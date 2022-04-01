# 加载环境

如果出现一直加载且没有排队消息, 考虑用chrome浏览器,不能用safari浏览器

# 数据处理

## 数据生成

TPU的keras训练只能使用tfrecords数据进行训练.  因此需要生成tfrecords 数据

[数据生成](./tfrecord.md)

##　数据读取

```
from kaggle_datasets import KaggleDatasets
def decode_fn(record_bytes):
    example = tf.io.parse_single_example(
        record_bytes, {
            "x": tf.io.FixedLenFeature([], dtype=tf.string),
            "y": tf.io.FixedLenFeature([], dtype=tf.string)
        }
    )
    x = tf.io.decode_raw(example["x"], tf.float32)
    y = tf.io.decode_raw(example["y"], tf.float32)
    return tf.cast(x, tf.float16), tf.cast(y, tf.float16) 

GCS_DS_PATH = KaggleDatasets().get_gcs_path('naic-dataset')

```

`'naic-dataset'`  公开数据集名字, TPU训练使用的数据必须上传到公开的dataset中, 不然无法加载用于训练

```
n_parse_threads = 1
shuffle_buffer_size = 10000
filenames = tf.io.gfile.glob(GCS_DS_PATH + "/tfrecords_train/*.tfrecords")
filenames2 = tf.io.gfile.glob(GCS_DS_PATH + "/tfrecords_test/*.tfrecords")
filenames = filenames + filenames2
dataset = tf.data.TFRecordDataset(filenames).map(decode_fn, num_parallel_calls=n_parse_threads)
dataset.shuffle(shuffle_buffer_size)
dataset = dataset.batch(batch_size, drop_remainder=True)

filenames = tf.io.gfile.glob(GCS_DS_PATH + "/tfrecords_test/*.tfrecords")
val_dataset = tf.data.TFRecordDataset(filenames).map(decode_fn, num_parallel_calls=n_parse_threads)
val_dataset.shuffle(shuffle_buffer_size)
val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

```



```
dataset = dataset.repeat(1000)
val_dataset = val_dataset.repeat(100)
```

数据量不够会导致训练结束, 因此要重复数据保证数据量.





# 模型

## 模型加载

```
tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
tf.tpu.experimental.initialize_tpu_system(tpu)
# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

with tpu_strategy.scope():
    model = build_recognizer(dtype="float32")
    model.compile(
        loss='binary_crossentropy',
        # loss=bce_with_logits(),
        metrics=[[accuracy()]],
        optimizer=AdaBeliefOptimizer(learning_rate=lr),
        steps_per_execution=batch_size,
    )  
```

初始化TPU



## callback

### 权重保存

```
tmp_ckpt_save_path = "/kaggle/working/naic2021-ai-wireless-semifinal/checkpoints"
save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')

ckpt_callback = callbacks.ModelCheckpoint(
        os.path.join(tmp_ckpt_save_path,
                     'ep{epoch:03d}_loss{loss:.4f}_val-loss{val_loss:.4f}_val-acc{val_accuracy:.4f}.h5'),
        prefix="model1",
        save_weights_only=False,
        monitor='loss',
        save_best_only=True,
        save_format='h5',
        options=save_locally,
    )
```

必须加上`save_locally`, 因为模型是在远程训练的

### 训练日志保存

```
csv_writer = callbacks.CSVLogger('./log.csv',separator=",",append=True)
```

由于页面会被关闭,需要保存日志



### 学习率缩减

```
reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.666,
            patience=patience,
            min_lr=1e-5,
            verbose=1,
        )

callback_list= [
    reduce_lr,
    ckpt_callback,
    csv_writer
]
```



##　模型训练

```
model.fit(
    x=dataset,
    steps_per_epoch=512 // batch_size,
    validation_data=val_dataset,
    validation_steps=512 // batch_size,
    initial_epoch=initial_epoch,
    epochs=max_epoch,
    callbacks=callback_list,
)
```

必须指定`steps_per_epoch` 和 `validation_steps`





# 问题

## 训练loss为nan

TPU需要设置batch数为128倍数(可能是8的倍数)

## kaggle tpu Unable to parse tensor proto

数据需要放到kaggle的dataset上,并公开. 或者也可以放在cloud上.











# 附录

## keras 调用TPU

```
# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
with tpu_strategy.scope():
    model = tf.keras.Sequential( … ) # define your model normally
    model.compile( … )

# train model normally
model.fit(training_dataset, epochs=EPOCHS, steps_per_epoch=…)
```

TPUs are network-connected accelerators and you must first locate them on the network. This is what `TPUClusterResolver.connect()` does. 

You then instantiate a `TPUStrategy`. This object contains the necessary distributed training code that will work on TPUs with their 8 compute cores (see [hardware section below](https://www.kaggle.com/docs/tpu#tpuhardware)).

Finally, you use the `TPUStrategy` by instantiating your model in the scope of the strategy. This creates the model on the TPU. Model size is constrained by the TPU RAM only, not by the amount of memory available on the VM running your Python code. Model creation and model training use the usual Keras APIs.





## Batch size, learning rate, steps_per_execution

To go fast on a TPU, increase the batch size. The rule of thumb is to use batches of 128 elements per core (ex: batch size of 128*8=1024 for a TPU with 8 cores). At this size, the 128x128 hardware matrix multipliers of the TPU (see [hardware section below](https://www.kaggle.com/docs/tpu#tpuhardware)) are most likely to be kept busy. You start seeing interesting speedups from a batch size of 8 per core though. In the sample above, the batch size is scaled with the core count through this line of code:

```
BATCH_SIZE = 16 * tpu_strategy.num_replicas_in_sync
```



With a TPUStrategy running on a single TPU v3-8, the core count is 8. This is the hardware available on Kaggle. It could be more on larger configurations called TPU pods available on Google Cloud.

![illustration of batch size and learning rate scaling rule of thumb on TPU](imgs/tpu_rule_of_thumb.png)

With larger batch sizes, TPUs will be crunching through the training data faster. This is only useful if the larger training batches produce more “training work” and get your model to the desired accuracy faster. That is why the rule of thumb also calls for increasing the learning rate with the batch size. You can start with a proportional increase but additional tuning may be necessary to find the optimal learning rate schedule for a given model and accelerator.

Starting with Tensorflow 2.4, model.compile() accepts a new `steps_per_execution` parameter. This parameter instructs Keras to send multiple batches to the TPU at once. In addition to lowering communications overheads, this gives the XLA compiler the opportunity to optimize TPU hardware utilization across multiple batches. With this option, it is no longer necessary to push batch sizes to very high values to optimize TPU performance. As long as you use batch sizes of at least 8 per core (>=64 for a TPUv3-8) performance should be acceptable. Example:

```
    model.compile( … ,
                  steps_per_execution=32)
    
```

设置`steps_per_execution`可以不需要大的batch也能提高tpu性能。（正常要128， 现在32的倍数就可以了）







# 参考文献 #