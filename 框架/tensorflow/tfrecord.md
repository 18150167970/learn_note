平台  tensorflow2.6

## 生成tfrecord

将图像写入tfrecord中

```
save_path = "data/tfrecords_train/data.tfrecords"
train_save_path_list.append(save_path)
with tf.io.TFRecordWriter(save_path) as file_writer:
    for i in tqdm(range(tr_gen.__len__())):
        if i > 100:
            break
        x, y = tr_gen.__getitem__(i)
        x = x[0].tobytes()
        y = y[0].tobytes()
        record_bytes = tf.train.Example(features=tf.train.Features(feature={
            "x": tf.train.Feature(bytes_list=tf.train.BytesList(value=[x])),
            "y": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y]))
        })).SerializeToString()
        file_writer.write(record_bytes)
```

其中，图像需要转成``` tybytes```,再用对应格式写入， 传入的是[x] 是列表



## 读tfrecord

```
def decode_fn(record_bytes):
    example = tf.io.parse_single_example(
        record_bytes, {
            "x": tf.io.FixedLenFeature([], dtype=tf.string),
            "y": tf.io.FixedLenFeature([], dtype=tf.string)
        }
    )
    return tf.io.decode_raw(example["x"], tf.float32), tf.io.decode_raw(example["y"], tf.float32)


n_parse_threads = 8
shuffle_buffer_size = 10000
dataset = tf.data.TFRecordDataset(train_save_path_list).map(decode_fn, num_parallel_calls=n_parse_threads)
dataset.shuffle(shuffle_buffer_size)
dataset = dataset.batch(batch_size)
```



若是kaggle上

```
GCS_DS_PATH = KaggleDatasets().get_gcs_path('naic-dataset')
filenames = tf.io.gfile.glob(GCS_DS_PATH + "/tfrecords_train/*.tfrecords")
filenames2 = tf.io.gfile.glob(GCS_DS_PATH + "/tfrecords_test/*.tfrecords")
filenames = filenames + filenames2
dataset = tf.data.TFRecordDataset(filenames).map(decode_fn, num_parallel_calls=n_parse_threads)
dataset.shuffle(shuffle_buffer_size)
dataset = dataset.batch(batch_size, drop_remainder=True)
```

`drop_remainder`为是否舍弃batch数剩余的样本



查看数据:

```
for batch in dataset:
    x = batch[0]
    y = batch[1]
    print(x, y)
```







# 参考文献 #