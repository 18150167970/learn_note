

# XLA 架构

## 为什么要构建 XLA？

对于将 XLA 用于 TensorFlow，我们有以下几项目标：

- *提高执行速度*。编译子计算图以减少短暂运算的执行时间，从而消除 TensorFlow 运行时的开销；融合流水线运算以降低内存开销；并针对已知张量形状执行专门优化以支持更积极的常量传播。
- *提高内存使用率*。分析和安排内存使用，原则上需要消除许多中间存储缓冲区。
- *降低对自定义运算的依赖*。通过提高自动融合的低级运算的性能，使之达到手动融合的自定义运算的性能水平，从而消除对多种自定义运算的需求。
- *减少移动资源占用量*。通过提前编译子计算图并发出可以直接链接到其他应用的对象/头文件对，消除 TensorFlow 运行时。这样，移动推断的资源占用量可降低几个数量级。
- *提高便携性*。使针对新颖硬件编写新后端的工作变得相对容易，在新硬件上运行时，大部分 TensorFlow 程序都能够以未经修改的方式运行。与针对新硬件专门设计各个整体运算的方式相比，这种模式不必重新编写 TensorFlow 程序即可有效利用这些运算。

## XLA 工作原理

XLA 的输入语言称为“HLO IR”或仅为“HLO”（高级优化器）。[运算语义](https://www.tensorflow.org/xla/operation_semantics?hl=zh-cn)页面中介绍了 HLO 的语义。可以将 HLO 简单理解为[编译器 IR](https://en.wikipedia.org/wiki/Intermediate_representation)。

XLA 接受在 HLO 中定义的计算图（“计算”）并将其编译为适用于各种架构的机器指令。XLA 采用模块化设计，可以轻松融入其他后端以[针对某些新颖的硬件架构](https://www.tensorflow.org/xla/developing_new_backend?hl=zh-cn)。TensorFlow 源代码树中包含适用于 x64 和 ARM64 架构的 CPU 后端，以及 NVIDIA GPU 后端。

下图显示了 XLA 中的编译过程：

![img](imgs/how-does-xla-work.png)

XLA 提供了多种与目标无关的优化和分析过程（例如 [CSE](https://en.wikipedia.org/wiki/Common_subexpression_elimination)）、与目标无关的运算融合，以及用于为计算分配运行时内存的缓冲区分析。

完成与目标无关的步骤之后，XLA 会将 HLO 计算发送到后端。后端可以执行进一步的 HLO 级优化，而此时将考虑目标特定的信息和需求。例如，XLA GPU 后端可以执行特别有利于 GPU 编程模型的运算融合，并确定如何将计算划分为计算流。在此阶段，后端还可能对某些运算或运算组合针对优化库调用执行模式匹配。

下一步是针对特定目标生成代码。XLA 所含的 CPU 和 GPU 后端使用 [LLVM](http://llvm.org/) 进行低级 IR、优化和代码生成。这些后端发出有效表示 XLA HLO 计算所需的 LLVM IR，然后调用 LLVM 以从此 LLVM IR 中发出原生代码。

目前，GPU 后端通过 LLVM NVPTX 后端支持 NVIDIA GPU。CPU 后端支持多个 CPU ISA

## XLA自动聚类

This tutorial trains a TensorFlow model to classify the [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10) dataset, and we compile it using XLA.

Load and normalize the dataset using the Keras API:

```python
import tensorflow as tf

# Check that GPU is available: cf. https://colab.research.google.com/notebooks/gpu.ipynb
assert(tf.test.gpu_device_name())

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(False) # Start with XLA disabled.

def load_data():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x_train = x_train.astype('float32') / 256
  x_test = x_test.astype('float32') / 256

  # Convert class vectors to binary class matrices.
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
  return ((x_train, y_train), (x_test, y_test))

(x_train, y_train), (x_test, y_test) = load_data()
Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
170500096/170498071 [==============================] - 4s 0us/step
```

We define the model, adapted from the Keras [CIFAR-10 example](https://keras.io/examples/cifar10_cnn/):

```python
def generate_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(32, (3, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(64, (3, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Activation('softmax')
  ])

model = generate_model()
```

We train the model using the [RMSprop](https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer?hl=zh-cn) optimizer:

```python
def compile_model(model):
  opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
  return model

model = compile_model(model)

def train_model(model, x_train, y_train, x_test, y_test, epochs=25):
  model.fit(x_train, y_train, batch_size=256, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

def warmup(model, x_train, y_train, x_test, y_test):
  # Warm up the JIT, we do not wish to measure the compilation time.
  initial_weights = model.get_weights()
  train_model(model, x_train, y_train, x_test, y_test, epochs=1)
  model.set_weights(initial_weights)

warmup(model, x_train, y_train, x_test, y_test)
%time train_model(model, x_train, y_train, x_test, y_test)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
196/196 [==============================] - 14s 12ms/step - loss: 2.1887 - accuracy: 0.1773 - val_loss: 1.8642 - val_accuracy: 0.3363
Epoch 1/25
196/196 [==============================] - 2s 10ms/step - loss: 2.1349 - accuracy: 0.2016 - val_loss: 1.8978 - val_accuracy: 0.3353
Epoch 2/25
196/196 [==============================] - 2s 9ms/step - loss: 1.8088 - accuracy: 0.3462 - val_loss: 1.7053 - val_accuracy: 0.3935
Epoch 3/25
196/196 [==============================] - 2s 8ms/step - loss: 1.6749 - accuracy: 0.3941 - val_loss: 1.5618 - val_accuracy: 0.4401
Epoch 4/25
196/196 [==============================] - 2s 8ms/step - loss: 1.5902 - accuracy: 0.4240 - val_loss: 1.5003 - val_accuracy: 0.4683
Epoch 5/25
196/196 [==============================] - 2s 8ms/step - loss: 1.5168 - accuracy: 0.4486 - val_loss: 1.4156 - val_accuracy: 0.4967
Epoch 6/25
196/196 [==============================] - 2s 8ms/step - loss: 1.4654 - accuracy: 0.4703 - val_loss: 1.4081 - val_accuracy: 0.4961
Epoch 7/25
196/196 [==============================] - 2s 8ms/step - loss: 1.4231 - accuracy: 0.4869 - val_loss: 1.3556 - val_accuracy: 0.5162
Epoch 8/25
196/196 [==============================] - 2s 9ms/step - loss: 1.3901 - accuracy: 0.5019 - val_loss: 1.3041 - val_accuracy: 0.5368
Epoch 9/25
196/196 [==============================] - 2s 9ms/step - loss: 1.3559 - accuracy: 0.5162 - val_loss: 1.2992 - val_accuracy: 0.5475
Epoch 10/25
196/196 [==============================] - 2s 9ms/step - loss: 1.3259 - accuracy: 0.5255 - val_loss: 1.2536 - val_accuracy: 0.5587
Epoch 11/25
196/196 [==============================] - 2s 9ms/step - loss: 1.2971 - accuracy: 0.5375 - val_loss: 1.2550 - val_accuracy: 0.5607
Epoch 12/25
196/196 [==============================] - 2s 8ms/step - loss: 1.2713 - accuracy: 0.5505 - val_loss: 1.1769 - val_accuracy: 0.5906
Epoch 13/25
196/196 [==============================] - 2s 8ms/step - loss: 1.2476 - accuracy: 0.5582 - val_loss: 1.1955 - val_accuracy: 0.5770
Epoch 14/25
196/196 [==============================] - 2s 8ms/step - loss: 1.2216 - accuracy: 0.5679 - val_loss: 1.1839 - val_accuracy: 0.5813
Epoch 15/25
196/196 [==============================] - 2s 8ms/step - loss: 1.2023 - accuracy: 0.5745 - val_loss: 1.1746 - val_accuracy: 0.5912
Epoch 16/25
196/196 [==============================] - 2s 8ms/step - loss: 1.1755 - accuracy: 0.5842 - val_loss: 1.1104 - val_accuracy: 0.6097
Epoch 17/25
196/196 [==============================] - 2s 8ms/step - loss: 1.1513 - accuracy: 0.5954 - val_loss: 1.0757 - val_accuracy: 0.6233
Epoch 18/25
196/196 [==============================] - 2s 8ms/step - loss: 1.1392 - accuracy: 0.5998 - val_loss: 1.0859 - val_accuracy: 0.6209
Epoch 19/25
196/196 [==============================] - 2s 8ms/step - loss: 1.1167 - accuracy: 0.6059 - val_loss: 1.0935 - val_accuracy: 0.6183
Epoch 20/25
196/196 [==============================] - 2s 8ms/step - loss: 1.0941 - accuracy: 0.6143 - val_loss: 1.0590 - val_accuracy: 0.6329
Epoch 21/25
196/196 [==============================] - 2s 8ms/step - loss: 1.0814 - accuracy: 0.6208 - val_loss: 1.0499 - val_accuracy: 0.6334
Epoch 22/25
196/196 [==============================] - 2s 9ms/step - loss: 1.0638 - accuracy: 0.6275 - val_loss: 0.9962 - val_accuracy: 0.6580
Epoch 23/25
196/196 [==============================] - 2s 8ms/step - loss: 1.0448 - accuracy: 0.6342 - val_loss: 1.0240 - val_accuracy: 0.6419
Epoch 24/25
196/196 [==============================] - 2s 9ms/step - loss: 1.0301 - accuracy: 0.6402 - val_loss: 0.9885 - val_accuracy: 0.6512
Epoch 25/25
196/196 [==============================] - 2s 8ms/step - loss: 1.0165 - accuracy: 0.6423 - val_loss: 0.9609 - val_accuracy: 0.6659
CPU times: user 55.2 s, sys: 8.16 s, total: 1min 3s
Wall time: 42.7 s
313/313 [==============================] - 1s 3ms/step - loss: 0.9609 - accuracy: 0.6659
Test loss: 0.9608545899391174
Test accuracy: 0.6658999919891357
```

Now let's train the model again, using the XLA compiler. To enable the compiler in the middle of the application, we need to reset the Keras session.

```python
# We need to clear the session to enable JIT in the middle of the program.
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA.
model = compile_model(generate_model())
(x_train, y_train), (x_test, y_test) = load_data()

warmup(model, x_train, y_train, x_test, y_test)
%time train_model(model, x_train, y_train, x_test, y_test)
196/196 [==============================] - 6s 14ms/step - loss: 2.1602 - accuracy: 0.1890 - val_loss: 1.8198 - val_accuracy: 0.3543
Epoch 1/25
196/196 [==============================] - 4s 18ms/step - loss: 2.1084 - accuracy: 0.2171 - val_loss: 1.8668 - val_accuracy: 0.3387
Epoch 2/25
196/196 [==============================] - 2s 8ms/step - loss: 1.8050 - accuracy: 0.3503 - val_loss: 1.7050 - val_accuracy: 0.3965
Epoch 3/25
196/196 [==============================] - 2s 8ms/step - loss: 1.6757 - accuracy: 0.3943 - val_loss: 1.5696 - val_accuracy: 0.4384
Epoch 4/25
196/196 [==============================] - 2s 8ms/step - loss: 1.5956 - accuracy: 0.4228 - val_loss: 1.5098 - val_accuracy: 0.4567
Epoch 5/25
196/196 [==============================] - 2s 8ms/step - loss: 1.5310 - accuracy: 0.4456 - val_loss: 1.4722 - val_accuracy: 0.4707
Epoch 6/25
196/196 [==============================] - 2s 8ms/step - loss: 1.4825 - accuracy: 0.4631 - val_loss: 1.5245 - val_accuracy: 0.4628
Epoch 7/25
196/196 [==============================] - 2s 8ms/step - loss: 1.4374 - accuracy: 0.4837 - val_loss: 1.4239 - val_accuracy: 0.4915
Epoch 8/25
196/196 [==============================] - 2s 8ms/step - loss: 1.3900 - accuracy: 0.5026 - val_loss: 1.3184 - val_accuracy: 0.5260
Epoch 9/25
196/196 [==============================] - 2s 8ms/step - loss: 1.3600 - accuracy: 0.5143 - val_loss: 1.2731 - val_accuracy: 0.5496
Epoch 10/25
196/196 [==============================] - 2s 8ms/step - loss: 1.3260 - accuracy: 0.5281 - val_loss: 1.2552 - val_accuracy: 0.5542
Epoch 11/25
196/196 [==============================] - 2s 8ms/step - loss: 1.2938 - accuracy: 0.5387 - val_loss: 1.2242 - val_accuracy: 0.5738
Epoch 12/25
196/196 [==============================] - 2s 9ms/step - loss: 1.2642 - accuracy: 0.5510 - val_loss: 1.2240 - val_accuracy: 0.5596
Epoch 13/25
196/196 [==============================] - 2s 8ms/step - loss: 1.2389 - accuracy: 0.5622 - val_loss: 1.1663 - val_accuracy: 0.5868
Epoch 14/25
196/196 [==============================] - 2s 8ms/step - loss: 1.2110 - accuracy: 0.5711 - val_loss: 1.1312 - val_accuracy: 0.5983
Epoch 15/25
196/196 [==============================] - 2s 8ms/step - loss: 1.1856 - accuracy: 0.5821 - val_loss: 1.1978 - val_accuracy: 0.5730
Epoch 16/25
196/196 [==============================] - 2s 8ms/step - loss: 1.1619 - accuracy: 0.5890 - val_loss: 1.2709 - val_accuracy: 0.5568
Epoch 17/25
196/196 [==============================] - 2s 8ms/step - loss: 1.1430 - accuracy: 0.5975 - val_loss: 1.0918 - val_accuracy: 0.6181
Epoch 18/25
196/196 [==============================] - 2s 8ms/step - loss: 1.1190 - accuracy: 0.6074 - val_loss: 1.0924 - val_accuracy: 0.6148
Epoch 19/25
196/196 [==============================] - 2s 8ms/step - loss: 1.0970 - accuracy: 0.6130 - val_loss: 1.0485 - val_accuracy: 0.6277
Epoch 20/25
196/196 [==============================] - 2s 8ms/step - loss: 1.0830 - accuracy: 0.6191 - val_loss: 1.0675 - val_accuracy: 0.6241
Epoch 21/25
196/196 [==============================] - 2s 8ms/step - loss: 1.0616 - accuracy: 0.6289 - val_loss: 1.0053 - val_accuracy: 0.6540
Epoch 22/25
196/196 [==============================] - 2s 8ms/step - loss: 1.0483 - accuracy: 0.6331 - val_loss: 0.9849 - val_accuracy: 0.6615
Epoch 23/25
196/196 [==============================] - 2s 8ms/step - loss: 1.0322 - accuracy: 0.6364 - val_loss: 0.9753 - val_accuracy: 0.6617
Epoch 24/25
196/196 [==============================] - 2s 8ms/step - loss: 1.0190 - accuracy: 0.6421 - val_loss: 0.9657 - val_accuracy: 0.6639
Epoch 25/25
196/196 [==============================] - 1s 8ms/step - loss: 0.9991 - accuracy: 0.6517 - val_loss: 0.9733 - val_accuracy: 0.6630
CPU times: user 45.4 s, sys: 6.48 s, total: 51.9 s
Wall time: 41.2 s
```

On a machine with a Titan V GPU and an Intel Xeon E5-2690 CPU the speed up is ~1.17x.



# Use XLA with tf.function

This tutorial trains a TensorFlow model to classify the MNIST dataset, where the training function is compiled using XLA.

First, load TensorFlow and enable eager execution.

```bsh
# In TF 2.4 jit_compile is called experimental_compile
pip install -q tf-nightly
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
```

Then define some necessary constants and prepare the MNIST dataset.

```python
# Size of each input image, 28 x 28 pixels
IMAGE_SIZE = 28 * 28
# Number of distinct number labels, [0..9]
NUM_CLASSES = 10
# Number of examples in each training batch (step)
TRAIN_BATCH_SIZE = 100
# Number of training steps to run
TRAIN_STEPS = 1000

# Loads MNIST dataset.
train, test = tf.keras.datasets.mnist.load_data()
train_ds = tf.data.Dataset.from_tensor_slices(train).batch(TRAIN_BATCH_SIZE).repeat()

# Casting from raw data to the required datatypes.
def cast(images, labels):
  images = tf.cast(
      tf.reshape(images, [-1, IMAGE_SIZE]), tf.float32)
  labels = tf.cast(labels, tf.int64)
  return (images, labels)
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 0s 0us/step
```

Finally, define the model and the optimizer. The model uses a single dense layer.

```python
layer = tf.keras.layers.Dense(NUM_CLASSES)
optimizer = tf.keras.optimizers.Adam()
```

## Define the training function

In the training function, you get the predicted labels using the layer defined above, and then minimize the gradient of the loss using the optimizer. In order to compile the computation using XLA, place it inside [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function?hl=zh-cn) with `jit_compile=True`.

```python
@tf.function(jit_compile=True)
def train_mnist(images, labels):
    images, labels = cast(images, labels)

    with tf.GradientTape() as tape:
      predicted_labels = layer(images)
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=predicted_labels, labels=labels
      ))
    layer_variables = layer.trainable_variables
    grads = tape.gradient(loss, layer_variables)
    optimizer.apply_gradients(zip(grads, layer_variables))
```

## Train and test the model

Once you have defined the training function, define the model.

```python
for images, labels in train_ds:
  if optimizer.iterations > TRAIN_STEPS:
    break
  train_mnist(images, labels)
```

And, finally, check the accuracy:

```python
images, labels = cast(test[0], test[1])
predicted_labels = layer(images)
correct_prediction = tf.equal(tf.argmax(predicted_labels, 1), labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Prediction accuracy after training: %s" % accuracy)
Prediction accuracy after training: tf.Tensor(0.8799, shape=(), dtype=float32)
```

Behind the scenes, the XLA compiler has compiled the entire TF function to HLO, which has enabled fusion optimizations. Using the introspection facilities, we can see the HLO code (other interesting possible values for "stage" are `optimized_hlo` for HLO after optimizations and `optimized_hlo_dot` for a Graphviz graph):

```python
print(train_mnist.experimental_get_compiler_ir(images, labels)(stage='hlo'))
HloModule a_inference_train_mnist_5289__.198, input_output_alias={ {0}: (2, {}, may-alias), {1}: (3, {}, may-alias), {2}: (5, {}, may-alias), {3}: (8, {}, may-alias), {4}: (9, {}, may-alias), {5}: (10, {}, may-alias), {6}: (11, {}, may-alias) }

%max_float_.59 (x.60: f32[], y.61: f32[]) -> f32[] {
  %x.60 = f32[] parameter(0)
  %y.61 = f32[] parameter(1)
  ROOT %maximum.62 = f32[] maximum(f32[] %x.60, f32[] %y.61)
}

%add_float_.69 (x.70: f32[], y.71: f32[]) -> f32[] {
  %x.70 = f32[] parameter(0)
  %y.71 = f32[] parameter(1)
  ROOT %add.72 = f32[] add(f32[] %x.70, f32[] %y.71)
}
```

# 参考文献 #