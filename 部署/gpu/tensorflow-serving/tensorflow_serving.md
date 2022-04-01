## 1.安装

### 1.1直接在系统安装 tensorflow-serving

```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list
# 添加gpg key
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install tensorflow-model-server
```
这种方式安装只能调用cpu,无法调用gpu
### 1.2docker安装tensorflow-serving

#### 1.2.1安装docker

[安装docker](../../docker/docker-nvidia.md)

#### 1.2.2服务器拉取tensorflow-serving

```
sudo docker pull tensorflow/serving:latest-gpu
```

docker tensorflow-serving 镜像官网https://hub.docker.com/r/tensorflow/serving/tags/

#### 客户端安装tensorflow-serving-api

```
pip3 install tensorflow-serving-api-gpu
```

## 2.模型导出

```
model.save(h5_model_save_path, save_format='tf')
```

## 3.服务端启动tensorflow-serving命令:

```
sudo docker run --gpus "device=0" -p 8500:8500 --mount type=bind,source=/home/chenli/formal_work/tensorflow_serving/models,target=/models -t --entrypoint=tensorflow_model_server 675505ceb9db --model_name=headcount --model_base_path=/models/model1 &
```

各参数意义:

---- docker 参数

- --gpus all :指定gpu使用情况  指定特定gpu "--gpus "device=0""

- -p 8500:8500 ：指的是开放8500这个gRPC端口。

- --mount type=bind, source=/your/local/model, target=/models：把你导出的本地模型文件夹挂载到docker container的/models这个文件夹，tensorflow serving会从容器内的/models文件夹里面找到你的模型。

- -t --entrypoint=tensorflow_model_server tensorflow/serving:latest-gpu：如果使用非devel版的docker，启动docker之后是不能进入容器内部bash环境的，--entrypoint的作用是允许你“间接”进入容器内部，然后调用tensorflow_model_server命令来启动TensorFlow Serving，这样才能输入后面的参数。紧接着指定使用tensorflow/serving:latest-gpu 这个镜像，可以换成你想要的任何版本。

----- tensorflow serving 参数

- --port=8500：开放8500这个gRPC端口（需要先设置上面entrypoint参数，否则无效。下面参数亦然）

- --entrypoint=tensorflow_model_server

- --per_process_gpu_memory_fraction=0.5：只允许模型使用多少百分比的显存，数值在[0, 1]之间。

- --model_name：模型名字，在导出模型的时候设置的名字。

- --model_base_path：模型所在容器内的路径，前面的mount已经挂载到了/models文件夹内，这里需要进一步指定到某个模型文件夹，例如/models/east_model指的是使用/models/east_model这个文件夹下面的模型。

- --model_config_file=/models/models.config   模型配置文件

- --enable_batching=true: 允许模型进行批推理，提高GPU使用效率。  使用batch预测

- --prefer_tflite_model=false      	bool	EXPERIMENTAL; CAN BE REMOVED ANYTIME! Prefer TensorFlow Lite model from `model.tflite` file in SavedModel directory, instead of the TensorFlow model from `saved_model.pb` file. If no TensorFlow Lite model found, fallback to TensorFlow model.

- --enable_signature_method_name_check=false	bool	Enable method_name check for SignatureDef. Disable this if serving native TF2 regression/classification models.

- --grpc_channel_arguments=""      	string	A comma separated list of arguments to be passed to the grpc server. (e.g. grpc.max_connection_age_ms=2000)

- --tensorflow_session_parallelism=0	int64	Number of threads to use for running a Tensorflow session. Auto-configured by default.Note that this option is ignored if --platform_config_file is non-empty.

- --tensorflow_intra_op_parallelism=0	int64	Number of threads to use to parallelize the executionof an individual op. Auto-configured by default.Note that this option is ignored if --platform_config_file is non-empty.

- --tensorflow_inter_op_parallelism=0	int64	Controls the number of operators that can be executed simultaneously. Auto-configured by default.Note that this option is ignored if --platform_config_file is non-empty.

-  --batching_parameters_file="models/batching_parameters_file.txt"  

  ```
  max_batch_size {value:128}
  batch_timeout_micros {value:100}
  max_enqueued_batches {value:1000000}
  num_batch_threads {value:8}
  ```

  ```
  max_batch_size: The maximum size of any batch. This parameter governs the throughput/latency tradeoff, and also avoids having batches that are so large they exceed some resource constraint (e.g. GPU memory to hold a batch's data).
  
  batch_timeout_micros: The maximum amount of time to wait before executing a batch (even if it hasn't reached max_batch_size). Used to rein in tail latency. (See basic_batch_scheduler.h for the exact latency contract.)
  
  num_batch_threads: The degree of parallelism, i.e. the maximum number of batches processed concurrently.
  
  max_enqueued_batches: The number of batches worth of tasks that can be enqueued to the scheduler. Used to bound queueing delay, by turning away requests that would take a long time to get to, rather than building up a large backlog.
  ```

  

**ps**: 注意,tensorflow-serving参数后面要带上`=`才能赋值进去,不然使用的是默认参数.

模型目录下要带版本号,不然会出错

```
|-total_model
	|- model_dir
        |- 1
            |- svaed_model.pb
            |- variables
                |- variables.data-00000-of-00001
                |- variables.index
        |- 2
            |- svaed_model.pb
            |- variables
                |- variables.data-00000-of-00001
                |- variables.index
	|- model_dir2
        |- 1
            |- svaed_model.pb
            |- variables
                |- variables.data-00000-of-00001
                |- variables.index
```

容器停止命令:

```
docker container list
docker container stop 37b952c2785c
```



## 4.客户端请求

查看模型

> ```text
> saved_model_cli show --dir 0/ --all
> ```
>
> 在模型文件夹的终端中,执行上面指令,查看`signature_name` ,`inputs`和`outputs`,用于后续的客户端代码调用.内容如下: 下面的`signature_name`="serving_default",`inputs`="input_bgr", `outputs`="probability_0"
>
> ```
> signature_def['serving_default']:
> The given SavedModel SignatureDef contains the following input(s):
> inputs['input_bgr'] tensor_info:
> outputs['probability_0'] tensor_info:
> ```
>
> 

```
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
from PIL import Image
import numpy as np
import tensorflow as tf



def request_server(img_resized, server_url, name="model3", signature_name="serving_default", inputs="input_1",
                   outputs="activation_19"):
    '''
    用于向TensorFlow Serving服务请求推理结果的函数。
    :param img_resized: 经过预处理的待推理图片数组，numpy array，shape：(h, w, 3)
    :param server_url: TensorFlow Serving的地址加端口，str，如：'0.0.0.0:8500'
    :return: 模型返回的结果数组，numpy array
    '''
    # Request.
    channel = grpc.insecure_channel(server_url)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = name  # 模型名称，启动容器命令的model_name参数
    request.model_spec.version.value = 1
    # request.model_spec.version_label = "stable"
    request.model_spec.signature_name = signature_name  # 签名名称，刚才叫你记下来的
    # "input_1"是你导出模型时设置的输入名称，刚才叫你记下来的
    request.inputs[inputs].CopyFrom(
        tf.make_tensor_proto(img_resized, shape=[1, ] + list(img_resized.shape)))
    response = stub.Predict(request, 100.0)  # 5 secs timeout
    res = np.asarray(response.outputs[outputs].float_val)

    print(res, time.time() - t1)
    # return np.asarray(response.outputs[outputs].float_val)  # fc2为输出名称，刚才叫你记下来的



imgpath = 'test.png'
import cv2

image = cv2.imread(imgpath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# grpc地址及端口，为你容器所在机器的ip + 容器启动命令里面设置的port
server_url = '0.0.0.0:8500'
res = request_server(image, server_url)

print(res.sum())
```



## 5.tensorflow-serving 多模型

本例中用使用savedModel保存模型中的相关代码生成三个模型，分别建立三个文件夹，将得到的模型分别放入，最后的文件结构如下图。其中100001文件夹表示模型的版本，可以在model1下放置不同版本的模型，默认情况下会加载具有较大版本号数字的模型。

### 多模型部署

  在multiModel文件夹下新建一个配置文件model.config，文件内容为：

```
model_config_list:{
    config:{
      name:"model1",
      base_path:"/models/model1",
      model_platform:"tensorflow"
    },
    config:{
      name:"model2",
      base_path:"/models/model2",
      model_platform:"tensorflow"
    },
    config:{
      name:"model3",
      base_path:"/models/model3",
      model_platform:"tensorflow"
    } 
}
```


配置文件定义了模型的名称和模型在容器内的路径，现在运行tfserving容器 :

```
sudo docker run --gpus "device=0" -p 8701:8701 --mount type=bind,source=/home/starnet/projects/chenli/dm-count/models,target=/models  -t 6dabf2c0a340 --model_config_file=/models/models.config --port=8701 --enable_batching=true --batching_parameters_file="models/batching_parameters_file.txt" --grpc_channel_arguments=grpc.max_concurrent_streams=1000 --per_process_gpu_memory_fraction=0.7 --tensorflow_session_parallelism=2
```



### 多模型调用

调用模型时,将模型名置为配置文件的中对应的名字,则可以调用:

demo:

```
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
from PIL import Image
import numpy as np
import tensorflow as tf


def request_server(img_resized, server_url, name="model3"):
    '''
    用于向TensorFlow Serving服务请求推理结果的函数。
    :param img_resized: 经过预处理的待推理图片数组，numpy array，shape：(h, w, 3)
    :param server_url: TensorFlow Serving的地址加端口，str，如：'0.0.0.0:8500'
    :return: 模型返回的结果数组，numpy array
    '''
    # Request.
    channel = grpc.insecure_channel(server_url)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = name  # 模型名称，启动容器命令的model_name参数
    request.model_spec.signature_name = "serving_default"  # 签名名称，刚才叫你记下来的
    # "input_1"是你导出模型时设置的输入名称，刚才叫你记下来的
    request.inputs["input_1"].CopyFrom(
        tf.make_tensor_proto(img_resized, shape=[1, ] + list(img_resized.shape)))
    response = stub.Predict(request, 5.0)  # 5 secs timeout
    return np.asarray(response.outputs["activation_19"].float_val)  # fc2为输出名称，刚才叫你记下来的


imgpath = 'test.png'
import cv2

image = cv2.imread(imgpath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# grpc地址及端口，为你容器所在机器的ip + 容器启动命令里面设置的port
server_url = '0.0.0.0:8500'
res = request_server(image, server_url, name="model1")
print(res.sum())

res = request_server(image, server_url, name="model2")
print(res.sum())

res = request_server(image, server_url, name="model3")
print(res.sum())
```



### 指定多模型版本
如果一个模型有多个版本，并在预测的时候希望指定模型的版本，可以通过以下方式实现。
修改model.config文件，增加model_version_policy：

```
model_config_list:{
    config:{
      name:"model1",
      base_path:"/models/model1",
      model_platform:"tensorflow",
      model_version_policy:{
        all:{}
      }
    },
    config:{
      name:"model2",
      base_path:"/models/model2",
      model_platform:"tensorflow"
    },
    config:{
      name:"model3",
      base_path:"/models/model3",
      model_platform:"tensorflow"
    } 
}
```

或者

```
model_config_list {
  config {
    name: 'model1',
    model_platform: "tensorflow",
    base_path: '/models/model1'
    model_version_policy{
      specific{
            version: 1,
            version: 2
        }
    }
    version_labels{
        key: "stable",
        value: 1    
    }
    version_labels{
        key: "test",
        value: 2    
    }
  }
}
```



tfserving支持模型的Hot Plug，上述容器运行起来之后，如果在宿主机的 /home/jerry/tmp/multiModel/model1/ 文件夹下新增模型文件如100003/，tfserving会自动加载新模型；同样如果移除现有模型，tfserving也会自动卸载模型

### 指定版本号或者版本别名调用

gRPC方式

使用别名：
```
channel = grpc.insecure_channel(server_url)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = name  # 模型名称，启动容器命令的model_name参数
# request.model_spec.version.value = 1

request.model_spec.version_label = "stable"

request.model_spec.signature_name = "serving_default"  # 签名名称，刚才叫你记下来的
# "input_1"是你导出模型时设置的输入名称，刚才叫你记下来的
request.inputs["input_1"].CopyFrom(
tf.make_tensor_proto(img_resized, shape=[1, ] + list(img_resized.shape)))
response = stub.Predict(request, 5.0)  # 5 secs timeout
return np.asarray(response.outputs["activation_19"].float_val)  # fc2为输出名称，刚才叫你记下来的
```

使用版本号：
```
 channel = grpc.insecure_channel(server_url)
 stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
 request = predict_pb2.PredictRequest()
 request.model_spec.name = name  # 模型名称，启动容器命令的model_name参数
 
 request.model_spec.version.value = 1
 
 request.model_spec.signature_name = "serving_default"  # 签名名称，刚才叫你记下来的
 # "input_1"是你导出模型时设置的输入名称，刚才叫你记下来的
 request.inputs["input_1"].CopyFrom(
 tf.make_tensor_proto(img_resized, shape=[1, ] + list(img_resized.shape)))
 response = stub.Predict(request, 5.0)  # 5 secs timeout
 return np.asarray(response.outputs["activation_19"].float_val)  # fc2为输出名称，刚才叫你记下来的
```

区别：model_spec.version_label与model_spec.version.value。



## 5.tensorflow-serving加速

**(1) 服务器端批处理**

服务器端批处理由TensorFlow Serving支持开箱即用。.

延迟和吞吐量之间的权衡取决于所支持的批处理参数。TensorFlow批处理最有效地利用硬件加速器承诺（保证）的高吞吐量。

若要启用批处理，请设置--enable_batching和--batching_parameters_file标志。可以将批处理参数设置为SessionBundleConfig。对于只使用CPU的系统，请考虑设置num_batch_threads可以使用的核心数量。批处理配置方法可访问这里，使用支持GPU的系统。

当服务器端的批处理请求全部到达时，推理请求在内部合并为单个大请求(Tensor)，并在合并请求上运行一个TensorFlow会话。在单个会话上运行批量请求，可以真正利用CPU/GPU并行性。

批处理过程中需要考虑的Tensorflow Serving Batching进程：

1. 在客户端使用异步请求，以在服务器端进行批处理
2. 在CPU/GPU上加入模型图组件，加速批处理
3. 在同一服务器服务多个模型时，对预测请求进行交织处理
4. 对于“脱机”大容量推理处理，强烈推荐批处理。

**(2) 客户端批处理**

客户端的批处理是将多个输入组合在一起，以发出单个请求。

由于ResNet模型要求以NHWC格式输入(第一个维度是输入的数量)，所以我们可以将多个输入图像聚合到一个RPC请求中：

```text
...
batch = []  
for jpeg in os.listdir(FLAGS.images_path):  
  path = os.path.join(FLAGS.images_path, jpeg)
  img = cv2.imread(path).astype(np.float32)
  batch.append(img)
...

batch_np = np.array(batch).astype(np.float32)  
dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=dim) for dim in batch_np.shape]  
t_shape = tensor_shape_pb2.TensorShapeProto(dim=dims)  
tensor = tensor_pb2.TensorProto(  
              dtype=types_pb2.DT_FLOAT,
              tensor_shape=t_shape,
              float_val=list(batched_np.reshape(-1)))
request.inputs['inputs'].CopyFrom(tensor)  
```

对N个图像的批处理，响应（相应）的输出Tensor对于请求批处理中相同数量的输入具有预测结果，在这种情况下，N=2（以下是N=2的情况）：

```text
outputs {  
  key: "classes"
  value {
    dtype: DT_INT64
    tensor_shape {
      dim {
        size: 2
      }
    }
    int64_val: 238
   
   int64_val: 121
  }
}
...
```



### Batch Scheduling Parameters and Tuning

控制 Batch Scheduling 的参数如下：

- `max_batch_size`：任何 Batch 的最大大小，该参数控制「吞吐量/延迟」的平衡，并且可以避免了 Batch 过大以至于超出了某些资源的限制（例：GPU 内存可以保存一批数据）。
- `batch_timeout_micros`：执行 Batch 之前等待的最长时间（即使未达到 `max_batch_size`），该参数用于控制尾部延迟。
- `num_batch_threads`：并行化程度，即同时处理的最大 Batch 数。
- `max_enqueued_batches`：可以排入调度程序的 Batch 数量。通过拒绝需要很长时间才能到达的请求，而不是全部积压到队列，从而来限制排队延迟。

最佳的参数设置取决于系统、模型、系统、环境以及吞吐量、延迟等，最好通过实验测试来选择其值。下面的一些准则可能会帮助您选择更优解。

**基本准则**

首先，在进行实验时，应将 `max_enqueued_batches` 临时设置为非常高的值。然后，对于生产环境，请考虑将 `max_enqueued_batches` 设置为等于 `num_batch_threads`，以便最大程度地减少服务器上的排队延迟，使服务器始终保持忙碌状态。对于 Batch 作业，请将 `max_enqueued_batches` 设置为足够大的值，但也应该理性设置，避免内存不足导致系统崩溃。

其次，如果出于系统架构的原因，需要限制 Batch Size（例如：设置为 100、200 或 400，而不是介于 1 ~ 400 之间的任何值）：如果使用 `BatchingSession`，则可以设置 `allowed_batch_sizes` 参数；否则，可以在回调代码中使用虚拟元素填充 Batch。

**CPU 服务器**

请考虑以下配置：`num_batch_threads` 等于 CPU 内核数； `max_batch_size` 值很高； `batch_timeout_micros` 设置为 0，然后使用 1 - 10 毫秒范围内的 `batch_timeout_micros` 值进行实验，0 可能是最佳值。

**GPU 服务器**

- 将 `num_batch_threads` 设置为 CPU 内核数；
- 调整 `max_batch_size` 时，将 `batch_timeout_micros` 临时设置为非常高的值，在吞吐量和平均延迟之间达到所需的平衡，推荐设置为 100 - 10000；
- `batch_timeout_micros` 的最佳值通常为几毫秒，具体取决于您的目标和上下文，在某些工作负载下，可以考虑设置为 0；对于 Batch 作业，请选择一个较大的值（可能是几秒钟），以确保良好的吞吐量，但是不要设置太大。



## 出现的错误

### No /dev/nvidia* exist

docker run 后面跟上--gpus all  并且使用管理员权限执行

### AbortionError(code=StatusCode.NOT_FOUND, details="Servable not found for request: Latest(“’mask”)

模型命名不对,检查下docker run 命令中的`model_name`参数,是不是`model_name = "headcount"`这样的格式,并且调用时,

request.model_spec.name 是不是上面的一致

类似问题可以检查ip之类的有没有错

### grpc: *received* *message* *larger* *than* *max* (4195017 vs. 4194304)

设置最长接收大小

```
MAX_MESSAGE_LENGTH = 100102410
channel = grpc.insecure_channel(server_url, options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                                     ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
```

### GPU利用率低

```
--gpus "device=0"

--enable_batching=true --batching_parameters_file="models/batching_parameters_file.txt" --grpc_channel_arguments=grpc.max_concurrent_streams=1000 --per_process_gpu_memory_fraction=0.7 --tensorflow_session_parallelism=2
```
docker gpus 指定为 单gpu,多gpu会导致gpu利用率低,目前不知道原因.

### docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].

未安装nvidia-docker2

[安装docker](../../docker/docker-nvidia.md)

### status = StatusCode.UNAVAILABLE 	details = "failed to connect to all address

grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with:

端口错误, grpc默认的是8500短裤, rest默认8501端口,  要指定不同的serving 可以通过指定 `-p 8xxx:8500` 来分配不同端口, 请求时 使用`0.0.0.0:8xxx`请求



# 参考文献 #

https://blog.csdn.net/chenguangchun1993/article/details/104971811

https://zhuanlan.zhihu.com/p/96917543

https://hackernoon.com/how-we-improved-tensorflow-serving-performance-by-over-70-f21b5dad2d98

https://www.jianshu.com/p/c0caa3af68e0

