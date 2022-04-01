## gprc 调优

```

def request_server(image, server_url, name="model3", signature_name="serving_default", inputs="input_1",
                   outputs="activation_19", wait_time=100):
    """
    用于向TensorFlow Serving服务请求推理结果的函数。
    :param image: 经过预处理的待推理图片数组，numpy array，shape：(h, w, 3)
    :param server_url: TensorFlow Serving的地址加端口，str，如：'0.0.0.0:8500'
    :param name:
    :param signature_name:
    :param inputs:
    :param outputs:
    :param index:
    :return:
    模型返回的结果数组，numpy array

    """
    # Request.
    # 设置最长接收大小
    max_message_length = 271360078 * 3
    channel = grpc.insecure_channel(server_url, options=[('grpc.max_send_message_length', max_message_length),
                                                         ('grpc.max_receive_message_length', max_message_length)])
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = name  # 模型名称，启动容器命令的model_name参数
    request.model_spec.version.value = 0
    # request.model_spec.version_label = "stable"
    request.model_spec.signature_name = signature_name  # 签名名称，刚才叫你记下来的
    # "input_1"是你导出模型时设置的输入名称，刚才叫你记下来的
    request.inputs[inputs].CopyFrom(
        tf.make_tensor_proto(image, shape=list(image.shape)))
    response = stub.Predict(request, wait_time)  # 5 secs timeout
    channel.close()
    res = np.asarray(response.outputs[outputs].float_val)  # outputs为输出名称，刚才叫你记下来的
    return res
```

客户端请求到返回 耗时过大, 经过分析

主要耗时在

```
res = np.asarray(response.outputs[outputs].float_val)
```

中的`.float_val` ,耗时约10s

经过google, 采用

```
tf.make_ndarray(response.outputs[outputs])
```

可以减少一半时间,耗时约5s,但是还是太久了 

因此最后采用

```
res = tf.io.parse_tensor(response.outputs[outputs].SerializeToString(),
                         out_type=tf.float32)  # outputs为输出名称，刚才叫你记下来的
```

耗时约 2.13s, 还是非常久





# 参考文献 #