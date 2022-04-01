## 生成tflite

**keras model 生成 tflite**

```
inputs = layers.Input(shape=(image_size[0], image_size[1], 3), dtype="uint8")
model = build_model(inputs, is_train=False, dtype="float32")
model.load_weights(model_path, by_name=True)
model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
if dtype == "float16":
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open(tflite_model_save_path, "wb") as f:
    f.write(tflite_model)
print("save tflite model done")
```



## 读取tflite

```
num_threads = 1
self.model_tflite, self.tflite_input_ids, self.tflite_output_ids = \
    self.load_tflite_model(model_path, num_threads=num_threads)

```



## 预测

```
def _tflite_predict(self, x):
   self.model_tflite.set_tensor(self.tflite_input_ids[0], x)
   self.model_tflite.invoke()
   return list(self.model_tflite.get_tensor(out_id) for out_id in self.tflite_output_ids)
   
outputs = self._tflite_predict(x)  # predict
```



# 参考文献 #