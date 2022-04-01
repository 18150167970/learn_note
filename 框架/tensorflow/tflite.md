

## keras to tflite

```
converter = tf.lite.TFLiteConverter.from_keras_model(model)
if dtype == "float16":
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()
with open(tflite_model_save_path, "wb") as f:
    f.write(tflite_model)
print("save tflite model done")
```

# 参考文献 #