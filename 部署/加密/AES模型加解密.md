## 模型加密

```
import pickle
from Crypto.Cipher import AES


def add_16(par):
    if type(par) == str:
        par = par.encode()
    while len(par) % 16 != 0:
        par += b'\x00'
    return par


def aesencrypt(text, key):
    key = add_16(key)
    text = add_16(text)
    aes = AES.new(key, AES.MODE_ECB)
    encrypt_text = aes.encrypt(text)
    return encrypt_text


def aesdecrypt(text, key):
    key = add_16(key)
    aes = AES.new(key, AES.MODE_ECB)
    decrypt_text = aes.decrypt(text)
    decrypt_text = decrypt_text.strip(b"\x00")
    return decrypt_text


if __name__ == '__main__':
    key = "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A"

    # 序类化模型，输出格式bytes
    model_path = "ocr-text-detector_bgr-None_tf26-x86_cpu-fp32.h5"
    text = open(model_path, 'rb').read()

    # AES加密
    en_text = aesencrypt(text, key)
    pickle.dump(en_text, open(model_path.replace("h5", "model"), "wb"))
```





## 模型解密

```
import pickle
from Crypto.Cipher import AES


def add_16(par):
    if type(par) == str:
        par = par.encode()
    while len(par) % 16 != 0:
        par += b'\x00'
    return par


def aesdecrypt(model_path, key):
    text = pickle.load(open(model_path.replace("h5", "model"), "rb"))

    key = add_16(key)
    aes = AES.new(key, AES.MODE_ECB)
    decrypt_text = aes.decrypt(text)
    decrypt_text = decrypt_text.strip(b"\x00")
    return decrypt_text


if __name__ == '__main__':
    key = "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A"

    # 序类化模型，输出格式bytes
    model_path = "ocr-text-detector_bgr-None_tf26-x86_cpu-fp32.model"

    en_text = pickle.load(open(model_path.replace("h5", "model"), "rb"))

    # 解密
    text_dec = aesdecrypt(en_text, key)
    print("success")

```








# 参考文献 #