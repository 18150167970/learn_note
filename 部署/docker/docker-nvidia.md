## 安装docker

### 卸载和安装依赖

```
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg lsb-release
```

### 添加Docker’s official GPG key:
```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```

```
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### 安装

```
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo docker run hello-world
```



### 配置源

打开

```
sudo vim /etc/docker/daemon.json
```

添加

```
{
   "registry-mirrors": ["http://hub-mirror.c.163.com"]
}
```

并重启docker



### nvidia-docker

```
curl https://get.docker.com | sh

sudo systemctl start docker && sudo systemctl enable docker

# 设置stable存储库和GPG密钥：
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 要访问experimental诸如WSL上的CUDA或A100上的新MIG功能之类的功能，您可能需要将experimental分支添加到存储库列表中.
# 可加可不加
curl -s -L https://nvidia.github.io/nvidia-container-runtime/experimental/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list

# nvidia-docker2更新软件包清单后，安装软件包（和依赖项）：
sudo apt-get update

sudo apt-get install -y nvidia-docker2

# 设置默认运行时后，重新启动Docker守护程序以完成安装：
sudo systemctl restart docker

```






# 参考文献 #

https://docs.docker.com/engine/install/ubuntu/