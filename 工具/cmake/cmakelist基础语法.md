

## 基础语法

```
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)  #设置最小版本
Project(cuda_code CXX C CUDA)  #设置项目
set(CMAKE_CUDA_FLAGS "-arch=compute_35 -g -G -O3") #设置cuda变量
include_directories(./include)  #添加头文件
add_subdirectory(4_sum_arrays_timer) #添加子文件
```











# 参考文献 #