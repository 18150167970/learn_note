**安装扩展**

1. vscode-cudacpp
   代码高亮
2. Nsight Visual Studio Code Edition
   debug

**配置文件**

按Ctrl+Shift+P或者点击查看-命令面板，调出命令面板

输入C/C++,选择C/C++:编辑配置(JSON)或者C/C++:编辑配置(UI)，实测两个默认产生的文件是一样的，但是推荐选择C/C++:编辑配置(UI)，选择C/C++:编辑配置(UI)会出现一个配置页面，比较方便，而选择编辑配置(JSON)只会产生一个配置文件。

c_cpp_properties.json

```bash
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/local/cuda-10.2/include"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/clang",
            "cStandard": "c11",
            "cppStandard": "c++14",
            "intelliSenseMode": "linux-clang-x64"
        }
    ],
    "version": 4
}
```

launch.json

```kotlin
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "(gdb) Launch",
            "type": "cppdbg",
            "request": "launch",
            "program": "${fileDirname}/${fileBasenameNoExtension}.out",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${fileDirname}",
            "environment": [],
            "externalConsole": false,
            "preLaunchTask": "build",
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "sourceFileMap": {"/build/glibc-S9d2JN": "/usr/src/glibc"}
        }
    ]
}
```

tasks.json

```kotlin
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build",
            "type": "shell",
            "command": "nvcc",
            "args":["-g","${file}","-o","${fileDirname}/${fileBasenameNoExtension}.out",
                // include 头文件
                "-I", "/usr/local/cuda/include",
                "-I", "/usr/local/cuda-10.2/samples/common/inc",
                // lib 库文件地址
                "-L", "/usr/local/cuda/lib64",   
                "-L", "/usr/local/cuda-10.2/samples/common/lib",  
                "-l", "cudart",                           
                "-l", "cublas",
                "-l", "cudnn",
                "-l", "curand",
                "-D_MWAITXINTRIN_H_INCLUDED"  
            ]
        }
    ]
}
```

**安装glibc**
这是因为调试cuda时，最后提示 vscode 无法打开 libc-start.c

```bash
sudo apt install glibc-source  
cd /usr/src/glibc/  
sudo tar -xvf glibc-2.27.tar.xz
```



# 参考文献 #

https://www.cnblogs.com/DLCannotBeAccelerated/p/15612820.html