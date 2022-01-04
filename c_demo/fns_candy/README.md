# FNS Candy
FNS Candy is a style transfer model. In this sample application, we use the ONNX Runtime C API to process an image using the FNS Candy model in ONNX format.

# Build Instructions
See [../README.md](../README.md)

# Prepare data
First, download the FNS Candy ONNX model from [here](https://raw.githubusercontent.com/microsoft/Windows-Machine-Learning/master/Samples/FNSCandyStyleTransfer/UWP/cs/Assets/candy.onnx).

Then, prepare an image:
1. PNG format
2. Dimension of 720x720

# Run
Command to run the application:
```
fns_candy_style_transfer.exe <model_path> <input_image_path> <output_image_path> [cpu|cuda|dml]
```

To use the CUDA or DirectML execution providers, specify `cuda` or `dml` on the command line. `cpu` is the default.



# 补充说明

　　原示例地址：https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/fns_candy_style_transfer

　　本示例改编自ONNX Runtime examples官方示例，在CentOS8系统运行无问题，其他Linux系统没有尝试过。

- img：目录下为模型的输入，格式为720*720的PNG图片，有一张示例输入图片；
- lib：目录下为onnxruntime和libpng动态库；
- model：目录下是ONNX模型；
- output：目录下是示例输出图片；



# 环境配置

创建onnxruntime和libpng动态库，出现Permission Denied请自行在命令前添加`sudo`或进入root权限`su root`：

```shell
tar xf onnxruntime-linux-x64-1.10.0.tgz
unzip -d ./libpng libpng.zip
cp -rfv onnxruntime-linux-x64-1.10.0/include/* /usr/local/include/
cp -rfv onnxruntime-linux-x64-1.10.0/lib/* /usr/local/lib/
cp -rfv libpng/include/* /usr/local/include/
cp -rfv libpng/* /usr/local/lib/
cat /etc/ld.so.conf
echo "/usr/local/lib" >> /etc/ld.so.conf
ldconfig
```

编译：

```shell
gcc -o fns_candy_style_transfer fns_candy_style_transfer.c -lonnxruntime -lpng
```

运行示例：

```shell
./fns_candy_style_transfer ./model/candy.onnx ./img/photo.png ./output/photo_infer.png cpu
```



