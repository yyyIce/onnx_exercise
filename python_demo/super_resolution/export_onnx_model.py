# 教程：https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
# 中文：
# 来源：https://github.com/pytorch/tutorials/blob/master/advanced_source/super_resolution_with_onnxruntime.py
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init

import onnx
import onnxruntime

from PIL import Image
import torchvision.transforms as transforms


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def export_onnx_model():
    # Create the super-resolution model by using the above model definition.
    torch_model = SuperResolutionNet(upscale_factor=3)

    model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
    batch_size = 1    # just a random number

    # Initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

    # set the model to inference mode
    torch_model.eval()

    # Input to the model，需要随便输入一个类型和大小正确的张量x，用于执行模型，让export能记录使用什么运算符计算输出的轨迹。
    # x的形状为(batch_size,1,224,224)即batch_size个通道数为1，图像高度为224，图像宽度为224的
    x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model,               # model being run
                  x,                             # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",       # where to save the model (can be a file or file-like object)
                  export_params=True,            # store the trained parameter weights inside the model file
                  opset_version=10,              # the ONNX version to export the model to
                  do_constant_folding=True,      # whether to execute constant folding for optimization
                  input_names = ['input'],       # the model's input names
                  output_names = ['output'],     # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
    
    return torch_model

def verify_onnx_model(torch_model):
    # 验证模型的有效性以及导出的ONNX模型是否和Pytorch模型精度匹配。
    onnx_model = onnx.load("super_resolution.onnx")

    # 验证onnx模型的有效性
    onnx.checker.check_model(onnx_model)

    # 创建会话
    ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

    # compute ONNX Runtime output prediction
    batch_size = 1    # just a random number
    x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
    torch_out = torch_model(x)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    # 使用run() API评估模型，它的输出是一个列表，其中包含由 ONNX Runtime 计算的模型的输出。
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    # 如果PyTorch 和 ONNX Runtime 的输出在数值上与给定的精度匹配(rtol = 1e-03 和 atol = 1e-05），说明从Pytorch导出的ONNX模型没有问题，如果不匹配，请联系Pytorch官方
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    return ort_session

def inference_onnx_demo(ort_session):
    img = Image.open("./img/cat.jpg")

    resize = transforms.Resize([224, 224])
    img = resize(img)

    '''
    加载图片，使用标准 PIL python 库对其进行预处理。 请注意，此预处理是处理数据以训练/测试神经网络的标准做法。
    我们首先调整图像大小以适合模型输入的大小(224x224）。 然后，我们将图像分为 Y，Cb 和 Cr 分量。 
    这些分量代表灰度图像(Y），蓝差(Cb）和红差(Cr）色度分量。 Y 分量对人眼更敏感，我们对将要转换的 Y 分量感兴趣。 
    提取 Y 分量后，我们将其转换为张量，这将是模型的输入。
    '''
    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()

    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)
    img_y.unsqueeze_(0)

    # 使用代表灰度尺寸调整后的猫图像的张量，并按照先前的说明在 ONNX Runtime 中运行超分辨率模型
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]
    # 输出shape为(1, 1, 672, 672)
    print(ort_outs[0].shape)

    # 模型的输出为张量。 处理模型的输出，根据输出张量构造最终的输出图像，并保存图像。
    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

    # get the output image follow post-processing step from PyTorch implementation
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]).convert("RGB")

    # Save the image, we will compare this with the output image from mobile device
    final_img.save("./img/cat_superres_with_ort.jpg")

if __name__ == '__main__':
    torch_model = export_onnx_model()
    ort_session = verify_onnx_model(torch_model)
    inference_onnx_demo(ort_session)
