import numpy as np
import onnxruntime
import torchvision.transforms as transforms
import cv2

def inference_onnx_demo():
    # 注意此处如果使用PIL，此时读取图像模式是RGB，所以如果我们使用PIL读入图像，得出的输出结果会与C示例不同。
    #img = Image.open("./img/photo.png") 
    # 注意CV2默认读取的是BGR，为与C的示例得到相同的输出，我们使用cv2读入图片
    img = cv2.imread("./img/photo.png")
    # print(img.shape)

    #img_arr = np.array(img).astype(np.float32)
    img_arr = img.astype(np.float32)
    img_arr = img_arr.transpose(2, 0, 1)
    # print(img_arr.shape)

    img_data = img_arr[np.newaxis, :, :, :]
    # print(img_data.shape)
    
    ort_session = onnxruntime.InferenceSession("candy.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: img_data}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out = ort_outs[0]
    
    # 这里也不需要np.uint8，cv2会处理，如果转换为np.uint8，会和C示例中的输出结果一致
    # 不转换会输出美观正常的图片，由此可见C示例导出的图像存在一点问题（绿色部分）。
    transposed_output_image = np.transpose(img_out, axes = [0,2,3,1]).squeeze()
    cv2.imwrite("./img/photo_candy.png", transposed_output_image)

if __name__ == '__main__':
    inference_onnx_demo()