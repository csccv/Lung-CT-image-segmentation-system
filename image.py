import cv2

# 读取图像
image = cv2.imread(r'E:\PycharmProjects\Unet\data\CXR_png\MCUCXR_0001_0.png')

# 获取图像的形状（高度、宽度和通道数）
height, width, channels = image.shape

print(f"Image Shape: Height={height}, Width={width}, Channels={channels}")

from PIL import Image

# 打开图像
image = Image.open(r'E:\PycharmProjects\Unet\data\Mask\MCUCXR_0001_0.png')

# 获取图像的通道数
channels = image.getbands()

print("Image Channels:", channels)
