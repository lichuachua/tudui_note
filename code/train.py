import torch

# 假设你有一个形状为 [4, 32, 32] 的图像张量 image
image = torch.randn(4, 32, 32)

# 选择前三个通道
image_rgb = image[:3, :, :]

# 打印结果形状
print("转换后的图像形状:", image_rgb.shape)
