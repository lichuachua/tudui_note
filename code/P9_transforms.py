import cv2
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 通过transforms.ToTensor()看两个问题
# 2.为什么需要tensor类型

# PIL Images

img_path = "../data/hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)
print(img)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

print(tensor_img)
print()
print()
print()
# numpy.ndarray
img2 = cv2.imread(img_path)
print(img2)
tensor_img2 = tensor_trans(img2)

print(tensor_img2)

# 显示
writer = SummaryWriter("logs")

# 1.如何在python中使用transforms
writer.add_image("Tensor_img", tensor_img, 1)
writer.add_image("Tensor_img", tensor_img2, 2)

writer.close()
