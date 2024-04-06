from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# ToTensor
img_path = "dataset/train/ants/5650366_e22b7e1065.jpg"
img = Image.open(img_path)
print(img)

writer = SummaryWriter("logs")
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
writer.add_image("ToTensor", img_tensor, 0)
# writer.close()

# Normalize——————output[channel] = (input[channel] - mean[channel]) / std[channel]
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.1, 0.1, 0.1], [0.1, 0.1, 0.1])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 3)

#
print(img)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
print(img_resize)

img_resize = trans_tensor(img_resize)
writer.add_image("Resize", img_resize, 0)

# Compose -> Resize ToTensor
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_tensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)
print(img_resize_2.shape)

# RandomCrop
trans_randomCrop = transforms.RandomCrop((100, 200))
trans_compose_2 = transforms.Compose([trans_randomCrop, trans_tensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
