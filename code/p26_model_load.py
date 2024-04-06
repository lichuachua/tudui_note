import torch
import torchvision
from torch import nn

model1 = torch.load("vgg16_model1.pth")
print(model1)

# model2 = torch.load("vgg16_model2.pth")
# print(model2)


vgg16 = torchvision.models.vgg16(weights=False)
vgg16.load_state_dict(torch.load("vgg16_model2.pth"))
print(vgg16)

# 陷阱1
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

model = torch.load('tudui_method1.pth')
print(model)
