import torch
import torchvision
from torch.nn import MaxPool2d
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self, x):
        x = self.maxpool1(x)
        return x


tudui = Tudui()
output = tudui(input)
print(output)



dataset = torchvision.datasets.CIFAR10("./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)


tudui = Tudui()
writer = SummaryWriter("logs19")
step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step += 1

writer.close()
