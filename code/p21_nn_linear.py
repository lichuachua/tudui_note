import torch
import torchvision
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn

dataset = torchvision.datasets.CIFAR10("./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64,drop_last=True)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear = Linear(196608,10)

    def forward(self, input):
        output = self.linear(input)
        return output

tudui =Tudui()
writer = SummaryWriter("logs21")
step = 0
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = tudui(output)
    print(output.shape)