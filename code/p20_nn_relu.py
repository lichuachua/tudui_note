from torch.nn import ReLU, Sigmoid

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        # output = self.relu1(input)
        output = self.sigmoid1(input)
        return output


tudui = Tudui()
output = tudui(input)
print(output)

dataset = torchvision.datasets.CIFAR10("./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)

tudui = Tudui()
writer = SummaryWriter("logs20")
step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    writer.add_images("input", imgs, global_step=step)
    writer.add_images("output", output, step)
    step += 1

writer.close()
