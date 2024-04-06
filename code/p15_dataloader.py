import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root='./dataset14', train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

img, target = test_set[0]
print(img)
print(target)
print(type(img))
print(type(target))
print(test_loader)

writer = SummaryWriter("logs15")
step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images("test_data_drop_last_true", imgs, step)
    step += 1

writer.close()
