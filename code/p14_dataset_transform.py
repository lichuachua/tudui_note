import torchvision

train_set = torchvision.datasets.CIFAR10(root='./dataset14', train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset14', train=False, download=True)

print(test_set[0])
print(test_set.classes)
img, target = test_set[0]
print(img)
print(target)
print(test_set.classes[target])
img.show()
