import torchvision
import ssl

from torch.utils.tensorboard import SummaryWriter

compose = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

ssl._create_default_https_context = ssl._create_unverified_context
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=compose, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=compose, download=True)

print(test_set[0])
print(test_set.classes)

img, target = test_set[0]
print(test_set.classes[target])
# img.show()

writer = SummaryWriter("dataset_test/")

for i in range(10):
    img, tar = train_set[i]
    writer.add_image("dataset_test", img, i)
writer.close()
