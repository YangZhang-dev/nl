import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)

loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, drop_last=False, num_workers=0)

img, tar = test_set[0]
print(img.shape)
print(tar)

step = 0
writer = SummaryWriter("dataloader/")

for data in loader:
    img, tar = data
    # 注意有s
    writer.add_images("dataloader_test", img, step)  # add_images: Add batched image data to summary
    step = step + 1

writer.close()
