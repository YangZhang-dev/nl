import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)

loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, drop_last=False, num_workers=0)


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        output = self.conv1(x)
        return output


module = MyModule()
step = 0
writer = SummaryWriter("./log")
for data in loader:
    img, tar = data
    conv_img = module(img)
    writer.add_images("img", img, global_step=step)
    conv_img=torch.reshape(conv_img, (-1, 3, 30, 30))
    writer.add_images("conv_img", conv_img, global_step=step)
    step = step + 1

writer.close()
