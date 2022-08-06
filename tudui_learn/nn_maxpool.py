import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)

loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, drop_last=False, num_workers=0)


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        output = self.maxpool1(x)
        return output


module = MyModule()
step = 0
writer = SummaryWriter("./maxpool")
for data in loader:
    img, tar = data
    conv_img = module(img)
    writer.add_images("after_maxpool_img", img, global_step=step)
    conv_img = torch.reshape(conv_img, (-1, 3, 30, 30))
    writer.add_images("max_img", conv_img, global_step=step)
    step = step + 1

writer.close()
