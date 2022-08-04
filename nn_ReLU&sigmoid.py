import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, drop_last=False, num_workers=0)


class MyModule(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # self.ReLU1 = ReLU(inplace=False)  # True表示直接覆盖原来的值，而FALSE是新产生一个output
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


module = MyModule()
writer = SummaryWriter("./sigmoid")
step = 0
for data in loader:
    img, tar = data
    writer.add_images("img", img, step)
    sigmoid_img = module(img)
    writer.add_images("after_sigmoid_img", sigmoid_img, step)
    step += 1
writer.close()