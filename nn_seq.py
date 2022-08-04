import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter


class MyModule(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.module1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        return self.module1(input)


input = torch.ones(64, 3, 32, 32)
module = MyModule()
output = module(input)
writer = SummaryWriter("./sequential")
writer.add_graph(module, input)
writer.close()