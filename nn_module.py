import torch
from torch import nn


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x + 1


module = MyModule()
tensor = torch.Tensor(1)
out = module(tensor)
print(out.size)

