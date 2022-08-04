import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d, Conv2d, Sequential, Flatten, Linear
from torch.utils.data import DataLoader

test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)

loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, drop_last=False, num_workers=0)


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


module = MyModule()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(module.parameters(), lr=0.01, momentum=0.9)
for epoch in range(20):
    loss_res = 0.0
    for data in loader:
        img, tar = data
        output = module(img)
        res = loss(output, tar)
        optim.zero_grad()
        res.backward()
        optim.step()
        loss_res += res
    print(loss_res)
