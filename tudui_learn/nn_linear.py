import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, drop_last=False, num_workers=0)


class MyModule(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


module = MyModule()
step = 0
writer = SummaryWriter("linear")
for data in loader:
    img, tar = data
    writer.add_images("img", img, step)
    img = torch.flatten(img)
    after_linear = module(img)
    temp = torch.reshape(after_linear, (1, 1, 1, 10))
    print(temp.shape)
    # writer.add_images("after_linear_img", temp, step)
    # step += 1

writer.close()
