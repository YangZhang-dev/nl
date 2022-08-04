import torch
import torchvision.models

# 模型保存方式一
from torch import nn

vgg = torchvision.models.vgg16(pretrained=False)

torch.save(vgg, "./mymodel/model_save_1.pth")

# 方式二

torch.save(vgg.state_dict(), "./mymodel/model_save_2.pth")


# 注意
class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        output = self.conv1(x)
        return output


module = MyModule()
torch.save(module,"mymodel/mymodel.pth")
