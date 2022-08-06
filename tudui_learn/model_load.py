import torch

# 第一种加载方式
import torchvision.models
from model_save import MyModule

vgg = torch.load("mymodel/model_save_1.pth")

# 第二种

vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("mymodel/model_save_2.pth"))

# 注意

module = torch.load("mymodel/mymodel.pth")
print("ok")
