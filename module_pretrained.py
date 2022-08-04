import torchvision
from torch import nn
from torch.nn.functional import linear

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

vgg16_true.add_module("my_linear", nn.Linear(1000, 10))

vgg16_false.classifier[6] = nn.Linear(4096, 10)
