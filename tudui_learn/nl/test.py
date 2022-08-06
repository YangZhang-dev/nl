# -*- coding: UTF-8 -*-
# Create by YangZhang on 2022/8/5
import torch
import torchvision.transforms
from PIL import Image
from model import MyModule

# from train import train_data

img_path = "test_img/dog_0.jpg"
img = Image.open(img_path)

compose = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
img = compose(img)
img = torch.reshape(img, (1, 3, 32, 32))

model = torch.load("./model_save/model_9.pth", map_location="cpu")
model.eval()
with torch.no_grad():
    outputs = model(img)
print(outputs.argmax(1))
# print(train_data.classes[outputs.argmax(1)])
