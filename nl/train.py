# -*- coding: UTF8 -*-
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from model import MyModule

# 定义训练的设备

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("train is on {}".format(device))
# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
# 打印数据集的长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("Length of training dataset is:{}".format(train_data_size))
print("Length of testing dataset is:{}".format(test_data_size))

# 加载数据
train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, drop_last=False, num_workers=0)
test_dataloader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, drop_last=False, num_workers=0)

# 创建网络模型
module = MyModule()
module = module.to(device)

# 创建损失函数(分类问题使用交叉熵)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learn_rate = 1e-2
optimizer = torch.optim.SGD(module.parameters(), lr=learn_rate)

# 设置训练网络的一些参数

# 训练的次数
total_train_step = 0
# 测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("./logs")

start_time = time.time()
for i in range(epoch):
    print("----Start of training Round {}----".format(i + 1))

    # 开始训练
    module.train()
    for data in train_dataloader:
        img, targets = data
        img = img.to(device)
        targets = targets.to(device)
        outputs = module(img)
        loss = loss_fn(outputs, targets)
        # 优化器梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("Training times: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss, total_train_step)

    # 测试步骤开始
    module.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            img, targets = data
            img = img.to(device)
            targets = targets.to(device)
            outputs = module(img)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            total_accuracy += (outputs.argmax(1) == targets).sum()
        total_test_step += 1
        print("Loss of the overall test set: {}".format(total_test_loss))
        print("Overall accuracy in test dataset is {}%".format(total_accuracy / test_data_size * 100))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)

    torch.save(module, "./model_save/model_{}.pth".format(i))
    print("model is saved")
over_time = time.time()
print("time is {}".format(over_time - start_time))
writer.close()
