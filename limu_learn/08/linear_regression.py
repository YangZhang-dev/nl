# -*- coding: UTF-8 -*-
# Create by YangZhang on 2022/8/7


import torch
from torch.utils import data


# 创建数据集

def synthetic_data(w, b, num_examples):  # @save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))  # 生成均值为0，标准差为1，长度为w的个数，数量为自定义的随机数x
    y = torch.matmul(X, w) + b  # 生成y
    y += torch.normal(0, 0.01, y.shape)  # 加入噪音
    return X, y.reshape((-1, 1))  # torch.Size([1000])->torch.Size([1000,1])


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)  # 对入参进行"打包"
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # shuffle 是否打乱


batch_size = 10
data_iter = load_array((features, labels), batch_size)
print(next(iter(data_iter)))

from torch import nn

# 定义模型
net = nn.Sequential(nn.Linear(2, 1))
# 初始化参数
# 通过net[0]选择网络中的第一个图层， 然后使用weight.data和bias.data方法访问参数。 还可以使用替换方法normal_和fill_来重写参数值。
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义回归函数
loss = nn.MSELoss()

# 定义优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 模型训练

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')


w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)