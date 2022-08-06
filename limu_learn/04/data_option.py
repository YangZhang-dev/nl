# -*- coding: UTF-8 -*-
# Create by YangZhang on 2022/8/6
import torch

x = torch.arange(6)
print(x)
print("---1---")
# shape访问张量的形状
print(x.shape)
print("---2---")
# numel访问元素数量
print(x.numel())

print("---3---")
# reshape 改变张量的形状而不改变元素的数量和元素值
x = x.reshape(3, 2)
print(x)

print("---4---")
# 全0，全1，其他常量或从特定分布随机采样的数字
zero = torch.zeros((2, 3, 4))
print(zero)
ones = torch.ones((2, 3, 4))
print(ones)

print("---5---")
# 通过列表来确定一个张量
y = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(y.shape)

print("---6---")
# tensor可以直接四则运算(形状相同)
x = x.reshape(2, 3)
print(x - y)
ex = torch.exp(x)  # e的x次方

print("---7---")
# 张量的连结
get1 = torch.cat((x, y), dim=0)  # 在行上进行叠加
get2 = torch.cat((x, y), dim=1)  # 在列上叠加

print("---8---")
# 逻辑运算构建张量
print(x == y)

print("---9---")
# 张量所有元素求和
xsum = x.sum()
print(xsum)

print("---10---")
# 广播机制（易错）
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a + b)

print("---11---")
# 转化为Numpy
print(type(x.numpy()))
print(type(torch.tensor(x.numpy())))

print("---12---")
# 大小为一的tensor转化为标量
one = torch.tensor([1])
print(one, one.item(), float(one), int(one))
