# -*- coding: UTF-8 -*-
# Create by YangZhang on 2022/8/6
import torch
from torch import float32

A = torch.arange(12, dtype=float32).reshape(3, 4)
print("---1---")
# 矩阵转置
print(A.T)

print("---2---")
# 对称矩阵
t = torch.tensor([[1, 0], [0, 1]])
print(t == t.T)

print("---3---")
# 指定维度求和,非降维求和(可用于广播机制)
print(A)
print(A.sum(axis=0)) # axis为几，就消掉哪个维度（那个维度的值为零）
print(A.sum(axis=1))
print(A.sum(axis=0, keepdims=True))  # 计算指定维度的元素和，保留维度(即被消掉的维度的维度值不会消失，转化为一)
print(A / (A.sum(axis=1, keepdims=True)))  # 每个元素在本行的占比
print(A.cumsum(axis=0))  # 计算每个元素在指定维度的和并保留每个元素

print("---4---")
# 求平均
print(A.mean())
print(A.mean(axis=0))

print("---5---")
# 一维向量点乘
y = torch.ones(4, dtype=torch.float32)
x = torch.zeros(4, dtype=torch.float32)
print(x, y, torch.dot(x, y))

print("---6---")
# 向量积（m*v）
print(A.shape, x.shape, torch.mv(A, y))

print("---7---")
# 矩阵乘法
B = torch.ones(4, 3)
print(torch.mm(A, B))

print("---8---")
# L1范数
u = torch.tensor([3.0, -4.0])
print(torch.abs(u).sum())

print("---9---")
# L2范数
print(torch.norm(u))

print("---10---")
# Frobenius范数
print(torch.ones((4, 9)))
print(torch.norm(torch.ones((4, 9))))

# L2是针对向量，F是针对矩阵，F可以看作是讲矩阵拉成向量

t = torch.ones(12).reshape((2, 3, 2))
print(t.shape)
print(t)
print(t.sum(axis=[0, 1],keepdims=True).shape)
