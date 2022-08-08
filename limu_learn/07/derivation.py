# -*- coding: UTF-8 -*-
# Create by YangZhang on 2022/8/7


import torch

print("---1---")
x = torch.arange(4.0)
print("x:", x)
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
print(x.grad)  # 默认值是None
y = 2 * torch.dot(x, x)  # y=2*x^2
print(y)
y.backward()
print(x.grad)
print(x.grad == 4 * x)

print("---2---")
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
print(y)
y.backward()
print(x.grad)

print("---3---")
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 在我们的例子中，我们只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)

print("---4---")
# 分离计算
x.grad.zero_()
y = x * x
u = y.detach()  # y当常数
z = u * x
z.sum().backward()
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)  # 均值为0，方差为一的正态分布张量
d = f(a)
d.backward()
print(a.grad == d / a)
