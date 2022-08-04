import torch
from torch import float32
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

output = torch.tensor([1, 2, 3], dtype=float32)
target = torch.tensor([1, 2, 4], dtype=float32)

loss1 = L1Loss()
loss2 = L1Loss(reduction="sum")
res1 = loss1(output, target)
res2 = loss2(output, target)
print(res1)
print(res2)

# 回归问题
loss3 = MSELoss()
res3 = loss3(output, target)
print(res3)

# 分类问题
loss4 = CrossEntropyLoss()
output1 = torch.tensor([0.1, 0.2, 0.3])
output1 = torch.reshape(output1, (1, 3))
target1 = torch.tensor([1])
res4 = loss4(output1, target1)
print(res4)
