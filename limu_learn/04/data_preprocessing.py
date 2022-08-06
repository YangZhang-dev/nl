# -*- coding: UTF-8 -*-
# Create by YangZhang on 2022/8/6

import os
import pandas as pd
import torch

os.makedirs(os.path.join('.', 'data'), exist_ok=True)
data_file = os.path.join('.', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('col1,col2,col3\n')  # 列名
    f.write('not,ok,1\n')
    f.write('ok,not,\n')

print("---1---")
# pandas 读取csv
data = pd.read_csv(data_file)
print(data)
print("\n")

print("---2---")
# 插入缺失值,以当前列的平均值
input = data.iloc[:, 2]
input = input.fillna(input.mean())
print(input)

print("---3---")
# 生成one-hot，并转化为tensor
onehot = pd.get_dummies(data.iloc[:, 0:2], dummy_na=False)  # dummy_na是指是否生成非空类别
print(onehot)
print(torch.tensor(onehot.values))
