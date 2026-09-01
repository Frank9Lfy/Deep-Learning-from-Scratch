# -*- coding: utf-8 -*-
import numpy as np


# 阶跃函数
def step_function0(x):  # 参数x只能是一个实数，不能是数组
    if x > 0:
        return 1
    else:
        return 0
def step_function1(x):   # 参数x可以是数组
    y = x > 0
    print(y)  # 显示布尔值数组
    print(y.astype(np.int32))  # 将布尔值数组转换为整数数组
    return y.astype(np.int64) # 返回布尔值数组，True为1，False为0
def step_function(x):
    return np.array(x > 0, dtype=np.int64)  # 返回布尔值数组，True为1，False为0
# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# relu函数
def relu(x):
    return np.maximum(0, x)
# softmax函数
def softmax(a):
    c = np.max(a)  # 为了防止溢出，减去最大值
    exp_a1 = np.exp(a - c)
    sum_exp_a = np.sum(exp_a1)
    y_ = exp_a1 / sum_exp_a
    return y_
