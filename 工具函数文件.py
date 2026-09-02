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

# 均方误差（MSE）
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 交叉熵误差（CEE）
def cross_entropy_error(y, t):
    delta = 1e-7  # 防止对数为无穷大
    return -np.sum(t * np.log(y + delta))  
# mini-batch版交叉熵误差（CEE）
def cross_entropy_error_batch1(y, t):  # 监督数据为one-hot编码形式
    if y.ndim == 1:  # 如果y是一维数组（单个样本）
        t = t.reshape(1, t.size)  # 将t转换为二维数组
        y = y.reshape(1, y.size)  # 将y转换为二维数组
    batch_size = y.shape[0]  # 获取批量大小
    return -np.sum(t * np.log(y + 1e-7)) / batch_size  # 返回平均交叉熵误差
def cross_entropy_error_batch2(y, t):  # 监督数据为标签形式
    if y.ndim == 1:  # 如果y是一维数组（单个样本）
        t = t.reshape(1, t.size)  # 将t转换为二维数组
        y = y.reshape(1, y.size)  # 将y转换为二维数组
    batch_size = y.shape[0]  # 获取批量大小
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size  

# 数值微分
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

# 梯度
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # 创建一个与x形状相同的数组，元素全为0
    for idx in range(x.size):  
        tmp_val = x[idx]  
        x[idx] = tmp_val + h  # 计算f(x+h)
        fxh1 = f(x)  # f(x+h)
        x[idx] = tmp_val - h  # 计算f(x-h)
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)  # 中心差分公式计算偏导数
        x[idx] = tmp_val  # 恢复当前元素的值
    return grad

# 梯度下降法
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num): # 重复迭代100次
        grad = numerical_gradient(f, x)  # 计算梯度
        x -= lr * grad  # 沿着梯度的反方向更新参数
    return x