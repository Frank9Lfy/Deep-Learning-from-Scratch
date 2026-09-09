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
# def softmax(a):
#     c = np.max(a)  # 为了防止溢出，减去最大值
#     exp_a1 = np.exp(a - c)
#     sum_exp_a = np.sum(exp_a1)
#     y_ = exp_a1 / sum_exp_a
#     return y_
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

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

def cross_entropy_error_batch(y, t):
    if y.ndim == 1:  # 如果y是一维数组（单个样本）
        t = t.reshape(1, t.size)  # 将t转换为二维数组
        y = y.reshape(1, y.size)  # 将y转换为二维数组

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]  # 获取批量大小
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# 数值微分
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

# 梯度
# def numerical_gradient(f, x):
#     h = 1e-4
#     grad = np.zeros_like(x)  # 创建一个与x形状相同的数组，元素全为0
#     for idx in range(x.size):
#         tmp_val = x[idx]
#         x[idx] = tmp_val + h  # 计算f(x+h)
#         fxh1 = f(x)  # f(x+h)
#         x[idx] = tmp_val - h  # 计算f(x-h)
#         fxh2 = f(x)  # f(x-h)
#         grad[idx] = (fxh1 - fxh2) / (2*h)  # 中心差分公式计算偏导数
#         x[idx] = tmp_val  # 恢复当前元素的值
#     return grad

def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad
"https://chat.deepseek.com/share/tbmubkpj7xpcnx0qc7"
"上述数值计算梯度代码（多维）的语法解释与拓展"

# 梯度下降法
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num): # 重复迭代100次
        grad = numerical_gradient(f, x)  # 计算梯度
        x -= lr * grad  # 沿着梯度的反方向更新参数
    return x

# 简单神经网络构造
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 初始化权重，正态分布随机数(高斯分布)，形状为(2,3)
    def predict(self, x1):
        return np.dot(x1, self.W)  # 前向传播，计算输出
    def loss(self, x1, t):
        z = self.predict(x1)  # 预测值
        y0 = softmax(z)  # softmax函数，预测值
        loss = cross_entropy_error(y0, t)  # 损失函数
        return loss