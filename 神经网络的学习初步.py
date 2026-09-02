import sys, os
sys.path.append(os.pardir) 
import numpy as np
from dataset.mnist import load_mnist

print("学习--是从训练数据中获取到某种规律\n"
"此处指自动获取最优权重参数的过程")

"https://chat.deepseek.com/share/5uzunablzlmh0oj2zl"
"问deepseek特征量相关知识"
"在神经网络中，重要的特征量都是机器来学习的，而不是人工来设计的。"
print('='*15,"损失函数",'='*20)
print("""损失函数：衡量预测值与真实值之间的差距
作为神经网络的学习指标，损失函数的值越小，说明输出的预测值与真实值之间的差距越小，性能越好。
一般用均方误差（MSE）或交叉熵误差（CEE）作为损失函数。""")
print("均方误差（MSE）：(1/n)*Σ(yi-ti)^2  yi表示神经网络的输出，ti表示监督数据（即训练数据,t_train）,i表示样本的总索引（数据的维数）")
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
# 设“2”为正确解
t0 = np.array([0,0,1,0,0,0,0,0,0,0])  # 正确解，one-hot编码形式
# 例一："2"概率最高
y1 = np.array([0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.0,0.0,0.0])  # 神经网络的输出1
print("均方误差y1：",mean_squared_error(y1, t0))  # 0.0975
# 例二："7"概率最高
y2 = np.array([0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0])  # 神经网络的输出2
print("均方误差y2：",mean_squared_error(y2, t0))  # 0.5975

print("交叉熵误差（CEE）：-Σti*log(yi)  yi表示神经网络的输出，ti表示正确解标签（即训练数据,t_train）,i表示样本的总索引（数据的维数）" \
"\n交叉熵误差的值是由正确解标签的输出结果决定的。交叉熵衡量两个概率分布之间的差异，常用于分类问题中。")
def cross_entropy_error(y, t):
    delta = 1e-7  # 防止对数为无穷大
    return -np.sum(t * np.log(y + delta))  
print("交叉熵误差y1：",cross_entropy_error(y1, t0))  # 0.510825457099338
print("交叉熵误差y2：",cross_entropy_error(y2, t0))  # 2.302584092994546

print("="*15,"小批量学习（mini-batch）",'='*20) 
# 类比批处理，见手写数字识别推理处理部分（前向传播）

(x_train, t_train), (x_test, t_test) = \
load_mnist(one_hot_label=True, normalize=True)

print("从训练数据集中随机抽取10个样本")
train_size = x_train.shape[0]
print("train_size:", train_size)  # 60000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # 随机抽取10个样本的索引
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print("x_batch.shape:", x_batch.shape)  # (10, 784)
print("t_batch.shape:", t_batch.shape)  # (10, 10)
print("np.random.choice(train_size, batch_size):", batch_mask)  # 随机抽取10个样本的索引
print("mini-batch的损失函数：利用一部分样本数据的平均损失函数来近似整个训练数据集的损失函数")
print("mini-batch版CEE的实现")
def cross_entropy_error1(y, t):  # 监督数据为one-hot编码形式
    if y.ndim == 1:  # 如果y是一维数组（单个样本）
        t = t.reshape(1, t.size)  # 将t转换为二维数组
        y = y.reshape(1, y.size)  # 将y转换为二维数组
    batch_size = y.shape[0]  # 获取批量大小
    return -np.sum(t * np.log(y + 1e-7)) / batch_size  # 返回平均交叉熵误差
def cross_entropy_error2(y, t):  # 监督数据为标签形式
    if y.ndim == 1:  # 如果y是一维数组（单个样本）
        t = t.reshape(1, t.size)  # 将t转换为二维数组
        y = y.reshape(1, y.size)  # 将y转换为二维数组
    batch_size = y.shape[0]  # 获取批量大小
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size  
'''
语法解释
np.arange(batch_size)
生成 [0, 1, 2, ..., batch_size-1]
例如：batch_size = 3 → [0, 1, 2]
y[np.arange(batch_size), t]
这是"花式索引"（fancy indexing）
y 形状为 (3, 10)（3 个样本，10 个类别）
假设t 为 [2, 5, 1]（三个样本的正确类别标签）
y[[0,1,2], [2,5,1]] 取出 [y[0,2], y[1,5], y[2,1]]
也就是：第 0 个样本的第 2 类概率、第 1 个样本的第 5 类概率、第 2 个样本的第 1 类概率
结果形状为 (3,)
'''
"https://chat.deepseek.com/share/3f95xik388mwgntosn"
"针对损失函数的设计原因（书p92-93），问deepseek"

print("="*15,"数值微分",'='*20)
# 导数
# bad example
def numerical_diff0(f, x):
    h = 1e-50  # 过小的h会导致数值误差(舍入误差)
    return (f(x+h) - f(x)) / h  # 使用的是前向差分，计算结果一定不是该点导数
print("1e-50:", np.float32(1e-50), "1e-4:", np.float32(1e-4), "1e-5:", np.float32(1e-5))  
# good example
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h) # 使用的是中心差分，计算结果接近导数

def function_1(x):
    return 0.01*x**2 + 0.1*x

import matplotlib.pyplot as plt
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()
print("x=5时的导数：", numerical_diff(function_1, 5))  # 0.1999999999990898
print("x=10时的导数：", numerical_diff(function_1, 10))  # 0.2999999999986347
# 偏导
def function_2(x):  # x是一个二维数组,假设x = np.array([x0, x1])
    return x[0]**2 + x[1]**2
    # 或者：return np.sum(x**2)  # 也可以使用numpy的广播机制
# 函数图像见复杂图像.py

print("x0=3, x1=4时,关于x0的偏导数：", numerical_diff(lambda x: function_2(np.array([x, 4])), 3))  # 6.00000000000378
print("x0=3, x1=4时,关于x1的偏导数：", numerical_diff(lambda x: function_2(np.array([3, x])), 4))  # 7.999999999999119
'''
lambda x: function_2(np.array([x, 4]))表示一个匿名函数
学过c++的同学们，应该很熟悉lambda表达式了🐶
'''
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
print("x0=3, x1=4时的梯度：", numerical_gradient(function_2, np.array([3.0, 4.0])))  # [6. 8.]
# 梯度可视化（含等高线）见复杂图像.py
print("="*15,"梯度下降法",'='*20)
print("""梯度法：沿着梯度的(反)方向更新参数，直到找到最值
神经网络中，梯度法指梯度下降法，寻找最小值
""")
print("梯度下降法的实现")
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num): # 重复迭代100次
        grad = numerical_gradient(f, x)  # 计算梯度
        x -= lr * grad  # 沿着梯度的反方向更新参数
    return x
print("迭代公式：x0 = x0 - lr * ∂f/∂x0\n\t  x1 = x1 - lr * ∂f/∂x1" \
"\n其中，lr为学习率（即步长）")

"https://chat.deepseek.com/share/vu49hp52i9zo3unoq6"
"拓展：针对梯度下降法与牛顿迭代法，问deepseek（没有精细化，可追问）"
# 梯度法求函数f(x0, x1) = x0^2 + x1^2的最小值
init_x = np.array([-3.0, 4.0])
print("初始值：", init_x)
print("最小值：", gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)) 
init_x = np.array([-3.0, 4.0]) 
print("学习率过大：", gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))
init_x = np.array([-3.0, 4.0])
print("学习率过小：", gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))

print("\n超参数：学习率lr、迭代次数step_num等参数需要人工设定，称为超参数。超参数需要尝试不同的值，才能找到最优的超参数组合。\n")

print("""神经网络的梯度：损失函数关于权重参数的梯度
\n""")
from 工具函数文件 import softmax
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 初始化权重，正态分布随机数(高斯分布)，形状为(2,3)
    def predict(self, x):
        return np.dot(x, self.W)  # 前向传播，计算输出
    def loss(self, x, t):
        z = self.predict(x)  # 预测值
        y = softmax(z)  # softmax函数，预测值
        loss = cross_entropy_error(y, t)  # 损失函数
        return loss