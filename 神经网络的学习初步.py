import sys, os
sys.path.append(os.pardir) 
import numpy as np
from dataset.mnist import load_mnist

print("学习--是从训练数据中获取到某种规律\n"
"此处指自动获取最优权重参数的过程")

"https://chat.deepseek.com/share/0vfopa9ovpuuv1ocrc"
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