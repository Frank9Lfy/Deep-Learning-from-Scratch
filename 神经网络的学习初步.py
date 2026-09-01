import sys, os
sys.path.append(os.pardir) 
import numpy as np
from dataset.mnist import load_mnist

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