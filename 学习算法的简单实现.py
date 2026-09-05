# 学习步骤
"""
1. 从训练数据中随机[^1]选出一部分数据（mini-batch）
2. 为了减小mini-batch损失函数的值，求出各权重参数的梯度
3.（用梯度下降法）（沿梯度方向）更新参数
4.重复1,2,3
"""
"""
[^1]:随机选择的数据，则该方法为随机梯度下降法（SGD）
"""
import sys,os
sys.path.append(os.pardir)
from 工具函数文件 import *
import numpy as np
print('='*15,"先实现一个2层神经网络的类",'='*20)

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        # weight_init_std: 用于初始化权重
        self.params = {}   # 保存神经网络参数的字典型变量（实例变量）
        self.params['W1'] = weight_init_std *\
                            np.random.randn(input_size,hidden_size) #生成一个符合正态分布的随机矩阵
        self.params['b1'] = np.zeros(hidden_size) # 一维数组
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)
    def predict(self,x):
        W1,W2 = self.params['W1'],self.params['W2']
        b1,b2 = self.params['b1'],self.params['b2']

        a1 = np.dot(x,W1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2)+b2
        y = softmax(a2)
        return y

    # x:输入数据 t:监督数据（即训练标签）
    def loss(self,x,t):
        y = self.predict(x)
        return cross_entropy_error(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)
        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy

    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])
        return grads
# 上述是推理处理部分
# 先看看效果
net_0 = TwoLayerNet(input_size=784,hidden_size=100,output_size=10)
print(net_0.params['W1'].shape)
print(net_0.params['W2'].shape)
print(net_0.params['b1'].shape)
print(net_0.params['b2'].shape)
x_0 = np.random.rand(100,784) # 伪输入数据（100份）
y_0 = net_0.predict(x_0)
t_0 = np.random.rand(100,10) # 伪正确解标签（100份）
#  numerical_gradient函数计算梯度较慢，可使用下一章BP算法，gradient(self,x,t)
grads_0 = net_0.numerical_gradient(x_0,t_0)
print(grads_0['W1'].shape)
print(grads_0['W2'].shape)
print(grads_0['b1'].shape)
print(grads_0['b2'].shape)

print('='*15,"mini-batch的实现",'='*20)
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True,one_hot_label=True)
train_loss_list = []
# 基于测试数据的评价
train_acc_list = []
test_acc_list = []
'''
epoch: 是一个单位，表示学习时遍历一次所有训练数据，
比如对于不是完全随机的大小为100的mini-batch进行学习时，（每个mini-batch不重叠）
若有60000份训练数据，需重复随机梯度下降600次，600次即为一个epoch
【对于mini-batch，一般做法是事先将训练数据全部打乱，然后按指定批次大小，按序生成】
'''

# 超参数
iters_num = 50 # 书上是10000，太慢了
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
network = TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

# 平均每个epoch的重复次数
iter_per_epoch = max(train_size // batch_size,1)
print("iter_per_epoch:",iter_per_epoch)

for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.numerical_gradient(x_batch,t_batch)
    # print(grad.keys())
    for key in ('W1','b1','W1','b2'):
        network.params[key] -= learning_rate * grad[key]
    # 记录学习过程
    loss = network.loss(x_batch,t_batch)
    train_loss_list.append(loss)
    # 计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train,t_train)
        test_acc = network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc,test acc | "+str(train_acc)+", "+str(test_acc))

y1 = train_loss_list
x1 = np.arange(0,iters_num,1)
plt.plot(x1,y1)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()

# 由于上述代码中，acc太少，故先不画图