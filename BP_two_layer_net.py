import sys,os
sys.path.append(os.pardir)
from 工具函数文件 import *
from 工具Layers import *
import numpy as np
from collections import OrderedDict

# 参见学习算法的简单实现.py，
# #初始化权重,numerical_gradient的部分与原来相同

class TwoLayerNet:
    def __init__(self, input_size, hidden_size,
                 output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)  # 生成一个符合正态分布的随机矩阵
        self.params['b1'] = np.zeros(hidden_size)  # 一维数组
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
    #OrderedDict 是一个能记住键值对插入顺序的字典子类。它保证你遍历字典时，拿到的顺序和当初添加的顺序完全一致。
        self.layers['Affine1'] = \
            Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = \
            Affine(self.params['W2'],self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
# x:输入数据 t:监督数据（即训练标签）
    def loss(self,x,t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        if t.ndim != 1:t = np.argmax(t,axis=1)  # 没人教过我Python的if可以不换行啊🤣
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

    def gradient(self,x,t):
        # forward
        self.loss(x,t)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())  # 转换成列表类型
        # print(layers)
        layers.reverse()
        # print(layers)
        for layer in layers:   # 从最后一层往前传
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        return grads

# 梯度确认  数值梯度和BP的比较
# from dataset.mnist import load_mnist
#
# # 读入数据
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
#
# print(x_train.shape, t_train.shape)
# print(x_test.shape, t_test.shape)
# network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
#
# x_batch = x_train[:3]
# print(x_batch)
# print(x_batch.shape)
# t_batch = t_train[:3]
# print(t_batch.shape)
#
# grad_numerical = network.numerical_gradient(x_batch, t_batch)
# grad_backprop = network.gradient(x_batch, t_batch)
#
# #求各个权重的绝对误差的平均值
# for key in grad_numerical.keys():
#     diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
#     print(key + ":" + str(diff))