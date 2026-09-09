# BP原理见书P121-134
import numpy as np

print('*'*15,"简单层的实现",'*'*20,'\n')
print('='*15,"乘法层",'='*20)
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x*y
        return out

    def backward(self,dout):  # dout:前一层的导数值
        dx = dout * self.y
        dy = dout * self.x

        return dx,dy

# 使用一下乘法层
apple = 100
apple_num = 2
tax = 1.1
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()
# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print("price:",price)
# backward
dprice = 1
dapple_price,dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print("dapple,dapple_num,dtax:",dapple,dapple_num,dtax)

'''
apple(100) ──┐
             ├──[ × ]──→ apple_price(220) ──┐
apple_num(2)─┘                              ├──[ × ]──→ price(242)
                                 tax(1.1)───┘
'''
"相关计算图见复杂图像.py"

print('='*15,"加法层",'='*20)
class AddLayer:
    def __init__(self):
        pass
    def forward(self,x,y):
        return x + y
    def backward(self,dout):
        dx = dout * 1
        dy = dout * 1
        return dx,dy


# 实现购买2个苹果和3个橘子
orange = 150
orange_num = 3
mul_orange_layer = MulLayer()
add_orange_layer = AddLayer()
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_orange_layer.forward(apple_price, orange_price)
print("all_price:",all_price)
price2 = mul_tax_layer.forward(all_price, tax)
print("price2:",price2)

dprice2 = 1
dall_price,dtax2= mul_tax_layer.backward(dprice2)
dapple_price2, dorange_price = add_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple2, dapple_num2 = mul_apple_layer.backward(dapple_price2)
print(dapple2,dapple_num2,dorange,dorange_num,dtax2)

print('*'*15,"激活函数层的实现",'*'*20,'\n')
print('='*15,"ReLU层",'='*20)

# 假定参数为numpy数组
class Relu:
    def __init__(self):
        self.mask = None
    def forward(self,x):   # 假定参数x为numpy数组（可以是向量，矩阵，甚至3维，4维……的张量）
        self.mask = x <=0   # mask 变成了“bool数组”
        out = x.copy
        out[self.mask] = 0
        return out
    def backward(self,dout):
        dout[self.mask] = 0 # mask在正向传播的时候已经修改
        dx = dout * 1
        return dx

print('='*15,"Sigmoid层",'='*20)
# 原理见书p142-143(求导化简小技巧)
class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self,x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    def backward(self,dout):
        dx = dout*(1.0 - self.out)*self.out
        return dx

print('='*15,"Affine层的实现",'='*20)
# Affine :全链接层
print("简单理解Affine层： 见手写数字识别推理处理部分（前向传播）的predict函数"
      "\n这里增加了backward")
"数学推导见  https://chat.deepseek.com/share/b0nxigacakzuh1y7tb"
"实际只要记住维度对应即可"

# 理解偏置的计算
X_dot_W = np.array([[0,0,0],[10,10,10]])
B = np.array([1,2,3])
print(X_dot_W)
print("X_dot_W+B:\n",X_dot_W+B)  # 广播
dY = np.array([[1,2,3],[4,5,6]])
print("dY:\n",dY)
dB = np.sum(dY,axis=0)  # 注意求和结果
print("dB:",dB)

class Affine:  # 假设x<=2维
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W) + self.b
        return out
    def backward(self,dout):
        dx = np.dout(dout,self.W.T)
        dW = np.dout(self.x.T,dout)
        db = np.sum(dout,axis=0)
        return dx

print('='*15,"Softmax-with-Loss层的实现",'='*20)
"""
为了反向传播得到漂亮的结果(y1-t1,y2-t2,y3-t3,  ),
为softmax特意设计了交叉熵误差函数？！
回归问题中输出层使用恒等函数，损失函数使用平方和误差，也是同样的效果
"""
from 工具函数文件 import *
class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.loss = None
        self.t = None  #(one-hot vector)
    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    def backward(self,dout=1):

        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx