print('='*15,"先用感知机做一个与门",'='*20)
def AND0(x1,x2):
    return x1&x2
def AND1(x1,x2):
    w1,w2,theta = 0.5,0.5,0.7 #(元组用法，解包)，参数有无穷多组
    tmp = w1 * x1 + w2 * x2
    if tmp > theta:
        return 1
    else:
        return 0
    # 或者： tmp = int((w1*x1+w2*x2)>theta)
    # return tmp
print("1,1 y =",AND1(1,1))
print("1,0 y =",AND1(1,0))
print("0,0 y =",AND1(0,0))
print("改：将theta移项，变为-b(偏置) ","(w1*x1 + w2*x2 + b) ? 0")
print('='*15,"使用NumPy实现",'='*20)
import numpy as np
x = np.array([0,1])
w = np.array([0.5,0.5])  # 权重：控制参数的“话语权”
b = -0.7  # 偏置：调整神经元被激活的容易程度
# 有时将b,wi(i=1,2,...)这些参数统称为权重
print(w*x)
print(np.sum(w*x))
print(np.sum(w*x)+b) #大约为-0.2(由浮点小数运算造成的误差，计算机使用二进制浮点数来表示小数)
def AND2(x1,x2):
    x_ = np.array([x1,x2])
    w_ = np.array([0.5,0.5])
    b_ = -0.7
    tmp = np.sum(w * x) + b
    if tmp > 0:
        return 1
    else:
        return 0
def NAND(x1,x2):
    x_ = np.array([x1,x2])
    w_ = np.array([-0.5,-0.5]) # 仅权重和偏置与AND不同
    b_ = 0.7
    tmp = np.sum(w * x) + b
    if tmp > 0:
        return 1
    else:
        return 0
def OR(x1,x2):
    x_ = np.array([x1,x2])
    w_ = np.array([0.5,0.5]) # 仅权重和偏置与AND不同
    b_ = -0.2
    tmp = np.sum(w * x) + b
    if tmp > 0:
        return 1
    else:
        return 0
print('='*15,"异或门的实现：无法直接用感知机（调整参数）实现",'='*20)
