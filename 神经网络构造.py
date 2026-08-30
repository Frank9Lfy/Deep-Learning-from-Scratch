import numpy as np
print('*'*15,"激活函数的实现及显示",'*'*20)
print('='*15,"跃阶函数的实现及显示",'='*20)
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
'''
本来括号里是np.int, 但是报错了：
AttributeError: module 'numpy' has no attribute 'int'.`np.int` was a deprecated alias for the builtin `int`. 
To avoid this error in existing code, use `int` by itself. Doing this will not modify any behavior and is safe. 
When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
'''
print(step_function1(np.array([-1.0, 1.0, 2.0])))
import matplotlib.pylab as plt
def step_function(x):
    return np.array(x > 0, dtype=np.int64)  # 返回布尔值数组，True为1，False为0
x0 = np.arange(-5.0, 5.0, 0.1) # 生成-5.0到5.0的数组，以0.1为单位(不包含5.0，离散的点连线)
print(x0)
y0 = step_function(x0)
print(y0)
plt.plot(x0, y0)
plt.ylim(-0.1, 1.1) # 指定设置y轴的范围
plt.show()

print('='*15,"sigmoid函数的实现及显示",'='*20)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
t = np.array([1.0, 2.0, 3.0])
print(sigmoid(t))
print(1.0+t)   # Numpy数组的广播机制(见NumPy基础知识笔记.py)
print(1.0/t)
x1 = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x1)  
plt.plot(x1, y1)
plt.ylim(-0.1, 1.1)
plt.show()
plt.plot(x0, y0,label='step',linestyle='--')
plt.plot(x1, y1,label='sigmoid')
plt.ylim(-0.1, 1.1)
plt.legend()  # 添加图例说明框
plt.show()
print("""
跃阶函数和sigmoid函数均为非线性函数，且sigmoid函数是跃阶函数的平滑版本。
激活函数必须是非线性函数，否则无论隐藏层有多少层，整个网络都只能表示线性函数。""")

print('='*15,"ReLU函数的实现及显示",'='*20)
def relu(x):
    return np.maximum(0, x)
x2 = x1
y2 = relu(x2)
plt.plot(x2,y2)
plt.ylim(-1, 5.5)
plt.show()

print('='*15,"神经网络的内积",'='*20)
print("矩阵相关代码用法 见NumPy基础知识笔记")
print("内积：np.dot(a,b)或a.dot(b)或a@b")
X = np.array([1,2])
print(X.shape)
W = np.array([[1,3,5],[2,4,6]])
print(W)
print(W.shape)
Y = np.dot(X,W)
print(Y)

print('='*15,"3层神经网络的实现",'='*20)