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
神经网络的激活函数必须是非线性函数，否则无论隐藏层有多少层，整个网络都只能表示线性函数。""")

print('='*15,"ReLU函数的实现及显示",'='*20)
def relu(x):
    return np.maximum(0, x)
x2 = x1
y2 = relu(x2)
plt.plot(x2,y2)
plt.ylim(-1, 5.5)
plt.show()
print("神经网络的激活函数使用ReLU函数的原因：\n1.计算简单，\n2.在正区间内梯度恒为1，\n3.在负区间内梯度恒为0，\n4.收敛速度快，\n5.非线性函数，\n6.不受输入数据的影响（sigmoid函数受输入数据的影响）")

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
print("A = XW + B")
print("输入层到第一（隐藏）层")
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X,W1) + B1
Z1 = sigmoid(A1)  # 使用上文定义的sigmoid函数
print("A1:",A1)
print("Z1:",Z1)
print("第一层到第二层（隐藏层）")
W2 = np.array([[0.1, 0.4],[0.2, 0.5],[0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = np.dot(Z1,W2) + B2
Z2 = sigmoid(A2)
print("A2:",A2)
print("Z2:",Z2)
print("第二层到输出层")
def identity_function(x):
    return x
W3 = np.array([[0.1, 0.3],[0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2,W3) + B3
Y = identity_function(A3)  # 使用恒等函数作为输出层的激活函数
print("A3:",A3)
print("Y:",Y)
print("""一般来说，输出层的激活函数是恒等函数（回归问题）
或softmax函数（分类问题） 见下文输出层的设计""")
print("整理三层神经网络代码成函数形式")
# 为了防止函数中使用的变量被外部同名变量覆盖，必须小写
def init_network():
    network = {}
    network['w1'] = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['w2'] = np.array([[0.1, 0.4],[0.2, 0.5],[0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['w3'] = np.array([[0.1, 0.3],[0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network
def forward(network, x):  # 从输入到输出方向的传递处理，故名为前向
    w1, w2, w3 = network['w1'], network['w2'], network['w3']  # 多重赋值（使代码简洁） 元组解包
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = identity_function(a3)
    return y
network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print("y:",y)

print('='*15,"输出层的设计",'='*20)
# https://chat.deepseek.com/share/luf0qnn2nr0u334sj0
print("分类问题--softmax函数的实现及显示")
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print("exp_a:",exp_a)
sum_exp_a = np.sum(exp_a)
print("sum_exp_a:",sum_exp_a)
print("exp_a/sum_exp_a:",exp_a/sum_exp_a)
a__1 = np.array([1010, 1000, 990])
print("exp(a__1):",np.exp(a__1))
print("exp(a__1)/sum(exp(a__1)):",np.exp(a__1)/np.sum(np.exp(a__1)))
def softmax(a):
    c = np.max(a)  # 为了防止溢出，减去最大值
    exp_a1 = np.exp(a - c)
    sum_exp_a = np.sum(exp_a1)
    y_ = exp_a1 / sum_exp_a
    return y_
print("softmax(a): ",softmax(np.array([0.3, 2.9, 4.0]))
      ,"\n总和为1，故可解释为概率分布")
print("套用函数前后（z和y）各个元素大小的相对关系不变（exp为单调递增函数）")
print("拓展注意点见书p68-69（3.5.4前后）")