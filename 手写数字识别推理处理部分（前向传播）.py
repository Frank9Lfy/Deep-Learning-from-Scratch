import sys, os
sys.path.append(os.pardir) # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import pickle
from 工具文件 import sigmoid, softmax

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 第一次调用会花点时间
(x_train0, t_train0), (x_test0, t_test0) = \
load_mnist(flatten=True, normalize=False) # 将图像展开为一维数组，不进行归一化处理（值仍为0-255）
# 输出各个数据集的形状
print("x_train0.shape:", x_train0.shape)  # (60000, 784)
print("t_train0.shape:", t_train0.shape)  # (60000,)
print("x_test0.shape:", x_test0.shape)   # (10000, 784)
print("t_test0.shape:", t_test0.shape)   # (10000,)
img = x_train0[0]
label = t_train0[0]
print("label:", label)  # 输出标签值
print("img.shape:", img.shape)  # (784,)
img = img.reshape(28, 28)  # 将一维数组重新变为28*28的二维数组(因为需要显示图像)
print("img.shape:", img.shape)  # (28, 28)
img_show(img) 

print('='*15,"神经网络的推理处理",'='*20)
def get_data():
    (x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=True, one_hot_label=False) 
    return (x_train, t_train), (x_test, t_test)

"""
normalize=True：将图像的像素值（数据）归一化为0.0~1.0之间（限定在某种范围）的浮点数（：正规化，是一种预处理）
预处理： 对神经网络的输入数据进行某种既定的转换
one_hot_label=False：标签数据为整数形式（0~9），而不是one-hot编码形式
"""

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y
# 原代码疑似有误
# x, t = get_data()
# print(x)
# network = init_network()
# print(len(x))
# print(predict(network, x[0]))
# print(t[0])
# accuracy_cnt = 0
# for i in range(len(x)):
#     y = predict(network, x[i])
#     p = np.argmax(y)  # 获取预测结果中最大值的索引
#     print(p)
#     if p == t[i]:
#         accuracy_cnt += 1   
# print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
(x_train1, t_train1), (x_test1, t_test1) = get_data()
print("x_test1:", x_test1)
print("t_test1:", t_test1)
print("x_test1.shape:", x_test1.shape)
print("t_test1.shape:", t_test1.shape)
network1 = init_network()
accuracy_cnt = 0
for i in range(len(x_test1)):
    y = predict(network1, x_test1[i])
    # print("y:", y)
    p = np.argmax(y)  # 获取预测结果中最大值的索引
    if p == t_test1[i]:
        accuracy_cnt += 1   

print("Accuracy:" + str(float(accuracy_cnt) / len(x_test1)))

print('='*15,"批处理:更高效的处理方式",'='*20)
print("先确认数据和参数的形状")
x, _ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']
"""
书中代码默认x是二维数组，x.shape = (10000, 784)，但实际上x是元组
x[0]是二维数组，x[1]是一维数组（即上文的t_train）
"""
# print("x[0]:", x[0])
print("x[0].shape:", x[0].shape,"\n书中为(10000, 784),认为x[0]是x_test，但x[0]实则为x_train")
# print("x[0][0]:", x[0][0])
print("x[0][0].shape:", x[0][0].shape)
print("W1.shape:", W1.shape)
print("W2.shape:", W2.shape)
print("W3.shape:", W3.shape,"\n最终输出层的神经元个数为10（元素个数为10的一维数组）,对应10个数字类别")
print("批处理：打包输入多个样本数据")
print("只需将输入数据x改为二维数组（每一行是一个样本数据）即可")

(x_train2, t_train2), (x_test2, t_test2) = get_data()
batch_size = 100  # 批处理的样本数
accuracy_cnt = 0

for i in range(0, len(x_train2), batch_size):
    x_batch = x_train2[i:i+batch_size]
    # print("x_batch.shape:", x_batch.shape)
    y_batch = predict(network, x_batch)
    # print("y_batch.shape:", y_batch.shape)
    p = np.argmax(y_batch, axis=1)  # 获取预测结果中最大值的索引 行,沿第一维的方向
    # print("p.shape:", p.shape)
    # print("t_train2[i:i+batch_size].shape:", t_train2[i:i+batch_size].shape)
    # print("p:", p)
    accuracy_cnt += np.sum(p == t_train2[i:i+batch_size])  #统计预测正确的样本数
print("Accuracy:" + str(float(accuracy_cnt) / len(x_train2)))
print("说明：必须使用一致的训练集或测试集（因为他们的大小不同），否则数组可能为空，报错。")