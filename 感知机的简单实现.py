"https://chat.deepseek.com/share/r462vk3tc2g6s85epq"
"这是向DeepSeek问到的感知机的起源与理解"
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
    tmp = np.sum(w_ * x_) + b_
    if tmp > 0:
        return 1
    else:
        return 0
def NAND(x1,x2):
    x_ = np.array([x1,x2])
    w_ = np.array([-0.5,-0.5]) # 仅权重和偏置与AND不同
    b_ = 0.7
    tmp = np.sum(w_ * x_) + b_
    if tmp > 0:
        return 1
    else:
        return 0
def OR(x1,x2):
    x_ = np.array([x1,x2])
    w_ = np.array([0.5,0.5]) # 仅权重和偏置与AND不同
    b_ = -0.2
    tmp = np.sum(w_ * x_) + b_
    if tmp > 0:
        return 1
    else:
        return 0
print('='*15,"异或门的实现：无法直接用单层感知机（调整参数）实现",'='*20)

print("感知机的局限性：只能表示由一条直线分割的空间（线性空间）")

# 画一个可视图（LLM辅助代码🐶）
import matplotlib.pyplot as plt
# 四个点
points = {
    (0, 0): "00",
    (1, 0): "10",
    (0, 1): "01",
    (1, 1): "11"
}

# 颜色：XOR 的正例是 10, 01；负例是 00, 11
colors = {
    (0, 0): "tab:orange",
    (1, 0): "tab:blue",
    (0, 1): "tab:blue",
    (1, 1): "tab:orange",
}
fig, ax = plt.subplots(figsize=(6, 6))
# 画出四个点
for (x, y), label in points.items():
    ax.scatter(x, y, s=120, c=colors[(x, y)], edgecolors="black", zorder=3)
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 8),
                fontsize=12, fontweight="bold")

# 画一条“尝试分割”的直线（仅用于说明无法分开）
x = np.linspace(-0.2, 1.5, 100)
y = -x + 0.5
ax.plot(x, y, "k--", linewidth=2, label="division line")

# 坐标轴设置
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.2, 1.2)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.grid(True, linestyle=":", alpha=0.5)
ax.set_title("XOR Problem: Not Linearly Separable")
ax.legend()
plt.show()

def XOR(x1, x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y0 = AND0(s1,s2)
    return y0
print("XOR(0,0):",XOR(0,0))
print("XOR(0,1):",XOR(0,1))
print("XOR(1,1):",XOR(1,1))
print("""异或门是二层感知机
总共由三层组成，拥有权重的只有2层（两个“之间”）
单层感知机之间表示线性空间，多层感知机可表示非线性空间
""")