import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print('='*15,"3D图像绘制(二元函数)",'='*20)
# 生成网格
x0 = np.linspace(-5, 5, 100)
x1 = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x0, x1)

# 计算 z = x0^2 + x1^2
Z = X**2 + Y**2

# 绘图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('z')
ax.set_title('z = x0^2 + x1^2')

plt.show()

print("\n等高线图\n")
x0 = np.linspace(-5, 5, 200)
x1 = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x0, x1)
Z = X**2 + Y**2

plt.figure(figsize=(7, 6))
plt.contourf(X, Y, Z, levels=20, cmap='jet')
plt.colorbar()
plt.xlabel('x0')
plt.ylabel('x1')
plt.title('z = x0^2 + x1^2 等高线图')
plt.show()
print("偏导数\n")
x0 = np.linspace(-5, 5, 200)
x1 = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x0, x1)

d1 = 2 * X
d2 = 2 * Y

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.contourf(X, Y, d1, levels=20, cmap='coolwarm')
plt.colorbar()
plt.title('∂f/∂x0 = 2x0')

plt.subplot(1, 2, 2)
plt.contourf(X, Y, d2, levels=20, cmap='coolwarm')
plt.colorbar()
plt.title('∂f/∂x1 = 2x1')

plt.show()


print("梯度下降显示\n")

x = np.linspace(-2, 2, 30)
y = np.linspace(-2, 2, 30)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

U = -2 * X
V = -2 * Y

fig, ax = plt.subplots(figsize=(7, 6))

contour = ax.contourf(X, Y, Z, levels=10, cmap='viridis', alpha=0.8)
fig.colorbar(contour, ax=ax, label='f(x, y)')

ax.quiver(X, Y, U, V,
          angles='xy',
          scale_units='xy',
           scale=10,
          color='white')

ax.set_title('f(x, y)=x^2+y^2 的等高线图 + 负梯度场')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.show()



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
a = input("是否绘制XOR线性不可分问题的可视化图？(若是，请敲字母y):")
if a == "y":
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

print('='*15,"计算图",'='*20)


import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle

fig, ax = plt.subplots(figsize=(13.5, 6.5))
ax.set_xlim(-0.2, 14); ax.set_ylim(0.6, 7.4); ax.axis('off')

C_NODE, C_LEAF, C_INTER, C_OUT = '#4C9BE8', '#8BC34A', '#FFB74D', '#EF5350'
C_FWD, C_BWD = '#1B5E20', '#B71C1C'

def node(x, y, r, color, text, fs=11):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor='black', lw=1.5, zorder=3))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            color='white', fontweight='bold', zorder=4)

def seg(p1, p2, color, style='-|>', ls='-'):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                                 color=color, lw=2.4, linestyle=ls, zorder=2))

def lab(x, y, text, color, fs=10.5, boxed=False):
    kw = dict(fontsize=fs, color=color, fontweight='bold',
              ha='center', va='center', zorder=6)
    if boxed:
        kw['bbox'] = dict(boxstyle='round,pad=0.22', fc='white', ec=color, lw=1)
    ax.text(x, y, text, **kw)

Y, YB = 3.3, 6.2   # 正向主链 / 反向回流线

# 节点（只画一次）
node(1.6, Y+1.3, 0.62, C_LEAF,  'apple\n100')
node(1.6, Y-1.3, 0.68, C_LEAF,  'apple_num\n2')
node(4.0, Y, 0.55, C_NODE,  '×')
node(6.6, Y, 0.80, C_INTER, 'apple_price\n220')
node(9.0, Y, 0.55, C_NODE,  '×')
node(9.0, 1.1, 0.62, C_LEAF,  'tax\n1.1')
node(11.6, Y, 0.72, C_OUT,  'price\n242')

# 正向：绿色连续箭头
seg((2.2, Y+1.1), (3.5, Y+0.3), C_FWD)
seg((2.25, Y-1.05), (3.5, Y-0.3), C_FWD)
seg((4.55, Y), (5.75, Y), C_FWD)
seg((7.42, Y), (8.42, Y), C_FWD)
seg((9.0, 1.75), (9.0, 2.72), C_FWD)
seg((9.55, Y), (10.85, Y), C_FWD)
lab(2.6, Y+1.15, '100', C_FWD); lab(2.5, Y-0.8, '2', C_FWD)
lab(5.15, Y+0.45, '220', C_FWD); lab(9.45, 2.15, '1.1', C_FWD)
lab(10.2, Y+0.45, '242', C_FWD)

# 反向：红色连续回流线
seg((11.6, Y+0.75), (11.6, YB), C_BWD, style='-[')      # price 汇入
lab(12.55, Y+1.35, 'dprice = 1', C_BWD, fs=9.5, boxed=True)
seg((11.6, YB), (9.55, YB), C_BWD)                      # 回流段 1
seg((8.45, YB), (4.55, YB), C_BWD)                      # 回流段 2
lab(10.55, YB+0.35, '1', C_BWD); lab(6.6, YB+0.35, '1.1', C_BWD)
seg((9.0, YB), (9.0, YB-0.5), C_BWD, style='-[')        # 分流 → tax
lab(9.0, YB-0.9, 'dtax = 220', C_BWD, fs=9.5, boxed=True)
seg((4.0, YB), (1.6, YB), C_BWD)                        # 分流 → apple
seg((1.6, YB), (1.6, Y+2.0), C_BWD, style='-[')
lab(2.8, YB+0.35, '2.2', C_BWD)
lab(0.55, Y+2.35, 'dapple = 2.2', C_BWD, fs=9.5, boxed=True)
seg((3.3, YB), (3.3, Y-1.15), C_BWD)                    # 分流 → apple_num
seg((3.3, Y-1.3), (2.32, Y-1.3), C_BWD, style='-[')
lab(3.62, Y-0.75, '110', C_BWD)
lab(4.15, Y-1.85, 'dapple_num = 110', C_BWD, fs=9.5, boxed=True)
#, ha='left' ？多余参数
ax.legend(handles=[mpatches.Patch(color=C_FWD, label='Forward 正向传播'),
                   mpatches.Patch(color=C_BWD, label='Backward 反向传播')],
          loc='lower left', fontsize=9.5)
plt.tight_layout(); plt.show()

