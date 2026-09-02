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
a = input("是否绘制XOR线性不可分问题的可视化图？(y/n):")
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

