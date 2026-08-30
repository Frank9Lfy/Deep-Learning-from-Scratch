import numpy as np
import matplotlib.pyplot as plt

print('='*15,"绘制简单图形",'='*20)
x = np.arange(0,6,0.1) #以0.1为单位，生成0-6的数据
y1 = np.sin(x)
plt.plot(x,y1) # 绘制图像
plt.show()  
# 追加cos函数，添加标题，x轴标签名
y2 = np.cos(x)
plt.plot(x,y1,label="sin")
plt.plot(x,y2,linestyle ="--", label ="cos")
# ValueError: '-_' is not a valid value for ls.
# Did you mean one of: '-'(和直线没区别), '-.'（点划线）, '--'（虚线）?
plt.xlabel("x") # x轴标签名
plt.ylabel("y") 
plt.title('sin&cos')
plt.legend()  # 添加图例说明框
plt.show()
print('='*15,"显示图像",'='*20)

from matplotlib.image import imread
img = imread('Dataset-level annotation errors.png') # 读入图像
plt.imshow(img)
plt.show()
