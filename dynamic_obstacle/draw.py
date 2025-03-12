import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取txt文件中的路径点
def read_points_from_file(filename):
    # 读取数据并转换为numpy数组
    points = np.loadtxt(filename)
    return points

# 读取两个文件的路径点
points_1 = read_points_from_file('distance.txt')
# points_2 = read_points_from_file('2.txt')
# points_3 = read_points_from_file('3.txt')

# 创建一个3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 提取x, y, z坐标
x1, y1, z1 = points_1[:, 0], points_1[:, 1], points_1[:, 2]
# x2, y2, z2 = points_2[:, 0], points_2[:, 1], points_2[:, 2]
# x3, y3, z3 = points_3[:, 0], points_3[:, 1], points_3[:, 2]

# 绘制路径点
ax.scatter(x1, y1, z1, color='r', label='Path 1')  # 用红色表示路径1
# ax.scatter(x2, y2, z2, color='b', label='Path 2')  # 用蓝色表示路径2
# ax.scatter(x3, y3, z3, color='g', label='Path 3')  # 用绿色表示路径3

# 设置标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 设置标题
ax.set_title('3D Paths')

# 显示图例
ax.legend()

# 显示图形
plt.show()
