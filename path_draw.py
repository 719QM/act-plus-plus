import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取文件并解析路径数据
file_path = 'Astar_data/output_0.txt'

# 用于存储读取的路径点，起点、终点、障碍物
path_points = []
start_point = None
target_point = None
obstacle = None

# 打开文件并读取内容
with open(file_path, 'r') as file:
    for line in file:
        # 去除每行的空格和换行符
        line = line.strip()

        # 检查是否是特殊的起点、终点或障碍物
        if line.startswith("start:"):
            start_point = tuple(map(float, line.split(":")[1].split()))
        elif line.startswith("target:"):
            target_point = tuple(map(float, line.split(":")[1].split()))
        elif line.startswith("obstacle:"):
            obstacle = tuple(map(float, line.split(":")[1].split()))
        else:
            # 处理普通路径点
            point = tuple(map(float, line.split()))
            path_points.append(point)

# 提取路径点的x、y、z坐标
x_path, y_path, z_path = zip(*path_points)

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制路径点
ax.plot(x_path, y_path, z_path, label='Path', color='blue')

# 绘制起点、终点和障碍物
ax.scatter(*start_point, color='green', s=100, label='Start')
ax.scatter(*target_point, color='red', s=100, label='Target')
ax.scatter(*obstacle, color='orange', s=100, label='Obstacle')

# 设置标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图例
ax.legend()

# 展示图形
plt.show()