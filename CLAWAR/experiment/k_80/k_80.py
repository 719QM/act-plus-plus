import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 读取轨迹数据
def load_trajectory(file_name):
    return np.loadtxt(file_name)

# 轨迹文件列表及对应颜色
trajectories = {
    "0.txt": "blue",
    "1.txt": "red",
    "2.txt": "green",
    "3.txt": "orange",
    "4.txt": "purple",


}

# 创建 3D 图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 读取并绘制每条轨迹
for file_name, color in trajectories.items():
    traj = load_trajectory(file_name)
    ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], label=file_name, color=color, s=10)  # s 控制点的大小

# 画长方体（障碍物）
center = np.array([-0.45, 0.75, 0.25])
length, width, height = 0.2, 0.2, 0.3

# 计算长方体的8个顶点
x, y, z = center
dx, dy, dz = length / 2, width / 2, height / 2

vertices = np.array([
    [x - dx, y - dy, z - dz], [x + dx, y - dy, z - dz],
    [x + dx, y + dy, z - dz], [x - dx, y + dy, z - dz],
    [x - dx, y - dy, z + dz], [x + dx, y - dy, z + dz],
    [x + dx, y + dy, z + dz], [x - dx, y + dy, z + dz]
])

# 定义六个面（按顶点索引）
faces = [
    [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
    [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
    [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
    [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
    [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
    [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面
]

# 使用 Poly3DCollection 绘制障碍物
obstacle = Poly3DCollection(faces, alpha=0.5, edgecolor="k", facecolor="red")
ax.add_collection3d(obstacle)

# 设置标签和图例
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.title("3D Trajectories with Obstacle")

# 显示图像
plt.show()
