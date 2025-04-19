import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 读取轨迹数据
def load_trajectory(file_name):
    return np.loadtxt(file_name)

# 轨迹文件列表及对应颜色
trajectories = {
    "experiment/shared_control_verify/policy_trajectory.txt": "red",
    "experiment/shared_control_verify/teleoperation_trajectory.txt": "blue",
    "experiment/shared_control_verify/ee_pos.txt": "green",
    "experiment/shared_control_verify/weighted_trajectory.txt": "purple"
}

# 创建 3D 图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 读取并绘制每条轨迹（点 + 线）
for file_name, color in trajectories.items():
    traj = load_trajectory(file_name)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, label=file_name, linewidth=4)  # 连线
    # ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], color=color, s=8)  # 小点更清晰

# 添加障碍物（长方体）
center = np.array([-0.45, 0.75, 0.25])
length, width, height = 0.2, 0.2, 0.3
x, y, z = center
dx, dy, dz = length / 2, width / 2, height / 2

vertices = np.array([
    [x - dx, y - dy, z - dz], [x + dx, y - dy, z - dz],
    [x + dx, y + dy, z - dz], [x - dx, y + dy, z - dz],
    [x - dx, y - dy, z + dz], [x + dx, y - dy, z + dz],
    [x + dx, y + dy, z + dz], [x - dx, y + dy, z + dz]
])

faces = [
    [vertices[0], vertices[1], vertices[2], vertices[3]],
    [vertices[4], vertices[5], vertices[6], vertices[7]],
    [vertices[0], vertices[1], vertices[5], vertices[4]],
    [vertices[2], vertices[3], vertices[7], vertices[6]],
    [vertices[1], vertices[2], vertices[6], vertices[5]],
    [vertices[0], vertices[3], vertices[7], vertices[4]],
]

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
