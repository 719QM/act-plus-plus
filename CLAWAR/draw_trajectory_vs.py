import numpy as np
import matplotlib.pyplot as plt

# 读取轨迹数据
def load_trajectory(file_name):
    return np.loadtxt(file_name)

# 轨迹文件列表及对应颜色
trajectories = {
    "policy_trajectory.txt": "red",
    "teleoperation_trajectory.txt": "blue",
    "distancetest.txt": "green",
    "weighted_trajectory.txt": "purple"
}

# 创建 3D 图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 读取并绘制每条轨迹
for file_name, color in trajectories.items():
    traj = load_trajectory(file_name)
    ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], label=file_name, color=color, s=10)  # s 控制点的大小

# 设置标签和图例
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.title("3D Trajectories")

# 显示图像
plt.show()
