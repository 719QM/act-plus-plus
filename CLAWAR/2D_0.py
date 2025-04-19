import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

title_fontsize = 26
label_fontsize = 24
legend_fontsize = 22
tick_fontsize = 20


# 读取轨迹数据
def load_trajectory(file_name):
    return np.loadtxt(file_name)

# 文件与颜色对应表
trajectories = {
    "experiment/teleoperation/0.txt": ("blue", "human"),
    "experiment/TAB/3.txt": ("green", "manual switch"),
    "experiment/k_1000/1.txt": ("orange", "fixed_weight"),
    "experiment/k_10/1.txt": ("purple", "shared_k10"),
    "experiment/k_50/0.txt": ("red", "shared_k50"),
    "experiment/k_80/1.txt": ("magenta", "shared_k80"),
}

# 固定窗口大小（英寸）
fig, ax = plt.subplots(figsize=(12, 12))

# 绘制轨迹
for file_name, (color, label) in trajectories.items():
    traj = load_trajectory(file_name)
    ax.plot(traj[:, 1], traj[:, 0], color=color, label=label, linewidth=4)

# 绘制障碍物投影（在 XY 平面上）
center = np.array([-0.4, 0.7])
length, width = 0.2, 0.2
# 横轴是 y，纵轴是 x，所以 rectangle 的 lower_left 要按照 y,x 来
lower_left = np.array([center[1] - width / 2, center[0] - length / 2])
obstacle_rect = Rectangle(lower_left, width, length, linewidth=1.5,
                          edgecolor='black', facecolor='red', alpha=0.8, label="Obstacle")
ax.add_patch(obstacle_rect)
plt.xlim(0.34, 0.92)      # 横轴（-Y）
plt.ylim(-0.525, -0.04)   # 纵轴（X）

# 设置坐标轴
ax.set_xlabel("Y", fontsize=label_fontsize)
ax.set_ylabel("X", fontsize=label_fontsize)
ax.set_aspect('equal')  # 保持1:1比例
# ax.set_title("Comparison of trajectories of different methods", fontsize=title_fontsize)
ax.legend(fontsize=legend_fontsize)
ax.tick_params(labelsize=tick_fontsize)
ax.invert_xaxis()  # 将 Y 坐标轴反向，使其为负方向


plt.grid(True)
plt.tight_layout()
plt.show()
