import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 读取轨迹数据
def load_trajectory(file_name):
    return np.loadtxt(file_name)

title_fontsize = 26
label_fontsize = 24
legend_fontsize = 22
tick_fontsize = 20

# 轨迹文件列表及对应颜色
trajectories = {
    "experiment/shared_control_verify/policy_trajectory.txt": "green",
    "experiment/shared_control_verify/teleoperation_trajectory.txt": "blue",
    "experiment/shared_control_verify/ee_pos.txt": "gray",
    "experiment/shared_control_verify/weighted_trajectory.txt": "red"
}
# 图例映射：文件名关键词 -> 简洁图例名
legend_names = {
    "policy_trajectory": "policy",
    "teleoperation_trajectory": "human",
    "ee_pos": "actual",
    "weighted_trajectory": "shared_control"
}

# 创建 2D 图
fig, ax = plt.subplots(figsize=(12, 12))  # 或者你想要的任意尺寸


# 读取并绘制每条轨迹（Y为横轴，X为纵轴）
for file_path, color in trajectories.items():
    traj = load_trajectory(file_path)

    # 获取简洁图例名
    for key, label in legend_names.items():
        if key in file_path:
            legend_label = label
            break
    else:
        legend_label = file_path  # 如果没匹配上，就用原路径名

    ax.plot(traj[:, 1], traj[:, 0], color=color, label=legend_label, linewidth=4)
    # ax.scatter(traj[:, 1], traj[:, 0], color=color, s=8)

# 添加障碍物（在 XY 平面上画一个矩形，对应坐标也要转轴）
center = np.array([-0.4, 0.7])  # [x, y]
length, width = 0.2, 0.2
# 横轴是 y，纵轴是 x，所以 rectangle 的 lower_left 要按照 y,x 来
lower_left = np.array([center[1] - width / 2, center[0] - length / 2])
obstacle_rect = Rectangle(lower_left, width, length, linewidth=1.5,
                          edgecolor='black', facecolor='red', alpha=0.8, label="Obstacle")
ax.add_patch(obstacle_rect)

plt.xlim(0.34, 0.92)      # 横轴（-Y）
plt.ylim(-0.525, -0.04)   # 纵轴（X）
# 坐标设置
# ax.set_title("2D Trajectories (Y as X-axis, X as Y-axis)", fontsize=title_fontsize)
ax.set_xlabel("Y", fontsize=label_fontsize)
ax.set_ylabel("X", fontsize=label_fontsize)
ax.set_aspect('equal')
ax.tick_params(labelsize=tick_fontsize)
ax.invert_xaxis()  # 将 Y 坐标轴反向，使其为负方向

plt.grid(True)
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.show()
