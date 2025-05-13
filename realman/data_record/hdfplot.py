import h5py
import numpy as np
import matplotlib.pyplot as plt

# 替换成你的 hdf5 路径
hdf5_path = "episode_v_20.hdf5"

# 读取数据
with h5py.File(hdf5_path, "r") as f:
    action_qpos = np.array(f["action"])
    obs_qpos = np.array(f["observations/qpos"])

# 时间步
T = action_qpos.shape[0]
time_steps = np.arange(T)

# 验证维度
assert action_qpos.shape == obs_qpos.shape, "action 和 observations/qpos 的形状不一致"
num_joints = action_qpos.shape[1]

# 创建子图
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 12))
axes = axes.flatten()

for i in range(num_joints):
    ax = axes[i]
    ax.plot(time_steps, action_qpos[:, i], label="action", color='blue')
    ax.plot(time_steps, obs_qpos[:, i], label="observation", color='orange')
    ax.set_title(f"Joint {i}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
