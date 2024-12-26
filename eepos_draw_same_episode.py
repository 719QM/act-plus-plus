# 用于对比在相同初始点和目标点的情况下，teleoperation和对应policy的末端点轨迹对比
import matplotlib.pyplot as plt
import numpy as np
import re

def parse_line(line):
    """
    解析每一行数据，返回一个包含 x, y, z 的数组。
    """
    # 去掉方括号
    line = re.sub(r'[\[\]]', '', line.strip())
    # 判断分隔符是逗号还是空格，并分割数据
    if ',' in line:
        return np.array([float(x) for x in line.split(',')])
    else:
        return np.array([float(x) for x in line.split()])  # 按空格分割

def read_trajectory(file_path):
    """
    从 txt 文件中读取轨迹数据。
    """
    trajectory = []
    with open(file_path, 'r') as file:
        for line in file:
            # 忽略空行或非数值内容
            if line.strip():
                trajectory.append(parse_line(line))
    return np.array(trajectory)


# 文件路径（替换为实际文件路径）
file1 = '/home/juyiii/ALOHA/act-plus-plus/EEpos/20_3/1.txt'
file2 = '/home/juyiii/ALOHA/act-plus-plus/EEpos/20_3/1_policy_end_effector_position.txt'
file3 = '/home/juyiii/ALOHA/act-plus-plus/EEpos/20_3/2.txt'
file4 = '/home/juyiii/ALOHA/act-plus-plus/EEpos/20_3/2_policy_end_effector_position.txt'
file5 = '/home/juyiii/ALOHA/act-plus-plus/EEpos/20_3/3.txt'
file6 = '/home/juyiii/ALOHA/act-plus-plus/EEpos/20_3/3_policy_end_effector_position.txt'
# file7 = '/home/juyiii/ALOHA/act-plus-plus/EEpos/18_3/18_Astar.txt'

# 读取轨迹
trajectory1 = read_trajectory(file1)
trajectory2 = read_trajectory(file2)
trajectory3 = read_trajectory(file3)
trajectory4 = read_trajectory(file4)
trajectory5 = read_trajectory(file5)
trajectory6 = read_trajectory(file6)
# trajectory7 = read_trajectory(file7)


# 检查是否读取正确
print("轨迹1 shape:", trajectory1.shape)
print("轨迹2 shape:", trajectory2.shape)
print("轨迹3 shape:", trajectory3.shape)
print("轨迹4 shape:", trajectory4.shape)
print("轨迹5 shape:", trajectory5.shape)
print("轨迹6 shape:", trajectory6.shape)
# print("轨迹7 shape:", trajectory7.shape)

# 绘制轨迹
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制4条轨迹
ax.plot(trajectory1[:, 0], trajectory1[:, 1], trajectory1[:, 2], label='1_teleoperation', color='r')
ax.plot(trajectory2[:, 0], trajectory2[:, 1], trajectory2[:, 2], label='1_policy', color='g')
ax.plot(trajectory3[:, 0], trajectory3[:, 1], trajectory3[:, 2], label='2_teleoperation', color='b')
ax.plot(trajectory4[:, 0], trajectory4[:, 1], trajectory4[:, 2], label='2_policy', color='y')
ax.plot(trajectory5[:, 0], trajectory5[:, 1], trajectory5[:, 2], label='3_teleoperation', color='pink')
ax.plot(trajectory6[:, 0], trajectory6[:, 1], trajectory6[:, 2], label='3_policy', color='purple')
# ax.plot(trajectory7[:, 0], trajectory7[:, 1], trajectory7[:, 2], label='A*', color='black')


# 设置图例和标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# 显示图形
plt.title("3D Trajectories")
plt.show()
