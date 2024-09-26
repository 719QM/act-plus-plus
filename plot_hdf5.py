import h5py
import matplotlib.pyplot as plt
import numpy as np

def plot_qpos(hdf5_file_path):
    # 打开HDF5文件
    with h5py.File(hdf5_file_path, 'r') as file:
        # 获取时间步数
        num_timesteps = len(file['observations']['qpos'])

        # 初始化存储qpos数据的列表
        qpos_data = []

        # 遍历每个时间步
        for timestep in range(num_timesteps):
            # 获取当前时间步的qpos数据
            qpos = file['observations']['qpos'][timestep][:6]
            qpos_data.append(qpos)

        # 将qpos数据转换为二维数组，每一行为一个时间步的qpos
        qpos_data = np.array(qpos_data)

        # 绘制qpos曲线
        plt.figure(figsize=(10, 6))
        for i in range(qpos_data.shape[1]):
            plt.plot(qpos_data[:, i], label=f'qpos_{i+1}')

        plt.title('qpos Curves')
        plt.xlabel('Timestep')
        plt.ylabel('qpos')
        plt.legend()
        plt.grid(True)
        plt.show()

# 调用函数，传入你的HDF5文件路径
plot_qpos('/home/juyiii/data/aloha/sim_RM_Astar/episode_0.hdf5')