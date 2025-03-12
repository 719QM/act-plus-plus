import h5py
import numpy as np
import os


def update_lqpos_with_finger(hdf5_file):
    with h5py.File(hdf5_file, 'r+') as f:
        if 'finger' not in f or 'lqpos' not in f:
            raise KeyError("HDF5文件中缺少 'finger' 或 'lqpos' 数据集")

        finger_data = f['finger'][:]
        lqpos_data = f['lqpos'][:]

        if finger_data.shape[0] != lqpos_data.shape[0]:
            raise ValueError("'finger' 和 'lqpos' 数据集的时间步数不匹配")

        # 预定义开关状态的数值
        open_state = np.array([999.0, 999.0, 999.0, 999.0, 999.0, 1.0])
        close_state_1 = np.array([550.0, 550.0, 550.0, 550.0, 550.0, 1.0])
        close_state_2 = np.array([600.0, 600.0, 600.0, 600.0, 999.0, 1.0])

        # 计算手指状态
        mask_open = np.all(finger_data == open_state, axis=1)
        mask_close_1 = np.all(finger_data == close_state_1, axis=1)
        mask_close_2 = np.all(finger_data == close_state_2, axis=1)

        # 更新 lqpos 数据的最后一列
        lqpos_data[:, -1] = np.where(mask_open, 1, 0)

        # 写回数据集
        f['lqpos'][:] = lqpos_data

        # 删除原始 finger 数据集
        del f['finger']
        print("HDF5 文件已更新：'lqpos' 数据集的末列已根据 'finger' 数据修改，并删除 'finger' 数据集")


def create_action_and_observations(hdf5_file):
    with h5py.File(hdf5_file, 'r+') as f:
        if 'lqpos' not in f or 'rqpos' not in f:
            raise KeyError("HDF5文件中缺少 'lqpos' 或 'rqpos' 数据集")

        lqpos_data = f['lqpos'][:]
        rqpos_data = f['rqpos'][:]

        if lqpos_data.shape[0] != rqpos_data.shape[0]:
            raise ValueError("'lqpos' 和 'rqpos' 数据集的时间步数不匹配")

        # 创建 action 数据集
        action_data = np.hstack((lqpos_data, rqpos_data))
        if 'action' in f:
            del f['action']
        f.create_dataset('action', data=action_data)

        # 创建 observations 组
        if 'observations' in f:
            del f['observations']
        obs_group = f.create_group('observations')

        # 创建 qpos 数据集
        obs_group.create_dataset('qpos', data=action_data)

        # 复制 image 数据
        image_folder = obs_group.create_group('images')
        if 'image_1' in f and 'image_2' in f:
            image_folder.create_dataset('image_1', data=f['image_1'][:])
            image_folder.create_dataset('image_2', data=f['image_2'][:])

            # 删除原始 image 数据集
            del f['image_1']
            del f['image_2']

        # 删除原始 lqpos 和 rqpos 数据集
        del f['lqpos']
        del f['rqpos']

        print("HDF5 文件已更新：创建 'action' 数据集和 'observations' 组，并删除原始数据集")


if __name__ == "__main__":
    for data_num in range(30):
        old_filename = f"/home/juyiii/data/aloha/20250311/data{data_num + 1}.hdf5"
        new_filename = f"/home/juyiii/data/aloha/20250311/episode_{data_num}.hdf5"

        if os.path.exists(old_filename):
            os.rename(old_filename, new_filename)
            try:
                print(f"正在处理: {new_filename}")
                update_lqpos_with_finger(new_filename)
                create_action_and_observations(new_filename)
            except Exception as e:
                print(f"处理 {new_filename} 失败: {e}")
        else:
            print(f"文件未找到: {old_filename}")
