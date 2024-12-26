import h5py
import numpy as np
import random
import os


def adjust_frame_count(txt_data, target_length):
    """
    调整 txt 数据的帧数，使其与 target_length 一致。
    """
    current_length = len(txt_data)

    if current_length > target_length:
        # 随机抽帧
        indices = sorted(random.sample(range(current_length), target_length))
        adjusted_data = txt_data[indices]
    elif current_length < target_length:
        # 随机加帧
        deficit = target_length - current_length
        adjusted_data = txt_data.tolist()

        for _ in range(deficit):
            insert_index = random.randint(0, len(adjusted_data) - 1)
            adjusted_data.insert(insert_index, adjusted_data[insert_index])

        adjusted_data = np.array(adjusted_data)
    else:
        adjusted_data = txt_data  # 长度一致，不做调整

    return adjusted_data


def copy_hdf5_structure(src, dest, adjusted_qpos_data=None):
    """
    递归复制 HDF5 文件的结构和数据，并用 adjusted_qpos_data 替换 `observations/qpos` 数据。
    """
    for key, item in src.items():
        if isinstance(item, h5py.Group):
            # 如果是组（Group），递归创建并复制
            group = dest.create_group(key)
            copy_hdf5_structure(item, group, adjusted_qpos_data)
        elif isinstance(item, h5py.Dataset):
            # 如果是数据集（Dataset），根据需要替换或直接复制
            if key == 'qpos' and adjusted_qpos_data is not None and '/observations/qpos' in item.name:
                print(f"Replacing dataset '{item.name}' with new data.")
                dest.create_dataset(key, data=adjusted_qpos_data, dtype=item.dtype)
            else:
                print(f"Copying dataset '{item.name}'.")
                dest.create_dataset(key, data=item[:], dtype=item.dtype)


def process_hdf5_and_txt(hdf5_path, txt_path, output_path):
    # 确保目标文件不存在，避免锁定问题
    if os.path.exists(output_path):
        os.remove(output_path)

    # 打开源文件，提取目标数据
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        qpos_data = hdf5_file['observations/qpos'][:]  # 获取 observations/qpos 数据
        target_length = len(qpos_data)

    # 读取 TXT 文件
    with open(txt_path, 'r') as txt_file:
        txt_data = np.array([list(map(float, line.strip().split(','))) for line in txt_file.readlines()])

    # 调整 TXT 数据帧数
    adjusted_qpos_data = adjust_frame_count(txt_data, target_length)

    # 创建新的 HDF5 文件并复制数据
    with h5py.File(output_path, 'w', libver='latest') as output_file:
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            copy_hdf5_structure(hdf5_file, output_file, adjusted_qpos_data)

    print(f"处理完成，新的 HDF5 文件已保存到：{output_path}")


# for j in range(0, 20):
#     for i in range(0, 5):
#         if j == 6 or j == 17:
#             #  跳出循环
#             break
#         elif 6 < j < 17:
#             hdf5_path = f'/home/juyiii/data/aloha/sim_RM_Astar_teleoperation/episode_{j+80}.hdf5'
#             txt_path = f'/home/juyiii/ALOHA/act-plus-plus/teleoperation_data/source_txt/teleoperation_qpos_{5 * (j-1) + i}.txt'
#             output_path = f'/home/juyiii/data/aloha/sim_RM_Astar_teleoperation_adjusted/episode_{5 * (j-1) + i}.hdf5'
#             process_hdf5_and_txt(hdf5_path, txt_path, output_path)
#
#         elif j > 17:
#             hdf5_path = f'/home/juyiii/data/aloha/sim_RM_Astar_teleoperation/episode_{j+80}.hdf5'
#             txt_path = f'/home/juyiii/ALOHA/act-plus-plus/teleoperation_data/source_txt/teleoperation_qpos_{5 * (j-2) + i}.txt'
#             output_path = f'/home/juyiii/data/aloha/sim_RM_Astar_teleoperation_adjusted/episode_{5 * (j-2) + i}.hdf5'
#             process_hdf5_and_txt(hdf5_path, txt_path, output_path)
#
#         else:
#             hdf5_path = f'/home/juyiii/data/aloha/sim_RM_Astar_teleoperation/episode_{j+80}.hdf5'
#             txt_path = f'/home/juyiii/ALOHA/act-plus-plus/teleoperation_data/source_txt/teleoperation_qpos_{5*j+i}.txt'
#             output_path = f'/home/juyiii/data/aloha/sim_RM_Astar_teleoperation_adjusted/episode_{5*j+i}.hdf5'
#             process_hdf5_and_txt(hdf5_path, txt_path, output_path)

hdf5_path = f'/home/juyiii/data/aloha/sim_RM_Astar_teleoperation/episode_98.hdf5'
txt_path = f'/home/juyiii/ALOHA/act-plus-plus/EEpos/20_3/teleoperation_qpos_3.txt'
output_path = f'/home/juyiii/ALOHA/act-plus-plus/EEpos/20_3/teleoperation_qpos_3.hdf5'
process_hdf5_and_txt(hdf5_path, txt_path, output_path)