import h5py
import numpy as np

# 路径替换为你自己的
file_path = '/home/juyiii/data/aloha/data_pc_test/episode_0.hdf5'
num_points = 1024
num_frames = 603

with h5py.File(file_path, 'a') as f:
    # 创建 group 路径
    if 'observations' not in f:
        obs_group = f.create_group('observations')
    else:
        obs_group = f['observations']

    if 'point_cloud' not in obs_group:
        pc_group = obs_group.create_group('point_cloud')
    else:
        pc_group = obs_group['point_cloud']

    # 如果已有同名数据集就删除
    for name in ['xyz', 'rgb']:
        if name in pc_group:
            del pc_group[name]

    # 随机生成点云 xyz 数据：形状 [3, num_points, num_frames]
    xyz_data = np.random.uniform(low=-1.0, high=1.0, size=(3, num_points, num_frames)).astype(np.float32)

    # 随机生成点云 rgb 数据：形状 [3, num_points, num_frames]，数值 0-255
    rgb_data = np.random.randint(low=0, high=256, size=(3, num_points, num_frames)).astype(np.uint8)

    # 保存到 HDF5 文件中
    pc_group.create_dataset('xyz', data=xyz_data)
    pc_group.create_dataset('rgb', data=rgb_data)

    print(f"✅ 已成功添加随机点云数据，形状：{xyz_data.shape}")

    # 创建 /observations/object_pose group
    if 'object_pose' not in obs_group:
        pose_group = obs_group.create_group('object_pose')
    else:
        pose_group = obs_group['object_pose']

    # 如果已有同名数据集就删除
    for name in ['R', 'T']:
        if name in pose_group:
            del pose_group[name]

    # 随机生成平移向量 T，范围可调，形状 (603, 3)
    T_data = np.random.uniform(low=-0.5, high=0.5, size=(num_frames, 3)).astype(np.float32)

    # 随机生成旋转矩阵（简化处理：使用正交矩阵近似）
    def random_rotation_matrix_9():
        # 使用QR分解得到随机正交矩阵
        A = np.random.randn(3, 3)
        Q, _ = np.linalg.qr(A)
        return Q.reshape(9)

    R_data = np.stack([random_rotation_matrix_9() for _ in range(num_frames)], axis=0).astype(np.float32)  # (603, 9)

    # 保存
    pose_group.create_dataset('R', data=R_data)
    pose_group.create_dataset('T', data=T_data)

    print(f"✅ 已成功添加随机目标位姿数据，R 形状：{R_data.shape}，T 形状：{T_data.shape}")

