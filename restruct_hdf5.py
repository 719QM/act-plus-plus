import h5py

max_episode = 38
for i in range(max_episode):
    # 原始 HDF5 文件路径
    input_file = f"/home/juyiii/data/aloha/rmreal_pick/data{i}.hdf5"
    # 新 HDF5 文件路径
    output_file = f"/home/juyiii/data/aloha/rmreal_pick/episode_{i}.hdf5"

    # 打开原始 HDF5 文件并创建新的文件
    with h5py.File(input_file, "r") as infile, h5py.File(output_file, "w") as outfile:
        # Step 1: 读取原始数据
        qpos = infile["qpos"][:]
        qvel = infile["qvel"][:]
        image_110 = infile["image_110"][:]
        image_100 = infile["image_100"][:]

        # Step 2: 创建 action 数据集
        outfile.create_dataset("action", data=qpos)

        # Step 3: 创建 observations 组并存入 qpos 和 qvel
        observations_group = outfile.create_group("observations")
        observations_group.create_dataset("qpos", data=qpos)
        observations_group.create_dataset("qvel", data=qvel)

        # Step 4: 创建 observations/images 子组并存入图像数据
        images_group = observations_group.create_group("images")
        images_group.create_dataset("image_110", data=image_110)
        images_group.create_dataset("image_100", data=image_100)

    print(f"Restructured HDF5 file saved to: {output_file}")

