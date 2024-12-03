import h5py
import numpy as np
import os


def adjust_hdf5_data(input_file, output_file, target_length=90):
    """
    Adjust HDF5 data to a fixed frame length by random sampling or duplication.

    Parameters:
    - input_file: str, the path to the input HDF5 file.
    - output_file: str, the path to the output HDF5 file.
    - target_length: int, the target number of frames for all datasets.
    """

    def random_sample(data, length):
        """Randomly sample indices from data to reach the desired length."""
        indices = np.sort(np.random.choice(len(data), size=length, replace=False))
        return data[indices]

    def random_pad(data, length):
        """Randomly pad data to reach the desired length."""
        current_length = len(data)
        while len(data) < length:
            idx = np.random.randint(current_length)
            data = np.insert(data, idx + 1, data[idx], axis=0)  # Duplicate randomly
        return data

    def process_group(in_group, out_group):
        """
        Recursively process HDF5 groups and adjust datasets.

        Parameters:
        - in_group: h5py.Group, the input group to process.
        - out_group: h5py.Group, the output group to write adjusted data.
        """
        for key in in_group.keys():
            if isinstance(in_group[key], h5py.Dataset):
                # Handle datasets (e.g., action, qpos, qvel, images)
                data = in_group[key][:]
                if len(data) > target_length:
                    adjusted_data = random_sample(data, target_length)
                elif len(data) < target_length:
                    adjusted_data = random_pad(data, target_length)
                else:
                    adjusted_data = data
                out_group.create_dataset(key, data=adjusted_data)
            elif isinstance(in_group[key], h5py.Group):
                # Handle nested groups (e.g., observations, images)
                subgroup = out_group.create_group(key)
                process_group(in_group[key], subgroup)

    # Open input and output HDF5 files
    with h5py.File(input_file, 'r') as infile, h5py.File(output_file, 'w') as outfile:
        process_group(infile, outfile)

    print(f"Adjusted data saved to {output_file}")


max_episode = 38
for i in range(max_episode):
    input_hdf5 = f"/home/juyiii/data/aloha/rmreal_pick_90/episode_{i}.hdf5"
    output_hdf5 = f"/home/juyiii/data/aloha/rmreal_pick/episode_{i}.hdf5"
    adjust_hdf5_data(input_hdf5, output_hdf5, target_length=100)
# input_hdf5 = "/home/juyiii/data/aloha/test/episode_0.hdf5"
# output_hdf5 = "/home/juyiii/data/aloha/test/episode_0_length.hdf5"
# adjust_hdf5_data(input_hdf5, output_hdf5, target_length=90)
