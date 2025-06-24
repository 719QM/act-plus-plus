import os
import h5py
import numpy as np

# 存储所有 action 长度
action_lengths = []

# 遍历 episode_0 到 episode_39
for i in range(40):
    filename = f"episode_{i}.hdf5"
    if os.path.exists(filename):
        try:
            with h5py.File(filename, "r") as f:
                if "action" in f:
                    length = len(f["action"])
                    action_lengths.append(length)
                    print(f"{filename}: action length = {length}")
                else:
                    print(f"{filename}: no 'action' dataset found.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    else:
        print(f"{filename} does not exist.")

# 输出最长和最短长度
if action_lengths:
    print("\nSummary:")
    print(f"Max action length: {max(action_lengths)}")
    print(f"Min action length: {min(action_lengths)}")
else:
    print("No valid 'action' datasets found.")

lengths = np.array(action_lengths)
mean_len = np.mean(lengths)
median_len = np.median(lengths)
max_len = np.max(lengths)
min_len = np.min(lengths)
std_len = np.std(lengths)
percentile_90 = np.percentile(lengths, 90)
percentile_95 = np.percentile(lengths, 95)

print(f"Mean: {mean_len:.2f}")
print(f"Median: {median_len}")
print(f"Std: {std_len:.2f}")
print(f"Min: {min_len}")
print(f"Max: {max_len}")
print(f"90th Percentile: {percentile_90}")
print(f"95th Percentile: {percentile_95}")
