import numpy as np
import os

def point_to_aabb_distance(point, aabb_min, aabb_max):
    """计算点到AABB（轴对齐包围盒）的最小距离"""
    clipped = np.maximum(aabb_min, np.minimum(point, aabb_max))
    return np.linalg.norm(point - clipped)

def calculate_oae(trajectory, obstacle_positions, box_size, sampling_frequency=50, epsilon=1e-3):
    """计算避障效率指数（OAE）"""
    distances = []
    for pt, obs_pos in zip(trajectory, obstacle_positions):
        aabb_min = obs_pos - box_size / 2
        aabb_max = obs_pos + box_size / 2
        dist = point_to_aabb_distance(pt, aabb_min, aabb_max)
        distances.append(dist)
    distances = np.array(distances)

    step_length = len(trajectory)
    total_time = step_length / sampling_frequency
    threat_integral = np.sum(1.0 / (distances + epsilon))
    min_distance = np.min(distances)
    oae = (min_distance / total_time) * (1.0 / threat_integral)

    return oae, min_distance, total_time, threat_integral

def load_trajectory(file_path):
    return np.loadtxt(file_path)

def load_obstacle_positions(file_path):
    data = np.loadtxt(file_path)
    return data[:, :3]  # 只保留位置 x, y, z

def find_switch_index(obstacle_positions, stable_threshold=0.01, min_stable_frames=10):
    num_points = len(obstacle_positions)

    for start in range(num_points - min_stable_frames, 0, -1):
        segment = obstacle_positions[start:start + min_stable_frames]
        max_displacement = np.max(np.linalg.norm(segment - segment[0], axis=1))
        if max_displacement < stable_threshold:
            stable_position = np.mean(segment, axis=0)

            for i in range(start - 1, -1, -1):
                dist = np.linalg.norm(obstacle_positions[i] - stable_position)
                if dist > 2 * stable_threshold:
                    return i + 1
    return 0  # 如果没找到明显变化，就从头开始

if __name__ == "__main__":
    folder_path = "TAB"  # 文件夹路径
    sampling_frequency = 50
    box_size = np.array([0.2, 0.2, 0.3])  # AABB 尺寸

    oae_values = []
    min_distances = []
    total_times = []
    threat_integrals = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt") and not file_name.endswith("_obs.txt"):
            traj_path = os.path.join(folder_path, file_name)
            obs_path = os.path.join(folder_path, file_name.replace(".txt", "_obs.txt"))

            if not os.path.exists(obs_path):
                print(f"⚠️ 缺少障碍物文件: {obs_path}")
                continue

            trajectory = load_trajectory(traj_path)
            obstacle_positions = load_obstacle_positions(obs_path)

            min_len = min(len(trajectory), len(obstacle_positions))
            trajectory = trajectory[:min_len]
            obstacle_positions = obstacle_positions[:min_len]

            switch_idx = find_switch_index(obstacle_positions)
            trajectory = trajectory[switch_idx:]
            obstacle_positions = obstacle_positions[switch_idx:]

            if len(trajectory) == 0:
                print(f"❌ 跳过空轨迹: {file_name}")
                continue

            oae, min_d, total_time, threat = calculate_oae(trajectory, obstacle_positions, box_size, sampling_frequency)

            oae_values.append(oae)
            min_distances.append(min_d)
            total_times.append(total_time)
            threat_integrals.append(threat)

            print(f"{file_name}: OAE={oae:.10f}, min_d={min_d:.4f}, time={total_time:.4f}s, threat={threat:.4f} (start from idx {switch_idx})")

    print("\n===== 统计结果 =====")
    if oae_values:
        print(f"OAE 平均值: {np.mean(oae_values):.10f}")
        print(f"最小障碍物距离 平均值: {np.mean(min_distances):.4f} m")
        print(f"轨迹执行总时间 平均值: {np.mean(total_times):.4f} s")
        print(f"威胁积分 平均值: {np.mean(threat_integrals):.4f} m⁻¹")
    else:
        print("没有有效轨迹用于统计。")
