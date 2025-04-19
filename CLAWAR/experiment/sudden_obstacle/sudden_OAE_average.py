import numpy as np
import os

def point_to_aabb_distance(point, aabb_min, aabb_max):
    """计算点到AABB（轴对齐包围盒）的最小距离"""
    clipped = np.maximum(aabb_min, np.minimum(point, aabb_max))  # 最近的包围盒点
    return np.linalg.norm(point - clipped)

def calculate_oae_dynamic(trajectory, obstacle_centers, obstacle_size, sampling_frequency=50, epsilon=1e-3):
    """计算动态障碍物的 OAE 指标"""
    if len(trajectory) != len(obstacle_centers):
        raise ValueError("轨迹和障碍物位置长度不一致")

    distances = []
    half_size = obstacle_size / 2.0
    for pt, obs_center in zip(trajectory, obstacle_centers):
        obs_min = obs_center - half_size
        obs_max = obs_center + half_size
        d = point_to_aabb_distance(pt, obs_min, obs_max)
        distances.append(d)
    distances = np.array(distances)

    # 轨迹时间（秒）
    step_length = len(trajectory)
    total_time = step_length / sampling_frequency

    # 威胁积分
    threat_integral = np.sum(1.0 / (distances + epsilon))

    # 最小距离
    min_distance = np.min(distances)

    # OAE
    oae = (min_distance / total_time) * (1.0 / threat_integral)

    return oae, min_distance, total_time, threat_integral

def load_trajectory(file_path):
    return np.loadtxt(file_path)

def load_obstacle_positions(file_path):
    """只提取障碍物的位置（前三列）"""
    data = np.loadtxt(file_path)
    return data[:, :3]  # 只保留位置部分（x, y, z）


if __name__ == "__main__":
    folder_path = "TAB"  # 文件夹路径
    obstacle_size = np.array([0.2, 0.2, 0.3])  # 固定障碍物尺寸
    sampling_frequency = 50

    oae_values = []
    min_distances = []
    total_times = []
    threat_integrals = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt") and "_obs" not in file_name:
            traj_path = os.path.join(folder_path, file_name)
            obs_path = os.path.join(folder_path, file_name.replace(".txt", "_obs.txt"))

            if not os.path.exists(obs_path):
                print(f"缺少障碍物文件: {obs_path}")
                continue

            trajectory = load_trajectory(traj_path)
            obstacle_centers = load_obstacle_positions(obs_path)

            oae, min_d, total_time, threat = calculate_oae_dynamic(
                trajectory, obstacle_centers, obstacle_size, sampling_frequency
            )

            oae_values.append(oae)
            min_distances.append(min_d)
            total_times.append(total_time)
            threat_integrals.append(threat)

            print(f"{file_name}: OAE={oae:.10f}, min_d={min_d:.4f}, time={total_time:.4f}s, threat={threat:.4f}")

    # 汇总统计
    print("\n===== 统计结果 =====")
    if oae_values:
        print(f"OAE 平均值: {np.mean(oae_values):.10f}")
        print(f"最小障碍物距离 平均值: {np.mean(min_distances):.4f} m")
        print(f"轨迹执行总时间 平均值: {np.mean(total_times):.4f} s")
        print(f"威胁积分 平均值: {np.mean(threat_integrals):.4f} m⁻¹")
    else:
        print("未成功计算任何轨迹。")
