import numpy as np
import os

def point_to_aabb_distance(point, aabb_min, aabb_max):
    """计算点到AABB（轴对齐包围盒）的最小距离"""
    clipped = np.maximum(aabb_min, np.minimum(point, aabb_max))  # 最近的包围盒点
    return np.linalg.norm(point - clipped)  # 计算欧氏距离

def calculate_oae(trajectory, obstacle_min, obstacle_max, sampling_frequency=50, epsilon=1e-3):
    """计算避障效率指数（OAE）"""
    distances = np.array([point_to_aabb_distance(pt, obstacle_min, obstacle_max) for pt in trajectory])

    # 计算执行时间（秒）
    step_length = len(trajectory)
    total_time = step_length / sampling_frequency

    # 计算威胁积分
    threat_integral = np.sum(1.0 / (distances + epsilon))

    # 计算最小距离
    min_distance = np.min(distances)

    # 计算 OAE
    oae = (min_distance / total_time) * (1.0 / threat_integral)

    return oae, min_distance, total_time, threat_integral

def load_trajectory(file_path):
    """读取txt轨迹数据，每行x, y, z"""
    return np.loadtxt(file_path)

if __name__ == "__main__":
    folder_path = "k_10"  # 轨迹文件夹路径
    sampling_frequency = 50  # 轨迹采集频率（Hz）

    # 障碍物信息（AABB）
    obstacle_center = np.array([-0.45, 0.75, 0.25])
    obstacle_size = np.array([0.2, 0.2, 0.3])  # (长, 宽, 高)
    obstacle_min = obstacle_center - obstacle_size / 2
    obstacle_max = obstacle_center + obstacle_size / 2

    # 读取所有轨迹文件
    oae_values = []
    min_distances = []
    total_times = []
    threat_integrals = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):  # 只处理txt文件
            file_path = os.path.join(folder_path, file_name)
            trajectory = load_trajectory(file_path)

            oae, min_d, total_time, threat = calculate_oae(trajectory, obstacle_min, obstacle_max, sampling_frequency)

            oae_values.append(oae)
            min_distances.append(min_d)
            total_times.append(total_time)
            threat_integrals.append(threat)

            print(f"{file_name}: OAE={oae:.10f}, min_d={min_d:.4f}, time={total_time:.4f}s, threat={threat:.4f}")

    # 计算均值
    mean_oae = np.mean(oae_values)
    mean_min_distance = np.mean(min_distances)
    mean_total_time = np.mean(total_times)
    mean_threat_integral = np.mean(threat_integrals)

    print("\n===== 统计结果 =====")
    print(f"OAE 平均值: {mean_oae:.10f}")
    print(f"最小障碍物距离 平均值: {mean_min_distance:.4f} m")
    print(f"轨迹执行总时间 平均值: {mean_total_time:.4f} s")
    print(f"威胁积分 平均值: {mean_threat_integral:.4f} m⁻¹")
