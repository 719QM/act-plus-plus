import numpy as np

def point_to_aabb_distance(point, aabb_min, aabb_max):
    """计算点到AABB（轴对齐包围盒）的最小距离"""
    clipped = np.maximum(aabb_min, np.minimum(point, aabb_max))  # 最近的包围盒点
    return np.linalg.norm(point - clipped)  # 计算欧氏距离

def calculate_oae(trajectory, obstacle_min, obstacle_max, sampling_frequency=50, epsilon=1e-3):
    """
    计算避障效率指数（OAE）

    参数：
    - trajectory: N×3数组，轨迹点坐标（x, y, z）
    - obstacle_min: 障碍物AABB的最小坐标
    - obstacle_max: 障碍物AABB的最大坐标
    - sampling_frequency: 轨迹采集频率（Hz），默认为50Hz
    - epsilon: 防止除以零的小常数

    返回：
    - oae: 避障效率指数
    - min_distance: 轨迹中的最小障碍物距离
    - total_time: 轨迹执行的总时间（秒）
    - threat_integral: 威胁积分
    """
    # 计算每个轨迹点到障碍物包围盒的距离
    distances = np.array([point_to_aabb_distance(pt, obstacle_min, obstacle_max) for pt in trajectory])

    # 计算执行时间（秒）
    step_length = len(trajectory)  # 时间步数
    total_time = step_length / sampling_frequency  # 计算时间

    # 计算威胁积分（避免除以零）
    threat_integral = np.sum(1.0 / (distances + epsilon))

    # 计算最小距离
    min_distance = np.min(distances)

    # 计算OAE
    oae = (min_distance / total_time) * (1.0 / threat_integral)

    return oae, min_distance, total_time, threat_integral

# 读取轨迹文件
def load_trajectory(file_path):
    """读取txt轨迹数据，每行x, y, z"""
    return np.loadtxt(file_path)

if __name__ == "__main__":
    trajectory_file = "k_1000.txt"  # 轨迹文件路径
    trajectory = load_trajectory(trajectory_file)  # 读取轨迹

    # 障碍物信息（AABB）
    obstacle_center = np.array([-0.45, 0.75, 0.25])
    obstacle_size = np.array([0.2, 0.2, 0.3])  # (长, 宽, 高)
    obstacle_min = obstacle_center - obstacle_size / 2
    obstacle_max = obstacle_center + obstacle_size / 2

    # 计算OAE
    oae, min_d, total_time, threat = calculate_oae(trajectory, obstacle_min, obstacle_max)

    print(f"避障效率指数 (OAE): {oae:.10f}")
    print(f"最小障碍物距离: {min_d:.4f} m")
    print(f"轨迹执行总时间: {total_time:.4f} s")
    print(f"威胁积分: {threat:.4f} m⁻¹")
