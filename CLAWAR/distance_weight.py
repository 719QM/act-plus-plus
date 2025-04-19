import numpy as np
import matplotlib.pyplot as plt

def point_to_aabb_distance(point, aabb_min, aabb_max):
    """计算点到AABB（轴对齐包围盒）的最小距离"""
    clipped = np.maximum(aabb_min, np.minimum(point, aabb_max))
    return np.linalg.norm(point - clipped)

def sigmoid_weight(d, d0=0.2, k=50):
    """基于距离计算轨迹权重，使用 Sigmoid 函数平滑过渡"""
    return 1 / (1 + np.exp(-k * (d - d0)))

def load_trajectory(file_path):
    """读取轨迹点 (x, y, z)"""
    return np.loadtxt(file_path)

if __name__ == "__main__":
    # === 设置路径和参数 ===
    file_path = "experiment/shared_control_verify/ee_pos.txt"
    sampling_frequency = 50  # Hz

    # === 设置障碍物的 AABB ===
    obstacle_center = np.array([-0.45, 0.75, 0.25])
    obstacle_size = np.array([0.2, 0.2, 0.3])
    obstacle_min = obstacle_center - obstacle_size / 2
    obstacle_max = obstacle_center + obstacle_size / 2

    # === 加载末端执行器轨迹 ===
    trajectory = load_trajectory(file_path)

    # === 计算每个时间步的距离和权重 ===
    distances = []
    weights = []

    for pt in trajectory:
        d = point_to_aabb_distance(pt, obstacle_min, obstacle_max)
        distances.append(d)
        weights.append(sigmoid_weight(d))  # 使用默认参数 d0=0.33, k=50

    distances = np.array(distances)
    weights = np.array(weights)
    timesteps = np.arange(len(distances)) / sampling_frequency  # 时间轴（单位：秒）

    # === 绘图 ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # 上图：距离 vs 时间
    ax1.plot(timesteps, distances, label="Distance to Obstacle", color='blue')
    ax1.axhline(0.2, color='gray', linestyle='--', label='Threshold d0=0.33')
    ax1.set_ylabel("Distance (m)")
    ax1.set_title("Distance to Obstacle Over Time")
    ax1.grid(True)
    ax1.legend()

    # 下图：权重 vs 时间
    ax2.plot(timesteps, weights, label="Sigmoid Weight", color='red')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Weight")
    ax2.set_title("Sigmoid-based Weight Over Time")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()
