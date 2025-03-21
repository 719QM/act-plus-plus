import numpy as np
from scipy.signal import butter, filtfilt

def load_trajectory(file_path):
    """加载轨迹文件，返回N×3的NumPy数组"""
    return np.loadtxt(file_path)

def calculate_velocity(position, dt):
    """通过位置差分计算速度"""
    return np.gradient(position, axis=0) / dt

def calculate_acceleration(velocity, dt):
    """通过速度差分计算加速度"""
    return np.gradient(velocity, axis=0) / dt

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """设计低通滤波器，滤除高频噪声"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def calculate_rms_acceleration(acceleration):
    """计算加速度的均方根（RMS）"""
    return np.sqrt(np.mean(np.square(acceleration), axis=0))

def main():
    # 参数设置
    file_path = 'k_1000.txt'  # 轨迹文件路径
    sampling_rate = 100  # 采样频率 (Hz)
    dt = 1.0 / sampling_rate  # 时间步长 (s)
    filter_cutoff = 10  # 低通滤波器截止频率 (Hz)

    # 加载轨迹数据
    trajectory = load_trajectory(file_path)

    # 计算速度
    velocity = calculate_velocity(trajectory, dt)

    # 计算加速度
    acceleration = calculate_acceleration(velocity, dt)

    # 滤波加速度数据
    acceleration_filtered = butter_lowpass_filter(acceleration, filter_cutoff, sampling_rate)

    # 计算加速度RMS
    rms_acceleration = calculate_rms_acceleration(acceleration_filtered)

    # 输出结果
    print(f"X轴加速度RMS: {rms_acceleration[0]:.4f} m/s²")
    print(f"Y轴加速度RMS: {rms_acceleration[1]:.4f} m/s²")
    print(f"Z轴加速度RMS: {rms_acceleration[2]:.4f} m/s²")
    print(f"三维合成加速度RMS: {np.linalg.norm(rms_acceleration):.4f} m/s²")

if __name__ == "__main__":
    main()