import numpy as np
import matplotlib.pyplot as plt

def plot_distance_from_txt(file_path):
    # 读取数据
    distances = np.loadtxt(file_path)

    # 生成步长索引
    steps = np.arange(len(distances))

    # 绘制图像
    plt.figure(figsize=(10, 5))
    plt.plot(steps, distances, marker='o', linestyle='-', color='b', label="Distance to Obstacle")
    plt.xlabel("Step")
    plt.ylabel("Distance")
    plt.title("Distance to Obstacle Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

# 运行示例
file_path = "distance2.txt"  # 替换为你的txt文件路径
plot_distance_from_txt(file_path)
