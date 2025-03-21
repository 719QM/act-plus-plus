import numpy as np
import matplotlib.pyplot as plt

# Sigmoid函数定义
def sigmoid_weight(d, d0=0.33, k=10):
    return 1 / (1 + np.exp(-k * (d - d0)))

# 生成距离数据
d_values = np.linspace(0, 0.6, 100)  # 从 0 到 0.6，100个点
weights = sigmoid_weight(d_values)

# 绘制曲线
plt.figure(figsize=(8, 5))
plt.plot(d_values, weights, label=r'$\sigma(d) = \frac{1}{1 + e^{-k(d - d_0)}}$', color='b', linewidth=2)
plt.axvline(x=0.33, color='r', linestyle='--', label=r'$d_0 = 0.3$')
plt.xlabel("Distance to Obstacle (d)")
plt.ylabel("Weight")
plt.title("Sigmoid Function for Weight Adjustment")
plt.legend()
plt.grid(True)
plt.show()
