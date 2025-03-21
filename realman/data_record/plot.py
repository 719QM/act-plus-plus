import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    """加载txt文件，每行表示一个时间步，每列表示一个关节角度"""
    return np.loadtxt(file_path)


def plot_joint_angles(ref_file, actual_file, save_path=None):
    """绘制16个关节角度的参考值和实际值随时间变化的曲线"""
    ref_data = load_data(ref_file)  # 读取参考角度 
    actual_data = load_data(actual_file)  # 读取实际角度

    num_joints = 16  # 关节数
    time_steps = ref_data.shape[0]  # 时间步长
    time = np.arange(time_steps)  # 生成时间序列

    fig, axes = plt.subplots(num_joints, 1, figsize=(8, 2 * num_joints), sharex=True)

    for i in range(num_joints):
        axes[i].plot(time, ref_data[:, i], label='Reference', linestyle='--', color='blue')
        axes[i].plot(time, actual_data[:, i], label='Actual', linestyle='-', color='red')
        axes[i].set_ylabel(f'Joint {i + 1} (deg)')
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel('Time Step')
    fig.suptitle('Joint Angle Comparison')
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


# 示例调用
ref_file = 'policy_reference_1.txt'  # 参考角度文件
actual_file = 'qpos_1.txt'  # 实际角度文件
plot_joint_angles(ref_file, actual_file, save_path='joint_angle_comparison_1.png')
