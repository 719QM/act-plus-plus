import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# 读取txt文件并处理数据
def read_data(file_path):
    start = None
    target = None
    obstacle = None
    path_points = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # 读取前三行
        for i, line in enumerate(lines[:3]):
            if line.startswith("start:"):
                start = list(map(float, line.split()[1:]))
            elif line.startswith("target:"):
                target = list(map(float, line.split()[1:]))
            elif line.startswith("obstacle:"):
                obstacle = list(map(float, line.split()[1:]))

        # 读取路径点
        for line in lines[3:]:
            path_points.append(list(map(float, line.split())))

    return start, target, obstacle, path_points


# 绘制路径和长方体
def plot_path_and_obstacle(path_points, obstacle):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 连接路径点
    path_points = np.array(path_points)
    ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], marker='o', color='b', label='Path')

    # 绘制障碍物长方体
    # 长方体的中心在obstacle点，长宽高分别为0.6, 0.6, 0.8
    dx, dy, dz = 0.6, 0.6, 0.8
    x, y, z = obstacle

    # 定义长方体的8个顶点
    vertices = np.array([
        [x - dx / 2, y - dy / 2, z - dz / 2],
        [x + dx / 2, y - dy / 2, z - dz / 2],
        [x + dx / 2, y + dy / 2, z - dz / 2],
        [x - dx / 2, y + dy / 2, z - dz / 2],
        [x - dx / 2, y - dy / 2, z + dz / 2],
        [x + dx / 2, y - dy / 2, z + dz / 2],
        [x + dx / 2, y + dy / 2, z + dz / 2],
        [x - dx / 2, y + dy / 2, z + dz / 2]
    ])

    # 定义长方体的6个面
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]]
    ]

    # 绘制长方体
    ax.add_collection3d(Poly3DCollection(faces, facecolors='r', linewidths=1, edgecolors='k', alpha=0.3))

    # 设置图形显示范围
    ax.set_xlim([min(path_points[:, 0]) - 1, max(path_points[:, 0]) + 1])
    ax.set_ylim([min(path_points[:, 1]) - 1, max(path_points[:, 1]) + 1])
    ax.set_zlim([min(path_points[:, 2]) - 1, max(path_points[:, 2]) + 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


# 主函数
def main():
    file_path = "output_108.txt"  # 请替换成你文件的路径
    start, target, obstacle, path_points = read_data(file_path)

    if start and target and obstacle and path_points:
        print("Start:", start)
        print("Target:", target)
        print("Obstacle:", obstacle)
        print("Path Points:", path_points)

        plot_path_and_obstacle(path_points, obstacle)
    else:
        print("文件格式错误或缺少数据")


if __name__ == "__main__":
    main()
