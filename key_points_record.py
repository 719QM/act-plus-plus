from realman.robotic_arm_package.robotic_arm import *
import numpy as np
import argparse
import yaml
import os

class FlowStyleDumper(yaml.SafeDumper):
    pass

def represent_list_flow(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

FlowStyleDumper.add_representer(list, represent_list_flow)



class Realman_Dual_Arms:
    def __init__(self):
        left_ip = "192.168.1.19"
        right_ip = "192.168.1.18"
        self.left_arm = Arm(RM75, left_ip)
        self.right_arm = Arm(RM75, right_ip)

    def read_key_points(self):
        _, left_qpos = self.left_arm.Get_Joint_Degree()
        _, right_qpos = self.right_arm.Get_Joint_Degree()

        # TEST
        # left_qpos = np.random.uniform(low=-10.0, high=10.0, size=7)
        # right_qpos = np.random.uniform(low=-10.0, high=10.0, size=7)

        left_hand = np.array([1])
        right_hand = np.array([1])

        print("Left Arm Joint Positions:", left_qpos)
        print("Right Arm Joint Positions:", right_qpos)
        print("Left Hand Position:", left_hand)
        print("Right Hand Position:", right_hand)

        return left_qpos, right_qpos, left_hand, right_hand

    def save_key_points(self, filepath='joint_angles.yaml', new_episode=False):
        left_qpos, right_qpos, left_hand, right_hand = self.read_key_points()

        # 1. 将 numpy 转成 list
        # left_list = left_qpos.tolist()
        # right_list = right_qpos.tolist()
        left_hand_list = left_hand.tolist()
        right_hand_list = right_hand.tolist()

        left_list = left_qpos
        right_list = right_qpos


        # 2. 读取已有数据或初始化
        if new_episode or not os.path.exists(filepath):
            data = {}
        else:
            with open(filepath, 'r') as f:
                try:
                    data = yaml.safe_load(f) or {}
                except yaml.YAMLError:
                    data = {}

        # 3. 获取 step 索引
        step_id = len(data)
        step_key = f"step_{step_id}"

        # 4. 添加新数据
        data[step_key] = {
            'left_qpos': left_list,
            'right_qpos': right_list,
            'left_hand': left_hand_list,
            'right_hand': right_hand_list
        }

        # 5. 保存为 YAML，使用 FlowStyleDumper 控制列表横向显示
        with open(filepath, 'w') as f:
            yaml.dump(data, f, Dumper=FlowStyleDumper, sort_keys=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_episode', action='store_true')
    parser.add_argument('--filepath', type=str, default='collection/key_points.yaml')
    args = parser.parse_args()

    robot = Realman_Dual_Arms()
    robot.save_key_points(filepath=args.filepath, new_episode=args.new_episode)