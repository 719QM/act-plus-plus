# 模拟环境的脚本化策略
# 返回action
import imageio
import math

import numpy
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from utils import increment_function
from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
import time
import os
import h5py

import IPython
e = IPython.embed


class BasePolicy:
    # 定义了__init__, generate_trajectory, interpolate, __call__函数，
    # 其中__init__和__call__函数会自动调用
    # generate_trajectory 函数在子类中被定义
    # 在__call__函数中会调用generate_trajectory, interpolate函数
    def __init__(self, inject_noise=False):
        # 初始化，接收参数inject_noise， 并定义步数和左右轨迹
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        # 该函数需要在子类中进行实现，如果没有实现则会抛异常
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        # 静态方法，可以在不创建类实例的情况下调用。静态方法没有self参数，不能访问实例属性和方法。
        # 用于在两个路标点之间插值得到当前姿势和夹爪指令
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        # 如果 t_frac 是0，表示当前时间与当前路标点的时间相等，如果 t_frac 是1，表示当前时间与下一个路标点的时间相等
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # 自动调用,当给一个对象传入参数后，就会自动调用call函数
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        # 这段代码的作用是从左右轨迹中获取当前时刻的路标点和下一个将要到达的路标点，以供后续的插值计算使用
        if self.left_trajectory[0]['t'] == self.step_count:
            # .pop(0) 会移除并返回列表中的第一个元素
            # current_waypoint为trajectory的第0个元素，next_waypoint为第一个元素
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([0, 0.5, 0.25])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # approach meet position
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # move to meet position
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0}, # close gripper
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # move left
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0}, # stay
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, # approach meet position
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # move to meet position
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # move to right
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1}, # stay
        ]


class InsertionPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, 0.5, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # insertion
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion

        ]
# TODO 添加RM机械臂的类，需要修改轨迹内容
class RMPolicy_simpletrajectory(BasePolicy):

    @staticmethod
    def rpy2R(rpy):  # [r,p,y] 单位rad
        rot_x = np.array([[1, 0, 0],
                          [0, math.cos(rpy[0]), -math.sin(rpy[0])],
                          [0, math.sin(rpy[0]), math.cos(rpy[0])]])
        rot_y = np.array([[math.cos(rpy[1]), 0, math.sin(rpy[1])],
                          [0, 1, 0],
                          [-math.sin(rpy[1]), 0, math.cos(rpy[1])]])
        rot_z = np.array([[math.cos(rpy[2]), -math.sin(rpy[2]), 0],
                          [math.sin(rpy[2]), math.cos(rpy[2]), 0],
                          [0, 0, 1]])
        R = np.dot(rot_z, np.dot(rot_y, rot_x))
        return R

    def count_lines(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()  # 读取所有行到一个列表
        return len(lines)  # 返回行数

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        # print(f'pos:',init_mocap_pose_right[0],init_mocap_pose_right[1],init_mocap_pose_right[2])

        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        init_quat_left = Quaternion(init_mocap_pose_left[3:])

        target_quat_left = Quaternion([1, 0, 0, 0])
        # 构造表示绕 z 轴旋转的四元数
        rotate_z = Quaternion(axis=[0, 0, 1], angle=0.4)
        rotate_y = Quaternion(axis=[0, 1, 0], angle=0.3) #正是向上抬
        target_quat_left = target_quat_left * rotate_z
        # print(f"target_quat: ",target_quat_left)

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:7]
        # print(f"Generate trajectory for {box_xyz=}")

        # forward_quat = Quaternion(axis=[0.0, 0.0, 1.0], degrees=0)
        # #
        # # euler = [0, -1, 0]
        # # R = self.rpy2R(euler)
        # # test_quat = Quaternion._from_matrix(R)
        #
        # target_euler=[0, -1.3, 0]
        # target_R = self.rpy2R(target_euler)
        # target_Q = Quaternion._from_matrix(target_R)
        #
        # init_euler = [0, -1, -1.57]
        # print(f"policy: ",increment_function())
        episode_number = increment_function()
        print(f"policy: ", episode_number)
        # 使用格式化字符串创建文件名
        filename = f"Astar_data/output_{episode_number}.txt"
        # filename = f"Astar_data/output_20.txt"
        with open(filename, 'r') as file:
            # 初始化行号计数器
            line_number = 0

            # 获得文件行数
            lines = file.readlines()
            lines_total = len(lines)-3
            print(f"The file has {lines_total} lines.")
            # 重置文件读取到文件的开始位置
            file.seek(0)

            self.left_trajectory= [{"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1},]

            # 逐行读取文件
            for process_Q in Quaternion.intermediates(init_quat_left,target_quat_left, lines_total):
                line = file.readline()
                line_number += 1  # 每读取一行，行号加1
                target_line = line.strip()
                if target_line.startswith('target'):
                    target_pos = list(map(float, target_line.split(':')[1].strip().split()))
                # 如果是前3行，跳过不处理
                if line_number <= 3:
                    continue
                if line_number ==lines_total:  # 如果读取到文件末尾，结束循环
                    self.left_trajectory.append({"t": 1000, "xyz":target_pos, "quat": target_quat_left.elements, "gripper": 0},)
                    break
                # 去除行尾的换行符并分割行
                numbers = line.strip().split()
                # 检查是否有数字存在
                if numbers:
                    # 获取第一个数字，即列表中的第一个元素
                    text_x = numbers[0]
                    text_y = numbers[1]
                    text_z = numbers[2]
                    self.left_trajectory.append({"t": line_number*10, "xyz":np.array([float(text_x), float(text_y), float(text_z)]), "quat": process_Q.elements, "gripper": 1},)

        # self.left_trajectory = [
        #     {"t": 0, "xyz": init_mocap_pose_left[:3] , "quat": init_mocap_pose_left[3:], "gripper": 1},
        #     {"t": 700, "xyz": [-0.4206847, 0.56161671, 0.40360408], "quat": [1,0,0,0], "gripper": 1},
        # ]
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1},
                    {"t": 1000, "xyz": init_mocap_pose_right[:3] , "quat": init_mocap_pose_right[3:], "gripper": 1},
                ]
        # self.left_trajectory =  [
        # 搬方块
        # ////////////////////////////////////////////////////////////////////////////////////////////////////////
        # {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1},  # sleep
        # # approach meet position
        # {"t": 100, "xyz": box_xyz + np.array([0, 0.15, 0+0.1]), "quat": approach_quat_left.elements, "gripper": 1},
        # # rotate
        # {"t": 200, "xyz": box_xyz + np.array([0, 0.15, 0+0.1]), "quat": rotate_quat_left.elements, "gripper": 1},
        # {"t": 300, "xyz": box_xyz + np.array([0, 0.05, 0+0.1]), "quat": rotate_quat_left.elements, "gripper": 1},
        # {"t": 350, "xyz": box_xyz + np.array([0, 0.05, 0+0.1]), "quat": rotate_quat_left.elements, "gripper": 0},
        # {"t": 400, "xyz": box_xyz + np.array([0, 0.05, 0+0.1]), "quat": rotate_quat_left.elements, "gripper": 0},
        # {"t": 450, "xyz": box_xyz + np.array([0, 0.05, 0.15]), "quat": rotate_quat_left.elements, "gripper": 0},
        # {"t": 500, "xyz": box_xyz + np.array([0, 0.05, 0.15]), "quat": rotate_quat_left.elements, "gripper": 0},
        # ////////////////////////////////////////////////////////////////////////////////////////////////////////

        # {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1},  # sleep
        # # approach meet position
        # # {"t": 100, "xyz": box_xyz + np.array([0+0.1, 0, 0]), "quat": approach_quat_left.elements,
        # #  "gripper": 1},
        # # # rotate
        # # {"t": 200, "xyz": box_xyz + np.array([0, 0, 0]), "quat": approach_quat_left.elements, "gripper": 1},
        # # {"t": 300, "xyz": box_xyz + np.array([0-0.1, 0, 0]), "quat": approach_quat_left.elements, "gripper": 1},
        # {"t": 500, "xyz": [-0.4, 0.4, 0.2], "quat": approach_quat_left.elements, "gripper": 0},

        # ]

        # self.right_trajectory = [
        #     # 搬方块
        #     # ////////////////////////////////////////////////////////////////////////////////////////////////////////
        #     # {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1},  # sleep
        #     # # approach meet position
        #     # {"t": 100, "xyz": box_xyz + np.array([0, -0.15, 0+0.1]), "quat": approach_quat_right.elements, "gripper": 1},
        #     # # rotate
        #     # {"t": 200, "xyz": box_xyz + np.array([0, -0.15, 0+0.1]), "quat": rotate_quat_right.elements, "gripper": 1},
        #     # {"t": 300, "xyz": box_xyz + np.array([0, -0.05, 0+0.1]), "quat": rotate_quat_right.elements, "gripper": 1},
        #     # {"t": 350, "xyz": box_xyz + np.array([0, -0.05, 0+0.1]), "quat": rotate_quat_right.elements, "gripper": 0},
        #     # {"t": 400, "xyz": box_xyz + np.array([0, -0.05, 0+0.1]), "quat": rotate_quat_right.elements, "gripper": 0},
        #     # {"t": 450, "xyz": box_xyz + np.array([0, -0.05, 0.15]), "quat": rotate_quat_right.elements, "gripper": 0},
        #     # {"t": 500, "xyz": box_xyz + np.array([0, -0.05, 0.15]), "quat": rotate_quat_right.elements, "gripper": 0},
        #     # ////////////////////////////////////////////////////////////////////////////////////////////////////////
        #     {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1},
        #     {"t": 1000, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1},
        #
        # ]

class RMPolicy_fireextinguisher(BasePolicy):

    @staticmethod
    def rpy2R(rpy):  # [r,p,y] 单位rad
        rot_x = np.array([[1, 0, 0],
                          [0, math.cos(rpy[0]), -math.sin(rpy[0])],
                          [0, math.sin(rpy[0]), math.cos(rpy[0])]])
        rot_y = np.array([[math.cos(rpy[1]), 0, math.sin(rpy[1])],
                          [0, 1, 0],
                          [-math.sin(rpy[1]), 0, math.cos(rpy[1])]])
        rot_z = np.array([[math.cos(rpy[2]), -math.sin(rpy[2]), 0],
                          [math.sin(rpy[2]), math.cos(rpy[2]), 0],
                          [0, 0, 1]])
        R = np.dot(rot_z, np.dot(rot_y, rot_x))
        return R

    def count_lines(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()  # 读取所有行到一个列表
        return len(lines)  # 返回行数

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        # print(f'pos:',init_mocap_pose_right[0],init_mocap_pose_right[1],init_mocap_pose_right[2])


        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        init_quat_left = Quaternion(init_mocap_pose_left[3:])

        fire_extinguisher_info = np.array(ts_first.observation['env_state'])
        fire_extinguisher_xyz = fire_extinguisher_info[:3]
        fire_extinguisher_quat = fire_extinguisher_info[3:7]
        # print(f"Generate trajectory for {box_xyz=}")

        # approach_quat_left = Quaternion(axis=[0.0, 0.0, 1.0], degrees=90)
        # approach_quat_right = Quaternion(axis=[0.0, 0.0, 1.0], degrees=-90)
        #
        rotate_quat_left = Quaternion(axis=[0.0, 1.0, 0.0], degrees=20)
        rotate_quat_left = init_quat_left * rotate_quat_left

        lift_quat_left = Quaternion(axis=[0.0, 1.0, 0.0], degrees=20)
        lift_quat_left = rotate_quat_left * lift_quat_left
        # rotate_quat_right = Quaternion(axis=[1.0, 0.0, 0.0], degrees=-90)
        # rotate_quat_right = approach_quat_right * rotate_quat_right
        #
        #
        # gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        # gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)
        #
        # meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)
        #
        # meet_xyz = np.array([-0.2, 0.5, 0.05])

        forward_quat = Quaternion(axis=[0.0, 0.0, 1.0], degrees=0)
        #
        # euler = [0, -1, 0]
        # R = self.rpy2R(euler)
        # test_quat = Quaternion._from_matrix(R)

        target_euler=[0, -1.3, 0]
        target_R = self.rpy2R(target_euler)
        target_Q = Quaternion._from_matrix(target_R)

        init_euler = [0, -1, -1.57]
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1},  # sleep
            # {"t": 100, "xyz": fire_extinguisher_xyz + np.array([-0.2, 0.2, 0.5]), "quat": init_mocap_pose_left[3:],
            #  "gripper": 1},  # approach the fire_extinguisher
            # {"t": 200, "xyz": fire_extinguisher_xyz + np.array([-0.2, 0.2, 0.5]), "quat": rotate_quat_left.elements,
            #  "gripper": 1},  # go closer
            # {"t": 300, "xyz": fire_extinguisher_xyz + np.array([-0.25, 0.03, 0.5]), "quat": rotate_quat_left.elements,
            #  "gripper": 1},  # go closer
            # {"t": 400, "xyz": fire_extinguisher_xyz + np.array([-0.3, 0.038, 0.45]), "quat": rotate_quat_left.elements,
            #  "gripper": 0},  # close gripper
            # {"t": 500, "xyz": fire_extinguisher_xyz + np.array([-0.3, 0.03, 0.6]), "quat": lift_quat_left.elements,
            #  "gripper": 0},  # lift
            {"t": 500, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1},  # sleep
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1},  # sleep
            {"t": 500, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1},  # stay

        ]

def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False
    end_effector_pos = []

    for episode_idx in range(20):
        # setup the environment
        episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
        if 'sim_transfer_cube' in task_name:
            env = make_ee_sim_env('sim_transfer_cube')
        elif 'sim_insertion' in task_name:
            env = make_ee_sim_env('sim_insertion')
        elif 'sim_RM_simpletrajectory' in task_name:
            env = make_ee_sim_env('sim_RM_simpletrajectory')
        elif 'sim_RM_fire_extinguisher' in task_name:
            env = make_ee_sim_env('sim_RM_fire_extinguisher')
        else:
            raise NotImplementedError

        ts = env.reset()
        # 改变机械臂初始位置时需要以下信息
        # print(f"left_init pos:",env._physics.named.data.xpos['handforcesensor3'])
        # print(f"left_init quat:",env._physics.named.data.xquat['handforcesensor3'])
        # print(f"left_mocap pos:",env._physics.named.data.xpos['mocap_left'])
        # print(f"left_mocap quat:",env._physics.named.data.xquat['mocap_left'])
        # print(f"right_init pos:",env._physics.named.data.xpos['handforcesensor4'])
        # print(f"right_init quat:",env._physics.named.data.xquat['handforcesensor4'])
        # print(f"right_mocap pos:",env._physics.named.data.xpos['mocap_right'])
        # print(f"right_mocap quat:",env._physics.named.data.xquat['mocap_right'])
        episode = [ts]
        if onscreen_render:
            # 录视频
            # writer = imageio.get_writer(f'episode_{episode_idx}.mp4', fps=30)
            # ax = plt.subplot()
            # plt_img = ax.imshow(ts.observation['images']['top'])

            # 创建两个子图，分别用于展示 'top' 和 'angle' 相机视角
            fig, (ax1, ax2) = plt.subplots(1, 2)  # 1行2列的子图布局

            # 显示 'top' 相机图像
            plt_img_top = ax1.imshow(ts.observation['images']['top'])
            ax1.set_title('Top Camera')

            # 显示 'angle' 相机图像
            plt_img_angle = ax2.imshow(ts.observation['images']['angle'])
            ax2.set_title('Angle Camera')
            plt.ion()

        policy = RMPolicy_simpletrajectory(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            # print(f"left_xpos:", env._physics.named.data.xpos['mocap_left'])
            # print(f"right_xpos:", env._physics.named.data.xpos['mocap_right'])
            # print(f"left_xquat:", env._physics.named.data.xquat['mocap_left'])
            # print(f"right_xquat:", env._physics.named.data.xquat['mocap_right'])
            # print(f"left_qpos1:",env._physics.named.data.qpos['left_joint_1'])
            # print(f"left_qpos2:",env._physics.named.data.qpos['left_joint_2'])
            # print(f"left_qpos3:",env._physics.named.data.qpos['left_joint_3'])
            # print(f"left_qpos4:",env._physics.named.data.qpos['left_joint_4'])
            # print(f"left_qpos5:",env._physics.named.data.qpos['left_joint_5'])
            # print(f"left_qpos6:",env._physics.named.data.qpos['left_joint_6'])
            # print(f"box: ",env._physics.named.data.qpos['red_box_joint'])

            episode.append(ts)
            if onscreen_render:
                # image = ts.observation['images']['angle']  # 假设这是图像数据
                # # writer.append_data(image)  # 将图像帧写入视频
                #
                # plt_img.set_data(ts.observation['images']['angle'])
                # 更新 'top' 相机的图像
                plt_img_top.set_data(ts.observation['images']['top'])

                # 更新 'angle' 相机的图像
                plt_img_angle.set_data(ts.observation['images']['angle'])
                plt.pause(0.02)
        plt.close()
        # writer.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        data_dict = {
            '/observations/qpos': [],
            '/action': [],
        }
        camera_names = SIM_TASK_CONFIGS['sim_RM_simpletrajectory']['camera_names']
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []
        episode = episode[:-1]
        max_timesteps = episode_len
        for t in range(max_timesteps):
            ts = episode[t]
            data_dict['/action'].append(ts.observation['qpos'])
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            # data_dict['/action'].append(ts.action)
            # for cam_name in camera_names:
            #     data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
            data_dict[f'/observations/images/top'].append(ts.observation['images']['top'])
            data_dict[f'/observations/images/angle'].append(ts.observation['images']['angle'])
            end_effector = np.copy(ts.observation['position'])
            end_effector_pos.append(end_effector)
            print(f"Accumulated Positions: {end_effector_pos[-1:]}")  # 打印最后一个位置


        # HDF5
        t0 = time.time()
        dataset_path = os.path.join('/home/juyiii/data/aloha/sim_RM_Astar_teleoperation', f'episode_{episode_idx}')
        # dataset_path = os.path.join('/home/juyiii/ALOHA/act-plus-plus/EEpos/20', f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')
        # np.savetxt('/home/juyiii/ALOHA/act-plus-plus/EEpos/20/Astar_new.txt', np.array(end_effector_pos), fmt='%.6f', delimiter=',', comments='')
        # print("End effector position has been saved! 'v'")



if __name__ == '__main__':
    test_task_name = 'sim_RM_simpletrajectory'
    test_policy(test_task_name)

