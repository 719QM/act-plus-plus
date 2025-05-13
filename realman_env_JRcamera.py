import time
import numpy as np
import collections
import matplotlib.pyplot as plt
import dm_env
from pyquaternion import Quaternion
import pyrealsense2 as rs
import cv2
import h5py
# from Robotic_Arm.rm_robot_interface import *
from RM_real_constants import HAND_UNNORMALIZE, HAND_NORMALIZE
from realman.robotic_arm_package.robotic_arm import *
from visualize_episodes import save_videos
import pickle
import argparse
from einops import rearrange
import torch
from torchvision import transforms
import threading
from threading import Barrier
import queue
from pynput import keyboard
import os
import sys
sys.path.insert(0, "")  # 把上级目录加入 sys.path
datarecord = True

import IPython
e = IPython.embed

all_actions = []
pre_action = []
target_action = []

next_step_larm = []
next_step_rarm = []
next_step_lhand = []
next_step_rhand = []
next_step_action = []

qpos_record = []
policy_reference_record = []
image_record = []
# timestep_num = []

isemergency = False

# 全局变量
position_queue = queue.Queue()  # 存储目标位置
right_position_queue = queue.Queue()  # 存储右手目标位置
stop_flag = threading.Event()  # 控制线程停止
move_thread = None  # 左机械臂线程，初始为空
right_move_thread = None  # 右机械臂线程，初始为空
camera_thread = None  # 相机线程，初始为空
read_thread = None  # 读取关节角线程，初始为空
right_read_thread = None  # 读取右关节角线程，初始为空
read_left_joint = None
# read_right_joint = None
read_right_joint = [0, 0, 0, 0, 0, 0, 0]
read_lock = threading.Lock()


query_frequency = 100

# from ..constants import SIM_TASK_CONFIGS
# camera_names = SIM_TASK_CONFIGS['sim_RM_simpletrajectory']['camera_names']
camera_names = ['image_1', 'image_2']
image_1_thread = None
image_2_thread = None
image_lock = threading.Lock()


key_press = None

all_time_actions = torch.zeros([5000, 5000+100, 18]).cuda()

barrier = Barrier(2)  # 两个线程同步点
read_barrier = Barrier(2)  # 两个读取线程同步点

class RealmanEnv:
    """
    Environment for real robot bi-manual manipulation
    Action space:      [left_arm_qpos (7),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (7),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (7),          # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (7),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (7),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (7),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam": (480x640x3),        # h, w, c, dtype='uint8'
    """
    def __init__(self, level=3, mode=2):
        # Initialize camera
        self.cap = None
        self.cap1 = None
        self.init_JRcamera()
        self.qpos_obs = []
        # 等待一会儿
        time.sleep(1)
        # 初始化机器人，初始化相机
        self.start_keyboard_listener()
        left_ip = "192.168.1.19"
        right_ip = "192.168.1.18"
        self.left_arm = Arm(RM75, left_ip)
        self.right_arm = Arm(RM75, right_ip)
        self.leftarm_init = [-13.04299259185791,	25.120895385742188,	141.19898986816406,	-93.86701202392578,	11.084973335266113,	-28.841115951538086,	102.23300170898438]
        self.rightarm_init = [34.81800079345703,	11.810999870300293,	70.34600067138672,	98.68399810791016,	120.12000274658203,	-72.572998046875,	-51.494998931884766]
        # self.leftarm_init = [0, 0, 0, 0, 0, 0, 0]
        # self.rightarm_init = [0, 0, 0, 0, 0, 0, 0]
        self.lefthand_init = [999, 999, 999, 999, 999, 999]
        self.righthand_init = [999, 999, 999, 999, 999, 999]



    # 机械臂运动线程
    def move_arm(self):
        global key_press, move_thread, stop_flag
        print("move_left_arm")
        move_start_time = time.time()
        move_last_time = move_start_time
        while not stop_flag.is_set():
            barrier.wait()  # 等右手准备好
            if key_press is not None:
                if key_press.char.lower() == 's':  # 处理大小写
                    if move_thread and move_thread.is_alive():
                        self.emergency_stop()
                        stop_flag.set()  # 触发停止标志
                        print(f"stop_flag:{stop_flag}, 机械臂线程已停止")
            if not stop_flag.is_set():
                try:
                    next_step_action = position_queue.get(timeout=0.1)
                    # print("move arm next step: ", next_step_action)
                    if next_step_action is None:
                        print("收到 None，等待新指令...")
                        pass
                    else:
                        state_len = int(len(next_step_action) / 2)
                        next_step_larm = next_step_action[:state_len-1]
                        next_step_rarm = next_step_action[state_len:-1]
                        # print("next_step_action: ", next_step_action)

                        self.left_arm.Movej_Cmd(next_step_larm, v=20, r=20, trajectory_connect=0, block=0)
                        # self.right_arm.Movej_Cmd(next_step_rarm, v=1, r=0, trajectory_connect=0, block=0)


                        # print("归一化手指：", next_step_action[7], next_step_action[15])

                        # print("手指动作：", HAND_UNNORMALIZE(next_step_action[7]), HAND_UNNORMALIZE(next_step_action[15]))

                        left_hand_angle = [int(a) for a in HAND_UNNORMALIZE(next_step_action[7])]
                        right_hand_angle = [int(a) for a in HAND_UNNORMALIZE(next_step_action[15])]
                        self.left_arm.Set_Hand_Angle(left_hand_angle, block=0)
                        # self.right_arm.Set_Hand_Angle(right_hand_angle, block=0)

                        move_now_time = time.time()
                        print(f"Move Arm Time: {move_now_time - move_start_time:.2f}, Move Arm FPS: {1 / (move_now_time - move_last_time):.2f} \n")
                        move_last_time = move_now_time
                except queue.Empty:
                    pass

    # 机械臂右臂运动线程
    def move_right_arm(self):
        global key_press, right_move_thread, stop_flag
        print("move right arm")
        move_start_time = time.time()
        move_last_time = move_start_time
        while not stop_flag.is_set():
            barrier.wait()  # 等左手准备好
            if key_press is not None:
                if key_press.char.lower() == 's':  # 处理大小写
                    if right_move_thread and right_move_thread.is_alive():
                        self.emergency_stop()
                        stop_flag.set()  # 触发停止标志
                        print(f"stop_flag:{stop_flag}, 机械臂线程已停止")
            if not stop_flag.is_set():
                try:
                    next_step_action = right_position_queue.get(timeout=0.1)
                    # print("move arm next step: ", next_step_action)
                    if next_step_action is None:
                        print("收到 None，等待新指令...")
                        pass
                    else:
                        state_len = int(len(next_step_action) / 2)
                        next_step_larm = next_step_action[:state_len-1]
                        next_step_rarm = next_step_action[state_len:-1]
                        # print("next_step_action: ", next_step_action)

                        # self.left_arm.Movej_Cmd(next_step_larm, v=1, r=0, trajectory_connect=0, block=0)
                        self.right_arm.Movej_Cmd(next_step_rarm, v=20, r=20, trajectory_connect=0, block=0)


                        # print("归一化手指：", next_step_action[7], next_step_action[15])

                        # print("手指动作：", HAND_UNNORMALIZE(next_step_action[7]), HAND_UNNORMALIZE(next_step_action[15]))

                        left_hand_angle = [int(a) for a in HAND_UNNORMALIZE(next_step_action[7])]
                        right_hand_angle = [int(a) for a in HAND_UNNORMALIZE(next_step_action[15])]
                        # self.left_arm.Set_Hand_Angle(left_hand_angle, block=0)
                        self.right_arm.Set_Hand_Angle(right_hand_angle, block=0)

                        move_now_time = time.time()
                        print(f"Move Right Arm Time: {move_now_time - move_start_time:.2f}, Move Right Arm FPS: {1 / (move_now_time - move_last_time):.2f} \n")
                        move_last_time = move_now_time
                except queue.Empty:
                    pass

    def read_joint(self):
        global read_left_joint, read_right_joint
        read_start_time = time.time()
        read_last_time = read_start_time
        while not stop_flag.is_set():
            read_barrier.wait()
            try:
                _, left_qpos = self.left_arm.Get_Joint_Degree()
                # _, right_qpos = self.right_arm.Get_Joint_Degree()
                with read_lock:
                    read_left_joint = left_qpos
                    # read_right_joint = right_qpos

            except Exception as e:
                print(f"Error reading joint angles: {e}")

            read_now_time = time.time()
            print(f"Read Joint Time: {read_now_time - read_start_time:.2f}, Read Joint FPS: {1 / (read_now_time - read_last_time):.2f} \n")
            read_last_time = read_now_time

    def right_read_joint(self):
        global read_left_joint, read_right_joint
        read_start_time = time.time()
        read_last_time = read_start_time
        while not stop_flag.is_set():
            read_barrier.wait()
            try:
                # _, left_qpos = self.left_arm.Get_Joint_Degree()
                _, right_qpos = self.right_arm.Get_Joint_Degree()
                with read_lock:
                    # read_left_joint = left_qpos
                    read_right_joint = right_qpos

            except Exception as e:
                print(f"Error reading joint angles: {e}")

            read_now_time = time.time()
            print(f"Right Read Joint Time: {read_now_time - read_start_time:.2f}, Right Read Joint FPS: {1 / (read_now_time - read_last_time):.2f} \n")
            read_last_time = read_now_time

    def start_read_thread(self):
        global read_thread
        if read_thread is None or not read_thread.is_alive():
            stop_flag.clear()
            read_thread = threading.Thread(target=self.read_joint, daemon=True)
            read_thread.start()
            print("读取左臂关节角线程已启动")

    def start_right_read_thread(self):
        global right_read_thread
        if right_read_thread is None or not right_read_thread.is_alive():
            stop_flag.clear()
            right_read_thread = threading.Thread(target=self.right_read_joint, daemon=True)
            right_read_thread.start()
            print("读取右臂关节角线程已启动")

    # 启动线程函数
    def start_thread(self):
        global move_thread
        if move_thread is None or not move_thread.is_alive():  # 防止重复启动
            stop_flag.clear()  # 复位停止标志
            move_thread = threading.Thread(target=self.move_arm, daemon=True)  # 设置守护线程
            move_thread.start()
            print("move_thread is alive? ", move_thread.is_alive())

            print("机械臂线程已启动")

    def start_right_thread(self):
        global right_move_thread
        if right_move_thread is None or not right_move_thread.is_alive():  # 防止重复启动
            stop_flag.clear()  # 复位停止标志
            right_move_thread = threading.Thread(target=self.move_right_arm, daemon=True)  # 设置守护线程
            right_move_thread.start()
            print("right_move_thread is alive? ", right_move_thread.is_alive())

            print("右机械臂线程已启动")

    # 停止线程函数
    def stop_thread(self):
        global move_thread
        if move_thread and move_thread.is_alive():
            stop_flag.set()  # 触发停止标志
            position_queue.put(None)  # 发送退出信号
            move_thread.join()  # 等待线程结束
            move_thread = None  # 释放资源
            print("机械臂线程已停止")

    def on_press(self, key):
        global move_thread, key_press
        print(f"按下： {key}")
        key_press = key
        print("move_thread is alive? ", move_thread.is_alive())

        if move_thread.is_alive():
            print("机械臂线程为启动状态")

        # try:
        #     if key.char.lower() == 's':  # 处理大小写
        #         if move_thread and move_thread.is_alive():
        #             self.emergency_stop()
        #             # position_queue.put(None)  # 发送退出信号
        #             move_thread.join()  # 等待线程结束
        #             move_thread = None  # 释放资源
        #             print("机械臂线程已停止")
        # except AttributeError:
        #     pass  # 遇到 `Key` 对象（如 `space`）直接跳过

        if key == keyboard.Key.space:
            print("程序退出")
            if datarecord:
                print("保存数据")
                np.savetxt('realman/data_record/qpos_test_1.txt', qpos_record, fmt='%f')
                np.savetxt('realman/data_record/policy_reference_test_1.txt', policy_reference_record, fmt='%f')
                save_videos(image_record, 0.05, video_path='realman/data_record/video_test_1.mp4')

                data_dict = {
                    '/observations/qpos': [],
                    '/observations/qvel': [],
                    '/action': [],
                }
                for cam_name in camera_names:
                    data_dict[f'/observations/images/{cam_name}'] = []

                data_length = len(qpos_record)
                action_array = np.stack(policy_reference_record)  # [T, 14]
                qpos_array = np.stack(qpos_record)  # [T, 14]

                # 图像转换，假设 image_1/image_2 每帧为 [480, 640, 3]
                image_1_array = np.stack([frame['image_1'] for frame in image_record])  # [T, 480, 640, 3]
                image_2_array = np.stack([frame['image_2'] for frame in image_record])  # [T, 480, 640, 3]

                dataset_dir = 'realman/data_record'

                dataset_path = os.path.join(dataset_dir, f'episode_v_20.hdf5')

                with h5py.File(dataset_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                    root.attrs['sim'] = False

                    # 创建分组
                    obs = root.create_group('observations')
                    image_grp = obs.create_group('images')

                    # 创建并写入图像数据集
                    image_grp.create_dataset('image_1', data=image_1_array, chunks=(1, 480, 640, 3), dtype='uint8')
                    image_grp.create_dataset('image_2', data=image_2_array, chunks=(1, 480, 640, 3), dtype='uint8')

                    # 创建并写入状态和动作数据集
                    obs.create_dataset('qpos', data=qpos_array)
                    root.create_dataset('action', data=action_array)

                print(f'✅ 保存完成，共 {data_length} 步，文件：{dataset_path}')


            os._exit(0)

    def start_keyboard_listener(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener_thread = threading.Thread(target=listener.start, daemon=True)
        listener_thread.start()

    def emergency_stop(self):
        print("急停触发！")
        self.right_arm.Move_Stop_Cmd(block=False)
        self.left_arm.Move_Stop_Cmd(block=False)
        stop_flag.set()  # 终止线程
        print("急停命令已发送，线程已停止")

    def get_JRimages_thread(self):
        global image_1_thread, image_2_thread
        image_start_time = time.time()
        image_last_time = image_start_time
        while not stop_flag.is_set():
            # print("get JR images")
            ret, frame = self.cap.read()
            ret1, frame1 = self.cap1.read()
            if ret and ret1:
                with image_lock:
                    # print("获取图像成功")
                    image_1_thread = frame
                    image_2_thread = frame1
                    image_now_time = time.time()
                    print(
                        f"Get image Time: {image_now_time - image_start_time:.2f}, Get image FPS: {1 / (image_now_time - image_last_time):.2f} \n")
                    image_last_time = image_now_time
            else:
                print("无法获取图像")
            time.sleep(0.01)


    def start_camera_thread(self):
        global camera_thread
        if camera_thread is None or not camera_thread.is_alive():
            camera_thread = threading.Thread(target=self.get_JRimages_thread, daemon=True)
            camera_thread.start()
            print("相机线程启动")


    def init_JRcamera(self):
        print("open JRcamera")
        self.cap = cv2.VideoCapture(0)
        if not self.cap or not self.cap.isOpened():
            print("无法打开相机 2 ，请检查设备连接！")
            exit()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率为 30 FPS

        self.cap1 = cv2.VideoCapture(2)
        if not self.cap1 or not self.cap1.isOpened():
            print("无法打开相机 4 ，请检查设备连接！")
            exit()
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap1.set(cv2.CAP_PROP_FPS, 30)
        self.start_camera_thread()

    def init_L515(self):

        # # Get device product line for setting a supporting resolution
        # pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        # pipeline_profile = config.resolve(pipeline_wrapper)
        context = rs.context()
        devices = []
        for device in context.devices:
            if device.get_info(rs.camera_info.name):
                devices.append(device.get_info(rs.camera_info.serial_number))
        print(f"Connected devices: {devices}")

        for serial in devices:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            if serial == self.camera_serial_100:
                config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                print(serial)
            if serial == self.camera_serial_110:
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                print(serial)

            pipeline.start(config)

            self.pipelines[serial] = pipeline
            # self.pipelines.append(pipeline)

        # device = pipeline_profile.get_device()
        # device_product_line = str(device.get_info(rs.camera_info.product_line))
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # self.pipeline.start(config)


    def get_JRimages(self):
        # print("get JR images")
        image_dict = dict()
        camera_names = ['image_1', 'image_2']

        ret, frame = self.cap.read()
        ret1, frame1 = self.cap1.read()
        if ret:
            image_dict['image_1'] = frame  # 直接存入 numpy 数组
            image_dict['image_2'] = frame1
        else:
            print("无法获取图像")

        return image_dict


    def release_camera(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def get_images(self):
        print(f"enter get_images")
        image_dict = dict()
        camera_names = ['image_110', 'image_100']

        for serial, pipeline in self.pipelines.items():
            print(f"serial: ", serial)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            if serial == self.camera_serial_110:
                # color_image = cv2.resize(color_image, (640, 480))
                image_dict['image_110'] = color_image
            if serial == self.camera_serial_100:
                color_image = cv2.resize(color_image, (640, 480))
                image_dict['image_100'] = color_image
                print(f"find image_100")

        # frames = self.pipeline.wait_for_frames()
        # color_frame = frames.get_color_frame()
        # for cam_name in camera_names:
        #     # NOTE 将color_frame转换为numpy array，根据需要进行更改
        #     image_dict[cam_name] = np.asanyarray(color_frame.get_data())
        # color_image = np.asanyarray(color_frame.get_data())
        return image_dict

    def get_qpos(self):
        """
        Get joint positions.

        Returns:
            list: Joint positions.
        """
        ret, qpos_raw = self.robot.rm_get_joint_degree()
        if ret == 0:
            qpos = qpos_raw[:6]
        else:
            # 输出报错
            print("\nFailed to get joint positions\n")
            return None
        return qpos

    # NOTE 多半没法这么用，机械臂读取数据周期很长
    def get_observation(self):
        obs = collections.OrderedDict()
        _, left_joint = self.left_arm.Get_Joint_Degree()
        _, right_joint = self.right_arm.Get_Joint_Degree()

        obs['qpos'] = self.qpos_obs
        # obs['qvel'] = self.get_qvel()
        obs['images'] = self.get_JRimages()
        return obs

    def reset(self, v=1, r=0, connect=0, block=1):
        global move_thread, qpos_record, policy_reference_record
        print("reset robot")

        lefthand_init = [1]
        righthand_init = [1]

        self.start_thread()
        self.start_right_thread()
        self.start_read_thread()
        self.start_right_read_thread()
        time.sleep(0.5)  # 等待线程启动

        print("机械臂线程已启动")
        next_action = self.leftarm_init + [HAND_NORMALIZE(self.lefthand_init)] + self.rightarm_init + [HAND_NORMALIZE(self.righthand_init)]

        position_queue.put(next_action)
        right_position_queue.put(next_action)

        # 初始的qpos=leftarm_init+lefthand_init+rightarm_init+righthand_init
        qpos_init = np.concatenate((self.leftarm_init, [HAND_NORMALIZE(self.lefthand_init)],
                                    self.rightarm_init, [HAND_NORMALIZE(self.righthand_init)]))
        obs = collections.OrderedDict()

        # Note 通过realman的API获取关节数据周期过长，直接改成开环
        # _, left_qpos = self.left_arm.Get_Joint_Degree()
        # _, right_qpos = self.right_arm.Get_Joint_Degree()
        # self.qpos_obs = np.concatenate((left_qpos, [HAND_NORMALIZE(self.lefthand_init)],
        #                            right_qpos, [HAND_NORMALIZE(self.righthand_init)]))
        with read_lock:
            left_qpos = read_left_joint
            right_qpos = read_right_joint
        self.qpos_obs = np.concatenate((left_qpos, [HAND_NORMALIZE(self.lefthand_init)],
                                        right_qpos, [HAND_NORMALIZE(self.righthand_init)]))
        # self.qpos_obs = next_action

        image_dict = dict()
        camera_names = ['image_1', 'image_2']

        with image_lock:
            image_dict['image_1'] = image_1_thread.copy() if image_1_thread is not None else None
            image_dict['image_2'] = image_2_thread.copy() if image_2_thread is not None else None

        # image_dict = self.get_JRimages()
        obs['qpos'] = self.qpos_obs
        obs['action'] = self.qpos_obs
        obs['images'] = dict()
        # 确保获取的图像有效
        if image_dict is not None:
            obs['images']['image_1'] = image_dict['image_1']  # NumPy 格式
            obs['images']['image_2'] = image_dict['image_2']
            print("图像类型", type(obs['images']['image_1']))
        else:
            print("图像数据为空，obs['images'] 未填充")
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=obs)

    # 输入的action是7+1+7+1的关节，前一半是left，后一半是right
    def step(self, action, v=1, r=0, connect=0, block=1):
        global qpos_record, policy_reference_record, image_record
        """
        关节空间运动

        Args:
            joint (list): 各关节目标角度数组，单位：°
            v (int): 速度百分比系数，1~100
            r (int, optional): 交融半径百分比系数，0~100。
            connect (int): 轨迹连接标志
                - 0：立即规划并执行轨迹，不与后续轨迹连接。
                - 1：将当前轨迹与下一条轨迹一起规划，但不立即执行。阻塞模式下，即使发送成功也会立即返回。
            block (int): 阻塞设置
                - 多线程模式：
                    - 0：非阻塞模式，发送指令后立即返回。
                    - 1：阻塞模式，等待机械臂到达目标位置或规划失败后才返回。
                - 单线程模式：
                    - 0：非阻塞模式。
                    - 其他值：阻塞模式并设置超时时间，单位为秒。

        Returns:
            int: 函数执行的状态码。
            - 0: 成功。
            - 1: 控制器返回false，参数错误或机械臂状态发生错误。
            - -1: 数据发送失败，通信过程中出现问题。
            - -2: 数据接收失败，通信过程中出现问题或者控制器长久没有返回。
            - -3: 返回值解析失败，接收到的数据格式不正确或不完整。
            - -4: 当前到位设备校验失败，即当前到位设备不为关节。
            - -5: 单线程模式超时未接收到返回，请确保超时时间设置合理。
        """
        state_len = int(len(action) / 2)
        left_action = action[:state_len]
        right_action = action[state_len:]

        left_qpos = left_action[:7]
        right_qpos = right_action[:7]

        # next_step_action = left_qpos + [HAND_NORMALIZE(left_action[7])] + right_qpos + [HAND_NORMALIZE(right_action[7])]
        position_queue.put(action)
        right_position_queue.put(action)

        obs = collections.OrderedDict()
        # Note 通过realman的API获取关节数据周期过长，直接改成开环
        # _, left_qpos = self.left_arm.Get_Joint_Degree()
        # _, right_qpos = self.right_arm.Get_Joint_Degree()
        # # print("读取关节角度：", left_qpos, right_qpos)
        # self.qpos_obs = np.concatenate((left_qpos, [left_action[7]],
        #                            right_qpos, [right_action[7]]))
        with read_lock:
            left_qpos = read_left_joint
            right_qpos = read_right_joint
        # print("读取关节角度：", left_qpos, right_qpos)
        self.qpos_obs = np.concatenate((left_qpos, [left_action[7]], right_qpos, [right_action[7]]))
        # self.qpos_obs = action
        image_dict = dict()
        camera_names = ['image_1', 'image_2']
        with image_lock:
            image_dict['image_1'] = image_1_thread.copy() if image_1_thread is not None else None
            image_dict['image_2'] = image_2_thread.copy() if image_2_thread is not None else None

        # image_dict = self.get_JRimages()
        obs['qpos'] = self.qpos_obs
        obs['action'] = action
        obs['images'] = dict()
        obs['images']['image_1'] = image_dict['image_1']
        obs['images']['image_2'] = image_dict['image_2']

        if datarecord:

            policy_reference_record.append(action)
            qpos_record.append(self.qpos_obs)
            image_record.append(obs['images'])

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs)

    def get_reward(self):
        return 0

def make_rm_real_env():
    env = RealmanEnv()
    return env


def test_realenv():
    render_cams = ['image_1']  # Camera names
    env = make_rm_real_env()
    ts = env.reset()
    episode = [ts]
    # from visualize_episodes import load_hdf5
    # qpos_list, _, _, _ = load_hdf5('/home/juyiii/data/aloha/rmreal_pick', 'episode_7')
    # qpos = np.array(qpos_list)
    # qpos = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    # num_ts, num_dim = qpos.shape
    start_time = time.time()
    last_time = start_time
    for t in range(2000):
        leftarm_init = [-13.04299259185791, 25.120895385742188, 141.19898986816406, -93.86701202392578,
                             11.084973335266113, -28.841115951538086, 102.23300170898438]
        rightarm_init = [34.81800079345703, 11.810999870300293, 70.34600067138672, 98.68399810791016,
                              120.12000274658203, -72.572998046875, -51.494998931884766]

        action = [-13.04299259185791, 25.120895385742188, 141.19898986816406, -93.86701202392578,
                             11.084973335266113, -28.841115951538086, 102.23300170898438, 1, 34.81800079345703, 11.810999870300293, 70.34600067138672, 98.68399810791016,
                              120.12000274658203, -72.572998046875, -51.494998931884766, 1]


        # action = qpos[t, :]

        ts = env.step(action)
        episode.append(ts)
        image = ts.observation['images']['image_1']
        # cv2.imshow('image_1', image)

        # images = [ts.observation['images'][cam] for cam in render_cams]
        # # Combine images horizontally or vertically
        # combined_image = cv2.hconcat(images)  # Combine horizontally
        # cv2.imshow('image', combined_image)

        # 设定频率为20Hz (50ms per frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下'q'键退出
            break
        now_time = time.time()
        print(f"Time: {now_time - start_time:.2f}, FPS: {1 / (now_time - last_time):.2f}")
        last_time = now_time
        time.sleep(0.05)
        print(f"timestep: {t} \n")

    print("All timesteps have been run.")
    # 手臂急停
    env.emergency_stop()

    cv2.destroyAllWindows()
    # for pipeline in env.pipelines:
    #     pipeline.stop()

def load_policy(args):
        # from imitate_episodes_teleoperation import make_policy
        policy_config = {'lr': 1e-5,
                         'num_queries': args['chunk_size'],
                         'kl_weight': 10,
                         'hidden_dim': 512,
                         'dim_feedforward': 3200,
                         'lr_backbone': 1e-5,
                         'backbone': 'resnet18',
                         'enc_layers': 4,
                         'dec_layers': 7,
                         'nheads': 8,
                         'camera_names': camera_names,
                         'vq': False,
                         'vq_class': None,
                         'vq_dim': None,
                         'action_dim': 18,
                         'no_encoder': False,
                         'ckpt_dir': args['ckpt_dir'],
                         'policy_class': "ACT",
                         'seed': 0,
                         'num_steps': 20000,
                         }
        # MYBpolicy = make_policy("ACT", policy_config)
        from policy import ACTPolicy
        print("here")

        MYBpolicy = ACTPolicy(policy_config)
        ckpt_dir = args['ckpt_dir']
        ckpt_name = "policy_best.ckpt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        loading_status = MYBpolicy.deserialize(torch.load(ckpt_path))
        print(loading_status)
        MYBpolicy.cuda()
        MYBpolicy.eval()
        temporal_agg = args['temporal_agg']
        if temporal_agg:
            query_frequency = 1
        return MYBpolicy


def get_curr_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []  # 存储从每个摄像头获取的图像
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)  # 将图像列表堆叠成数组
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(
        0)  # 从 numpy 转为 torch，并归一化到0～1之间， 转移到GPU上， 添加一个新的维度

    if rand_crop_resize:
        print('rand crop resize is used!')
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)

    return curr_image


def query_policy(args):
        # global target_qpos, query_timestep, all_actions, ts, interpolated_trajectory, isinterpolated, interpolate_time

        env = make_rm_real_env()
        ts = env.reset()
        episode = [ts]

        temporal_agg = args['temporal_agg']
        qpos_numpy = []

        query_frequency = 1
        max_timesteps = 20000

        # start_time = time.time()
        # last_time = start_time
        start_time = time.time()
        last_time = start_time
        for t in range(max_timesteps):

            if t == 0:
                ckpt_dir = args['ckpt_dir']
                stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
                print(os.path.getsize(stats_path))  # 检查文件大小，如果返回 0 表示文件为空
                with open(stats_path, 'rb') as f:
                    stats = pickle.load(f)
                MYBpolicy = load_policy(args)

            with torch.inference_mode():

                pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']

                obs = ts.observation
                qpos_numpy = np.array(obs['qpos'])
                # print(qpos_numpy)
                pre_action.append(qpos_numpy)

                # curr_image = get_image(ts, camera_names, rand_crop_resize=False)
                curr_image = get_curr_image(ts, camera_names, rand_crop_resize=False)

                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

                if t % query_frequency == 0:

                    if t == 0:
                        # warm up
                        for _ in range(1):
                            MYBpolicy(qpos, curr_image)
                        print('network warm up done')
                    all_actions = MYBpolicy(qpos, curr_image)
                # print(f"all action(10)", all_actions[:10])
                if temporal_agg:
                    all_time_actions[[t], t:t + args['chunk_size']] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1) # 用来检查哪些动作在所有维度中都不为零，从而筛选出已填充的有效动作
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]

                raw_action = raw_action.squeeze(0).cpu().numpy()

                # 后处理 去归一化
                post_process = lambda a: a * stats['action_std'] + stats['action_mean']
                action = post_process(raw_action)
                target_qpos = action[:-2]

            ts = env.step(target_qpos)
            # print("policy target pos: ", target_qpos)
            episode.append(ts)
            now_time = time.time()
            print(f"Time: {now_time - start_time:.2f}, FPS: {1 / (now_time - last_time):.2f}")
            last_time = now_time
            print(f"Now is the {t} step")
            print("\n")
            time.sleep(0.1)

        # 输出英文已经跑完了所有的timesteps
        print("All timesteps have been run.")
        # 手臂急停
        env.emergency_stop()

        # target_action.append(target_qpos)
        # print(f"qpos_target: ", target_qpos)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_steps', action='store', type=int, help='num_steps', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--eval_every', action='store', type=int, default=500, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=500, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class')
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim')
    parser.add_argument('--no_encoder', action='store_true')


    query_policy(vars(parser.parse_args()))


    # test_realenv()





