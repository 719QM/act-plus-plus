import time
import numpy as np
import collections
import matplotlib.pyplot as plt
import dm_env
from pyquaternion import Quaternion
import pyrealsense2 as rs
import cv2
from Robotic_Arm.rm_robot_interface import *


import IPython
e = IPython.embed

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
        # 初始化机器人，初始化相机
        """
        Initialize and connect to the robotic arm.

        Args:
            ip (str): IP address of the robot arm.
            port (int): Port number.
            level (int, optional): Connection level. Defaults to 3.
            mode (int, optional): Thread mode (0: single, 1: dual, 2: triple). Defaults to 2.
        """
        left_ip = "192.168.1.18"
        right_ip = "192.168.1.19"
        port = 8080
        self.thread_mode = rm_thread_mode_e(mode)
        self.left_arm = RoboticArm(self.thread_mode)
        self.left_handle = self.left_arm.rm_create_robot_arm(left_ip, port, level)
        self.right_arm = RoboticArm(self.thread_mode)
        self.right_handle = self.right_arm.rm_create_robot_arm(right_ip, port, level)

        # self.robot = RoboticArm(self.thread_mode)
        # self.handle = self.robot.rm_create_robot_arm(ip, port, level)

        if self.left_handle.id == -1 or self.right_handle.id == -1:
            print("\nFailed to connect to the robot arm\n")
            exit(1)
        else:
            print(f"\nSuccessfully connected to the left arm: {self.left_handle.id} and the right arm: {self.right_handle.id}\n")

        # Initialize camera
        self.camera_serial_110 = 'f1381658'
        self.camera_serial_100 = 'f0233166'
        self.pipelines = {}
        self.init_L515()

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
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            if serial == self.camera_serial_110:
                config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

            pipeline.start(config)

            self.pipelines[serial] = pipeline
            # self.pipelines.append(pipeline)

        # device = pipeline_profile.get_device()
        # device_product_line = str(device.get_info(rs.camera_info.product_line))
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # self.pipeline.start(config)

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
                color_image = cv2.resize(color_image, (640, 480))
                image_dict['image_110'] = color_image
            if serial == self.camera_serial_100:
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
    # NOTE 角速度如何获取？

    @staticmethod
    def rm_set_hand_angle(hand_angle: list[int], handle) -> int:
        """
        设置灵巧手各自由度角度
        @details 设置灵巧手角度，灵巧手有6个自由度，从1~6分别为小拇指，无名指，中指，食指，大拇指弯曲，大拇指旋转
        Args:
            hand_angle (list[int]): 手指角度数组，范围：0~1000. 另外，-1代表该自由度不执行任何操作，保持当前状态

        Returns:
            int: 函数执行的状态码。
            - 0: 成功。
            - 1: 控制器返回false，参数错误或机械臂状态发生错误。
            - -1: 数据发送失败，通信过程中出现问题。
            - -2: 数据接收失败，通信过程中出现问题或者控制器长久没有返回。
            - -3: 返回值解析失败，接收到的数据格式不正确或不完整。
            - -4: 超时未返回
        """
        angle = (c_int * 6)(*hand_angle)
        tag = rm_set_hand_angle(handle, angle)
        return tag

    # NOTE 多半没法这么用，机械臂读取数据周期很长
    def get_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['qvel'] = self.get_qvel()
        obs['images'] = self.get_images()
        return obs

    def reset(self, v=20, r=0, connect=0, block=1):
        # 重置机械臂
        leftarm_init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        rightarm_init = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        lefthand_init = [999, 999, 999, 999, 999, 999]
        righthand_init = [999, 999, 999, 999, 999, 999]

        array_type = ctypes.c_float * len(leftarm_init)
        ctypes_array = array_type(*leftarm_init)
        ctypes_array_right = array_type(*rightarm_init)
        LP_c_float = ctypes.POINTER(ctypes.c_float)
        LP_c_float_right = ctypes.POINTER(ctypes.c_float)
        pointer_to_array = ctypes.cast(ctypes_array, LP_c_float)
        pointer_to_array_right = ctypes.cast(ctypes_array_right, LP_c_float_right)

        rm_movej(self.left_handle, pointer_to_array, v, r, connect, block)
        rm_movej(self.right_handle, pointer_to_array_right, v, r, connect, block)
        self.rm_set_hand_angle(lefthand_init, self.left_handle)
        self.rm_set_hand_angle(righthand_init, self.right_handle)
        # 初始的qpos=leftarm_init+lefthand_init+rightarm_init+righthand_init
        qpos_init = np.concatenate((leftarm_init, [1], rightarm_init, [1]))
        obs = collections.OrderedDict()
        obs['qpos'] =qpos_init
        # NOTE qvel?
        obs['images'] = self.get_images()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=obs)

    # 输入的action是7+1+7+1的关节，前一半是left，后一半是right
    def step(self, action, v=20, r=0, connect=0, block=1):
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

        array_type = ctypes.c_float * len(left_qpos)
        ctypes_array = array_type(*left_qpos)
        ctypes_array_right = array_type(*right_qpos)
        LP_c_float = ctypes.POINTER(ctypes.c_float)
        LP_c_float_right = ctypes.POINTER(ctypes.c_float)
        pointer_to_array = ctypes.cast(ctypes_array, LP_c_float)
        pointer_to_array_right = ctypes.cast(ctypes_array_right, LP_c_float_right)

        left_movetag = rm_movej(self.left_handle, pointer_to_array, v, r, connect, block)
        right_movetag = rm_movej(self.right_handle, pointer_to_array_right, v, r, connect, block)
        left_handtag = self.rm_set_hand_angle([left_action[7]], self.left_handle)
        right_handtag = self.rm_set_hand_angle([right_action[7]], self.right_handle)
        print(f"left_arm: ", left_movetag, "right_arm: ", right_movetag, "left_hand: ", left_handtag, "right_hand: ", right_handtag)
        obs = collections.OrderedDict()
        obs['qpos'] = action
        # NOTE qvel?
        obs['images'] = self.get_images()

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
    render_cams = ['image_100']  # Camera names
    env = make_rm_real_env()
    ts = env.reset()
    episode = [ts]
    from visualize_episodes import load_hdf5
    qpos_list, _, _, _ = load_hdf5('/home/juyiii/data/aloha/rmreal_pick', 'episode_7')
    qpos = np.array(qpos_list)
    num_ts, num_dim = qpos.shape
    start_time = time.time()
    last_time = start_time
    for t in range(num_ts):
        action = [0, 0, 0, 0, 0, 0, 90, 999, 0, 0, 0, 0, 0, 0, 0, 999]


        # action = qpos[t, :]

        ts = env.step(action)
        episode.append(ts)
        image = ts.observation['images']['image_100']
        cv2.imshow('image_100', image)

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

    cv2.destroyAllWindows()
    for pipeline in env.pipelines:
        pipeline.stop()

if __name__ == '__main__':
    test_realenv()





