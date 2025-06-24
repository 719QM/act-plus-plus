from realman.robotic_arm_package.robotic_arm import *
import numpy as np
import argparse
import yaml
import threading
from threading import Barrier
from pynput import keyboard
from interpolator import SmoothInterpolator
from RM_real_constants import HAND_UNNORMALIZE, HAND_NORMALIZE
import pyrealsense2 as rs
import cv2
import time
import queue
import dm_env
import collections
from visualize_episodes import save_videos
import h5py
from collections import defaultdict

import os
# os.environ["DISPLAY"] = ":0"

episode_num = 39

yaml_file_path = f'CLAWAR/real_experiment.yaml'
camera_names = ['camera_rgb']


class Realman_Dual_react:
    def __init__(self):
        os.environ["DISPLAY"] = ":0"
        left_ip = "192.168.1.19"
        right_ip = "192.168.1.18"
        self.left_arm = Arm(RM75, left_ip)
        self.right_arm = Arm(RM75, right_ip)
        self.barrier = Barrier(2)
        self.hand_barrier = Barrier(2)
        self.stop_flag = threading.Event()  # 控制线程停止
        self.key_press = None
        self.move_left_thread = None
        self.move_right_thread = None
        self.left_interpolator = SmoothInterpolator(dim=7, policy_freq=20, control_freq=100)
        self.right_interpolator = SmoothInterpolator(dim=7, policy_freq=20, control_freq=100)
        self.right_hand_control_thread = None
        self.left_hand_control_thread = None
        self.camera_serial_110 = 'f1420921'
        self.camera_serial_100 = 'f1381658'
        self.pipelines = {}
        self.listener_thread = None
        self.realsense_thread = None
        self.JRcamera_thread = None
        self.step_num = 0
        self.max_step = self.count_steps_in_yaml(yaml_file_path)
        print(f"一共{self.max_step}steps")
        self.position_queue = queue.Queue()  # 存储目标位置
        self.right_position_queue = queue.Queue()  # 存储目标位置
        self.left_hand_position_queue = queue.Queue()  # 存储目标位置
        self.right_hand_position_queue = queue.Queue()  # 存储目标位置
        self.left_point = []
        self.right_point = []
        self.left_hand_angle = []
        self.right_hand_angle = []
        self.policy_reference_record = []
        self.qpos_record = []
        self.image_record = []
        self.image_lock = threading.Lock()
        self.JRimage_lock = threading.Lock()
        self.image_rgb = []
        self.image_depth = []
        self.first_interp_done = False
        self.qpos_obs = []
        self.fps_display_thread = None
        self.fps_dict = defaultdict(str)
        self.fps_lock = threading.Lock()
        self.left_camera = []
        self.right_camera = []

    def count_steps_in_yaml(self, filepath):
        """
        读取 YAML 文件并返回 step 的数量（以 step_ 开头的 key 数量）
        """
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return 0

        step_keys = [key for key in data.keys() if key.startswith('step_')]
        return len(step_keys)

    # 判断是否轨迹是否走完
    def is_trajectory_finished(self):
        # 当已经走到了关键点的最后一步，即self.step_num >= self.max_step，且走完了最后一步的所有插值点，则返回True
        if self.step_num > self.max_step and self.left_interpolator.current_index >= len(self.left_interpolator.trajectory) \
                and self.right_interpolator.current_index >= len(self.right_interpolator.trajectory):
            return True
        else:
            return False

    def load_key_points(self, filepath, step_id):
        step_key = f"step_{step_id}"

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        if step_key not in data:
            raise ValueError(f"Step {step_id} not found in the file.")

        left_qpos = data[step_key]['left_qpos']
        right_qpos = data[step_key]['right_qpos']
        left_hand = data[step_key]['left_hand']
        right_hand = data[step_key]['right_hand']

        print("Left Arm Joint Positions:", left_qpos)
        print("Right Arm Joint Positions:", right_qpos)
        print("Left Hand Position:", left_hand)
        print("Right Hand Position:", right_hand)

        return left_qpos, right_qpos, left_hand, right_hand

    def move_left_arm(self):
        print("move_left_arm")
        move_start_time = time.time()
        move_last_time = move_start_time
        last_action = None
        last_target = None
        # 当没有急停，且走完最后一步的所有轨迹时才会退出
        while not self.stop_flag.is_set() and not self.is_trajectory_finished():
            self.barrier.wait()  # 等右手准备好
            if self.key_press is not None:
                if self.key_press.char.lower() == 's':  # 处理大小写
                    if self.move_left_thread and self.move_left_thread.is_alive():
                        self.emergency_stop()
                        self.stop_flag.set()  # 触发停止标志
                        print(f"stop_flag:{self.stop_flag}, 机械臂线程已停止")
            if not self.stop_flag.is_set():
                # ✅ 只有当左右手臂当前轨迹都走完时，才尝试读取新的目标
                try:
                    new_action = self.position_queue.get_nowait()
                    # new_action = position_queue.get_nowait()
                    current_target = new_action[:7]
                    # 判断目标是否发生变化（注意使用 np.allclose 防止浮点误差）
                    if (last_target is None) or (not np.allclose(current_target, last_target)):
                        print("进行插值")
                        self.left_interpolator.update_target(current_target)
                        last_target = current_target
                        print("左臂插值轨迹：", self.left_interpolator.trajectory)
                    last_action = new_action

                except queue.Empty:
                    pass  # 没有新目标也没关系，继续插值

                self.left_point = self.left_interpolator.get_next_point()
                # 打印left_point的大小
                # print("left_point size: ", len(self.left_point))
                # print(f"move left arm next step: {self.left_point}, hand: {last_action[7]}")

                self.left_arm.Movej_CANFD(joint=self.left_point[:7], follow=False, expand=0)


                move_now_time = time.time()
                with self.fps_lock:
                    self.fps_dict["Left Arm FPS:"] = f"{1 / (move_now_time - move_last_time):.2f} FPS"
                # print(f"Move Arm Time: {move_now_time - move_start_time:.2f}, Move Arm FPS: {1 / (move_now_time - move_last_time):.2f} \n")
                move_last_time = move_now_time
            time.sleep(0.008)

    def start_move_left_thread(self):
        self.move_left_thread = threading.Thread(target=self.move_left_arm, daemon=True)
        self.move_left_thread.start()
        print("move_thread is alive? ", self.move_left_thread.is_alive())

    def move_right_arm(self):
        print("move right arm")
        move_start_time = time.time()
        move_last_time = move_start_time
        last_action = None
        last_target = None
        while not self.stop_flag.is_set() and not self.is_trajectory_finished():
            self.barrier.wait()  # 等左手准备好
            if self.key_press is not None:
                if self.key_press.char.lower() == 's':  # 处理大小写
                    if self.move_right_thread and self.move_right_thread.is_alive():
                        self.emergency_stop()
                        self.stop_flag.set()  # 触发停止标志
                        print(f"stop_flag:{self.stop_flag}, 机械臂线程已停止")
            if not self.stop_flag.is_set():
                try:
                    new_action = self.right_position_queue.get_nowait()
                    current_target = new_action[8:15]
                    # 判断目标是否发生变化（注意使用 np.allclose 防止浮点误差）
                    if (last_target is None) or (not np.allclose(current_target, last_target)):
                        print("进行插值")
                        self.right_interpolator.update_target(current_target)
                        last_target = current_target
                        print("右臂插值轨迹：", self.right_interpolator.trajectory)
                    last_action = new_action
                except queue.Empty:
                    pass  # 没有新目标也没关系，继续插值

                self.right_point = self.right_interpolator.get_next_point()
                # print("现在走到了插值轨迹的第", self.right_interpolator.current_index, "步")
                # print("move right arm next step: ", point)

                self.right_arm.Movej_CANFD(joint=self.right_point[:7], follow=False, expand=0)

            move_now_time = time.time()
            with self.fps_lock:
                self.fps_dict["Right Arm FPS:"] = f"{1 / (move_now_time - move_last_time):.2f} FPS"
            # print(f"Move Right Arm Time: {move_now_time - move_start_time:.2f}, Move Right Arm FPS: {1 / (move_now_time - move_last_time):.2f} \n")
            move_last_time = move_now_time
            time.sleep(0.008)

    def start_move_right_thread(self):
        self.move_right_thread = threading.Thread(target=self.move_right_arm, daemon=True)
        self.move_right_thread.start()
        print("move_thread is alive? ", self.move_right_thread.is_alive())

    def on_press(self, key):
        print(f"press： {key}")
        self.key_press = key
        # print("move_thread is alive? ", self.move_left_thread.is_alive())

    def start_keyboard_listener(self):
        def _listen():
            with keyboard.Listener(on_press=self.on_press) as listener:
                listener.join()

        self.listener_thread = threading.Thread(target=_listen, daemon=True)
        self.listener_thread.start()

    def emergency_stop(self):
        print("急停触发！")
        self.right_arm.Move_Stop_Cmd(block=False)
        self.left_arm.Move_Stop_Cmd(block=False)
        self.stop_flag.set()  # 终止线程
        print("急停命令已发送，线程已停止")

    def right_hand(self):
        # global key_press, right_hand_control_thread, stop_flag
        print("move right hand")
        move_start_time = time.time()
        move_last_time = move_start_time
        while not self.stop_flag.is_set():
            self.hand_barrier.wait()
            if self.key_press is not None:
                if self.key_press.char.lower() == 's':
                    if self.right_hand_control_thread and self.right_hand_control_thread.is_alive():
                        self.emergency_stop()
                        self.stop_flag.set()
                        print(f"stop_flag:{self.stop_flag}, 机械臂线程已停止")
            if not self.stop_flag.is_set():
                try:
                    new_action = self.right_hand_position_queue.get_nowait()
                    self.right_hand_angle = [int(a) for a in HAND_UNNORMALIZE(new_action[15])]
                    # self.right_arm.Set_Hand_Angle(self.right_hand_angle, block=0)
                except queue.Empty:
                    pass
            move_now_time = time.time()
            with self.fps_lock:
                self.fps_dict["Right Hand FPS:"] = f"{1 / (move_now_time - move_last_time):.2f} FPS"
            # print(
            #     f"Move Right hand Time: {move_now_time - move_start_time:.2f}, Move Right hand FPS: {1 / (move_now_time - move_last_time):.2f} \n")
            move_last_time = move_now_time
            time.sleep(0.008)


    def left_hand(self):
        # global key_press, left_hand_control_thread, stop_flag
        print("move left hand")
        move_start_time = time.time()
        move_last_time = move_start_time
        while not self.stop_flag.is_set():
            self.hand_barrier.wait()
            if self.key_press is not None:
                if self.key_press.char.lower() == 's':
                    if self.left_hand_control_thread and self.left_hand_control_thread.is_alive():
                        self.emergency_stop()
                        self.stop_flag.set()
                        print(f"stop_flag:{self.stop_flag}, 机械臂线程已停止")
            if not self.stop_flag.is_set():
                try:
                    new_action = self.left_hand_position_queue.get_nowait()
                    self.left_hand_angle = [int(a) for a in HAND_UNNORMALIZE(new_action[7])]
                    print("左手角度：", self.left_hand_angle)
                    self.left_arm.Set_Hand_Angle(self.left_hand_angle, block=0)
                except queue.Empty:
                    pass
            move_now_time = time.time()
            with self.fps_lock:
                self.fps_dict["Left Hand FPS:"] = f"{1 / (move_now_time - move_last_time):.2f} FPS"
            # print(
            #     f"Move Left hand Time: {move_now_time - move_start_time:.2f}, Move Left hand FPS: {1 / (move_now_time - move_last_time):.2f} \n")
            move_last_time = move_now_time
            time.sleep(0.008)

    def start_right_hand_thread(self):
        if self.right_hand_control_thread is None or not self.right_hand_control_thread.is_alive():  # 防止重复启动
            self.stop_flag.clear()  # 复位停止标志
            self.right_hand_control_thread = threading.Thread(target=self.right_hand, daemon=True)  # 设置守护线程
            self.right_hand_control_thread.start()
            print("right_hand_control_thread is alive? ", self.right_hand_control_thread.is_alive())
            print("右手线程已启动")

    def start_left_hand_thread(self):
        if self.left_hand_control_thread is None or not self.left_hand_control_thread.is_alive():
            self.stop_flag.clear()
            self.left_hand_control_thread = threading.Thread(target=self.left_hand, daemon=True)
            self.left_hand_control_thread.start()
            print("left_hand_control_thread is alive? ", self.left_hand_control_thread.is_alive())
            print("左手线程已启动")

    def init_L515(self):
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
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            print(serial)

            # if serial == self.camera_serial_100:
            #     config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            #     print(serial)
            # if serial == self.camera_serial_110:
            #     config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            #     print(serial)

            pipeline.start(config)

            self.pipelines[serial] = pipeline

    def get_realsense_image(self):
        print(f"enter get_images")
        image_start_time = time.time()
        image_last_time = image_start_time
        image_dict = dict()
        camera_names = ['image_110', 'image_100']

        while not self.stop_flag.is_set():
            pipeline = list(self.pipelines.values())[0]
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            with self.image_lock:
                self.image_rgb = color_image

            image_now_time = time.time()
            with self.fps_lock:
                self.fps_dict["Realsense FPS:"] = f"{1 / (image_now_time - image_last_time):.2f} FPS"
            # print(f"Get Image Time: {image_now_time - image_start_time:.2f}, Get Image FPS: {1 / (image_now_time - image_last_time):.2f} \n")
            image_last_time = image_now_time

    def start_realsense(self):
        if self.realsense_thread is None or not self.realsense_thread.is_alive():
            self.stop_flag.clear()
            self.realsense_thread = threading.Thread(target=self.get_realsense_image, daemon=True)
            self.realsense_thread.start()
            print("realsense_thread is alive? ", self.realsense_thread.is_alive())
            print("realsense相机线程已启动")

    def display_fps(self):
        # while True:
        img = np.zeros((300, 400, 3), dtype=np.uint8)  # 黑底画布
        with self.fps_lock:
            for idx, (k, v) in enumerate(self.fps_dict.items()):
                text = f"{k}: {v}"
                cv2.putText(img, text, (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
        cv2.imshow("Thread FPS Monitor", img)
        # time.sleep(0.1)  # 每秒刷新 10 次
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break

    def get_JRimages_thread(self):
        # global image_1_thread, image_2_thread
        image_start_time = time.time()
        image_last_time = image_start_time
        while not self.stop_flag.is_set():
            # print("get JR images")
            ret, frame = self.cap.read()
            ret1, frame1 = self.cap1.read()
            if ret and ret1:
                with self.JRimage_lock:
                    # print("获取图像成功")
                    self.left_camera = frame
                    self.right_camera = frame1
                    image_now_time = time.time()
                    # print(
                    #     f"Get image Time: {image_now_time - image_start_time:.2f}, Get image FPS: {1 / (image_now_time - image_last_time):.2f} \n")
                    image_last_time = image_now_time
            else:
                print("无法获取图像")
            time.sleep(0.01)


    def start_JRcamera_thread(self):
        # global camera_thread
        if self.JRcamera_thread is None or not self.JRcamera_thread.is_alive():
            self.JRcamera_thread = threading.Thread(target=self.get_JRimages_thread, daemon=True)
            self.JRcamera_thread.start()
            print("相机线程启动")


    def init_JRcamera(self):
        print("open JRcamera")
        self.cap = cv2.VideoCapture(4)
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
        self.start_JRcamera_thread()

    def reset(self):
        print("reset robot")


        lefthand_init = [1]
        righthand_init = [1]

        # self.start_read_thread()
        # self.start_right_read_thread()
        # time.sleep(0.5)  # 等待线程启动
        # print("机械臂读取线程已启动")

        # _, read_left_joint = self.left_arm.Get_Joint_Degree()
        # _, read_right_joint = self.right_arm.Get_Joint_Degree()
        #
        # self.left_interpolator.last_pos = read_left_joint
        # self.right_interpolator.last_pos = read_right_joint
        #
        # self.left_interpolator.last_pos = [-1, 5, 3, -3, 4, 6, 6]
        # self.right_interpolator.last_pos = [1, -5, -3, 3, -4, -6, -6]
        #
        # # next_action = self.leftarm_init + [HAND_NORMALIZE(self.lefthand_init)] + self.rightarm_init + [HAND_NORMALIZE(self.righthand_init)]
        #
        # left_qpos, right_qpos, left_hand, right_hand = self.load_key_points(yaml_file_path, self.step_num)
        # self.step_num += 1
        # next_action = left_qpos + left_hand + right_qpos + right_hand
        # self.position_queue.put(next_action)
        # self.right_position_queue.put(next_action)
        # self.left_hand_position_queue.put(next_action)
        # self.right_hand_position_queue.put(next_action)
        # listener = keyboard.Listener(on_press=self.on_press)
        # listener.start()

        # self.start_keyboard_listener()
        self.start_move_left_thread()
        self.start_move_right_thread()
        self.start_right_hand_thread()
        self.start_left_hand_thread()
        self.init_L515()
        self.start_realsense()
        self.init_JRcamera()
        self.start_JRcamera_thread()

        time.sleep(0.5)  # 等待线程启动

        print("机械臂线程已启动")


        # obs = collections.OrderedDict()

        # self.qpos_obs = next_action
        #
        # image_dict = dict()
        # camera_names = ['image_1', 'image_2']
        #
        # with self.image_lock:
        #     image_dict['image_rgb'] = self.image_rgb.copy() if self.image_rgb is not None else None
        #     # image_dict['image_2'] = image_2_thread.copy() if image_2_thread is not None else None
        #
        # # image_dict = self.get_JRimages()
        # obs['qpos'] = self.qpos_obs
        # obs['action'] = self.qpos_obs
        # obs['images'] = dict()
        # # 确保获取的图像有效
        # if image_dict is not None:
        #     obs['images']['image_rgb'] = image_dict['image_rgb']  # NumPy 格式
        #     # obs['images']['image_2'] = image_dict['image_2']
        #     print("图像类型", type(obs['images']['image_rgb']))
        # else:
        #     print("图像数据为空，obs['images'] 未填充")
        # return dm_env.TimeStep(
        #     step_type=dm_env.StepType.FIRST,
        #     reward=self.get_reward(),
        #     discount=None,
        #     observation=obs)

    # 输入的action是7+1+7+1的关节，前一半是left，后一半是right
    def step(self):
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
        # self.display_fps()
        # print("fps没卡进程")
        if not self.first_interp_done:
            print(f"------------------第{self.step_num}步-----------------------")

            _, read_left_joint = self.left_arm.Get_Joint_Degree()
            _, read_right_joint = self.right_arm.Get_Joint_Degree()

            self.left_interpolator.last_pos = read_left_joint
            self.right_interpolator.last_pos = read_right_joint

            # self.left_interpolator.last_pos = [-1, 5, 3, -3, 4, 6, 6]
            # self.right_interpolator.last_pos = [1, -5, -3, 3, -4, -6, -6]

            # next_action = self.leftarm_init + [HAND_NORMALIZE(self.lefthand_init)] + self.rightarm_init + [HAND_NORMALIZE(self.righthand_init)]

            left_qpos, right_qpos, left_hand, right_hand = self.load_key_points(yaml_file_path, self.step_num)
            self.step_num += 1
            next_action = left_qpos + left_hand + right_qpos + right_hand
            self.position_queue.put(next_action)
            self.right_position_queue.put(next_action)
            self.left_hand_position_queue.put(next_action)
            self.right_hand_position_queue.put(next_action)

            self.first_interp_done = True
            time.sleep(1)

        # 只有当左右手臂当前轨迹都走完时，才尝试读取新的目标
        elif self.step_num < self.max_step and self.left_interpolator.current_index >= len(self.left_interpolator.trajectory) \
                and self.right_interpolator.current_index >= len(self.right_interpolator.trajectory):
                left_action, right_action, left_hand, right_hand = self.load_key_points(yaml_file_path, self.step_num)
                print(f"------------------第{self.step_num}步-----------------------")
                self.step_num += 1

                next_action = left_action + left_hand + right_action + right_hand
                self.position_queue.put(next_action)
                self.right_position_queue.put(next_action)
                self.left_hand_position_queue.put(next_action)
                self.right_hand_position_queue.put(next_action)

                self.first_interp_done = True
        elif self.step_num >= self.max_step and self.left_interpolator.current_index >= len(self.left_interpolator.trajectory) \
                and self.right_interpolator.current_index >= len(self.right_interpolator.trajectory):
            self.step_num += 1
            print("已经走完所有的轨迹")
            self.stop_flag.is_set()
            pass

        obs = collections.OrderedDict()

        # self.qpos_obs = self.left_point + HAND_NORMALIZE(self.left_hand_angle) + self.right_point + HAND_NORMALIZE(self.right_hand_angle)
        self.qpos_obs = np.concatenate([
            np.array(self.left_point),
            np.array([HAND_NORMALIZE(self.left_hand_angle)]),
            np.array(self.right_point),
            np.array([HAND_NORMALIZE(self.right_hand_angle)])
        ])
        # self.qpos_obs = action
        image_dict = dict()
        # camera_names = ['image_1', 'image_2']
        with self.image_lock:
            image_dict['image_rgb'] = self.image_rgb.copy() if self.image_rgb is not None else None
        with self.JRimage_lock:
            # image_dict['image_left'] = self.left_camera.copy() if self.left_camera is not None else None
            # image_dict['image_right'] = self.right_camera.copy() if self.right_camera is not None else None
            image_dict['image_left'] = np.array(self.left_camera.copy()) if self.left_camera is not None else None
            image_dict['image_right'] = np.array(self.right_camera.copy()) if self.right_camera is not None else None

        # image_dict = self.get_JRimages()
        # obs['qpos'] = self.qpos_obs
        obs['qpos'] = self.qpos_obs
        obs['action'] = self.qpos_obs
        obs['images'] = dict()
        obs['images']['image_rgb'] = image_dict['image_rgb']
        obs['images']['image_left'] = image_dict['image_left']
        obs['images']['image_right'] = image_dict['image_right']
        # obs['images']['image_2'] = image_dict['image_2']

        self.policy_reference_record.append(self.qpos_obs)
        self.qpos_record.append(self.qpos_obs)
        self.image_record.append(obs['images'])

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs)

    def get_reward(self):
        return 0


def make_rm_real_env():
    env = Realman_Dual_react()
    return env


if __name__ == '__main__':
    render_cams = ['image_rgb']  # Camera names
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
    t = 0
    while t < 20000 and not env.is_trajectory_finished():

        ts = env.step()
        episode.append(ts)
        image = ts.observation['images']['image_rgb']
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
        t += 1

    print("All timesteps have been run.")

    print("保存数据")
    np.savetxt(f'CLAWAR/qpos_test_{episode_num}', env.qpos_record, fmt='%f')
    np.savetxt(f'CLAWAR/policy_reference_test_{episode_num}', env.policy_reference_record, fmt='%f')
    save_videos(env.image_record, 0.05, video_path=f'CLAWAR/video_test_{episode_num}.mp4')

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/action': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    data_length = len(env.qpos_record)
    action_array = np.stack(env.policy_reference_record)  # [T, 14]
    qpos_array = np.stack(env.qpos_record)  # [T, 14]

    # 图像转换，假设 image_1/image_2 每帧为 [480, 640, 3]
    # 输出env.image_record中每个图像的步数、类型和大小
    # print(f"image_1: {len(env.image_record)}, {type(env.image_record[0]['image_rgb'])}")
    # print(f"image_2: {len(env.image_record)}, {type(env.image_record[0]['image_left'])}")
    # print(f"image_3: {len(env.image_record)}, {type(env.image_record[0]['image_right'])}")
    # for idx, frame in enumerate(env.image_record):
    #     img = frame['image_left']
    #     if img is None:
    #         print(f"Frame {idx}: image is None")
    #     else:
    #         print(f"Frame {idx}: shape = {img.shape}")

    image_1_array = np.stack([frame['image_rgb'] for frame in env.image_record])  # [T, 480, 640, 3]
    image_2_array = np.stack([frame['image_left'] for frame in env.image_record])  # [T, 480, 640, 3]
    image_3_array = np.stack([frame['image_right'] for frame in env.image_record])  # [T, 480, 640, 3]


    dataset_dir = 'collection'

    dataset_path = os.path.join(dataset_dir, f'episode_{episode_num}.hdf5')

    with h5py.File(dataset_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False

        # 创建分组
        obs = root.create_group('observations')
        image_grp = obs.create_group('images')

        # 创建并写入图像数据集
        image_grp.create_dataset('image_rgb', data=image_1_array, chunks=(1, 480, 640, 3), dtype='uint8')
        image_grp.create_dataset('image_left', data=image_2_array, chunks=(1, 480, 640, 3), dtype='uint8')
        image_grp.create_dataset('image_right', data=image_3_array, chunks=(1, 480, 640, 3), dtype='uint8')

        # 创建并写入状态和动作数据集
        obs.create_dataset('qpos', data=qpos_array)
        root.create_dataset('action', data=action_array)

    print(f'✅ 保存完成，共 {data_length} 步，文件：{dataset_path}')

    # 手臂急停
    env.emergency_stop()

    cv2.destroyAllWindows()
    # for pipeline in env.pipelines:
    #     pipeline.stop()

