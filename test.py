import torch
# print(torch.cuda.is_available())

# class MyClass:
#     @staticmethod
#     def my_static_method(x, y):
#         return x + y
#     def __call__(self, x):
#         print("Calling with",x)
#
# result = MyClass.my_static_method(1, 2)
# print(result)
# obj = MyClass()
# obj(110)

# 读取文件内容
# def read_data(filename):
#     with open(filename, 'r') as file:
#         # 初始化行号计数器
#         line_number = 0
#         # 逐行读取文件
#         while True:
#             line = file.readline()
#             if not line:  # 如果读取到文件末尾，结束循环
#                 break
#             line_number += 1  # 每读取一行，行号加1
#
#             # 去除行尾的换行符并分割行
#             numbers = line.strip().split()
#
#             # 检查是否有数字存在
#             if numbers:
#                 # 获取第一个数字，即列表中的第一个元素
#                 first_x = numbers[0]
#
#                 # 将字符串转换为整数或浮点数
#                 try:
#                     x_value = int(first_x)  # 尝试转换为整数
#                 except ValueError:
#                     x_value = float(first_x)  # 如果失败，转换为浮点数
#
#                 print(f"Line {line_number}: The first x value is: {x_value}")
#             else:
#                 print(f"Line {line_number}: No numbers found in line.")
#
#
# # 调用函数
# read_data('output_0.txt')

# def make_conditional_counter(increment_interval):
#     count = 0
#     result = 0
#
#     def conditional_counter():
#         nonlocal count, result
#         if count % increment_interval == 0:
#             result += 1
#             count = 0  # 重置计数器
#         count += 1
#         return result
#
#     return conditional_counter
#
# # 创建一个每三次调用增加一次的计数器
# my_conditional_counter = make_conditional_counter(3)
#
# # 使用计数器
# print(my_conditional_counter())  # 输出: 1 (因为初始调用时result为0)
# print(my_conditional_counter())  # 输出: 1
# print(my_conditional_counter())  # 输出: 2
# print(my_conditional_counter())  # 输出: 2
# print(my_conditional_counter())  # 输出: 2
# print(my_conditional_counter())  # 输出: 2
# print(my_conditional_counter())  # 输出: 2

# def increment_every_third_call():
#     if not hasattr(increment_every_third_call, 'counter'):
#         increment_every_third_call.counter = 0  # 初始化计数器
#         increment_every_third_call.call_count = 0  # 初始化调用次数计数器
#
#
#     # 检查是否达到增加数值的条件（每三次）
#     if increment_every_third_call.call_count >= 2:
#         increment_every_third_call.counter += 1  # 增加数值
#         increment_every_third_call.call_count = 0  # 重置调用次数计数器
#
#     increment_every_third_call.call_count += 1  # 每次调用函数时，调用计数加1
#
#     return increment_every_third_call.counter  # 返回当前的数值
#
# # 使用函数
# print(increment_every_third_call())  # 输出: 1 (第一次调用，counter从0变为1)
# print(increment_every_third_call())  # 输出: 1 (第二次调用，不满足条件，counter不变)
# episode_number = increment_every_third_call()
# print(f"episode_number: ", episode_number)
# print(f"episode_number: ", episode_number)
# print(f"episode_number: ", episode_number)
# print(increment_every_third_call())  # 输出: 2 (第三次调用，满足条件，counter增加1)
# print(increment_every_third_call())  # 输出: 2 (第四次调用，重置计数器，counter不变)
# print(increment_every_third_call())  # 输出: 2 (第四次调用，重置计数器，counter不变)
# print(increment_every_third_call())  # 输出: 2 (第四次调用，重置计数器，counter不变)
# print(increment_every_third_call())  # 输出: 2 (第四次调用，重置计数器，counter不变)

# def incrementing_function():
#     if not hasattr(incrementing_function, 'counter'):
#         incrementing_function.counter = -1  # 初始化计数器
#     incrementing_function.counter += 1
#     return incrementing_function.counter
#
# # 使用函数
# print(incrementing_function())  # 输出: 1
# print(incrementing_function())  # 输出: 2
#


# from pyquaternion import Quaternion
# q0 = Quaternion(axis=[1, 0, 1], angle=0.0)
# q1 = Quaternion(axis=[1, 1, 0], angle=2 * 3.141592 / 3)
# for q in Quaternion.intermediates(q0, q1, 8, include_endpoints=True):
#     v = q.rotate(q0)
#
#     print(v)

from dm_control import suite
from dm_control import viewer
import numpy as np
from dm_control.viewer.gui import glfw_gui
from dm_control import mujoco
import glfw
from dm_control.rl import control
from ee_sim_env import RMsimpletrajectoryEETask
from constants import DT
from scripted_policy import RMPolicy_simpletrajectory
from pyquaternion import Quaternion


xml_path = 'assets/models/rm_bimanual_ee.xml'
# MuJoCo data structures
model = mujoco.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mujoco.MjData(model)                     # MuJoCo data
cam = mujoco.MjvCamera()                        # Abstract camera
opt = mujoco.MjvOption()                        # visualization options
physics = mujoco.Physics.from_xml_path(xml_path)
task = RMsimpletrajectoryEETask(random=False)  # 这个task是之前的task，改掉
env = control.Environment(physics, task, time_limit=200, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
ts = env.reset()
episode = [ts]
policy = RMPolicy_simpletrajectory(False)
# 创建窗口实例
window = glfw_gui.GlfwWindow(width=640, height=480, title="My Title")

# 相机参数
# 根据相机的初始位置设定 camera_distance, camera_yaw 和 camera_pitch
initial_pos = physics.named.model.cam_pos['left_pillar']
camera_distance = np.linalg.norm(initial_pos)  # 计算初始相机距离
camera_yaw = np.degrees(np.arctan2(initial_pos[1], initial_pos[0]))  # 计算初始相机偏航角度
camera_pitch = np.degrees(np.arcsin(initial_pos[2] / camera_distance))  # 计算初始相机俯仰角度
is_dragging = False  # 拖拽状态
last_mouse_pos = None


# 鼠标滚轮缩放回调函数
def scroll_callback(window, x_offset, y_offset):
    global camera_distance
    camera_distance = max(0.5, camera_distance - y_offset * 0.1)  # 限制缩放最小距离

# 鼠标按键回调函数
def mouse_button_callback(window, button, action, mods):
    global is_dragging, last_mouse_pos
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            is_dragging = True
            last_mouse_pos = glfw.get_cursor_pos(window)  # 记录按下时的位置
        elif action == glfw.RELEASE:
            is_dragging = False

# 鼠标移动回调函数，用于处理视角旋转
def cursor_pos_callback(window, xpos, ypos):
    global last_mouse_pos, camera_pitch, camera_yaw
    if is_dragging and last_mouse_pos:
        # 计算鼠标移动的差值
        dx = xpos - last_mouse_pos[0]
        dy = ypos - last_mouse_pos[1]

        # 更新相机俯仰和偏航角度
        camera_yaw += dx * 0.1
        camera_pitch -= dy * 0.1
        camera_pitch = np.clip(camera_pitch, -89, 89)  # 限制俯仰角范围

        # 更新记录的最后鼠标位置
        last_mouse_pos = (xpos, ypos)

# 将鼠标事件绑定到窗口
glfw.set_scroll_callback(window._context.window, scroll_callback)
glfw.set_mouse_button_callback(window._context.window, mouse_button_callback)
glfw.set_cursor_pos_callback(window._context.window, cursor_pos_callback)

class Teleoperation_Policy:
    def __init__(self,env, window, move_speed=0.001, rotate_speed=1):
        self.env = env
        self.window = window
        self.move_speed = move_speed
        self.rotate_speed = rotate_speed
        self.mocap_left_xpos = np.array(env._physics.named.data.xpos['mocap_left'])
        self.mocap_left_quat = Quaternion(env._physics.named.data.mocap_quat['mocap_left'])
        self.mocap_right_xpos = np.array(env._physics.named.data.xpos['mocap_right'])
        self.mocap_right_quat = Quaternion(env._physics.named.data.mocap_quat['mocap_right'])

        # print(f"initial_quat: ", self.mocap_left_quat)

    def update_mocap_position(self, direction):
        # 更新mocap的位置信息
        if direction == 'x+':
            self.mocap_left_xpos[0] += self.move_speed
        elif direction == 'x-':
            self.mocap_left_xpos[0] -= self.move_speed
        elif direction == 'y+':
            self.mocap_left_xpos[1] += self.move_speed
        elif direction == 'y-':
            self.mocap_left_xpos[1] -= self.move_speed
        elif direction == 'z+':
            self.mocap_left_xpos[2] += self.move_speed
        elif direction == 'z-':
            self.mocap_left_xpos[2] -= self.move_speed
        elif direction == '0':
            self.mocap_left_xpos = np.array(env._physics.named.data.xpos['mocap_left'])

    def update_mocap_quat(self, direction):
        #  更新mocap姿态信息
        rotate_x_quat_positive = Quaternion(axis=[1.0, 0.0, 0.0], degrees=self.rotate_speed)
        rotate_x_quat_negative = Quaternion(axis=[1.0, 0.0, 0.0], degrees=-self.rotate_speed)
        rotate_y_quat_positive = Quaternion(axis=[0.0, 1.0, 0.0], degrees=self.rotate_speed)
        rotate_y_quat_negative = Quaternion(axis=[0.0, 1.0, 0.0], degrees=-self.rotate_speed)
        rotate_z_quat_positive = Quaternion(axis=[0.0, 0.0, 1.0], degrees=self.rotate_speed)
        rotate_z_quat_negative = Quaternion(axis=[0.0, 0.0, 1.0], degrees=-self.rotate_speed)
        if direction == 'x+':
            self.mocap_left_quat = self.mocap_left_quat * rotate_x_quat_positive
        elif direction == 'x-':
            self.mocap_left_quat = self.mocap_left_quat * rotate_x_quat_negative
        elif direction == 'y+':
            self.mocap_left_quat = self.mocap_left_quat * rotate_y_quat_positive
        elif direction == 'y-':
            self.mocap_left_quat = self.mocap_left_quat * rotate_y_quat_negative
        elif direction == 'z+':
            self.mocap_left_quat = self.mocap_left_quat * rotate_z_quat_positive
        elif direction == 'z-':
            self.mocap_left_quat = self.mocap_left_quat * rotate_z_quat_negative
        elif direction == '0':
            self.mocap_left_quat = Quaternion(env._physics.named.data.mocap_quat['mocap_left'])

    def handle_keyboard_pos_input(self, window):
        # 获取按键输入
        glfw_window = window._context.window  # 取出真实的 GLFW 窗口实例
        if glfw.get_key(glfw_window, glfw.KEY_W) == glfw.PRESS:
            self.update_mocap_position('y+')
            print(f"press: w \n")
        elif glfw.get_key(glfw_window, glfw.KEY_S) == glfw.PRESS:
            self.update_mocap_position('y-')
            print(f"press: s \n")
        elif glfw.get_key(glfw_window, glfw.KEY_A) == glfw.PRESS:
            self.update_mocap_position('x-')
            print(f"press: a \n")
        elif glfw.get_key(glfw_window, glfw.KEY_D) == glfw.PRESS:
            self.update_mocap_position('x+')
            print(f"press: d \n")
        elif glfw.get_key(glfw_window, glfw.KEY_Q) == glfw.PRESS:
            self.update_mocap_position('z-')
            print(f"press: q \n")
        elif glfw.get_key(glfw_window, glfw.KEY_E) == glfw.PRESS:
            self.update_mocap_position('z+')
            print(f"press: e \n")
        else:
            self.update_mocap_position('0')

    def handle_keyboard_quat_input(self, window):
        # 获取按键输入
        glfw_window = window._context.window  # 取出真实的 GLFW 窗口实例
        if glfw.get_key(glfw_window, glfw.KEY_J) == glfw.PRESS:
            self.update_mocap_quat('y+')
            print(f"press: J \n")
        elif glfw.get_key(glfw_window, glfw.KEY_L) == glfw.PRESS:
            self.update_mocap_quat('y-')
            print(f"press: L \n")
        elif glfw.get_key(glfw_window, glfw.KEY_K) == glfw.PRESS:
            self.update_mocap_quat('x-')
            print(f"press: K \n")
        elif glfw.get_key(glfw_window, glfw.KEY_I) == glfw.PRESS:
            self.update_mocap_quat('x+')
            print(f"press: I \n")
        elif glfw.get_key(glfw_window, glfw.KEY_U) == glfw.PRESS:
            self.update_mocap_quat('z-')
            print(f"press: U \n")
        elif glfw.get_key(glfw_window, glfw.KEY_O) == glfw.PRESS:
            self.update_mocap_quat('z+')
            print(f"press: O \n")
        else:
            self.update_mocap_quat('0')

    def __call__(self, window):
        self.handle_keyboard_pos_input(window)
        self.handle_keyboard_quat_input(window)
        # left_quat = env._physics.named.data.xquat['mocap_left']
        left_gripper = 1
        right_gripper = 1
        action_left = np.concatenate([self.mocap_left_xpos, self.mocap_left_quat.elements, [left_gripper]])
        action_right = np.concatenate([self.mocap_right_xpos, self.mocap_right_quat.elements, [right_gripper]])
        # print(f"action: ", action_left)
        return np.concatenate([action_left, action_right])

    # mocap_left_xpos = env._physics.named.data.xpos['mocap_left']
    # print(f"mocap_left_xpos: ", mocap_left_xpos)

# 定义渲染函数
def render_func():
    global camera_distance, camera_pitch, camera_yaw, ts


    policy = Teleoperation_Policy(env,window)
    action = policy(window)
    ts = env.step(action)

    episode.append(ts)
    # # 更新模拟状态
    # physics.step()

    # 相机位置信息 (根据拖拽和滚轮调整相机位置)
    physics.named.model.cam_pos['left_pillar'][0] = camera_distance * np.cos(np.radians(camera_yaw)) * np.cos(
        np.radians(camera_pitch))
    physics.named.model.cam_pos['left_pillar'][1] = camera_distance * np.sin(np.radians(camera_yaw)) * np.cos(
        np.radians(camera_pitch))
    physics.named.model.cam_pos['left_pillar'][2] = camera_distance * np.sin(np.radians(camera_pitch))

    # 渲染图像 (分辨率: width x height)
    width, height = window.shape
    camera = physics.render(camera_id=0, width=width, height=height)

    # 返回渲染的图像，转换为 np.uint8 格式的 3D 数组
    return np.array(camera, dtype=np.uint8).reshape(height, width, 3)

# 运行窗口事件循环并展示图像
window.event_loop(render_func)
# env = suite.load(domain_name="humanoid", task_name="stand")
# action_spec = env.action_spec()
#
# # Define a uniform random policy.
# def random_policy(time_step):
#   del time_step  # Unused.
#   return np.random.uniform(low=action_spec.minimum,
#                            high=action_spec.maximum,
#                            size=action_spec.shape)
#
# # Launch the viewer application.
# viewer.launch(env, policy=random_policy)
