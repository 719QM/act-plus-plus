import numpy as np
from dm_control.viewer.gui import glfw_gui
from dm_control.utils import inverse_kinematics as ik
from dm_control import mujoco
import glfw
from dm_control.rl import control
from ee_sim_env import RMsimpletrajectoryEETask
from constants import DT
from scripted_policy import RMPolicy_simpletrajectory
from pyquaternion import Quaternion
import time
import os
import h5py
import argparse

from constants import SIM_TASK_CONFIGS
camera_names = SIM_TASK_CONFIGS['sim_RM_simpletrajectory']['camera_names']
camera_top = []
camera_angle = []



xml_path = 'assets/models/rm_bimanual_ee.xml'
# MuJoCo data structures
model = mujoco.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mujoco.MjData(model)                     # MuJoCo data
cam = mujoco.MjvCamera()                        # Abstract camera
opt = mujoco.MjvOption()                        # visualization options
physics = mujoco.Physics.from_xml_path(xml_path)
task = RMsimpletrajectoryEETask(random=False)  # 这个task是之前的task，改掉
env = control.Environment(physics, task, time_limit=2000, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
ts = env.reset()
episode = [ts]
# policy = RMPolicy_simpletrajectory(False)
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
policy = None
teleoperation_qpos = []
num_episode = None

def ik_test():
    # 目标参数
    SITE_NAME = 'mocap_left_site1'  # Site名称，需在XML文件中定义
    # TARGET_POS = np.array([-0.46, 0.57, 0.4])  # 目标位置
    # 修改为mocap的位置
    TARGET_POS = np.array(physics.named.data.site_xpos[SITE_NAME])  # 目标位置
    TARGET_QUAT = None  # 目标方向 (四元数)，如果不需要可以设为None
    JOINT_NAMES = ['left_joint_1', 'left_joint_2', 'left_joint_3', 'left_joint_4', 'left_joint_5',
                   'left_joint_6']  # 需要控制的关节
    TOL = 1e-4  # 误差容限
    MAX_STEPS = 200  # 最大迭代次数

    result = ik.qpos_from_site_pose(
        physics=physics,
        site_name=SITE_NAME,
        target_pos=TARGET_POS,
        target_quat=TARGET_QUAT,
        joint_names=JOINT_NAMES,
        tol=TOL,
        max_steps=MAX_STEPS,
        inplace=False  # 设置为False，不直接修改physics中的qpos
    )
    # 检查求解结果
    if result.success:
        print("IK解算成功!")
        print("解算的关节角: ", result.qpos)
        print("误差范数: ", result.err_norm)

        # 将解算结果应用到模型中
        physics.data.qpos[:len(result.qpos)] = result.qpos
        physics.forward()  # 更新前向运动学

        # 验证目标位置
        current_pos = physics.named.data.site_xpos[SITE_NAME]
        print("当前末端位置: ", current_pos)
        print("目标末端位置: ", TARGET_POS)
    else:
        print("IK解算失败，尝试调整参数或模型初始状态！")


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
    def __init__(self, env, window, move_speed=0.001, rotate_speed=1):
        self.env = env
        self.window = window
        self.move_speed = move_speed
        self.rotate_speed = rotate_speed
        self.mocap_left_xpos = np.array(env._physics.named.data.xpos['mocap_left'])
        self.mocap_left_quat = Quaternion(env._physics.named.data.mocap_quat['mocap_left'])
        self.mocap_right_xpos = np.array(env._physics.named.data.xpos['mocap_right'])
        self.mocap_right_quat = Quaternion(env._physics.named.data.mocap_quat['mocap_right'])
        self.left_gripper = 1
        self.right_gripper = 1
        self.last_key_time = {}  # 记录每个按键的最后一次按下时间
        self.debounce_interval = 0.05  # 去抖动时间间隔，单位：秒

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

    def update_gripper(self, direction):
        #  更新夹抓的位置
        if direction == "1":
            self.left_gripper = 1
        elif direction == "0":
            self.left_gripper = 0

    def handle_keyboard_input(self, window):
        glfw_window = window._context.window  # 获取真实的 GLFW 窗口实例
        current_time = time.time()  # 获取当前时间戳

        # 定义所有需要检测的按键和对应的动作
        key_actions = {
            glfw.KEY_I: ('x+', self.update_mocap_quat),
            glfw.KEY_K: ('x-', self.update_mocap_quat),
            glfw.KEY_J: ('y+', self.update_mocap_quat),
            glfw.KEY_L: ('y-', self.update_mocap_quat),
            glfw.KEY_U: ('z-', self.update_mocap_quat),
            glfw.KEY_O: ('z+', self.update_mocap_quat),

            glfw.KEY_A: ('y+', self.update_mocap_position),
            glfw.KEY_D: ('y-', self.update_mocap_position),
            glfw.KEY_W: ('x+', self.update_mocap_position),
            glfw.KEY_S: ('x-', self.update_mocap_position),
            glfw.KEY_Q: ('z+', self.update_mocap_position),
            glfw.KEY_E: ('z-', self.update_mocap_position),

            glfw.KEY_1: ('1', self.update_gripper),
            glfw.KEY_0: ('0', self.update_gripper),
        }
        # 遍历所有按键
        for key, (action, func) in key_actions.items():
            if glfw.get_key(glfw_window, key) == glfw.PRESS:
                # 检查去抖动时间间隔
                last_time = self.last_key_time.get(key, 0)
                if current_time - last_time > self.debounce_interval:
                    func(action)  # 执行动作
                    print(f"press: {action} \n")
                    # ik_test()
                    teleoperation_qpos.append(self.get_qpos())
                    camera_top.append(physics.render(height=480, width=640, camera_id='top'))
                    camera_angle.append(physics.render(height=480, width=640, camera_id='angle'))

                    self.last_key_time[key] = current_time  # 更新按键的最后按下时间
                    break  # 防止多个按键同时触发

        # # 如果没有按下任何按键，更新到初始状态
        # if all(glfw.get_key(glfw_window, key) != glfw.PRESS for key in key_actions.keys()):
        #     self.update_mocap_quat('0')


    # def handle_keyboard_pos_input(self, window):
    #     # 获取按键输入
    #     glfw_window = window._context.window  # 取出真实的 GLFW 窗口实例
    #     if glfw.get_key(glfw_window, glfw.KEY_W) == glfw.PRESS:
    #         self.update_mocap_position('y+')
    #         print(f"press: w \n")
    #         teleoperation_qpos.append(self.get_qpos())
    #     elif glfw.get_key(glfw_window, glfw.KEY_S) == glfw.PRESS:
    #         self.update_mocap_position('y-')
    #         print(f"press: s \n")
    #         teleoperation_qpos.append(self.get_qpos())
    #     elif glfw.get_key(glfw_window, glfw.KEY_A) == glfw.PRESS:
    #         self.update_mocap_position('x-')
    #         print(f"press: a \n")
    #         teleoperation_qpos.append(self.get_qpos())
    #     elif glfw.get_key(glfw_window, glfw.KEY_D) == glfw.PRESS:
    #         self.update_mocap_position('x+')
    #         print(f"press: d \n")
    #         teleoperation_qpos.append(self.get_qpos())
    #     elif glfw.get_key(glfw_window, glfw.KEY_Q) == glfw.PRESS:
    #         self.update_mocap_position('z-')
    #         print(f"press: q \n")
    #         teleoperation_qpos.append(self.get_qpos())
    #     elif glfw.get_key(glfw_window, glfw.KEY_E) == glfw.PRESS:
    #         self.update_mocap_position('z+')
    #         print(f"press: e \n")
    #         teleoperation_qpos.append(self.get_qpos())
    #     else:
    #         self.update_mocap_position('0')
    #
    # def handle_keyboard_quat_input(self, window):
    #     # 获取按键输入
    #     glfw_window = window._context.window  # 取出真实的 GLFW 窗口实例
    #     if glfw.get_key(glfw_window, glfw.KEY_J) == glfw.PRESS:
    #         self.update_mocap_quat('y+')
    #         print(f"press: J \n")
    #         teleoperation_qpos.append(self.get_qpos())
    #     elif glfw.get_key(glfw_window, glfw.KEY_L) == glfw.PRESS:
    #         self.update_mocap_quat('y-')
    #         print(f"press: L \n")
    #         teleoperation_qpos.append(self.get_qpos())
    #     elif glfw.get_key(glfw_window, glfw.KEY_K) == glfw.PRESS:
    #         self.update_mocap_quat('x-')
    #         print(f"press: K \n")
    #         teleoperation_qpos.append(self.get_qpos())
    #     elif glfw.get_key(glfw_window, glfw.KEY_I) == glfw.PRESS:
    #         self.update_mocap_quat('x+')
    #         print(f"press: I \n")
    #         teleoperation_qpos.append(self.get_qpos())
    #     elif glfw.get_key(glfw_window, glfw.KEY_U) == glfw.PRESS:
    #         self.update_mocap_quat('z-')
    #         print(f"press: U \n")
    #         teleoperation_qpos.append(self.get_qpos())
    #     elif glfw.get_key(glfw_window, glfw.KEY_O) == glfw.PRESS:
    #         self.update_mocap_quat('z+')
    #         print(f"press: O \n")
    #         teleoperation_qpos.append(self.get_qpos())
    #     else:
    #         self.update_mocap_quat('0')
    #
    # def handle_keyboard_gripper_input(self, window):
    #     # 获取按键输入
    #     glfw_window = window._context.window  # 取出真实的 GLFW 窗口实例
    #     if glfw.get_key(glfw_window, glfw.KEY_1) == glfw.PRESS:
    #         self.update_gripper('1')
    #         print(f"press: 1 \n")
    #         teleoperation_qpos.append(self.get_qpos())
    #     elif glfw.get_key(glfw_window, glfw.KEY_0) == glfw.PRESS:
    #         self.update_gripper('0')
    #         print(f"press: 0 \n")
    #         teleoperation_qpos.append(self.get_qpos())

    def get_qpos(self):
        # 获得关节角
        qpos = np.zeros(14)
        for i in range(1, 7):
            qpos[i-1] = env._physics.named.data.qpos[f'left_joint_{i}'].item()
            qpos[i+6] = env._physics.named.data.qpos[f'right_joint_{i}'].item()

        qpos[6] = self.left_gripper
        qpos[13] = self.right_gripper
        # print("qpos: ", qpos)
        return qpos

    def __call__(self, window):
        # self.handle_keyboard_pos_input(window)
        # self.handle_keyboard_quat_input(window)
        # self.handle_keyboard_gripper_input(window)
        self.handle_keyboard_input(window)
        # left_quat = env._physics.named.data.xquat['mocap_left']
        action_left = np.concatenate([self.mocap_left_xpos, self.mocap_left_quat.elements, [self.left_gripper]])
        action_right = np.concatenate([self.mocap_right_xpos, self.mocap_right_quat.elements, [self.right_gripper]])

        # qpos = self.get_qpos()
        # print(qpos)
        return np.concatenate([action_left, action_right])

    # mocap_left_xpos = env._physics.named.data.xpos['mocap_left']
    # print(f"mocap_left_xpos: ", mocap_left_xpos)


# 文件保存函数
def save_qpos_to_txt(file_path):
    qpos_array = np.array(teleoperation_qpos)
    # np.savetxt(file_path, qpos_array, fmt='%.6f', delimiter=',', header='Joint Angles', comments='')
    np.savetxt(file_path, qpos_array, fmt='%.6f', delimiter=',', comments='')
    print(f"关节角度已保存到 {file_path}")


# 定义渲染函数
def render_func():
    global camera_distance, camera_pitch, camera_yaw, ts, policy, episode, teleoperation_qpos, num_episode

    # 只在首次调用时初始化 policy
    if policy is None:
        policy = Teleoperation_Policy(env, window)

    action = policy(window)
    ts = env.step(action)

    episode.append(ts)

    # 相机位置信息 (根据拖拽和滚轮调整相机位置)
    physics.named.model.cam_pos['left_pillar'][0] = camera_distance * np.cos(np.radians(camera_yaw)) * np.cos(
        np.radians(camera_pitch))
    physics.named.model.cam_pos['left_pillar'][1] = camera_distance * np.sin(np.radians(camera_yaw)) * np.cos(
        np.radians(camera_pitch))
    physics.named.model.cam_pos['left_pillar'][2] = camera_distance * np.sin(np.radians(camera_pitch))

    # 渲染图像 (分辨率: width x height)
    width, height = window.shape
    camera = physics.render(camera_id=0, width=width, height=height)

    # 检查空格键，如果按下则退出并保存数据
    glfw_window = window._context.window  # 获取 GLFW 窗口实例
    if glfw.get_key(glfw_window, glfw.KEY_SPACE) == glfw.PRESS:
        print("空格键按下，退出遥控模式...")
        # save_qpos_to_txt(f"teleoperation_data/source_txt/teleoperation_qpos_{num_episode}.txt")
        save_qpos_to_txt(f"EEpos/20_3/teleoperation_qpos_{num_episode}.txt")

        data_dict = {
            '/observations/qpos': [],
            '/action': [],
        }

        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []
        episode = episode[:-1]
        max_timesteps = len(teleoperation_qpos)
        for t in range(max_timesteps):
            ts = episode[t]
            data_dict['/action'].append(teleoperation_qpos[t])
            data_dict['/observations/qpos'].append(teleoperation_qpos[t])
            # data_dict['/action'].append(ts.action)
            # for cam_name in camera_names:
            #     data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
            data_dict[f'/observations/images/top'].append(camera_top[t])
            data_dict[f'/observations/images/angle'].append(camera_angle[t])
        # HDF5
        t0 = time.time()
        # dataset_path = os.path.join('/home/juyiii/data/aloha/sim_RM_teleoperation', f'episode_18_test')
        dataset_path = os.path.join('/home/juyiii/ALOHA/act-plus-plus/EEpos/20_3', f'episode_{num_episode}')
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

        window.close()  # 关闭窗口，结束事件循环
        return

    # 返回渲染的图像，转换为 np.uint8 格式的 3D 数组
    return np.array(camera, dtype=np.uint8).reshape(height, width, 3)


def main(args):
    global num_episode
    num_episode = args['num_episodes']
    # 运行窗口事件循环并展示图像
    window.event_loop(render_func)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    main(vars(parser.parse_args()))


