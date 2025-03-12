# 希望实现可以根据与障碍物之间的位置，对键盘操作和自主控制的轨迹灵活加权
import numpy as np
from dm_control.viewer.gui import glfw_gui
from dm_control.utils import inverse_kinematics as ik
from dm_control import mujoco
import mujoco as mj
import glfw
from dm_control.rl import control
from ee_sim_env import RMsimpletrajectoryEETask, RMpaperEETask
from constants import DT
from scripted_policy import RMPolicy_simpletrajectory
from pyquaternion import Quaternion
import time
import os
import h5py
import argparse
import pickle
from einops import rearrange
import torch
from torchvision import transforms
from scipy.spatial.transform import Rotation as R



from constants import SIM_TASK_CONFIGS
camera_names = SIM_TASK_CONFIGS['sim_RM_simpletrajectory']['camera_names']
camera_top = []
camera_angle = []



xml_path = 'assets/models/rm_bimanual_both.xml'
# MuJoCo data structures
model = mujoco.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mujoco.MjData(model)                     # MuJoCo data
cam = mujoco.MjvCamera()                        # Abstract camera
opt = mujoco.MjvOption()                        # visualization options
physics = mujoco.Physics.from_xml_path(xml_path)
task = RMpaperEETask(random=False)
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
image_list = []
qpos_history_raw = []
timestep = 0
target_qpos = []
ispolicy = False
all_actions = []
pre_action = []
target_action = []
query_timestep = 0
box_move = ['right', 'left', 'forward', 'back']
ee_pos = []
isinterpolated = False
interpolated_trajectory = []
target_qpos = []
interpolate_time = 0
obstacle_distance = []
box_left_gripper_distance = None
left_eepos_policy = None
left_eequat_policy = None

def compute_forward_kinematics(qpos):
    temp_physics = physics.copy(share_model=True)
    temp_physics.named.data.qpos[:len(qpos)] = qpos
    temp_physics.forward()
    left_xpos = temp_physics.named.data.xpos['handforcesensor3']
    left_quat = Quaternion(temp_physics.named.data.xquat['handforcesensor3'])
    right_xpos = temp_physics.named.data.xpos['handforcesensor4']
    right_quat = Quaternion(temp_physics.named.data.xquat['handforcesensor4'])
    print("left_eepos_reference: ", left_xpos)
    # print("right_eepos_reference: ", right_xpos)
    print("left_eequat_reference: ", left_quat)
    # print("right_eequat_reference: ", right_quat)
    left_eepos_policy = left_xpos
    left_eequat_policy = left_quat


def weighted_pose(position_teleop, quat_teleop, position_policy, quat_policy, weight):
    """
    计算加权后的目标位姿（位置 + 姿态）

    参数：
    - position_teleop: 遥操作输入的位置 (numpy 数组, shape=(3,))
    - quat_teleop: 遥操作输入的姿态（四元数, shape=(4,)）
    - position_policy: policy 预测的位置 (numpy 数组, shape=(3,))
    - quat_policy: policy 预测的姿态（四元数, shape=(4,)）
    - weight: 遥操作输入的权重，取值范围 [0, 1]，policy 的权重为 (1 - weight)

    返回：
    - position_weighted: 加权后的位置 (numpy 数组, shape=(3,))
    - quat_weighted: 加权后的四元数 (numpy 数组, shape=(4,))
    """

    # 1. 位置加权（直接加权平均）
    position_weighted = weight * position_teleop + (1 - weight) * position_policy

    # 2. 姿态加权插值（Slerp）
    quat_teleop = R.from_quat(quat_teleop)  # 转换为 Rotation 对象
    quat_policy = R.from_quat(quat_policy)

    # 计算球面线性插值（Slerp）
    quat_weighted = R.slerp(weight, [quat_teleop, quat_policy])  # 进行 Slerp
    quat_weighted = quat_weighted.as_quat()  # 转回四元数格式

    return position_weighted, quat_weighted


def compute_xy_distance(body1_name, body2_name):
    """
    计算两个 geom 在 XY 平面内的中心点距离。

    参数：
    model -- MuJoCo 模型 (mjModel)
    data -- MuJoCo 仿真数据 (mjData)
    body1_name -- 第一个 geom 的名称 (str)
    body2_name -- 第二个 geom 的名称 (str)

    返回：
    两个 body 在 XY 平面的距离 (float)
    """


    body1_pos = env._physics.named.data.xpos[body1_name]
    body2_pos = env._physics.named.data.xpos[body2_name]

    print("box_pos: ", body1_pos)
    print("left_gripper_pos: ", body2_pos)

    # 计算 XY 平面的欧几里得距离
    xy_distance = np.sqrt((body1_pos[0] - body2_pos[0]) ** 2 + (body1_pos[1] - body2_pos[1]) ** 2)

    return xy_distance

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


def get_image(ts, camera_names, rand_crop_resize=False):
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

class Teleoperation_Policy:
    def __init__(self, env, window, args, move_speed=0.001, rotate_speed=1):
        self.env = env
        self.window = window
        self.args = args
        self.move_speed = move_speed
        self.rotate_speed = rotate_speed
        self.mocap_left_xpos = np.array(env._physics.named.data.xpos['mocap_left'])
        self.mocap_left_quat = Quaternion(env._physics.named.data.mocap_quat['mocap_left'])
        self.mocap_right_xpos = np.array(env._physics.named.data.xpos['mocap_right'])
        self.mocap_right_quat = Quaternion(env._physics.named.data.mocap_quat['mocap_right'])
        self.left_gripper = 1
        self.right_gripper = 1
        self.last_key_time = {}  # 记录每个按键的最后一次按下时间
        self.debounce_interval = 0.1  # 去抖动时间间隔，单位：秒
        self.MYBpolicy = None
        self.stats = None
        self.query_frequency = 100
        self.all_time_actions = torch.zeros([2000, 2000+100, 16]).cuda()

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

    def update_box(self, direction):
        box_length = 0.005
        # 更新box的位置信息
        box_pos = env._physics.named.data.qpos['red_box_joint']
        print(box_pos)
        if direction == "right":
            # box的位置往右移0.05, 也就是y的负方向
            box_pos[1] -= box_length
        elif direction == "left":
            box_pos[1] += box_length  # 左移0.05单位

        elif direction == "forward":
            box_pos[0] -= box_length  # 上移0.05单位

        elif direction == "back":
            box_pos[0] += box_length  # 下移0.05单位

    def load_policy(self):
        from imitate_episodes_teleoperation import make_policy
        policy_config = {'lr': 1e-5,
                         'num_queries': self.args['chunk_size'],
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
                         'action_dim': 16,
                         'no_encoder': False,
                         'ckpt_dir': self.args['ckpt_dir'],
                         'policy_class': "ACT",
                         'seed': 0,
                         'num_steps': 20000,
                         }
        # MYBpolicy = make_policy("ACT", policy_config)
        from policy import ACTPolicy
        self.MYBpolicy = ACTPolicy(policy_config)
        ckpt_dir = self.args['ckpt_dir']
        ckpt_name = "policy_best.ckpt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        loading_status = self.MYBpolicy.deserialize(torch.load(ckpt_path))
        print(loading_status)
        self.MYBpolicy.cuda()
        self.MYBpolicy.eval()
        temporal_agg = self.args['temporal_agg']
        if temporal_agg:
            self.query_frequency = 1
        return self.MYBpolicy

    def query_policy(self, tab):
        global target_qpos, query_timestep, all_actions, ts, interpolated_trajectory, isinterpolated, interpolate_time

        temporal_agg = self.args['temporal_agg']
        qpos_numpy = []

        if query_timestep == 0 and not isinterpolated:
            ckpt_dir = self.args['ckpt_dir']
            stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
            print(os.path.getsize(stats_path))  # 检查文件大小，如果返回 0 表示文件为空
            with open(stats_path, 'rb') as f:
                self.stats = pickle.load(f)
            self.MYBpolicy = self.load_policy()

        with torch.inference_mode():
            if not isinterpolated:

                pre_process = lambda s_qpos: (s_qpos - self.stats['qpos_mean']) / self.stats['qpos_std']

                obs = ts.observation
                qpos_numpy = np.array(obs['qpos'])
                # print(qpos_numpy)
                pre_action.append(qpos_numpy)

                curr_image = get_image(ts, camera_names, rand_crop_resize=False)

                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

                if query_timestep % self.query_frequency == 0:

                    if query_timestep == 0:
                        # warm up
                        for _ in range(1):
                            self.MYBpolicy(qpos, curr_image)
                        print('network warm up done')
                    all_actions = self.MYBpolicy(qpos, curr_image)
                # print(f"all action(10)", all_actions[:10])
                if temporal_agg:
                    self.all_time_actions[[timestep], timestep:timestep + self.args['chunk_size']] = all_actions
                    actions_for_curr_step = self.all_time_actions[:, timestep]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1) # 用来检查哪些动作在所有维度中都不为零，从而筛选出已填充的有效动作
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, query_timestep % self.query_frequency]

                raw_action = raw_action.squeeze(0).cpu().numpy()

                # 后处理 去归一化
                post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']
                action = post_process(raw_action)
                target_qpos = action[:-2]

                query_timestep = query_timestep + 1

            target_action.append(target_qpos)

            # 正运动学计算target_qpos对应的末端位置
            compute_forward_kinematics(target_qpos)


            # print(f"qpos_target: ", target_qpos)

    def handle_keyboard_input(self, window):
        global timestep, ispolicy, ts, query_timestep
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

            glfw.KEY_TAB: ('TAB', self.query_policy),

            glfw.KEY_RIGHT: ('right', self.update_box),
            glfw.KEY_LEFT: ('left', self.update_box),
            glfw.KEY_UP: ('forward', self.update_box),
            glfw.KEY_DOWN: ('back', self.update_box),

        }
        # 遍历所有按键
        for key, (action, func) in key_actions.items():
            if glfw.get_key(glfw_window, key) == glfw.PRESS:
                # 检查去抖动时间间隔
                last_time = self.last_key_time.get(key, 0)
                if current_time - last_time > self.debounce_interval:
                    print(f"press: {action}")
                    func(action)  # 执行动作
                    if action not in box_move:
                        if action == 'TAB':
                            ispolicy = True
                            # 获取所有 weld 约束的数量
                            # num_welds = env._physics.model.eq_type.size
                            # print(num_welds)
                            # 关闭 weld 约束（例如，移除 mocap 绑定）
                            env._physics.model.eq_active[:] = 0  # 禁用所有 weld 约束

                        else:
                            if ispolicy:
                                query_timestep = 0
                                print("修改mocap的位姿")
                                # 获取当前位置机械臂末端的位姿并得到对应的mocap的位置
                                # 姿态偏移量
                                lefthand_set_quat = Quaternion([-0.53739432, -0.45963982, 0.53696655, -0.46000599])
                                leftmocap_set_quat = Quaternion([1, 0, 0, 0])
                                leftdelt_quat = leftmocap_set_quat / lefthand_set_quat
                                righthand_set_quat = Quaternion([-0.75968791, 0, 0, 0.65028784])
                                rightmocap_set_quat = Quaternion([1, 0, 0, 0])
                                rightdelt_quat = rightmocap_set_quat / righthand_set_quat
                                # 位置偏移量
                                leftdelt_pos = np.array([-0.0393153, 0.00838329, -0.00360406])
                                rightdelt_pos = np.array([-0.0401231, -0.0012093, -0.0036001])

                                # 获取当前机械臂末端的位姿
                                lefthand_xpos = env._physics.named.data.xpos['handforcesensor3']
                                lefthand_quat = Quaternion(env._physics.named.data.xquat['handforcesensor3'])
                                righthand_xpos = env._physics.named.data.xpos['handforcesensor4']
                                righthand_quat = Quaternion(env._physics.named.data.xquat['handforcesensor4'])

                                # 更新mocap的位姿
                                leftmocap_quat = lefthand_quat * leftdelt_quat
                                leftmocap_xpos = lefthand_xpos + leftdelt_pos
                                rightmocap_quat = righthand_quat * rightdelt_quat
                                rightmocap_xpos = righthand_xpos + rightdelt_pos

                                self.mocap_left_xpos = leftmocap_xpos
                                self.mocap_left_quat = leftmocap_quat
                                self.mocap_right_xpos = rightmocap_xpos
                                self.mocap_right_quat = rightmocap_quat

                            ispolicy = False
                            self.all_time_actions.detach().zero_()

                            env._physics.model.eq_active[:] = 1  # 开启所有 weld 约束

                        # ik_test()
                        teleoperation_qpos.append(self.get_qpos())
                        camera_top.append(physics.render(height=480, width=640, camera_id='top'))
                        camera_angle.append(physics.render(height=480, width=640, camera_id='angle'))

                        self.last_key_time[key] = current_time  # 更新按键的最后按下时间

                        obs = ts.observation
                        if 'images' in obs:
                            image_list.append(obs['images'])
                        else:
                            image_list.append({'main': obs['image']})
                        qpos_numpy = np.array(obs['qpos'])
                        # 将当前时间步的 qpos 添加到列表中
                        qpos_history_raw.append(qpos_numpy)

                        # qpos_history_raw[timestep] = qpos_numpy

                        left_xpos = env._physics.named.data.xpos['handforcesensor3']
                        print("left_eepos: ", left_xpos)
                        left_quat = Quaternion(env._physics.named.data.xquat['handforcesensor3'])
                        print("left_eequat: ", left_quat)
                        # right_xpos = env._physics.named.data.xpos['handforcesensor4']
                        # right_quat = Quaternion(env._physics.named.data.xquat['handforcesensor4'])
                        ee_pos.append(np.array(left_xpos))
                        # print("eepos last: ", ee_pos[-1])

                        timestep = timestep + 1
                        print(timestep)


                    # 计算box和left_gripper_geom之间的距离
                    box_left_gripper_distance = compute_xy_distance('box', 'left_7')
                    print("box_left_gripper_distance: ", box_left_gripper_distance)
                    obstacle_distance.append(box_left_gripper_distance)
                    print("\n")

                    break  # 防止多个按键同时触发


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

        # 如果是policy，那么action应该是一个长度为1+14的向量；如果是遥操作，那么action应该是长度为1+16的向量
        if ispolicy:
            action = np.concatenate([np.array([1]), target_qpos])
        else:
            action_left = np.concatenate([self.mocap_left_xpos, self.mocap_left_quat.elements, [self.left_gripper]])
            action_right = np.concatenate([self.mocap_right_xpos, self.mocap_right_quat.elements, [self.right_gripper]])
            action = np.concatenate([np.array([0]), action_left, action_right])

        # qpos = self.get_qpos()
        # print(qpos)
        return action

    # mocap_left_xpos = env._physics.named.data.xpos['mocap_left']
    # print(f"mocap_left_xpos: ", mocap_left_xpos)


# 文件保存函数
def save_qpos_to_txt(file_path):
    qpos_array = np.array(teleoperation_qpos)
    # np.savetxt(file_path, qpos_array, fmt='%.6f', delimiter=',', header='Joint Angles', comments='')
    np.savetxt(file_path, qpos_array, fmt='%.6f', delimiter=',', comments='')
    print(f"关节角度已保存到 {file_path}")


# 定义渲染函数
def render_func(args):
    global camera_distance, camera_pitch, camera_yaw, ts, policy, episode, teleoperation_qpos, num_episode, timestep, ee_pos

    # 只在首次调用时初始化 policy
    if policy is None:
        policy = Teleoperation_Policy(env, window,args)

    action = policy(window)
    # print(action)
    ts = env.step(action)

    episode.append(ts)
    #
    # obs = ts.observation
    # if 'images' in obs:
    #     image_list.append(obs['images'])
    # else:
    #     image_list.append({'main': obs['image']})
    # qpos_numpy = np.array(obs['qpos'])
    #
    # qpos_history_raw[timestep] = qpos_numpy
    #
    # timestep = timestep + 1

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
        # np.savetxt('pre_action.txt', pre_action, fmt='%f')  # 使用 '%f' 作为格式，表示浮点数
        # np.savetxt('target_action.txt', target_action, fmt='%f')
        np.savetxt('CLAWAR/distancetest.txt', ee_pos, fmt='%f')
        # 保存与障碍距离数据至txt中
        np.savetxt('CLAWAR/distance2.txt', obstacle_distance, fmt='%f')

        # # save_qpos_to_txt(f"teleoperation_data/source_txt/teleoperation_qpos_{num_episode}.txt")
        # save_qpos_to_txt(f"EEpos/20_3/teleoperation_qpos_{num_episode}.txt")
        #
        # data_dict = {
        #     '/observations/qpos': [],
        #     '/action': [],
        # }
        #
        # for cam_name in camera_names:
        #     data_dict[f'/observations/images/{cam_name}'] = []
        # episode = episode[:-1]
        # max_timesteps = len(teleoperation_qpos)
        # for t in range(max_timesteps):
        #     ts = episode[t]
        #     data_dict['/action'].append(teleoperation_qpos[t])
        #     data_dict['/observations/qpos'].append(teleoperation_qpos[t])
        #     # data_dict['/action'].append(ts.action)
        #     # for cam_name in camera_names:
        #     #     data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
        #     data_dict[f'/observations/images/top'].append(camera_top[t])
        #     data_dict[f'/observations/images/angle'].append(camera_angle[t])
        # # HDF5
        # t0 = time.time()
        # # dataset_path = os.path.join('/home/juyiii/data/aloha/sim_RM_teleoperation', f'episode_18_test')
        # dataset_path = os.path.join('/home/juyiii/ALOHA/act-plus-plus/EEpos/20_3', f'episode_{num_episode}')
        # with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        #     root.attrs['sim'] = True
        #     obs = root.create_group('observations')
        #     image = obs.create_group('images')
        #     for cam_name in camera_names:
        #         _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
        #                                  chunks=(1, 480, 640, 3), )
        #     # compression='gzip',compression_opts=2,)
        #     # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        #     qpos = obs.create_dataset('qpos', (max_timesteps, 14))
        #     action = root.create_dataset('action', (max_timesteps, 14))
        #
        #     for name, array in data_dict.items():
        #         root[name][...] = array
        # print(f'Saving: {time.time() - t0:.1f} secs\n')

        window.close()  # 关闭窗口，结束事件循环
        return

    # 返回渲染的图像，转换为 np.uint8 格式的 3D 数组
    return np.array(camera, dtype=np.uint8).reshape(height, width, 3)


def main(args):
    global num_episode
    num_episode = args['num_episodes']
    # 运行窗口事件循环并展示图像
    window.event_loop(lambda: render_func(args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)

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

    main(vars(parser.parse_args()))


