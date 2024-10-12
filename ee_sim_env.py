# 具有EE space control 的 Mujoco +DM_Control 环境（EE end effector，工作空间）
# 定义的函数在control.py中对应
import numpy as np
import collections
import os
import math

from constants import DT, XML_DIR, START_ARM_POSE, START_ARM_POSE_RM, START_ARM_POSE_SHADOWHAND
from constants import PUPPET_GRIPPER_POSITION_CLOSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from constants import RM_GRIPPER_UNNORMALIZE, RM_GRIPPER_NORMALIZE, RM_GRIPPER_VELOCITY_NORMALIZE
from constants import SHADOW_HAND_UNNORMALIZE, SHADOW_HAND_NORMALIZE, SHADOW_HAND_VELOCITY_NORMALIZE

from utils import sample_box_pose, sample_insertion_pose, sample_box_pose_RM, increment_function, sample_fireextinguisher_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from pyquaternion import Quaternion


import IPython
e = IPython.embed


def make_ee_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:      [left_arm_pose (7),             # position and quaternion for end effector
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_pose (7),            # position and quaternion for end effector
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_RM_simpletrajectory' in task_name:
        xml_path = os.path.join(XML_DIR, f'models/rm_bimanual_ee.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = RMsimpletrajectoryEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_RM_fire_extinguisher' in task_name:
        xml_path = os.path.join(XML_DIR, f'models/rm_bimanual_ee.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = RMFireExtinguisherEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        # 在每一步动作之前被调用。它接受动作和物理模型作为参数。动作被分为左右两部分，分别对应左右手的动作。这些动作被用来设置模拟环境中的位置和方向
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))

    def initialize_robots(self, physics):
        # 用于初始化机器人的状态。它首先重置关节位置，然后设置模拟环境中的位置和方向，最后重置夹具控制
        # reset joint position
        physics.named.data.qpos[:16] = START_ARM_POSE

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        np.copyto(physics.data.mocap_pos[0], [-0.31718881+0.1, 0.5, 0.29525084])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # right
        np.copyto(physics.data.mocap_pos[1], np.array([0.31718881-0.1, 0.49999888, 0.29525084]))
        np.copyto(physics.data.mocap_quat[1],  [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array([
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
        ])
        np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        # obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        # obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class TransferCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        # 调用函数 sample_box_pose()，该函数返回一个表示盒子新位置和姿态的数组 cube_pose。
        cube_pose = sample_box_pose()
        # 获取与盒子关联的关节在 qpos（关节位置数组）中的起始索引。具体来说，它使用 name2id 方法，通过给定的关节名称 'red_box_joint' 查找对应的索引。'joint' 指定了查找的类别为关节。
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        # 将新采样的盒子位置和姿态 (cube_pose) 复制到物理模拟的数据结构中。
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        return reward


class InsertionEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize peg and socket position
        peg_pose, socket_pose = sample_insertion_pose()
        id2index = lambda j_id: 16 + (j_id - 16) * 7 # first 16 is robot qpos, 7 is pose dim # hacky

        peg_start_id = physics.model.name2id('red_peg_joint', 'joint')
        peg_start_idx = id2index(peg_start_id)
        np.copyto(physics.data.qpos[peg_start_idx : peg_start_idx + 7], peg_pose)
        # print(f"randomized cube position to {cube_position}")

        socket_start_id = physics.model.name2id('blue_socket_joint', 'joint')
        socket_start_idx = id2index(socket_start_id)
        np.copyto(physics.data.qpos[socket_start_idx : socket_start_idx + 7], socket_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward

class RMsimpletrajectoryEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 2

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

    def before_step(self, action, physics):
        # print(f"left: ", physics.named.data.xpos['handforcesensor3'])
        # print(f"right: ", physics.named.data.xpos['handforcesensor4'])
        # 在每一步动作之前被调用。它接受动作和物理模型作为参数。动作被分为左右两部分，分别对应左右手的动作。这些动作被用来设置模拟环境中的位置和方向
        # 动作前一半是action_left, 后一半是action_right
        # action_left: mocap_pos[3] + mocap_quat[4] + gripper_ctrl[1]
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        # pos(3) quat(4) gripper(1)
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = RM_GRIPPER_UNNORMALIZE(action_left[7])
        g_right_ctrl = RM_GRIPPER_UNNORMALIZE(action_right[7])
        physics.data.ctrl[0] = g_left_ctrl
        physics.data.ctrl[1] = g_right_ctrl

        # np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))
    # vx300s_left/left_finger,  vx300s_left/right_finger, vx300s_right/left_finger, vx300s_right/right_finger

    def initialize_robots(self, physics):
        # 用于初始化机器人的状态。它首先重置关节位置，然后设置模拟环境中的位置和方向，最后重置夹具控制
        # reset joint position
        # note 换成小夹抓之后只有6+1+6+1个joint
        physics.named.data.qpos[:14] = START_ARM_POSE_RM

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side

        # 当机械臂关节为全0时，左sensor的pos为：[-0.4206847   0.56161671  0.40360408]，quat为：[-0.53739432 -0.45963982  0.53696655 -0.46000599],对应的euler为[89.954, -8.873, 89.993]
        #                   左mocap的pos为：[-0.46 0.57 0.4],quat为：[1,0,0,0], 对应的euler为：[ 0 0 0]
        # 所以mocap相对于sensor有一个固定的偏差：pos:[-0.0393153 0.00838329 -0.00360406],euler:[-89.954, 8.873, -89.993](degree)[-1.5691975,0.155,-1.569877](radians)

        # 姿态偏移量
        lefthand_set_quat = Quaternion([-0.53739432, -0.45963982,  0.53696655, -0.46000599])
        leftmocap_set_quat = Quaternion([1, 0, 0, 0])
        leftdelt_quat = leftmocap_set_quat / lefthand_set_quat
        righthand_set_quat = Quaternion([-0.75968791, 0, 0, 0.65028784])
        rightmocap_set_quat = Quaternion([1, 0, 0, 0])
        rightdelt_quat = rightmocap_set_quat / righthand_set_quat
        # 初始化后修改init，不修改偏移量！！！只有修改模型后偏移量需要重新校准
        lefthand_init_quat = Quaternion([-0.11732551, -0.16290137,  0.9796316,   0.00444887])
        leftmocap_init_quat = lefthand_init_quat * leftdelt_quat
        print(f"leftmocap init quat: ", leftmocap_init_quat)

        righthand_init_quat = Quaternion([-0.75968791, 0, 0, 0.65028784])
        rightmocap_init_quat =righthand_init_quat* rightdelt_quat
        print(f"rightmocap init quat: ", rightmocap_init_quat)

        # 位置偏移量
        leftdelt_pos = np.array([-0.0393153, 0.00838329, -0.00360406])
        rightdelt_pos = np.array([-0.0401231, -0.0012093, -0.0036001])

        # left
        np.copyto(physics.data.mocap_pos[0], np.array([-0.03911368,  0.87658713,  0.40874427]) + leftdelt_pos)
        np.copyto(physics.data.mocap_quat[0], leftmocap_init_quat.elements)
        # right
        np.copyto(physics.data.mocap_pos[1], np.array([-0.41987693, -0.56879072,  0.4036001]) + rightdelt_pos)
        np.copyto(physics.data.mocap_quat[1], rightmocap_init_quat.elements)

        # reset gripper control
        # close_gripper_control = np.array([
        #     PUPPET_GRIPPER_POSITION_CLOSE,
        #     -PUPPET_GRIPPER_POSITION_CLOSE,
        #     PUPPET_GRIPPER_POSITION_CLOSE,
        #     -PUPPET_GRIPPER_POSITION_CLOSE,
        # ])
        # physics.data.ctrl[0] = close_gripper_control[0]
        # physics.data.ctrl[1] = close_gripper_control[1]
        # physics.data.ctrl[2] = close_gripper_control[2]
        # physics.data.ctrl[3] = close_gripper_control[3]

        physics.data.ctrl[0] = 0
        physics.data.ctrl[1] = 0
        # physics.data.ctrl[2] = 0
        # physics.data.ctrl[3] = 0
        # np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        # 调用函数 sample_box_pose()，该函数返回一个表示盒子新位置和姿态的数组 cube_pose。
        cube_pose = sample_box_pose_RM()
        # 获取与盒子关联的关节在 qpos（关节位置数组）中的起始索引。具体来说，它使用 name2id 方法，通过给定的关节名称 'red_box_joint' 查找对应的索引。'joint' 指定了查找的类别为关节。
        # NOTE 源代码用index赋角度的方法不好
        # box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        # print("box_start_idx: ",box_start_idx)
        # 将新采样的盒子位置和姿态 (cube_pose) 复制到物理模拟的数据结构中。
        np.copyto(physics.named.data.qpos['red_box_joint'], cube_pose)
        # print(f"box: ", physics.named.data.qpos['red_box_joint'])
        # print(f"randomized cube position to {cube_position}")
        # print(f"initialize_episode: ", increment_function())
        # 调用函数以获取当前的计数器值
        episode_number = increment_function()
        print(f"initialize_episode: ", episode_number)
        # 使用格式化字符串创建文件名
        filename = f"Astar_data/output_{episode_number}.txt"
        with open(filename, 'r') as file:
            for line in file:
                # 去除行尾的换行符并按空格分割
                line = line.strip()
                if line.startswith('target'):
                    target_pos = list(map(float, line.split(':')[1].strip().split()))
                    target_quat = np.array([1, 0, 0, 0])
                    target = np.concatenate([target_pos, target_quat])
                    # print('target:', target)
                    if len(target) == 7:
                        # ball_idx = physics.model.name2id('ball_joint', 'joint')
                        # print("ball_idx: ",ball_idx)
                        np.copyto(physics.named.data.qpos['ball_joint'], target)
                        print(f"ball_pos: ", physics.named.data.qpos['ball_joint'])
                        np.copyto(physics.named.model.site_pos['hook'], target_pos)
                        np.copyto(physics.named.model.site_pos['anchor'], target_pos)
                    else:
                        print("Target position does not contain exactly 3 values.")

                if line.startswith('obstacle'):
                    obstacle_pose = list(map(float, line.split(':')[1].strip().split()))
                    obstacle_quat = np.array([1, 0, 0, 0])
                    obstcale = np.concatenate([obstacle_pose, obstacle_quat])
                    if len(obstcale) == 7:
                        np.copyto(physics.named.data.qpos['red_box_joint'], obstcale)
                        print(f"box_pos: ", physics.named.data.qpos['red_box_joint'])
                    else:
                        print("Box position does not contain exactly 3 values.")

                    # line.split(':') 将字符串按 : 分割，得到 ['target', '-0.49 0.58 0.4']。
                    # line.split(':')[1] 选择分割后数组的第二个元素，即 '-0.49 0.58 0.4'。
                    # .strip() 去除选择元素的首尾空白字符（包括换行符）。
                    # .split() 按空格分割处理后的字符串，得到 ['-0.49', '0.58', '0.4']。
                    # map(float, ...) 将列表中的每个字符串元素转换成浮点数。
                    # list(...) 将 map 对象转换成列表。
        # print(physics.data.qpos)


        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        # qpos: joint[6]+gripper[1]
        left_qpos_raw = qpos_raw[:7]
        right_qpos_raw = qpos_raw[7:14]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [RM_GRIPPER_NORMALIZE(left_qpos_raw[6])]
        right_gripper_qpos = [RM_GRIPPER_NORMALIZE(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:7]
        right_qvel_raw = qvel_raw[7:14]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [RM_GRIPPER_VELOCITY_NORMALIZE(left_qvel_raw[6])]
        right_gripper_qvel = [RM_GRIPPER_VELOCITY_NORMALIZE(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[14:]
        return env_state

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        # obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)
            # print(contact_pair)

        touch_left_gripper = ("gripper_left_u", "red_box") in all_contact_pairs or ("gripper_left_d", "red_box") in all_contact_pairs
        touch_right_gripper = ("gripper_right_u", "red_box") in all_contact_pairs or ("gripper_right_d", "red_box") in all_contact_pairs
        touch_table = ("red_box", "floortop") in all_contact_pairs
        touch_gripper_ball = ("ball_geom", "left_gripper_geom") in all_contact_pairs or ("ball_geom","left_6_geom") in all_contact_pairs
        touch_arm_box = (("red_box", "left_gripper_geom") in all_contact_pairs
                         or ("red_box", "left_6_geom") in all_contact_pairs
                         or ("red_box", "left_5_geom") in all_contact_pairs
                         or ("red_box", "left_4_geom") in all_contact_pairs
                         or ("red_box", "left_3_geom") in all_contact_pairs
                         or ("red_box", "left_2_geom") in all_contact_pairs
                         or ("red_box", "left_1_geom") in all_contact_pairs)


        reward = 0
        # if touch_right_gripper and touch_left_gripper:
        #     reward = 1
        # if (touch_right_gripper and touch_left_gripper) and not touch_table: # lifted
        #     reward = 2
        # if touch_left_gripper: # attempted transfer
        #     reward = 3
        # if touch_left_gripper and not touch_table: # successful transfer
        #     reward = 4
        if touch_arm_box:
            reward = -1
        if touch_gripper_ball:
            reward = 1
        if touch_gripper_ball and not touch_arm_box:
            reward = 2
        # print(reward)
        return reward

class RMFireExtinguisherEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 2

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

    def before_step(self, action, physics):
        # print(f"left: ", physics.named.data.xpos['handforcesensor3'])
        # print(f"right: ", physics.named.data.xpos['handforcesensor4'])
        # 在每一步动作之前被调用。它接受动作和物理模型作为参数。动作被分为左右两部分，分别对应左右手的动作。这些动作被用来设置模拟环境中的位置和方向
        # 动作前一半是action_left, 后一半是action_right
        # action_left: mocap_pos[3] + mocap_quat[4] + gripper_ctrl[1]
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        # pos(3) quat(4) gripper(1)
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        # todo 这里的gripper 赋给 ctrl 需要修改，因为修改了夹抓为灵巧手，左右各有20个joint
        g_left_ctrl = SHADOW_HAND_UNNORMALIZE(action_left[7])
        g_right_ctrl = SHADOW_HAND_UNNORMALIZE(action_right[7])
        physics.data.ctrl[0:24] = g_left_ctrl
        physics.data.ctrl[24:48] = g_right_ctrl

        # np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))
    # vx300s_left/left_finger,  vx300s_left/right_finger, vx300s_right/left_finger, vx300s_right/right_finger

    def initialize_robots(self, physics):
        # 用于初始化机器人的状态。它首先重置关节位置，然后设置模拟环境中的位置和方向，最后重置夹具控制
        # reset joint position
        # note 换成灵巧手之后有6+24+6+24个joint
        physics.named.data.qpos[:60] = START_ARM_POSE_SHADOWHAND


        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side

        # todo mocap 的位置和姿态需要修改
        euler = [0, -0.52, -0.52]
        R = self.rpy2R(euler)
        init_left_quat = Quaternion._from_matrix(R)
        # init_left_quat = Quaternion(euler[0], euler[1], euler[2])

        # x_quat = Quaternion(axis=[1.0, 0.0, 0.0],degrees=-30)
        # init_left_quat = Quaternion(axis=[0.0, 0.0, 1.0], degrees=-30)
        # init_left_quat = init_left_quat * x_quat
        init_right_quat = Quaternion(axis=[0.0, 0.0, 1.0], degrees=68)


        np.copyto(physics.data.mocap_pos[0], np.array([0.2, 0.6, 0.4]))
        np.copyto(physics.data.mocap_quat[0], init_left_quat.elements)
        # right
        np.copyto(physics.data.mocap_pos[1], np.array([0.2, -0.6, 0.4]))
        np.copyto(physics.data.mocap_quat[1], init_right_quat.elements)

        # reset gripper control
        # close_gripper_control = np.array([
        #     PUPPET_GRIPPER_POSITION_CLOSE,
        #     -PUPPET_GRIPPER_POSITION_CLOSE,
        #     PUPPET_GRIPPER_POSITION_CLOSE,
        #     -PUPPET_GRIPPER_POSITION_CLOSE,
        # ])
        # physics.data.ctrl[0] = close_gripper_control[0]
        # physics.data.ctrl[1] = close_gripper_control[1]
        # physics.data.ctrl[2] = close_gripper_control[2]
        # physics.data.ctrl[3] = close_gripper_control[3]
        # todo 这里的夹抓控制记得修改
        physics.data.ctrl[0:24] = [0] * 24
        physics.data.ctrl[24:48] = [0] * 24
        # physics.data.ctrl[2] = 0
        # physics.data.ctrl[3] = 0
        # np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # todo 从修改box位置改为修改灭火器位置
        fire_extinguisher_pose = sample_fireextinguisher_pose()
        np.copyto(physics.named.data.qpos['fire_extinguisher_joint'], fire_extinguisher_pose)
        # print(f"box: ", physics.named.data.qpos['red_box_joint'])
        # print(f"randomized cube position to {cube_position}")
        # print(physics.data.qpos)

        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        # qpos: joint[6]+gripper[24]
        left_qpos_raw = qpos_raw[:30]
        right_qpos_raw = qpos_raw[30:60]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [SHADOW_HAND_NORMALIZE(left_qpos_raw[6:30])]
        right_gripper_qpos = [SHADOW_HAND_NORMALIZE(right_qpos_raw[6:30])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:30]
        right_qvel_raw = qvel_raw[30:60]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [SHADOW_HAND_VELOCITY_NORMALIZE(left_qvel_raw[6:30])]
        right_gripper_qvel = [SHADOW_HAND_VELOCITY_NORMALIZE(right_qvel_raw[6:30])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[60:]
        return env_state

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        # obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)
            # print(contact_pair)

        # todo 嘶，这个 reward 怎么写啊
        touch_left_gripper = ("gripper_left_u", "red_box") in all_contact_pairs or ("gripper_left_d", "red_box") in all_contact_pairs
        touch_right_gripper = ("gripper_right_u", "red_box") in all_contact_pairs or ("gripper_right_d", "red_box") in all_contact_pairs
        touch_table = ("red_box", "floortop") in all_contact_pairs
        touch_gripper_ball = ("ball_geom", "left_gripper_geom") in all_contact_pairs or ("ball_geom","left_6_geom") in all_contact_pairs
        touch_arm_box = (("red_box", "left_gripper_geom") in all_contact_pairs
                         or ("red_box", "left_6_geom") in all_contact_pairs
                         or ("red_box", "left_5_geom") in all_contact_pairs
                         or ("red_box", "left_4_geom") in all_contact_pairs
                         or ("red_box", "left_3_geom") in all_contact_pairs
                         or ("red_box", "left_2_geom") in all_contact_pairs
                         or ("red_box", "left_1_geom") in all_contact_pairs)


        reward = 0
        # if touch_right_gripper and touch_left_gripper:
        #     reward = 1
        # if (touch_right_gripper and touch_left_gripper) and not touch_table: # lifted
        #     reward = 2
        # if touch_left_gripper: # attempted transfer
        #     reward = 3
        # if touch_left_gripper and not touch_table: # successful transfer
        #     reward = 4
        if touch_arm_box:
            reward = -1
        if touch_gripper_ball:
            reward = 1
        if touch_gripper_ball and not touch_arm_box:
            reward = 2
        # print(reward)
        return reward

