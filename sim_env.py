# 具有joint space control 的 Mujoco +DM_Control 环境 (通过关节控制）
import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import DT, XML_DIR, START_ARM_POSE, START_ARM_POSE_RM, START_ARM_POSE_SHADOWHAND
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from constants import RM_GRIPPER_UNNORMALIZE, RM_GRIPPER_NORMALIZE, RM_GRIPPER_VELOCITY_NORMALIZE
from constants import SHADOW_HAND_NORMALIZE, SHADOW_HAND_VELOCITY_NORMALIZE, SHADOW_HAND_UNNORMALIZE
from utils import increment_function_jointspace, sample_fireextinguisher_pose, sample_ball_pose_RM

import IPython
e = IPython.embed

BOX_POSE = [None] # to be changed from outside

def make_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position 此处与ee不同
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
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
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_RM_simpletrajectory' in task_name:
        xml_path = os.path.join(XML_DIR, f'models/rm_bimanual.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = RMsimpletrajectoryTask(random=False)
        env = control.Environment(physics, task, time_limit=1000, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_RM_fire_extinguisher' in task_name:
        xml_path = os.path.join(XML_DIR, f'models/rm_bimanual.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = RMFireExtinguisherTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7:7+6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7+6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
        super().before_step(env_action, physics)
        return

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
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['left_wrist'] = physics.render(height=480, width=640, camera_id='left_wrist')
        obs['images']['right_wrist'] = physics.render(height=480, width=640, camera_id='right_wrist')
        # obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        # obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
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


class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7*2:] = BOX_POSE[0] # two objects
            # print(f"{BOX_POSE=}")
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

class RMsimpletrajectoryTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 2
        self.prev_error = np.zeros(14)
        self.integral_error = np.zeros(14)


    def pid_control(self, physics, action):
        # 初始设定
        Kp = 1.0  # 比例增益
        Ki = 0.1  # 积分增益
        Kd = 0.1  # 微分增益

        # 获取当前关节角度
        current_angles = np.array(physics.data.qpos[:14])
        # 目标角度
        target_angles = np.array(action[:14])  # 假设action是一个包含14个关节角度的数组
        # 计算误差
        error = target_angles - current_angles

        # 计算控制信号
        control_signal = Kp * error + Ki * self.integral_error + Kd * (error - self.prev_error)

        # 更新积分误差
        self.integral_error += error

        # 更新前一次误差
        self.prev_error = error

        # 应用控制信号
        pid_pose = current_angles + control_signal

        return pid_pose

    def before_step(self, action, physics):
        # print(f"action: ", action[7])
        # print(f"qpos: ", physics.data.qpos[8])
        # note 这里尝试了一下用PID控制解决关节角突变以及震动的问题，但是没有效果
        # pid_pose = self.pid_control(physics, action)
        # left_arm_action = pid_pose[:6]
        # right_arm_action = pid_pose[7:7 + 6]

        left_arm_action = action[:6]
        right_arm_action = action[7:7+6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7+6]

        left_gripper_action = RM_GRIPPER_UNNORMALIZE(normalized_left_gripper_action)
        right_gripper_action = RM_GRIPPER_UNNORMALIZE(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action]
        full_right_gripper_action = [right_gripper_action]

        velocity_servo = np.zeros(14)
        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action, velocity_servo])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:14] = START_ARM_POSE_RM
            np.copyto(physics.data.ctrl[:14], START_ARM_POSE_RM)

            # NOTE 如果是执行imitate_episodes.py程序，进行evaluate需要对障碍物箱子目标小球位置进行随机采样，执行以下代码
            #
            # assert BOX_POSE[0] is not None
            # physics.named.data.qpos[-14:-7] = BOX_POSE[0]
            #
            # ball_pose = sample_ball_pose_RM()
            # np.copyto(physics.named.data.qpos['ball_joint'], ball_pose)
            # np.copyto(physics.named.model.site_pos['hook'], ball_pose[:3])
            #
            # np.copyto(physics.named.model.site_pos['anchor'], ball_pose[:3])

            # NOTE 如果是读取txt文件读取路径，执行以下代码读取障碍物位置以及目标小球位置
            print(f"{BOX_POSE=}")
            episode_number = increment_function_jointspace()
            print(f"sim_env: ", episode_number)
            # 使用格式化字符串创建文件名
            filename = f"Astar_data/output_19.txt"
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
                            # print(f"box_qpos: ", physics.named.data.qpos['ball_joint'])

                            np.copyto(physics.named.model.site_pos['hook'], target_pos)

                            np.copyto(physics.named.model.site_pos['anchor'], target_pos)
                        else:
                            print("Target position does not contain exactly 3 values.")

                    if line.startswith('obstacle'):
                        obstacle_pose = list(map(float, line.split(':')[1].strip().split()))
                        obstacle_quat = np.array([1, 0, 0, 0])
                        obstcale = np.concatenate([obstacle_pose, obstacle_quat])
                        if len(obstcale) == 7:
                            physics.named.data.qpos[-14:-7] = obstcale
                        else:
                            print("Box position does not contain exactly 3 values.")

            # print(physics.data.qpos)
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        # print(qpos_raw)
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
    def get_pos(physics):
        left_pos = physics.named.data.xpos['left_7']
        # print(f'left_pos: ',left_pos)
        return left_pos


    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        # obs['images']['left_wrist'] = physics.render(height=480, width=640, camera_id='left_wrist')
        # obs['images']['right_wrist'] = physics.render(height=480, width=640, camera_id='right_wrist')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        # obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        obs['position'] = self.get_pos(physics)

        return obs

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

        touch_left_gripper = ("gripper_left_u", "red_box") in all_contact_pairs or ("gripper_left_d", "red_box") in all_contact_pairs
        touch_right_gripper = ("gripper_right_u", "red_box") in all_contact_pairs or ("gripper_right_d", "red_box") in all_contact_pairs
        touch_table = ("red_box", "floortop") in all_contact_pairs

        touch_gripper_ball = ("ball_geom", "left_gripper_geom") in all_contact_pairs or (
        "ball_geom", "left_6_geom") in all_contact_pairs
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
        # if (touch_right_gripper and touch_left_gripper) and not touch_table:  # lifted
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

class RMFireExtinguisherTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 2


    def before_step(self, action, physics):
        # print(f"action: ", action[7])
        # print(f"qpos: ", physics.data.qpos[8])

        left_arm_action = action[:6]
        right_arm_action = action[7:7+6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7+6]

        left_gripper_action = SHADOW_HAND_UNNORMALIZE(normalized_left_gripper_action)
        right_gripper_action = SHADOW_HAND_UNNORMALIZE(normalized_right_gripper_action)

        full_left_gripper_action = left_gripper_action
        full_right_gripper_action = right_gripper_action

        # velocity_servo = np.zeros(12)
        env_action = np.concatenate([full_left_gripper_action, full_right_gripper_action, left_arm_action, right_arm_action])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:60] = START_ARM_POSE_SHADOWHAND
            np.copyto(physics.data.ctrl[48:54], START_ARM_POSE_SHADOWHAND[:6])  # left_joint
            np.copyto(physics.data.ctrl[:24], START_ARM_POSE_SHADOWHAND[6:30])  # left_hand
            np.copyto(physics.data.ctrl[54:60], START_ARM_POSE_SHADOWHAND[30:36])  # right_joint
            np.copyto(physics.data.ctrl[24:48], START_ARM_POSE_SHADOWHAND[36:60])  # right_hand


        # todo 从修改box位置改为修改灭火器位置
        fire_extinguisher_pose = sample_fireextinguisher_pose()
        np.copyto(physics.named.data.qpos['fire_extinguisher_joint'], fire_extinguisher_pose)
            # print(physics.data.qpos)
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
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

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        # obs['images']['left_wrist'] = physics.render(height=480, width=640, camera_id='left_wrist')
        # obs['images']['right_wrist'] = physics.render(height=480, width=640, camera_id='right_wrist')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        # obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')

        return obs

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[60:]
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

        touch_left_gripper = ("gripper_left_u", "red_box") in all_contact_pairs or ("gripper_left_d", "red_box") in all_contact_pairs
        touch_right_gripper = ("gripper_right_u", "red_box") in all_contact_pairs or ("gripper_right_d", "red_box") in all_contact_pairs
        touch_table = ("red_box", "floortop") in all_contact_pairs

        touch_gripper_ball = ("ball_geom", "left_gripper_geom") in all_contact_pairs or (
        "ball_geom", "left_6_geom") in all_contact_pairs
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
        # if (touch_right_gripper and touch_left_gripper) and not touch_table:  # lifted
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

# todo 下面的两个函数，用于测试在仿真环境中的双臂控制，需要在硬件上测试，如果改成自己的硬件，需要修改函数中的参数
def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # gripper action
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
    action[6] = normalized_left_pos
    action[7+6] = normalized_right_pos
    return action

def test_sim_teleop():
    """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=False)

    # setup the environment
    env = make_sim_env('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]
    # setup plotting
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['angle'])
    plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        plt_img.set_data(ts.observation['images']['angle'])
        plt.pause(0.02)


if __name__ == '__main__':
    test_sim_teleop()

