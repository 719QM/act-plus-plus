# 跨文件共享的常量
import pathlib
import os

import numpy as np

### Task parameters
# DATA_DIR = '/home/zfu/interbotix_ws/src/act/data' if os.getlogin() == 'zfu' else '/scr/tonyzhao/datasets'
DATA_DIR = '/home/juyiii/data/aloha/'
REALMAN_TASK_CONFIGS = {
    'rmreal_pick':{
        'dataset_dir': DATA_DIR + '/rmreal_pick',
        'num_episodes': 38,
        'episode_len': 90,
        'camera_names': ['image_100', 'image_110']
    },

}
SIM_TASK_CONFIGS = {
    'sim_RM_simpletrajectory':{
        'dataset_dir': DATA_DIR + '/sim_RM_Astar_teleoperation_adjusted',
        'num_episodes': 123,
        'episode_len': 1000,
        # 'camera_names': ['top', 'left_wrist', 'right_wrist']
        'camera_names': ['top', 'angle']
    },
    'rmreal_task': {
        'dataset_dir': DATA_DIR + '/rmreal',
        'num_episodes': 100,
        'episode_len': 1000,
        # 'camera_names': ['top', 'left_wrist', 'right_wrist']
        'camera_names': ['top']
    },
    'sim_RM_fire_extinguisher':{
        'dataset_dir': DATA_DIR + '/sim_RM_fire_extinguisher',
        'num_episodes': 20,
        'episode_len': 500,
        # 'camera_names': ['top', 'left_wrist', 'right_wrist']
        'camera_names': ['top']

    },
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        # 'camera_names': ['top', 'left_wrist', 'right_wrist']
        'camera_names': ['top']

    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        # 'camera_names': ['top', 'left_wrist', 'right_wrist']
        'camera_names': ['top']
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['top']
    },
    'all': {
        'dataset_dir': DATA_DIR + '/',
        'num_episodes': None,
        'episode_len': None,
        'name_filter': lambda n: 'sim' not in n,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    'sim_transfer_cube_scripted_mirror':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted_mirror',
        'num_episodes': None,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_scripted_mirror': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted_mirror',
        'num_episodes': None,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

}


### Simulation envs fixed constants
DT = 0.05
FPS = 20
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
# START_ARM_POSE_RM = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
START_ARM_POSE_RM = [0, -0.431, -0.613, -0.777, -0.067, -0.565, 0, 0, 0, 0, 0, 0, 0, 0]


# SHADOWHAND
LEFT_ARM_START = np.array([1.5633898, 0.27038634, -1.4091092, -2.009177, 2.0931191, -4.208923])
RIGHT_ARM_START = np.array([0.017297983, -0.048058573, 1.2328076, -0.016972838, -2.313671, 0.12582023])
LEFT_SHADOWHAND_START = np.zeros(24)
RIGHT_SHADOWHAND_START = np.zeros(24)

START_ARM_POSE_SHADOWHAND = np.concatenate([LEFT_ARM_START, LEFT_SHADOWHAND_START, RIGHT_ARM_START, RIGHT_SHADOWHAND_START])

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = -0.8
MASTER_GRIPPER_JOINT_CLOSE = -1.65
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2

RM_GRIPPER_OPEN = 0
RM_GRIPPER_CLOSE = 0.1


# x 为gripper张开的程度， 输出为gripper的位置
RM_GRIPPER_UNNORMALIZE = lambda x: x * (RM_GRIPPER_OPEN - RM_GRIPPER_CLOSE) + RM_GRIPPER_CLOSE
# x 为gripper的位置， 输出为gripper张开的程度
RM_GRIPPER_NORMALIZE = lambda x: (x - RM_GRIPPER_CLOSE) / (RM_GRIPPER_OPEN - RM_GRIPPER_CLOSE)
RM_GRIPPER_VELOCITY_NORMALIZE = lambda x: x / (RM_GRIPPER_OPEN - RM_GRIPPER_CLOSE)


# OPEN:1 CLOSED:0
# SHADOW_HAND_OPEN = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
SHADOW_HAND_OPEN = np.zeros(24)
SHADOW_HAND_CLOSED = np.array([0.001, 0.001, -0.112, 1.08, 1.34, 1.3, 0.001, 1.08, 1.34, 2, 0.001, 1.08, 1.34, 2, 0.001, 0.001, 1.08, 1.34,
                      1.5, 0.001, 0.837, 0.023, 0.538, 0.0223])
# SHADOW_HAND_UNNORMALIZE = lambda x: x * (SHADOW_HAND_OPEN - SHADOW_HAND_CLOSED) + SHADOW_HAND_CLOSED
SHADOW_HAND_UNNORMALIZE = lambda x: [(open_val - closed_val) * x + closed_val
                                     for open_val, closed_val
                                     in zip(SHADOW_HAND_OPEN, SHADOW_HAND_CLOSED)]
# SHADOW_HAND_NORMALIZE = lambda x: (x - SHADOW_HAND_CLOSED) / (SHADOW_HAND_OPEN - SHADOW_HAND_CLOSED)
# SHADOW_HAND_VELOCITY_NORMALIZE = lambda x: x / (SHADOW_HAND_OPEN - SHADOW_HAND_CLOSED)
# SHADOW_HAND_NORMALIZE = lambda x: [(sum(abs(x_val - closed_val)) / sum(abs(open_val - closed_val)))
#                                    for x_val, closed_val, open_val
#                                    in zip(x, SHADOW_HAND_CLOSED, SHADOW_HAND_OPEN)]
# SHADOW_HAND_VELOCITY_NORMALIZE = lambda x: [(sum(abs(x_val)) / sum(abs(open_val - closed_val)))
#                                    for x_val, closed_val, open_val
#                                    in zip(x, SHADOW_HAND_CLOSED, SHADOW_HAND_OPEN)]


def SHADOW_HAND_NORMALIZE(x):
    diff_closed_open = SHADOW_HAND_OPEN - SHADOW_HAND_CLOSED
    diff_x_closed = x - SHADOW_HAND_CLOSED
    sum_abs_diff_closed_open = np.sum(np.abs(diff_closed_open))
    sum_abs_diff_x_closed = np.sum(np.abs(diff_x_closed))

    # 避免除以零
    if sum_abs_diff_closed_open == 0:
        return 0
    else:
        return sum_abs_diff_x_closed / sum_abs_diff_closed_open

def SHADOW_HAND_VELOCITY_NORMALIZE(x):
    diff_closed_open = SHADOW_HAND_OPEN - SHADOW_HAND_CLOSED
    sum_abs_diff_closed_open = np.sum(np.abs(diff_closed_open))
    sum_abs_diff_x = np.sum(np.abs(x))

    # 避免除以零
    if sum_abs_diff_closed_open == 0:
        return 0
    else:
        return sum_abs_diff_x / sum_abs_diff_closed_open
