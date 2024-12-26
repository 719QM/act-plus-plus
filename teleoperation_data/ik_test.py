from dm_control import mujoco
from dm_control.utils import inverse_kinematics as ik
import numpy as np

# 模型文件路径（替换为你的实际 XML 文件路径）
ARM_XML = '/home/juyiii/ALOHA/act-plus-plus/assets/models/rm_bimanual_ee.xml'

# 目标参数
SITE_NAME = 'mocap_left_site1'  # Site名称，需在XML文件中定义
TARGET_POS = np.array([-0.46, 0.57, 0.4])  # 目标位置
TARGET_QUAT = None  # 目标方向 (四元数)，如果不需要可以设为None
JOINT_NAMES = ['left_joint_1', 'left_joint_2', 'left_joint_3', 'left_joint_4', 'left_joint_5', 'left_joint_6']  # 需要控制的关节
TOL = 1e-4  # 误差容限
MAX_STEPS = 200  # 最大迭代次数

# 加载模型
physics = mujoco.Physics.from_xml_path(ARM_XML)

# 逆运动学求解
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
