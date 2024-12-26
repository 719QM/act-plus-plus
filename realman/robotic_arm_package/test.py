from robotic_arm import *

robot= Arm(RM75, '192.168.1.18')
# API 版本信息
print(robot.API_Version())

# # 设置关节使能
# lim = [178, 130, 135, 178, 130, 360, 360]
# # 设置关节 1-6 最小限位
# test = [150, 120, 100, 150, 120, 180, 180]
# for i in range(1, 7):
#     robot.Set_Joint_EN_State(i, False)
#     time.sleep(1)
#     robot.Set_Joint_Min_Pos(i, -test[i])
#     time.sleep(1)
#     robot.Set_Joint_EN_State(i, True)
#     time.sleep(1)
# print(f'关节最小限位：{robot.Get_Joint_Min_Pos()}')


_, joint, pose, Arm_Err, Sys_Err = robot.Get_Current_Arm_State()
print(joint)

initial_joint = [0, 0, 0, 0, 0, 0, 0]
block = 1
robot.Set_Arm_Init_Pose(initial_joint, block)
robot.Movej_Cmd(initial_joint, v=1, r=0, trajectory_connect=0, block=1)

# 断开连接
robot.RM_API_UnInit()
robot.Arm_Socket_Close()