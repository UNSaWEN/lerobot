# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SO101 手机遥操作示例

此脚本使用手机 (iOS 或 Android) 遥操作 SO101 机械臂。
相比 phone_to_so100，此版本使用了专门为 SO101 调整的坐标映射。

使用方法:
1. 确保 SO101 机械臂已连接并校准
2. 确保 URDF 文件路径正确 (默认: ./SO101/so101_new_calib.urdf)
3. 根据你的设置修改串口 (port) 和手机系统 (PhoneOS.IOS 或 PhoneOS.ANDROID)
4. 运行: python examples/phone_to_so101/teleoperate.py

校准姿势:
- 手机屏幕朝上
- 手机顶部指向机械臂前方 (与机械臂伸展方向相同)
- 操作者站在机械臂后方

如果方向仍然不对，可以调整 so101_phone_processor.py 中的坐标映射。
"""

import time

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# 导入 SO101 专用的手机动作映射处理器
from so101_phone_processor import MapPhoneActionToRobotActionSO101

# ============== 配置参数 ==============
# 根据你的实际设置修改以下参数

# 串口配置 (使用 lerobot-find-port 查找)
ROBOT_PORT = "/dev/ttyACM0"

# 机器人 ID (与校准时使用的 ID 一致)
ROBOT_ID = "R_follower_arm"

# 手机系统
PHONE_OS = PhoneOS.IOS  # 或 PhoneOS.ANDROID

# URDF 文件路径
URDF_PATH = "./SO101/so101_new_calib.urdf"

# 控制频率
FPS = 30

# 末端执行器步长 (控制灵敏度，值越大移动越快)
EE_STEP_SIZES = {"x": 0.3, "y": 0.3, "z": 0.3}

# 工作空间边界 (安全限制，单位: 米)
# 根据你的机械臂安装位置调整
EE_BOUNDS = {
    "min": [-0.3, -0.3, 0.02],  # X_min, Y_min, Z_min (Z_min > 0 防止碰撞底座)
    "max": [0.3, 0.3, 0.4]      # X_max, Y_max, Z_max
}

# 单步最大移动距离 (安全限制，单位: 米)
MAX_EE_STEP_M = 0.05

# 夹爪速度因子
GRIPPER_SPEED_FACTOR = 20.0

# ======================================


def main():
    # 创建机器人配置
    robot_config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        use_degrees=True
    )

    # 创建手机遥操作配置
    teleop_config = PhoneConfig(phone_os=PHONE_OS)

    # 初始化机器人和遥操作设备
    robot = SO101Follower(robot_config)
    teleop_device = Phone(teleop_config)

    # 创建运动学求解器
    # 使用 SO101 的 URDF 文件
    kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # 构建处理管道: 手机动作 -> 末端执行器位姿 -> 关节角度
    phone_to_robot_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            # 第1步: 手机动作 -> 目标增量 (使用 SO101 专用映射)
            MapPhoneActionToRobotActionSO101(platform=teleop_config.phone_os),

            # 第2步: 目标增量 -> 末端执行器绝对位姿
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes=EE_STEP_SIZES,
                motor_names=list(robot.bus.motors.keys()),
                use_latched_reference=True,
            ),

            # 第3步: 安全边界检查和限制
            EEBoundsAndSafety(
                end_effector_bounds=EE_BOUNDS,
                max_ee_step_m=MAX_EE_STEP_M,
            ),

            # 第4步: 夹爪速度 -> 夹爪位置
            GripperVelocityToJoint(
                speed_factor=GRIPPER_SPEED_FACTOR,
            ),

            # 第5步: 末端执行器位姿 -> 关节角度 (逆运动学)
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=list(robot.bus.motors.keys()),
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # 连接机器人和遥操作设备
    robot.connect()
    teleop_device.connect()

    # 初始化可视化 (Rerun)
    init_rerun(session_name="phone_so101_teleop")

    if not robot.is_connected or not teleop_device.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("=" * 60)
    print("SO101 手机遥操作已启动")
    print("=" * 60)
    print(f"机器人端口: {ROBOT_PORT}")
    print(f"机器人 ID: {ROBOT_ID}")
    print(f"手机系统: {PHONE_OS.value}")
    print(f"控制频率: {FPS} Hz")
    print("-" * 60)
    print("操作说明:")
    print("  - iOS: 按住 B1 启用控制，A3 控制夹爪")
    print("  - Android: 按住 Move 启用控制，A/B 控制夹爪")
    print("-" * 60)
    print("如果方向不对，请修改 so101_phone_processor.py 中的坐标映射")
    print("按 Ctrl+C 退出")
    print("=" * 60)

    try:
        while True:
            t0 = time.perf_counter()

            # 获取机器人当前状态
            robot_obs = robot.get_observation()

            # 获取手机遥操作动作
            phone_obs = teleop_device.get_action()

            # 处理管道: 手机动作 -> 末端执行器位姿 -> 关节角度
            joint_action = phone_to_robot_joints_processor((phone_obs, robot_obs))

            # 发送动作到机器人
            _ = robot.send_action(joint_action)

            # 可视化
            log_rerun_data(observation=phone_obs, action=joint_action)

            # 控制循环频率
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\n正在断开连接...")
    finally:
        robot.disconnect()
        teleop_device.disconnect()
        print("已安全退出")


if __name__ == "__main__":
    main()
