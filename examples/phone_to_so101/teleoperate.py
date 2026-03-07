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

import sys
import time
from pathlib import Path

# Allow importing so101_phone_processor when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

from so101_phone_processor import MapPhoneActionToRobotActionSO101

# Config: change port, id, and phone_os to match your setup
ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "so101_follower"
PHONE_OS = PhoneOS.IOS  # or PhoneOS.ANDROID
URDF_PATH = "./SO101/so101_new_calib.urdf"
FPS = 30
EE_STEP_SIZES = {"x": 0.3, "y": 0.3, "z": 0.3}
EE_BOUNDS = {"min": [-0.3, -0.3, 0.02], "max": [0.3, 0.3, 0.4]}
MAX_EE_STEP_M = 0.05
GRIPPER_SPEED_FACTOR = 20.0


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

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo:
    # https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # Build pipeline to convert phone action to ee pose action to joint action (SO101 coordinate mapping)
    phone_to_robot_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotActionSO101(platform=teleop_config.phone_os),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes=EE_STEP_SIZES,
                motor_names=list(robot.bus.motors.keys()),
                use_latched_reference=True,
            ),

            EEBoundsAndSafety(
                end_effector_bounds=EE_BOUNDS,
                max_ee_step_m=MAX_EE_STEP_M,
            ),
            GripperVelocityToJoint(speed_factor=GRIPPER_SPEED_FACTOR),
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

    print("Starting teleop loop. Move your phone to teleoperate the SO101 robot...")
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

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    finally:
        robot.disconnect()
        teleop_device.disconnect()


if __name__ == "__main__":
    main()
