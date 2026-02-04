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
SO101 Leader to SO101 Follower Teleoperation (End-Effector Space)

使用 SO101 主臂遥操作 SO101 从臂，通过末端执行器空间进行控制。
主臂的关节位置通过正向运动学转换为末端执行器姿态，
然后通过逆向运动学转换为从臂的关节位置。

用法:
    python examples/so101_to_so101_EE/teleoperate.py

配置说明:
    - FOLLOWER_PORT: 从臂串口路径
    - LEADER_PORT: 主臂串口路径
    - URDF_PATH: SO101 URDF 文件路径
    - FPS: 控制频率
"""

import time

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    robot_action_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ==================== 配置参数 ====================
# 串口配置 (根据实际硬件修改)
FOLLOWER_PORT = "/dev/ttyACM0"  # SO101 从臂串口
LEADER_PORT = "/dev/ttyACM1"    # SO101 主臂串口

# 机器人 ID
FOLLOWER_ID = "so101_follower"
LEADER_ID = "so101_leader"

# URDF 路径
URDF_PATH = "./SO101/so101_new_calib.urdf"

# 控制参数
FPS = 30

# 安全限制
EE_BOUNDS_MIN = [-1.0, -1.0, -1.0]  # 末端执行器位置下限 (米)
EE_BOUNDS_MAX = [1.0, 1.0, 1.0]     # 末端执行器位置上限 (米)
MAX_EE_STEP = 0.10                   # 末端执行器最大步进 (米)
# =================================================


def main():
    # Initialize the robot and teleoperator config
    follower_config = SO101FollowerConfig(
        port=FOLLOWER_PORT, id=FOLLOWER_ID, use_degrees=True
    )
    leader_config = SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID)

    # Initialize the robot and teleoperator
    follower = SO101Follower(follower_config)
    leader = SO101Leader(leader_config)

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo:
    # https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    follower_kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )

    leader_kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=list(leader.bus.motors.keys()),
    )

    # Build pipeline to convert teleop joints to EE action
    leader_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=leader_kinematics_solver, motor_names=list(leader.bus.motors.keys())
            ),
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )

    # build pipeline to convert EE action to robot joints
    ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        [
            EEBoundsAndSafety(
                end_effector_bounds={"min": EE_BOUNDS_MIN, "max": EE_BOUNDS_MAX},
                max_ee_step_m=MAX_EE_STEP,
            ),
            InverseKinematicsEEToJoints(
                kinematics=follower_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
                initial_guess_current_joints=False,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect to the robot and teleoperator
    follower.connect()
    leader.connect()

    # Init rerun viewer
    init_rerun(session_name="so101_so101_EE_teleop")

    if not follower.is_connected or not leader.is_connected:
        raise ValueError("Follower or leader is not connected!")

    print("Starting teleop loop...")
    print(f"  Follower: {FOLLOWER_PORT} ({FOLLOWER_ID})")
    print(f"  Leader: {LEADER_PORT} ({LEADER_ID})")
    print("  Press Ctrl+C to stop.")

    try:
        while True:
            t0 = time.perf_counter()

            # Get robot observation
            robot_obs = follower.get_observation()

            # Get teleop observation
            leader_joints_obs = leader.get_action()

            # teleop joints -> teleop EE action
            leader_ee_act = leader_to_ee(leader_joints_obs)

            # teleop EE -> robot joints
            follower_joints_act = ee_to_follower_joints((leader_ee_act, robot_obs))

            # Send action to robot
            _ = follower.send_action(follower_joints_act)

            # Visualize
            log_rerun_data(observation=leader_ee_act, action=follower_joints_act)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\nStopping teleop...")
    finally:
        follower.disconnect()
        leader.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
