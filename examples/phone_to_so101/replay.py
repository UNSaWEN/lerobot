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
SO101 数据集回放示例

此脚本回放已录制的数据集，让 SO101 机械臂重现录制的动作。
"""

import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    InverseKinematicsEEToJoints,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

# ============== 配置参数 ==============

# 回放参数
EPISODE_IDX = 0
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"

# 机器人配置
ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "R_follower_arm"

# URDF 文件路径
URDF_PATH = "./SO101/so101_new_calib.urdf"

# ======================================


def main():
    # 创建机器人配置
    robot_config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        use_degrees=True
    )

    # 初始化机器人
    robot = SO101Follower(robot_config)

    # 创建运动学求解器
    kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # 构建管道: 末端执行器动作 -> 关节动作
    # 注意: 回放使用开环控制 (initial_guess_current_joints=False)
    robot_ee_to_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=list(robot.bus.motors.keys()),
                initial_guess_current_joints=False,  # 开环控制
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # 加载数据集
    dataset = LeRobotDataset(HF_REPO_ID, episodes=[EPISODE_IDX])
    # 过滤出指定 episode 的帧
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == EPISODE_IDX)
    actions = episode_frames.select_columns(ACTION)

    # 连接机器人
    robot.connect()

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    print("=" * 60)
    print("SO101 数据回放已启动")
    print("=" * 60)
    print(f"回放 episode: {EPISODE_IDX}")
    print(f"帧数: {len(episode_frames)}")
    print(f"FPS: {dataset.fps}")
    print("=" * 60)

    log_say(f"Replaying episode {EPISODE_IDX}")

    try:
        for idx in range(len(episode_frames)):
            t0 = time.perf_counter()

            # 从数据集获取录制的动作
            ee_action = {
                name: float(actions[idx][ACTION][i])
                for i, name in enumerate(dataset.features[ACTION]["names"])
            }

            # 获取机器人当前状态
            robot_obs = robot.get_observation()

            # 末端执行器动作 -> 关节动作
            joint_action = robot_ee_to_joints_processor((ee_action, robot_obs))

            # 发送动作到机器人
            _ = robot.send_action(joint_action)

            # 控制循环频率
            precise_sleep(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\n回放被中断")
    finally:
        robot.disconnect()
        print("回放完成")


if __name__ == "__main__":
    main()
