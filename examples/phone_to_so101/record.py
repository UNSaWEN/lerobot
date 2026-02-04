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
SO101 手机遥操作数据录制示例

此脚本使用手机遥操作 SO101 机械臂并录制数据集。
录制的数据可用于训练模仿学习策略。
"""

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEE,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# 导入 SO101 专用的手机动作映射处理器
from so101_phone_processor import MapPhoneActionToRobotActionSO101

# ============== 配置参数 ==============

# 录制参数
NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 30
TASK_DESCRIPTION = "My task description"
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"

# 机器人配置
ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "R_follower_arm"

# 相机配置 (索引 0 通常是默认相机)
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# 手机系统
PHONE_OS = PhoneOS.IOS  # 或 PhoneOS.ANDROID

# URDF 文件路径
URDF_PATH = "./SO101/so101_new_calib.urdf"

# 末端执行器配置
EE_STEP_SIZES = {"x": 0.3, "y": 0.3, "z": 0.3}
EE_BOUNDS = {
    "min": [-0.3, -0.3, 0.02],
    "max": [0.3, 0.3, 0.4]
}
MAX_EE_STEP_M = 0.10

# ======================================


def main():
    # 创建相机配置
    camera_config = {
        "front": OpenCVCameraConfig(
            index_or_path=CAMERA_INDEX,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=FPS
        )
    }

    # 创建机器人配置
    robot_config = SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        cameras=camera_config,
        use_degrees=True,
    )

    # 创建手机遥操作配置
    teleop_config = PhoneConfig(phone_os=PHONE_OS)

    # 初始化机器人和遥操作设备
    robot = SO101Follower(robot_config)
    phone = Phone(teleop_config)

    # 创建运动学求解器
    kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # 构建管道: 手机动作 -> 末端执行器动作
    phone_to_robot_ee_pose_processor = RobotProcessorPipeline[
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
            GripperVelocityToJoint(speed_factor=20.0),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # 构建管道: 末端执行器动作 -> 关节动作
    robot_ee_to_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=list(robot.bus.motors.keys()),
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # 构建管道: 关节观测 -> 末端执行器观测
    robot_joints_to_ee_pose = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=kinematics_solver,
                motor_names=list(robot.bus.motors.keys())
            )
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # 创建数据集
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=phone_to_robot_ee_pose_processor,
                initial_features=create_initial_features(action=phone.action_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_joints_to_ee_pose,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=True,
            ),
        ),
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # 连接机器人和遥操作设备
    robot.connect()
    phone.connect()

    # 初始化键盘监听和可视化
    listener, events = init_keyboard_listener()
    init_rerun(session_name="phone_so101_record")

    if not robot.is_connected or not phone.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("=" * 60)
    print("SO101 数据录制已启动")
    print("=" * 60)
    print(f"录制 {NUM_EPISODES} 个 episode，每个 {EPISODE_TIME_SEC} 秒")
    print(f"任务描述: {TASK_DESCRIPTION}")
    print("-" * 60)
    print("快捷键:")
    print("  - 空格: 提前结束当前 episode")
    print("  - R: 重新录制当前 episode")
    print("  - Q: 停止录制")
    print("=" * 60)

    episode_idx = 0
    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

        # 主录制循环
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=phone,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=phone_to_robot_ee_pose_processor,
            robot_action_processor=robot_ee_to_joints_processor,
            robot_observation_processor=robot_joints_to_ee_pose,
        )

        # 重置环境
        if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
            log_say("Reset the environment")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=phone,
                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
                teleop_action_processor=phone_to_robot_ee_pose_processor,
                robot_action_processor=robot_ee_to_joints_processor,
                robot_observation_processor=robot_joints_to_ee_pose,
            )

        if events["rerecord_episode"]:
            log_say("Re-recording episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        # 保存 episode
        dataset.save_episode()
        episode_idx += 1

    # 清理
    log_say("Stop recording")
    robot.disconnect()
    phone.disconnect()
    listener.stop()

    dataset.finalize()
    dataset.push_to_hub()


if __name__ == "__main__":
    main()
