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
SO101 专用的手机动作映射处理器

SO101 的 URDF 坐标系与 SO100 不同，需要调整坐标映射：
- SO100: 使用混合轴定义 (X, Y 轴)
- SO101: 所有关节轴使用 Z 轴，且 base 坐标系有 180° 旋转

此处理器适配 SO101 的坐标系定义。
"""

from dataclasses import dataclass, field

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotAction, RobotActionProcessorStep
from lerobot.teleoperators.phone.config_phone import PhoneOS


@ProcessorStepRegistry.register("map_phone_action_to_robot_action_so101")
@dataclass
class MapPhoneActionToRobotActionSO101(RobotActionProcessorStep):
    """
    Maps calibrated phone pose actions to standardized robot action inputs for SO101.

    This processor step acts as a bridge between the phone teleoperator's output
    and the SO101 robot's expected action format. It remaps the phone's 6-DoF pose
    (position and rotation) to the robot's target end-effector pose, applying
    necessary axis inversions and swaps specific to SO101's URDF coordinate system.

    SO101 URDF 坐标系说明:
    - shoulder_pan 关节: rpy="3.14159 ... -3.14159" 表示绕 X 和 Z 轴各旋转 180°
    - 所有关节轴: axis="0 0 1" (Z 轴)
    - gripper_frame_link: 有额外的 180° 旋转

    相比 SO100 的映射调整:
    - Z 轴方向反转 (向上抬手机 → 机械臂向上)
    - X/Y 轴映射可能需要调整

    Attributes:
        platform: The operating system of the phone (iOS or Android), used
            to determine the correct button mappings for the gripper.
    """

    platform: PhoneOS
    _enabled_prev: bool = field(default=False, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        """
        Processes the phone action dictionary to create a robot action dictionary.

        Args:
            action: The input action dictionary from the phone teleoperator.

        Returns:
            A new action dictionary formatted for the SO101 robot controller.

        Raises:
            ValueError: If 'pos' or 'rot' keys are missing from the input action.
        """
        # Pop them from the action
        enabled = bool(action.pop("phone.enabled"))
        pos = action.pop("phone.pos")
        rot = action.pop("phone.rot")
        inputs = action.pop("phone.raw_inputs")

        if pos is None or rot is None:
            raise ValueError("pos and rot must be present in action")

        rotvec = rot.as_rotvec()  # Absolute orientation as rotvec

        # Map certain inputs to certain actions
        if self.platform == PhoneOS.IOS:
            gripper_vel = float(inputs.get("a3", 0.0))
        else:
            a = float(inputs.get("reservedButtonA", 0.0))
            b = float(inputs.get("reservedButtonB", 0.0))
            gripper_vel = (
                a - b
            )  # Positive if a is pressed, negative if b is pressed, 0 if both or neither are pressed

        # SO101 坐标系映射 (与 SO100 不同)
        # 根据 SO101 URDF 的坐标系定义调整映射
        #
        # 手机坐标系 (校准后):
        #   - pos[0]: 手机 X 方向 (左右)
        #   - pos[1]: 手机 Y 方向 (前后)
        #   - pos[2]: 手机 Z 方向 (上下)
        #
        # SO101 机械臂坐标系:
        #   - X: 机械臂前方 (伸展方向)
        #   - Y: 机械臂左侧
        #   - Z: 向上
        #
        # 映射关系 (需要根据实际测试调整):
        #   - 手机向前 (pos[1]+) → 机械臂向前 (target_x+)
        #   - 手机向左 (pos[0]-) → 机械臂向左 (target_y+)
        #   - 手机向上 (pos[2]+) → 机械臂向上 (target_z+)
        #
        # 注意: 以下映射基于 SO101 URDF 分析，可能需要根据实际测试微调
        action["enabled"] = enabled
        action["target_x"] = pos[1] if enabled else 0.0    # 手机 Y → 机械臂 X (前后)
        action["target_y"] = -pos[0] if enabled else 0.0   # 手机 X → 机械臂 -Y (左右反转)
        action["target_z"] = pos[2] if enabled else 0.0    # 手机 Z → 机械臂 Z (上下)
        action["target_wx"] = rotvec[1] if enabled else 0.0
        action["target_wy"] = -rotvec[0] if enabled else 0.0
        action["target_wz"] = rotvec[2] if enabled else 0.0
        action["gripper_vel"] = gripper_vel  # Still send gripper action when disabled
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["enabled", "pos", "rot", "raw_inputs"]:
            features[PipelineFeatureType.ACTION].pop(f"phone.{feat}", None)

        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION][f"{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features
