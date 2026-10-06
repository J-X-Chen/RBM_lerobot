import dataclasses


@dataclasses.dataclass(frozen=True)
class RobotConfig:
    motors: list[str]
    cameras: list[str]
    camera_to_image_key: dict[str, str]
    json_state_data_name: list[str]
    json_action_data_name: list[str]


ASTRIBOT_S1_CONFIG = RobotConfig(
    motors=[
        "astribot_chassis_x",
        "astribot_chassis_y",
        "astribot_chassis_yaw",
        "astribot_torso_joint_1",
        "astribot_torso_joint_2",
        "astribot_torso_joint_3",
        "astribot_torso_joint_4",
        "astribot_arm_left_joint_1",
        "astribot_arm_left_joint_2",
        "astribot_arm_left_joint_3",
        "astribot_arm_left_joint_4",
        "astribot_arm_left_joint_5",
        "astribot_arm_left_joint_6",
        "astribot_arm_left_joint_7",
        "astribot_gripper_left",
        "astribot_arm_right_joint_1",
        "astribot_arm_right_joint_2",
        "astribot_arm_right_joint_3",
        "astribot_arm_right_joint_4",
        "astribot_arm_right_joint_5",
        "astribot_arm_right_joint_6",
        "astribot_arm_right_joint_7",
        "astribot_gripper_right",
        "astribot_head_joint_1",
        "astribot_head_joint_2",
    ],
    cameras=[
        "head",
        "left",
        "right",
    ],
    camera_to_image_key={
        "color_0": "head",
        "color_1": "left",
        "color_2": "right",
    },
    json_state_data_name=[
        "chassis.qpos",
        "torso.qpos",
        "left_arm.qpos",
        "left_ee.qpos",
        "right_arm.qpos",
        "right_ee.qpos",
        "head.qpos",
    ],
    json_action_data_name=[
        "chassis.qpos",
        "torso.qpos",
        "left_arm.qpos",
        "left_ee.qpos",
        "right_arm.qpos",
        "right_ee.qpos",
        "head.qpos",
    ],
)


ROBOT_CONFIGS = {
    "Astribot_S1": ASTRIBOT_S1_CONFIG,
    "astribot_s1": ASTRIBOT_S1_CONFIG,
}
