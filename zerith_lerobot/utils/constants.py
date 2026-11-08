"""Zerith H1 Pro data, SDK and URDF ordering (not EtherCAT enum ordering)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SDK_ROOT = PROJECT_ROOT / "zerith_H1_SDK1.3.6"
URDF_PATH = SDK_ROOT / "ZR_H1PRO-1.2.00.H.V4.10_URDF_2026.08.12/urdf/ZR_H1PRO-1.2.00.H.V4.10_URDF_2026.08.12.urdf"
ARM_JOINT_SUFFIXES = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                      "wrist_roll", "wrist_yaw", "wrist_pitch"]
# Gripper values are SDK motor positions, NOT the prismatic finger positions in URDF.
URDF_JOINTS = [f"left_{name}_joint" for name in ARM_JOINT_SUFFIXES] + [None] + [
    f"right_{name}_joint" for name in ARM_JOINT_SUFFIXES
] + [None, "daogui_joint", "body_pitch_joint", "body_yaw_joint", "neck_yaw_joint",
     "neck_pitch_joint", None, None]
SDK_MOTORS = [f"MOTOR_LEFT_ARM_{i}" for i in range(1, 9)] + [
    f"MOTOR_RIGHT_ARM_{i}" for i in range(1, 9)
] + ["MOTOR_LIFT", "MOTOR_WAIST_DOWN", "MOTOR_WAIST_UP", "MOTOR_HEAD_DOWN", "MOTOR_HEAD_UP"]
CAMERA_KEYS = ["observation.images.cam_high", "observation.images.cam_left_wrist",
               "observation.images.cam_right_wrist"]
