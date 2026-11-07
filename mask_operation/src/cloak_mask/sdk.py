"""Read-only adapter for an already connected local Astribot SDK instance.

No SDK initialization, control-right acquisition, camera activation or motion is
performed here. Host read times are NOT hardware timestamps; live RGB sync must
be supplied by the caller. Offline dataset rendering uses recorded feedback.
"""
import time
import numpy as np

GROUPS = {
    'astribot_chassis': ['astribot_chassis_x', 'astribot_chassis_y', 'astribot_chassis_yaw'],
    'astribot_torso': [f'astribot_torso_joint_{i}' for i in range(1,5)],
    'astribot_arm_left': [f'astribot_arm_left_joint_{i}' for i in range(1,8)],
    'astribot_gripper_left': ['astribot_gripper_left'],
    'astribot_arm_right': [f'astribot_arm_right_joint_{i}' for i in range(1,8)],
    'astribot_gripper_right': ['astribot_gripper_right'],
    'astribot_head': ['astribot_head_joint_1', 'astribot_head_joint_2'],
}


def read_current_feedback(sdk, joint_names):
    wanted = set(joint_names)
    available = {joint for joints in GROUPS.values() for joint in joints}
    if wanted-available:
        raise ValueError(f'Unknown SDK joint names: {wanted-available}')
    groups = [name for name, joints in GROUPS.items() if wanted.intersection(joints)]
    started = time.monotonic()
    values = sdk.get_current_joints_position(groups)
    finished = time.monotonic()
    if values is None or len(values) != len(groups):
        raise ValueError('SDK feedback group count mismatch')
    positions = {}
    for group, group_values in zip(groups, values, strict=True):
        q = np.asarray(group_values, dtype=float)
        if q.shape != (len(GROUPS[group]),) or not np.isfinite(q).all():
            raise ValueError(f'Invalid SDK feedback for {group}: {q}')
        positions.update(zip(GROUPS[group], q, strict=True))
    return np.array([positions[name] for name in joint_names]), {
        'host_read_start_monotonic': started, 'host_read_end_monotonic': finished,
        'hardware_timestamp': None,
    }
