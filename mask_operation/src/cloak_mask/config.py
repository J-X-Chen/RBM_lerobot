"""Explicit calibration schema and refusal to invent missing physical parameters."""
import json
from pathlib import Path

import numpy as np

CAMERAS = ('head', 'left', 'right')
CHASSIS = ('astribot_chassis_x', 'astribot_chassis_y', 'astribot_chassis_yaw')


def s1_state_order():
    return [*CHASSIS, *[f'astribot_torso_joint_{i}' for i in range(1, 5)],
            *[f'astribot_arm_left_joint_{i}' for i in range(1, 8)], 'astribot_gripper_left',
            *[f'astribot_arm_right_joint_{i}' for i in range(1, 8)], 'astribot_gripper_right',
            'astribot_head_joint_1', 'astribot_head_joint_2']


def template(info, model):
    return {
        'schema_version': 1,
        'state': {
            'key': 'observation.state', 'confirmed_measured_feedback': False,
            'recording_source': None,
            'joint_names': s1_state_order(),
            'mapping_note': 'CANDIDATE from RBM_lerobot/astribot_lerobot/utils/constants.py; '
                            'state_0..24 alone does not establish recording provenance. Confirm before rendering.',
            'units': 'revolute_rad; chassis_xy_m; gripper_sdk_feedback',
        },
        'grippers': {side: {
            'feedback_name': f'astribot_gripper_{side}', 'source': None,
            'joint_names': [n for n in model.moving_joints if n.startswith(f'astribot_gripper_{side}_')],
            'samples': None,
            'samples_format': [{'feedback': 'measured scalar', 'angles_rad': 'one value per joint_names entry'}],
        } for side in ('left', 'right')},
        'cameras': {name: {
            'feature': f'observation.images.{name}',
            'width': info['features'][f'observation.images.{name}']['shape'][1],
            'height': info['features'][f'observation.images.{name}']['shape'][0],
            'parent_link': None, 'T_parent_optical': None, 'K': None,
            'distortion_model': None, 'distortion_coefficients': None,
            'rgb_time_offset_seconds': None, 'source': None,
            'confirmed_recording_setup': False,
        } for name in CAMERAS},
        'conventions': {
            'T_parent_optical': '4x4 maps OpenCV optical coordinates into parent link; metres; x right y down z forward',
            'K': '3x3 pixel intrinsics for the STORED RGB resolution/crop; fx/fy need not match; skew must be zero',
            'distortion': 'none explicitly for rectified/pinhole RGB; opencv for 4/5/8 Brown-Conrady coefficients',
            'rgb_time_offset_seconds': 'video timestamp = episode video start + row timestamp + this offset',
            'grippers': 'measured lookup samples covering observed feedback range; no implicit 0..100 to radians mapping',
            'root': 'torso base local coordinates; chassis common rigid motion cancels for robot-mounted cameras',
        },
    }


def calibration_errors(config, info, model):
    errors = []
    if set(config.get('cameras', {})) != set(CAMERAS):
        errors.append('cameras must contain exactly head, left and right')
    if config.get('schema_version') != 1:
        errors.append('schema_version must be 1')
    state = config.get('state', {})
    if state.get('key') != 'observation.state':
        errors.append('state.key must be observation.state, never action')
    if state.get('confirmed_measured_feedback') is not True or not state.get('recording_source'):
        errors.append('state: confirm measured feedback and document the recording/conversion source')
    names = state.get('joint_names', [])
    size = info['features']['observation.state']['shape'][0]
    if len(names) != size or len(set(names)) != len(names):
        errors.append(f'state.joint_names must give {size} unique measured component names')
    required = {n for n in model.moving_joints if not n.startswith('astribot_gripper_')}
    required |= {'astribot_gripper_left', 'astribot_gripper_right'}
    if required - set(names):
        errors.append(f'state mapping missing: {sorted(required-set(names))}')
    if set(names) - required - set(CHASSIS):
        errors.append(f'Unrecognized state components: {sorted(set(names)-required-set(CHASSIS))}')
    if state.get('units') != 'revolute_rad; chassis_xy_m; gripper_sdk_feedback':
        errors.append('state.units must explicitly match the supported SDK units; convert other units first')
    for side in ('left', 'right'):
        cfg = config.get('grippers', {}).get(side, {})
        expected = {n for n in model.moving_joints if n.startswith(f'astribot_gripper_{side}_')}
        joint_names = cfg.get('joint_names', [])
        if set(joint_names) != expected or len(joint_names) != len(expected):
            errors.append(f'grippers.{side}.joint_names must cover exactly {sorted(expected)}')
        if cfg.get('feedback_name') != f'astribot_gripper_{side}':
            errors.append(f'grippers.{side}.feedback_name does not match measured state component')
        if not cfg.get('source'):
            errors.append(f'grippers.{side}: missing measured linkage/mapping provenance')
        try:
            samples = cfg.get('samples')
            x = np.array([s['feedback'] for s in samples], float)
            y = np.array([s['angles_rad'] for s in samples], float)
            if len(x) < 2 or y.shape != (len(x), len(expected)) or not np.isfinite(x).all() or not np.isfinite(y).all() or not (np.diff(x) > 0).all():
                raise ValueError()
        except (TypeError, ValueError, KeyError):
            errors.append(f'grippers.{side}.samples: need >=2 increasing feedback samples with all finger angles in radians')
    for camera in CAMERAS:
        cfg = config.get('cameras', {}).get(camera, {})
        prefix = f'cameras.{camera}'
        ft = info['features'].get(cfg.get('feature'), {})
        if ft.get('dtype') != 'video' or ft.get('shape', [])[:2] != [cfg.get('height'), cfg.get('width')]:
            errors.append(f'{prefix}: feature/resolution must match dataset RGB video')
        if cfg.get('parent_link') not in model.links:
            errors.append(f'{prefix}.parent_link: missing or unknown measured mounting link')
        if not cfg.get('source'):
            errors.append(f'{prefix}.source: missing calibration provenance')
        if cfg.get('confirmed_recording_setup') is not True:
            errors.append(f'{prefix}: confirm calibration matches the recorded camera, crop, resolution and raw/rectified RGB')
        try:
            k = np.asarray(cfg.get('K'), float)
            if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]) or not np.allclose([k[0, 1], k[1, 0]], 0):
                raise ValueError()
        except (ValueError, TypeError):
            errors.append(f'{prefix}.K: missing/invalid pinhole intrinsics')
        try:
            t = np.asarray(cfg.get('T_parent_optical'), float)
            if t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1]) or not np.allclose(t[:3,:3].T @ t[:3,:3], np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(t[:3,:3]), 1, atol=1e-6):
                raise ValueError()
        except (ValueError, TypeError):
            errors.append(f'{prefix}.T_parent_optical: missing/invalid rigid optical-to-link transform')
        mode, coefficients = cfg.get('distortion_model'), cfg.get('distortion_coefficients')
        try:
            d = np.asarray(coefficients, float)
            if d.ndim != 1 or not np.isfinite(d).all() or not (
                (mode == 'none' and len(d) == 0) or (mode == 'opencv' and len(d) in (4, 5, 8))):
                raise ValueError()
        except (ValueError, TypeError):
            errors.append(f'{prefix}: specify distortion_model and coefficients explicitly')
        if not isinstance(cfg.get('rgb_time_offset_seconds'), (float, int)) or not np.isfinite(cfg['rgb_time_offset_seconds']):
            errors.append(f'{prefix}.rgb_time_offset_seconds: missing explicit synchronization offset')
    return errors


def feedback_positions(state, config):
    values = np.asarray(state, float)
    names = config['state']['joint_names']
    if values.shape != (len(names),) or not np.isfinite(values).all():
        raise ValueError('Invalid measured feedback vector')
    positions = dict(zip(names, values, strict=True))
    for side in ('left', 'right'):
        cfg = config['grippers'][side]
        samples = cfg['samples']
        x = np.asarray([s['feedback'] for s in samples], float)
        y = np.asarray([s['angles_rad'] for s in samples], float)
        value = positions[cfg['feedback_name']]
        if not x[0] <= value <= x[-1]:
            raise ValueError(f'{side} gripper feedback {value} outside calibrated [{x[0]}, {x[-1]}]; no clamping/extrapolation')
        for i, joint in enumerate(cfg['joint_names']):
            positions[joint] = float(np.interp(value, x, y[:, i]))
    return positions


def load_config(path):
    return json.loads(Path(path).read_text())
