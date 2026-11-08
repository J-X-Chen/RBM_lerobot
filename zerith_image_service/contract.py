"""Wire contract v1: measured state 23D, commanded arms/grippers/waist 19D."""
from pathlib import Path
import json
import socket
import xml.etree.ElementTree as ET

import numpy as np

PROTOCOL = 'zerith_h1_v1'
CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
NAMES = [f'left_arm_{i}' for i in range(1, 8)] + ['left_gripper'] + [
    f'right_arm_{i}' for i in range(1, 8)
] + ['right_gripper', 'waist_lift', 'waist_pitch', 'waist_yaw', 'head_yaw',
     'head_pitch', 'base_linear_speed', 'base_angular_speed']
MOTORS = [f'MOTOR_LEFT_ARM_{i}' for i in range(1, 9)] + [
    f'MOTOR_RIGHT_ARM_{i}' for i in range(1, 9)
] + ['MOTOR_LIFT', 'MOTOR_WAIST_DOWN', 'MOTOR_WAIST_UP', 'MOTOR_HEAD_DOWN', 'MOTOR_HEAD_UP']
ARM = list(range(7)) + list(range(8, 15))
GRIPPERS = [7, 15]
BODY = ARM + [16, 17, 18]
SUFFIXES = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_yaw', 'wrist_pitch']
JOINTS = [f'left_{s}_joint' for s in SUFFIXES] + [None] + [
    f'right_{s}_joint' for s in SUFFIXES
] + [None, 'daogui_joint', 'body_pitch_joint', 'body_yaw_joint', 'neck_yaw_joint', 'neck_pitch_joint']


def vector(value, size):
    out = np.asarray(value, dtype=float)
    if out.shape != (size,) or not np.isfinite(out).all():
        raise ValueError(f'Expected finite {size}D vector')
    return out.copy()


def limits_from_urdf(path):
    joints = {j.get('name'): j for j in ET.parse(path).findall('joint')}
    # SDK gripper motor coordinates; URDF finger translations are not motor coordinates.
    low, high = np.zeros(21), np.full(21, 1.5)
    for i, name in enumerate(JOINTS):
        if name:
            limit = joints[name].find('limit')
            low[i], high[i] = float(limit.get('lower')), float(limit.get('upper'))
    return low, high


def default_urdf(sdk_root):
    matches = sorted(Path(sdk_root).glob('*/urdf/*.urdf'))
    if len(matches) != 1:
        raise ValueError('Specify --urdf: SDK must contain exactly one URDF')
    return matches[0]


def send_json(sock, value):
    data = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
    if len(data) > 2_000_000:
        raise ValueError('Control message too large')
    sock.sendall(f'{len(data):010d}'.encode() + data)


def recv_json(sock, timeout=2.0):
    sock.settimeout(timeout)
    def read(count):
        result = bytearray()
        while len(result) < count:
            part = sock.recv(count - len(result))
            if not part:
                raise ConnectionError('Peer disconnected')
            result.extend(part)
        return bytes(result)
    header = read(10)
    if not header.isdigit() or not 0 < int(header) <= 2_000_000:
        raise ValueError('Invalid frame size')
    value = json.loads(read(int(header)))
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object')
    return value


class ControlClient:
    def __init__(self, host, port=6200, timeout=2.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.timeout = timeout

    def request(self, **payload):
        send_json(self.sock, dict(payload, protocol=PROTOCOL))
        reply = recv_json(self.sock, self.timeout)
        if reply.get('protocol') != PROTOCOL or not reply.get('ok'):
            raise RuntimeError(f'H1 service rejected request: {reply}')
        return reply

    def close(self):
        self.sock.close()
