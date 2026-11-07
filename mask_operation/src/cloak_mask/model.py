"""Visual-only Astribot SDK model import and deterministic kinematics.

The SDK URDF supplies torso, head and arms. Its SDF supplies the two grippers,
including the base visuals baked into arm link 7. We retain every visual mesh,
not collision proxies. Bodies are fixed in MJCF and their relative transforms
are updated from measured joints before mj_forward: no physics stepping,
synthetic inertias, inferred finger coupling, or IK is involved.
"""
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    result[:3, 3] = xyz
    return result


def numbers(text, default):
    return np.asarray([float(x) for x in text.split()] if text else default, dtype=float)


def urdf_pose(element):
    if element is None:
        return np.eye(4)
    return transform(numbers(element.get('xyz'), [0]*3), numbers(element.get('rpy'), [0]*3))


def sdf_pose(element, expected_frame):
    if element is None:
        return np.eye(4)
    relative = element.get('relative_to', expected_frame)
    if relative != expected_frame:
        raise ValueError(f'Unsupported SDF pose frame {relative!r}, expected {expected_frame!r}')
    values = numbers(element.text, [0]*6)
    if values.shape != (6,):
        raise ValueError('SDF pose must contain xyz roll pitch yaw')
    return transform(values[:3], values[3:])


def quaternion(matrix):
    return np.roll(Rotation.from_matrix(matrix[:3, :3]).as_quat(), 1)


def fmt(values):
    return ' '.join(f'{x:.15g}' for x in values)


@dataclass
class Visual:
    mesh: Path
    pose: np.ndarray
    scale: np.ndarray
    group: str


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    kind: str
    pose: np.ndarray
    axis: np.ndarray
    child_pose: np.ndarray = field(default_factory=lambda: np.eye(4))

    def relative(self, value):
        motion = np.eye(4)
        if self.kind in ('revolute', 'continuous'):
            motion[:3, :3] = Rotation.from_rotvec(self.axis * value).as_matrix()
        elif self.kind == 'prismatic':
            motion[:3, 3] = self.axis * value
        elif self.kind != 'fixed':
            raise ValueError(f'Unsupported joint type {self.kind}')
        return self.pose @ motion @ self.child_pose


class AstribotVisualModel:
    def __init__(self, sdk_root):
        self.sdk_root = Path(sdk_root).resolve()
        self.directory = self.sdk_root / 'astribot_config/robot_config/astribot_s1/model'
        self.urdf = self.directory / 'astribot_whole_body_with_head.urdf'
        self.sdf = self.directory / 'astribot_whole_body_with_gripper.sdf'
        self.links = {}
        self.joints = {}
        self.mesh_sources = set()
        self._read_urdf()
        self._append_grippers()
        children = {j.child for j in self.joints.values()}
        roots = set(self.links) - children
        if len(roots) != 1:
            raise ValueError(f'Model must have one root, got {roots}')
        self.root = roots.pop()
        if len(children) != len(self.joints):
            raise ValueError('Multiple parent joints for one link')

    def _resolve(self, reference):
        candidates = [self.directory / reference, self.directory / 'meshes' / reference,
                      self.directory.parent / 'meshes' / reference]
        for path in candidates:
            if path.is_file():
                path = path.resolve()
                self.mesh_sources.add(path)
                return path
        raise FileNotFoundError(f'SDK visual mesh not found: {reference}; searched {candidates}')

    @staticmethod
    def group(name):
        if 'gripper_left' in name:
            return 'left_gripper'
        if 'gripper_right' in name:
            return 'right_gripper'
        if 'arm_left' in name:
            return 'left_arm'
        if 'arm_right' in name:
            return 'right_arm'
        if 'head' in name:
            return 'head'
        return 'torso'

    def _read_urdf(self):
        root = ET.parse(self.urdf).getroot()
        for link in root.findall('link'):
            name = link.attrib['name']
            self.links[name] = []
            for visual in link.findall('visual'):
                mesh = visual.find('geometry/mesh')
                if mesh is None:
                    raise ValueError(f'Unsupported non-mesh SDK visual: {name}')
                self.links[name].append(Visual(self._resolve(mesh.attrib['filename']),
                    urdf_pose(visual.find('origin')), numbers(mesh.get('scale'), [1]*3), self.group(name)))
        for joint in root.findall('joint'):
            if joint.find('mimic') is not None:
                raise ValueError('URDF mimic must be explicitly implemented before use')
            axis = joint.find('axis')
            axis = numbers(axis.get('xyz') if axis is not None else None, [1, 0, 0])
            axis /= np.linalg.norm(axis)
            j = Joint(joint.attrib['name'], joint.find('parent').attrib['link'],
                      joint.find('child').attrib['link'], joint.attrib['type'],
                      urdf_pose(joint.find('origin')), axis)
            self.joints[j.name] = j

    def _append_grippers(self):
        model = ET.parse(self.sdf).getroot().find('model')
        for link in model.findall('link'):
            name = link.attrib['name']
            is_finger = name.startswith('astribot_gripper_')
            if is_finger:
                self.links[name] = []
            for visual in link.findall('visual'):
                if not is_finger and 'gripper_' not in visual.attrib['name']:
                    continue
                if name not in self.links:
                    raise ValueError(f'Gripper attachment link missing from URDF: {name}')
                mesh = visual.find('geometry/mesh')
                if mesh is None:
                    raise ValueError('Unsupported non-mesh gripper visual')
                group = 'left_gripper' if 'left' in name else 'right_gripper'
                self.links[name].append(Visual(self._resolve(mesh.findtext('uri')),
                    sdf_pose(visual.find('pose'), name), numbers(mesh.findtext('scale'), [1]*3), group))
        for joint in model.findall('joint'):
            name = joint.attrib['name']
            if not name.startswith('astribot_gripper_'):
                continue
            parent, child = joint.findtext('parent'), joint.findtext('child')
            child_element = next(x for x in model.findall('link') if x.get('name') == child)
            axis_el = joint.find('axis/xyz')
            if axis_el.get('expressed_in'):
                raise ValueError('Unsupported SDF axis frame')
            axis = numbers(axis_el.text, [1, 0, 0])
            axis /= np.linalg.norm(axis)
            self.joints[name] = Joint(name, parent, child, joint.attrib['type'],
                sdf_pose(joint.find('pose'), parent), axis, sdf_pose(child_element.find('pose'), name))

    @property
    def moving_joints(self):
        return [name for name, joint in self.joints.items() if joint.kind != 'fixed']

    def mjcf(self, width=1280, height=720):
        root = ET.Element('mujoco', model='astribot_s1_visual_fk')
        ET.SubElement(root, 'compiler', angle='radian', fusestatic='false')
        visual = ET.SubElement(root, 'visual')
        ET.SubElement(visual, 'global', offwidth=str(width), offheight=str(height))
        ET.SubElement(visual, 'map', znear='0.001')
        asset, world = ET.SubElement(root, 'asset'), ET.SubElement(root, 'worldbody')
        self.geom_groups = {}
        mesh_names = {}
        visited = set()

        def add_link(name, parent, pose):
            if name in visited:
                raise ValueError('Cycle in SDK model')
            visited.add(name)
            body = ET.SubElement(parent, 'body', name=name, pos=fmt(pose[:3, 3]), quat=fmt(quaternion(pose)))
            for index, item in enumerate(self.links[name]):
                key = (str(item.mesh), tuple(item.scale))
                if key not in mesh_names:
                    mesh_names[key] = f'mesh_{len(mesh_names)}'
                    ET.SubElement(asset, 'mesh', name=mesh_names[key], file=str(item.mesh),
                                  scale=fmt(item.scale), maxhullvert='64')
                geom_name = f'{name}__visual_{index}'
                ET.SubElement(body, 'geom', name=geom_name, type='mesh', mesh=mesh_names[key],
                              pos=fmt(item.pose[:3, 3]), quat=fmt(quaternion(item.pose)),
                              contype='0', conaffinity='0', group='0', rgba='0.65 0.7 0.8 1')
                self.geom_groups[geom_name] = item.group
            for joint in self.joints.values():
                if joint.parent == name:
                    add_link(joint.child, body, joint.relative(0))
        add_link(self.root, world, np.eye(4))
        if visited != set(self.links):
            raise ValueError(f'Unreachable links: {set(self.links)-visited}')
        # Optical pose and frustum are supplied from explicit calibration at render time.
        ET.SubElement(world, 'camera', name='calibrated_camera', ipd='0')
        return ET.tostring(root, encoding='unicode')

    def set_feedback(self, mj_model, mj_data, positions):
        import mujoco
        missing = set(self.moving_joints) - set(positions)
        if missing:
            raise ValueError(f'No measured/mapped positions for model joints: {sorted(missing)}')
        for joint in self.joints.values():
            value = 0.0 if joint.kind == 'fixed' else float(positions[joint.name])
            if not np.isfinite(value):
                raise ValueError(f'Non-finite feedback: {joint.name}')
            pose = joint.relative(value)
            body_id = mj_model.body(joint.child).id
            mj_model.body_pos[body_id] = pose[:3, 3]
            mj_model.body_quat[body_id] = quaternion(pose)
        mujoco.mj_forward(mj_model, mj_data)
