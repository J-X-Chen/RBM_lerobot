#!/usr/bin/env python3
"""Explicitly build a nominal-model CANDIDATE, never a measured calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from scipy.spatial.transform import Rotation
from cloak_mask.aloha import AlohaVisualModel, DEFAULT_URDF, import_capture, template, validate, collada_to_obj
from cloak_mask.dataset import DatasetFrames
from cloak_mask.model import urdf_pose
from cloak_mask.cli import write_json


def prepare(args):
    ds = DatasetFrames(args.dataset_root)
    visual = AlohaVisualModel(args.urdf, args.mesh_cache)
    cfg = import_capture(template(ds, visual), args.capture)
    tree = ET.parse(args.urdf).getroot()
    def joint_pose(name, value=0.):
        joint = tree.find(f"joint[@name='{name}']")
        pose = urdf_pose(joint.find('origin'))
        if joint.get('type') != 'fixed':
            bounds = joint.find('limit')
            if not float(bounds.get('lower')) <= value <= float(bounds.get('upper')):
                raise ValueError(f'{name}: preview value outside URDF range')
            axis = np.fromstring(joint.find('axis').get('xyz'),sep=' ')
            motion = np.eye(4)
            if joint.get('type') == 'prismatic':
                motion[:3,3] = axis*value
            else:
                motion[:3,:3] = Rotation.from_rotvec(axis*value).as_matrix()
            pose = pose @ motion
        return pose
    # Explicit convention hypothesis: housing +X forward,+Y left,+Z up.
    # Include each camera visual's rotation; a URDF visual pose is NOT a calibrated optical pose.
    housing_to_cv = np.eye(4)
    housing_to_cv[:3,:3] = [[0,0,1],[-1,0,0],[0,-1,0]]
    chains = {
        'head': joint_pose('camera_stand_1_joint',args.head_height) @ joint_pose('camera_stand_2_joint',args.head_pitch) @ joint_pose('mid_camera_joint'),
        'left': joint_pose('fl_camera_stand_joint') @ joint_pose('fl_camera_joint'),
        'right': joint_pose('fr_camera_stand_joint') @ joint_pose('fr_camera_joint'),
    }
    front_surface = None
    if getattr(args, 'wrist_optical_origin', 'mesh-origin') == 'mesh-front':
        mesh = Path(args.urdf).resolve().parents[1] / 'meshes/dabai.dae'
        converted = collada_to_obj(mesh, Path(args.mesh_cache).resolve())
        vertices = np.array([[float(x) for x in line.split()[1:]]
                             for line in converted.read_text().splitlines() if line.startswith('v ')])
        front_surface = float(vertices[:, 0].max())
    for name, link in [('head','mid_camera_link'),('left','fl_camera_link'),('right','fr_camera_link')]:
        v = tree.find(f"link[@name='{link}']/visual")
        optical = chains[name] @ urdf_pose(v.find('origin')) @ housing_to_cv
        if name != 'head' and front_surface is not None:
            # Only the previously missing optical-origin offset is hypothesized.
            # No official mounting joint, mesh vertex, K, or camera orientation changes.
            visual_world = chains[name] @ urdf_pose(v.find('origin'))
            optical[:3, 3] += visual_world[:3, :3] @ np.array([front_surface, 0., 0.])
        cfg['cameras'][name]['T_parent_optical'] = optical.tolist()
        cfg['cameras'][name]['exclude_camera_housing'] = [] if name == 'head' else [link]
        cfg['cameras'][name]['source'] = {
            'intrinsics': str(args.capture), 'extrinsics': str(args.urdf),
            'status': 'NOMINAL_MODEL_AND_AXIS_HYPOTHESIS_NOT_MEASURED',
            'optical_origin': 'Assumed at camera visual origin; true RGB optical center offset unknown',
            'optical_axes': 'Assumed housing +X forward,+Y left,+Z up; apply camera visual rotation then OpenCV axes',
            'distortion': 'Captured D retained; driver zeros do not establish zero physical distortion',
        }
        if name != 'head' and front_surface is not None:
            cfg['cameras'][name]['source']['optical_origin'] = 'CAD front surface candidate, NOT a factory optical calibration'
            cfg['cameras'][name]['optical_origin_trial'] = dict(
                status='CAD_FRONT_SURFACE_HYPOTHESIS_NOT_OPTICAL_CALIBRATION',
                front_surface_x_m=front_surface, mesh=str(mesh),
                note='Preserve official mounting transforms, orientation and intrinsics. RGB lateral optical offset and entrance pupil remain unknown. No manual -3mm/+2deg correction.')
    for side in ('left','right'):
        cfg['grippers'][side] = dict(scale=1.,offset_m=0.,
            source='PREVIEW HYPOTHESIS: treat recorded SDK /1e6 as physical opening in metres; no clipping to URDF limit',
            verified=False)
    cfg['preview_assumptions'] = dict(
        head_stand_height_m=args.head_height, head_stand_pitch_rad=args.head_pitch,
        head_stand_source='Explicit trial coordinates, not recorded feedback or fitted calibration',
        gripper_note='~0.10 m observed vs 0.08 m URDF nominal maximum remains unresolved',
        camera_note='Captured K reused at matching resolution; recording-time sensor identity/settings still need verification',
        purpose='User-requested initial visual preview; all mounting poses and gripper scale remain unverified')
    cfg['reference_capture_sha256'] = hashlib.sha256(args.capture.read_bytes()).hexdigest()
    errors = validate(cfg, ds, visual, allow_candidate=True)
    if errors:
        raise ValueError('\n'.join(errors))
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous.get('status') != 'INCOMPLETE':
            raise FileExistsError('Refusing to replace a populated calibration; choose another output')
        backup = args.output.with_name('calibration.incomplete.original.json')
        with backup.open('x') as f:
            json.dump(previous, f, indent=2)
    write_json(args.output, cfg)
    print(f'CANDIDATE preview configuration: {args.output}')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,default=Path('aloha_calibration_reference_20261031/capture.json'))
    p.add_argument('--dataset-root',type=Path,default=Path('lerobotv3_dataset/aloha_cup_selection_red'))
    p.add_argument('--urdf',type=Path,default=Path(DEFAULT_URDF))
    p.add_argument('--mesh-cache',type=Path,default=Path('artifacts/aloha_mask/mesh_cache'))
    p.add_argument('--head-height',type=float,default=0.,help='Explicit trial head stand displacement (m), not measured')
    p.add_argument('--head-pitch',type=float,default=0.,help='Explicit trial head stand pitch (rad), not measured')
    p.add_argument('--wrist-optical-origin', choices=['mesh-origin', 'mesh-front'], default='mesh-origin',
                   help='Explicit optical-origin hypothesis; mesh-front uses CAD front plane, NOT calibrated optical center')
    p.add_argument('--output',type=Path,required=True)
    prepare(p.parse_args())
