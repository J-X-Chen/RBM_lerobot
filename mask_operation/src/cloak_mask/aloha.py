"""Offline Piper/ALOHA visual FK. No hardware control or invented calibration."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .model import AstribotVisualModel, Joint, Visual, numbers, urdf_pose
from .renderer import mujoco, LABELS, projection_plan, set_frustum, overlay
from .dataset import DatasetFrames
from .cli import write_json, save_rgb, contact_sheet, inspect_rgb

STATE_NAMES = [f'{side}.{name}' for side in ('left', 'right')
               for name in [*(f'joint{i}' for i in range(1, 7)), 'gripper']]
DEFAULT_URDF = 'mobile_aloha_sim/aloha_new_description/urdf/aloha_tracer2_dabai_dark.urdf'


def collada_to_obj(source, cache):
    """Preserve visual-scene transforms and all triangles in the supplied Z_UP assets.

    Deliberately reject unsupported COLLADA constructs rather than silently changing
    geometry. Materials are irrelevant for segmentation; collision meshes are unused.
    """
    dest = cache / (hashlib.sha256(source.read_bytes()).hexdigest() + '.obj')
    if dest.exists():
        return dest
    root = ET.parse(source).getroot()
    for el in root.iter():
        el.tag = el.tag.split('}')[-1]
    if root.findtext('asset/up_axis') != 'Z_UP':
        raise ValueError(f'Unsupported asset axis: {source}')
    unit = root.find('asset/unit')
    scale = float(unit.get('meter', '1')) if unit is not None else 1.
    geometries = {}
    for geom in root.findall('library_geometries/geometry'):
        mesh = geom.find('mesh')
        if mesh is None or any(e.tag not in ('source', 'vertices', 'triangles') for e in mesh):
            raise ValueError(f'Unsupported mesh primitives: {source}')
        vertices = {v.get('id'): v.find("input[@semantic='POSITION']").get('source')[1:]
                    for v in mesh.findall('vertices')}
        sources = {}
        for s in mesh.findall('source'):
            accessor = s.find('technique_common/accessor')
            if int(accessor.get('offset', '0')) != 0:
                raise ValueError('Nonzero COLLADA accessor offset')
            sources[s.get('id')] = np.fromstring(s.findtext('float_array'), sep=' ').reshape(-1, int(accessor.get('stride')))
        parts = []
        for tri in mesh.findall('triangles'):
            inputs = tri.findall('input')
            v = next(i for i in inputs if i.get('semantic') == 'VERTEX')
            stride = max(int(i.get('offset')) for i in inputs) + 1
            idx = np.fromstring(tri.findtext('p'), sep=' ', dtype=int).reshape(-1, stride)[:, int(v.get('offset'))]
            if len(idx) != int(tri.get('count')) * 3:
                raise ValueError('Invalid COLLADA triangle count')
            parts.append(sources[vertices[v.get('source')[1:]]][idx, :3])
        geometries[geom.get('id')] = np.concatenate(parts)
    scene_id = root.find('scene/instance_visual_scene').get('url')[1:]
    scene = root.find(f"library_visual_scenes/visual_scene[@id='{scene_id}']")
    result = []

    def visit(node, parent):
        pose = parent.copy()
        for el in node:
            if el.tag == 'matrix':
                pose = pose @ np.fromstring(el.text, sep=' ').reshape(4, 4)
            elif el.tag in ('translate', 'rotate', 'scale', 'lookat', 'skew', 'instance_node', 'instance_controller'):
                raise ValueError(f'Unsupported COLLADA scene transform: {el.tag} in {source}')
        for instance in node.findall('instance_geometry'):
            points = geometries[instance.get('url')[1:]]
            points = (points @ pose[:3, :3].T + pose[:3, 3]) * scale
            if np.linalg.det(pose[:3, :3]) < 0:
                points = points.reshape(-1, 3, 3)[:, ::-1].reshape(-1, 3)
            result.append(points)
        for child in node.findall('node'):
            visit(child, pose)
    for node in scene.findall('node'):
        visit(node, np.eye(4))
    points = np.concatenate(result)
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite mesh vertices')
    cache.mkdir(parents=True, exist_ok=True)
    with dest.open('w') as f:
        for p in points:
            f.write('v %.10g %.10g %.10g\n' % tuple(p))
        for i in range(0, len(points), 3):
            f.write(f'f {i+1} {i+2} {i+3}\n')
    return dest


class AlohaVisualModel(AstribotVisualModel):
    """Front execution arms only, rooted at body_Link; rear teaching arms excluded."""
    def __init__(self, urdf, cache):
        self.urdf = Path(urdf).resolve()
        packages = self.urdf.parents[2]
        tree = ET.parse(self.urdf).getroot()
        self.root = 'body_Link'
        self.links = {self.root: []}
        self.joints = {}
        self.mesh_sources = set()
        for link in tree.findall('link'):
            name = link.get('name')
            if not name.startswith(('fl_', 'fr_')):
                continue
            self.links[name] = []
            side = 'left' if name.startswith('fl_') else 'right'
            group = side + ('_gripper' if name.endswith(('link7', 'link8')) else '_arm')
            for visual in link.findall('visual'):
                mesh = visual.find('geometry/mesh')
                if mesh is None:
                    raise ValueError(f'Unsupported visual: {name}')
                reference = mesh.get('filename')
                if not reference.startswith('package://'):
                    raise ValueError(f'Unexpected mesh URI: {reference}')
                source = packages / reference.removeprefix('package://')
                self.mesh_sources.add(source)
                path = collada_to_obj(source, Path(cache).resolve()) if source.suffix.lower() == '.dae' else source
                self.links[name].append(Visual(path, urdf_pose(visual.find('origin')),
                                              numbers(mesh.get('scale'), [1, 1, 1]), group))
        for el in tree.findall('joint'):
            child, parent = el.find('child').get('link'), el.find('parent').get('link')
            if child not in self.links or child == self.root:
                continue
            if parent not in self.links or el.find('mimic') is not None:
                raise ValueError(f'Unsupported ancestry/mimic: {el.get("name")}')
            axis = el.find('axis')
            axis = numbers(axis.get('xyz') if axis is not None else None, [1, 0, 0])
            axis /= np.linalg.norm(axis)
            j = Joint(el.get('name'), parent, child, el.get('type'), urdf_pose(el.find('origin')), axis)
            self.joints[j.name] = j


def positions(state, grippers):
    state = np.asarray(state, float)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError('Expected finite 14D observation.state')
    result = {}
    for side, prefix, offset in [('left', 'fl', 0), ('right', 'fr', 7)]:
        result.update({f'{prefix}_joint{i+1}': float(state[offset+i]) for i in range(6)})
        cfg = grippers[side]
        opening = state[offset+6] * cfg['scale'] + cfg['offset_m']
        if opening < 0:
            raise ValueError(f'Negative physical gripper opening: {side}')
        # URDF opposed prismatic fingers; no clipping to the model's 0.08 m limit.
        result[f'{prefix}_joint7'] = float(opening / 2)
        result[f'{prefix}_joint8'] = float(-opening / 2)
    return result


def template(dataset, model):
    return dict(status='INCOMPLETE', model_sha256=hashlib.sha256(model.urdf.read_bytes()).hexdigest(),
                model_scope='front execution arms and wrist cameras; excludes chassis, head stand, teaching arms, cables',
                grippers={s: dict(scale=None, offset_m=None, source=None) for s in ('left', 'right')},
                cameras={s: dict(width=dataset.info['features'][f'observation.images.{s}']['shape'][1],
                                height=dataset.info['features'][f'observation.images.{s}']['shape'][0],
                                parent_link=p, K=None, distortion_model=None, distortion_coefficients=None,
                                T_parent_optical=None, source=None)
                         for s, p in [('head', 'body_Link'), ('left', 'fl_link6'), ('right', 'fr_link6')]})


def validate(cfg, dataset, model, allow_candidate=False):
    errors = []
    if cfg.get('status') not in (('VERIFIED', 'CANDIDATE') if allow_candidate else ('VERIFIED',)):
        errors.append('status must be VERIFIED (or CANDIDATE with --allow-candidate)')
    if cfg.get('model_sha256') != hashlib.sha256(model.urdf.read_bytes()).hexdigest():
        errors.append('model_sha256 mismatch')
    for side in ('left', 'right'):
        g = cfg.get('grippers', {}).get(side, {})
        try:
            if not g.get('source') or not np.isfinite([g['scale'], g['offset_m']]).all() or g['scale'] <= 0:
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            errors.append(f'{side}: missing gripper scale/offset/source')
    for name in ('head', 'left', 'right'):
        try:
            c = cfg['cameras'][name]
            k, t = np.asarray(c['K'], float), np.asarray(c['T_parent_optical'], float)
            if k.shape != (3, 3) or t.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(t).all():
                raise ValueError('missing/invalid K or T_parent_optical')
            if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]) or not np.allclose([k[0, 1], k[1, 0]], 0):
                raise ValueError('invalid pinhole K')
            if not np.allclose(t[3], [0, 0, 0, 1]) or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(t[:3, :3]), 1):
                raise ValueError('invalid rigid transform')
            if c['parent_link'] not in model.links or not c.get('source'):
                raise ValueError('invalid parent_link or missing parameter source')
            shape = dataset.info['features'][f'observation.images.{name}']['shape']
            if (c['height'], c['width']) != tuple(shape[:2]):
                raise ValueError('calibration/image resolution mismatch')
            d = c['distortion_coefficients']
            if c['distortion_model'] == 'none':
                if d != []:
                    raise ValueError('none requires empty distortion coefficients')
            elif c['distortion_model'] == 'plumb_bob':
                if len(d) != 5 or not np.isfinite(d).all():
                    raise ValueError('plumb_bob requires five finite coefficients')
            else:
                raise ValueError('unsupported/missing distortion model')
        except (KeyError, TypeError, ValueError) as e:
            errors.append(f'{name}: {e}')
    return errors


def import_capture(cfg, path):
    """Import only observed ROS values; unresolved fields stay null."""
    from scipy.spatial.transform import Rotation
    capture = json.loads(path.read_text())
    cfg['status'] = 'CANDIDATE'
    for name, entry in capture['cameras'].items():
        if name not in cfg['cameras']:
            continue
        c = cfg['cameras'][name]
        info = entry.get('camera_info')
        if info:
            c.update(width=info['width'], height=info['height'],
                     K=np.asarray(info['K']).reshape(3, 3).tolist(),
                     distortion_model=info['distortion_model'], distortion_coefficients=info['D'],
                     source=f'ROS1 capture {path.resolve()}; driver fallback not ruled out')
        if 'T_parent_optical_quaternion_xyzw' in entry:
            t = np.eye(4)
            t[:3, :3] = Rotation.from_quat(entry['T_parent_optical_quaternion_xyzw']).as_matrix()
            t[:3, 3] = entry['T_parent_optical_translation_xyz']
            c.update(parent_link=entry['parent_link'], T_parent_optical=t.tolist())
    return cfg


class AlohaRenderer:
    def __init__(self, visual, cfg):
        self.visual, self.cfg = visual, cfg
        self.plans = {n: projection_plan(c) for n, c in cfg['cameras'].items()}
        w, h = max(p[0] for p in self.plans.values()), max(p[1] for p in self.plans.values())
        self.model = mujoco.MjModel.from_xml_string(visual.mjcf(w, h))
        self.data = mujoco.MjData(self.model)
        self.labels = np.zeros(self.model.ngeom, np.uint8)
        for n, g in visual.geom_groups.items():
            self.labels[self.model.geom(n).id] = LABELS[g]
        self.renderers = {}
        try:
            for n, p in self.plans.items():
                r = mujoco.Renderer(self.model, width=p[0], height=p[1])
                self.renderers[n] = r
                r.enable_segmentation_rendering()
        except Exception:
            self.close()
            raise

    def render(self, state, camera_names=None):
        if camera_names is not None and not set(camera_names) <= set(self.cfg["cameras"]):
            raise ValueError("Unknown camera requested")
        self.visual.set_feedback(self.model, self.data, positions(state, self.cfg['grippers']))
        cid = self.model.camera('calibrated_camera').id
        masks, poses = {}, {}
        for name, c in self.cfg['cameras'].items():
            if camera_names is not None and name not in camera_names:
                continue
            bid = self.model.body(c['parent_link']).id
            parent = np.eye(4)
            parent[:3, :3], parent[:3, 3] = self.data.xmat[bid].reshape(3, 3), self.data.xpos[bid]
            optical = parent @ np.asarray(c['T_parent_optical'])
            poses[name] = optical.tolist()
            self.data.cam_xpos[cid] = optical[:3, 3]
            self.data.cam_xmat[cid] = (optical[:3, :3] @ np.diag([1, -1, -1])).ravel()
            r = self.renderers[name]
            # A trial optical origin can lie inside the camera housing mesh.
            # Hide only explicitly listed enclosure visuals, never arm/finger labels.
            excluded = c.get('exclude_camera_housing', [])
            allowed = {'left': ['fl_camera_link'], 'right': ['fr_camera_link'], 'head': []}[name]
            if any(link not in allowed for link in excluded):
                raise ValueError('Only the viewing camera housing may be excluded')
            groups = self.model.geom_group.copy()
            option = mujoco.MjvOption()
            option.geomgroup[5] = 0
            try:
                for link in excluded:
                    for gid in range(self.model.ngeom):
                        if self.model.body(self.model.geom_bodyid[gid]).name == link:
                            self.model.geom_group[gid] = 5
                r.update_scene(self.data, camera='calibrated_camera', scene_option=option)
            finally:
                self.model.geom_group[:] = groups
            w, h, k, mapping = self.plans[name]
            set_frustum(r.scene, k, w, h)
            seg = r.render()
            ids = seg[..., 0]
            valid = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0) & (ids < len(self.labels))
            labels = np.zeros(ids.shape, np.uint8)
            labels[valid] = self.labels[ids[valid]]
            if mapping is not None:
                labels = cv2.remap(labels, mapping[..., 0], mapping[..., 1], cv2.INTER_NEAREST)
            masks[name] = labels
        return masks, poses

    def close(self):
        for r in self.renderers.values():
            r.close()


def run(args):
    dataset = DatasetFrames(args.dataset_root)
    if dataset.info['features']['observation.state']['names'] != STATE_NAMES:
        raise ValueError('Dataset state names/order do not match Piper execution arms')
    model = AlohaVisualModel(args.urdf, args.mesh_cache)
    cfg = json.loads(args.config.read_text()) if args.config else template(dataset, model)
    if args.ros_capture:
        cfg = import_capture(cfg, args.ros_capture)
    errors = validate(cfg, dataset, model, args.allow_candidate)
    if args.command == 'render' and errors:
        raise ValueError('Calibration incomplete; run audit first:\n- ' + '\n- '.join(errors))
    # Permit the documented preview command to be rerun with identical inputs.
    # An unrelated directory or changed calibration must use a new output path.
    identity = dict(schema=1, dataset_root=str(Path(args.dataset_root).resolve()),
                    urdf=str(Path(args.urdf).resolve()), episode=args.episode,
                    calibration_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest())
    manifest = args.output / 'render_run.json'
    if args.command == 'render' and args.output.exists():
        if not manifest.is_file() or json.loads(manifest.read_text()) != identity:
            raise FileExistsError('Output already exists with different or unknown inputs; choose a new --output')
        print(f'Regenerating preview with identical inputs: {args.output}')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    if args.command == 'render':
        write_json(manifest, identity)
    write_json(args.output / 'calibration.json', cfg)
    if args.command == 'audit':
        from .autocalibrate import discover_files
        paths = discover_files([args.service_root, Path(args.dataset_root) / 'meta'])
        xml = model.mjcf(640, 480)
        (args.output / 'execution_arms.xml').write_text(xml)
        compiled = mujoco.MjModel.from_xml_string(xml)
        for frame in (args.frames if args.frames is not None else [180, 240, 300]):
            inspect_rgb(dataset, args.output / f'episode_{args.episode:04d}_frame_{frame:06d}', args.episode, frame)
        write_json(args.output / 'audit.json', dict(status='BLOCKED_MISSING_CALIBRATION' if errors else 'READY',
            errors=errors, urdf=str(model.urdf), state_names=STATE_NAMES, camera_resolution=[640,480],
            model_geoms=compiled.ngeom, model_links=len(model.links), calibration_candidates=paths,
            issues=['No recording-specific K/D or RGB-optical-to-robot transform found in dataset metadata.',
                    'URDF camera links describe housings; optical offsets are not provided.',
                    'Head stand height/pitch are movable and absent from 14D feedback.',
                    'Recorded gripper values near 0.10 m exceed URDF nominal 0.08 m opening; do not clip or assume scale.',
                    'Static right-camera dark background makes unseeded silhouette calibration ambiguous.',
                    'CameraInfo driver may publish fallback K; capture alone does not verify calibration.'],
            excluded_geometry=['teaching arms', 'chassis', 'head stand', 'cables'],
            next_step='Capture ROS1 CameraInfo/TF; resolve mounting and gripper mapping, then render or refine a sourced candidate.'))
        print(f'Audit and RGB: {args.output}; calibration blockers: {len(errors)}')
        return
    from .video import MaskVideos
    renderer = AlohaRenderer(model, cfg)
    video = None
    rgb_frames = None
    full_episode = args.frames is None or args.all_frames
    try:
        if full_episode:
            video = MaskVideos(args.output, args.episode, dataset.info)
            rgb_frames = dataset.iter_episode_rgb(args.episode)
        frames = range(int(dataset.episodes[args.episode]['length'])) if full_episode else args.frames
        for frame in frames:
            row = dataset.row(args.episode, frame)
            masks, poses = renderer.render(row['observation.state'])
            if video is not None:
                rgbs = next(rgb_frames)
                video.write(masks, rgbs)
                if video.count % 30 == 0:
                    print(f'Episode {args.episode}: {video.count}/{len(frames)} frames', flush=True)
                continue
            dest = args.output / f'episode_{args.episode:04d}_frame_{frame:06d}'
            dest.mkdir(exist_ok=True)
            panels, timing = [], {}
            for name, labels in masks.items():
                rgb, timing[name] = dataset.rgb(row, f'observation.images.{name}')
                save_rgb(dest / f'{name}_rgb.png', rgb)
                cv2.imwrite(str(dest / f'{name}_native_robot_mask.png'), (labels > 0).astype(np.uint8) * 255)
                np.save(dest / f'{name}_native_labels.npy', labels)
                combined = overlay(rgb, labels)
                save_rgb(dest / f'{name}_native_overlay.png', combined)
                panels.append(contact_sheet([rgb, np.repeat((labels > 0)[..., None], 3, axis=2).astype(np.uint8)*255, combined],
                                           [name+' RGB', name+' native mask', cfg['status']+' overlay']))
            save_rgb(dest / 'comparison.png', np.concatenate(panels, axis=0))
            write_json(dest / 'frame.json', dict(row=row, video_alignment=timing, T_body_optical=poses,
                                               calibration_status=cfg['status']))

        if video is not None:
            write_json(args.output / 'video_report.json', dict(episode=args.episode, calibration_status=cfg['status'], **video.finish()))
    finally:
        if rgb_frames is not None:
            rgb_frames.close()
        if video is not None:
            video.close()
        renderer.close()
    print(f'Masks: {args.output}')
