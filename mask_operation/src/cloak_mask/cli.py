"""Standalone audit, RGB inspection and calibrated dataset-mask export."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .config import CAMERAS, calibration_errors, load_config, template
from .dataset import DatasetFrames
from .model import AstribotVisualModel


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def save_rgb(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise IOError(f'Cannot write {path}')


def contact_sheet(images, captions, width=640):
    panels = []
    for image, caption in zip(images, captions, strict=True):
        h = round(image.shape[0] * width/image.shape[1])
        panel = cv2.resize(image, (width, h), interpolation=cv2.INTER_AREA)
        panel = cv2.copyMakeBorder(panel, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24,24,24))
        cv2.putText(panel, caption, (10,23), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255,255,255), 1, cv2.LINE_AA)
        panels.append(panel)
    target = max(p.shape[0] for p in panels)
    return np.concatenate([cv2.copyMakeBorder(p,0,target-p.shape[0],0,0,cv2.BORDER_CONSTANT) for p in panels], axis=1)


def inspect_rgb(dataset, output, episode, frame):
    row = dataset.row(episode, frame)
    images, provenance = [], {}
    for camera in CAMERAS:
        feature = f'observation.images.{camera}'
        rgb, timing = dataset.rgb(row, feature)
        save_rgb(output / f'{camera}_rgb.png', rgb)
        images.append(rgb)
        provenance[camera] = timing
    save_rgb(output / 'rgb_contact_sheet.png', contact_sheet(images,
        [f'{c}: recorded RGB ONLY (no mask)' for c in CAMERAS]))
    write_json(output / 'frame.json', {'row': row, 'video_alignment': provenance,
        'note': 'Dataset timestamp alignment only; unknown hardware latency is not corrected.'})


def audit(args, dataset, model):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    info = dataset.info
    draft = template(info, model)
    if args.config:
        config = load_config(args.config)
    else:
        config = draft
    errors = calibration_errors(config, info, model)
    # Configuration candidates exclude unrelated third-party robots/calibration.
    candidates = []
    for folder in ('astribot_config', 'astribot_sdk', 'examples', 'config'):
        for path in (model.sdk_root/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.json','.yaml','.yml','.xml','.urdf','.sdf'):
                text = path.read_text(errors='replace').lower()
                if any(word in text for word in ('camera_matrix', 'intrinsics', 'intrinsic_matrix', 'extrinsic', 'distortion_coefficients', '<camera')):
                    candidates.append(str(path.relative_to(model.sdk_root)))
    row = dataset.row(args.episode, args.frames[0])
    # Ranges over the recorded data, not inferred state names or commanded actions.
    import pyarrow.parquet as pq
    lo, hi, count = None, None, 0
    for path in sorted((dataset.root/'data').rglob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(columns=['observation.state']):
            values = np.asarray(batch.column(0).to_pylist(), dtype=float)
            if values.ndim != 2 or not np.isfinite(values).all():
                raise ValueError(f'Invalid feedback in {path}')
            lo = values.min(axis=0) if lo is None else np.minimum(lo, values.min(axis=0))
            hi = values.max(axis=0) if hi is None else np.maximum(hi, values.max(axis=0))
            count += len(values)
    # Compile with local MuJoCo to detect missing/incompatible geometry immediately.
    from .renderer import mujoco
    xml = model.mjcf()
    compiled = mujoco.MjModel.from_xml_string(xml)
    (output / 'astribot_visual.xml').write_text(xml)
    report = {
        'status': 'BLOCKED_MISSING_CALIBRATION' if errors else 'CALIBRATION_SCHEMA_READY_NOT_VISUALLY_VERIFIED',
        'sdk_root': str(model.sdk_root), 'dataset_root': str(dataset.root),
        'sdk_feedback_api': 'Astribot.get_current_joints_position(names) -> get_current_joint_position_list(names)',
        'sdk_feedback_source': 'astribot_sdk/core/astribot_api/astribot_client.py',
        'sdk_camera_api': 'get_cameras_info() currently provides activation status, not numeric calibration',
        'sdk_gripper_semantics': 'open_effector uses 0; close_effector uses 100; no SDF six-joint coupling found',
        'camera_calibration_file_candidates': candidates,
        'model': {'urdf': str(model.urdf), 'gripper_sdf': str(model.sdf),
                  'links': len(model.links), 'visual_geoms': compiled.ngeom,
                  'moving_kinematic_joints': model.moving_joints,
                  'mesh_files': [str(p) for p in sorted(model.mesh_sources)],
                  'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (model.urdf,model.sdf)},
                  'mjcf_semantics': 'Visual-only FK tree: body transforms updated from joint feedback; do not mj_step this model'},
        'dataset': {'frames_scanned': count, 'state_feature': info['features']['observation.state'],
                    'state_min': lo.tolist(), 'state_max': hi.tolist(),
                    'cameras': {c:info['features'][f'observation.images.{c}'] for c in CAMERAS},
                    'selected_recorded_state': row['observation.state']},
        'state_order_candidate_source': '/home/jchen572/code_file/RBM_lerobot/astribot_lerobot/utils/constants.py:ASTRIBOT_S1_CONFIG',
        'state_order_note': 'Candidate only; hdf5_robot + state_N names do not prove this recording used that exporter',
        'missing_or_invalid': errors,
        'limitations': ['No real scene depth: masks handle robot self-occlusion only.',
                        'Camera mounts/housings absent from SDK visuals are not invented.',
                        'Recorded feedback, not current live SDK feedback, must accompany historical RGB.',
                        'All physical calibration fields must come from the actual recording setup.'],
    }
    write_json(output / 'audit.json', report)
    template_path = output / 'calibration.template.json'
    if not template_path.exists():
        write_json(template_path, draft)
    inspect_rgb(dataset, output / f'episode_{args.episode:04d}_frame_{args.frames[0]:06d}', args.episode, args.frames[0])
    lines = ['# Astribot S1 Mask 自动检查', '', f"状态：{report['status']}", '',
             f'模型编译成功：{len(model.links)} 个连杆、{compiled.ngeom} 个视觉 geom、{len(model.moving_joints)} 个运动学关节。',
             f'数据集检查：{count} 帧，仅读取 observation.state，不以 action 代替反馈。', '',
             '## 尚未满足的条件', '']
    lines += [f'- {e}' for e in errors] if errors else ['- 配置结构完整；仍需人工检查 RGB 叠加对齐。']
    lines += ['', '## 已生成的内容', '', '- audit.json：SDK/模型/数据字段、实测范围和缺失项。',
              '- calibration.template.json：待补齐的真实标定；未知值为 null。',
              '- astribot_visual.xml：使用原 SDK 网格的 MuJoCo 视觉模型（仅供 FK，不用于动力学）。',
              f'- episode_{args.episode:04d}_frame_{args.frames[0]:06d}/rgb_contact_sheet.png：真实录制 RGB，未叠加任何猜测 mask。',
              '', '缺少标定时不输出 mask 或 RGB 叠加图；空白/黑色 mask 也不会被当作成功结果。',
              '候选 25 维关节顺序来自本机 RBM_lerobot 配置，但此数据集元数据缺少录制来源，需要核实。',
              'SDK 夹爪反馈 0 表示开、100 表示关；数据中存在负值，不能直接缩放成模型角度或静默裁剪。']
    (output / 'report.md').write_text('\n'.join(lines)+'\n')
    print(f"{report['status']}: {output/'report.md'}")
    return 0


def render(args, dataset, model):
    if not args.config:
        raise ValueError('render requires --config with measured calibration')
    config = load_config(args.config)
    errors = calibration_errors(config, dataset.info, model)
    if errors:
        write_json(args.output/'render_blocked.json', {'status':'BLOCKED', 'missing_or_invalid':errors})
        raise ValueError('Rendering blocked; no masks generated:\n- '+'\n- '.join(errors))
    from .renderer import AstribotMaskRenderer, COLORS, LABELS, overlay
    # Preflight every selected sample before producing output images.
    from .config import feedback_positions
    rows = [dataset.row(args.episode, frame) for frame in args.frames]
    for row in rows:
        feedback_positions(row['observation.state'], config)
    manifest = {'status':'RENDERED_REQUIRES_VISUAL_REVIEW', 'episode':args.episode,
                'state_source':'observation.state', 'label_ids':LABELS,
                'calibration_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(), 'frames':[]}
    with AstribotMaskRenderer(model, config, dataset.info) as renderer:
        for row in rows:
            labels = renderer.render(row['observation.state'])
            target = args.output/f"episode_{args.episode:04d}_frame_{row['frame_index']:06d}"
            target.mkdir(parents=True, exist_ok=False)
            panels, captions, timing, fractions = [], [], {}, {}
            for camera, camera_labels in labels.items():
                cfg = config['cameras'][camera]
                rgb, timing[camera] = dataset.rgb(row, cfg['feature'], cfg['rgb_time_offset_seconds'])
                mixed = overlay(rgb, camera_labels)
                save_rgb(target/f'{camera}_rgb.png', rgb)
                save_rgb(target/f'{camera}_overlay.png', mixed)
                save_rgb(target/f'{camera}_labels_color.png', COLORS[camera_labels])
                for suffix, mask in [('robot_mask',camera_labels>0), ('left_gripper_mask',camera_labels==5),
                                     ('right_gripper_mask',camera_labels==6)]:
                    if not cv2.imwrite(str(target/f'{camera}_{suffix}.png'),mask.astype(np.uint8)*255):
                        raise IOError('Mask write failed')
                np.save(target/f'{camera}_labels.npy',camera_labels)
                fractions[camera] = float((camera_labels>0).mean())
                panels.append(mixed)
                captions.append(f'{camera}: robot mask overlay (review calibration)')
            save_rgb(target/'overlay_contact_sheet.png',contact_sheet(panels,captions))
            entry = {'frame_index':row['frame_index'], 'timestamp':row['timestamp'],
                     'dataset_index':row['index'], 'video_alignment':timing, 'mask_fraction':fractions,
                     'feedback':row['observation.state'],
                     'warnings':[f'{c}: empty mask; inspect calibration/view' for c,f in fractions.items() if f==0]}
            write_json(target/'frame.json',entry)
            manifest['frames'].append(entry)
    write_json(args.output/'manifest.json',manifest)
    write_json(args.output/'calibration.used.json',config)
    print(f'Rendered {len(rows)} frames per camera into {args.output}; visual validation is still required.')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['audit','render'])
    parser.add_argument('--sdk-root',type=Path,default=Path('astribot_sdk_ros2'))
    parser.add_argument('--dataset-root',type=Path,default=Path('lerobotv3_dataset/lerobotv3_red_cup_only'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/astribot_mask'))
    parser.add_argument('--config',type=Path)
    parser.add_argument('--episode',type=int,default=0)
    parser.add_argument('--frames',type=int,nargs='+',default=[0])
    args = parser.parse_args()
    try:
        dataset, model = DatasetFrames(args.dataset_root), AstribotVisualModel(args.sdk_root)
        return audit(args,dataset,model) if args.command=='audit' else render(args,dataset,model)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.exit(2,f'{exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
