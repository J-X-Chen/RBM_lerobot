"""Discover calibration evidence and attempt Cloak silhouette target extraction.

Never reuse Franka/DROID intrinsics, poses, or gripper linkage constants for S1.
Candidate masks are explicitly NOT ground truth or accepted calibration.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .config import s1_state_order


from .ros_capture import probe_ros


def discover_files(roots):
    """Find calibration/ROS bag evidence without importing SDKs or scanning private home state."""
    results = []
    ignore = {'third_party','.git','.venv','node_modules','site-packages','artifacts','__pycache__'}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for folder, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in ignore]
            for name in files:
                path = Path(folder)/name
                lower = name.lower()
                if path.suffix.lower() in ('.mcap','.db3','.bag'):
                    results.append({'path':str(path.resolve()),'kind':'rosbag_candidate'})
                elif path.suffix.lower() in ('.json','.yaml','.yml','.xml','.txt','.urdf','.sdf') and path.stat().st_size < 4_000_000:
                    if not any(token in lower for token in ('camera','calib','intrinsic','extrinsic','robot')):
                        continue
                    text = path.read_text(errors='replace')
                    tokens = [token for token in ('camera_matrix','distortion_coefficients','intrinsics',
                        'T_parent_optical','<camera','CameraInfo') if token in text]
                    if tokens:
                        results.append({'path':str(path.resolve()),'kind':'calibration_candidate','matched_keys':tokens})
    return results


def iou(a,b):
    return float((a&b).sum()/max(1,(a|b).sum()))


def silhouette_attempt(dataset, output, max_samples=32, open_tolerance=1.0):
    # Reuse the actual Cloak algorithm, but not its DROID intrinsics/extrinsic seed.
    from openpi.calibration.silhouette import CalibOptions, build_target_mask, estimate_ee_color
    from .cli import save_rgb, contact_sheet, write_json
    from .renderer import overlay
    results = {}
    # Exploratory state order only; not accepted as the recording's verified mapping.
    candidate_order = s1_state_order()
    episode_ids = sorted(dataset.episodes)
    chosen_episodes = [episode_ids[i] for i in np.unique(np.linspace(0,len(episode_ids)-1,min(16,len(episode_ids)),dtype=int))]
    for camera in ('left','right'):
        candidates = []
        index = candidate_order.index(f'astribot_gripper_{camera}')
        for episode in chosen_episodes:
            length = dataset.episodes[episode]['length']
            for frame in np.unique(np.linspace(0,length-1,8,dtype=int)):
                row = dataset.row(episode,int(frame))
                # SDK explicitly documents zero=open; tolerance is only frame selection.
                if abs(row['observation.state'][index]) <= open_tolerance:
                    candidates.append(row)
        indices = np.unique(np.linspace(0,len(candidates)-1,min(max_samples,len(candidates)),dtype=int)) if candidates else []
        rows = [candidates[i] for i in indices]
        if len(rows)<10:
            results[camera]={'status':'INSUFFICIENT_OPEN_FRAMES','open_frames':len(rows)}
            continue
        images = [dataset.rgb(r,f'observation.images.{camera}')[0] for r in rows]
        # 320px wide for the exploratory target, preserve aspect ratio.
        width=320
        height=round(images[0].shape[0]*width/images[0].shape[1])
        frames=np.stack([cv2.resize(im,(width,height),interpolation=cv2.INTER_AREA) for im in images])
        split=np.arange(len(frames))%2==0
        # original build_target_mask's gripper filter: preselected frames all count as open.
        def target(subset):
            positions=np.zeros(len(subset))
            color=estimate_ee_color(positions,subset,crop='rigid')
            opts=CalibOptions(crop='rigid',ee_color=tuple(color))
            return build_target_mask(positions,subset,opts),color
        train,color=target(frames[split])
        heldout,_=target(frames[~split])
        score=iou(train,heldout)
        target_dir=output/camera
        target_dir.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(target_dir/'candidate_train_silhouette.png'),train.astype(np.uint8)*255)
        cv2.imwrite(str(target_dir/'candidate_holdout_silhouette.png'),heldout.astype(np.uint8)*255)
        preview=contact_sheet([frames[0],overlay(frames[0],train.astype(np.uint8)*5),
            overlay(frames[1],heldout.astype(np.uint8)*5)],
            ['Recorded RGB','CANDIDATE target - NOT calibrated','Held-out candidate - NOT calibrated'],width=480)
        save_rgb(target_dir/'silhouette_attempt.png',preview)
        results[camera]={'status':'TARGET_ONLY_NOT_ACCEPTED_CALIBRATION',
            'sample_count':len(rows),'sample_rows':[{'episode':r['episode_index'],'frame':r['frame_index']} for r in rows],
            'candidate_state_index':index,'open_tolerance_sdk_units':open_tolerance,
            'estimated_rigid_rgb':color.tolist(),'split_target_iou':score,
            'train_area_fraction':float(train.mean()),'heldout_area_fraction':float(heldout.mean()),
            'target_consistency_gate':bool(score>=0.65 and 0.01<float(train.mean())<0.5),
            'important':'Split-target IoU measures only heuristic repeatability, NOT calibration accuracy. '
                        'The candidate state mapping, gripper kinematics and intrinsics remain unverified.'}
        write_json(target_dir/'target_report.json',results[camera])
    results['head']={'status':'ORIGINAL_WRIST_ASSUMPTION_NOT_APPLICABLE',
        'reason':'Head camera sees arms moving relative to it. Static dark/low-variance pixels describe background, not the whole robot.'}
    results['pose_optimization']={'status':'BLOCKED_MISSING_PHYSICAL_INPUTS',
        'reason':'Original Cloak optimize_pose requires known focal length, known end-effector kinematics '
                 'and a camera-to-gripper initialization with the gripper in view. '
                 'S1 has no K, no 6-joint gripper mapping, and no mount transform in the supplied SDK. '
                 'Fitting all of these to a single heuristic static silhouette is underconstrained. '
                 'Franka defaults were not substituted; no candidate was promoted to calibration.'}
    return results


def run(args,dataset,model):
    from .cli import write_json
    args.output.mkdir(parents=True,exist_ok=True)
    roots=[model.sdk_root, dataset.root/'meta', Path('/opt'),Path('/usr/local'),
           Path('/home/jchen572/code_file/RBM_lerobot/astribot_lerobot')]
    roots.append(Path('astribot_simulation'))
    roots.extend(args.search_root)
    report={'file_search_roots':[str(p) for p in roots], 'files':discover_files(roots)}
    report['ros']=json.loads(args.ros_capture.read_text()) if args.ros_capture else probe_ros(args.ros_timeout)
    from .config import template
    from .ros_import import import_candidates
    candidate, notes = import_candidates(template(dataset.info,model),report['ros'],model)
    report['ros_import_notes']=notes
    write_json(args.output/'calibration.candidate.json',candidate)
    write_json(args.output/'discovery.json',report)
    report['silhouette']=silhouette_attempt(dataset,args.output/'silhouette',args.max_samples,args.open_tolerance)
    report['status']='NO_VERIFIED_CALIBRATION_FOUND'
    write_json(args.output/'autocalibration_report.json',report)
    print(f"{report['status']}: {args.output/'autocalibration_report.json'}")
    return 0
