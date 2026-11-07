"""Inspect and preview official simulation parameters; never certify real calibration."""
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .config import s1_state_order
from .renderer import mujoco, LABELS, overlay, set_frustum, constrain_wrist_mask
from .model import AstribotVisualModel
from .tcp import project_tcp, draw_tcp

MODEL = 'mjcf/astribot_s1_mjcf/astribot_s1_for_aloha_with_gripper.xml'


def coupled_positions(model, seeds):
    """Solve compiled affine joint equalities, respecting MuJoCo qpos0 offsets."""
    values = dict(seeds)
    edges = []
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT or not model.eq_active0[i]:
            continue
        a, b = int(model.eq_obj1id[i]), int(model.eq_obj2id[i])
        if b < 0:
            continue
        coeff = model.eq_data[i, :5]
        if np.any(np.abs(coeff[2:]) > 1e-12) or abs(coeff[1]) < 1e-12:
            raise ValueError('Only invertible affine joint coupling is supported')
        an, bn = model.joint(a).name, model.joint(b).name
        offset = model.qpos0[model.jnt_qposadr[a]] + coeff[0] - coeff[1]*model.qpos0[model.jnt_qposadr[b]]
        edges.append((an, bn, float(offset), float(coeff[1])))
    for _ in range(len(edges)+1):
        known_before = len(values)
        for a,b,offset,scale in edges:
            if a in values and b in values:
                if not np.isclose(values[a], offset+scale*values[b], atol=1e-9):
                    raise ValueError(f'Inconsistent equality: {a}, {b}')
            elif b in values:
                values[a] = offset+scale*values[b]
            elif a in values:
                values[b] = (values[a]-offset)/scale
        # A full pass with no new values has checked all reachable constraints.
        if len(values) == known_before:
            break
    return values


def load_simulation(root):
    root = Path(root).resolve()
    # Absolute paths are required for nested includes in MuJoCo 3.17.
    model = mujoco.MjModel.from_xml_path(str(root/'astribot_descriptions'/MODEL))
    model.vis.global_.offwidth = max(1280, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(720, model.vis.global_.offheight)
    return model


def camera_evidence(model, data, camera, sdk_links):
    cid = model.camera(camera).id
    parent = int(model.cam_bodyid[cid])
    while parent and model.body(parent).name not in sdk_links:
        parent = int(model.body_parentid[parent])
    if not parent:
        raise ValueError(f'No SDK mounting ancestor for {camera}')
    rotation = data.xmat[parent].reshape(3,3)
    t = np.eye(4)
    t[:3,:3] = rotation.T @ data.cam_xmat[cid].reshape(3,3) @ np.diag([1,-1,-1])
    t[:3,3] = rotation.T @ (data.cam_xpos[cid]-data.xpos[parent])
    return {'name':camera, 'parent_link':model.body(parent).name,
            'T_parent_optical':t.tolist(), 'fovy_degrees':float(model.cam_fovy[cid]),
            'transform_note':'Fixed mount in the unflipped OpenCV convention, NOT a world extrinsic. Apply current joint FK and the preview optical-axis rotation before projection.',
            'status':'SIMULATION_PARAMETERS_NOT_REAL_CALIBRATION'}


def frame_camera_transforms(model, data, camera):
    """Export actual per-frame FK camera pose, including the preview's 180-degree roll.

    Both preview variants now flip both pixel axes overall. This is a proper
    optical-axis rotation, not a world reflection. World-to-camera is the inverse.
    """
    cid = model.camera(camera).id
    parent = int(model.cam_bodyid[cid])
    world_parent = np.eye(4)
    world_parent[:3,:3] = data.xmat[parent].reshape(3,3)
    world_parent[:3,3] = data.xpos[parent]
    local_gl = np.eye(4)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.cam_quat[cid])
    local_gl[:3,:3] = rotation.reshape(3,3)
    local_gl[:3,3] = model.cam_pos[cid]
    world_gl = world_parent @ local_gl
    # Independently verify composition against MuJoCo's forward-kinematics output.
    np.testing.assert_allclose(world_gl[:3,:3],data.cam_xmat[cid].reshape(3,3),atol=1e-10)
    np.testing.assert_allclose(world_gl[:3,3],data.cam_xpos[cid],atol=1e-10)
    gl_to_cv = np.diag([1.,-1.,-1.,1.])
    image_roll = np.diag([-1.,-1.,1.,1.])
    world_raw = world_gl @ gl_to_cv
    world_preview = world_raw @ image_roll
    return {'camera':camera, 'parent_link':model.body(parent).name,
            'T_world_parent':world_parent.tolist(),
            'T_parent_camera_gl':local_gl.tolist(),
            'T_world_optical_raw':world_raw.tolist(),
            'T_world_optical_preview':world_preview.tolist(),
            'T_optical_preview_world':np.linalg.inv(world_preview).tolist(),
            'composition':'T_world_parent(q) @ T_parent_camera_gl @ diag(1,-1,-1,1) @ diag(-1,-1,1,1)'}


def run(args):
    from .dataset import DatasetFrames
    from .cli import write_json, save_rgb, contact_sheet
    dataset = getattr(args, 'live_dataset', None)
    if dataset is None:
        dataset = DatasetFrames(args.dataset_root)
    wrist_path=getattr(args,'wrist_candidate',None)
    wrist_candidate=None if wrist_path is None else json.loads(Path(wrist_path).read_text())
    if wrist_candidate is not None:
        if set(wrist_candidate['cameras']) != {'left','right'}:
            raise ValueError('Wrist candidate must contain left and right')
        for name,cfg in wrist_candidate['cameras'].items():
            shape=dataset.info['features'][f'observation.images.{name}']['shape']
            if (cfg['height'],cfg['width']) != tuple(shape[:2]):
                raise ValueError('Wrist candidate resolution mismatch')
            if not np.isfinite([cfg['scale_x'],cfg['offset_x']]).all() or not .9<=cfg['scale_x']<=1.1 or abs(cfg['offset_x'])>30:
                raise ValueError('Wrist candidate exceeds supported small correction range')
    sdk = AstribotVisualModel(args.sdk_root)
    model = load_simulation(args.simulation_root)
    head_translation = float(getattr(args,'head_camera_y_m',0.))
    head_cy = float(getattr(args,'head_cy_offset_px',0.))
    if not np.isfinite([head_translation,head_cy]).all():
        raise ValueError('Head diagnostic offsets must be finite')
    head_id = model.camera('head_rgbd').id
    mount_before = model.cam_pos[head_id].copy()
    mount_rotation = np.empty(9)
    mujoco.mju_quat2Mat(mount_rotation,model.cam_quat[head_id])
    # Corrected optical +y equals original OpenGL camera +y after the 180deg roll.
    model.cam_pos[head_id] += mount_rotation.reshape(3,3) @ np.array([0.,head_translation,0.])
    head_candidate_path = getattr(args,'head_candidate',None)
    head_candidate = None
    if head_candidate_path:
        if head_translation != 0 or head_cy != 0:
            raise ValueError('Use head-candidate or manual diagnostic offsets, not both')
        head_candidate = json.loads(Path(head_candidate_path).read_text())
        shape = dataset.info['features']['observation.images.head']['shape']
        if (head_candidate['width'],head_candidate['height']) != (shape[1],shape[0]):
            raise ValueError('Head candidate resolution does not match recorded RGB')
        if head_candidate['parent_link'] != model.body(int(model.cam_bodyid[head_id])).name:
            raise ValueError('Head candidate mounting link does not match MJCF camera parent')
        if head_candidate.get('distortion_model') != 'none' or head_candidate.get('distortion_coefficients') != []:
            raise ValueError('Simulation candidate preview currently supports only explicit zero distortion')
        pose = np.asarray(head_candidate['T_parent_optical'],float)
        k_candidate = np.asarray(head_candidate['K'],float)
        if (pose.shape != (4,4) or not np.isfinite(pose).all() or
            not np.allclose(pose[3],[0,0,0,1]) or
            not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-6) or
            not np.isclose(np.linalg.det(pose[:3,:3]),1) or
            k_candidate.shape != (3,3) or not np.isfinite(k_candidate).all() or
            k_candidate[0,0]<=0 or k_candidate[1,1]<=0 or
            not np.allclose(k_candidate[2],[0,0,1]) or
            not np.allclose([k_candidate[0,1],k_candidate[1,0]],0)):
            raise ValueError('Invalid candidate rigid pose or intrinsics')
        model.cam_pos[head_id] = pose[:3,3]
        # Preview flips both axes after rendering, so undo that roll for the GL mount.
        mujoco.mju_mat2Quat(model.cam_quat[head_id],(pose[:3,:3]@np.diag([-1.,1.,-1.])).ravel())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    root = args.simulation_root.resolve()
    backend = root/'src/astribot_envs/astribot_mujoco_env.py'
    # Refuse silently reusing this adapter if the inspected conversion changed.
    if 'self.data.qpos[qpos_index] / 0.93 * 100' not in backend.read_text():
        raise ValueError('Simulation feedback conversion changed; review adapter')
    args.output.mkdir(parents=True,exist_ok=True)
    cameras = {'head':'head_rgbd','left':'left_wrist_rgbd','right':'right_wrist_rgbd'}
    evidence = {k:camera_evidence(model,data,v,sdk.links) for k,v in cameras.items()}
    linkage = {side:coupled_positions(model,{f'astribot_gripper_{side}_joint_L1':.93}) for side in ('left','right')}
    for name, item in evidence.items():
        height,width = dataset.info['features'][f'observation.images.{name}']['shape'][:2]
        focal = height/(2*np.tan(np.deg2rad(item['fovy_degrees'])/2))
        item['native_pinhole_K_at_dataset_resolution'] = [[focal,0,(width-1)/2],[0,focal,(height-1)/2],[0,0,1]]
        if name == 'head':
            item['diagnostic_preview_K'] = [[focal,0,(width-1)/2],[0,focal,(height-1)/2+head_cy],[0,0,1]]
            if head_candidate is not None:
                item['diagnostic_preview_K'] = head_candidate['K']
        item['intrinsics_note'] = 'Derived from explicit MJCF fovy for native rendering, NOT real sensor K or resized/flipped simulation pipeline K.'
    # Show visual mesh surfaces only, excluding floor, props and collision proxies.
    labels = np.zeros(model.ngeom,np.uint8)
    for g in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[g])).name
        if body.startswith('astribot_') and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            labels[g] = LABELS[sdk.group(body)]
        model.geom_group[g] = 0 if labels[g] else 5
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[0] = 1
    option.sitegroup[:] = 0
    report = {'status':'UNVERIFIED_SIMULATION_PREVIEW_NOT_FOR_TRAINING',
        'wrist_mask_constraint':'left/right: zero rows y < ceil(image_height/2), after orientation and resizing; head unchanged',
        'wrist_native_candidate':wrist_candidate,
        'model_commit':subprocess.check_output(['git','-C',str(root/'astribot_descriptions'),'rev-parse','HEAD'],text=True).strip(),
        'backend_sha256':hashlib.sha256(backend.read_bytes()).hexdigest(),
        'camera_parameters':evidence, 'finger_angles_at_feedback_100':linkage,
        'head_fitted_candidate':None if head_candidate is None else {
            'path':str(head_candidate_path),'sha256':hashlib.sha256(Path(head_candidate_path).read_bytes()).hexdigest(),
            'status':'FITTED_CANDIDATE_NOT_VERIFIED_CALIBRATION'},
        'head_diagnostic_offsets':{'camera_y_m':head_translation,'cy_offset_px':head_cy,
            'original_parent_camera_position':mount_before.tolist(),
            'status':'EXPLICIT_DIAGNOSTIC_HYPOTHESIS_NOT_ESTIMATED_CALIBRATION'},
        'assumptions':['Dataset state order is the existing unconfirmed RBM exporter candidate.',
            'SDK feedback is converted with simulation q=feedback*0.93/100; passive fingers use compiled equalities.',
            'Measured feedback is not clipped, even outside simulator limits; such values are reported.',
            'Unrecorded wheel/free-object coordinates retain model qpos0; no physics stepping.',
            'Native hypothesis uses the named camera at dataset resolution, then flips both pixel axes per user visual review.',
            'Simulation pipeline hypothesis uses 640x480, swaps wrists, resizes then vertically flips; an additional horizontal flip corrects the reviewed image orientation.',
            'No real distortion, camera synchronization, or recording provenance has been verified.'], 'frames':[]}
    renderer = mujoco.Renderer(model,height=480,width=640)
    renderer.enable_segmentation_rendering()
    native_renderers = {}
    video = None
    rgb_frames = None
    mask_callback = getattr(args, "mask_callback", None)
    try:
        for camera in cameras:
            shape = dataset.info['features'][f'observation.images.{camera}']['shape']
            native_renderers[camera] = mujoco.Renderer(model,height=shape[0],width=shape[1])
            native_renderers[camera].enable_segmentation_rendering()
        def segmentation(r,c,wrist=None):
            r.update_scene(data,camera=c,scene_option=option)
            focal=r.height/(2*np.tan(np.deg2rad(model.cam_fovy[model.camera(c).id])/2))
            k=np.array([[focal,0,(r.width-1)/2],[0,focal,(r.height-1)/2],[0,0,1.]])
            if c == 'head_rgbd' and head_candidate is not None:
                # Convert corrected full-resolution K to unflipped render K.
                sx,sy=r.width/head_candidate['width'],r.height/head_candidate['height']
                k=np.asarray(head_candidate['K'],float).copy()
                k[0,0]*=sx;k[1,1]*=sy
                k[0,2]=r.width-1-((k[0,2]+.5)*sx-.5)
                k[1,2]=r.height-1-((k[1,2]+.5)*sy-.5)
                set_frustum(r.scene,k,r.width,r.height)
            elif c == 'head_rgbd' and head_cy != 0:
                height,width = r.height,r.width
                focal=height/(2*np.tan(np.deg2rad(model.cam_fovy[head_id])/2))
                stored_height=dataset.info['features']['observation.images.head']['shape'][0]
                # Final preview flips y, hence the sign here is the inverse.
                k=np.array([[focal,0,(width-1)/2],[0,focal,(height-1)/2-head_cy*height/stored_height],[0,0,1]])
                set_frustum(r.scene,k,width,height)
            if wrist_candidate is not None and wrist in ('left','right'):
                cfg=wrist_candidate['cameras'][wrist]
                focal=r.height/(2*np.tan(np.deg2rad(model.cam_fovy[model.camera(c).id])/2))
                # Corrected-view x'=a*x+b corresponds to fx'=a*fx, cx'=a*cx+b.
                # Rendering precedes the 180-degree image roll, hence invert cx.
                corrected_cx=cfg['scale_x']*(r.width-1)/2+cfg['offset_x']
                k=np.array([[cfg['scale_x']*focal,0,r.width-1-corrected_cx],
                            [0,focal,(r.height-1)/2],[0,0,1.]])
                set_frustum(r.scene,k,r.width,r.height)
            seg = r.render()
            valid = (seg[...,1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[...,0]>=0)
            out = np.zeros(seg.shape[:2],np.uint8)
            out[valid] = labels[seg[...,0][valid]]
            return out,k
        from .video import MaskVideos
        if args.frames is None and mask_callback is None:
            video = MaskVideos(args.output, args.episode, dataset.info)
            rgb_frames = dataset.iter_episode_rgb(args.episode)
        frames = range(int(dataset.episodes[args.episode]['length'])) if video is not None else args.frames
        rows = (getattr(args, 'mask_rows') if mask_callback is not None
                else (dataset.row(args.episode, frame) for frame in frames))
        for row in rows:
            frame = int(row['frame_index'])
            positions = dict(zip(s1_state_order(),row['observation.state'],strict=True))
            seeds = {f'astribot_gripper_{side}_joint_L1':positions[f'astribot_gripper_{side}']*.93/100 for side in ('left','right')}
            positions.update(coupled_positions(model,seeds))
            positions['astribot_chassis_zrot'] = positions['astribot_chassis_yaw']
            data.qpos[:] = model.qpos0
            outside=[]
            for j in range(model.njnt):
                name=model.joint(j).name
                if name.startswith('astribot_gripper_') and name not in positions:
                    raise ValueError(f'No coupling for {name}')
                if name in positions:
                    if int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),int(mujoco.mjtJoint.mjJNT_SLIDE)):
                        raise ValueError(f'Non-scalar measured joint {name}')
                    value=positions[name]
                    data.qpos[model.jnt_qposadr[j]]=value
                    if model.jnt_limited[j] and not model.jnt_range[j,0]<=value<=model.jnt_range[j,1]:
                        outside.append({'joint':name,'value':value,'range':model.jnt_range[j].tolist()})
            mujoco.mj_forward(model,data)
            if video is not None or mask_callback is not None:
                masks = {}
                for camera, cname in cameras.items():
                    raw, _ = segmentation(native_renderers[camera], cname, wrist=camera)
                    masks[camera] = constrain_wrist_mask(cv2.flip(raw, -1), camera)
                if mask_callback is not None:
                    mask_callback(row, masks)
                    continue
                rgbs = next(rgb_frames)
                video.write(masks, rgbs)
                report['frames'].append({'episode':args.episode, 'frame':frame,
                    'timestamp':float(row['timestamp']), 'outside_simulator_limits':outside})
                if video.count % 30 == 0:
                    print(f'Episode {args.episode}: {video.count}/{len(frames)} frames', flush=True)
                continue
            target=args.output/f'episode_{args.episode:04d}_frame_{frame:06d}'
            target.mkdir(exist_ok=False)
            panels,captions,timing,areas=[],[],{},{}
            tcp_panels,tcp_captions,tcp_records=[],[],{}
            for camera,cname in cameras.items():
                rgb,timing[camera]=dataset.rgb(row,f'observation.images.{camera}')
                # Correct rendered labels before overlays/export; recorded RGB stays unchanged.
                raw_native,native_k=segmentation(native_renderers[camera],cname,wrist=camera)
                native=cv2.flip(raw_native,-1)
                swapped=cameras[{'left':'right','right':'left'}.get(camera,camera)]
                raw_pipeline,pipeline_k=segmentation(renderer,swapped)
                pipeline=cv2.flip(cv2.resize(raw_pipeline,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST),0)
                pipeline=cv2.flip(pipeline,1)
                size=(rgb.shape[1],rgb.shape[0])
                tcp_records[camera]={
                    'native':project_tcp(model,data,cname,native_k,size,size),
                    'simulation_pipeline':project_tcp(model,data,swapped,pipeline_k,(renderer.width,renderer.height),size)}
                native=constrain_wrist_mask(native,camera)
                pipeline=constrain_wrist_mask(pipeline,camera)
                save_rgb(target/f'{camera}_rgb.png',rgb)
                for variant,mask in [('native',native),('simulation_pipeline',pipeline)]:
                    mixed=overlay(rgb,mask)
                    cv2.putText(mixed,'UNVERIFIED SIMULATION CANDIDATE',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,60,60),1)
                    save_rgb(target/f'{camera}_{variant}_overlay.png',mixed)
                    save_rgb(target/f'{camera}_{variant}_tcp_overlay.png',draw_tcp(mixed,tcp_records[camera][variant]))
                    cv2.imwrite(str(target/f'{camera}_{variant}_robot_mask.png'),(mask>0).astype(np.uint8)*255)
                    np.save(target/f'{camera}_{variant}_labels.npy',mask)
                    areas[f'{camera}_{variant}']=float((mask>0).mean())
                panels.extend([rgb,overlay(rgb,native),overlay(rgb,pipeline)])
                captions.extend([f'{camera}: recorded RGB',f'{camera}: UNVERIFIED native',f'{camera}: UNVERIFIED sim pipeline'])
                tcp_rgb=draw_tcp(rgb,tcp_records[camera]['native'])
                save_rgb(target/f'{camera}_tcp_rgb.png',tcp_rgb)
                tcp_panels.extend([tcp_rgb,draw_tcp(overlay(rgb,native),tcp_records[camera]['native']),
                                   draw_tcp(overlay(rgb,pipeline),tcp_records[camera]['simulation_pipeline'])])
                tcp_captions.extend([f'{camera}: RGB + native TCP',f'{camera}: native + TCP',f'{camera}: sim pipeline + TCP'])
            # Three columns, one camera per row.
            sheets=[contact_sheet(panels[i:i+3],captions[i:i+3],width=480) for i in range(0,9,3)]
            save_rgb(target/'comparison.png',np.concatenate(sheets,axis=0))
            tcp_sheets=[contact_sheet(tcp_panels[i:i+3],tcp_captions[i:i+3],width=480) for i in range(0,9,3)]
            save_rgb(target/'tcp_comparison.png',np.concatenate(tcp_sheets,axis=0))
            write_json(target/'tcp_points.json',{'definition':'MJCF arm_left_tool / arm_right_tool reference sites; NOT calibrated fingertips. Cyan=L, yellow=R. Points are not occlusion-tested or constrained to the wrist lower half.',
                'episode':args.episode,'frame':frame,'cameras':tcp_records})
            report['frames'].append({'episode':args.episode,'frame':frame,'video_alignment':timing,'mask_fraction':areas,'outside_simulator_limits':outside,
                'head_feedback_rad':[positions['astribot_head_joint_1'],positions['astribot_head_joint_2']],
                'head_camera_transforms':frame_camera_transforms(model,data,cameras['head'])})
        if video is not None:
            report['video_export'] = video.finish()
        write_json(args.output/'simulation_report.json',report)
    finally:
        if rgb_frames is not None:
            rgb_frames.close()
        if video is not None:
            video.close()
        renderer.close()
        for r in native_renderers.values(): r.close()
    print(f'UNVERIFIED simulation previews: {args.output}')
