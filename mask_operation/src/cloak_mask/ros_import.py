"""Extract unambiguous intrinsics and static optical-to-link TF candidates."""
from collections import deque
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation


def nearest_static_mount(frame, transforms, model_links):
    graph={}
    for item in transforms:
        if item['topic']!='/tf_static':
            continue
        matrix=np.eye(4)
        matrix[:3,:3]=Rotation.from_quat(item['quaternion_xyzw']).as_matrix()
        matrix[:3,3]=item['translation']
        graph.setdefault(item['child'],[]).append((item['parent'],matrix))
        graph.setdefault(item['parent'],[]).append((item['child'],np.linalg.inv(matrix)))
    queue=deque([(frame,np.eye(4),[frame])])
    seen={frame}
    matches=[]
    while queue:
        current,matrix,path=queue.popleft()
        if matches and len(path)>len(matches[0][2]):
            break
        if current in model_links:
            matches.append((current,matrix,path));continue
        for target,edge in graph.get(current,[]):
            if target not in seen:
                seen.add(target);queue.append((target,edge@matrix,path+[target]))
    if len(matches)!=1:
        raise ValueError(f'No unique nearest SDK link in static TF for {frame}: {[m[0] for m in matches]}')
    return matches[0]


def import_candidates(config,capture,model):
    result=deepcopy(config)
    notes=[]
    for name,cfg in result['cameras'].items():
        candidates=[item for item in capture.get('camera_info',[])
                    if name in (item['topic']+' '+item['frame_id']).lower()
                    and item['width']==cfg['width'] and item['height']==cfg['height']]
        if len(candidates)!=1:
            notes.append(f'{name}: expected one CameraInfo matching name and stored resolution, found {len(candidates)}')
            continue
        item=candidates[0]
        k=np.asarray(item['K'],float).reshape(3,3)
        if k[0,0]<=0 or k[1,1]<=0 or not np.isfinite(k).all():
            notes.append(f'{name}: invalid/uninitialized CameraInfo K');continue
        if item.get('binning_x',0)>1 or item.get('binning_y',0)>1 or any(item.get('roi',{}).get(x,0) for x in ('x_offset','y_offset','width','height')):
            notes.append(f'{name}: binned/ROI CameraInfo needs explicit image geometry handling');continue
        if item['distortion_model'] not in ('plumb_bob','rational_polynomial') or len(item['D']) not in (4,5,8):
            notes.append(f'{name}: unsupported or unspecified raw-image distortion model');continue
        cfg['K']=k.tolist()
        cfg['distortion_model']='opencv'
        cfg['distortion_coefficients']=item['D']
        cfg['source']=f"ROS2 CameraInfo {item['topic']}, optical frame {item['frame_id']}, stamp {item.get('stamp')}; RAW RGB candidate"
        cfg['confirmed_recording_setup']=False
        try:
            link,t,path=nearest_static_mount(item['frame_id'],capture.get('tf',[]),set(model.links))
            cfg['parent_link']=link
            cfg['T_parent_optical']=t.tolist()
            cfg['source']+='; static TF chain '+' -> '.join(path)
        except ValueError as exc:
            notes.append(str(exc))
    return result,notes
