#!/usr/bin/env python3
"""Fit a review-only head camera candidate to recorded RGB edges and robot FK."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from cloak_mask.simulation import load_simulation, coupled_positions, frame_camera_transforms
from cloak_mask.renderer import mujoco, LABELS, overlay, set_frustum
from cloak_mask.config import s1_state_order
from cloak_mask.model import AstribotVisualModel
from cloak_mask.dataset import DatasetFrames
from cloak_mask.cli import save_rgb, contact_sheet, write_json


class HeadFit:
    def __init__(self,args):
        self.dataset=DatasetFrames(args.dataset_root)
        self.model=load_simulation(args.simulation_root)
        self.data=mujoco.MjData(self.model)
        self.cid=self.model.camera('head_rgbd').id
        mujoco.mj_forward(self.model,self.data)
        self.mount=np.asarray(frame_camera_transforms(self.model,self.data,'head_rgbd')['T_world_optical_preview'])
        parent=int(self.model.cam_bodyid[self.cid])
        world=np.eye(4);world[:3,:3]=self.data.xmat[parent].reshape(3,3);world[:3,3]=self.data.xpos[parent]
        self.mount=np.linalg.inv(world)@self.mount
        self.parent=parent
        shape=self.dataset.info['features']['observation.images.head']['shape']
        self.width,self.height=640,round(shape[0]*640/shape[1])
        self.full_width,self.full_height=shape[1],shape[0]
        focal=self.height/(2*np.tan(np.deg2rad(self.model.cam_fovy[self.cid])/2))
        self.k=np.array([[focal,0,(self.width-1)/2],[0,focal,(self.height-1)/2],[0,0,1.]])
        self.renderer=mujoco.Renderer(self.model,height=self.height,width=self.width)
        self.option=mujoco.MjvOption();self.option.geomgroup[:]=0;self.option.geomgroup[0]=1;self.option.sitegroup[:]=0
        self.labels=np.zeros(self.model.ngeom,np.uint8)
        self.fit_geoms=[]
        for g in range(self.model.ngeom):
            name=self.model.body(int(self.model.geom_bodyid[g])).name
            visual=name.startswith('astribot_') and int(self.model.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_MESH)
            self.model.geom_group[g]=0 if visual else 5
            if visual:
                self.labels[g]=LABELS[AstribotVisualModel.group(name)]
                # Use rigid white shells/bases, excluding uncertain finger linkage and cups.
                if name.endswith(('_link_5','_link_6','_link_7')) and 'arm_' in name or name in ('astribot_gripper_left_base','astribot_gripper_right_base'):
                    self.fit_geoms.append(g)

    def pose(self,p):
        delta=np.eye(4);delta[:3,:3]=Rotation.from_rotvec(p[:3]).as_matrix();delta[:3,3]=p[3:6]
        return self.mount@delta

    def intrinsics(self,p):
        k=self.k.copy();k[0,0]*=np.exp(p[6]);k[1,1]*=np.exp(p[6]);return k

    def state(self,row):
        q=dict(zip(s1_state_order(),row['observation.state'],strict=True))
        q.update(coupled_positions(self.model,{f'astribot_gripper_{s}_joint_L1':q[f'astribot_gripper_{s}']*.93/100 for s in ('left','right')}))
        q['astribot_chassis_zrot']=q['astribot_chassis_yaw']
        self.data.qpos[:]=self.model.qpos0
        for j in range(self.model.njnt):
            if self.model.joint(j).name in q:self.data.qpos[self.model.jnt_qposadr[j]]=q[self.model.joint(j).name]
        mujoco.mj_forward(self.model,self.data)

    def render(self,row,p,depth=False):
        self.state(row)
        parent=np.eye(4);parent[:3,:3]=self.data.xmat[self.parent].reshape(3,3);parent[:3,3]=self.data.xpos[self.parent]
        world=parent@self.pose(p)
        self.data.cam_xpos[self.cid]=world[:3,3]
        self.data.cam_xmat[self.cid]=(world[:3,:3]@np.diag([1.,-1.,-1.])).ravel()
        r=self.renderer
        r.disable_depth_rendering();r.enable_segmentation_rendering()
        r.update_scene(self.data,camera='head_rgbd',scene_option=self.option)
        set_frustum(r.scene,self.intrinsics(p),self.width,self.height)
        seg=r.render();ids=seg[...,0]
        valid=(seg[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids>=0)
        labels=np.zeros(ids.shape,np.uint8);labels[valid]=self.labels[ids[valid]]
        if not depth:return labels
        r.disable_segmentation_rendering();r.enable_depth_rendering()
        z=r.render();r.disable_depth_rendering()
        return labels,ids,z

    def sample(self,episode,frame):
        row=self.dataset.row(episode,frame)
        rgb,timing=self.dataset.rgb(row,'observation.images.head')
        small=cv2.resize(rgb,(self.width,self.height),interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
        edge=cv2.Canny(cv2.GaussianBlur(gray,(3,3),.7),45,110)
        distance=cv2.distanceTransform(255-edge,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        distance=cv2.GaussianBlur(distance,(5,5),.8)
        return {'row':row,'rgb':rgb,'small':small,'distance':distance,'edge':edge,'timing':timing}

    def points(self,sample,p):
        _,ids,z=self.render(sample['row'],p,True)
        pixels=[]
        for geom in self.fit_geoms:
            mask=(ids==geom).astype(np.uint8)
            eroded=cv2.erode(mask,np.ones((3,3),np.uint8))
            y,x=np.nonzero(mask-eroded)
            valid=(x>3)&(y>3)&(x<self.width-4)&(y<self.height-4)
            x,y=x[valid],y[valid]
            if len(x):
                idx=np.linspace(0,len(x)-1,min(90,len(x)),dtype=int)
                pixels.extend(zip(x[idx],y[idx]))
        if not pixels:raise ValueError('No visible rigid robot contours')
        uv=np.array(pixels);zz=z[uv[:,1],uv[:,0]]
        xyz=(np.c_[uv,np.ones(len(uv))]@np.linalg.inv(self.intrinsics(p)).T)*zz[:,None]
        pose=self.pose(p)
        return xyz@pose[:3,:3].T+pose[:3,3]

    def residual(self,p,samples,points):
        pose=self.pose(p);k=self.intrinsics(p);all_r=[]
        for sample,xyz in zip(samples,points):
            cam=(xyz-pose[:3,3])@pose[:3,:3]
            uv=cam@k.T;uv=uv[:,:2]/np.maximum(cam[:,2:],1e-4)
            # Distance to actual RGB edges: heuristic evidence, not labelled ground truth.
            dist=map_coordinates(sample['distance'],[uv[:,1],uv[:,0]],order=1,mode='nearest')
            outside=np.maximum(-uv[:,0],0)+np.maximum(uv[:,0]-self.width+1,0)+np.maximum(-uv[:,1],0)+np.maximum(uv[:,1]-self.height+1,0)
            all_r.append(dist+outside+100*np.maximum(.02-cam[:,2],0))
        return np.concatenate(all_r)

    def score(self,s,p):
        try:
            points=self.points(s,p)
        except ValueError:
            return {'mean_edge_distance_px_at_640':None,'median_edge_distance_px_at_640':None,'contour_samples':0,'reason':'No visible selected rigid shell contours'}
        residual=self.residual(p,[s],[points])
        return {'mean_edge_distance_px_at_640':float(np.mean(residual)),
                'median_edge_distance_px_at_640':float(np.median(residual)),
                'contour_samples':len(points)}

    def close(self):self.renderer.close()


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    engine=HeadFit(args)
    try:
        train=[];heldout=[]
        for episodes,destination in [(args.train_episodes,train),(args.validation_episodes,heldout)]:
            for ep in episodes:
                length=engine.dataset.episodes[ep]['length']
                for f in np.unique(np.linspace(0,length-1,4,dtype=int)):
                    destination.append(engine.sample(ep,int(f)))
        zero=np.zeros(7)
        # The previously inspected translation trial is an initialization only.
        initial=zero.copy();initial[4]=.12
        points=[engine.points(s,initial) for s in train]
        lower=np.array([-.25,-.25,-.25,-.12,0.,0.,np.log(.8)])
        upper=np.array([.25,.25,.25,.12,.25,.12,np.log(1.2)])
        starts=[]
        for y in (.06,.12,.18):
            p=zero.copy();p[4]=y;starts.append(p)
        trials=[]
        for i,start in enumerate(starts):
            fit=least_squares(engine.residual,start,args=(train,points),bounds=(lower,upper),loss='soft_l1',f_scale=2.,diff_step=1e-4,max_nfev=160)
            trials.append(fit)
            print('fit',i,'cost',fit.cost,'params',fit.x.tolist(),flush=True)
        best=min(trials,key=lambda x:x.cost)
        # Refresh visible contour samples once, then refine the same shared camera.
        points=[engine.points(s,best.x) for s in train]
        best=least_squares(engine.residual,best.x,args=(train,points),bounds=(lower,upper),loss='soft_l1',f_scale=2.,diff_step=1e-4,max_nfev=160)
        fitted=best.x
        records=[];overview=[]
        for split,samples in [('fit',train),('heldout',heldout)]:
            sheets=[]
            for s in samples:
                row=s['row'];ep,f=row['episode_index'],row['frame_index']
                target=args.output/f'{split}_ep{ep:03d}_frame{f:06d}';target.mkdir()
                before=engine.render(row,zero);after=engine.render(row,fitted)
                before=cv2.resize(before,(engine.full_width,engine.full_height),interpolation=cv2.INTER_NEAREST)
                after=cv2.resize(after,(engine.full_width,engine.full_height),interpolation=cv2.INTER_NEAREST)
                save_rgb(target/'rgb.png',s['rgb']);save_rgb(target/'before.png',overlay(s['rgb'],before));save_rgb(target/'candidate_overlay.png',overlay(s['rgb'],after))
                cv2.imwrite(str(target/'candidate_robot_mask.png'),(after>0).astype(np.uint8)*255);np.save(target/'candidate_labels.npy',after)
                sheet=contact_sheet([s['rgb'],overlay(s['rgb'],before),overlay(s['rgb'],after)],
                    [f'{split} ep{ep} f{f}: RGB','Before: native','FITTED CANDIDATE - REVIEW'],width=480)
                save_rgb(target/'comparison.png',sheet);sheets.append(sheet)
                record={'split':split,'episode':ep,'frame':f,'before':engine.score(s,zero),'initial_trial':engine.score(s,initial),'candidate':engine.score(s,fitted),'video_alignment':s['timing']};records.append(record)
            save_rgb(args.output/f'{split}_overview.jpg',np.concatenate(sheets,axis=0))
        k=engine.intrinsics(fitted);scale=engine.full_width/engine.width
        k[:2,:2]*=scale;k[:2,2]=(k[:2,2]+.5)*scale-.5
        report={'status':'FITTED_CANDIDATE_REQUIRES_USER_REVIEW_NOT_VERIFIED_CALIBRATION',
            'method':'Robust multi-frame rigid-shell contour to real RGB Canny edge distance; fixed correspondences refreshed once. No ground-truth masks or manual labels.',
            'train_episodes':args.train_episodes,'validation_episodes':args.validation_episodes,
            'parameter_order':['optical_rx_rad','optical_ry_rad','optical_rz_rad','optical_tx_m','optical_ty_m','optical_tz_m','log_focal_scale'],
            'parameters':fitted.tolist(),'bounds':[lower.tolist(),upper.tolist()],
            'parent_link':engine.model.body(engine.parent).name,'T_parent_optical':engine.pose(fitted).tolist(),'K':k.tolist(),
            'width':engine.full_width,'height':engine.full_height,'distortion_model':'none','distortion_coefficients':[],
            'convention':'OpenCV optical frame AFTER the reviewed 180-degree roll; do not flip again.',
            'assumptions':['Zero distortion is an experiment assumption, not measured.','Centered principal point retained to limit pose/intrinsic ambiguity.','Shared fx/fy scale fitted; SDK state order remains the candidate mapping.','Edge alignment can match unrelated edges; numeric score is not calibration accuracy.','Estimated pose may absorb errors in model geometry or intrinsics; not a measured mounting displacement.'],
            'records':records}
        write_json(args.output/'head_candidate.json',report)
        print('Saved candidate and comparisons:',args.output,flush=True)
    finally:engine.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--simulation-root',type=Path,default=Path('astribot_simulation'))
    p.add_argument('--dataset-root',type=Path,default=Path('lerobotv3_dataset/lerobotv3_red_cup_only'))
    p.add_argument('--output',type=Path,default=Path('artifacts/astribot_mask/head_fitted_candidate'))
    p.add_argument('--train-episodes',type=int,nargs='+',default=[0,2,4])
    p.add_argument('--validation-episodes',type=int,nargs='+',default=[1,3])
    args=p.parse_args()
    if set(args.train_episodes)&set(args.validation_episodes):p.error('Fit and validation episodes must be disjoint')
    run(args)
