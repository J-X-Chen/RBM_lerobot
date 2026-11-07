#!/usr/bin/env python3
"""Bounded physical camera-translation trial against RGB edges; never ground-truth calibration."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import cv2
import numpy as np
from cloak_mask.aloha import AlohaVisualModel, AlohaRenderer, DEFAULT_URDF, validate
from cloak_mask.dataset import DatasetFrames
from cloak_mask.renderer import overlay
from cloak_mask.cli import write_json, save_rgb, contact_sheet


def boundary_score(labels, distance, wrist):
    contours,_=cv2.findContours((labels>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours: return 8.
    uv=np.concatenate(contours)[:,0,:];h,w=labels.shape
    uv=uv[(uv[:,0]>3)&(uv[:,0]<w-4)&(uv[:,1]>3)&(uv[:,1]<h-4)]
    if len(uv)<20: return 8.
    dist=np.minimum(distance[uv[:,1],uv[:,0]],8.)
    if wrist:
        # Give both fingers equal weight; disappearance of either side is penalized.
        return float(np.mean([dist[uv[:,0]<w//2].mean() if np.any(uv[:,0]<w//2) else 8.,
                              dist[uv[:,0]>=w//2].mean() if np.any(uv[:,0]>=w//2) else 8.]))
    return float(dist.mean())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,default=Path('lerobotv3_dataset/aloha_cup_selection_red'))
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fit-episode',type=int,default=0)
    p.add_argument('--heldout-episode',type=int,default=1)
    p.add_argument('--frames',nargs='+',type=int,default=[180,240,300])
    a=p.parse_args()
    if a.fit_episode==a.heldout_episode: p.error('Use a different episode for held-out validation')
    ds=DatasetFrames(a.dataset_root);cfg=json.loads(a.config.read_text())
    visual=AlohaVisualModel(Path(DEFAULT_URDF),Path('artifacts/aloha_mask/mesh_cache'))
    errors=validate(cfg,ds,visual,allow_candidate=True)
    if errors: raise ValueError('\n'.join(errors))
    rows={split:[ds.row(ep,f) for f in a.frames] for split,ep in [('fit',a.fit_episode),('heldout',a.heldout_episode)]}
    a.output.mkdir(parents=True,exist_ok=False)
    low=copy.deepcopy(cfg)
    for c in low['cameras'].values():
        w,h=c['width'],c['height'];nw,nh=w//2,h//2;sx,sy=nw/w,nh/h
        k=np.array(c['K'],float);k[0,0]*=sx;k[1,1]*=sy
        k[0,2]=(k[0,2]+.5)*sx-.5;k[1,2]=(k[1,2]+.5)*sy-.5
        c.update(width=nw,height=nh,K=k.tolist())
    renderer=AlohaRenderer(visual,low)
    proposed=copy.deepcopy(cfg);candidate=copy.deepcopy(cfg)
    report=dict(status='EDGE_MATCHING_TRIAL_NOT_CALIBRATION',base_config=str(a.config),
        method='Local grid search of optical XY translation; exact MuJoCo silhouette re-render. Fixed K/D, orientation, geometry and gripper mapping. Equal wrist-finger weights, capped edge distance and displacement penalty.',
        limitations='RGB edges include cups, cables and background. Lower edge distance does not prove robot silhouette accuracy. No automatic promotion to current order.txt.',
        fit_episode=a.fit_episode,heldout_episode=a.heldout_episode,frames=a.frames,fit_resolution_scale=.5,cameras={})
    try:
        for camera in ('head','left','right'):
            samples={}
            c=low['cameras'][camera];base=np.array(c['T_parent_optical'],float)
            for split,rs in rows.items():
                samples[split]=[]
                for r in rs:
                    rgb=cv2.resize(ds.rgb(r,'observation.images.'+camera)[0],(c['width'],c['height']),interpolation=cv2.INTER_AREA)
                    edges=cv2.Canny(cv2.GaussianBlur(rgb,(3,3),.8),45,110)
                    dt=cv2.distanceTransform(255-edges,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
                    samples[split].append((r,rgb,dt,edges))
            bound=.01 if camera=='head' else .003
            cache={}
            def masks_at(xy,split):
                t=base.copy();t[:3,3]+=t[:3,:3]@np.array([*xy,0.]);c['T_parent_optical']=t.tolist()
                return [renderer.render(r['observation.state'],camera_names=[camera])[0][camera] for r,_,_,_ in samples[split]]
            def score(xy,split):
                key=(split,*np.round(xy,8))
                if key not in cache:
                    masks=masks_at(xy,split)
                    cache[key]=float(np.mean([boundary_score(m,dt,camera!='head') for m,(_,_,dt,_) in zip(masks,samples[split])]))
                return cache[key]
            def loss(xy):return score(xy,'fit')+.08*float(np.sum((np.array(xy)/bound)**2))
            best=np.zeros(2);best_loss=loss(best)
            for step in (bound/2,bound/4,bound/8):
                center=best.copy()
                for dx in (-2,-1,0,1,2):
                    for dy in (-2,-1,0,1,2):
                        xy=np.clip(center+np.array([dx,dy])*step,-bound,bound)
                        value=loss(xy)
                        if value<best_loss:best,best_loss=xy,value
            before={s:score([0,0],s) for s in samples};after={s:score(best,s) for s in samples}
            at_bound=bool(np.any(np.isclose(np.abs(best),bound)))
            accepted=after['fit']<before['fit']-.02 and after['heldout']<before['heldout']-.02 and not at_bound
            t=np.array(cfg['cameras'][camera]['T_parent_optical'],float);t[:3,3]+=t[:3,:3]@np.array([*best,0.])
            proposed['cameras'][camera]['T_parent_optical']=t.tolist()
            proposed['cameras'][camera]['edge_fit_trial']=dict(optical_xy_delta_m=best.tolist(),accepted_by_heldout_edge_score=accepted)
            if accepted:candidate['cameras'][camera]=copy.deepcopy(proposed['cameras'][camera])
            report['cameras'][camera]=dict(optical_xy_delta_m=best.tolist(),bound_m=bound,before=before,after=after,applied_to_candidate=accepted,at_bound=at_bound,rejection_reason=None if accepted else ('Search boundary reached: keep baseline pending review' if at_bound else 'Insufficient improvement on both splits'),evaluations=len(cache))
            for split in samples:
                before_masks=masks_at([0,0],split);after_masks=masks_at(best,split)
                panels=[]
                for (r,rgb,_,edges),bm,am in zip(samples[split],before_masks,after_masks):
                    panels.append(contact_sheet([rgb,overlay(rgb,bm),overlay(rgb,am),np.repeat(edges[...,None],3,axis=2)],
                        [f'{camera} ep{r["episode_index"]} f{r["frame_index"]}','Baseline','Edge-fit PROPOSAL','RGB edges (not robot labels)'],width=320))
                save_rgb(a.output/f'{camera}_{split}_review.png',np.concatenate(panels,axis=0))
            c['T_parent_optical']=base.tolist()
            print(camera,json.dumps(report['cameras'][camera]),flush=True)
    finally:renderer.close()
    for value in (candidate,proposed):
        value['status']='CANDIDATE'
        value['edge_fit_trial']=dict(base_config=str(a.config),report=str(a.output/'report.json'),note='Offline edge fit; requires visual review, not verified physical calibration.')
        errors=validate(value,ds,visual,allow_candidate=True)
        if errors:raise ValueError('\n'.join(errors))
    write_json(a.output/'calibration.json',candidate)
    write_json(a.output/'proposed_calibration.json',proposed)
    write_json(a.output/'report.json',report)
    print(a.output)

if __name__=='__main__':main()
