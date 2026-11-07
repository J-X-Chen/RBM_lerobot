#!/usr/bin/env python3
"""Fit small native wrist fx/cx corrections from saved previews; no physical calibration claim."""
import argparse
from pathlib import Path
import numpy as np,cv2,json
from scipy.ndimage import map_coordinates
from scipy.optimize import differential_evolution
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('artifacts/astribot_mask/wrist_fitted_review'))
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
root=Path('artifacts/astribot_mask')
result={'status':'EDGE_FITTED_NATIVE_INTRINSICS_CANDIDATE_REQUIRES_REVIEW','fit_episode':0,'fit_frames':[0,60,120],'heldout_episode':1,'heldout_frames':[0,65,130],
 'method':'Robust RGB edge distance, horizontal fx/cx adjustment, soft constraint retaining image-left finger alignment. Not ground-truth accuracy.',
 'scope':'native wrist views only; head and simulation_pipeline unchanged','cameras':{}}
for cam in ['left','right']:
 samples=[]
 for split,path,frames in [('fit',root/'s1_fixed_head_masks_run1',[0,60,120]),('heldout',root/'head_fitted_three_cameras',[0,65,130])]:
  ep=0 if split=='fit' else 1
  for f in frames:
   folder=path/f'episode_{ep:04d}_frame_{f:06d}'
   rgb=cv2.imread(str(folder/f'{cam}_rgb.png'));lab=np.load(folder/f'{cam}_native_labels.npy');h,w=lab.shape
   mask=(lab>0).astype(np.uint8);mask[:h//2]=0
   edge=cv2.Canny(cv2.GaussianBlur(rgb,(3,3),.8),45,110)
   dt=cv2.distanceTransform(255-edge,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
   contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
   pts=np.concatenate(contours)[:,0,:];pts=pts[(pts[:,0]>5)&(pts[:,0]<w-6)&(pts[:,1]>h//2+3)&(pts[:,1]<h-6)]
   samples.append((split,f,pts,dt))
 def score(p,split):
  a,b=p;vals=[]
  for sp,f,uv,dt in samples:
   if sp!=split:continue
   x=a*uv[:,0]+b
   dist=map_coordinates(dt,[uv[:,1],x],order=1,mode='nearest')
   # Equal contribution from the two fingers, cap interference from object edges.
   vals.extend([np.mean(np.minimum(dist[uv[:,0]<320],12)),np.mean(np.minimum(dist[uv[:,0]>=320],12))])
  return float(np.mean(vals))
 def loss(p):
  # Keep the already-good image-left finger within a few pixels.
  anchor=p[0]*100+p[1]-100
  return score(p,'fit')+.03*max(abs(anchor)-3,0)**2
 fit=differential_evolution(loss,[(.94,1.03),(-12,12)],seed=2,tol=1e-6)
 print(cam,fit.x, 'fit',score([1,0],'fit'),score(fit.x,'fit'),'heldout',score([1,0],'heldout'),score(fit.x,'heldout'),flush=True)
 result['cameras'][cam] = {'scale_x':fit.x[0],'offset_x':fit.x[1],'fit_before':score([1,0],'fit'),'fit_after':score(fit.x,'fit'),'heldout_before':score([1,0],'heldout'),'heldout_after':score(fit.x,'heldout'),'width':640,'height':360}

(args.output/'wrist_candidate.json').write_text(json.dumps(result,indent=2)+'\n')
