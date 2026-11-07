"""Project model TCP reference sites using the same camera and image operations as masks."""
import cv2
import numpy as np


def project_tcp(model, data, camera_name, k_raw, render_size, output_size):
    rw,rh=render_size
    width,height=output_size
    cid=model.camera(camera_name).id
    # MuJoCo camera: x right, y up, looks along -z. OpenCV: y down, +z forward.
    rotation=data.cam_xmat[cid].reshape(3,3)
    results={}
    for side in ('left','right'):
        site=f'arm_{side}_tool'
        point=data.site(site).xpos.copy()
        optical=((point-data.cam_xpos[cid])@rotation)*np.array([1.,-1.,-1.])
        item={'site':site,'world_m':point.tolist(),'camera_optical_raw_m':optical.tolist(),
              'source_camera':camera_name,'pixel_xy':None,'in_frame':False,
              'occlusion_checked':False}
        if optical[2]>1e-8:
            uv=(np.asarray(k_raw)@optical)[:2]/optical[2]
            # Resize using pixel-centre convention, then flip both axes as the preview does.
            uv=(uv+.5)*np.array([width/rw,height/rh])-.5
            uv=np.array([width-1,height-1])-uv
            item['pixel_xy']=uv.tolist()
            item['in_frame']=bool(0<=uv[0]<width and 0<=uv[1]<height)
        results[side]=item
    return results


def draw_tcp(rgb, points):
    image=rgb.copy()
    for side,item in points.items():
        if not item['in_frame']:continue
        x,y=np.rint(item['pixel_xy']).astype(int)
        color=(0,255,255) if side=='left' else (255,220,0)  # RGB, not BGR
        cv2.circle(image,(x,y),7,(0,0,0),3,cv2.LINE_AA)
        cv2.drawMarker(image,(x,y),color,cv2.MARKER_CROSS,15,2,cv2.LINE_AA)
        text=f'{side[0].upper()} TCP'
        tx=max(0,min(x+10,image.shape[1]-70));ty=max(18,min(y-10,image.shape[0]-5))
        cv2.putText(image,text,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(image,text,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
    return image
