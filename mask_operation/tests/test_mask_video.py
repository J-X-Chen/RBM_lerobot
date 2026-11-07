import av
import cv2
import numpy as np
import pytest
from cloak_mask.video import MaskVideos
from cloak_mask.renderer import overlay


def info():
    return {'fps':30,'features':{f'observation.images.{n}':{'shape':[h,w,3]} for n,h,w in [('head',96,128),('left',48,64),('right',48,64)]}}


def test_composite_video_order_scaling_fps_and_count(tmp_path):
    writer=MaskVideos(tmp_path,2,info())
    expected=[]; mixed=[]
    try:
        for i in range(4):
            masks={};rgbs={}
            for j,n in enumerate(writer.cameras):
                h,w=writer.shapes[n]
                a=np.zeros((h,w),np.uint8);a[7:31,3+i+10*j:23+i+10*j]=3+j
                masks[n]=a
                rgbs[n]=np.full((h,w,3),[40+60*j,60,100],np.uint8)
            parts=[cv2.resize((masks[n]>0).astype(np.uint8)*255,(64,48),interpolation=cv2.INTER_NEAREST) for n in writer.cameras]
            expected.append(np.concatenate(parts,axis=1))
            parts=[cv2.resize(overlay(rgbs[n],masks[n]),(64,48),interpolation=cv2.INTER_AREA) for n in writer.cameras]
            mixed.append(np.concatenate(parts,axis=1))
            np.testing.assert_array_equal(writer.compose(masks,rgbs)['mask_rgb'],mixed[-1])
            writer.write(masks,rgbs)
        report=writer.finish()
    finally:
        writer.close()
    assert report['frames']==4
    assert len(list(tmp_path.rglob('*.mp4')))==2
    for name,path in report['videos'].items():
        with av.open(path) as container:
            s=container.streams.video[0]
            assert s.average_rate==30 and (s.width,s.height)==(192,48)
            frames=list(container.decode(s))
            assert len(frames)==4
            for i,f in enumerate(frames):
                if name=='masks':
                    np.testing.assert_array_equal(f.to_ndarray(format='gray'),expected[i])
                else:
                    assert np.abs(f.to_ndarray(format='rgb24').astype(float)-mixed[i]).mean()<5
                assert float(f.pts*f.time_base)==pytest.approx(i/30)
    with pytest.raises(FileExistsError):
        MaskVideos(tmp_path,2,info())


def test_failed_export_does_not_publish_partial_video(tmp_path):
    writer=MaskVideos(tmp_path,0,info())
    try:
        with pytest.raises(ValueError):
            writer.write({'head':np.zeros((48,64))},{})
    finally:
        writer.close()
    assert not list(tmp_path.rglob('*.mp4'))
