"""Export stored robot masks as ordinary LeRobot v3 RGB video features."""
import copy
from contextlib import closing
from fractions import Fraction
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import uuid

import av
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from PIL import Image

from .dataset import DatasetFrames
from .export import dump, MASK_KEYS

OUTPUT_KEYS = {'head':'observation.images.cam_head_mask',
               'right':'observation.images.cam_right_wrist_mask',
               'left':'observation.images.cam_left_wrist_mask'}
THERMAL = 'observation.images.cam_thermal'


def stats_record(acc):
    n, pixels, lo, hi, sums, squares = acc
    mean = sums / pixels
    return dict(min=(lo/255).reshape(3,1,1).tolist(), max=(hi/255).reshape(3,1,1).tolist(),
                mean=mean.reshape(3,1,1).tolist(),
                std=np.sqrt(np.maximum(0, squares/pixels-mean**2)).reshape(3,1,1).tolist(), count=[n])


def fresh_stats():
    return [0,0,np.full(3,255.),np.zeros(3),np.zeros(3),np.zeros(3)]


def add_stats(acc, image):
    values=image.reshape(-1,3)
    acc[0]+=1; acc[1]+=len(values)
    acc[2]=np.minimum(acc[2],values.min(0));acc[3]=np.maximum(acc[3],values.max(0))
    f=values.astype(np.float64)/255
    acc[4]+=f.sum(0);acc[5]+=(f*f).sum(0)


def resolve_camera_keys(info, explicit=None):
    aliases={'head':['head','cam_head','cam_left_high'],
             'left':['left','cam_left_wrist'], 'right':['right','cam_right_wrist']}
    if explicit is not None and (not isinstance(explicit,dict) or set(explicit)!=set(aliases)):
        raise ValueError('--camera-keys must map exactly head, left, right')
    result={}
    for camera,names in aliases.items():
        candidates=([explicit[camera]] if explicit is not None else
                    [f'observation.images.{name}' for name in names if f'observation.images.{name}' in info['features']])
        if len(candidates)!=1:
            raise ValueError(f'Ambiguous or missing {camera} RGB camera; specify --camera-keys')
        key=candidates[0]
        feature=info['features'].get(key,{})
        if feature.get('dtype')!='video' or len(feature.get('shape',[]))!=3 or feature['shape'][-1]!=3:
            raise ValueError(f'Requires RGB video feature: {key}')
        result[camera]=key
    if len(set(result.values()))!=3:raise ValueError('Source camera keys must be distinct')
    return result


def export_filled_videos(args):
    source=args.dataset_root.resolve()
    destination=(args.output or source.with_name(source.name+'_rgb_mask')).resolve()
    if source==destination or source in destination.parents or destination in source.parents:
        raise ValueError('Output must be separate from input')
    if destination.exists(): raise FileExistsError(destination)
    rgb=args.fill_rgb
    lossless=getattr(args,'video_lossless',False)
    crf=getattr(args,'video_crf',23)
    if not 0<=crf<=51:raise ValueError('--video-crf must be in [0,51]')
    encoder='libx264rgb' if lossless else 'libx264'
    pixel_format='rgb24' if lossless else 'yuv420p'
    if len(rgb)!=3 or any(type(v) is not int or not 0<=v<=255 for v in rgb):
        raise ValueError('--fill-rgb needs three integers in [0,255]')
    dataset=DatasetFrames(source);info=copy.deepcopy(dataset.info)
    if info['codebase_version']!='v3.0':raise ValueError('Requires LeRobot v3.0')
    source_keys=resolve_camera_keys(info,getattr(args,'camera_keys',None))
    replace=getattr(args,'mask_replace',False)
    output_keys=source_keys if replace else OUTPUT_KEYS
    for camera, key in MASK_KEYS.items():
        if key not in info['features']:
            raise ValueError(f'Missing {key}: run mask export first, then pass the exported dataset')
    if not replace and set(OUTPUT_KEYS.values()) & set(info['features']):raise ValueError('Filled video keys already exist')
    for template in (info['data_path'],info['video_path']):
        if Path(template).is_absolute() or '..' in Path(template).parts:raise ValueError('Unsafe dataset path')
    stage=destination.with_name('.'+destination.name+'.partial-'+uuid.uuid4().hex[:8])
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,stage)
    manifest={'status':'IN_PROGRESS','source':str(source),'output':str(destination),
              'fill_rgb':rgb,'video_keys':output_keys,'source_video_keys':source_keys,'mask_replace':replace,'mask_keys':MASK_KEYS,
              'meaning':'Opaque RGB fill where stored mask > 0; no augmentation',
              'removed_feature':THERMAL,'original_rgb':'replaced in output copy only' if replace else 'preserved',
              'video_layout':'same per-camera chunk/file grouping and episode time offsets as source RGB'}
    meta_path=stage/'meta/rgb_mask_export.json';dump(meta_path,manifest)
    fps=Fraction(str(info['fps']));global_stats={c:fresh_stats() for c in output_keys};episodes={}
    try:
        total=0
        for camera,key in output_keys.items():
            prefix=f'videos/{source_keys[camera]}'
            groups={}
            for ep,meta in dataset.episodes.items():
                group=(meta[f'{prefix}/chunk_index'],meta[f'{prefix}/file_index'])
                groups.setdefault(group,[]).append((ep,meta))
            h,w,_=info['features'][source_keys[camera]]['shape']
            if not lossless and (h%2 or w%2):raise ValueError('YUV420 requires even image dimensions')
            for (chunk,file),members in sorted(groups.items()):
                relative=info['video_path'].format(video_key=key,chunk_index=chunk,file_index=file)
                path=stage/relative;path.parent.mkdir(parents=True,exist_ok=True)
                members.sort(key=lambda item: item[1][f'{prefix}/from_timestamp'])
                last_pts=-1
                with av.open(str(path),'w',format='mp4') as container:
                    stream=container.add_stream(encoder,rate=fps)
                    stream.width=w;stream.height=h;stream.pix_fmt=pixel_format
                    stream.options={'crf':'0' if lossless else str(crf),'preset':'medium'}
                    for ep,meta in members:
                        data_path=info['data_path'].format(chunk_index=meta['data/chunk_index'],file_index=meta['data/file_index'])
                        rows=pq.read_table(source/data_path,columns=['episode_index','frame_index','timestamp',*MASK_KEYS.values()],
                                           filters=[('episode_index','=',ep)]).to_pylist()
                        rows.sort(key=lambda r:r['frame_index'])
                        if len(rows)!=meta['length'] or [r['frame_index'] for r in rows]!=list(range(len(rows))):
                            raise ValueError(f'Invalid frame sequence in episode {ep}')
                        start=float(meta[f'{prefix}/from_timestamp'])
                        stop=float(meta[f'{prefix}/to_timestamp'])
                        acc=fresh_stats()
                        with closing(dataset.rgb_sequence(rows,source_keys[camera])) as reader:
                            for row,(image,_) in zip(rows,reader,strict=True):
                                timestamp=start+float(row['timestamp'])
                                pts=round(timestamp*float(fps))
                                if abs(pts/float(fps)-timestamp)>1e-4 or pts<=last_pts:
                                    raise ValueError('Source episode timestamps overlap or are not on the video FPS grid')
                                if not start<=timestamp<stop:
                                    raise ValueError('Frame outside episode video interval')
                                last_pts=pts
                                stored=row[MASK_KEYS[camera]]
                                if not stored.get('bytes'):raise ValueError('Requires embedded PNG mask bytes')
                                with Image.open(BytesIO(stored['bytes'])) as decoded:mask=np.array(decoded)
                                if mask.shape!=(h,w) or not np.isin(mask,[0,1,255]).all():raise ValueError('Invalid binary mask')
                                image=image.copy();image[mask>0]=rgb
                                add_stats(acc,image);add_stats(global_stats[camera],image)
                                frame=av.VideoFrame.from_ndarray(image,format='rgb24');frame.pts=pts;frame.time_base=1/fps
                                for packet in stream.encode(frame):container.mux(packet)
                        episodes.setdefault(ep,{})[camera]={'chunk_index':chunk,'file_index':file,
                            'from_timestamp':start,'to_timestamp':stop,'stats':stats_record(acc)}
                        if camera=='head':total+=len(rows)
                    for packet in stream.encode():container.mux(packet)
                print(f'RGB mask videos: {camera}, chunk {chunk}, file {file}, {len(members)} episodes',flush=True)
            info['features'][key]={'dtype':'video','shape':[h,w,3],'names':['height','width','channels'],
                'info':{'video.height':h,'video.width':w,'video.codec':'h264','video.pix_fmt':'gbrp' if lossless else 'yuv420p',
                        'video.fps':float(fps),'video.channels':3,'video.is_depth_map':False,'has_audio':False}}
        if total!=info['total_frames']:raise ValueError('Dataset frame count mismatch')
        info['features'].pop(THERMAL,None)
        dump(stage/'meta/info.json',info)
        stats_path=stage/'meta/stats.json'
        stats=json.loads(stats_path.read_text()) if stats_path.exists() else {}
        stats.pop(THERMAL,None)
        for c,k in output_keys.items():stats[k]=stats_record(global_stats[c])
        dump(stats_path,stats)
        for path in (stage/'meta/episodes').rglob('*.parquet'):
            table=pq.read_table(path);indices=table['episode_index'].to_pylist()
            table=table.select([k for k in table.column_names if not k.startswith((f'videos/{THERMAL}/',f'stats/{THERMAL}/'))])
            for c,k in output_keys.items():
                # Replace existing video/stat columns instead of appending duplicate names.
                table=table.select([name for name in table.column_names if not name.startswith((f'videos/{k}/',f'stats/{k}/'))])
                for field in ('chunk_index','file_index','from_timestamp','to_timestamp'):
                    table=table.append_column(f'videos/{k}/{field}',pa.array([episodes[e][c][field] for e in indices]))
                for field in ('min','max','mean','std','count'):
                    table=table.append_column(f'stats/{k}/{field}',pa.array([episodes[e][c]['stats'][field] for e in indices]))
            metadata=dict(table.schema.metadata or {});metadata.pop(b'huggingface',None)
            pq.write_table(table.replace_schema_metadata(metadata),path,compression='zstd')
        # Thermal is not an output camera. Remove only its copied data and metadata.
        for path in (stage/'data').rglob('*.parquet'):
            schema=pq.read_schema(path)
            if THERMAL in schema.names:
                table=pq.read_table(path).drop([THERMAL]);metadata=dict(table.schema.metadata or {})
                metadata.pop(b'huggingface',None)
                pq.write_table(table.replace_schema_metadata(metadata),path,compression='zstd')
        shutil.rmtree(stage/'videos'/THERMAL,ignore_errors=True)
        manifest.update(status='COMPLETE',frames=total,episodes=len(episodes),codec=encoder,
                        crf=0 if lossless else crf,lossless=lossless,
                        stats_basis='filled RGB pixels before video encoding')
        dump(meta_path,manifest)
        if destination.exists():raise FileExistsError(destination)
        os.rename(stage,destination)
    except BaseException as error:
        manifest.update(status='FAILED_INCOMPLETE_DO_NOT_TRAIN',error=str(error));dump(meta_path,manifest)
        raise
    return destination
