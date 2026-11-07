from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import av
import datasets
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cloak_mask.export import export_dataset, MASK_KEYS


def fixture_dataset(root, states=None, shapes=None, state_names=None):
    root.mkdir()
    states = states if states is not None else [[0., 1.], [2., 3.]]
    shapes = shapes or {c:[48,64,3] for c in MASK_KEYS}
    features = {'observation.state':dict(dtype='float32',shape=[len(states[0])],names=state_names),
                'action':dict(dtype='float32',shape=[len(states[0])],names=state_names),
                **{k:dict(dtype='int64',shape=[1],names=None) for k in ['index','episode_index','frame_index','task_index']},
                'timestamp':dict(dtype='float32',shape=[1],names=None)}
    ep = dict(episode_index=0,length=2,tasks=['test'],dataset_from_index=0,dataset_to_index=2,
              **{'data/chunk_index':0,'data/file_index':0,'meta/episodes/chunk_index':0,'meta/episodes/file_index':0})
    for camera,shape in shapes.items():
        key=f'observation.images.{camera}'
        features[key]=dict(dtype='video',shape=shape,names=['height','width','channels'],info={'video.fps':30,'video.codec':'h264','video.pix_fmt':'yuv420p','video.height':shape[0],'video.width':shape[1],'video.channels':3,'video.is_depth_map':False,'has_audio':False})
        path=root/'videos'/key/'chunk-000/file-000.mp4';path.parent.mkdir(parents=True)
        with av.open(str(path),'w') as out:
            s=out.add_stream('libx264',rate=30);s.height,s.width=shape[:2];s.pix_fmt='yuv420p'
            for i in range(2):
                f=av.VideoFrame.from_ndarray(np.full(shape,40+i,np.uint8),format='rgb24');f.pts=i
                for packet in s.encode(f):out.mux(packet)
            for packet in s.encode():out.mux(packet)
        ep.update({f'videos/{key}/chunk_index':0,f'videos/{key}/file_index':0,
                   f'videos/{key}/from_timestamp':0.,f'videos/{key}/to_timestamp':2/30})
    meta=root/'meta/episodes/chunk-000';meta.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([ep]),meta/'file-000.parquet')
    pq.write_table(pa.table({'task_index':[0],'task':['test']}),root/'meta/tasks.parquet')
    info=dict(codebase_version='v3.0',features=features,fps=30,total_frames=2,total_episodes=1,total_tasks=1,
              chunks_size=1000,data_files_size_in_mb=100,video_files_size_in_mb=200,robot_type='test',
              splits={'train':'0:1'},data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
              video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
    (root/'meta/info.json').write_text(json.dumps(info));(root/'meta/stats.json').write_text('{}')
    data=root/'data/chunk-000';data.mkdir(parents=True)
    records=[{'observation.state':s,'action':list(reversed(s)),'timestamp':i/30,'episode_index':0,'frame_index':i,'index':i,'task_index':0} for i,s in enumerate(states)]
    pq.write_table(pa.Table.from_pylist(records),data/'file-000.parquet')
    return info


def args_for(source, **kwargs):
    return SimpleNamespace(dataset_root=source,output=None,robot='aloha',config=None,mask_scope='robot',**kwargs)


def fake_renderer(args, dataset, rows, sink, stage):
    for row in rows:
        labels={}
        for c in MASK_KEYS:
            h,w=dataset.info['features']['observation.images.'+c]['shape'][:2]
            labels[c]=np.zeros((h,w),np.uint8)
            labels[c][10:20,10:20]=5
            labels[c][20:30,10:20]=3
        sink(row,labels)


def tree_hash(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}


def test_export_is_independent_lossless_and_hf_loadable(tmp_path):
    source=tmp_path/'source';fixture_dataset(source);before=tree_hash(source)
    destination=export_dataset(args_for(source),render_fn=fake_renderer)
    assert tree_hash(source)==before
    src=pq.read_table(source/'data/chunk-000/file-000.parquet');dst=pq.read_table(destination/'data/chunk-000/file-000.parquet')
    assert src.equals(dst.select(src.column_names),check_metadata=False)
    ds=datasets.Dataset.from_parquet(str(destination/'data/chunk-000/file-000.parquet'), cache_dir=str(tmp_path/'cache'))
    for key in MASK_KEYS.values():
        image=ds[0][key]
        assert image.mode=='L' and image.size==(64,48)
        assert set(np.unique(image))=={0,255}
        assert (np.array(image)>0).sum()==200
    stats=json.loads((destination/'meta/stats.json').read_text())
    assert stats[MASK_KEYS['head']]['count']==[2]
    assert stats[MASK_KEYS['head']]['mean'][0][0][0]==pytest.approx(200/(48*64))
    for p in (source/'videos').rglob('*.mp4'):
        target=destination/p.relative_to(source)
        assert p.read_bytes()==target.read_bytes() and p.stat().st_ino!=target.stat().st_ino
    with pytest.raises(FileExistsError):export_dataset(args_for(source),render_fn=fake_renderer)
    bad=args_for(source);bad.output=source/'nested'
    with pytest.raises(ValueError):export_dataset(bad,render_fn=fake_renderer)


def test_gripper_scope_and_failed_export_not_published(tmp_path):
    source=tmp_path/'source';fixture_dataset(source)
    args=args_for(source);args.mask_scope='grippers'
    dest=export_dataset(args,render_fn=fake_renderer)
    ds=datasets.Dataset.from_parquet(str(dest/'data/chunk-000/file-000.parquet'), cache_dir=str(tmp_path/'cache'))
    assert (np.array(ds[0][MASK_KEYS['left']])>0).sum()==100
    args.output=tmp_path/'failed'
    def fail(*_):raise RuntimeError('injected failure')
    with pytest.raises(RuntimeError):export_dataset(args,render_fn=fail)
    assert not args.output.exists()
    manifest=next(tmp_path.glob('.failed.partial-*/meta/cloak/export.json'))
    assert json.loads(manifest.read_text())['status']=='FAILED_INCOMPLETE_DO_NOT_TRAIN'


def test_filled_videos_are_lossless_aligned_and_non_destructive(tmp_path):
    from cloak_mask.filled_videos import export_filled_videos, OUTPUT_KEYS, THERMAL
    from cloak_mask.dataset import DatasetFrames
    source=tmp_path/'source';fixture_dataset(source)
    masked=export_dataset(args_for(source),render_fn=fake_renderer)
    before=tree_hash(masked)
    args=SimpleNamespace(dataset_root=masked,output=tmp_path/'filled',fill_rgb=[0,255,0],video_lossless=True)
    result=export_filled_videos(args)
    assert tree_hash(masked)==before
    ds=DatasetFrames(result)
    for c,key in OUTPUT_KEYS.items():
        assert ds.info['features'][key]['dtype']=='video'
        for i in range(2):
            row=ds.row(0,i);image,_=ds.rgb(row,key)
            original,_=DatasetFrames(source).rgb(row,'observation.images.'+c)
            expected=original.copy();expected[10:30,10:20]=[0,255,0]
            assert np.array_equal(image,expected)
    assert THERMAL not in ds.info['features']
    stats=json.loads((result/'meta/stats.json').read_text())
    assert stats[OUTPUT_KEYS['head']]['count']==[2]
    assert np.asarray(stats[OUTPUT_KEYS['head']]['mean']).shape==(3,1,1)
    src=pq.read_table(masked/'data/chunk-000/file-000.parquet')
    dst=pq.read_table(result/'data/chunk-000/file-000.parquet')
    assert src.equals(dst)
    with pytest.raises(FileExistsError):export_filled_videos(args)


def test_filled_videos_preserve_shared_shard_and_episode_offsets(tmp_path):
    from cloak_mask.filled_videos import export_filled_videos, OUTPUT_KEYS
    from cloak_mask.dataset import DatasetFrames
    source=tmp_path/'source';info=fixture_dataset(source)
    info['total_episodes']=2;info['splits']={'train':'0:2'}
    (source/'meta/info.json').write_text(json.dumps(info))
    ep_path=source/'meta/episodes/chunk-000/file-000.parquet'
    base=pq.read_table(ep_path).to_pylist()[0];eps=[]
    for ep in range(2):
        row=dict(base,episode_index=ep,length=1,dataset_from_index=ep,dataset_to_index=ep+1)
        for c in MASK_KEYS:
            prefix=f'videos/observation.images.{c}'
            row[prefix+'/from_timestamp']=ep/30
            row[prefix+'/to_timestamp']=(ep+1)/30
        eps.append(row)
    pq.write_table(pa.Table.from_pylist(eps),ep_path)
    data_path=source/'data/chunk-000/file-000.parquet'
    rows=pq.read_table(data_path).to_pylist()
    rows[1].update(episode_index=1,frame_index=0,timestamp=0.)
    pq.write_table(pa.Table.from_pylist(rows),data_path)
    masked=export_dataset(args_for(source),render_fn=fake_renderer)
    out=export_filled_videos(SimpleNamespace(dataset_root=masked,output=tmp_path/'merged',fill_rgb=[0,255,0],video_lossless=True))
    dataset=DatasetFrames(out)
    for c,key in OUTPUT_KEYS.items():
        assert len(list((out/'videos'/key).rglob('*.mp4')))==1
        for ep in range(2):
            metadata=dataset.episodes[ep]
            assert metadata[f'videos/{key}/file_index']==0
            assert metadata[f'videos/{key}/from_timestamp']==ep/30
            row=dataset.row(ep,0)
            actual,_=dataset.rgb(row,key)
            expected,_=DatasetFrames(source).rgb(row,'observation.images.'+c)
            expected[10:30,10:20]=[0,255,0]
            assert np.array_equal(actual,expected)


def test_default_filled_video_uses_compact_h264(tmp_path):
    from cloak_mask.filled_videos import export_filled_videos, OUTPUT_KEYS
    from cloak_mask.dataset import DatasetFrames
    source=tmp_path/'source';fixture_dataset(source)
    masked=export_dataset(args_for(source),render_fn=fake_renderer)
    out=export_filled_videos(SimpleNamespace(dataset_root=masked,output=tmp_path/'compact',fill_rgb=[0,255,0]))
    ds=DatasetFrames(out)
    for key in OUTPUT_KEYS.values():
        assert ds.info['features'][key]['info']['video.pix_fmt']=='yuv420p'
        image,_=ds.rgb(ds.row(0,1),key)
        assert image[15,15,1]>230 and image[15,15,0]<25
    manifest=json.loads((out/'meta/rgb_mask_export.json').read_text())
    assert manifest['crf']==23 and not manifest['lossless']


@pytest.mark.parametrize('renamed', [False, True])
def test_mask_replace_uses_original_names_and_preserves_source(tmp_path, renamed):
    from cloak_mask.filled_videos import export_filled_videos, OUTPUT_KEYS
    from cloak_mask.dataset import DatasetFrames
    source=tmp_path/'source';fixture_dataset(source)
    masked=export_dataset(args_for(source),render_fn=fake_renderer)
    mapping={c:'observation.images.'+c for c in MASK_KEYS}
    if renamed:
        mapping={'head':'observation.images.cam_head','left':'observation.images.cam_left_wrist','right':'observation.images.cam_right_wrist'}
        path=masked/'meta/info.json';info=json.loads(path.read_text())
        ep_path=masked/'meta/episodes/chunk-000/file-000.parquet';ep=pq.read_table(ep_path)
        stats_path=masked/'meta/stats.json';stats=json.loads(stats_path.read_text())
        for c,key in mapping.items():
            old='observation.images.'+c
            info['features'][key]=info['features'].pop(old)
            (masked/'videos'/old).rename(masked/'videos'/key)
            ep=ep.rename_columns([name.replace(old+'/',key+'/') for name in ep.column_names])
            if old in stats:stats[key]=stats.pop(old)
        path.write_text(json.dumps(info));stats_path.write_text(json.dumps(stats));pq.write_table(ep,ep_path)
    before=tree_hash(masked)
    args=SimpleNamespace(dataset_root=masked,output=tmp_path/'replaced',fill_rgb=[0,255,0],mask_replace=True,video_lossless=True)
    result=export_filled_videos(args)
    assert tree_hash(masked)==before
    ds=DatasetFrames(result)
    assert set(ds.info['features'])==set(DatasetFrames(masked).info['features'])
    assert not set(OUTPUT_KEYS.values()) & set(ds.info['features'])
    assert {p.name for p in (result/'videos').iterdir()}==set(mapping.values())
    table=pq.read_table(result/'meta/episodes/chunk-000/file-000.parquet')
    assert len(table.column_names)==len(set(table.column_names))
    for c,key in mapping.items():
        row=ds.row(0,1);rgb,_=ds.rgb(row,key)
        original,_=DatasetFrames(masked).rgb(row,key)
        original[10:30,10:20]=[0,255,0]
        assert np.array_equal(rgb,original)
    args.output=masked
    with pytest.raises(ValueError):export_filled_videos(args)
