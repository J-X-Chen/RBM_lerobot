"""Non-destructive LeRobot v3 export with embedded, lossless Cloak masks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import cv2
import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .dataset import DatasetFrames

CAMERAS = ('head', 'left', 'right')
MASK_KEYS = {c: f'observation.cloak.{c}_mask' for c in CAMERAS}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def stats_value(values):
    frames, pixels, foreground = values
    mean = foreground / pixels
    return dict(min=[[[float(foreground == pixels)]]], max=[[[float(foreground > 0)]]],
                mean=[[[mean]]], std=[[[float(np.sqrt(mean*(1-mean)))]]], count=[frames])


class MaskSink:
    """Append PNG image columns without reconstructing or changing original columns."""
    def __init__(self, stage, info, scope):
        self.stage, self.info, self.scope = stage, info, scope
        self.writer = None
        self.total = 0
        self.episode_stats = {}
        self.counts = {}
        self.global_stats = {c: [0, 0, 0] for c in CAMERAS}

    def start_file(self, relative):
        self.destination = self.stage / relative
        self.writer = None

    def start_batch(self, table):
        self.table = table
        self.images = {c: [] for c in CAMERAS}

    def __call__(self, row, labels):
        episode, frame = int(row['episode_index']), int(row['frame_index'])
        if frame != self.counts.get(episode, 0):
            raise ValueError(f'Non-contiguous or duplicate frames in episode {episode}: {frame}')
        for camera in CAMERAS:
            mask = np.asarray(labels[camera])
            shape = tuple(self.info['features'][f'observation.images.{camera}']['shape'][:2])
            if mask.shape != shape:
                raise ValueError(f'{camera}: mask {mask.shape}, expected native resolution {shape}')
            binary = (mask > 0) if self.scope == 'robot' else np.isin(mask, [5, 6])
            ok, encoded = cv2.imencode('.png', binary.astype(np.uint8)*255)
            if not ok:
                raise RuntimeError('PNG encoding failed')
            self.images[camera].append({'bytes': encoded.tobytes(), 'path': None})
            ep_stats = self.episode_stats.setdefault(episode, {c: [0, 0, 0] for c in CAMERAS})
            for accumulator in (self.global_stats[camera], ep_stats[camera]):
                for i, value in enumerate((1, binary.size, int(binary.sum()))):
                    accumulator[i] += value
        self.counts[episode] = frame + 1
        self.total += 1
        if self.total % 100 == 0:
            print(f'Masks {self.total}/{self.info["total_frames"]}; episode {episode}, frame {frame}', flush=True)

    def finish_batch(self):
        table = self.table
        features = datasets.Features.from_arrow_schema(table.schema)
        for camera, key in MASK_KEYS.items():
            if len(self.images[camera]) != len(table):
                raise ValueError('Renderer did not return one mask per input row')
            image = datasets.Image(mode='L')
            features[key] = image
            table = table.append_column(key, pa.array(self.images[camera], type=image.pa_type))
        metadata = dict(table.schema.metadata or {})
        hf = json.loads(metadata.get(b'huggingface', b'{}'))
        hf.setdefault('info', {})['features'] = features.to_dict()
        metadata[b'huggingface'] = json.dumps(hf).encode()
        table = table.replace_schema_metadata(metadata)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.destination, table.schema, compression='zstd')
        self.writer.write_table(table)
        self.table = None
        self.images = None

    def close_file(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def render_rows(args, dataset, rows, sink, stage):
    if args.robot == 'aloha':
        from .live import make_aloha_renderer
        renderer = make_aloha_renderer(args.config, dataset.info, args.urdf, args.mesh_cache,
                                       allow_candidate=args.allow_candidate)
        try:
            for row in rows:
                masks, _ = renderer.render(row['observation.state'])
                sink(row, masks)
        finally:
            renderer.close()
    else:
        from types import SimpleNamespace
        from .simulation import run
        if dataset.info['features']['observation.state']['shape'] != [25]:
            raise ValueError('S1 adapter requires the existing 25-dimensional RBM feedback convention')
        if not args.allow_candidate:
            raise ValueError('S1 uses review-only camera/state assumptions; pass --allow-candidate explicitly')
        run(SimpleNamespace(dataset_root=args.dataset_root, simulation_root=args.simulation_root,
            sdk_root=args.sdk_root, head_candidate=args.head_candidate, wrist_candidate=args.wrist_candidate,
            episode=0, frames=None, output=stage/'meta/cloak/s1_renderer', mask_rows=rows, mask_callback=sink))


def export_dataset(args, render_fn=render_rows):
    source = args.dataset_root.resolve()
    destination = (args.output or source.with_name(source.name+'_cloak')).resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError('Output must be separate from, and not contain, the input dataset')
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    dataset = DatasetFrames(source)
    info = copy.deepcopy(dataset.info)
    if info.get('codebase_version') != 'v3.0':
        raise ValueError('Only LeRobot v3.0 is supported')
    for camera, key in MASK_KEYS.items():
        if key in info['features']:
            raise ValueError(f'Dataset already contains {key}')
        image = info['features'].get(f'observation.images.{camera}')
        if image is None or len(image['shape']) != 3 or image['shape'][2] != 3:
            raise ValueError(f'Missing supported RGB feature observation.images.{camera}')
    paths = sorted({Path(info['data_path'].format(chunk_index=m['data/chunk_index'],
                    file_index=m['data/file_index'])) for m in dataset.episodes.values()})
    for path in paths:
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe dataset data_path')
    if len(dataset.episodes) != info['total_episodes']:
        raise ValueError('Episode metadata count does not match info.json')
    video_template = Path(info.get('video_path') or 'videos')
    if video_template.is_absolute() or '..' in video_template.parts:
        raise ValueError('video_path must be local and relative for a self-contained copy')
    config_paths = {'calibration': args.config} if args.robot == 'aloha' else {
        'head_candidate': args.head_candidate, 'wrist_candidate': args.wrist_candidate}
    for path in config_paths.values():
        if path is not None:
            json.loads(path.read_text())  # fail before copying a large dataset
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name('.'+destination.name+'.partial-'+uuid.uuid4().hex[:8])
    # Physical copies, never hardlinks: editing the output cannot mutate the source.
    shutil.copytree(source, stage, copy_function=shutil.copy2)
    sink = MaskSink(stage, info, args.mask_scope)
    provenance = dict(status='IN_PROGRESS',source=str(source), output=str(destination),robot=args.robot,
                      mask_scope=args.mask_scope, storage='embedded lossless grayscale PNG in Parquet',
                      meaning='255=robot pixels to hide, 0=background; loader yields float32 [0,1]',
                      original_columns='preserved without value changes', calibrations={})
    for name, path in config_paths.items():
        if path is not None:
            provenance['calibrations'][name] = dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            target = stage/'meta/cloak'/f'{name}.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    manifest = stage/'meta/cloak/export.json'
    dump(manifest, provenance)
    def rows():
        for path in paths:
            sink.start_file(path)
            try:
                for batch in pq.ParquetFile(source/path).iter_batches(batch_size=32):
                    table = pa.Table.from_batches([batch])
                    if set(MASK_KEYS.values()) & set(table.column_names):
                        raise ValueError('Existing mask columns cannot be overwritten')
                    sink.start_batch(table)
                    for row in table.select(['observation.state','timestamp','episode_index','frame_index','index']).to_pylist():
                        yield row
                    sink.finish_batch()
            finally:
                sink.close_file()
    iterator = rows()
    try:
        render_fn(args, dataset, iterator, sink, stage)
        if sink.total != info['total_frames']:
            raise ValueError(f'Frame count mismatch: {sink.total} != {info["total_frames"]}')
        for episode, meta in dataset.episodes.items():
            if sink.counts.get(episode) != int(meta['length']):
                raise ValueError(f'Incomplete episode {episode}')
        for camera, key in MASK_KEYS.items():
            h, w = info['features'][f'observation.images.{camera}']['shape'][:2]
            info['features'][key] = dict(dtype='image',shape=[h,w,1],names=['height','width','channels'])
        dump(stage/'meta/info.json', info)
        stats_path = stage/'meta/stats.json'
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        for camera, key in MASK_KEYS.items():
            stats[key] = stats_value(sink.global_stats[camera])
        dump(stats_path, stats)
        for path in (stage/'meta/episodes').rglob('*.parquet'):
            table = pq.read_table(path)
            episodes = table['episode_index'].to_pylist()
            for camera, key in MASK_KEYS.items():
                records = [stats_value(sink.episode_stats[e][camera]) for e in episodes]
                for stat in ('min','max','mean','std','count'):
                    table = table.append_column(f'stats/{key}/{stat}',pa.array([r[stat] for r in records]))
            # Include descriptors for added statistics if the source uses HF metadata.
            metadata = dict(table.schema.metadata or {})
            metadata.pop(b'huggingface', None)
            pq.write_table(table.replace_schema_metadata(metadata),path,compression='zstd')
        mapping = {f'observation.images.{c}':k for c,k in MASK_KEYS.items()}
        dump(stage/'meta/cloak_mask_keys.json',mapping)
        dump(stage/'meta/cloak/pi05_overrides.json',dict(policy=dict(cloak_enabled=True,cloak_mask_keys=mapping),
             dataset=dict(root=str(destination),video_backend='pyav',image_transforms=dict(enable=False))))
        provenance.update(status='COMPLETE_CANDIDATE_MASKS_REQUIRE_REVIEW',frames=sink.total,episodes=len(sink.counts),
                          mask_keys=mapping, training_note='Use policy.cloak_enabled=true and policy.cloak_mask_keys mapping. This does not verify camera calibration.')
        dump(manifest,provenance)
        if destination.exists():
            raise FileExistsError(f'Output appeared during generation: {destination}')
        os.rename(stage,destination)
    except BaseException as error:
        provenance.update(status='FAILED_INCOMPLETE_DO_NOT_TRAIN',error=str(error))
        dump(manifest,provenance)
        print(f'Incomplete export retained for inspection: {stage}',flush=True)
        raise
    finally:
        iterator.close()
        sink.close_file()
    print(f'New LeRobot v3 dataset: {destination}',flush=True)
    return destination
