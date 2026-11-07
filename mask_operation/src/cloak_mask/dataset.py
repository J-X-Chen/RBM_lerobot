"""Read recorded feedback and timestamp-aligned RGB without importing LeRobot/torch."""
from pathlib import Path
import json

import av
import numpy as np
import pyarrow.parquet as pq


class DatasetFrames:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.info = json.loads((self.root / 'meta/info.json').read_text())
        self.episodes = {}
        for path in sorted((self.root / 'meta/episodes').rglob('*.parquet')):
            table = pq.read_table(path)
            columns = [n for n in table.column_names if not n.startswith('stats/')]
            for row in table.select(columns).to_pylist():
                self.episodes[int(row['episode_index'])] = row
        if not self.episodes:
            raise ValueError('No v3 episode metadata found')
        self._tables = {}

    def row(self, episode, frame):
        meta = self.episodes[episode]
        path = self.root / self.info['data_path'].format(
            chunk_index=meta['data/chunk_index'], file_index=meta['data/file_index'])
        if path not in self._tables:
            # Never read action as a stand-in for measured feedback.
            self._tables[path] = pq.read_table(path, columns=[
                'observation.state', 'timestamp', 'frame_index', 'episode_index', 'index']).to_pylist()
            indexed = {}
            for row in self._tables[path]:
                indexed.setdefault((row['episode_index'], row['frame_index']), []).append(row)
            self._tables[path] = indexed
        matching = self._tables[path].get((episode, frame), [])
        if len(matching) != 1:
            raise ValueError(f'Expected exactly one row for episode={episode} frame={frame}, found {len(matching)}')
        return matching[0]

    def rgb(self, row, feature, time_offset=0.0):
        meta = self.episodes[int(row['episode_index'])]
        prefix = f'videos/{feature}'
        path = self.root / self.info['video_path'].format(video_key=feature,
            chunk_index=meta[f'{prefix}/chunk_index'], file_index=meta[f'{prefix}/file_index'])
        start = float(meta[f'{prefix}/from_timestamp'])
        stop = float(meta[f'{prefix}/to_timestamp'])
        target = start + float(row['timestamp']) + time_offset
        if not start <= target < stop:
            raise ValueError(f'{feature}: requested timestamp {target} outside episode video interval [{start}, {stop})')
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            container.seek(max(0, int(target / stream.time_base)), stream=stream, backward=True)
            best, best_delta, best_time = None, float('inf'), None
            for decoded in container.decode(stream):
                if decoded.pts is None:
                    continue
                timestamp = float(decoded.pts * decoded.time_base)
                if start - 1e-6 <= timestamp < stop:
                    delta = abs(timestamp - target)
                    if delta < best_delta:
                        best, best_delta, best_time = decoded, delta, timestamp
                if timestamp >= target:
                    break
            tolerance = 0.51 / self.info['fps'] + 1e-6
            if best is None or best_delta > tolerance:
                raise ValueError(f'{feature}: no RGB within {tolerance:.6f}s of {target:.6f}s (delta={best_delta})')
            image = best.to_ndarray(format='rgb24')
        shape = tuple(self.info['features'][feature]['shape'])
        if image.shape != shape:
            raise ValueError(f'{feature}: decoded {image.shape}, expected metadata {shape}')
        return image, {'video_path': str(path), 'target_timestamp': target,
                       'decoded_timestamp': best_time, 'delta_seconds': best_delta}

    def rgb_sequence(self, rows, feature, time_offset=0.0):
        """Decode monotonic rows from one episode once, preserving nearest-PTS selection.

        The caller must close this generator if it stops before exhaustion.
        Only two decoded frames are retained, independent of episode length.
        """
        rows = iter(rows)
        first = next(rows, None)
        if first is None:
            return
        from itertools import chain
        episode = int(first['episode_index'])
        meta = self.episodes[episode]
        prefix = f'videos/{feature}'
        path = self.root / self.info['video_path'].format(video_key=feature,
            chunk_index=meta[f'{prefix}/chunk_index'], file_index=meta[f'{prefix}/file_index'])
        start, stop = float(meta[f'{prefix}/from_timestamp']), float(meta[f'{prefix}/to_timestamp'])
        first_target = start + float(first['timestamp']) + time_offset
        tolerance = 0.51 / self.info['fps'] + 1e-6
        shape = tuple(self.info['features'][feature]['shape'])
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            container.seek(max(0, int(first_target / stream.time_base)), stream=stream, backward=True)
            decoded = ((float(f.pts * f.time_base), f) for f in container.decode(stream) if f.pts is not None)
            previous, current = None, next(decoded, None)
            last_target = float('-inf')
            for row in chain([first], rows):
                target = start + float(row['timestamp']) + time_offset
                if int(row['episode_index']) != episode or target < last_target:
                    raise ValueError('rgb_sequence requires monotonic timestamps in one episode')
                if not start <= target < stop:
                    raise ValueError(f'{feature}: requested timestamp {target} outside episode video interval [{start}, {stop})')
                while current is not None and current[0] < target:
                    previous, current = current, next(decoded, None)
                choices = [item for item in (previous, current) if item is not None and start - 1e-6 <= item[0] < stop]
                best = min(choices, key=lambda item: abs(item[0]-target), default=None)
                delta = float('inf') if best is None else abs(best[0]-target)
                if best is None or delta > tolerance:
                    raise ValueError(f'{feature}: no RGB within {tolerance:.6f}s of {target:.6f}s (delta={delta})')
                image = best[1].to_ndarray(format='rgb24')
                if image.shape != shape:
                    raise ValueError(f'{feature}: decoded {image.shape}, expected metadata {shape}')
                last_target = target
                yield image, {'video_path':str(path), 'target_timestamp':target,
                              'decoded_timestamp':best[0], 'delta_seconds':delta}

    def iter_episode_rgb(self, episode):
        """Yield synchronized RGB triplets, closing all readers even on early exit."""
        from contextlib import ExitStack, closing
        rows = [self.row(episode, frame) for frame in range(int(self.episodes[episode]['length']))]
        with ExitStack() as stack:
            readers = {name: stack.enter_context(closing(self.rgb_sequence(rows, f'observation.images.{name}')))
                       for name in ('head', 'left', 'right')}
            for _ in rows:
                yield {name: next(reader)[0] for name, reader in readers.items()}
