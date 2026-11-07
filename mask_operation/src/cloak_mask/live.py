"""Shared ALOHA native renderer construction and thread-owned online masks."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np


def make_aloha_renderer(config_path, info, urdf, mesh_cache, allow_candidate=False):
    from .aloha import AlohaVisualModel, AlohaRenderer, STATE_NAMES, validate
    if info['features']['observation.state']['names'] != STATE_NAMES:
        raise ValueError('ALOHA state names/order do not match the supported execution arms')
    visual = AlohaVisualModel(urdf, mesh_cache)
    config = json.loads(Path(config_path).read_text())
    errors = validate(config, SimpleNamespace(info=info), visual, allow_candidate=allow_candidate)
    if errors:
        raise ValueError('\n'.join(errors))
    return AlohaRenderer(visual, config)


class LiveAlohaMasks:
    """All MuJoCo/EGL creation, rendering and cleanup run on one dedicated thread.

    Can be called by either synchronous inference or its async prefetch worker.
    No RGB decoding, mask augmentation or parameter fitting is performed here.
    """
    def __init__(self, config_path, info, urdf, mesh_cache, allow_candidate=False, scope='robot'):
        if scope not in ('robot', 'grippers'):
            raise ValueError('Mask scope must be robot or grippers')
        self.scope = scope
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='cloak-render')
        self.lock = Lock()
        self.closed = False
        try:
            self.renderer = self.pool.submit(make_aloha_renderer, config_path, info, urdf, mesh_cache, allow_candidate).result()
        except BaseException:
            self.pool.shutdown(wait=True)
            raise
        self.evidence = dict(config=str(Path(config_path).resolve()),
            sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),scope=scope,
            source='same native renderer as mask_generation.py; current measured 14D state',augmentation=False)

    def _render(self, state, cameras):
        labels, _ = self.renderer.render(state, camera_names=cameras)
        return {name: ((mask > 0) if self.scope == 'robot' else np.isin(mask, [5, 6])).astype(np.float32)[None]
                for name, mask in labels.items()}

    def render(self, state, cameras):
        with self.lock:
            if self.closed:
                raise RuntimeError('Mask renderer is closed')
            future = self.pool.submit(self._render, np.array(state, dtype=float, copy=True), tuple(cameras))
        return future.result()

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            future = self.pool.submit(self.renderer.close)
        try:
            future.result()
        finally:
            self.pool.shutdown(wait=True)
