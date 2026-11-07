"""Online RGB filling using the same native renderers as dataset export."""
import atexit
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue, Empty
from threading import Lock
from types import SimpleNamespace

import numpy as np

ALIASES = {
    'head': ('head', 'cam_head', 'cam_left_high'),
    'left': ('left', 'cam_left_wrist'),
    'right': ('right', 'cam_right_wrist'),
}


class LiveS1Masks:
    """Keep the export renderer alive on its owning thread across observations."""
    def __init__(self, assets, info, output, allow_candidate=False):
        if not allow_candidate:
            raise ValueError('S1 export calibration is a candidate; specify allow_candidate_mask explicitly')
        self.requests, self.results = Queue(), Queue()
        self.lock = Lock()
        self.closed = False
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='s1-mask')
        self.future = self.pool.submit(self._run, assets, info, output)

    def _run(self, assets, info, output):
        from .simulation import run
        def rows():
            while True:
                state = self.requests.get()
                if state is None:
                    return
                yield {'observation.state': state, 'frame_index': 0}
        run(SimpleNamespace(live_dataset=SimpleNamespace(info=info),
            simulation_root=assets/'astribot_simulation', sdk_root=assets/'astribot_sdk_ros2',
            head_candidate=assets/'artifacts/astribot_mask/head_fitted_review/head_candidate.json',
            wrist_candidate=assets/'artifacts/astribot_mask/wrist_fitted_review/wrist_candidate.json',
            output=output, episode=0, frames=None, mask_rows=rows(),
            mask_callback=lambda row, masks: self.results.put(masks)))

    def render(self, state, cameras):
        state = np.asarray(state, dtype=float)
        if state.shape != (25,) or not np.isfinite(state).all():
            raise ValueError('S1 masks require full measured 25D feedback, before policy state selection')
        with self.lock:
            if self.closed:
                raise RuntimeError('Mask renderer is closed')
            self.requests.put(state.copy())
            while True:
                try:
                    labels = self.results.get(timeout=.1)
                    return {c: (labels[c] > 0)[None] for c in cameras}
                except Empty:
                    if self.future.done():
                        self.future.result()
                        raise RuntimeError('S1 mask renderer stopped unexpectedly')

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.requests.put(None)
        try:
            self.future.result()
        finally:
            self.pool.shutdown(wait=True)


class RobotMask:
    def __init__(self, robot, *, assets_root, config=None, fill_rgb=(0, 255, 0),
                 allow_candidate=False, cache=None):
        color = np.asarray(fill_rgb)
        if color.shape != (3,) or color.dtype.kind not in 'iu' or np.any((color < 0) | (color > 255)):
            raise ValueError('mask_fill_rgb must contain three integers in [0,255] (RGB)')
        self.color = color.tolist()
        self.robot = robot
        self.assets = Path(assets_root).expanduser().resolve()
        self.config = Path(config) if config else self.assets/'artifacts/aloha_mask/wrist_right5mm_up7p5mm_run1/calibration.json'
        self.cache = Path(cache or Path.cwd()/'mask_operation/.cache')
        self.allow_candidate = allow_candidate
        self.provider = None
        self.lock = Lock()
        atexit.register(self.close)

    def apply(self, observation, state, *, layout):
        import torch
        with self.lock:
            keys = {c: [f'observation.images.{a}' for a in aliases
                        if f'observation.images.{a}' in observation] for c, aliases in ALIASES.items()}
            keys = {c: v for c, v in keys.items() if v}
            if not keys:
                raise ValueError('No supported RGB cameras for robot mask')
            info = {'features': {}}
            for c, names in keys.items():
                shape = tuple(observation[names[0]].shape)
                if len(shape) != 3 or shape[0 if layout == 'CHW' else 2] != 3:
                    raise ValueError(f'Expected {layout} RGB for {names[0]}')
                h, w = shape[1:] if layout == 'CHW' else shape[:2]
                info['features'][f'observation.images.{c}'] = {'shape': [h, w, 3], 'dtype': 'video'}
            if self.provider is None:
                os.environ.setdefault('MUJOCO_GL', 'egl')
                if self.robot == 'aloha':
                    from .aloha import STATE_NAMES, DEFAULT_URDF
                    from .live import LiveAlohaMasks
                    info['features']['observation.state'] = {'names': STATE_NAMES}
                    self.provider = LiveAlohaMasks(self.config, info, self.assets/DEFAULT_URDF,
                        self.cache/'meshes', self.allow_candidate, 'robot')
                else:
                    if set(keys) != set(ALIASES):
                        raise ValueError('S1 export renderer requires head and both wrist cameras')
                    self.provider = LiveS1Masks(self.assets, info, self.cache/'s1', self.allow_candidate)
            masks = self.provider.render(state, tuple(keys))
            result = dict(observation)
            for c, names in keys.items():
                for key in names:
                    image = torch.as_tensor(observation[key])
                    mask = torch.as_tensor(masks[c][0], dtype=torch.bool, device=image.device)
                    spatial = image.shape[1:] if layout == 'CHW' else image.shape[:2]
                    if tuple(mask.shape) != tuple(spatial):
                        raise ValueError(f'{key}: mask/image resolution mismatch; check calibration')
                    if image.dtype != torch.uint8 and not image.is_floating_point():
                        raise ValueError('Expected uint8 RGB or float RGB in [0,1]')
                    if image.is_floating_point() and (not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1):
                        raise ValueError('Float RGB must be in [0,1]')
                    color = torch.tensor(self.color, device=image.device, dtype=image.dtype)
                    if image.is_floating_point():
                        color = color / 255
                    result[key] = torch.where(mask[None] if layout == 'CHW' else mask[..., None],
                        color[:, None, None] if layout == 'CHW' else color, image)
            return result

    def close(self):
        with self.lock:
            if self.provider is not None:
                provider, self.provider = self.provider, None
                provider.close()
