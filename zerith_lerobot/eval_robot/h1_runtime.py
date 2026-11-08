"""PI0.5/DM0.5 checkpoint runtime using the root LeRobot implementation."""
from pathlib import Path
import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.zerith_action import ZerithActionLayout, NAMES, CAMERAS


class H1Policy:
    def __init__(self, checkpoint, device='cuda', n_action_steps=None):
        checkpoint = Path(checkpoint)
        self.layout = ZerithActionLayout.load(checkpoint / 'zerith_action_layout.json')
        if self.layout.indices != list(range(19)) or self.layout.names != NAMES:
            raise ValueError('H1 live evaluation requires the 19D arms/grippers/waist checkpoint')
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device = device
        cfg.compile_model = False
        for name in ('gradient_checkpointing', 'vlm_gradient_checkpointing', 'ae_gradient_checkpointing'):
            if hasattr(cfg, name):
                setattr(cfg, name, False)
        if cfg.type not in {'pi05', 'dm05', 'act'} or getattr(cfg, 'use_relative_actions', False):
            raise ValueError('Expected absolute PI05/DM05/ACT checkpoint')
        if tuple(cfg.input_features['observation.state'].shape) != (23,) or tuple(cfg.output_features['action'].shape) != (19,):
            raise ValueError('Expected checkpoint state=23, action=19')
        if set(cfg.image_features) != set(CAMERAS):
            raise ValueError(f'Expected Zerith camera keys, got {list(cfg.image_features)}')
        self.n_action_steps = cfg.n_action_steps if n_action_steps is None else n_action_steps
        if not 1 <= self.n_action_steps <= cfg.chunk_size:
            raise ValueError('n_action_steps must be within checkpoint chunk_size')
        self.policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg).to(device).eval()
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=str(checkpoint),
            preprocessor_overrides={'device_processor': {'device': device}})

    def predict(self, observation, reset=False):
        if reset:
            self.policy.reset()
        with torch.inference_mode():
            batch = self.pre(observation)
            chunk = self.policy.predict_action_chunk(batch)
            actions = self.post(chunk).detach().float().cpu().numpy()
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != 19 or not np.isfinite(actions).all():
            raise ValueError(f'Invalid policy action chunk: {actions.shape}')
        return actions[0]


def live_observation(state, frames, task):
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (23,) or not np.isfinite(state).all():
        raise ValueError('Invalid measured state')
    obs = {'observation.state': torch.from_numpy(state.copy()), 'task': task}
    for key in CAMERAS:
        bgr = frames[key.removeprefix('observation.images.')]
        obs[key] = torch.from_numpy(bgr[..., ::-1].copy()).permute(2, 0, 1).float() / 255.
    return obs


def load_start(root, episode):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset('local/zerith', root=root, episodes=[episode], video_backend='pyav')
    if ds.meta.robot_type != 'zerith_h1_pro' or ds.meta.features['observation.state']['names'] != NAMES:
        raise ValueError('Expected Zerith full23 dataset')
    sample = ds[0]
    observation = {k: v for k, v in sample.items() if k.startswith('observation.') or k == 'task'}
    return observation
