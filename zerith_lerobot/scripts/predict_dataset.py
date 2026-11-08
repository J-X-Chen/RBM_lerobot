"""Offline checkpoint inference; save reduced and SDK-order full actions without robot IO."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'lerobot/src'))
import numpy as np
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.zerith_action import NAMES, ZerithActionLayout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=Path('zerith_prediction.npz'))
    args = parser.parse_args()
    layout = ZerithActionLayout.load(args.checkpoint / 'zerith_action_layout.json')
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.device = args.device
    policy = get_policy_class(config.type).from_pretrained(args.checkpoint, config=config).eval()
    pre, post = make_pre_post_processors(config, pretrained_path=str(args.checkpoint),
        preprocessor_overrides={'device_processor': {'device': args.device}})
    data = LeRobotDataset('local/zerith', root=args.root, video_backend='pyav')
    if data.meta.robot_type != 'zerith_h1_pro' or data.meta.features['observation.state']['names'] != NAMES:
        raise ValueError('Expected Zerith full23 data')
    sample = data[args.frame]
    observation = {k: v for k, v in sample.items() if k.startswith('observation.') or k == 'task'}
    with torch.inference_mode():
        batch = pre(observation)
        normalized = policy.predict_action_chunk(batch)
        # Postprocess one time step at a time for pipelines expecting [batch, action_dim].
        actions = torch.stack([post(normalized[:, i]) for i in range(normalized.shape[1])], dim=1)
    full = layout.expand(actions).cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, action=actions.cpu().numpy(), action_full23=full,
             action_indices=np.asarray(layout.indices), state=sample['observation.state'].numpy())
    print(f'Saved action {tuple(actions.shape)}, full action {full.shape} to {args.output}')


if __name__ == '__main__':
    main()
