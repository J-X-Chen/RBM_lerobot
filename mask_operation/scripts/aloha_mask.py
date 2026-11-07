#!/usr/bin/env python3
"""Audit/render offline ALOHA execution-arm masks from measured LeRobot v3 feedback."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cloak_mask.aloha import DEFAULT_URDF, run

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['audit', 'render'])
    p.add_argument('--dataset-root', type=Path, default=Path('lerobotv3_dataset/aloha_cup_selection_red'))
    p.add_argument('--urdf', type=Path, default=Path(DEFAULT_URDF))
    p.add_argument('--service-root', type=Path, default=Path('agilex_image_service'))
    p.add_argument('--mesh-cache', type=Path, default=Path('artifacts/aloha_mask/mesh_cache'))
    p.add_argument('--config', type=Path)
    p.add_argument('--ros-capture', type=Path, help='Import observed CameraInfo/TF as unverified candidates')
    p.add_argument('--allow-candidate', action='store_true')
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--frames', type=int, nargs='+', default=None, help='Selected PNG preview frames; omit to export the whole episode as mask MP4')
    p.add_argument('--all-frames', action='store_true', help='Alias for omitting --frames: export the whole episode as mask MP4')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'render' and not a.config:
        p.error('render requires --config; audit writes an incomplete template')
    if a.all_frames and a.frames is not None:
        p.error('--all-frames cannot be combined with --frames')
    run(a)
