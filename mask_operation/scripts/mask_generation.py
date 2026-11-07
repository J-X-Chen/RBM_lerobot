#!/usr/bin/env python3
"""Copy a LeRobot v3 dataset and add native Cloak masks; never overwrite the source."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'mask_operation/src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets-root', type=Path, default=Path.home()/'code_file/cloak')
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--robot',choices=['aloha','s1'],help='Required for mask rendering')
    p.add_argument('--output-mode',choices=['masks','filled-videos'],default='masks')
    p.add_argument('--fill-rgb',type=int,nargs=3,default=[0,255,0])
    p.add_argument('--video-crf',type=int,default=23,help='Filled videos: H264 quality 0-51 (lower = larger)')
    p.add_argument('--video-lossless',action='store_true',help='Opt-in lossless RGB encoding; much larger files')
    p.add_argument('--mask_replace','--mask-replace',action='store_true',
                   help='filled-videos: replace original RGB streams only inside the new output copy')
    p.add_argument('--camera-keys',type=json.loads,
                   help='Optional JSON mapping head/left/right to exact source RGB feature names')
    p.add_argument('--output',type=Path,help='New directory; default: sibling <dataset>_cloak')
    p.add_argument('--mask-scope',choices=['robot','grippers'],default='robot',help='robot matches the reviewed previews; grippers keeps only labels 5/6')
    p.add_argument('--allow-candidate',action='store_true',help='Explicitly allow the currently reviewed but unverified calibration candidates')
    p.add_argument('--config',type=Path)
    p.add_argument('--head-candidate',type=Path)
    p.add_argument('--wrist-candidate',type=Path)
    p.add_argument('--urdf',type=Path)
    p.add_argument('--mesh-cache',type=Path,default=ROOT/'mask_operation/.cache/meshes')
    p.add_argument('--simulation-root',type=Path)
    p.add_argument('--sdk-root',type=Path)
    args = p.parse_args()
    if args.config is None:
        args.config = args.assets_root/'artifacts/aloha_mask/wrist_right5mm_up7p5mm_run1/calibration.json'
    if args.head_candidate is None:
        args.head_candidate = args.assets_root/'artifacts/astribot_mask/head_fitted_review/head_candidate.json'
    if args.wrist_candidate is None:
        args.wrist_candidate = args.assets_root/'artifacts/astribot_mask/wrist_fitted_review/wrist_candidate.json'
    if args.urdf is None:
        args.urdf = args.assets_root/'mobile_aloha_sim/aloha_new_description/urdf/aloha_tracer2_dabai_dark.urdf'
    if args.simulation_root is None:
        args.simulation_root = args.assets_root/'astribot_simulation'
    if args.sdk_root is None:
        args.sdk_root = args.assets_root/'astribot_sdk_ros2'
    if args.output_mode == 'filled-videos':
        from cloak_mask.filled_videos import export_filled_videos
        print(export_filled_videos(args))
        return
    if args.robot is None:
        p.error('--robot is required for masks mode')
    if args.mask_replace:
        p.error('--mask_replace requires --output-mode filled-videos')
    from cloak_mask.export import export_dataset
    export_dataset(args)

if __name__=='__main__':main()
