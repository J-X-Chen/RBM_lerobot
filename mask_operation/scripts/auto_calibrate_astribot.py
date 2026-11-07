#!/usr/bin/env python3
"""Read-only discovery + original Cloak silhouette feasibility attempt on S1 RGB."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cloak_mask.autocalibrate import run
from cloak_mask.dataset import DatasetFrames
from cloak_mask.model import AstribotVisualModel

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk-root',type=Path,default=Path('astribot_sdk_ros2'))
    p.add_argument('--dataset-root',type=Path,default=Path('lerobotv3_dataset/lerobotv3_red_cup_only'))
    p.add_argument('--output',type=Path,default=Path('artifacts/astribot_mask/automatic_calibration'))
    p.add_argument('--search-root',type=Path,action='append',default=[])
    p.add_argument('--ros-timeout',type=float,default=5.0)
    p.add_argument('--ros-capture',type=Path,help='CameraInfo/TF JSON captured on the robot using the independent ROS2 capture script')
    p.add_argument('--max-samples',type=int,default=32)
    p.add_argument('--open-tolerance',type=float,default=1.0)
    args=p.parse_args()
    if args.max_samples<10 or args.open_tolerance<=0 or args.ros_timeout<=0:
        p.error('max-samples>=10, open-tolerance>0, ros-timeout>0 required')
    raise SystemExit(run(args,DatasetFrames(args.dataset_root),AstribotVisualModel(args.sdk_root)))
