#!/usr/bin/env python3
"""Run with the robot's sourced ROS2 Python; read CameraInfo, /tf, /tf_static only."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cloak_mask.ros_capture import probe_ros

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=15.0)
    a=p.parse_args()
    if a.seconds<=0 or a.seconds>60:
        p.error('seconds must be in (0,60]')
    if a.output.exists():
        p.error('Output already exists; choose a new path')
    report=probe_ros(a.seconds)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:
        json.dump(report,f,indent=2)
    print(report['status'],a.output)
