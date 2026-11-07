#!/usr/bin/env python3
"""Preview sourced simulation cameras on recorded RGB; outputs are NOT validated masks."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cloak_mask.simulation import run

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--simulation-root',type=Path,default=Path('astribot_simulation'))
    p.add_argument('--sdk-root',type=Path,default=Path('astribot_sdk_ros2'))
    p.add_argument('--dataset-root',type=Path,default=Path('lerobotv3_dataset/lerobotv3_red_cup_only'))
    p.add_argument('--output',type=Path,default=Path('artifacts/astribot_mask/simulation_preview'))
    p.add_argument('--episode',type=int,default=0)
    p.add_argument('--frames',type=int,nargs='+',default=None,help='Selected PNG preview frames; omit to export the whole episode as native mask MP4')
    p.add_argument('--head-camera-y-m',type=float,default=0.,help='Diagnostic mount translation along corrected head optical +y (down), in metres; not accepted calibration')
    p.add_argument('--head-cy-offset-px',type=float,default=0.,help='Diagnostic principal-point y offset at stored head resolution; negative moves projection up')
    p.add_argument('--head-candidate',type=Path,help='Review-only head_candidate.json from fit_astribot_head.py; wrist parameters are unchanged')
    p.add_argument('--wrist-candidate',type=Path,help='Small native wrist fx/cx corrections from fit_astribot_wrists.py')
    run(p.parse_args())
