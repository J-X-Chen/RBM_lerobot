#!/usr/bin/env python3
"""Attempt the existing Cloak wrist silhouette extraction; never certify calibration."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cloak_mask.renderer import overlay
from cloak_mask.dataset import DatasetFrames
from cloak_mask.cli import write_json, save_rgb, contact_sheet
from openpi.calibration.silhouette import CalibOptions, build_target_mask, estimate_ee_color
import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, default=Path('lerobotv3_dataset/aloha_cup_selection_red'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    ds = DatasetFrames(a.dataset_root)
    episodes = sorted(ds.episodes)
    episodes = [episodes[i] for i in np.unique(np.linspace(0,len(episodes)-1,8,dtype=int))]
    rows = [ds.row(e,int(f)) for e in episodes for f in np.linspace(0,ds.episodes[e]['length']-1,4,dtype=int)]
    report = {}
    for name, index in [('left',6), ('right',13)]:
        openings = np.array([r['observation.state'][index] for r in rows])
        # Dataset-relative wide-open selection only, NOT physical opening calibration.
        selected = [r for r in rows if abs(r['observation.state'][index]-np.max(openings)) <= .002]
        if len(selected) < 8:
            report[name] = dict(status='INSUFFICIENT_SIMILAR_OPENINGS', count=len(selected))
            continue
        frames = np.stack([cv2.resize(ds.rgb(r,f'observation.images.{name}')[0], (320,240)) for r in selected])
        targets = []
        for batch in (frames[::2], frames[1::2]):
            g = np.zeros(len(batch))  # preselected matching openings; Cloak's zero=open convention
            color = estimate_ee_color(g, batch, crop='rigid')
            targets.append(build_target_mask(g,batch,CalibOptions(crop='rigid',ee_color=tuple(color))))
        train, hold = targets
        iou = float((train & hold).sum()/max(1,(train | hold).sum()))
        save_rgb(a.output/f'{name}_silhouette_review.png', contact_sheet(
            [frames[0], overlay(frames[0],train.astype(np.uint8)*5),overlay(frames[1],hold.astype(np.uint8)*5)],
            ['Recorded RGB','Heuristic target ONLY','Held-out heuristic target'],width=480))
        cv2.imwrite(str(a.output/f'{name}_target.png'),train.astype(np.uint8)*255)
        report[name] = dict(status='HEURISTIC_TARGET_NOT_CALIBRATION', split_iou=iou,
                            area_fraction=float(train.mean()), sample_count=len(selected),
                            rows=[dict(episode=r['episode_index'],frame=r['frame_index']) for r in selected])
    report['optimization'] = dict(status='BLOCKED', reason='No recorded K or optical mount seed; unverified gripper mapping. '
        'Cloak pose fitting requires these inputs. Static dark background can also satisfy the target heuristic. '
        'No Franka/S1 defaults substituted; head is not a static wrist silhouette.')
    write_json(a.output/'report.json',report)
    print(a.output/'report.json')

if __name__ == '__main__':
    main()
