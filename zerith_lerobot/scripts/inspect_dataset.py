"""Inspect all numeric frames without loading models, videos or robot SDK binaries."""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "zerith_lerobot/lerobot/src"))

import numpy as np
import pyarrow.parquet as pq
from lerobot.utils.zerith_action import NAMES, ZerithActionLayout
from zerith_lerobot.utils.constants import URDF_PATH, URDF_JOINTS


def inspect(root, output):
    info = json.loads((root / "meta/info.json").read_text())
    if info['robot_type'] != 'zerith_h1_pro':
        raise ValueError('Expected zerith_h1_pro dataset')
    for key in ('action', 'observation.state'):
        if info['features'][key]['names'] != NAMES:
            raise ValueError(f'Unexpected {key} ordering')
    tables = [pq.read_table(p, columns=['action', 'observation.state'])
              for p in sorted((root / 'data').rglob('*.parquet'))]
    arrays = {key: np.concatenate([np.asarray(t[key].to_pylist()) for t in tables])
              for key in ('action', 'observation.state')}
    if len(arrays['action']) != info['total_frames'] or not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Frame count mismatch or nonfinite data')
    layout = ZerithActionLayout.fit(arrays['action'])
    joints = {j.get('name'): j for j in ET.parse(URDF_PATH).findall('joint')}
    violations = []
    rows = []
    for i, name in enumerate(NAMES):
        low, high = arrays['action'][:, i].min(), arrays['action'][:, i].max()
        if URDF_JOINTS[i]:
            limit = joints[URDF_JOINTS[i]].find('limit')
            if low < float(limit.get('lower')) or high > float(limit.get('upper')):
                violations.append(name)
        rows.append({'index': i, 'name': name, 'action_min': float(low), 'action_max': float(high),
                     'state_min': float(arrays['observation.state'][:, i].min()),
                     'state_max': float(arrays['observation.state'][:, i].max()), 'retained': i in layout.indices})
        print(f'{i:2} {name:22} action=[{low:.8f}, {high:.8f}] {"KEEP" if i in layout.indices else "CONSTANT"}')
    report = {'dataset': str(root.resolve()), 'frames': len(arrays['action']),
              'episodes': info['total_episodes'], 'fps': info['fps'], 'state_dim': 23,
              'action_dim': len(layout.indices), 'urdf_limit_violations': violations, 'dimensions': rows}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'dataset_report.json').write_text(json.dumps(report, indent=2) + '\n')
    layout.save(output / 'zerith_action_layout.json')
    print(f"state=23 action={len(layout.indices)}; URDF violations={violations}; report={output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'lerobotv3_datasets/task_003_red_300ml_coca_cola_into_basket_full23')
    parser.add_argument('--output', type=Path, default=ROOT / 'zerith_lerobot/dataset_analysis')
    args = parser.parse_args()
    inspect(args.root, args.output)
