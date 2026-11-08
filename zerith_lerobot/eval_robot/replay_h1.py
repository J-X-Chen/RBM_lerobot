"""Replay recorded absolute H1 actions, without loading a policy. r/s/p/b/q match eval_h1."""
import argparse
import json
from pathlib import Path
import select
import sys
import termios
import time
import tty

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / 'lerobot/src'):
    sys.path.insert(0, str(path))

import numpy as np
from zerith_image_service.contract import NAMES
from zerith_lerobot.eval_robot.eval_h1 import EvaluationState, log_line
from zerith_lerobot.eval_robot.robot_control.robot_arm_zerith import ZerithH1Controller


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--control-port', type=int, default=6200)
    p.add_argument('--frequency', type=float)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--log', type=Path, default=Path('eval_action_logs/h1_replay.jsonl'))
    args = p.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset('local/zerith', root=args.dataset_root, episodes=[args.episode], video_backend='pyav')
    if ds.meta.robot_type != 'zerith_h1_pro' or ds.meta.features['action']['names'] != NAMES:
        raise ValueError('Expected H1 full23 dataset')
    table = ds.hf_dataset.with_format('numpy')
    actions = np.asarray(table['action'])
    states = np.asarray(table['observation.state'])
    if not np.isfinite(actions).all() or not np.all(actions[:, 19:] == 0):
        raise ValueError('Replay expects finite full23 absolute actions with inactive head/chassis')
    frequency = args.frequency or ds.fps
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError('Invalid frequency')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open('x') as log:
        if not args.execute:
            log_line(log, mode='preview', episode=args.episode, frames=len(actions),
                     state_start=states[0].tolist(), action_min=actions.min(0).tolist(), action_max=actions.max(0).tolist())
            print(f'Read-only dataset replay check: {len(actions)} frames, {frequency} Hz, 19 active dimensions')
            return
        if not sys.stdin.isatty():
            raise RuntimeError('Interactive terminal required')
        controller = ZerithH1Controller(args.host, args.control_port)
        old = None
        session = EvaluationState(states[0])
        index = 0
        due = last_heartbeat = 0.
        aligning_since = None
        try:
            old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            print('Dataset replay: r align, s play, p hold, b return, q stop', flush=True)
            while session.phase != 'stopped':
                now = time.monotonic()
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1).lower()
                    session.key(key)
                    if key == 'q': break
                    if key in ('r', 'b'):
                        if not controller.armed: controller.arm()
                        controller.action(session.start)
                        aligning_since = now
                        index = 0
                    elif key == 'p' and controller.armed:
                        controller.hold()
                if controller.armed and now-last_heartbeat >= .1:
                    controller.heartbeat()
                    last_heartbeat = now
                state, control = controller.state()
                if session.aligned(state):
                    controller.hold()
                    print('Aligned; s starts replay.', flush=True)
                if session.phase == 'aligning' and now-aligning_since > 60:
                    raise TimeoutError('Alignment timeout')
                if session.phase == 'running' and now >= due:
                    reply = controller.action(actions[index, :19])
                    log_line(log, event='replay', frame=index, state=state.tolist(), action=actions[index,:19].tolist(), control=reply['control'])
                    due = now + 1/frequency
                    index += 1
                    if index == len(actions):
                        controller.hold()
                        session.key('p')
                        print('Replay complete; held current pose. b returns, q exits.', flush=True)
                time.sleep(.005)
        finally:
            if old is not None: termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
            controller.close()


if __name__ == '__main__':
    main()
