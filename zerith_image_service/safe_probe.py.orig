"""Astribot-style bounded H1 diagnostic: read-only, current-pose hold, then one arm joint."""
import argparse
import json
from pathlib import Path
import select
import sys
import termios
import time
import tty

import numpy as np
from .contract import ARM, BODY, NAMES, ControlClient, vector


def probe_target(origin, joint, delta, elapsed):
    target = vector(origin, 23)[:19]
    if joint is not None:
        target[joint] += np.sign(delta)*min(abs(delta), max(0., elapsed-2)*.005)
    return target


def check_feedback(origin, current, previous, dt, joint):
    if not 0 < dt < .5:
        raise RuntimeError('Feedback interval exceeded limit')
    drift = abs(current-origin)
    others = [i for i in BODY if i != joint]
    if np.max(drift[BODY]) > .03 or np.max(drift[others]) > .01:
        raise RuntimeError('Unexpected body motion')
    if np.max(abs(current[BODY]-previous[BODY])) / dt > .1:
        raise RuntimeError('Measured joint speed exceeded probe limit')
    if np.max(drift[[7, 15]]) > .05 or np.max(drift[19:21]) > .01 or np.max(abs(current[21:])) > .03:
        raise RuntimeError('Unexpected gripper/head/chassis motion')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=6200)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--joint', type=int, choices=ARM)
    p.add_argument('--delta', type=float, default=.01)
    p.add_argument('--log', type=Path, default=Path('eval_action_logs/h1_probe.jsonl'))
    args = p.parse_args()
    if not np.isfinite(args.delta) or abs(args.delta) > .02:
        p.error('abs(delta) must be <= 0.02 rad')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    client = ControlClient(args.host, args.port)
    armed = False
    old = None
    try:
        with args.log.open('x') as log:
            reply = client.request(get_state=True)
            origin = vector(reply['state'], 23)
            print(json.dumps(dict(zip(NAMES, origin.tolist())), indent=2))
            log.write(json.dumps(reply)+'\n')
            if not args.execute:
                return
            if not reply['control']['execute'] or reply['control']['fault']:
                raise RuntimeError('Service must be ready in execute mode')
            if not sys.stdin.isatty():
                raise RuntimeError('Interactive terminal required')
            print('8-second bounded test; q stops without return. Software disarm is not an emergency stop.')
            print('Joint:', NAMES[args.joint] if args.joint is not None else 'HOLD CURRENT')
            if input('Type r then Enter: ').strip() != 'r':
                return
            old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            origin = vector(client.request(arm=True)['state'], 23)
            armed = True
            previous = origin.copy()
            started = last = time.monotonic()
            while time.monotonic()-started < 8:
                if select.select([sys.stdin], [], [], 0)[0] and sys.stdin.read(1).lower() == 'q':
                    break
                time.sleep(.05)
                reply = client.request(get_state=True)
                current = vector(reply['state'], 23)
                now = time.monotonic()
                if reply['control']['fault'] or reply['control']['state_age_s'] > .5:
                    raise RuntimeError('Service feedback fault')
                check_feedback(origin, current, previous, now-last, args.joint)
                target = probe_target(origin, args.joint, args.delta, now-started)
                client.request(action=target.tolist())
                log.write(json.dumps(dict(t=now-started, state=current.tolist(), target=target.tolist(), control=reply['control']))+'\n')
                log.flush()
                previous, last = current, now
    finally:
        if old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        try:
            if armed:
                client.request(disarm=True)
        finally:
            client.close()


if __name__ == '__main__':
    main()
