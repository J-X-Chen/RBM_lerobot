"""H1 evaluation: offline -> read-only live preview -> r align / s run / p hold / b return / q stop."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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
from zerith_image_service.cameras import CameraSubscriber
from zerith_lerobot.eval_robot.robot_control.robot_arm_zerith import ZerithH1Controller


class EvaluationState:
    """Policy actions are allowed only after measured alignment and an explicit s key."""
    def __init__(self, start):
        start = np.asarray(start, dtype=float)
        if start.shape != (23,) or not np.isfinite(start).all():
            raise ValueError('Dataset start must be a finite 23D state')
        self.start = start[:19].copy()
        if np.any(self.start[[7, 15]] < -.03) or np.any(self.start[[7, 15]] > 1.53):
            raise ValueError('Dataset gripper feedback outside SDK tolerance')
        self.start[[7, 15]] = np.clip(self.start[[7, 15]], 0., 1.5)
        self.phase = 'idle'
        self.generation = 0
        self.tolerance = np.full(19, .03)
        self.tolerance[[7, 15]] = .05
        self.tolerance[16] = .005

    def key(self, key):
        if key in ('r', 'b'):
            self.phase = 'aligning'
            self.generation += 1
        elif key == 's' and self.phase == 'ready':
            self.phase = 'running'
            self.generation += 1
        elif key == 'p':
            self.phase = 'paused'
            self.generation += 1
        elif key == 'q':
            self.phase = 'stopped'
            self.generation += 1
        return self.phase

    def aligned(self, measured):
        if self.phase == 'aligning' and np.all(abs(np.asarray(measured)[:19]-self.start) <= self.tolerance):
            self.phase = 'ready'
            return True
        return False


def log_line(file, **record):
    file.write(json.dumps(dict(time=time.time(), **record), allow_nan=False) + '\n')
    file.flush()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, default=ROOT / 'lerobotv3_datasets/task_003_red_300ml_coca_cola_into_basket_full23')
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--mode', choices=['offline', 'preview', 'live'], default='preview')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--control-port', type=int, default=6200)
    p.add_argument('--camera-ports', nargs=3, type=int, default=[55555, 55556, 55557])
    p.add_argument('--device', default='cuda')
    p.add_argument('--task')
    p.add_argument('--frequency', type=float, default=30.)
    p.add_argument('--n-action-steps', type=int)
    p.add_argument('--async-prefetch', action='store_true')
    p.add_argument('--prefetch-steps', type=int, default=5)
    p.add_argument('--max-inference-seconds', type=float, default=15.)
    p.add_argument('--alignment-timeout', type=float, default=60.)
    p.add_argument('--preview-steps', type=int, default=1)
    p.add_argument('--log', type=Path, default=Path('eval_action_logs/zerith_eval.jsonl'))
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.execute != (args.mode == 'live'):
        parser.error('--execute must be supplied only with --mode=live')
    limits = (args.frequency, args.max_inference_seconds, args.alignment_timeout, args.preview_steps)
    if not np.isfinite(limits).all() or min(limits) <= 0:
        parser.error('Frequency, timeouts and preview steps must be finite and positive')
    if args.prefetch_steps < 0:
        parser.error('prefetch-steps must be nonnegative')
    if args.mode == 'live' and not sys.stdin.isatty():
        parser.error('Live evaluation requires an interactive terminal')
    from zerith_lerobot.eval_robot.h1_runtime import H1Policy, live_observation, load_start
    start_obs = load_start(args.dataset_root, args.episode)
    task = args.task or start_obs.get('task')
    if not isinstance(task, str) or not task.strip():
        raise ValueError('A nonempty task is required')
    runtime = H1Policy(args.checkpoint, args.device, args.n_action_steps)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    controller = cameras = None
    old_terminal = None
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        with args.log.open('x') as log:
            if args.mode == 'offline':
                start_obs['task'] = task
                actions = runtime.predict(start_obs, reset=True)
                log_line(log, mode='offline', action=actions.tolist(), full_action=runtime.layout.expand(actions).tolist())
                print(f'Offline: {actions.shape}, finite={np.isfinite(actions).all()}')
                return
            controller = ZerithH1Controller(args.host, args.control_port)
            cameras = CameraSubscriber(args.host, args.camera_ports)
            if args.mode == 'preview':
                for _ in range(args.preview_steps):
                    q, status = controller.state()
                    frames = cameras.read(server_time=status["server_time"])
                    actions = runtime.predict(live_observation(q, frames, task), reset=True)
                    log_line(log, mode='preview', state=q.tolist(), action=actions.tolist(), control=status)
                    print(f'Read-only preview: {actions.shape}; first target={actions[0].round(4)}')
                return
            session = EvaluationState(start_obs['observation.state'].numpy())
            pending = None
            queue = deque()
            sent = 0
            reset = True
            aligned_since = None
            next_action = time.monotonic()
            last_heartbeat = 0.
            print('r: align to episode first state; s: start; p: hold; b: return; j: camera snapshot; q: disarm/exit', flush=True)
            old_terminal = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            while session.phase != 'stopped':
                now = time.monotonic()
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1).lower()
                    before = session.generation
                    phase = session.key(key)
                    if key == 'q':
                        break
                    if key in ('r', 'b'):
                        if not controller.armed:
                            controller.arm()
                        controller.action(session.start)
                        aligned_since = now
                        print('Aligning slowly using H1-side rate limits...', flush=True)
                    if key == 'p' and controller.armed:
                        controller.hold()
                        print('Held. r/b aligns again before s.', flush=True)
                    if key == 'j':
                        import cv2
                        _, camera_status = controller.state()
                        for name, frame in cameras.read(timeout=.1, server_time=camera_status["server_time"]).items():
                            cv2.imwrite(str(args.log.parent / f'{name}_{time.time_ns()}.jpg'), frame)
                    if session.generation != before:
                        queue.clear()
                        reset = True
                        log_line(log, event='key', key=key, phase=phase)
                if controller.armed and now-last_heartbeat >= .1:
                    reply = controller.heartbeat()
                    if reply['control']['fault']:
                        raise RuntimeError(reply['control']['fault'])
                    last_heartbeat = now
                q, status = controller.state()
                if session.aligned(q):
                    controller.hold()
                    print('Aligned. Press s to start policy.', flush=True)
                if session.phase == 'aligning' and now-aligned_since > args.alignment_timeout:
                    raise TimeoutError('Alignment did not converge')
                if pending is not None:
                    future, generation, sent_at, started = pending
                    if now-started > args.max_inference_seconds:
                        raise TimeoutError('Policy inference timeout')
                    if future.done():
                        actions = future.result()
                        pending = None
                        if generation == session.generation and session.phase == 'running':
                            overlap = sent-sent_at
                            # Drop steps corresponding to actions already executed since the observation.
                            queue = deque(actions[overlap:overlap+runtime.n_action_steps])
                            log_line(log, event='inference', action=actions.tolist(), overlap=overlap, latency=now-started)
                if session.phase == 'running':
                    # Even while using a cached chunk, stale images stop the run.
                    frames = cameras.read(timeout=.05, server_time=status["server_time"])
                    need = not queue or (args.async_prefetch and len(queue) <= args.prefetch_steps)
                    if pending is None and need:
                        obs = live_observation(q, frames, task)
                        future = executor.submit(runtime.predict, obs, reset)
                        reset = False
                        pending = future, session.generation, sent, time.monotonic()
                    if queue and now >= next_action:
                        target = queue.popleft()
                        reply = controller.action(target)
                        log_line(log, event='action', state=q.tolist(), action=target.tolist(), control=reply['control'])
                        sent += 1
                        next_action = now + 1/args.frequency
                time.sleep(.005)
    finally:
        if old_terminal is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal)
        try:
            if controller:
                controller.close()
        finally:
            if cameras:
                cameras.close()
            executor.shutdown(wait=False, cancel_futures=True)


if __name__ == '__main__':
    main()
