"""Terminal commands and bounded, feedback-checked alignment/normal return."""

import json
import os
import select
import signal
import sys
import termios
import time
import tty

import numpy as np

JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


class TerminalCommands:
    def __init__(self):
        self.quit = False
        self.returning = False
        self.abort = False
        self.saved_terminal = None
        self.fd = None

    def __enter__(self):
        self.old_signal = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.interrupt)
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.saved_terminal = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def interrupt(self, signum, frame):
        if self.quit or self.returning:
            self.abort = True
        self.quit = True

    def poll(self):
        if self.fd is not None and select.select([self.fd], [], [], 0)[0]:
            key = os.read(self.fd, 1).decode(errors="ignore").lower()
            if key == "q":
                if self.quit or self.returning:
                    self.abort = True
                self.quit = True
            if key in ("r", "s", "q"):
                return key
        return "q" if self.quit else None

    def __exit__(self, *exc):
        signal.signal(signal.SIGINT, self.old_signal)
        if self.saved_terminal is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved_terminal)


def slow_move(
    robot,
    cameras,
    goal,
    keys,
    *,
    frequency=30,
    joint_speed=0.15,
    gripper_speed=0.01,
    timeout=60,
    log=None,
    phase="return",
):
    """No inference during motion. Stop on fault/timeout, never recover a fault here."""
    if not robot.execute or not robot.enabled:
        raise RuntimeError("Slow move requires an explicitly enabled controller")
    goal = robot.prepare_target(goal)
    state = robot.read_state()
    previous = getattr(robot, "last", None)
    previous = state.copy() if previous is None else previous.copy()
    # Alignment/return do not run inference: keep these smooth even at a low policy rate.
    period = 1 / max(30, frequency)
    limit = np.full(14, joint_speed * period)
    limit[[6, 13]] = gripper_speed * period
    origin = previous.copy()
    delta = goal - origin
    # Quintic smoothstep has zero endpoint velocity/acceleration and peak slope 1.875.
    duration = max(0.5, 1.875 * float(np.max(np.abs(delta) / (limit / period))))
    trajectory_time = 0.0
    deadline = time.monotonic() + timeout
    stable = 0
    while True:
        start = time.monotonic()
        keys.poll()
        if keys.abort:
            raise RuntimeError("Return/alignment interrupted: stopping immediately")
        if keys.quit and phase != "return":
            return False
        if start >= deadline:
            raise RuntimeError("Slow %s timed out; stopping without forcing the pose" % phase)
        cameras.snapshot()
        state = robot.read_state()
        # Advance by one period, never jump ahead after a network scheduling delay.
        trajectory_time = min(duration, trajectory_time + period)
        u = trajectory_time / duration
        blend = u * u * u * (10 + u * (-15 + 6 * u))
        desired = origin + blend * delta
        target = previous + np.clip(desired - previous, -limit, limit)
        command = robot.send(target, period)
        previous = command
        if log is not None:
            log.write(
                json.dumps(
                    dict(
                        phase=phase,
                        state=state.tolist(),
                        command=command.tolist(),
                        target=goal.tolist(),
                        time_unix_s=time.time(),
                    )
                )
                + "\n"
            )
            log.flush()
        close_command = np.max(np.abs(command - goal)) < 1e-4
        close_feedback = (
            np.max(np.abs(state[JOINTS] - goal[JOINTS])) <= 0.03
            and np.max(np.abs(state[[6, 13]] - goal[[6, 13]])) <= 0.003
        )
        stable = stable + 1 if trajectory_time >= duration and close_command and close_feedback else 0
        if stable >= 3:
            return True
        time.sleep(max(0, period - (time.monotonic() - start)))
