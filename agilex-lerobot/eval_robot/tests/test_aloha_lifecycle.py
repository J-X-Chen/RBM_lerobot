import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aloha_lifecycle import TerminalCommands, slow_move


class Robot:
    def __init__(self):
        self.execute = True
        self.enabled = True
        self.command_timeout = 5
        self.state = np.array([0.01] * 6 + [0.05] + [0.01] * 6 + [0.05])
        self.last = self.state.copy()
        self.anchor = self.state.copy()
        self.sent = []
        self.closed = False

    def connect(self):
        pass

    def read_state(self):
        return self.state.copy()

    def enable(self):
        self.anchor = self.state.copy()
        self.enabled = True

    def prepare_target(self, goal):
        return np.array(goal).copy()

    def send(self, action, dt):
        self.last = np.array(action).copy()
        if self.execute:
            self.state = self.last.copy()
            self.sent.append((self.last.copy(), dt))
        return self.last.copy()

    def close(self):
        self.closed = True
        self.enabled = False

    def finish_hold(self):
        self.enabled = False


def keys():
    return NS(abort=False, quit=False, poll=lambda: None)


def test_slow_return_rate_and_feedback_convergence():
    robot = Robot()
    start = robot.state.copy()
    goal = start.copy()
    goal[:6] = 0
    assert slow_move(robot, NS(snapshot=lambda: {}), goal, keys(), frequency=100, joint_speed=0.15, timeout=1)
    previous = start
    for command, dt in robot.sent:
        assert np.max(np.abs(command[:6] - previous[:6])) <= 0.15 * dt + 1e-9
        np.testing.assert_allclose(command[[6, 13]], start[[6, 13]])
        previous = command
    np.testing.assert_allclose(robot.state, goal)


def test_low_policy_rate_still_has_smooth_start_and_finish():
    robot = Robot()
    start = robot.state.copy()
    goal = start.copy()
    goal[0] += 0.02
    assert slow_move(robot, NS(snapshot=lambda: {}), goal, keys(), frequency=10, timeout=2)
    commands = np.array([start] + [command for command, dt in robot.sent])
    increments = np.diff(commands[:, 0])
    assert all(dt <= 1 / 30 for command, dt in robot.sent)
    assert np.min(increments) >= -1e-12
    assert increments[0] < increments.max() / 4
    moving = increments[increments > 1e-12]
    assert moving[-1] < increments.max() / 4
    assert increments.max() <= 0.15 / 30 + 1e-12
    np.testing.assert_allclose(robot.state, goal)


@pytest.mark.parametrize("failure", ["abort", "camera", "feedback", "timeout"])
def test_return_failure_stops_advancing(failure):
    robot = Robot()
    camera = NS(snapshot=lambda: {})
    control = keys()
    if failure == "abort":
        control.abort = True
    elif failure == "camera":
        camera.snapshot = lambda: (_ for _ in ()).throw(RuntimeError("stale camera"))
    elif failure == "feedback":
        robot.read_state = lambda: (_ for _ in ()).throw(RuntimeError("bad feedback"))
    else:
        robot.send = lambda action, dt: robot.last  # stuck command/feedback
    with pytest.raises(RuntimeError):
        slow_move(robot, camera, robot.state * 0, control, frequency=100, timeout=0.02)
    if failure != "timeout":
        assert not robot.sent


def test_second_interrupt_requests_immediate_abort():
    control = TerminalCommands()
    control.interrupt(None, None)
    assert control.quit and not control.abort
    control.returning = True
    control.interrupt(None, None)
    assert control.abort


@pytest.mark.parametrize("mode", ["q", "ctrl_c", "fault", "preview", "r"])
def test_live_shutdown_paths(monkeypatch, tmp_path, mode):
    import eval_aloha
    import aloha_cameras
    import robot_control.remote_piper as remote

    robot = Robot()
    robot.enabled = False
    robot.execute = mode != "preview"
    initial = robot.state.copy()
    aligned = []
    monkeypatch.setattr(eval_aloha, "load_alignment_target", lambda args: initial + 0.002)
    monkeypatch.setattr(remote, "RemotePiper", lambda *a, **k: robot)
    monkeypatch.setattr(eval_aloha.sys, "stdin", NS(isatty=lambda: True))
    monkeypatch.setattr(
        aloha_cameras, "AlohaCameras", lambda *a, **k: NS(connect=lambda: None, close=lambda: None, snapshot=lambda: {})
    )

    class Keys(TerminalCommands):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def poll(self):
            if mode == "r" and not aligned:
                return "r"
            return "q" if self.quit else "s"

    control = Keys()
    monkeypatch.setattr(eval_aloha, "TerminalCommands", lambda: control)

    class Policy:
        queue = []
        calls = 0

        def reset(self):
            pass

        def warmup(self, *a):
            assert not robot.enabled

        def action(self, obs, task):
            self.calls += 1
            if self.calls == 2:
                if mode == "fault":
                    raise RuntimeError("model failure")
                if mode == "ctrl_c":
                    control.interrupt(None, None)
                else:
                    control.quit = True
            return obs["observation.state"] + 0.001

    phases = []

    def move(r, c, goal, k, **kwargs):
        phases.append(kwargs["phase"])
        if kwargs["phase"] == "alignment":
            np.testing.assert_allclose(goal, initial + 0.002)
            aligned.append(True)
            return True
        np.testing.assert_allclose(goal[:6], initial[:6])
        np.testing.assert_allclose(goal[[6, 13]], robot.state[[6, 13]])
        return True

    monkeypatch.setattr(eval_aloha, "slow_move", move)
    args = NS(
        robot_host="fake",
        robot_port=6200,
        execute=robot.execute,
        joint_speed=0.3,
        gripper_speed=0.02,
        active_arm="both",
        command_timeout=5,
        camera_timeout=2,
        image_server_address=None,
        head_camera_port=6100,
        left_camera_port=6101,
        right_camera_port=6102,
        task="test",
        policy_warmup_steps=1,
        log=tmp_path / "test.jsonl",
        steps=None if mode in ("q", "ctrl_c", "fault") else 4,
        frequency=100,
        return_joint_speed=0.15,
        return_gripper_speed=0.01,
        return_timeout=1,
    )
    if mode == "fault":
        with pytest.raises(RuntimeError, match="model failure"):
            eval_aloha.live(args, Policy())
    else:
        eval_aloha.live(args, Policy())
    assert phases == (["alignment", "return"] if mode == "r" else ["return"] if mode in ("q", "ctrl_c") else [])
    assert robot.closed
    if mode == "preview":
        assert not robot.sent
