import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_control.piper_bimanual import RAD_TO_MDEG, ArmConfig, PiperBimanual


class FakeSDK:
    def __init__(self, can_name):
        self.name = can_name
        self.calls = []
        self.stale = False
        self.error = 0
        self.fail = False
        self.joints = [0] * 6
        self.grip = 50000

    def stamp(self):
        return time.time() - (10 if self.stale else 0)

    def ConnectPort(self, **kwargs):
        self.calls.append(("connect", kwargs))

    def DisconnectPort(self):
        self.calls.append(("disconnect",))

    def GetArmJointMsgs(self):
        return NS(time_stamp=self.stamp(), joint_state=NS(**{f"joint_{i + 1}": v for i, v in enumerate(self.joints)}))

    def GetArmGripperMsgs(self):
        return NS(time_stamp=self.stamp(), gripper_state=NS(grippers_angle=self.grip))

    def GetArmStatus(self):
        return NS(time_stamp=self.stamp(), arm_status=NS(err_code=self.error, arm_status=0))

    def EnablePiper(self):
        self.calls.append(("enable",))
        return True

    def MotionCtrl_1(self, *args):
        self.calls.append(("stop", *args))

    def MotionCtrl_2(self, *args):
        self.calls.append(("mode", *args))

    def JointCtrl(self, *args):
        if self.fail:
            raise RuntimeError("CAN send failed")
        self.calls.append(("joint", *args))
        self.joints = args

    def GripperCtrl(self, *args):
        self.calls.append(("gripper", *args))
        self.grip = args[0]


@pytest.fixture
def robot():
    configs = [ArmConfig(can, [-3.0] * 6, [3.0] * 6, 0.1) for can in ("left", "right")]
    instance = PiperBimanual(*configs, sdk_factory=FakeSDK, execute=True)
    instance.connect()
    yield instance
    instance.close()


def test_sdk_units_order_and_rate_limit(robot):
    robot.enable()
    target = np.array([0.1] * 6 + [0.07] + [-0.1] * 6 + [0.03])
    sent = robot.send(target, 0.1)
    np.testing.assert_allclose(sent, [0.03] * 6 + [0.052] + [-0.03] * 6 + [0.048])
    assert ("joint", *([round(0.03 * RAD_TO_MDEG)] * 6)) in robot.arms[0].calls
    assert ("gripper", 52000, 500, 1, 0) in robot.arms[0].calls
    assert ("gripper", 48000, 500, 1, 0) in robot.arms[1].calls
    np.testing.assert_allclose(robot.read_state(), sent, atol=1e-5)


def test_read_only_never_enables_or_commands(robot):
    robot.execute = False
    robot.enable()
    robot.send(robot.read_state(), 0.1)
    robot.stop()
    assert all(call[0] == "connect" for arm in robot.arms for call in arm.calls)


@pytest.mark.parametrize("invalid", [np.full(14, np.nan), np.zeros(16), np.full(14, 10.0)])
def test_invalid_action_stops_before_joint_commands(robot, invalid):
    robot.enable()
    with pytest.raises(ValueError):
        robot.send(invalid, 0.1)
    assert all(not any(c[0] == "joint" for c in arm.calls) for arm in robot.arms)
    assert all(any(c[0] == "stop" for c in arm.calls) for arm in robot.arms)


@pytest.mark.parametrize("fault", ["stale", "error", "fail"])
def test_fault_stops_both_and_latches(robot, fault):
    robot.enable()
    setattr(robot.arms[1], fault, True)
    with pytest.raises(RuntimeError):
        robot.send(np.array([0.0] * 6 + [0.05] + [0.0] * 6 + [0.05]), 0.1)
    assert robot.faulted
    assert all(any(c[0] == "stop" for c in arm.calls) for arm in robot.arms)
    with pytest.raises(RuntimeError):
        robot.enable()


def test_lock_inactive_arm(robot):
    robot.active_arm = "left"
    robot.enable()
    sent = robot.send(np.array([0.1] * 6 + [0.08] + [0.1] * 6 + [0.08]), 0.1)
    np.testing.assert_allclose(sent[7:], robot.anchor[7:])


def test_watchdog_stops_without_new_commands(robot):
    robot.command_timeout = 0.05
    robot.enable()
    deadline = time.monotonic() + 1
    while not robot.faulted and time.monotonic() < deadline:
        time.sleep(0.01)
    assert robot.faulted
    assert all(any(c[0] == "stop" for c in arm.calls) for arm in robot.arms)


def test_gripper_calibration_roundtrip():
    configs = [
        ArmConfig(can, [-3.0] * 6, [3.0] * 6, 0.1, gripper_scale=0.5, gripper_offset_m=0.01) for can in ("l", "r")
    ]
    robot = PiperBimanual(*configs, sdk_factory=FakeSDK, execute=True)
    try:
        robot.connect()
        assert robot.read_state()[6] == pytest.approx(0.08)
        robot.enable()
        action = robot.read_state()
        action[6] = 0.1
        robot.send(action, 1.0)
        assert robot.arms[0].grip == 60000
        assert robot.read_state()[6] == pytest.approx(0.1)
    finally:
        robot.close()


def test_camera_rgb_and_freshness():
    import cv2
    from aloha_cameras import AlohaCameras, decode_rgb

    bgr = np.zeros((8, 10, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255
    ok, jpg = cv2.imencode(".jpg", bgr)
    assert ok
    rgb = decode_rgb(jpg.tobytes())
    assert rgb.shape == (3, 8, 10) and rgb[0].mean() > 250 and rgb[2].mean() < 5
    receiver = AlohaCameras("localhost", {"head": 6100, "left": 6101, "right": 6102})
    with pytest.raises(RuntimeError, match="Waiting"):
        receiver.snapshot()
    receiver.frames = {name: (time.monotonic(), rgb) for name in receiver.ports}
    assert len(receiver.snapshot()) == 3
    receiver.frames["left"] = (time.monotonic() - 2, rgb)
    with pytest.raises(RuntimeError, match="stale"):
        receiver.snapshot()
    with pytest.raises(ValueError):
        decode_rgb(b"invalid payload")


@pytest.fixture
def boundary_robot():
    class BoundarySDK(FakeSDK):
        def __init__(self, can_name):
            super().__init__(can_name)
            self.joints = [0, round(-0.03977605 * RAD_TO_MDEG), round(0.03481932 * RAD_TO_MDEG), 0, 0, 0]

    configs = [ArmConfig(name, [-3, 0, -3, -3, -3, -3], [3, 3, 0, 3, 3, 3], 0.1) for name in ("left", "right")]
    robot = PiperBimanual(*configs, sdk_factory=BoundarySDK, execute=True)
    robot.connect()
    yield robot
    robot.close()


def test_boundary_start_connects_without_commands_and_recovers_without_jump(boundary_robot):
    robot = boundary_robot
    start = robot.read_state()
    assert all(c[0] == "connect" for arm in robot.arms for c in arm.calls)
    robot.enable()
    np.testing.assert_allclose(robot.last, start)
    target = start.copy()  # noisy prediction near the same boundary is clipped to nominal limits
    previous = start
    for _ in range(8):
        sent = robot.send(target, 0.03)
        assert np.max(np.abs(sent[1:3] - previous[1:3])) <= 0.3 * 0.03 + 1e-10
        assert sent[1] >= previous[1] and sent[2] <= previous[2]
        previous = sent
    np.testing.assert_allclose(sent[1:3], [0, 0], atol=1e-10)


def test_boundary_inactive_arm_keeps_measured_anchor(boundary_robot):
    robot = boundary_robot
    robot.active_arm = "left"
    robot.enable()
    sent = robot.send(robot.read_state(), 0.03)
    np.testing.assert_allclose(sent[7:], robot.anchor[7:])


def test_minor_policy_overshoot_clips_without_expanding_hardware_limits(boundary_robot):
    robot = boundary_robot
    robot.enable()
    target = robot.last.copy()
    target[1] = -0.05230638
    previous = robot.last.copy()
    for _ in range(8):
        sent = robot.send(target, 0.03)
        assert previous[1] <= sent[1] <= 0
        assert sent[1] - previous[1] <= 0.3 * 0.03 + 1e-10
        previous = sent
    assert sent[1] == pytest.approx(0)
    target[1] = -0.081
    with pytest.raises(ValueError, match="outside configured"):
        robot.send(target, 0.03)
    assert robot.faulted


def test_right_joint4_small_prediction_overshoot_is_clipped(robot):
    robot.configs[1].joint_max[3] = np.deg2rad(100)
    robot.enable()
    robot.smooth_control = True
    target = robot.last.copy()
    target[7:13] = [-0.20198917, 1.65297902, -0.50865954, 1.80699205, 0.95715261, -2.53555536]
    before = robot.last.copy()
    robot.send(target, 0.1)
    np.testing.assert_allclose(robot.last, before)  # accepting a target does not jump to it
    assert robot.pending[10] == pytest.approx(np.deg2rad(100))
    target[10] = np.deg2rad(100) + 0.081
    with pytest.raises(ValueError, match="outside configured"):
        robot.send(target, 0.1)
    assert robot.faulted


def test_left_joint5_prediction_clips_to_hardware_lower_bound(robot):
    robot.configs[0].joint_min[4] = np.deg2rad(-70)
    robot.enable()
    robot.smooth_control = True
    target = robot.last.copy()
    target[:6] = [-0.09183726, 1.92146683, -0.26699352, 0.6446687, -1.29377401, -0.12553787]
    before = robot.last.copy()
    robot.send(target, 0.1)
    np.testing.assert_allclose(robot.last, before)
    assert robot.pending[4] == pytest.approx(np.deg2rad(-70))
    target[4] = np.deg2rad(-70) - 0.081
    with pytest.raises(ValueError, match="outside configured"):
        robot.send(target, 0.1)
    assert robot.faulted


@pytest.mark.parametrize("speed,acceleration", [(0.3, 0.6), (0.4, 0.8), (0.45, 0.9)])
@pytest.mark.parametrize("response", [0.15, 0.12])
def test_smooth_ticks_limit_speed_acceleration_and_do_not_renew_heartbeat(robot, speed, acceleration, response):
    robot.tracking_response_s = response
    robot.joint_speed = speed
    robot.joint_acceleration = acceleration
    robot.enable()
    robot.smooth_control = True  # manually clock ticks for deterministic testing
    start = robot.last.copy()
    target = start.copy()
    target[0] += 0.5
    np.testing.assert_allclose(robot.send(target, 0.1), start)
    heartbeat = robot.heartbeat
    old_velocity = 0
    for _ in range(20):
        previous = robot.last.copy()
        command = robot._send(robot.pending, 1 / 30, servo_tick=True)
        velocity = (command[0] - previous[0]) * 30
        assert abs(velocity) <= robot.joint_speed + 1e-10
        assert abs(velocity - old_velocity) <= robot.joint_acceleration / 30 + 1e-10
        old_velocity = velocity
    assert robot.heartbeat == heartbeat
    robot.stop()
    with pytest.raises(RuntimeError):
        robot._send(target, 1 / 30, servo_tick=True)


def test_faster_gripper_still_bounds_rate_acceleration_and_effort(robot):
    robot.gripper_speed = 0.04
    robot.gripper_acceleration = 0.08
    effort = robot.effort
    robot.enable()
    robot.smooth_control = True
    target = robot.last.copy()
    target[[6, 13]] = 0.09
    robot.send(target, 0.05)
    previous_velocity = np.zeros(2)
    for _ in range(12):
        previous = robot.last.copy()
        command = robot._send(robot.pending, 1 / 30, servo_tick=True)
        velocity = (command[[6, 13]] - previous[[6, 13]]) * 30
        assert np.max(np.abs(velocity)) <= 0.04 + 1e-10
        assert np.max(np.abs(velocity - previous_velocity)) <= 0.08 / 30 + 1e-10
        assert np.max(command[[6, 13]]) <= 0.09
        previous_velocity = velocity
    assert robot.effort == effort


def test_smooth_worker_stops_when_external_targets_expire():
    robot = PiperBimanual(*[ArmConfig(s, [-3]*6, [3]*6, 0.1) for s in ('l', 'r')],
                         sdk_factory=FakeSDK, execute=True, smooth_control=True, command_timeout=0.15)
    try:
        robot.connect()
        robot.enable()
        target = robot.last.copy()
        target[0] += 0.5
        robot.send(target, 0.1)
        deadline = time.monotonic() + 1
        while not robot.faulted and time.monotonic() < deadline:
            time.sleep(0.01)
        assert robot.faulted and not robot.enabled
        assert any(c[0] == 'joint' for c in robot.arms[0].calls)
        assert any(c[0] == 'stop' for c in robot.arms[0].calls)
    finally:
        robot.close()


def test_large_outside_feedback_is_readable_but_cannot_enable(boundary_robot):
    robot = boundary_robot
    robot.arms[0].joints[1] = round(-0.2 * RAD_TO_MDEG)
    assert robot.read_state()[1] < -0.19
    with pytest.raises(ValueError, match="outside configured"):
        robot.enable()
    assert all(c[0] == "connect" for arm in robot.arms for c in arm.calls)
