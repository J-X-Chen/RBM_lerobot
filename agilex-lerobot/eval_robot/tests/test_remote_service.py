"""Real TCP/JPEG transport with fake CAN/ROS; never touches real hardware."""

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agilex_image_service"))
sys.path.insert(0, str(ROOT / "agilex-lerobot/eval_robot"))
from piper_control import ArmConfig, PiperBimanual
from robot_control.remote_piper import RemotePiper
from server import Bridge, Handler, Server
from test_piper_bimanual import FakeSDK


@pytest.fixture
def network():
    robot = PiperBimanual(
        *[ArmConfig(s, [-3.0] * 6, [3.0] * 6, 0.1) for s in ("l", "r")], execute=True, sdk_factory=FakeSDK
    )
    robot.connect()
    bridge = Bridge.__new__(Bridge)
    bridge.robot = robot
    bridge.args = NS(
        execute_control=True,
        timeout=5.0,
        joint_speed=0.3,
        gripper_speed=0.02,
        camera_timeout=10.0,
        feedback_timeout=1.0,
    )
    bridge.lock = threading.RLock()
    bridge.frames = {s: (None, time.monotonic()) for s in ("head", "left", "right")}
    bridge.error = None
    bridge.stop = threading.Event()
    server = Server(("127.0.0.1", 0), Handler)
    server.bridge = bridge
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    client = RemotePiper("127.0.0.1", server.server_address[1], execute=True, camera_timeout=10.0)
    client.connect()
    try:
        yield client, bridge
    finally:
        client.close()
        bridge.stop.set()
        server.shutdown()
        server.server_close()
        thread.join(2)
        robot.close()


def test_preview_never_sends_actions_to_execute_server(network):
    client, bridge = network
    client.execute = False
    state = client.read_state()
    client.send(state + 0.01, 0.03)
    assert not bridge.robot.enabled
    assert all(call[0] == "connect" for arm in bridge.robot.arms for call in arm.calls)


def test_warmup_idle_connection_survives_then_limits_distant_target(network):
    client, bridge = network
    time.sleep(2.1)  # the baseline disconnected idle clients after two seconds
    client.read_state()
    assert not bridge.robot.faulted
    client.enable()
    time.sleep(0.04)
    target = np.array([1.0] * 6 + [0.07] + [-1.0] * 6 + [0.03])
    command = client.send(target, 1 / 30)
    assert np.max(np.abs(command[[0, 7]])) <= 0.010001
    assert abs(command[6] - 0.05) <= 0.02 / 30 + 1e-6
    client.close()
    assert bridge.robot.faulted and not bridge.robot.enabled


def test_network_invalid_target_and_stale_camera_stop(network):
    client, bridge = network
    client.enable()
    bridge.frames["left"] = (None, time.monotonic() - 20)
    with pytest.raises(RuntimeError, match="camera"):
        client.send(np.zeros(14), 0.03)
    assert bridge.robot.faulted


def test_disconnect_and_watchdog(network):
    client, bridge = network
    bridge.robot.command_timeout = 0.1
    client.enable()
    deadline = time.monotonic() + 1
    # Read-only queries must not refresh the action watchdog.
    while not bridge.robot.faulted and time.monotonic() < deadline:
        client.request({"get_state": True})
        time.sleep(0.02)
    assert bridge.robot.faulted


def test_abrupt_disconnect_stops(network):
    client, bridge = network
    client.enable()
    client.socket.close()
    client.socket = None
    deadline = time.monotonic() + 1
    while bridge.robot.enabled and time.monotonic() < deadline:
        time.sleep(0.01)
    assert bridge.robot.faulted


def test_normal_finish_holds_across_disconnect_and_can_restart(network):
    client, bridge = network
    client.enable()
    bridge.robot.command_timeout = 0.1
    client.finish_hold()
    client.close()
    time.sleep(0.2)
    assert bridge.robot.holding and bridge.robot.enabled and not bridge.robot.faulted
    assert not any(call[0] == "stop" for arm in bridge.robot.arms for call in arm.calls)
    client.connect()
    client.enable()
    assert not bridge.robot.holding
    client.close()  # A session without normal completion must still emergency-stop.
    deadline = time.monotonic() + 1
    while not bridge.robot.faulted and time.monotonic() < deadline:
        time.sleep(0.01)
    assert bridge.robot.faulted


def test_hold_still_stops_on_feedback_fault(network):
    client, bridge = network
    client.enable()
    client.finish_hold()
    client.close()
    bridge.robot.arms[0].stale = True
    deadline = time.monotonic() + 1
    while not bridge.robot.faulted and time.monotonic() < deadline:
        time.sleep(0.01)
    assert bridge.robot.faulted and not bridge.robot.enabled
    assert any(call[0] == "stop" for arm in bridge.robot.arms for call in arm.calls)


def test_accepted_target_is_not_latest_servo_output(network):
    client, bridge = network
    bridge.robot.smooth_control = True
    client.enable()
    target = client.last.copy()
    target[0] = 0.2
    command = client.send(target, 0.1)
    assert command[0] < 0.2
    assert client.accepted_target[0] == pytest.approx(0.2)


def test_server_cap_and_single_arm(network):
    client, bridge = network
    client.joint_speed = 9
    client.active_arm = "left"
    client.enable()
    assert bridge.robot.joint_speed == 0.3
    time.sleep(0.03)
    command = client.send(np.array([1.0] * 6 + [0.07] + [1.0] * 6 + [0.07]), 0.02)
    np.testing.assert_allclose(command[7:], bridge.robot.anchor[7:])


def test_model_minor_gripper_noise_clipped(network):
    client, bridge = network
    client.enable()
    command = client.send(np.array([0.0] * 6 + [-0.0001] + [0.0] * 6 + [0.1001]), 0.03)
    assert np.isfinite(command).all()
    with pytest.raises(RuntimeError, match="Gripper target"):
        client.send(np.array([0.0] * 6 + [0.3] + [0.0] * 6 + [0.1]), 0.03)
    assert bridge.robot.faulted


def test_three_jpeg_streams_rgb_and_stale():
    import cv2
    import zmq
    from aloha_cameras import AlohaCameras

    context = zmq.Context()
    pubs = [context.socket(zmq.PUB) for _ in range(3)]
    ports = {name: pub.bind_to_random_port("tcp://127.0.0.1") for name, pub in zip(("head", "left", "right"), pubs)}
    receiver = AlohaCameras("127.0.0.1", ports, max_age=0.25)
    stopped = threading.Event()
    bgr = np.zeros((480, 640, 3), np.uint8)
    bgr[:, :, 2] = 255
    ok, jpg = cv2.imencode(".jpg", bgr)
    assert ok

    def publish():
        while not stopped.wait(0.02):
            for pub in pubs:
                pub.send(jpg.tobytes())

    thread = threading.Thread(target=publish)
    thread.start()
    try:
        receiver.connect()
        images = receiver.snapshot()
        assert set(images) == {f"observation.images.{s}" for s in ports}
        for rgb in images.values():
            assert rgb.shape == (3, 480, 640) and rgb[0].mean() > 250 and rgb[2].mean() < 5
        stopped.set()
        thread.join()
        time.sleep(0.4)
        with pytest.raises(RuntimeError, match="stale"):
            receiver.snapshot()
    finally:
        stopped.set()
        thread.join()
        receiver.close()
        for pub in pubs:
            pub.close(0)
        context.term()


@pytest.mark.parametrize("smooth,async_mode,pacing", [(False, False, False), (True, False, False), (True, True, False), (True, True, True)])
def test_live_evaluator_remote_execution_lifecycle(network, monkeypatch, tmp_path, smooth, async_mode, pacing):
    """Run the actual live loop against the actual TCP handler and simulated CAN."""
    from collections import deque

    import aloha_cameras
    import eval_aloha

    client, bridge = network
    bridge.robot.smooth_control = smooth
    port = client.port
    client.close()  # live() owns its own connection
    images = {f"observation.images.{s}": np.zeros((3, 480, 640), np.uint8) for s in ("head", "left", "right")}
    events = []

    class Cameras:
        def __init__(self, host, ports, max_age):
            assert host == "127.0.0.1"
            assert list(ports.values()) == [55555, 55556, 55557]

        def connect(self):
            events.append("cameras")
            bridge.frames = {s: (None, time.monotonic()) for s in ("head", "left", "right")}

        def snapshot(self):
            bridge.frames = {s: (None, time.monotonic()) for s in ("head", "left", "right")}
            return images

        def close(self):
            events.append("closed")

    class Policy:
        queue = deque()
        n_action_steps = 50

        def reset(self):
            pass

        def predict_chunk(self, observation, task):
            time.sleep(0.01)
            target = observation["observation.state"] + 0.001
            if pacing:
                target[0] += 0.09
            return np.repeat(target[None, :], 50, axis=0)

        def warmup(self, observation, task, steps):
            assert not bridge.robot.enabled
            assert observation["observation.state"].shape == (14,)
            assert len(observation) == 4
            events.append("warmup")

        def action(self, observation, task):
            assert bridge.robot.enabled
            return observation["observation.state"] + 0.001

    monkeypatch.setattr(aloha_cameras, "AlohaCameras", Cameras)
    monkeypatch.setattr(eval_aloha.sys, "stdin", NS(isatty=lambda: True))

    class Keys:
        quit = False
        abort = False
        returning = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def poll(self):
            return "s"

    monkeypatch.setattr(eval_aloha, "TerminalCommands", Keys)
    monkeypatch.setattr(eval_aloha, "load_alignment_target", lambda args: bridge.robot.read_state())
    args = NS(
        robot_host="127.0.0.1",
        robot_port=port,
        execute=True,
        active_arm="both",
        joint_speed=0.3,
        gripper_speed=0.02,
        command_timeout=5,
        return_joint_speed=0.15,
        return_gripper_speed=0.01,
        return_timeout=5,
        camera_timeout=2,
        image_server_address=None,
        head_camera_port=55555,
        left_camera_port=55556,
        right_camera_port=55557,
        task="pick red cup",
        policy_warmup_steps=5,
        steps=15 if pacing else 3,
        frequency=30,
        async_prefetch=async_mode,
        adaptive_tracking=pacing,
        prefetch_steps=5,
        async_timeout=2,
        log=tmp_path / "actions.jsonl",
    )
    eval_aloha.live(args, Policy())
    import json

    rows = [json.loads(line) for line in args.log.read_text().splitlines()]
    policy_rows = [row for row in rows if row["phase"] == "policy"]
    assert (0 < len(policy_rows) <= args.steps if async_mode else len(policy_rows) == 3)
    assert all(row["execute"] for row in policy_rows)
    if async_mode:
        assert any(row["phase"] == "async_wait" for row in rows)
    if pacing:
        waits = [row for row in rows if row["phase"] == "tracking_wait"]
        assert waits
        assert len({row["queued_actions"] for row in waits}) == 1
        assert any(row["phase"] == "policy" and row["step"] > waits[0]["step"] for row in rows)
    assert any(row["phase"] == "return" for row in rows)
    assert all(len(row["command"]) == 14 for row in rows)
    assert events == ["cameras", "warmup", "closed"]
    assert bridge.robot.holding and bridge.robot.enabled and not bridge.robot.faulted
