"""Deployment smoke check; no ROS, CAN, or network connections."""

import threading
import time
from types import SimpleNamespace

import numpy as np
from server import Bridge


class Robot:
    anchor = np.zeros(14)
    enabled = False
    faulted = False
    holding = False
    smooth_control = False
    joint_speed = 0.3
    gripper_speed = 0.02

    def read_state(self):
        return np.zeros(14)

    def enable(self):
        self.enabled = True

    def stop(self):
        self.enabled = False
        self.faulted = True

    def send(self, action, dt):
        assert self.enabled and 0 < dt <= 0.1
        return np.asarray(action)


b = Bridge.__new__(Bridge)
b.lock = threading.RLock()
b.robot = Robot()
b.frames = {s: (None, time.monotonic()) for s in ("head", "left", "right")}
b.error = None
b.args = SimpleNamespace(execute_control=True, timeout=5, camera_timeout=2, joint_speed=0.3, gripper_speed=0.02)
assert b.request({"get_state": True})["protocol"] == 2
assert not b.robot.enabled
b.request({"enable": True, "active_arm": "left", "joint_speed": 1})
assert b.robot.joint_speed == 0.3 and b.robot.active_arm == "left"
assert b.request({"action": [0.0] * 14, "dt": 0.03})["executed"]
b.frames["left"] = (None, time.monotonic() - 3)
try:
    b.request({"action": [0.0] * 14})
except RuntimeError:
    pass
else:
    raise AssertionError("stale camera accepted")
b.request({"stop": True})
assert b.robot.faulted and not b.robot.enabled
print("PASS protocol 2, explicit enable, server speed cap, stale camera rejection, stop; fake hardware only")
