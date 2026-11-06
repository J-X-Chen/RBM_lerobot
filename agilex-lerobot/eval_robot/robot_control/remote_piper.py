"""TCP client for agilex_image_service; no CAN or ROS on the inference machine."""

import json
import logging
import socket

import numpy as np
from robot_control.piper_bimanual import vector

logger = logging.getLogger(__name__)


def receive(sock, size):
    data = b""
    while len(data) < size:
        part = sock.recv(size - len(data))
        if not part:
            raise ConnectionError("Piper service disconnected")
        data += part
    return data


class RemotePiper:
    def __init__(
        self,
        host,
        port=6200,
        *,
        execute=False,
        active_arm="both",
        joint_speed=0.3,
        gripper_speed=0.02,
        command_timeout=5,
        camera_timeout=2,
    ):
        self.host, self.port = host, port
        self.execute, self.active_arm = execute, active_arm
        self.joint_speed, self.gripper_speed = joint_speed, gripper_speed
        self.command_timeout, self.camera_timeout = command_timeout, camera_timeout
        self.socket = None
        self.enabled = False
        self.info = {}
        self.last = None
        self.anchor = None
        self.accepted_target = None

    def request(self, request):
        try:
            payload = json.dumps(request, allow_nan=False).encode()
            self.socket.sendall(f"{len(payload):010d}".encode() + payload)
            size = int(receive(self.socket, 10))
            if not 2 <= size <= 65536:
                raise RuntimeError("Invalid service response length")
            reply = json.loads(receive(self.socket, size))
            if not reply.get("ok"):
                raise RuntimeError(f"Piper service: {reply.get('error', reply)}")
            return reply
        except BaseException:
            # An interrupted/failed framed exchange cannot safely carry another request.
            if self.socket:
                self.socket.close()
                self.socket = None
            raise

    def connect(self):
        try:
            self.socket = socket.create_connection((self.host, self.port), timeout=10)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.info = self.request({"get_state": True})
            if self.info.get("protocol") != 2 or not self.info.get("calibrated"):
                raise RuntimeError("Update service and start with --hardware-config (protocol 2 required)")
            if self.execute and not {"prepare_target", "startup_anchor", "finish_hold"}.issubset(self.info.get("capabilities", [])):
                raise RuntimeError("Update agilex_image_service: startup anchor/return support is required")
            if self.execute and not self.info.get("execute_control"):
                raise RuntimeError("Restart service with --execute-control before executing")
            if self.info.get("faulted"):
                raise RuntimeError("Service stopped previously; inspect hardware and restart service")
            self.command_timeout = min(self.command_timeout, self.info["command_timeout"])
        except BaseException:
            self.close()
            raise

    def read_state(self):
        self.info = self.request({"get_state": True})
        if self.info.get("faulted") or self.info.get("error"):
            raise RuntimeError(f"Robot-side fault: {self.info}")
        ages = self.info.get("camera_age", {})
        if set(ages) != {"head", "left", "right"} or any(
            not np.isfinite(age) or age < 0 or age > self.camera_timeout for age in ages.values()
        ):
            raise RuntimeError("Robot-side camera input missing/stale")
        return vector(self.info["state"])

    def enable(self):
        if self.execute:
            # Mark ownership before the request so an uncertain reply still causes stop/close.
            self.enabled = True
            reply = self.request(
                {
                    "enable": True,
                    "active_arm": self.active_arm,
                    "joint_speed": self.joint_speed,
                    "gripper_speed": self.gripper_speed,
                }
            )
            self.anchor = vector(reply["anchor"])
            self.last = self.anchor.copy()
            print(f"Robot speed caps: joints {reply['joint_speed']} rad/s, grippers {reply['gripper_speed']} m/s")

    def send(self, action, dt):
        target = vector(action)
        if not self.execute:
            # A read-only evaluator must not send action requests to an execution-capable server.
            state = self.read_state()
            if self.active_arm == "left":
                target[7:] = state[7:]
            elif self.active_arm == "right":
                target[:7] = state[:7]
            return target
        if not self.enabled:
            raise RuntimeError("Remote controller is not enabled")
        reply = self.request({"action": target.tolist(), "dt": dt})
        if not reply.get("executed"):
            raise RuntimeError("Service did not execute action")
        self.last = vector(reply["target"])
        self.accepted_target = vector(reply.get("accepted_target", reply["target"]))
        return self.last.copy()

    def prepare_target(self, action):
        return vector(self.request({"prepare_target": vector(action).tolist()})["target"])

    def finish_hold(self):
        reply = self.request({"finish_hold": True})
        if not reply.get("holding"):
            raise RuntimeError("Service did not confirm normal completion")
        self.enabled = False  # Close only the connection; server retains monitored position hold.

    def close(self):
        if self.socket:
            try:
                if self.enabled:
                    self.request({"stop": True})
            except Exception:
                logger.exception("Remote stop reply missing; server disconnect/watchdog remains active")
            finally:
                if self.socket:
                    self.socket.close()
                self.socket = None
                self.enabled = False
