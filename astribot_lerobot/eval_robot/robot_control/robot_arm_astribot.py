from __future__ import annotations

import os
import atexit
import json
import socket
import sys
import platform
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import logging_mp
except ModuleNotFoundError:
    import logging as logging_mp

from astribot_lerobot.eval_robot.utils.logging_mp_compat import patch_logging_mp

_DLL_DIRECTORY_HANDLES = []

_ROS_MULTIARCH_BY_MACHINE = {
    "amd64": "x86_64-linux-gnu",
    "x86_64": "x86_64-linux-gnu",
    "arm64": "aarch64-linux-gnu",
    "aarch64": "aarch64-linux-gnu",
}


def _find_astribot_sdk_root(sdk_root: str | Path | None = None) -> Path:
    if sdk_root is not None:
        root = Path(sdk_root).expanduser()
        if root.exists():
            return root.resolve()
        raise FileNotFoundError(f"Astribot SDK root does not exist: {root}")

    env_root = os.environ.get("ASTRIBOT_SDK_ROOT")
    if env_root:
        root = Path(env_root).expanduser()
        if root.exists():
            return root.resolve()

    repo_root = Path(__file__).resolve().parents[3]
    sdk_names = ("astribot_sdk_aarch64", "astribot_sdk_ros2")
    if _ROS_MULTIARCH_BY_MACHINE.get(platform.machine().lower()) == "x86_64-linux-gnu":
        sdk_names = ("astribot_sdk_ros2", "astribot_sdk_aarch64")
    seen: set[Path] = set()
    for base in (Path.cwd(), repo_root, Path.cwd().parent, repo_root.parent):
        for name in sdk_names:
            candidate = (base / name).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        "Could not find Astribot SDK. Set ASTRIBOT_SDK_ROOT or pass sdk_root. "
        "On S1/Orin this must point to the aarch64 SDK, not a workstation x86 SDK."
    )


def _prepend_env_path(name: str, paths: list[Path]) -> None:
    existing = [p for p in os.environ.get(name, "").split(os.pathsep) if p]
    new_paths = [str(path) for path in paths if path.exists()]
    if new_paths:
        os.environ[name] = os.pathsep.join([*new_paths, *existing])


def _sdk_library_paths(root: Path) -> list[Path]:
    ros_multiarch = _ROS_MULTIARCH_BY_MACHINE.get(platform.machine().lower())
    return [
        Path("/opt/ros/humble/lib"),
        Path("/opt/ros/humble/lib") / ros_multiarch if ros_multiarch else Path("/opt/ros/humble/lib"),
        root / "astribot_sdk" / "core" / "common" / "robotics_library_py",
        root / "astribot_sdk" / "core" / "common" / "whole_body_control" / "third_party",
        root / "third_party" / "drake" / "lib",
        root / "astribot_msgs" / "lib",
        root / "third_party" / "software" / "astribot_ros_middleware" / "lib",
        root / "third_party" / "software" / "ruckig" / "lib",
        root / "third_party" / "third_pkg",
        root / "third_party" / "third_pkg" / "pinocchio" / "lib",
        root / "third_party" / "third_pkg" / "hpp-fcl" / "lib",
    ]


def configure_astribot_sdk_path(sdk_root: str | Path | None = None) -> Path:
    root = _find_astribot_sdk_root(sdk_root)
    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    python_paths = [
        root,
        Path("/opt/ros/humble/local/lib") / python_dir / "dist-packages",
        Path("/opt/ros/humble/lib") / python_dir / "site-packages",
        root / "astribot_msgs" / "local" / "lib" / python_dir / "dist-packages",
        root / "third_party" / "software" / "astribot_ros_middleware" / "lib" / python_dir / "site-packages",
        root / "third_party" / "astribot_ros_middleware_py",
        root / "third_party" / "drake" / "lib" / python_dir / "site-packages",
        root / "third_party" / "third_pkg" / "pinocchio" / "lib" / python_dir / "site-packages",
    ]
    for path in reversed([path for path in python_paths if path.exists()]):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    library_paths = _sdk_library_paths(root)
    _prepend_env_path("LD_LIBRARY_PATH", library_paths)
    _prepend_env_path("PATH", library_paths)
    if hasattr(os, "add_dll_directory"):
        for path in library_paths:
            if path.exists():
                try:
                    _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))
                except (FileNotFoundError, OSError):
                    pass

    os.environ.setdefault("ROBOT_TYPE", "S1")
    os.environ.setdefault("ASTRIBOT_SDK_ROOT", str(root))
    os.environ.setdefault("ROS_DOMAIN_ID", "25")
    os.environ.setdefault("ROS_LOCALHOST_ONLY", "0")
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    return root

patch_logging_mp(logging_mp)
logger_mp = logging_mp.get_logger(__name__)

WHOLE_BODY_NAMES = [
    "astribot_chassis",
    "astribot_torso",
    "astribot_arm_left",
    "astribot_gripper_left",
    "astribot_arm_right",
    "astribot_gripper_right",
    "astribot_head",
]
WHOLE_BODY_CONTROL_NAMES = [
    "astribot_torso",
    "astribot_arm_left",
    "astribot_gripper_left",
    "astribot_arm_right",
    "astribot_gripper_right",
    "astribot_head",
]
ARM_NAMES = ["astribot_arm_left", "astribot_arm_right"]
GRIPPER_NAMES = ["astribot_gripper_left", "astribot_gripper_right"]

CHASSIS_SLICE = slice(0, 3)
TORSO_SLICE = slice(3, 7)
LEFT_ARM_SLICE = slice(7, 14)
LEFT_GRIPPER_SLICE = slice(14, 15)
RIGHT_ARM_SLICE = slice(15, 22)
RIGHT_GRIPPER_SLICE = slice(22, 23)
HEAD_SLICE = slice(23, 25)
WHOLE_BODY_DOF = 25
DUAL_ARM_DOF = 14

_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def _send_length_prefixed_json(sock: socket.socket, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock.sendall(f"{len(data):010d}".encode("utf-8") + data)


def _recv_length_prefixed_json(sock: socket.socket, timeout_s: float | None) -> dict[str, Any]:
    sock.settimeout(timeout_s)
    length_bytes = b""
    while len(length_bytes) < 10:
        chunk = sock.recv(10 - len(length_bytes))
        if not chunk:
            raise ConnectionError("Astribot remote control server closed the connection.")
        length_bytes += chunk
    length = int(length_bytes.decode("utf-8"))
    data = b""
    while len(data) < length:
        chunk = sock.recv(min(1 << 20, length - len(data)))
        if not chunk:
            raise ConnectionError("Astribot remote control server closed the connection.")
        data += chunk
    message = json.loads(data.decode("utf-8"))
    if not isinstance(message, dict):
        raise RuntimeError(f"Astribot remote control server returned non-object JSON: {message!r}")
    return message


def _is_mutating_remote_payload(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in ("hold_current", "move", "move_to_pose", "action", "chunk"))


def get_astribot_client(
    freq: float = 250.0,
    sdk_root: str | Path | None = None,
    high_control_rights: bool = False,
    node_name: str = "eval_astribot_s1",
):
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            configure_astribot_sdk_path(sdk_root)
            from astribot_sdk.core.astribot_api.astribot_client import Astribot

            _CLIENT = Astribot(freq=freq, high_control_rights=high_control_rights, node_name=node_name)
    return _CLIENT


def _finite_vector(name: str, values: Any, expected_len: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape[0] != expected_len:
        raise ValueError(f"{name} length {array.shape[0]} != {expected_len}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values: {array}")
    return array


def _read_shared_value(value) -> float:
    if hasattr(value, "get_lock"):
        with value.get_lock():
            return float(value.value)
    return float(value.value)


def _write_shared_value(value, data: float) -> None:
    if hasattr(value, "get_lock"):
        with value.get_lock():
            value.value = float(data)
    else:
        value.value = float(data)


class AstribotS1ArmIK:
    def solve_tau(self, arm_q):
        return np.zeros_like(np.asarray(arm_q, dtype=np.float64))

    def solve_ik(self, *args, **kwargs):
        raise NotImplementedError(
            "Astribot_S1 eval shim does not include a whole-body IK solver. "
            "Use --ready_arm_pose or keep --ready_lift_m=0."
        )


class AstribotS1RemoteArmController:
    """S1 whole-body controller backed by astribot_image_service.server on the robot."""

    def __init__(
        self,
        *,
        host: str,
        port: int = 6200,
        timeout_s: float = 5.0,
        arm_velocity_limit=None,
        reconnect: bool = True,
        remote_interpolation_steps: int = 1,
    ):
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.reconnect = bool(reconnect)
        self.arm_velocity_limit = float(arm_velocity_limit) if arm_velocity_limit is not None else 8.0
        self.remote_interpolation_steps = max(1, int(remote_interpolation_steps))
        self.control_dt = 1.0 / 250.0
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self.q_target = np.zeros(WHOLE_BODY_DOF, dtype=np.float64)
        logger_mp.info(
            "Initialize AstribotS1RemoteArmController: %s:%d timeout=%.2fs interpolation_steps=%d.",
            self.host,
            self.port,
            self.timeout_s,
            self.remote_interpolation_steps,
        )
        self._connect()
        self.q_target = self.get_current_dual_arm_q().copy()
        atexit.register(self.close)
        logger_mp.info("Initialize AstribotS1RemoteArmController OK. Current q:\n%s\n", self.q_target)

    def _connect(self) -> None:
        self.close()
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self.timeout_s)
        self._sock = sock

    def _request(self, payload: dict[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
        with self._lock:
            last_exc: Exception | None = None
            for attempt in range(2 if self.reconnect else 1):
                try:
                    if self._sock is None:
                        self._connect()
                    assert self._sock is not None
                    _send_length_prefixed_json(self._sock, payload)
                    response = _recv_length_prefixed_json(
                        self._sock,
                        self.timeout_s if timeout_s is None else timeout_s,
                    )
                    if not response.get("ok", False):
                        raise RuntimeError(
                            "Astribot remote control request failed: "
                            f"{response.get('error', response)}"
                        )
                    control_status = response.get("control")
                    if (
                        _is_mutating_remote_payload(payload)
                        and isinstance(control_status, dict)
                        and control_status.get("execute") is False
                    ):
                        raise RuntimeError(
                            "Astribot remote control server is running in DRY-RUN mode "
                            "(control.execute=False). Restart the S1-side "
                            "astribot_image_service.server with --execute-control before "
                            "pressing r/s for real robot motion."
                        )
                    return response
                except (OSError, ConnectionError, TimeoutError, RuntimeError) as exc:
                    last_exc = exc
                    self.close()
                    if attempt == 0 and self.reconnect:
                        logger_mp.warning(
                            "Astribot remote control request failed once (%s); reconnecting to %s:%d.",
                            exc,
                            self.host,
                            self.port,
                        )
                        continue
                    break
            raise RuntimeError(
                f"Astribot remote control request to {self.host}:{self.port} failed: {last_exc}"
            ) from last_exc

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None

    def get_current_dual_arm_q(self):
        response = self._request({"get_state": True})
        state = np.asarray(response.get("state"), dtype=np.float64).reshape(-1)
        if state.shape[0] != WHOLE_BODY_DOF:
            raise RuntimeError(f"Astribot remote control state length {state.shape[0]} != {WHOLE_BODY_DOF}.")
        if not np.all(np.isfinite(state)):
            raise RuntimeError(f"Astribot remote control state contains non-finite values: {state}")
        return state

    def get_current_dual_arm_dq(self):
        return np.zeros(WHOLE_BODY_DOF, dtype=np.float64)

    def ctrl_dual_arm(self, q_target, tauff_target=None):
        del tauff_target
        q_target = _finite_vector("Astribot remote S1 whole-body q_target", q_target, WHOLE_BODY_DOF).copy()
        q_target[LEFT_GRIPPER_SLICE] = np.clip(q_target[LEFT_GRIPPER_SLICE], 0.0, 100.0)
        q_target[RIGHT_GRIPPER_SLICE] = np.clip(q_target[RIGHT_GRIPPER_SLICE], 0.0, 100.0)
        if self.remote_interpolation_steps > 1:
            start = _finite_vector(
                "Astribot remote S1 previous q_target",
                self.q_target,
                WHOLE_BODY_DOF,
            )
            weights = np.linspace(
                1.0 / float(self.remote_interpolation_steps),
                1.0,
                self.remote_interpolation_steps,
                dtype=np.float64,
            )
            action_chunk = start[None, :] + weights[:, None] * (q_target[None, :] - start[None, :])
            self.send_action_chunk(action_chunk, replace_buffer=True)
        else:
            self._request({"action": q_target.astype(np.float32).tolist(), "clear_buffer": True})
            self.q_target = q_target.copy()

    def send_action_chunk(self, actions, *, replace_buffer: bool = True):
        action_chunk = np.asarray(actions, dtype=np.float64)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk.reshape(1, -1)
        if action_chunk.ndim != 2 or action_chunk.shape[1] != WHOLE_BODY_DOF:
            raise RuntimeError(
                "Astribot remote S1 action chunk must have shape "
                f"(N, {WHOLE_BODY_DOF}), got {action_chunk.shape}."
            )
        if not np.all(np.isfinite(action_chunk)):
            bad_indices = np.argwhere(~np.isfinite(action_chunk)).tolist()
            raise RuntimeError(f"Astribot remote S1 action chunk contains non-finite values at {bad_indices}.")
        action_chunk = action_chunk.copy()
        action_chunk[:, LEFT_GRIPPER_SLICE] = np.clip(action_chunk[:, LEFT_GRIPPER_SLICE], 0.0, 100.0)
        action_chunk[:, RIGHT_GRIPPER_SLICE] = np.clip(action_chunk[:, RIGHT_GRIPPER_SLICE], 0.0, 100.0)
        response = self._request(
            {
                "chunk": action_chunk.astype(np.float32).tolist(),
                "replace_buffer": bool(replace_buffer),
            }
        )
        self.q_target = action_chunk[-1].copy()
        return response

    def clear_action_buffer(self):
        return self._request({"clear_buffer": True})

    def move_dual_arm_to_pose(self, q_target, duration: float, label: str = "target pose"):
        q_target = _finite_vector("Astribot remote S1 whole-body move target", q_target, WHOLE_BODY_DOF).copy()
        q_target[LEFT_GRIPPER_SLICE] = np.clip(q_target[LEFT_GRIPPER_SLICE], 0.0, 100.0)
        q_target[RIGHT_GRIPPER_SLICE] = np.clip(q_target[RIGHT_GRIPPER_SLICE], 0.0, 100.0)
        duration = max(float(duration), 0.2)
        logger_mp.info(
            "[AstribotS1RemoteArmController] move request to %s: duration=%.3fs.",
            label,
            duration,
        )
        response = self._request(
            {"move": q_target.astype(np.float32).tolist(), "duration": duration},
            timeout_s=max(self.timeout_s, duration + 10.0),
        )
        self.q_target = q_target.copy()
        return response

    def ctrl_dual_arm_go_home(self):
        self._request({"hold_current": True})
        self.q_target = self.get_current_dual_arm_q().copy()


class AstribotS1ArmController:
    def __init__(
        self,
        motion_mode=False,
        simulation_mode=False,
        arm_velocity_limit=None,
        sdk_root: str | Path | None = None,
        high_control_rights: bool = False,
        control_way: str = "direct",
        use_wbc: bool = False,
        add_default_torso: bool = True,
        fps: float = 250.0,
        node_name: str = "eval_astribot_s1",
    ):
        if simulation_mode:
            raise NotImplementedError("Astribot_S1 eval shim uses astribot_sdk_ros2 and does not support --sim.")

        logger_mp.info("Initialize AstribotS1ArmController...")
        self.motion_mode = motion_mode
        self.simulation_mode = simulation_mode
        self.control_dt = 1.0 / float(fps)
        self.arm_velocity_limit = float(arm_velocity_limit) if arm_velocity_limit is not None else 8.0
        self.control_way = control_way
        self.use_wbc = use_wbc
        self.add_default_torso = add_default_torso
        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None
        self._last_command_time = time.monotonic()
        self._lock = threading.Lock()
        self.astribot = get_astribot_client(
            freq=fps,
            sdk_root=sdk_root,
            high_control_rights=high_control_rights,
            node_name=node_name,
        )
        self.q_target = self.get_current_dual_arm_q().copy()
        logger_mp.info(f"Current Astribot S1 whole-body state q:\n{self.q_target}\n")
        logger_mp.info("Initialize AstribotS1ArmController OK!\n")

    def clip_arm_q_target(self, target_q, velocity_limit, current_q=None, control_dt=None):
        current_q = self.get_current_dual_arm_q() if current_q is None else _finite_vector(
            "Astribot S1 current q", current_q, WHOLE_BODY_DOF
        )
        effective_dt = max(float(self.control_dt if control_dt is None else control_dt), self.control_dt)
        target_q = np.asarray(target_q, dtype=np.float64).copy()
        # eval_s1 uses a 25D S1 state, but this SDK control path only commands
        # torso, arms, grippers, and head. Keep chassis feedback out of velocity
        # limiting so a dataset chassis value cannot throttle or stall upper-body
        # moves that the controller can actually execute.
        target_q[CHASSIS_SLICE] = current_q[CHASSIS_SLICE]
        delta = target_q - current_q
        velocity_delta = delta.copy()
        velocity_delta[CHASSIS_SLICE] = 0.0
        velocity_delta[LEFT_GRIPPER_SLICE] = 0.0
        velocity_delta[RIGHT_GRIPPER_SLICE] = 0.0
        motion_scale = np.max(np.abs(velocity_delta)) / (float(velocity_limit) * effective_dt)
        clipped = current_q + delta / max(motion_scale, 1.0)
        clipped[LEFT_GRIPPER_SLICE] = target_q[LEFT_GRIPPER_SLICE]
        clipped[RIGHT_GRIPPER_SLICE] = target_q[RIGHT_GRIPPER_SLICE]
        return clipped

    def ctrl_dual_arm(self, q_target, tauff_target=None):
        q_target = _finite_vector("Astribot S1 whole-body q_target", q_target, WHOLE_BODY_DOF).copy()
        now = time.monotonic()
        control_dt = now - self._last_command_time
        self._last_command_time = now
        current_q = self.get_current_dual_arm_q()
        q_target[CHASSIS_SLICE] = current_q[CHASSIS_SLICE]
        if self.arm_velocity_limit > 0:
            q_target = self.clip_arm_q_target(
                q_target,
                velocity_limit=self.arm_velocity_limit,
                current_q=current_q,
                control_dt=control_dt,
            )

        command_list = [
            q_target[TORSO_SLICE].tolist(),
            q_target[LEFT_ARM_SLICE].tolist(),
            np.clip(q_target[LEFT_GRIPPER_SLICE], 0.0, 100.0).tolist(),
            q_target[RIGHT_ARM_SLICE].tolist(),
            np.clip(q_target[RIGHT_GRIPPER_SLICE], 0.0, 100.0).tolist(),
            q_target[HEAD_SLICE].tolist(),
        ]
        with self._lock:
            self.q_target = q_target.copy()
            self.astribot.set_joints_position(
                WHOLE_BODY_CONTROL_NAMES,
                command_list,
                control_way=self.control_way,
                use_wbc=self.use_wbc,
                add_default_torso=self.add_default_torso,
            )

        if self._speed_gradual_max and self._gradual_start_time is not None:
            t_elapsed = time.time() - self._gradual_start_time
            self.arm_velocity_limit = 8.0 + (22.0 * min(1.0, t_elapsed / max(float(self._gradual_time), 1e-6)))

    def move_dual_arm_to_pose(self, q_target, duration: float, label: str = "target pose"):
        q_target = _finite_vector("Astribot S1 whole-body move target", q_target, WHOLE_BODY_DOF).copy()
        current_q = self.get_current_dual_arm_q()
        q_target[CHASSIS_SLICE] = current_q[CHASSIS_SLICE]
        command_list = [
            q_target[TORSO_SLICE].tolist(),
            q_target[LEFT_ARM_SLICE].tolist(),
            np.clip(q_target[LEFT_GRIPPER_SLICE], 0.0, 100.0).tolist(),
            q_target[RIGHT_ARM_SLICE].tolist(),
            np.clip(q_target[RIGHT_GRIPPER_SLICE], 0.0, 100.0).tolist(),
            q_target[HEAD_SLICE].tolist(),
        ]
        duration = max(float(duration), 0.2)
        logger_mp.info(
            "[AstribotS1ArmController] move_joints_position to %s: duration=%.3fs, names=%s.",
            label,
            duration,
            WHOLE_BODY_CONTROL_NAMES,
        )
        with self._lock:
            self.q_target = q_target.copy()
            return self.astribot.move_joints_position(
                WHOLE_BODY_CONTROL_NAMES,
                command_list,
                duration=duration,
                use_wbc=self.use_wbc,
                add_default_torso=False,
            )

    def get_current_dual_arm_q(self):
        positions = self.astribot.get_current_joints_position(WHOLE_BODY_NAMES)
        return np.concatenate([np.asarray(part, dtype=np.float64).reshape(-1) for part in positions])

    def get_current_dual_arm_dq(self):
        velocities = self.astribot.get_current_joints_velocity(WHOLE_BODY_NAMES)
        return np.concatenate([np.asarray(part, dtype=np.float64).reshape(-1) for part in velocities])

    def ctrl_dual_arm_go_home(self):
        logger_mp.info("[AstribotS1ArmController] move_to_home start...")
        self.astribot.move_to_home(duration=5.0, use_wbc=self.use_wbc)
        self.q_target = self.get_current_dual_arm_q().copy()
        logger_mp.info("[AstribotS1ArmController] move_to_home done.")

    def speed_gradual_max(self, t=5.0):
        self._gradual_start_time = time.time()
        self._gradual_time = float(t)
        self._speed_gradual_max = True

    def speed_instant_max(self):
        self.arm_velocity_limit = 30.0


class AstribotS1GripperController:
    def __init__(
        self,
        left_gripper_value_in,
        right_gripper_value_in,
        dual_gripper_data_lock=None,
        dual_gripper_state_out=None,
        dual_gripper_action_out=None,
        fps=100.0,
        Unit_Test=False,
        simulation_mode=False,
        sdk_root: str | Path | None = None,
        high_control_rights: bool = False,
        node_name: str = "eval_astribot_s1_gripper",
        control_way: str = "direct",
    ):
        if simulation_mode:
            raise NotImplementedError("Astribot_S1 gripper shim uses astribot_sdk_ros2 and does not support --sim.")

        logger_mp.info("Initialize AstribotS1GripperController...")
        self.fps = float(fps)
        self.Unit_Test = Unit_Test
        self.control_way = control_way
        self.running = True
        self.astribot = get_astribot_client(
            freq=max(float(fps), 1.0),
            sdk_root=sdk_root,
            high_control_rights=high_control_rights,
            node_name=node_name,
        )
        self.hold_current_pose(left_gripper_value_in, right_gripper_value_in)
        self.gripper_control_thread = threading.Thread(
            target=self.control_thread,
            args=(
                left_gripper_value_in,
                right_gripper_value_in,
                dual_gripper_data_lock,
                dual_gripper_state_out,
                dual_gripper_action_out,
            ),
            daemon=True,
        )
        self.gripper_control_thread.start()
        logger_mp.info("Initialize AstribotS1GripperController OK!\n")

    def ctrl_dual_gripper(self, dual_gripper_action):
        action = _finite_vector("Astribot S1 gripper action", dual_gripper_action, 2)
        action = np.clip(action, 0.0, 100.0)
        self.astribot.set_joints_position(
            GRIPPER_NAMES,
            [[float(action[0])], [float(action[1])]],
            control_way=self.control_way,
            use_wbc=False,
            add_default_torso=False,
        )

    def control_thread(
        self,
        left_gripper_value_in,
        right_gripper_value_in,
        dual_gripper_data_lock=None,
        dual_gripper_state_out=None,
        dual_gripper_action_out=None,
    ):
        try:
            while self.running:
                start_time = time.time()
                left_cmd = _read_shared_value(left_gripper_value_in)
                right_cmd = _read_shared_value(right_gripper_value_in)
                dual_gripper_action = np.array([left_cmd, right_cmd], dtype=np.float64)

                state = self.astribot.get_current_joints_position(GRIPPER_NAMES)
                dual_gripper_state = np.array(
                    [
                        float(np.asarray(state[0], dtype=np.float64).reshape(-1)[0]),
                        float(np.asarray(state[1], dtype=np.float64).reshape(-1)[0]),
                    ],
                    dtype=np.float64,
                )

                if dual_gripper_state_out is not None and dual_gripper_action_out is not None:
                    if dual_gripper_data_lock is not None:
                        with dual_gripper_data_lock:
                            dual_gripper_state_out[:] = dual_gripper_state
                            dual_gripper_action_out[:] = dual_gripper_action
                    else:
                        dual_gripper_state_out[:] = dual_gripper_state
                        dual_gripper_action_out[:] = dual_gripper_action

                self.ctrl_dual_gripper(dual_gripper_action)
                sleep_time = max(0.0, (1.0 / self.fps) - (time.time() - start_time))
                time.sleep(sleep_time)
        finally:
            logger_mp.info("AstribotS1GripperController has been closed.")

    def hold_current_pose(self, left_gripper_value_in, right_gripper_value_in):
        state = self.astribot.get_current_joints_position(GRIPPER_NAMES)
        _write_shared_value(left_gripper_value_in, float(np.asarray(state[0]).reshape(-1)[0]))
        _write_shared_value(right_gripper_value_in, float(np.asarray(state[1]).reshape(-1)[0]))
