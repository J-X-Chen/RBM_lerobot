#!/usr/bin/env python3
"""Publish Astribot S1 SDK camera images and optionally host the S1 control sender."""

from __future__ import annotations

import argparse
import collections
import contextlib
import faulthandler
import importlib.util
import json
import os
import platform
import queue
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REEXEC_ENV = "ASTRIBOT_IMAGE_SERVICE_REEXECED"
PROCESS_STOPPING = threading.Event()

ELF_MACHINE_NAMES = {
    62: "x86_64",
    183: "aarch64",
}
HOST_ELF_MACHINES = {
    "amd64": 62,
    "x86_64": 62,
    "arm64": 183,
    "aarch64": 183,
}
ROS_MULTIARCH_BY_MACHINE = {
    "amd64": "x86_64-linux-gnu",
    "x86_64": "x86_64-linux-gnu",
    "arm64": "aarch64-linux-gnu",
    "aarch64": "aarch64-linux-gnu",
}


def _find_astribot_sdk_root(sdk_root: str | Path | None = None) -> Path:
    if sdk_root:
        root = Path(sdk_root).expanduser()
        if root.exists():
            return root.resolve()
        raise FileNotFoundError(f"Astribot SDK root does not exist: {root}")

    env_root = os.environ.get("ASTRIBOT_SDK_ROOT")
    if env_root:
        root = Path(env_root).expanduser()
        if root.exists():
            return root.resolve()

    candidates: list[Path] = []
    seen: set[Path] = set()
    for base in (Path.cwd(), REPO_ROOT, Path.cwd().parent, REPO_ROOT.parent):
        for name in ("astribot_sdk_aarch64", "astribot_sdk_ros2"):
            candidate = (base / name).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                candidates.append(candidate)
    if candidates:
        expected_machine = HOST_ELF_MACHINES.get(platform.machine().lower())
        if expected_machine is not None:
            for candidate in candidates:
                probe = candidate / "astribot_sdk" / "core" / "common" / "util" / "astribot_function.so"
                if _read_elf_machine(probe) == expected_machine:
                    return candidate
        return candidates[0]

    raise FileNotFoundError(
        "Could not find Astribot SDK. Set ASTRIBOT_SDK_ROOT or use --sdk-root. "
        "On S1/Orin this must point to the aarch64 SDK, not a workstation x86 SDK."
    )


def _read_elf_machine(path: Path) -> int | None:
    try:
        header = path.read_bytes()[:20]
    except OSError:
        return None
    if len(header) < 20 or header[:4] != b"\x7fELF":
        return None
    if header[5] == 1:
        endian = "little"
    elif header[5] == 2:
        endian = "big"
    else:
        return None
    return int.from_bytes(header[18:20], endian)


def _elf_machine_name(machine: int | None) -> str:
    if machine is None:
        return "unknown"
    return ELF_MACHINE_NAMES.get(machine, f"ELF machine {machine}")


def validate_astribot_sdk_native_arch(root: Path) -> None:
    expected_machine = HOST_ELF_MACHINES.get(platform.machine().lower())
    if expected_machine is None:
        return

    probe = root / "astribot_sdk" / "core" / "common" / "util" / "astribot_function.so"
    actual_machine = _read_elf_machine(probe)
    if actual_machine is None or actual_machine == expected_machine:
        return

    raise RuntimeError(
        "Astribot SDK native library architecture mismatch. "
        f"Host is {platform.machine()} ({_elf_machine_name(expected_machine)}), "
        f"but {probe} is {_elf_machine_name(actual_machine)}. "
        "Use the S1/Orin aarch64 Astribot SDK, for example set "
        "ASTRIBOT_SDK_ROOT=./astribot_sdk_aarch64 and pass "
        "--sdk-root \"$ASTRIBOT_SDK_ROOT\". Do not copy the workstation/x86 "
        "astribot_sdk_ros2 directory onto S1."
    )


def _sdk_python_paths(root: Path) -> list[Path]:
    python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return [
        Path("/opt/ros/humble/local/lib") / python_dir / "dist-packages",
        Path("/opt/ros/humble/lib") / python_dir / "site-packages",
        root,
        root / "astribot_sdk" / "core" / "common",
        root / "astribot_msgs" / "local" / "lib" / python_dir / "dist-packages",
        root / "third_party" / "software" / "astribot_ros_middleware" / "lib" / python_dir / "site-packages",
        root / "third_party" / "astribot_ros_middleware_py",
        root / "third_party" / "drake" / "lib" / python_dir / "site-packages",
        root / "third_party" / "software" / "ruckig" / "lib" / python_dir / "site-packages",
        root / "third_party" / "third_pkg" / "pinocchio" / "lib" / python_dir / "site-packages",
        root / "third_party" / "third_pkg" / "hpp-fcl" / "lib" / python_dir / "site-packages",
    ]


def _sdk_library_paths(root: Path) -> list[Path]:
    ros_multiarch = ROS_MULTIARCH_BY_MACHINE.get(platform.machine().lower())
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


def _prepend_env_path(name: str, paths: list[Path]) -> None:
    existing = [p for p in os.environ.get(name, "").split(os.pathsep) if p]
    new_paths = [str(path) for path in paths if path.exists()]
    if new_paths:
        os.environ[name] = os.pathsep.join(_dedupe_path_entries([*new_paths, *existing]))


def _dedupe_path_entries(paths: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _parse_major_version(version: str) -> int | None:
    try:
        return int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def validate_numpy_for_full_sdk() -> None:
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("Astribot full SDK requires NumPy, but importing numpy failed.") from exc

    major = _parse_major_version(getattr(np, "__version__", ""))
    if major is not None and major >= 2:
        raise RuntimeError(
            "Astribot SDK native extensions were compiled against NumPy 1.x, "
            f"but the current environment has numpy {np.__version__}. "
            "Keep Python the same, but use NumPy < 2 in the S1 runtime environment, for example: "
            "python -m pip install 'numpy<2'."
        )


def validate_python_dependencies_for_full_sdk() -> None:
    missing = []
    for module_name in ("filterpy", "transforms3d", "tf_transformations"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    if missing:
        pip_packages = [module_name for module_name in missing if module_name != "tf_transformations"]
        apt_packages = ["ros-humble-tf-transformations"] if "tf_transformations" in missing else []
        install_parts = []
        if apt_packages:
            install_parts.append(f"sudo apt install -y {' '.join(apt_packages)}")
        if pip_packages:
            install_parts.append(f"python -m pip install {' '.join(pip_packages)}")
        raise RuntimeError(
            "Astribot full SDK is missing required Python package(s): "
            f"{', '.join(missing)}. Install them for the current S1 environment with: "
            + " && ".join(install_parts)
            + ". Do not install the PyPI package 'meta'; "
            "the SDK's later 'meta.whole_body_control' message is a fallback symptom of "
            "missing filterpy."
        )


def validate_ros_extension_imports_for_full_sdk() -> None:
    for module_name in ("tf2_py", "tf_transformations"):
        try:
            __import__(module_name)
        except Exception as exc:
            raise RuntimeError(
                f"Astribot full SDK cannot import ROS Python extension {module_name!r}: {exc}. "
                "On S1, source /opt/ros/humble/setup.bash before the Astribot SDK env.sh, "
                "make sure /opt/ros/humble/lib is before SDK third_party paths in LD_LIBRARY_PATH, "
                "and remove the unrelated PyPI package with: python -m pip uninstall -y meta."
            ) from exc


def configure_astribot_sdk_path(sdk_root: str | Path | None = None) -> Path:
    root = _find_astribot_sdk_root(sdk_root)
    validate_astribot_sdk_native_arch(root)
    python_paths = _sdk_python_paths(root)
    for path in reversed([path for path in python_paths if path.exists()]):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    library_paths = _sdk_library_paths(root)
    _prepend_env_path("LD_LIBRARY_PATH", library_paths)
    _prepend_env_path("PATH", library_paths)

    os.environ.setdefault("ASTRIBOT_LOG", "1")
    os.environ.setdefault("ROBOT_TYPE", "S1")
    os.environ.setdefault("ASTRIBOT_SDK_ROOT", str(root))
    os.environ.setdefault("ROS_DOMAIN_ID", "25")
    os.environ.setdefault("ROS_LOCALHOST_ONLY", "0")
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    return root


def ensure_astribot_sdk_runtime_env(sdk_root: str | Path | None = None, *, check_native_log_dir: bool = False) -> Path:
    root = _find_astribot_sdk_root(sdk_root)
    validate_astribot_sdk_native_arch(root)
    desired = [str(path) for path in _sdk_library_paths(root) if path.exists()]
    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    reordered = _dedupe_path_entries([*desired, *existing])
    needs_reexec = existing[: len(desired)] != desired
    if needs_reexec and os.environ.get(REEXEC_ENV) != "1":
        env = os.environ.copy()
        env[REEXEC_ENV] = "1"
        env.setdefault("ASTRIBOT_LOG", "1")
        env.setdefault("PYTHONFAULTHANDLER", "1")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(reordered)
        print("[astribot-image-service] Re-exec with Astribot SDK library paths.", flush=True)
        os.execvpe(sys.executable, [sys.executable, "-m", "astribot_image_service.server", *sys.argv[1:]], env)

    if check_native_log_dir:
        log_dir = Path("/opt/astribot_ros/log/python")
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            probe = log_dir / ".astribot_image_service_write_test"
            probe.write_text("", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(
                "Astribot SDK native logging directory is not writable: "
                f"{log_dir}. The full Astribot SDK may abort during import before Python can print a traceback. "
                "Please make this directory writable or update the SDK log config to use a writable log_dir."
            ) from exc
    return root


@dataclass(frozen=True)
class CameraRoute:
    label: str
    camera_name: str
    port: int


STATE_COMPONENTS = [
    ("astribot_chassis", 3),
    ("astribot_torso", 4),
    ("astribot_arm_left", 7),
    ("astribot_gripper_left", 1),
    ("astribot_arm_right", 7),
    ("astribot_gripper_right", 1),
    ("astribot_head", 2),
]
STATE_NAMES = [name for name, _ in STATE_COMPONENTS]
STATE_DIM = 25
ARM_GRIPPER_DIM = 16

TORSO_SLICE = slice(3, 7)
LEFT_ARM_SLICE = slice(7, 14)
LEFT_GRIPPER_SLICE = slice(14, 15)
RIGHT_ARM_SLICE = slice(15, 22)
RIGHT_GRIPPER_SLICE = slice(22, 23)
HEAD_SLICE = slice(23, 25)
LEFT_ARM_GRIPPER_16_SLICE = slice(0, 8)
RIGHT_ARM_GRIPPER_16_SLICE = slice(8, 16)

CONTROL_NAMES = [
    "astribot_torso",
    "astribot_arm_left",
    "astribot_gripper_left",
    "astribot_arm_right",
    "astribot_gripper_right",
    "astribot_head",
]


def send_json(sock: socket.socket, obj: dict[str, Any]) -> None:
    data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    sock.sendall(f"{len(data):010d}".encode("utf-8") + data)


def recv_json(sock: socket.socket, timeout: float | None = 10.0) -> dict[str, Any]:
    sock.settimeout(timeout)
    length_bytes = b""
    while len(length_bytes) < 10:
        chunk = sock.recv(10 - len(length_bytes))
        if not chunk:
            raise ConnectionError("client disconnected")
        length_bytes += chunk
    length = int(length_bytes.decode("utf-8"))
    data = b""
    while len(data) < length:
        chunk = sock.recv(min(1 << 20, length - len(data)))
        if not chunk:
            raise ConnectionError("client disconnected")
        data += chunk
    decoded = json.loads(data.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("control message must be a JSON object")
    return decoded


def _as_float_list(values: Any) -> list[float]:
    return np.asarray(values, dtype=np.float32).reshape(-1).tolist()


def _coerce_s1_action(action: Any, current_state: np.ndarray | None = None) -> np.ndarray:
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(action_array)):
        raise ValueError("action contains NaN or Inf")

    if action_array.shape[0] == STATE_DIM:
        return action_array.copy()

    if action_array.shape[0] == ARM_GRIPPER_DIM:
        if current_state is None:
            current_state = np.zeros(STATE_DIM, dtype=np.float32)
        target = np.asarray(current_state, dtype=np.float32).reshape(STATE_DIM).copy()
        target[LEFT_ARM_SLICE] = action_array[LEFT_ARM_GRIPPER_16_SLICE][0:7]
        target[LEFT_GRIPPER_SLICE] = action_array[LEFT_ARM_GRIPPER_16_SLICE][7:8]
        target[RIGHT_ARM_SLICE] = action_array[RIGHT_ARM_GRIPPER_16_SLICE][0:7]
        target[RIGHT_GRIPPER_SLICE] = action_array[RIGHT_ARM_GRIPPER_16_SLICE][7:8]
        return target

    raise ValueError(f"S1 action length {action_array.shape[0]} is not supported; expected 25D or 16D.")


class ZmqPublisherThread(threading.Thread):
    def __init__(self, label: str, host: str, port: int):
        super().__init__(daemon=True, name=f"zmq-pub-{label}-{port}")
        self.label = label
        self.host = host
        self.port = int(port)
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=1)
        self._ready = threading.Event()
        self._running = True
        self._lock = threading.Lock()
        self.frame_count = 0
        self.byte_count = 0
        self.last_send_time = 0.0
        self.error: Exception | None = None

    def publish(self, payload: bytes) -> None:
        if PROCESS_STOPPING.is_set() or not self._running:
            return
        if not payload:
            return
        if self._queue.full():
            with contextlib.suppress(queue.Empty):
                self._queue.get_nowait()
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(payload)

    def stop(self) -> None:
        self._running = False
        if self._queue.full():
            with contextlib.suppress(queue.Empty):
                self._queue.get_nowait()
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)
        self.join(timeout=1.0)

    def wait_until_ready(self, timeout_s: float = 2.0) -> bool:
        return self._ready.wait(timeout=timeout_s)

    def snapshot(self) -> tuple[int, int, float]:
        with self._lock:
            return self.frame_count, self.byte_count, self.last_send_time

    def run(self) -> None:
        try:
            import zmq
        except ModuleNotFoundError as exc:
            self.error = RuntimeError("pyzmq is required for astribot_image_service. Install the project deps first.")
            self._ready.set()
            return

        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.CONFLATE, 1)
        try:
            socket.bind(f"tcp://{self.host}:{self.port}")
            self._ready.set()
            while self._running:
                try:
                    payload = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if payload is None:
                    break
                try:
                    socket.send(payload, zmq.NOBLOCK)
                except zmq.Again:
                    continue
                with self._lock:
                    self.frame_count += 1
                    self.byte_count += len(payload)
                    self.last_send_time = time.time()
        except Exception as exc:
            self.error = exc
            self._ready.set()
        finally:
            socket.close()
            context.term()


class AstribotImageBridge:
    def __init__(
        self,
        routes: list[CameraRoute],
        *,
        bind_host: str,
        jpeg_quality: int,
        decode_before_publish: bool,
    ):
        self.routes = routes
        self.jpeg_quality = int(jpeg_quality)
        self.decode_before_publish = bool(decode_before_publish)
        self._publishers = {
            route.camera_name: ZmqPublisherThread(route.label, bind_host, route.port) for route in routes
        }
        self._received_by_camera = {route.camera_name: 0 for route in routes}
        self._dropped_by_camera = {route.camera_name: 0 for route in routes}
        self._lock = threading.Lock()
        self._stopped = threading.Event()

    @property
    def need_decode(self) -> bool:
        return self.decode_before_publish

    def start(self) -> None:
        for publisher in self._publishers.values():
            publisher.start()
        for camera_name, publisher in self._publishers.items():
            if not publisher.wait_until_ready():
                raise RuntimeError(f"Timed out binding ZMQ publisher for {camera_name} on {publisher.port}.")
            if publisher.error is not None:
                raise RuntimeError(
                    f"Failed to bind ZMQ publisher for {camera_name} on {publisher.port}: {publisher.error}"
                ) from publisher.error

    def stop(self) -> None:
        self._stopped.set()
        for publisher in self._publishers.values():
            publisher.stop()

    def image_callback(self, topic_name: str, msg: Any, width: int, height: int, array: np.ndarray) -> None:
        if self._stopped.is_set() or PROCESS_STOPPING.is_set():
            return
        camera_name = self._camera_name_from_topic(topic_name)
        publisher = self._publishers.get(camera_name)
        if publisher is None:
            return

        payload = self._payload_from_callback(msg, array)
        if not payload:
            with self._lock:
                self._dropped_by_camera[camera_name] += 1
            return

        with self._lock:
            self._received_by_camera[camera_name] += 1
        publisher.publish(payload)

    def log_status(self) -> None:
        with self._lock:
            received = dict(self._received_by_camera)
            dropped = dict(self._dropped_by_camera)

        now = time.time()
        for route in self.routes:
            publisher = self._publishers[route.camera_name]
            sent, byte_count, last_send_time = publisher.snapshot()
            age = now - last_send_time if last_send_time > 0 else float("inf")
            age_text = "never" if age == float("inf") else f"{age:.1f}s ago"
            mb = byte_count / (1024 * 1024)
            print(
                f"[astribot-image-service] {route.label} {route.camera_name} -> {publisher.host}:{publisher.port}: "
                f"received={received[route.camera_name]} sent={sent} dropped={dropped[route.camera_name]} "
                f"bytes={mb:.1f}MiB last={age_text}",
                flush=True,
            )

    @staticmethod
    def _camera_name_from_topic(topic_name: str) -> str:
        parts = str(topic_name).split("/")
        return parts[2] if len(parts) > 2 else str(topic_name)

    def _payload_from_callback(self, msg: Any, array: np.ndarray) -> bytes | None:
        if not self.decode_before_publish:
            data = getattr(msg, "data", None)
            if data is not None:
                return bytes(data)
            if isinstance(array, np.ndarray):
                return array.tobytes()
            return None

        if not isinstance(array, np.ndarray) or array.ndim != 3:
            return None
        import cv2

        ok, encoded = cv2.imencode(".jpg", array, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return None
        return encoded.tobytes()


class AstribotControlBridge:
    """Run the SDK-facing side of S1 whole-body position control on the robot host."""

    def __init__(
        self,
        astribot: Any,
        *,
        execute: bool,
        command_frequency: float,
        control_way: str,
        use_wbc: bool,
        add_default_torso: bool,
        gripper_binary: bool,
        gripper_threshold: float,
        control_mode: str = "legacy",
        tracking_arm_velocity: float = 1.0,
        tracking_torso_velocity: float = 0.1,
        tracking_head_velocity: float = 0.3,
        gripper_frequency: float = 100.0,
        gripper_velocity: float = 40.0,
        safe_joint_envelope: float = 0.15,
        safe_measured_velocity: float = 0.1,
        safe_measured_acceleration: float = 1.0,
        safe_tracking_error: float = 0.05,
        safe_policy_velocity: float = 0.02,
        safe_policy_acceleration: float = 0.05,
        safe_command_tracking: bool = False,
        safe_direct_gripper: bool = False,
        safe_gripper_feedback_tracking: bool = False,
        safe_gripper_limit_retention: float = 1.0,
        safe_gripper_output_smoothing_s: float = 0.0,
        safe_gripper_smoothing_s: float = 0.0,
        safe_gripper_velocity: float = 1.0,
        safe_gripper_acceleration: float = 2.0,
        target_timeout_s: float = 5.0,
    ):
        self.astribot = astribot
        if control_mode not in {"legacy", "g1", "safe"}:
            raise ValueError("Unknown control mode")
        rates = (tracking_arm_velocity, tracking_torso_velocity, tracking_head_velocity,
                 gripper_frequency, gripper_velocity, command_frequency, target_timeout_s,
                 safe_policy_velocity, safe_policy_acceleration, safe_gripper_velocity, safe_gripper_acceleration, safe_joint_envelope, safe_measured_velocity, safe_measured_acceleration, safe_tracking_error)
        if not all(np.isfinite(v) and v > 0 for v in rates):
            raise ValueError("Control frequencies and velocities must be finite and positive")
        if control_mode == "g1" and (control_way != "direct" or use_wbc or add_default_torso or gripper_binary):
            raise ValueError("g1 mode requires direct control, continuous grippers, no WBC/default torso")
        if control_mode == "safe" and (control_way not in {"filter", "direct"} or use_wbc or add_default_torso or gripper_binary):
            raise ValueError("safe mode requires filter/direct, continuous grippers, no WBC/default torso")
        self.safe_g1_combined = control_mode == "safe" and control_way == "direct"
        self._safe_alignment = False
        self._safe_origin = None
        self._safe_command = None
        self._safe_feedback = None
        self._safe_time = None
        self._safe_velocity = np.zeros(25)
        self._safe_window = None
        self.control_mode = control_mode
        self.tracking_arm_velocity = tracking_arm_velocity
        self.tracking_torso_velocity = tracking_torso_velocity
        self.tracking_head_velocity = tracking_head_velocity
        self.gripper_frequency = gripper_frequency
        self.gripper_velocity = gripper_velocity
        self.safe_joint_envelope = safe_joint_envelope
        self.safe_measured_velocity = safe_measured_velocity
        self.safe_measured_acceleration = safe_measured_acceleration
        self.safe_tracking_error = safe_tracking_error
        self.safe_policy_velocity = safe_policy_velocity
        self.safe_policy_acceleration = safe_policy_acceleration
        if not np.isfinite(safe_gripper_smoothing_s) or safe_gripper_smoothing_s < 0:
            raise ValueError("safe_gripper_smoothing_s must be finite and nonnegative")
        self.safe_gripper_smoothing_s = float(safe_gripper_smoothing_s)
        self.safe_direct_gripper = bool(safe_direct_gripper)
        if safe_command_tracking and not self.safe_g1_combined:
            raise ValueError("safe_command_tracking requires safe direct combined control")
        self.safe_command_tracking = bool(safe_command_tracking)
        if safe_gripper_feedback_tracking and not (self.safe_g1_combined and self.safe_direct_gripper):
            raise ValueError("safe_gripper_feedback_tracking requires safe direct combined control")
        self.safe_gripper_feedback_tracking = bool(safe_gripper_feedback_tracking)
        if not np.isfinite(safe_gripper_limit_retention) or not 0 < safe_gripper_limit_retention <= 1:
            raise ValueError("safe_gripper_limit_retention must be in (0, 1]")
        if safe_gripper_limit_retention != 1 and not self.safe_gripper_feedback_tracking:
            raise ValueError("Reduced gripper limit retention requires safe feedback gripper tracking")
        self.safe_gripper_limit_retention = float(safe_gripper_limit_retention)
        if not np.isfinite(safe_gripper_output_smoothing_s) or safe_gripper_output_smoothing_s < 0:
            raise ValueError("safe_gripper_output_smoothing_s must be finite and nonnegative")
        if safe_gripper_output_smoothing_s > 0 and not self.safe_gripper_feedback_tracking:
            raise ValueError("Output smoothing requires safe feedback gripper tracking")
        self.safe_gripper_output_smoothing_s = float(safe_gripper_output_smoothing_s)
        self.safe_gripper_velocity = safe_gripper_velocity
        self.safe_gripper_acceleration = safe_gripper_acceleration
        self.target_timeout_s = target_timeout_s
        self._last_target_monotonic = 0.0
        self._stale_hold = False
        self._fault: str | None = None
        self._sdk_lock = threading.RLock()
        self._gripper_thread = None
        self.execute = bool(execute)
        self.command_frequency = max(1.0, float(command_frequency))
        self.control_way = str(control_way)
        self.use_wbc = bool(use_wbc)
        self.add_default_torso = bool(add_default_torso)
        self.gripper_binary = bool(gripper_binary)
        self.gripper_threshold = float(gripper_threshold)
        self._last_gripper = {"left": None, "right": None}
        self._latest_names: list[str] | None = None
        self._latest_cmd: list[list[float]] | None = None
        self._latest_action: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._command_thread: threading.Thread | None = None
        self._tracking_diagnostic = None
        self._action_count = 0
        self._command_count = 0
        self._body_command_count = 0
        self._gripper_command_count = 0
        self._error_count = 0
        self._last_action_time = 0.0
        self._last_command_time = 0.0

    def start(self) -> None:
        if not self.execute:
            print(
                "[astribot-control] DRY-RUN mode: action server will accept targets but will not command SDK. "
                "Add --execute-control to move the robot.",
                flush=True,
            )
            return
        self._command_thread = threading.Thread(
            target=self._command_loop,
            daemon=True,
            name="astribot-control-250hz",
        )
        self._command_thread.start()
        if self.control_mode == "g1":
            self._gripper_thread = threading.Thread(target=self._gripper_loop, daemon=True,
                                                    name="astribot-independent-grippers")
            self._gripper_thread.start()
        print(
            f"[astribot-control] {self.command_frequency:.1f}Hz SDK command loop started "
            f"(control_way={self.control_way}, use_wbc={self.use_wbc}, "
            f"add_default_torso={self.add_default_torso}).",
            flush=True,
        )

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            self._latest_names = None
            self._latest_cmd = None
        if self._command_thread is not None:
            self._command_thread.join(timeout=1.0)

        if self._gripper_thread is not None:
            self._gripper_thread.join(timeout=1.0)

    def get_state(self) -> np.ndarray:
        if PROCESS_STOPPING.is_set():
            raise RuntimeError("Astribot control bridge is stopping.")
        with self._sdk_lock:
            joints = self.astribot.get_current_joints_position(STATE_NAMES)
        if joints is None:
            raise RuntimeError("Astribot SDK returned no joint state.")
        values: list[float] = []
        for component in joints:
            values.extend(_as_float_list(component))
        if len(values) != STATE_DIM:
            raise RuntimeError(f"Astribot S1 state length {len(values)} != {STATE_DIM}.")
        state = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(state)):
            raise RuntimeError("Astribot S1 state contains NaN or Inf.")
        return state

    def update_action(self, action: Any) -> list[str]:
        if self._stop_event.is_set() or PROCESS_STOPPING.is_set():
            raise RuntimeError("Astribot control bridge is stopping.")
        target = self._target_from_action(action)
        if self.control_mode == "safe":
            # Keep the stored trajectory target consistent with SDK gripper
            # commands. Startup/hold targets may contain encoder zero offsets;
            # clipping only _command_from_target leaves the tracker chasing them.
            target = target.copy()
            target[[14, 22]] = np.clip(target[[14, 22]], 0.0, 100.0)
        names, cmd_list = self._command_from_target(target)
        now = time.time()
        with self._lock:
            if self.control_mode in {"g1", "safe"} and self._fault is not None:
                raise RuntimeError(f"Control fault latched: {self._fault}; explicit hold_current or service restart required")
            self._last_target_monotonic = time.monotonic()
            self._stale_hold = False
            self._latest_names = names
            self._latest_cmd = cmd_list
            self._latest_action = target.copy()
            self._action_count += 1
            self._last_action_time = now
        return names

    def move_to_pose(self, action: Any, *, duration: float) -> list[str]:
        with self._sdk_lock:
            return self._move_to_pose_locked(action, duration=duration)

    def _move_to_pose_locked(self, action: Any, *, duration: float) -> list[str]:
        if self.control_mode in {"g1", "safe"}:
            raise RuntimeError("tracked modes prohibit blocking SDK move")
        if self._stop_event.is_set() or PROCESS_STOPPING.is_set():
            raise RuntimeError("Astribot control bridge is stopping.")
        target = self._target_from_action(action)
        names, cmd_list = self._command_from_target(target)
        duration = max(float(duration), 0.2)
        with self._lock:
            self._latest_names = None
            self._latest_cmd = None
            self._latest_action = target.copy()
            self._action_count += 1
            self._last_action_time = time.time()

        if self.execute:
            self.astribot.move_joints_position(
                names,
                cmd_list,
                duration=duration,
                use_wbc=self.use_wbc,
                add_default_torso=self.add_default_torso,
            )

        with self._lock:
            self._latest_names = names
            self._latest_cmd = cmd_list
        return names

    def _target_from_action(self, action: Any) -> np.ndarray:
        action_len = np.asarray(action, dtype=np.float32).reshape(-1).shape[0]
        current_state = self.get_state() if action_len == ARM_GRIPPER_DIM else None
        return _coerce_s1_action(action, current_state=current_state)

    def _command_from_target(self, target: np.ndarray) -> tuple[list[str], list[list[float]]]:
        command: dict[str, np.ndarray] = {
            "astribot_torso": target[TORSO_SLICE],
            "astribot_arm_left": target[LEFT_ARM_SLICE],
            "astribot_arm_right": target[RIGHT_ARM_SLICE],
            "astribot_head": target[HEAD_SLICE],
        }

        if self.control_mode == "g1":
            pass  # Dedicated gripper loop owns both continuous gripper targets.
        elif self.gripper_binary:
            self._update_binary_grippers(target)
        else:
            command["astribot_gripper_left"] = np.clip(target[LEFT_GRIPPER_SLICE], 0.0, 100.0)
            command["astribot_gripper_right"] = np.clip(target[RIGHT_GRIPPER_SLICE], 0.0, 100.0)

        names = [name for name in CONTROL_NAMES if name in command]
        cmd_list = [command[name].astype(np.float32).reshape(-1).tolist() for name in names]
        return names, cmd_list

    def hold_current(self) -> list[str]:
        if self._stop_event.is_set() or PROCESS_STOPPING.is_set():
            raise RuntimeError("Astribot control bridge is stopping.")
        with self._sdk_lock:
            measured = self.get_state()
            with self._lock:
                if self.control_mode == "safe" and self._fault is not None:
                    raise RuntimeError("Safe-mode fault is latched; inspect robot then restart service")
                self._fault = None
            return self.update_action(measured)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            last_action_time = self._last_action_time
            last_command_time = self._last_command_time
            return {
                "execute": self.execute,
                "control_mode": self.control_mode,
                "fault": self._fault,
                "stale_hold": self._stale_hold,
                "target_timeout_s": self.target_timeout_s,
                "command_frequency": self.command_frequency,
                "safe_protocol_version": 2 if self.control_mode == "safe" else None,
                "safe_control_revision": 6,
                "safe_command_tracking": self.safe_command_tracking,
                "safe_alignment_supported": True,
                "safe_alignment": self._safe_alignment,
                "safe_execution_style": "g1_combined" if self.safe_g1_combined else "filter",
                "tracking_arm_velocity": self.tracking_arm_velocity,
                "gripper_frequency": self.gripper_frequency,
                "safe_joint_envelope": self.safe_joint_envelope,
                "safe_measured_velocity": self.safe_measured_velocity,
                "safe_measured_acceleration": self.safe_measured_acceleration,
                "safe_tracking_error": self.safe_tracking_error,
                "safe_policy_velocity": self.safe_policy_velocity,
                "safe_policy_acceleration": self.safe_policy_acceleration,
                "safe_direct_gripper": self.safe_direct_gripper,
                "safe_gripper_feedback_tracking": self.safe_gripper_feedback_tracking,
                "safe_gripper_limit_retention": self.safe_gripper_limit_retention,
                "safe_gripper_output_smoothing_s": self.safe_gripper_output_smoothing_s,
                "safe_gripper_smoothing_s": self.safe_gripper_smoothing_s,
                "safe_gripper_velocity": self.safe_gripper_velocity,
                "safe_gripper_acceleration": self.safe_gripper_acceleration,

                "tracking_diagnostic": self._tracking_diagnostic,
                "tracking_diagnostic_age_s": (time.time() - self._tracking_diagnostic["server_time_s"]
                                               if self._tracking_diagnostic is not None else None),
                "actions": self._action_count,
                "commands": self._command_count,
                "body_commands": self._body_command_count,
                "gripper_commands": self._gripper_command_count,
                "errors": self._error_count,
                "has_target": self._latest_cmd is not None,
                "last_action_age_s": time.time() - last_action_time if last_action_time else None,
                "last_command_age_s": time.time() - last_command_time if last_command_time else None,
            }

    def log_status(self) -> None:
        stats = self.snapshot()
        last_action = "never" if stats["last_action_age_s"] is None else f"{stats['last_action_age_s']:.1f}s ago"
        last_command = (
            "never" if stats["last_command_age_s"] is None else f"{stats['last_command_age_s']:.1f}s ago"
        )
        print(
            "[astribot-control] "
            f"execute={stats['execute']} target={stats['has_target']} actions={stats['actions']} "
            f"commands={stats['commands']} errors={stats['errors']} "
            f"last_action={last_action} last_command={last_command}",
            flush=True,
        )

    @staticmethod
    def _velocity_target(target: np.ndarray, current: np.ndarray, velocity: float, dt: float) -> np.ndarray:
        delta = target - current
        scale = max(1.0, float(np.max(np.abs(delta))) / (velocity * dt))
        return current + delta / scale

    def _tracking_tick(self, *, grippers: bool) -> None:
        # Serialize SDK access; recheck the target under this lock so move/stop
        # cannot be followed by a stale target captured before the transition.
        with self._sdk_lock:
            with self._lock:
                if self._latest_names is None or self._stop_event.is_set() or PROCESS_STOPPING.is_set():
                    return
                target = self._latest_action.copy()
            measured = self.get_state()
            with self._lock:
                # Refresh the snapshot after the SDK read; a newer network target
                # may have arrived while feedback was being fetched.
                if self._latest_names is None or self._stop_event.is_set() or PROCESS_STOPPING.is_set():
                    return
                if time.monotonic() - self._last_target_monotonic > self.target_timeout_s and not self._stale_hold:
                    self._latest_action = measured.copy()
                    self._stale_hold = True
                    print("[astribot-control] target timeout: holding measured pose", flush=True)
                target = self._latest_action.copy()
            if grippers:
                names = ["astribot_gripper_left", "astribot_gripper_right"]
                slices = [LEFT_GRIPPER_SLICE, RIGHT_GRIPPER_SLICE]
                velocities = [self.gripper_velocity] * 2
                dt = 1.0 / self.gripper_frequency
            else:
                names = ["astribot_torso", "astribot_arm_left", "astribot_arm_right", "astribot_head"]
                slices = [TORSO_SLICE, LEFT_ARM_SLICE, RIGHT_ARM_SLICE, HEAD_SLICE]
                velocities = [self.tracking_torso_velocity, self.tracking_arm_velocity,
                              self.tracking_arm_velocity, self.tracking_head_velocity]
                dt = 1.0 / self.command_frequency
            # G1 scales the full 14D arm error by one factor, preserving the
            # direction of the dual-arm error vector rather than clipping joints independently.
            arm_indices = np.r_[7:14, 15:22]
            arm_command = None if grippers else self._velocity_target(
                target[arm_indices], measured[arm_indices], self.tracking_arm_velocity, dt)
            commands = []
            for sl, speed in zip(slices, velocities):
                desired = np.clip(target[sl], 0, 100) if grippers else target[sl]
                command = self._velocity_target(desired, measured[sl], speed, dt)
                if grippers:
                    command = np.clip(command, 0, 100)
                elif sl == LEFT_ARM_SLICE:
                    command = arm_command[:7]
                elif sl == RIGHT_ARM_SLICE:
                    command = arm_command[7:]
                commands.append(command.tolist())
            if self.execute:
                if self._stop_event.is_set() or PROCESS_STOPPING.is_set():
                    return
                self.astribot.set_joints_position(names, commands, control_way="direct",
                                                 use_wbc=False, add_default_torso=False)
                with self._lock:
                    self._command_count += 1
                    if grippers:
                        self._gripper_command_count += 1
                    else:
                        self._body_command_count += 1
                    self._last_command_time = time.time()

    def _tracking_loop(self, *, grippers: bool) -> None:
        period = 1.0 / (self.gripper_frequency if grippers else self.command_frequency)
        while not self._stop_event.is_set() and not PROCESS_STOPPING.is_set():
            started = time.monotonic()
            try:
                self._tracking_tick(grippers=grippers)
            except Exception as exc:
                with self._lock:
                    self._error_count += 1
                    # Latch SDK/feedback failures; ordinary policy targets cannot
                    # silently re-arm control. Explicit hold_current reads fresh state.
                    self._fault = str(exc)
                    self._latest_names = None
                    self._latest_cmd = None
                print(f"[astribot-control] tracking failed: {exc}", flush=True)
            self._stop_event.wait(max(0.0, period - (time.monotonic() - started)))

    def _gripper_loop(self) -> None:
        self._tracking_loop(grippers=True)

    def disarm(self) -> None:
        with self._sdk_lock, self._lock:
            self._safe_alignment = False
            self._fault = "explicit stop; restart service after inspection"
            self._latest_names = None
            self._latest_cmd = None

    def set_safe_alignment(self, enabled: bool) -> None:
        with self._sdk_lock, self._lock:
            if not self.safe_g1_combined or self._fault or self._stop_event.is_set():
                raise RuntimeError("Alignment requires healthy safe G1 combined control")
            joints = np.r_[3:14, 15:22, 23:25]
            if not enabled and np.max(np.abs(self._safe_velocity[joints])) > self.safe_policy_velocity:
                raise RuntimeError("Alignment has not decelerated to policy speed")
            self._safe_alignment = enabled

    def _safe_tick(self) -> None:
        with self._sdk_lock:
            with self._lock:
                if self._fault or self._latest_names is None or self._stop_event.is_set() or PROCESS_STOPPING.is_set():
                    return
                target = self._latest_action.copy()
                target_sequence = self._action_count
            q = self.get_state().astype(np.float64)
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(target)):
                raise RuntimeError("non-finite safe control feedback or target")
            now = time.monotonic()
            joints = np.r_[3:14, 15:22, 23:25]
            grips = [14, 22]
            measured_speed_limit = max(self.safe_measured_velocity, 0.6) if self._safe_alignment else self.safe_measured_velocity
            measured_acceleration_limit = max(self.safe_measured_acceleration, 3.0) if self._safe_alignment else self.safe_measured_acceleration
            if self._safe_origin is None:
                # Encoder zero can be slightly outside the command interval.
                # Preserve measured feedback; sanitize only the initial command.
                if np.any(q[grips] < -5.0) or np.any(q[grips] > 100.5):
                    raise RuntimeError("initial gripper feedback outside tolerated range [-5.0, 100.5]")
                self._safe_origin = q.copy()
                self._safe_command = q.copy()
                self._safe_command[grips] = np.clip(q[grips], 0, 100)
                self._safe_feedback = q.copy()
                self._safe_time = now
            elapsed = now - self._safe_time
            if elapsed > 0.2:
                raise RuntimeError("safe control feedback cycle exceeded 0.2s")
            measured_velocity = np.zeros(25)
            if elapsed > 1e-6:
                measured_velocity = (q - self._safe_feedback) / elapsed
                if np.max(np.abs(measured_velocity[joints])) > measured_speed_limit:
                    index = int(joints[np.argmax(np.abs(measured_velocity[joints]))])
                    offset = 0
                    for component, size in STATE_COMPONENTS:
                        if index < offset + size:
                            joint_name = f"{component}[{index - offset}]"
                            break
                        offset += size
                    raise RuntimeError(
                        f"measured joint speed exceeded {measured_speed_limit:g} rad/s: joint={index} ({joint_name}), "
                        f"velocity={measured_velocity[index]:.6f}, "
                        f"previous_q={self._safe_feedback[index]:.6f}, q={q[index]:.6f}, "
                        f"dt_s={elapsed:.6f}, last_command={self._safe_command[index]:.6f}; "
                        "grippers are excluded from this check"
                    )
            # SDK feedback can refresh more slowly than the command loop.
            # Estimate acceleration between non-overlapping >=100ms velocity
            # windows, using their midpoints rather than double-differencing
            # sample-and-hold encoder values at every command tick.
            if self._safe_window is None:
                self._safe_window = (now, q.copy(), None, None)
            window_start, window_q, previous_v, previous_mid = self._safe_window
            window_dt = now - window_start
            if window_dt >= 0.1:
                window_v = (q - window_q) / window_dt
                midpoint = (now + window_start) / 2
                if previous_v is not None:
                    acceleration = (window_v - previous_v) / (midpoint - previous_mid)
                    index = int(joints[np.argmax(np.abs(acceleration[joints]))])
                    if abs(acceleration[index]) > measured_acceleration_limit:
                        raise RuntimeError(
                            f"measured joint acceleration exceeded {measured_acceleration_limit:g} rad/s^2: joint={index}, "
                            f"acceleration={acceleration[index]:.4f}, velocity={window_v[index]:.4f}, "
                            f"previous_velocity={previous_v[index]:.4f}, window_s={window_dt:.4f}"
                        )
                self._safe_window = (now, q.copy(), window_v, midpoint)
            if np.max(np.abs(q[joints] - self._safe_origin[joints])) > self.safe_joint_envelope:
                index = int(joints[np.argmax(np.abs(q[joints] - self._safe_origin[joints]))])
                raise RuntimeError(
                    f"measured joint displacement exceeded {self.safe_joint_envelope:g} rad from initial pose: "
                    f"joint={index}, initial={self._safe_origin[index]:.6f}, q={q[index]:.6f}, "
                    f"last_command={self._safe_command[index]:.6f}, target={target[index]:.6f}"
                )
            if np.max(np.abs(target[joints] - self._safe_origin[joints])) > self.safe_joint_envelope:
                raise RuntimeError(f"requested pose outside {self.safe_joint_envelope:g} rad safe envelope; do not auto-align")
            if np.max(np.abs(self._safe_command[joints] - q[joints])) > self.safe_tracking_error:
                raise RuntimeError(f"safe joint tracking error exceeded {self.safe_tracking_error:g} rad")
            if time.monotonic() - self._last_target_monotonic > self.target_timeout_s:
                raise RuntimeError("safe target timeout")
            dt = min(max(elapsed, 0), 1.0 / self.command_frequency)
            command = self._safe_command.copy()
            # Slew from LAST COMMAND: feedback cannot jump the command.
            # Limits cannot be relaxed via legacy speed flags.
            # Acceleration-limited velocity, braking as remaining distance shrinks.
            # Do not snap to target: snapping would violate acceleration at reversal.
            velocity = self._safe_velocity.copy()
            groups = ((joints, 0.02, 0.05), (grips, self.safe_gripper_velocity, self.safe_gripper_acceleration))
            if self.safe_g1_combined:
                # Same feedback groups as G1; both arms share one 14D scale.
                # Grippers join the body SDK call, never a separate worker.
                groups = ((np.r_[7:14, 15:22], 0.02, 0.05),
                          (np.r_[3:7], 0.02, 0.05), (np.r_[23:25], 0.02, 0.05),
                          (grips, self.safe_gripper_velocity, self.safe_gripper_acceleration))
            gripper_baseline_command = None
            gripper_unsmoothed_command = None
            gripper_retention = 1.0 if self._safe_alignment else self.safe_gripper_limit_retention
            for indices, vmax, amax in groups:
                if self.safe_direct_gripper and np.array_equal(indices, grips):
                    if self.safe_gripper_feedback_tracking:
                        # Section 16's per-jaw feedback law, at the unchanged
                        # combined command cadence (no independent jaw thread).
                        gripper_baseline_command = []
                        gripper_unsmoothed_command = []
                        for index in grips:
                            desired = np.clip(target[index:index + 1], 0, 100)
                            limited = float(np.clip(self._velocity_target(
                                desired, q[index:index + 1], self.gripper_velocity,
                                1.0 / self.command_frequency)[0], 0, 100))
                            gripper_baseline_command.append(limited)
                            # Keep the configured fraction of the original removed
                            # displacement. This relaxes the feedback step limit,
                            # not the position bounds, and applies to both jaws.
                            goal = float(desired[0]) + gripper_retention * (limited - float(desired[0]))
                            gripper_unsmoothed_command.append(goal)
                            if self.safe_gripper_output_smoothing_s > 0 and not self._safe_alignment:
                                # Exact first-order low-pass update from last SDK
                                # command: unity steady-state gain, no target rescale.
                                alpha = -np.expm1(-max(elapsed, 0.0) / self.safe_gripper_output_smoothing_s)
                                command[index] += alpha * (goal - command[index])
                            else:
                                command[index] = goal
                        velocity[indices] = 0.0
                    elif self.safe_gripper_smoothing_s > 0:
                        # Smooth from the last command, never from measured jaw
                        # position: encoder deadband cannot stall target progress.
                        # Each jaw brakes independently when its target reverses.
                        desired = np.clip((target[indices] - command[indices]) /
                                          self.safe_gripper_smoothing_s, -vmax, vmax)
                        velocity[indices] += np.clip(desired - velocity[indices],
                                                     -amax * dt, amax * dt)
                        command[indices] += velocity[indices] * dt
                    else:
                        command[indices] = np.clip(target[indices], 0, 100)
                        velocity[indices] = 0.0
                    continue
                if not np.array_equal(indices, grips):
                    vmax, amax = self.safe_policy_velocity, self.safe_policy_acceleration
                    if self.safe_g1_combined:
                        if np.array_equal(indices, np.r_[3:7]):
                            vmax = min(vmax, self.tracking_torso_velocity)
                        elif np.array_equal(indices, np.r_[23:25]):
                            vmax = min(vmax, self.tracking_head_velocity)
                        else:
                            vmax = min(vmax, self.tracking_arm_velocity)
                if self._safe_alignment and not np.array_equal(indices, grips):
                    vmax, amax = 0.35, 0.5
                error = target[indices] - command[indices]
                desired = np.sign(error) * np.minimum(vmax, np.sqrt(2 * amax * np.abs(error)))
                if self.safe_g1_combined:
                    # Feedback target first, then a last-command acceleration
                    # backstop: encoder disturbances cannot jump the SDK target.
                    distance = float(np.max(np.abs(error)))
                    speed = min(vmax, np.sqrt(2 * amax * distance))
                    tracking_feedback = q[indices].copy()
                    if np.array_equal(indices, grips):
                        if np.any(tracking_feedback < -5.0) or np.any(tracking_feedback > 100.5):
                            raise RuntimeError("gripper feedback outside tolerated range [-5.0, 100.5]")
                        # Encoders may keep a small zero offset even while holding
                        # the physical endpoint. Sanitize the tracking reference
                        # every tick, preserving raw q for feedback diagnostics.
                        tracking_feedback = np.clip(tracking_feedback, 0, 100)
                    candidate = tracking_feedback.copy()
                    if speed > 0 and dt > 0:
                        candidate = self._velocity_target(target[indices], tracking_feedback, speed, dt)
                    # Feedback may lag, drift or overshoot the fixed target.
                    # It must not create a new reference beyond the interval
                    # between the last command and the requested target. In
                    # particular, an idle target collapses this interval to a
                    # single value, so encoder drift cannot move the command.
                    candidate = np.clip(candidate,
                                        np.minimum(command[indices], target[indices]),
                                        np.maximum(command[indices], target[indices]))
                    if self._safe_alignment or np.array_equal(indices, grips):
                        # r/b supply a smooth fixed-reference trajectory. A tiny
                        # feedback-relative step can stall below SDK deadband.
                        # Continuous grippers likewise follow the requested
                        # target rather than stalling at feedback deadband.
                        # Follow the trajectory with slew/acceleration backstops;
                        # measured tracking error still faults before SDK output.
                        candidate = target[indices]
                    if self.safe_command_tracking and not self._safe_alignment and np.array_equal(indices, np.r_[7:14, 15:22]):
                        # Advance from the last SDK command, preserving the full
                        # dual-arm error direction. Braking and acceleration
                        # limits remain active; feedback error still faults above.
                        candidate = self._velocity_target(target[indices], command[indices],
                                                          max(speed, 1e-12), dt) if dt > 0 else command[indices]
                    desired = (candidate - command[indices]) / dt if dt > 0 else np.zeros_like(error)
                    desired /= max(1.0, float(np.max(np.abs(desired))) / vmax)
                    change = desired - velocity[indices]
                    if dt > 0:
                        change /= max(1.0, float(np.max(np.abs(change))) / (amax * dt))
                        velocity[indices] += change
                else:
                    velocity[indices] += np.clip(desired - velocity[indices], -amax * dt, amax * dt)
                command[indices] += velocity[indices] * dt
            if np.max(np.abs(command[joints] - self._safe_origin[joints])) > self.safe_joint_envelope:
                raise RuntimeError("acceleration-limited command would exceed safe envelope")
            # Gripper endpoint saturation takes priority over acceleration
            # smoothing. Do not fault the whole robot on bounded overshoot of
            # an otherwise valid target during braking/reversal.
            for index in grips:
                if command[index] < 0 or command[index] > 100:
                    command[index] = np.clip(command[index], 0, 100)
                    velocity[index] = 0.0
            names, commands = self._command_from_target(command)
            sdk_started = time.monotonic()
            sdk_sent = False
            if self.execute and not self._stop_event.is_set() and not PROCESS_STOPPING.is_set():
                self.astribot.set_joints_position(names, commands, control_way=self.control_way,
                                                 use_wbc=False, add_default_torso=False)
                sdk_sent = True
            sdk_elapsed = time.monotonic() - sdk_started
            arms = np.r_[7:14, 15:22]
            diagnostic = {
                "server_time_s": time.time(), "target_sequence": target_sequence,
                "alignment": self._safe_alignment, "sdk_sent": sdk_sent,
                "cycle_elapsed_s": elapsed, "limiter_dt_s": dt,
                "sdk_call_elapsed_s": sdk_elapsed,
                "target": target.tolist(), "sdk_command": command.tolist(),
                "feedback_before_command": q.tolist(),
                "command_velocity": ((command - self._safe_command) / elapsed).tolist() if elapsed > 1e-6 else None,
                "measured_velocity": measured_velocity.tolist(),
                "arm_target_command_gap_max": float(np.max(np.abs(target[arms] - command[arms]))),
                "arm_command_feedback_gap_max": float(np.max(np.abs(command[arms] - q[arms]))),
                "arm_target_feedback_gap_max": float(np.max(np.abs(target[arms] - q[arms]))),
                "command_frequency": self.command_frequency,
                "command_tracking": self.safe_command_tracking,
                "policy_velocity_limit": self.safe_policy_velocity,
                "policy_acceleration_limit": self.safe_policy_acceleration,
                "tracking_arm_velocity_limit": self.tracking_arm_velocity,
                "gripper_velocity_limit": (None if self.safe_gripper_feedback_tracking and gripper_retention < 1
                                           else self.gripper_velocity if self.safe_gripper_feedback_tracking else self.safe_gripper_velocity),
                "gripper_baseline_velocity": self.gripper_velocity,
                "gripper_limit_retention": gripper_retention,
                "gripper_baseline_sdk_command": gripper_baseline_command,
                "gripper_unsmoothed_sdk_command": gripper_unsmoothed_command,
                "gripper_output_smoothing_s": 0.0 if self._safe_alignment else self.safe_gripper_output_smoothing_s,
            }
            self._safe_velocity = velocity
            self._safe_command = command
            self._safe_feedback = q.copy()
            self._safe_time = now
            with self._lock:
                self._tracking_diagnostic = diagnostic
                self._command_count += int(self.execute)
                if self.execute:
                    self._last_command_time = time.time()

    def _safe_loop(self) -> None:
        while not self._stop_event.is_set() and not PROCESS_STOPPING.is_set():
            started = time.monotonic()
            try:
                self._safe_tick()
            except Exception as exc:
                self.disarm()
                with self._lock:
                    self._fault = str(exc)
                    self._error_count += 1
                print(f"[astribot-control] SAFE FAULT: {exc}; no further SDK targets; use physical stop if moving", flush=True)
            self._stop_event.wait(max(0, 1.0 / self.command_frequency - (time.monotonic() - started)))

    def _command_loop(self) -> None:
        if self.control_mode == "safe":
            self._safe_loop()
            return
        if self.control_mode == "g1":
            self._tracking_loop(grippers=False)
            return
        period = 1.0 / self.command_frequency
        next_time = time.monotonic()
        while not self._stop_event.is_set():
            with self._lock:
                names = list(self._latest_names) if self._latest_names is not None else None
                cmd = [list(values) for values in self._latest_cmd] if self._latest_cmd is not None else None

            if names is not None and cmd is not None:
                try:
                    if self._stop_event.is_set() or PROCESS_STOPPING.is_set():
                        break
                    self.astribot.set_joints_position(
                        names,
                        cmd,
                        control_way=self.control_way,
                        use_wbc=self.use_wbc,
                        add_default_torso=self.add_default_torso,
                    )
                    with self._lock:
                        self._command_count += 1
                        self._last_command_time = time.time()
                except Exception as exc:
                    with self._lock:
                        self._error_count += 1
                    print(f"[astribot-control] set_joints_position error: {exc}", flush=True)

            next_time += period
            sleep_s = next_time - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_time = time.monotonic()

    def _update_binary_grippers(self, target: np.ndarray) -> None:
        if not self.execute:
            return
        for side, action_index in (("left", 14), ("right", 22)):
            closed = bool(float(target[action_index]) >= self.gripper_threshold)
            if self._last_gripper[side] == closed:
                continue
            name = self.astribot.effector_left_name if side == "left" else self.astribot.effector_right_name
            try:
                if closed:
                    self.astribot.set_effector_max_force([name], max_force=[80.0])
                    self.astribot.close_effector([name], duration=0.05)
                else:
                    self.astribot.open_effector([name], duration=0.05)
                self._last_gripper[side] = closed
                print(f"[astribot-control] gripper {side}: {'CLOSE' if closed else 'OPEN'}", flush=True)
            except Exception as exc:
                print(f"[astribot-control] gripper {side} failed: {exc}", flush=True)


class BufferedActionPlayerThread(threading.Thread):
    def __init__(
        self,
        control: AstribotControlBridge,
        *,
        action_frequency: float,
        max_buffer_steps: int,
    ):
        super().__init__(daemon=True, name="astribot-action-player")
        self.control = control
        self.action_frequency = max(1.0, float(action_frequency))
        self.max_buffer_steps = max(1, int(max_buffer_steps))
        self._buffer: collections.deque[Any] = collections.deque()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._played = 0
        self._errors = 0

    def append_chunk(self, actions: list[Any], *, replace: bool = False) -> int:
        with self._lock:
            if replace:
                self._buffer.clear()
            room = max(0, self.max_buffer_steps - len(self._buffer))
            kept = actions[:room]
            self._buffer.extend(kept)
            return len(kept)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"queued": len(self._buffer), "played": self._played, "errors": self._errors}

    def stop(self) -> None:
        self._stop_event.set()
        self.clear()
        self.join(timeout=1.0)

    def run(self) -> None:
        period = 1.0 / self.action_frequency
        next_time = time.monotonic()
        while not self._stop_event.is_set():
            action = None
            with self._lock:
                if self._buffer:
                    action = self._buffer.popleft()
            if action is not None:
                try:
                    self.control.update_action(action)
                    with self._lock:
                        self._played += 1
                except Exception as exc:
                    with self._lock:
                        self._errors += 1
                    print(f"[astribot-control] buffered action error: {exc}", flush=True)

            next_time += period
            sleep_s = next_time - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_time = time.monotonic()


class AstribotActionServerThread(threading.Thread):
    """Length-prefixed JSON action endpoint for workstation policy code."""

    def __init__(
        self,
        control: AstribotControlBridge,
        player: BufferedActionPlayerThread,
        *,
        host: str,
        port: int,
        replan_horizon: int,
    ):
        super().__init__(daemon=True, name=f"astribot-action-server-{port}")
        self.control = control
        self.player = player
        self.host = host
        self.port = int(port)
        self.replan_horizon = int(replan_horizon)
        self._stop_event = threading.Event()
        self._server_socket: socket.socket | None = None
        self._client_threads: list[threading.Thread] = []
        self._client_sockets: set[socket.socket] = set()
        self._client_lock = threading.Lock()

    def stop(self) -> None:
        self._stop_event.set()
        if self._server_socket is not None:
            with contextlib.suppress(Exception):
                self._server_socket.close()
        with self._client_lock:
            client_sockets = list(self._client_sockets)
        for conn in client_sockets:
            with contextlib.suppress(Exception):
                conn.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(Exception):
                conn.close()
        for thread in self._client_threads:
            thread.join(timeout=0.5)
        self.join(timeout=1.0)

    def run(self) -> None:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket = server_socket
        try:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(4)
            server_socket.settimeout(0.2)
            print(f"[astribot-control] action server listening on {self.host}:{self.port}", flush=True)
            while not self._stop_event.is_set():
                try:
                    conn, addr = server_socket.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop_event.is_set():
                        break
                    raise
                thread = threading.Thread(
                    target=self._serve_client,
                    args=(conn, addr),
                    daemon=True,
                    name=f"astribot-action-client-{addr}",
                )
                thread.start()
                self._client_threads.append(thread)
        finally:
            with contextlib.suppress(Exception):
                server_socket.close()

    def _serve_client(self, conn: socket.socket, addr: tuple[str, int]) -> None:
        print(f"[astribot-control] client connected: {addr}", flush=True)
        with self._client_lock:
            self._client_sockets.add(conn)
        try:
            while not self._stop_event.is_set():
                try:
                    request = recv_json(conn, timeout=0.5)
                except socket.timeout:
                    continue
                response = self._handle_request(request)
                send_json(conn, response)
        except (ConnectionError, BrokenPipeError, ConnectionResetError, OSError) as exc:
            print(f"[astribot-control] client {addr} disconnected: {exc}", flush=True)
        except Exception as exc:
            with contextlib.suppress(Exception):
                send_json(conn, {"ok": False, "error": str(exc)})
            print(f"[astribot-control] client {addr} error: {exc}", flush=True)
        finally:
            with contextlib.suppress(Exception):
                conn.close()
            with self._client_lock:
                self._client_sockets.discard(conn)

    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._stop_event.is_set() or PROCESS_STOPPING.is_set():
            return self._status_response(ok=False, extra={"error": "server is stopping"})

        required_mode = request.get("required_control_mode")
        if required_mode is not None and required_mode != self.control.control_mode:
            return self._status_response(ok=False, extra={"error": "control mode mismatch"})
        if self.control.control_mode == "g1" and required_mode != "g1" and any(
            key in request for key in ("action", "chunk", "move", "move_to_pose", "hold_current")
        ):
            return self._status_response(ok=False, extra={"error": "g1 mode requires a matching g1 client"})
        if self.control.control_mode == "g1" and "chunk" in request:
            return self._status_response(ok=False, extra={"error": "g1 mode accepts latest targets, not chunks"})

        if request.get("disarm"):
            self.player.clear()
            self.control.disarm()
            return self._status_response(extra={"disarmed": True})
        if self.control.control_mode == "safe" and any(k in request for k in ("action", "chunk", "move", "move_to_pose", "hold_current", "safe_alignment")):
            if required_mode != "safe" or "chunk" in request:
                return self._status_response(ok=False, extra={"error": "safe mode requires matching single-target client"})
            expected_style = "g1_combined" if self.control.safe_g1_combined else "filter"
            if request.get("required_safe_execution_style", "filter") != expected_style:
                return self._status_response(ok=False, extra={"error": "safe execution style mismatch; update client/service"})

        if "safe_alignment" in request:
            if type(request["safe_alignment"]) is not bool:
                raise ValueError("safe_alignment must be boolean")
            self.control.set_safe_alignment(request["safe_alignment"])
            return self._status_response()

        if request.get("get_state"):
            return self._status_response(state=self.control.get_state())

        if request.get("hold_current"):
            self.player.clear()
            names = self.control.hold_current()
            return self._status_response(extra={"names": names, "held_current": True})

        if (
            request.get("clear_buffer")
            and "action" not in request
            and "chunk" not in request
            and "move" not in request
            and "move_to_pose" not in request
        ):
            self.player.clear()
            return self._status_response(extra={"cleared": True})

        if "move" in request or "move_to_pose" in request:
            self.player.clear()
            move_action = request["move"] if "move" in request else request["move_to_pose"]
            names = self.control.move_to_pose(move_action, duration=float(request.get("duration", 3.0)))
            return self._status_response(
                extra={
                    "names": names,
                    "moved": bool(self.control.execute),
                    "dry_run": not self.control.execute,
                }
            )

        if "chunk" in request and request["chunk"] is not None:
            chunk = list(request["chunk"])
            if self.replan_horizon > 0:
                chunk = chunk[: self.replan_horizon]
            queued = self.player.append_chunk(chunk, replace=bool(request.get("replace_buffer", False)))
            return self._status_response(
                extra={
                    "queued": queued,
                    "received": len(chunk),
                    "dry_run": not self.control.execute,
                }
            )

        if "action" in request and request["action"] is not None:
            if bool(request.get("clear_buffer", True)):
                self.player.clear()
            names = self.control.update_action(request["action"])
            return self._status_response(extra={"names": names, "dry_run": not self.control.execute})

        return self._status_response(
            ok=False,
            extra={"error": "expected get_state, hold_current, move, action, or chunk"},
        )

    def _status_response(
        self,
        *,
        ok: bool = True,
        state: np.ndarray | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {
            "ok": bool(ok),
            "buffer": self.player.snapshot(),
            "control": self.control.snapshot(),
        }
        if state is not None:
            response["state"] = state.astype(np.float32).tolist()
        if extra:
            response.update(extra)
        return response


def parse_camera_routes(args: argparse.Namespace) -> list[CameraRoute]:
    routes = [
        CameraRoute("head", args.head_camera, int(args.head_port)),
    ]
    if not args.disable_wrist_cameras:
        routes.extend(
            [
                CameraRoute("left_wrist", args.left_wrist_camera, int(args.left_wrist_port)),
                CameraRoute("right_wrist", args.right_wrist_camera, int(args.right_wrist_port)),
            ]
        )

    for item in args.camera or []:
        try:
            label, camera_name, port_text = item.split(":", 2)
        except ValueError as exc:
            raise ValueError("--camera entries must be label:camera_name:port, e.g. torso:torso_rgbd:55558") from exc
        routes.append(CameraRoute(label.strip(), camera_name.strip(), int(port_text)))

    deduped: list[CameraRoute] = []
    seen_ports: set[int] = set()
    seen_cameras: set[str] = set()
    for route in routes:
        if not route.camera_name:
            continue
        if route.port in seen_ports:
            raise ValueError(f"Duplicate ZMQ port {route.port} in camera routes.")
        if route.camera_name in seen_cameras:
            raise ValueError(f"Duplicate camera name {route.camera_name!r} in camera routes.")
        seen_ports.add(route.port)
        seen_cameras.add(route.camera_name)
        deduped.append(route)
    return deduped


def shutdown_astribot_client(astribot: Any) -> None:
    """Best-effort SDK shutdown without letting ROS teardown noise block exit."""
    if isinstance(astribot, AstribotCameraClient):
        astribot.shutdown()
        return

    interface = getattr(astribot, "astribot_interface", None)
    shutdown = getattr(interface, "shutdown", None) if interface is not None else None
    if not callable(shutdown):
        return

    rclpy_module = sys.modules.get("rclpy")
    try:
        rclpy_ok = bool(rclpy_module.ok()) if rclpy_module is not None else True
    except Exception:
        rclpy_ok = False
    if not rclpy_ok:
        print("[astribot-image-service] ROS is already shut down; skipping SDK shutdown.", flush=True)
        return

    with contextlib.suppress(Exception):
        shutdown()


def wait_for_cameras(astribot: Any, routes: list[CameraRoute], timeout_s: float) -> None:
    deadline = time.time() + max(0.0, float(timeout_s))
    pending = {route.camera_name for route in routes}
    while pending and time.time() <= deadline and not PROCESS_STOPPING.is_set():
        info = astribot.get_cameras_info()
        pending = {
            camera_name for camera_name in pending if not bool(info.get(camera_name, {}).get("activate", False))
        }
        if pending:
            time.sleep(0.5)

    if pending:
        print(
            "[astribot-image-service] Warning: cameras not reported active before timeout: "
            + ", ".join(sorted(pending)),
            flush=True,
        )


class AstribotCameraClient:
    """Minimal SDK-compatible camera client that does not load whole-body control."""

    def __init__(self, *, node_name: str):
        import rclpy
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.node import Node
        from sensor_msgs.msg import CompressedImage
        from std_srvs.srv import SetBool

        import astribot_ros_middleware as ast_ros_middleware

        self.rclpy = rclpy
        self.ReentrantCallbackGroup = ReentrantCallbackGroup
        self.CompressedImage = CompressedImage
        self.SetBool = SetBool
        self._subscriptions = []
        self._temp_subscriptions = []

        node_name_with_pid = f"{node_name}_{os.getpid()}"
        if not ast_ros_middleware.is_inited():
            try:
                ast_ros_middleware.init(node_name=node_name_with_pid, anonymous=True)
            except RuntimeError:
                pass
        if not rclpy.ok():
            try:
                rclpy.init(args=None)
            except RuntimeError:
                pass

        self.node = Node(node_name_with_pid)
        self.logger = self.node.get_logger()
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self._stop_event = threading.Event()
        self._spin_thread = threading.Thread(
            target=self._spin,
            daemon=True,
            name=f"ROS2-camera-spin-{node_name}",
        )
        self._spin_thread.start()

    def _spin(self) -> None:
        while not self._stop_event.is_set() and not PROCESS_STOPPING.is_set() and self.rclpy.ok():
            try:
                self.executor.spin_once(timeout_sec=0.1)
            except Exception as exc:
                print(f"[astribot-image-service] ROS spin warning: {exc}", flush=True)

    def shutdown(self) -> None:
        self._stop_event.set()
        self._spin_thread.join(timeout=1.0)
        for subscription in [*self._subscriptions, *self._temp_subscriptions]:
            with contextlib.suppress(Exception):
                self.node.destroy_subscription(subscription)
        with contextlib.suppress(Exception):
            self.executor.remove_node(self.node)
        with contextlib.suppress(Exception):
            self.node.destroy_node()

    def activate_camera(self, cameras_setting: dict[str, dict] | None = None) -> bool:
        del cameras_setting
        return self._set_camera_power(True)

    def deactivate_camera(self) -> bool:
        return self._set_camera_power(False)

    def _set_camera_power(self, enabled: bool) -> bool:
        client = self.node.create_client(self.SetBool, "/astribot_camera_control")
        if not client.wait_for_service(timeout_sec=1.0):
            print("[astribot-image-service] Warning: service /astribot_camera_control not available.", flush=True)
            return False

        req = self.SetBool.Request()
        req.data = bool(enabled)
        future = client.call_async(req)
        deadline = time.time() + 5.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.001)
        if not future.done():
            print("[astribot-image-service] Warning: camera control service call timed out.", flush=True)
            return False

        response = future.result()
        if getattr(response, "success", False):
            return True
        print(
            "[astribot-image-service] Warning: camera control service returned failure: "
            f"{getattr(response, 'message', '')}",
            flush=True,
        )
        return False

    @staticmethod
    def camera_name_to_topic_name(camera_name: str, image_type: str) -> str:
        if image_type in ("color", "grey", "depth", "ir"):
            return f"/astribot_camera/{camera_name}/{image_type}_compress"
        return f"/astribot_camera/{camera_name}/{image_type}_image"

    def is_camera_activated(self, camera_name: str, image_type: str) -> bool:
        topic_name = self.camera_name_to_topic_name(camera_name, image_type)
        real_topic_name = f"{topic_name}/compressed" if image_type != "depth" else topic_name
        message_received = threading.Event()

        def temp_callback(msg: Any) -> None:
            del msg
            message_received.set()

        try:
            temp_sub = self.node.create_subscription(self.CompressedImage, real_topic_name, temp_callback, 10)
            self._temp_subscriptions.append(temp_sub)
            return message_received.wait(timeout=0.2)
        except Exception:
            return False
        finally:
            if "temp_sub" in locals():
                with contextlib.suppress(Exception):
                    self.node.destroy_subscription(temp_sub)
                with contextlib.suppress(ValueError):
                    self._temp_subscriptions.remove(temp_sub)

    def register_image_callback(self, camera_name: str, image_type: str, callback: Any, need_decode: bool):
        topic_name = self.camera_name_to_topic_name(camera_name, image_type)
        if not topic_name.startswith("/astribot_camera/"):
            print(f"[astribot-image-service] Invalid camera topic: {topic_name}", flush=True)
            return None

        if not self.is_camera_activated(camera_name, image_type):
            print(
                f"[astribot-image-service] Warning: camera {camera_name}({image_type}) not active yet.",
                flush=True,
            )

        parts = topic_name.split("/")
        if parts[3] == "color_compress":
            subscription_topic = topic_name + "/compressed"
        else:
            subscription_topic = topic_name

        def ros_callback(msg: Any) -> None:
            try:
                if need_decode and getattr(msg, "format", "").lower() == "jpeg":
                    import cv2

                    np_arr = np.frombuffer(msg.data, np.uint8)
                    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    callback(subscription_topic, msg, cv_image.shape[1], cv_image.shape[0], cv_image)
                else:
                    callback(subscription_topic, msg, 0, 0, None)
            except Exception as exc:
                print(f"[astribot-image-service] Error processing image from {subscription_topic}: {exc}", flush=True)

        subscription = self.node.create_subscription(
            self.CompressedImage,
            subscription_topic,
            ros_callback,
            10,
            callback_group=self.ReentrantCallbackGroup(),
        )
        self._subscriptions.append(subscription)
        return subscription

    def get_cameras_info(self) -> dict[str, dict[str, bool]]:
        camera_name_list = ["head_rgbd", "torso_rgbd", "left_wrist_rgbd", "right_wrist_rgbd", "head_stereo"]
        return {
            camera_name: {"activate": self.is_camera_activated(camera_name, "color")}
            for camera_name in camera_name_list
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sdk-root",
        default="",
        help="Relative or absolute Astribot SDK root. On S1/Orin this must be the aarch64 SDK.",
    )
    parser.add_argument("--bind-host", default="0.0.0.0", help="Host/interface for ZMQ PUB sockets.")
    parser.add_argument("--node-name", default="astribot_image_service", help="ROS2 node name prefix.")
    parser.add_argument("--sdk-freq", type=float, default=30.0, help="Frequency passed to Astribot SDK.")
    parser.add_argument("--high-control-rights", action="store_true", help="Request high SDK control rights.")
    parser.add_argument("--no-activate-camera", action="store_true", help="Skip Astribot.activate_camera().")
    parser.add_argument("--activate-timeout-s", type=float, default=10.0, help="Seconds to wait for cameras to activate.")
    parser.add_argument("--deactivate-on-exit", action="store_true", help="Call Astribot.deactivate_camera() on exit.")
    parser.add_argument("--head-camera", default="head_rgbd", help="S1 head camera name.")
    parser.add_argument("--left-wrist-camera", default="left_wrist_rgbd", help="S1 left wrist camera name.")
    parser.add_argument("--right-wrist-camera", default="right_wrist_rgbd", help="S1 right wrist camera name.")
    parser.add_argument("--head-port", type=int, default=55555, help="ZMQ port for the head camera.")
    parser.add_argument("--left-wrist-port", type=int, default=55556, help="ZMQ port for the left wrist camera.")
    parser.add_argument("--right-wrist-port", type=int, default=55557, help="ZMQ port for the right wrist camera.")
    parser.add_argument("--disable-wrist-cameras", action="store_true", help="Only publish the head camera.")
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help="Extra route as label:camera_name:port, e.g. torso:torso_rgbd:55558.",
    )
    parser.add_argument(
        "--decode-before-publish",
        action="store_true",
        help="Ask SDK for decoded BGR frames and JPEG-encode before publishing.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality when --decode-before-publish is used.")
    parser.add_argument("--status-interval-s", type=float, default=5.0, help="Periodic stream status log interval.")
    parser.add_argument(
        "--full-sdk-client",
        action="store_true",
        help="Use Astribot() full SDK client instead of the camera-only ROS client.",
    )
    parser.add_argument(
        "--enable-control",
        action="store_true",
        help="Also start the S1 action/control endpoint and 250Hz SDK command loop.",
    )
    parser.add_argument(
        "--execute-control",
        action="store_true",
        help="Actually send accepted action targets to the SDK. Without this, control runs in dry-run mode.",
    )
    parser.add_argument("--control-bind-host", default="", help="Host/interface for the action TCP server.")
    parser.add_argument("--control-port", type=int, default=6200, help="Length-prefixed JSON action TCP port.")
    parser.add_argument(
        "--control-frequency",
        type=float,
        default=250.0,
        help="SDK set_joints_position repeat frequency for the latest target.",
    )
    parser.add_argument(
        "--control-action-frequency",
        type=float,
        default=30.0,
        help="Frequency for consuming queued action chunks.",
    )
    parser.add_argument(
        "--control-way",
        default="filter",
        choices=["filter", "direct"],
        help="Astribot SDK control_way used by the high-frequency command loop.",
    )
    parser.add_argument("--control-mode", choices=["legacy", "g1", "safe"], default="legacy")
    parser.add_argument("--tracking-arm-velocity", type=float, default=1.0, help="rad/s")
    parser.add_argument("--tracking-torso-velocity", type=float, default=0.1, help="rad/s")
    parser.add_argument("--tracking-head-velocity", type=float, default=0.3, help="rad/s")
    parser.add_argument("--control-target-timeout-s", type=float, default=5.0)
    parser.add_argument("--gripper-frequency", type=float, default=100.0)
    parser.add_argument("--safe-joint-envelope", type=float, default=0.15)
    parser.add_argument("--safe-measured-velocity", type=float, default=0.1)
    parser.add_argument("--safe-measured-acceleration", type=float, default=1.0)
    parser.add_argument("--safe-tracking-error", type=float, default=0.05)
    parser.add_argument("--safe-policy-velocity", type=float, default=0.02, help="Policy body command rad/s; alignment limits unchanged")
    parser.add_argument("--safe-policy-acceleration", type=float, default=0.05, help="Policy body command rad/s^2")
    parser.add_argument("--safe-gripper-smoothing-s", type=float, default=0.0, help="Optional gripper smoothing time constant for direct action mode; 0 disables")
    parser.add_argument("--safe-gripper-output-smoothing-s", type=float, default=0.0, help="Policy-only post-retention gripper low-pass time constant; 0 disables")
    parser.add_argument("--safe-gripper-limit-retention", type=float, default=1.0, help="Fraction of original gripper clipping retained: 1 unchanged, 0.5 halves removed displacement; policy only, both jaws")
    parser.add_argument("--safe-gripper-feedback-tracking", action="store_true", help="Use G1 per-jaw feedback tracking at the combined command frequency")
    parser.add_argument("--safe-command-tracking", action="store_true", help="Advance policy arms from last SDK command; retain acceleration and feedback fault checks")
    parser.add_argument("--safe-direct-gripper", action="store_true", help="Pass continuous gripper targets directly in the combined SDK command; optionally smooth using --safe-gripper-smoothing-s")
    parser.add_argument("--safe-gripper-velocity", type=float, default=1.0, help="Safe mode gripper position units/s; does not change body limits")
    parser.add_argument("--safe-gripper-acceleration", type=float, default=2.0, help="Safe mode gripper position units/s^2")
    parser.add_argument("--gripper-velocity", type=float, default=40.0, help="gripper position units/s")
    parser.add_argument("--control-use-wbc", action="store_true", help="Pass use_wbc=True to set_joints_position.")
    parser.add_argument(
        "--control-add-default-torso",
        action="store_true",
        help="Pass add_default_torso=True to set_joints_position.",
    )
    parser.add_argument(
        "--control-gripper-binary",
        action="store_true",
        help="Use open/close gripper commands instead of continuous gripper position targets.",
    )
    parser.add_argument(
        "--control-gripper-threshold",
        type=float,
        default=50.0,
        help="Close threshold for --control-gripper-binary.",
    )
    parser.add_argument(
        "--control-replan-horizon",
        type=int,
        default=50,
        help="Keep at most this many actions from an incoming chunk; <=0 keeps the whole chunk.",
    )
    parser.add_argument(
        "--control-max-buffer-steps",
        type=int,
        default=200,
        help="Maximum queued action steps accepted by the control endpoint.",
    )
    return parser


def main() -> None:
    faulthandler.enable()
    PROCESS_STOPPING.clear()
    args = build_arg_parser().parse_args()
    if args.control_mode == "g1" and args.execute_control:
        raise RuntimeError("G1 direct hardware execution disabled after unsafe motion report; use safe dry-run for diagnosis")
    use_full_sdk_client = bool(args.full_sdk_client or args.enable_control)
    ensure_astribot_sdk_runtime_env(args.sdk_root or None, check_native_log_dir=use_full_sdk_client)
    sdk_root = configure_astribot_sdk_path(args.sdk_root or None)
    if use_full_sdk_client:
        validate_numpy_for_full_sdk()
        validate_python_dependencies_for_full_sdk()
        validate_ros_extension_imports_for_full_sdk()
    routes = parse_camera_routes(args)
    if not routes:
        raise ValueError("No camera routes configured.")

    print(f"[astribot-image-service] SDK root: {sdk_root}", flush=True)
    print(
        "[astribot-image-service] Publishing routes: "
        + ", ".join(f"{route.label}:{route.camera_name}->{args.bind_host}:{route.port}" for route in routes),
        flush=True,
    )

    bridge = AstribotImageBridge(
        routes,
        bind_host=args.bind_host,
        jpeg_quality=args.jpeg_quality,
        decode_before_publish=args.decode_before_publish,
    )
    bridge.start()

    stop_event = threading.Event()
    stop_signal: dict[str, int | None] = {"signum": None}
    stop_count = {"value": 0}

    def _request_stop(signum, frame):
        del frame
        stop_count["value"] += 1
        stop_signal["signum"] = int(signum)
        if stop_count["value"] > 1:
            print("[astribot-image-service] second stop signal received; forcing exit.", flush=True)
            os._exit(128 + int(signum))
        print("[astribot-image-service] stop requested; shutting down threads. Press Ctrl-C again to force.", flush=True)
        PROCESS_STOPPING.set()
        stop_event.set()

    def _install_stop_signal_handlers() -> None:
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGINT, _request_stop)
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGTERM, _request_stop)

    _install_stop_signal_handlers()

    astribot = None
    subscriptions = []
    control_bridge: AstribotControlBridge | None = None
    action_player: BufferedActionPlayerThread | None = None
    action_server: AstribotActionServerThread | None = None
    try:
        if use_full_sdk_client:
            print("[astribot-image-service] Importing full Astribot SDK client...", flush=True)
            from astribot_sdk.core.astribot_api.astribot_client import Astribot
            print("[astribot-image-service] Full Astribot SDK client imported.", flush=True)

            sdk_freq = float(args.control_frequency) if args.enable_control else float(args.sdk_freq)
            if args.enable_control:
                print(
                    "[astribot-image-service] Control enabled; using full Astribot SDK client "
                    f"at freq={sdk_freq:.1f}.",
                    flush=True,
                )
            if args.enable_control and not args.high_control_rights:
                print(
                    "[astribot-image-service] Note: --high-control-rights is not set. "
                    "The SDK may prompt for control rights if another user owns them.",
                    flush=True,
                )
            print("[astribot-image-service] Constructing Astribot SDK client...", flush=True)
            astribot = Astribot(
                freq=sdk_freq,
                high_control_rights=bool(args.high_control_rights),
                node_name=args.node_name,
            )
            _install_stop_signal_handlers()
            print("[astribot-image-service] Astribot SDK client constructed.", flush=True)
        else:
            print(
                "[astribot-image-service] Using camera-only SDK client; whole-body control is not loaded.",
                flush=True,
            )
            astribot = AstribotCameraClient(node_name=args.node_name)
            _install_stop_signal_handlers()

        if not args.no_activate_camera:
            result = astribot.activate_camera()
            print(f"[astribot-image-service] activate_camera result: {result}", flush=True)
        wait_for_cameras(astribot, routes, args.activate_timeout_s)

        for route in routes:
            subscription = astribot.register_image_callback(
                route.camera_name,
                "color",
                bridge.image_callback,
                need_decode=bridge.need_decode,
            )
            subscriptions.append(subscription)
            print(
                f"[astribot-image-service] Registered {route.camera_name} color callback on port {route.port}.",
                flush=True,
            )

        if args.enable_control:
            if args.execute_control:
                print(
                    "\n" + "=" * 72 + "\n"
                    "  !!! EXECUTE CONTROL MODE: incoming TCP actions will move Astribot S1 !!!\n"
                    "  Confirm workspace is clear, terminal control mode is active, and E-stop is ready.\n"
                    + "=" * 72 + "\n",
                    flush=True,
                )
                time.sleep(3.0)

            control_bridge = AstribotControlBridge(
                astribot,
                execute=bool(args.execute_control),
                command_frequency=float(args.control_frequency),
                control_way=args.control_way,
                use_wbc=bool(args.control_use_wbc),
                add_default_torso=bool(args.control_add_default_torso),
                gripper_binary=bool(args.control_gripper_binary),
                gripper_threshold=float(args.control_gripper_threshold),
                control_mode=args.control_mode,
                tracking_arm_velocity=args.tracking_arm_velocity,
                tracking_torso_velocity=args.tracking_torso_velocity,
                tracking_head_velocity=args.tracking_head_velocity,
                gripper_frequency=args.gripper_frequency,
                gripper_velocity=args.gripper_velocity,
                safe_joint_envelope=args.safe_joint_envelope,
                safe_measured_velocity=args.safe_measured_velocity,
                safe_measured_acceleration=args.safe_measured_acceleration,
                safe_tracking_error=args.safe_tracking_error,
                safe_policy_velocity=args.safe_policy_velocity,
                safe_policy_acceleration=args.safe_policy_acceleration,
                safe_command_tracking=args.safe_command_tracking,
                safe_direct_gripper=args.safe_direct_gripper,
                safe_gripper_feedback_tracking=args.safe_gripper_feedback_tracking,
                safe_gripper_limit_retention=args.safe_gripper_limit_retention,
                safe_gripper_output_smoothing_s=args.safe_gripper_output_smoothing_s,
                safe_gripper_smoothing_s=args.safe_gripper_smoothing_s,
                safe_gripper_velocity=args.safe_gripper_velocity,
                safe_gripper_acceleration=args.safe_gripper_acceleration,
            )
            control_bridge.start()
            action_player = BufferedActionPlayerThread(
                control_bridge,
                action_frequency=float(args.control_action_frequency),
                max_buffer_steps=int(args.control_max_buffer_steps),
            )
            action_player.start()
            action_server = AstribotActionServerThread(
                control_bridge,
                action_player,
                host=args.control_bind_host or args.bind_host,
                port=int(args.control_port),
                replan_horizon=int(args.control_replan_horizon),
            )
            action_server.start()
            _install_stop_signal_handlers()

        next_status_time = 0.0
        while not stop_event.is_set():
            now = time.time()
            if args.status_interval_s > 0 and now >= next_status_time:
                bridge.log_status()
                if control_bridge is not None:
                    control_bridge.log_status()
                next_status_time = now + float(args.status_interval_s)
            time.sleep(0.1)
    except KeyboardInterrupt:
        _request_stop(signal.SIGINT, None)
    finally:
        PROCESS_STOPPING.set()
        if action_server is not None:
            action_server.stop()
        if action_player is not None:
            action_player.stop()
        if control_bridge is not None:
            control_bridge.stop()
        bridge.stop()
        signal_requested = stop_signal["signum"] is not None
        if astribot is not None and args.deactivate_on_exit and not signal_requested:
            with contextlib.suppress(Exception):
                astribot.deactivate_camera()
        if astribot is not None and not signal_requested:
            shutdown_astribot_client(astribot)
        elif astribot is not None:
            print(
                "[astribot-image-service] signal stop: skipping SDK shutdown to avoid ROS teardown hang.",
                flush=True,
            )
        print("[astribot-image-service] stopped.", flush=True)
        if signal_requested:
            os._exit(0)


if __name__ == "__main__":
    main()
