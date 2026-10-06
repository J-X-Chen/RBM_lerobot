from __future__ import annotations

import threading
import time
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import logging_mp
except ModuleNotFoundError:
    import logging as logging_mp

from astribot_lerobot.eval_robot.utils.logging_mp_compat import patch_logging_mp

_DLL_DIRECTORY_HANDLES = []


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
    for candidate in (Path.cwd() / "astribot_sdk_ros2", repo_root / "astribot_sdk_ros2"):
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError("Could not find astribot_sdk_ros2. Set ASTRIBOT_SDK_ROOT or pass sdk_root.")


def _prepend_env_path(name: str, paths: list[Path]) -> None:
    existing = [p for p in os.environ.get(name, "").split(os.pathsep) if p]
    new_paths = [str(path) for path in paths if path.exists()]
    if new_paths:
        os.environ[name] = os.pathsep.join([*new_paths, *existing])


def configure_astribot_sdk_path(sdk_root: str | Path | None = None) -> Path:
    root = _find_astribot_sdk_root(sdk_root)
    python_paths = [
        root,
        root / "astribot_msgs" / "local" / "lib" / "python3.10" / "dist-packages",
        root / "third_party" / "software" / "astribot_ros_middleware" / "lib" / "python3.10" / "site-packages",
        root / "third_party" / "astribot_ros_middleware_py",
        root / "third_party" / "drake" / "lib" / "python3.10" / "site-packages",
        root / "third_party" / "third_pkg" / "pinocchio" / "lib" / "python3.10" / "site-packages",
    ]
    for path in reversed([path for path in python_paths if path.exists()]):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    library_paths = [
        root / "astribot_sdk" / "core" / "common" / "robotics_library_py",
        root / "astribot_sdk" / "core" / "common" / "whole_body_control" / "third_party",
        root / "third_party" / "drake" / "lib",
        root / "astribot_msgs" / "lib",
        root / "third_party" / "software" / "astribot_ros_middleware" / "lib",
    ]
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


def get_astribot_client(
    freq: float = 250.0,
    sdk_root: str | Path | None = None,
    high_control_rights: bool = False,
    node_name: str = "eval_astribot_s1",
):
    configure_astribot_sdk_path(sdk_root)
    from astribot_sdk.core.astribot_api.astribot_client import Astribot

    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
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

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = np.asarray(target_q, dtype=np.float64) - current_q
        motion_scale = np.max(np.abs(delta)) / (float(velocity_limit) * self.control_dt)
        return current_q + delta / max(motion_scale, 1.0)

    def ctrl_dual_arm(self, q_target, tauff_target=None):
        q_target = _finite_vector("Astribot S1 whole-body q_target", q_target, WHOLE_BODY_DOF)
        if self.arm_velocity_limit > 0:
            q_target = self.clip_arm_q_target(q_target, velocity_limit=self.arm_velocity_limit)

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
                add_default_torso=False,
            )

        if self._speed_gradual_max and self._gradual_start_time is not None:
            t_elapsed = time.time() - self._gradual_start_time
            self.arm_velocity_limit = 8.0 + (22.0 * min(1.0, t_elapsed / max(float(self._gradual_time), 1e-6)))

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
