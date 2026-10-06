"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import torch
import logging
import json
import select
import sys
import termios
import threading
import tty
from collections.abc import Callable
from queue import Empty, Queue

import cv2
import numpy as np
import pinocchio as pin
from pprint import pformat
from dataclasses import asdict, dataclass
from pathlib import Path
from torch import nn
from contextlib import nullcontext
from typing import Any
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from multiprocessing.sharedctypes import SynchronizedArray
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.utils.dex3_action import expand_dex3_action
from astribot_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from astribot_lerobot.eval_robot.utils.utils import (
    cleanup_resources,
    extract_observation,
    predict_action,
    to_list,
    to_scalar,
    EvalRealConfig,
)
from astribot_lerobot.eval_robot.eval_s1 import (
    RuntimeThermalInputAdapter,
    _policy_dataset_feature_names,
    _selected_camera_feature_names,
    _thermal_input_type,
    configure_policy_for_thermal_input_type,
    ensure_runtime_thermal_input,
    filter_observation_camera_features,
    make_runtime_thermal_input_adapter,
    save_runtime_thermal_input_preview,
    validate_camera_feature_selection,
    validate_thermal_input_selection,
)
from astribot_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data

import logging_mp
from astribot_lerobot.eval_robot.utils.logging_mp_compat import patch_logging_mp

patch_logging_mp(logging_mp)
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


ASYNC_AGGREGATE_FUNCTIONS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


@dataclass
class EvalRealAsyncConfig(EvalRealConfig):
    async_actions_per_chunk: int = 0
    async_chunk_size_threshold: float = 0.5
    async_aggregate_fn: str = "weighted_average"
    async_observation_queue_timeout_s: float = 0.05
    async_action_queue_wait_timeout_s: float = 5.0


@dataclass
class AsyncTimedObservation:
    timestamp: float
    timestep: int
    observation: dict[str, Any]
    must_go: bool = False


@dataclass
class AsyncTimedAction:
    timestamp: float
    timestep: int
    action: torch.Tensor
    observation_timestep: int
    inference_elapsed_s: float


class KeyCommandListener:
    def __init__(self):
        self._keys: list[str] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._old_terminal_settings = None
        self.shutdown_requested = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._old_terminal_settings is not None and sys.stdin.isatty():
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_terminal_settings)
            self._old_terminal_settings = None

    def _listen(self):
        if sys.stdin.isatty():
            self._old_terminal_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())

        while self._running:
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not readable:
                    continue
                key = sys.stdin.read(1).lower()
            except Exception as e:
                logger_mp.warning(f"Keyboard listener stopped: {e}")
                return

            if key:
                self.push(key)

    def push(self, key: str):
        key = key.lower()
        if key not in {"q", "r", "s", "c"}:
            return
        with self._lock:
            if key == "q":
                self.shutdown_requested = True
            self._keys.append(key)

    def consume(self, *allowed_keys: str) -> str | None:
        allowed = {key.lower() for key in allowed_keys}
        with self._lock:
            for idx, key in enumerate(self._keys):
                if key in allowed:
                    return self._keys.pop(idx)
        return None

    def wait_for(self, *allowed_keys: str) -> str:
        while True:
            key = self.consume(*allowed_keys)
            if key is not None:
                return key
            if self.shutdown_requested and "q" in allowed_keys:
                return "q"
            time.sleep(0.05)

    def wait_for_manual_key(self, *allowed_keys: str) -> str:
        while True:
            key = self.consume(*allowed_keys)
            if key is not None:
                return key
            time.sleep(0.05)


def parse_arm_pose(pose_text: str, arm_dof: int) -> np.ndarray | None:
    if not pose_text.strip():
        return None
    values = [float(item) for item in pose_text.replace(",", " ").split()]
    if len(values) != arm_dof:
        raise ValueError(f"Expected {arm_dof} arm pose values, got {len(values)} from '{pose_text}'.")
    return np.asarray(values, dtype=np.float64)


def _lift_current_arm_pose(arm_ik, current_arm_q: np.ndarray, lift_m: float) -> np.ndarray:
    fk_data = arm_ik.reduced_robot.model.createData()
    pin.framesForwardKinematics(arm_ik.reduced_robot.model, fk_data, current_arm_q)
    left_current = fk_data.oMf[arm_ik.L_hand_id]
    right_current = fk_data.oMf[arm_ik.R_hand_id]
    left_target = pin.SE3(left_current.rotation.copy(), left_current.translation.copy())
    right_target = pin.SE3(right_current.rotation.copy(), right_current.translation.copy())
    left_target.translation[2] += lift_m
    right_target.translation[2] += lift_m

    target_arm_q, _ = arm_ik.solve_ik(
        left_target.homogeneous,
        right_target.homogeneous,
        current_arm_q,
        np.zeros_like(current_arm_q),
    )
    target_arm_q = np.asarray(target_arm_q, dtype=np.float64)
    if target_arm_q.shape != current_arm_q.shape or not np.all(np.isfinite(target_arm_q)):
        raise RuntimeError("Ready lift IK returned an invalid arm pose.")
    return target_arm_q


def _smoothstep(alpha: float) -> float:
    alpha = max(0.0, min(1.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def get_ready_arm_pose(cfg: EvalRealConfig, arm_ik, current_arm_q: np.ndarray) -> np.ndarray:
    arm_dof = len(current_arm_q)
    pose = parse_arm_pose(cfg.ready_arm_pose, arm_dof)
    if pose is not None:
        return pose

    if cfg.ready_lift_m > 0:
        pose = _lift_current_arm_pose(arm_ik, current_arm_q, float(cfg.ready_lift_m))
        delta = np.abs(pose - current_arm_q)
        max_delta = float(np.max(delta)) if delta.size else 0.0
        max_delta_joint = int(np.argmax(delta)) if delta.size else -1
        logger_mp.info(
            f"Computed lifted ready pose by raising wrists {cfg.ready_lift_m:.3f} m; "
            f"max joint delta is {max_delta:.4f} rad on joint_{max_delta_joint}."
        )
        if cfg.ready_max_joint_delta > 0 and max_delta > cfg.ready_max_joint_delta:
            raise RuntimeError(
                "Ready lift IK rejected for safety: "
                f"max joint delta {max_delta:.4f} rad exceeds ready_max_joint_delta "
                f"{cfg.ready_max_joint_delta:.4f} rad."
            )
        return pose

    logger_mp.info("No ready_arm_pose provided; using the zero arm pose as ready pose.")
    return np.zeros(arm_dof, dtype=np.float64)


def hold_current_arm_pose(arm_ctrl, arm_ik):
    current_arm_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=np.float64)
    tau = np.zeros_like(current_arm_q)
    arm_ctrl.ctrl_dual_arm(current_arm_q, tau)
    return current_arm_q


def _json_array(values, precision: int = 6) -> list[float]:
    return np.round(np.asarray(values, dtype=np.float64), precision).tolist()


def _max_abs(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def _argmax_abs(values) -> int | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return int(np.argmax(np.abs(values)))


def open_action_log(cfg: EvalRealConfig):
    path_text = str(cfg.action_log_path).strip()
    if not path_text or path_text.lower() in {"false", "none", "off", "0"}:
        return None, None

    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.suffix != ".jsonl":
        path = path / f"eval_s1_actions_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    action_log_file = path.open("w", encoding="utf-8")
    logger_mp.info(f"Writing per-step action log to {path}")
    return action_log_file, path


def write_action_log(action_log_file, record: dict[str, Any]) -> None:
    if action_log_file is None:
        return
    action_log_file.write(json.dumps(record, separators=(",", ":")) + "\n")
    action_log_file.flush()


def save_head_camera_snapshot(tv_img_array, label: str = "debug_head_camera_on_r.png") -> Path | None:
    image = np.asarray(tv_img_array.copy())
    if image.size == 0:
        logger_mp.warning("Head camera snapshot skipped: empty image buffer.")
        return None
    if image.ndim != 3 or image.shape[2] != 3:
        logger_mp.warning(f"Head camera snapshot skipped: unexpected image shape {image.shape}.")
        return None

    path = Path.cwd() / label
    ok = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    mean_pixel = float(np.mean(image))
    if not ok:
        logger_mp.warning(f"Failed to save head camera snapshot to {path}.")
        return None
    if mean_pixel < 1.0:
        logger_mp.warning(
            f"Saved head camera snapshot to {path}, but mean pixel value is {mean_pixel:.2f}; "
            "the frame may be blank."
        )
    else:
        logger_mp.info(f"Saved head camera snapshot to {path} with shape {image.shape}.")
    return path


def get_async_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if name not in ASYNC_AGGREGATE_FUNCTIONS:
        available = ", ".join(sorted(ASYNC_AGGREGATE_FUNCTIONS))
        raise ValueError(f"Unknown async_aggregate_fn '{name}'. Available: {available}.")
    return ASYNC_AGGREGATE_FUNCTIONS[name]


def resolve_async_actions_per_chunk(cfg: EvalRealAsyncConfig, policy: PreTrainedPolicy) -> int:
    if int(cfg.async_actions_per_chunk) > 0:
        return int(cfg.async_actions_per_chunk)

    policy_cfg = getattr(policy, "config", None)
    for attr in ("n_action_steps", "action_chunk_size", "chunk_size", "horizon"):
        value = getattr(policy_cfg, attr, None)
        if value is not None and int(value) > 0:
            return int(value)
    return 1


def clone_observation_for_async(observation: dict[str, Any]) -> dict[str, Any]:
    cloned = {}
    for key, value in observation.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        elif isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        else:
            cloned[key] = value
    return cloned


def collect_real_observation(
    cfg: EvalRealConfig,
    arm_ctrl,
    ee_shared_mem: dict[str, Any],
    arm_dof: int,
    ee_dof: int,
    tv_img_array,
    wrist_img_array,
    thermal_img_array,
    tv_img_shape,
    wrist_img_shape,
    thermal_img_shape,
    is_binocular,
    has_wrist_cam,
    has_thermal_cam,
    runtime_thermal_input_adapter: RuntimeThermalInputAdapter | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    observation, current_arm_q = process_images_and_observations(
        tv_img_array,
        wrist_img_array,
        tv_img_shape,
        wrist_img_shape,
        is_binocular,
        has_wrist_cam,
        arm_ctrl,
        thermal_img_array=thermal_img_array,
        thermal_img_shape=thermal_img_shape,
        has_thermal_cam=has_thermal_cam,
    )
    observation = filter_observation_camera_features(cfg, observation)
    observation = ensure_runtime_thermal_input(cfg, observation, runtime_thermal_input_adapter)
    current_arm_q = np.asarray(current_arm_q, dtype=np.float64)

    left_ee_state = right_ee_state = np.array([], dtype=np.float64)
    if cfg.ee:
        with ee_shared_mem["lock"]:
            full_state = np.array(ee_shared_mem["state"][:], dtype=np.float64)
            left_ee_state = full_state[:ee_dof]
            right_ee_state = full_state[ee_dof:]

    state_tensor = torch.from_numpy(np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)).float()
    observation["observation.state"] = state_tensor
    return observation, current_arm_q, left_ee_state, right_ee_state, state_tensor


def prepare_observation_for_policy(
    observation: dict[str, Any],
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    task: str | None,
    use_dataset: bool = False,
    robot_type: str | None = None,
) -> dict[str, Any]:
    observation = dict(observation)
    for name, value in list(observation.items()):
        if not use_dataset:
            if not hasattr(value, "unsqueeze"):
                continue
            if "images" in name:
                value = value.type(torch.float32) / 255
                value = value.permute(2, 0, 1).contiguous()

        observation[name] = value.unsqueeze(0).to(device)

    observation["task"] = task if task else ""
    observation["robot_type"] = robot_type if robot_type else ""
    return preprocessor(observation)


def _normalise_action_chunk(action_chunk: torch.Tensor, actions_per_chunk: int) -> torch.Tensor:
    if action_chunk.ndim == 1:
        action_chunk = action_chunk.view(1, 1, -1)
    elif action_chunk.ndim == 2:
        action_chunk = action_chunk.unsqueeze(0)
    elif action_chunk.ndim == 3 and action_chunk.shape[0] != 1 and action_chunk.shape[1] == 1:
        action_chunk = action_chunk.transpose(0, 1)
    elif action_chunk.ndim != 3:
        raise RuntimeError(f"Policy returned action chunk with unsupported shape {tuple(action_chunk.shape)}.")

    if action_chunk.shape[0] != 1:
        raise RuntimeError(f"Expected a single-observation action chunk, got shape {tuple(action_chunk.shape)}.")
    if action_chunk.shape[1] <= 0:
        raise RuntimeError("Policy returned an empty action chunk.")

    return action_chunk[:, :actions_per_chunk, :]


def _select_action_chunk_fallback(
    policy: PreTrainedPolicy,
    observation: dict[str, Any],
    actions_per_chunk: int,
) -> torch.Tensor:
    actions = []
    for _ in range(actions_per_chunk):
        action = policy.select_action(observation)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        actions.append(action)
    return torch.stack(actions, dim=1)


_predict_chunk_fallback_warned = False


def predict_action_chunk_async(
    observation: dict[str, Any],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str,
    actions_per_chunk: int,
    use_dataset: bool = False,
    robot_type: str | None = None,
) -> list[torch.Tensor]:
    global _predict_chunk_fallback_warned

    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        policy_observation = prepare_observation_for_policy(
            observation,
            device,
            preprocessor,
            task,
            use_dataset=use_dataset,
            robot_type=robot_type,
        )

        try:
            action_chunk = policy.predict_action_chunk(policy_observation)
        except NotImplementedError:
            if not _predict_chunk_fallback_warned:
                logger_mp.warning("Policy does not expose predict_action_chunk; falling back to select_action.")
                _predict_chunk_fallback_warned = True
            action_chunk = _select_action_chunk_fallback(policy, policy_observation, actions_per_chunk)
        except RuntimeError as exc:
            if "stack expects a non-empty" not in str(exc):
                raise
            if not _predict_chunk_fallback_warned:
                logger_mp.warning(
                    "Policy predict_action_chunk needs an internal observation history; "
                    "falling back to select_action chunk extraction."
                )
                _predict_chunk_fallback_warned = True
            action_chunk = _select_action_chunk_fallback(policy, policy_observation, actions_per_chunk)

        action_chunk = _normalise_action_chunk(action_chunk, actions_per_chunk)
        _, chunk_size, _ = action_chunk.shape
        processed_actions = []
        for chunk_idx in range(chunk_size):
            processed_action = postprocessor(action_chunk[:, chunk_idx, :])
            processed_actions.append(processed_action.squeeze(0).to("cpu"))

    return processed_actions


class AsyncActionQueue:
    def __init__(
        self,
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        action_chunk_size: int,
    ):
        self._aggregate_fn = aggregate_fn
        self._actions: dict[int, AsyncTimedAction] = {}
        self._lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = max(1, int(action_chunk_size))
        self.queue_size_history: list[int] = []

    def _drop_stale_locked(self) -> None:
        for timestep in [t for t in self._actions if t <= self.latest_action]:
            del self._actions[timestep]

    def merge(self, incoming_actions: list[AsyncTimedAction]) -> tuple[int, int]:
        accepted = 0
        with self._lock:
            self.action_chunk_size = max(self.action_chunk_size, len(incoming_actions), 1)
            for incoming in incoming_actions:
                if incoming.timestep <= self.latest_action:
                    continue
                existing = self._actions.get(incoming.timestep)
                if existing is None:
                    self._actions[incoming.timestep] = incoming
                else:
                    self._actions[incoming.timestep] = AsyncTimedAction(
                        timestamp=incoming.timestamp,
                        timestep=incoming.timestep,
                        action=self._aggregate_fn(existing.action, incoming.action),
                        observation_timestep=incoming.observation_timestep,
                        inference_elapsed_s=incoming.inference_elapsed_s,
                    )
                accepted += 1
            self._drop_stale_locked()
            queue_size = len(self._actions)
        return accepted, queue_size

    def pop_next(self) -> AsyncTimedAction | None:
        with self._lock:
            self._drop_stale_locked()
            self.queue_size_history.append(len(self._actions))
            if not self._actions:
                return None
            next_timestep = min(self._actions)
            action = self._actions.pop(next_timestep)
            self.latest_action = next_timestep
            self._drop_stale_locked()
            return action

    def qsize(self) -> int:
        with self._lock:
            self._drop_stale_locked()
            return len(self._actions)

    def latest_timestep_for_observation(self) -> int:
        with self._lock:
            return max(self.latest_action, 0)

    def ready_to_send_observation(self, threshold: float) -> bool:
        threshold = max(0.0, min(1.0, float(threshold)))
        with self._lock:
            self._drop_stale_locked()
            return len(self._actions) / max(self.action_chunk_size, 1) <= threshold


def enqueue_latest_observation(observation_queue: Queue, observation: AsyncTimedObservation) -> bool:
    try:
        while True:
            try:
                observation_queue.get_nowait()
            except Empty:
                break
        observation_queue.put_nowait(observation)
        return True
    except Exception as exc:
        logger_mp.warning(f"Failed to enqueue observation for async inference: {exc}")
        return False


def execute_robot_action(
    cfg: EvalRealConfig,
    action: torch.Tensor,
    current_arm_q: np.ndarray,
    left_ee_state: np.ndarray,
    right_ee_state: np.ndarray,
    arm_dof: int,
    ee_dof: int,
    arm_ctrl,
    arm_ik,
    ee_shared_mem: dict[str, Any],
    prev_sent_action_np: np.ndarray | None,
) -> dict[str, Any]:
    raw_action_np = action.detach().cpu().numpy().reshape(-1)
    expected_dim = arm_dof + (2 * ee_dof if cfg.ee else 0)
    compressed_dex3_dim = arm_dof + 2
    if cfg.ee == "dex3" and ee_dof == 7 and raw_action_np.shape[0] == compressed_dex3_dim:
        raw_action_np = expand_dex3_action(raw_action_np, arm_dof=arm_dof)
    if raw_action_np.shape[0] != expected_dim:
        raise RuntimeError(f"Policy action dim {raw_action_np.shape[0]} does not match expected dim {expected_dim}.")
    if not np.all(np.isfinite(raw_action_np)):
        raise RuntimeError("Policy action contains NaN or Inf; refusing to send it to the robot.")

    current_arm_q = np.asarray(current_arm_q, dtype=np.float64)
    if current_arm_q.shape[0] != arm_dof or not np.all(np.isfinite(current_arm_q)):
        raise RuntimeError("Current arm state is invalid; refusing to send a policy action.")
    action_np = raw_action_np.copy()
    raw_arm_action = raw_action_np[:arm_dof].copy()
    raw_arm_delta = raw_arm_action - current_arm_q
    arm_action = raw_arm_action.copy()
    if cfg.max_policy_arm_delta > 0:
        arm_delta_limit = float(cfg.max_policy_arm_delta)
        arm_action = current_arm_q + np.clip(arm_action - current_arm_q, -arm_delta_limit, arm_delta_limit)
        if (
            cfg.limit_policy_delta_from_last_command
            and prev_sent_action_np is not None
            and prev_sent_action_np.shape[0] >= arm_dof
        ):
            prev_arm_action = prev_sent_action_np[:arm_dof]
            arm_action = prev_arm_action + np.clip(
                arm_action - prev_arm_action,
                -arm_delta_limit,
                arm_delta_limit,
            )
            arm_action = current_arm_q + np.clip(arm_action - current_arm_q, -arm_delta_limit, arm_delta_limit)
        action_np[:arm_dof] = arm_action

    tau = np.asarray(arm_ik.solve_tau(arm_action), dtype=np.float64)
    if tau.shape != arm_action.shape or not np.all(np.isfinite(tau)):
        raise RuntimeError("Gravity compensation tau is invalid; refusing to send a policy action.")
    arm_ctrl.ctrl_dual_arm(arm_action, tau)

    raw_left_ee_action = raw_right_ee_action = np.array([], dtype=np.float64)
    left_ee_action = right_ee_action = np.array([], dtype=np.float64)
    raw_left_delta = raw_right_delta = np.array([], dtype=np.float64)
    sent_left_delta = sent_right_delta = np.array([], dtype=np.float64)
    if cfg.ee:
        ee_action_start_idx = arm_dof
        raw_left_ee_action = raw_action_np[ee_action_start_idx : ee_action_start_idx + ee_dof].copy()
        raw_right_ee_action = raw_action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof].copy()
        left_ee_action = raw_left_ee_action.copy()
        right_ee_action = raw_right_ee_action.copy()
        if cfg.max_policy_ee_delta > 0:
            ee_delta_limit = float(cfg.max_policy_ee_delta)
            left_ee_action = left_ee_state + np.clip(left_ee_action - left_ee_state, -ee_delta_limit, ee_delta_limit)
            right_ee_action = right_ee_state + np.clip(
                right_ee_action - right_ee_state,
                -ee_delta_limit,
                ee_delta_limit,
            )
            if (
                cfg.limit_policy_delta_from_last_command
                and prev_sent_action_np is not None
                and prev_sent_action_np.shape[0] >= ee_action_start_idx + 2 * ee_dof
            ):
                prev_left_ee_action = prev_sent_action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                prev_right_ee_action = prev_sent_action_np[
                    ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof
                ]
                left_ee_action = prev_left_ee_action + np.clip(
                    left_ee_action - prev_left_ee_action,
                    -ee_delta_limit,
                    ee_delta_limit,
                )
                right_ee_action = prev_right_ee_action + np.clip(
                    right_ee_action - prev_right_ee_action,
                    -ee_delta_limit,
                    ee_delta_limit,
                )
                left_ee_action = left_ee_state + np.clip(
                    left_ee_action - left_ee_state,
                    -ee_delta_limit,
                    ee_delta_limit,
                )
                right_ee_action = right_ee_state + np.clip(
                    right_ee_action - right_ee_state,
                    -ee_delta_limit,
                    ee_delta_limit,
                )

            action_np[ee_action_start_idx : ee_action_start_idx + ee_dof] = left_ee_action
            action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof] = right_ee_action

        if isinstance(ee_shared_mem["left"], SynchronizedArray):
            ee_shared_mem["left"][:] = to_list(left_ee_action)
            ee_shared_mem["right"][:] = to_list(right_ee_action)
        elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
            ee_shared_mem["left"].value = to_scalar(left_ee_action)
            ee_shared_mem["right"].value = to_scalar(right_ee_action)

        raw_left_delta = raw_left_ee_action - left_ee_state
        raw_right_delta = raw_right_ee_action - right_ee_state
        sent_left_delta = left_ee_action - left_ee_state
        sent_right_delta = right_ee_action - right_ee_state

    return {
        "raw_action_np": raw_action_np,
        "action_np": action_np,
        "raw_arm_action": raw_arm_action,
        "arm_action": arm_action,
        "tau": tau,
        "raw_arm_delta": raw_arm_delta,
        "clipped_arm_delta": arm_action - current_arm_q,
        "raw_left_ee_action": raw_left_ee_action,
        "raw_right_ee_action": raw_right_ee_action,
        "left_ee_action": left_ee_action,
        "right_ee_action": right_ee_action,
        "raw_left_delta": raw_left_delta,
        "raw_right_delta": raw_right_delta,
        "sent_left_delta": sent_left_delta,
        "sent_right_delta": sent_right_delta,
    }


def _policy_feature_dim(features: dict[str, Any] | None, name: str) -> int | None:
    if not features or name not in features:
        return None
    feature = features[name]
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, dict):
        shape = feature.get("shape")
    if not shape:
        return None
    return int(shape[-1])


def compute_slow_move_tau(cfg: EvalRealConfig, arm_ik, arm_q: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    arm_q = np.asarray(arm_q, dtype=np.float64)
    if cfg.slow_move_tau_scale <= 0:
        return np.zeros_like(arm_q)

    try:
        tau = np.asarray(arm_ik.solve_tau(arm_q), dtype=np.float64)
    except Exception as exc:
        logger_mp.warning(f"Slow move gravity compensation failed; using zero tau. Error: {exc}")
        return np.zeros_like(arm_q)

    if tau.shape != arm_q.shape or not np.all(np.isfinite(tau)):
        logger_mp.warning("Slow move gravity compensation returned invalid tau; using zero tau.")
        return np.zeros_like(arm_q)

    tau = tau * float(cfg.slow_move_tau_scale) * max(0.0, min(1.0, float(alpha)))
    if cfg.slow_move_tau_limit > 0:
        tau = np.clip(tau, -float(cfg.slow_move_tau_limit), float(cfg.slow_move_tau_limit))
    return tau


def move_arm_to_pose_slowly(
    cfg: EvalRealConfig,
    arm_ctrl,
    arm_ik,
    target_arm_q,
    label: str,
    key_listener: KeyCommandListener | None = None,
    allow_q_interrupt: bool = True,
    trajectory_velocity_limit: float | None = None,
    wait_for_clearance_on_block: bool = False,
) -> bool:
    target_arm_q = np.asarray(target_arm_q, dtype=np.float64)
    current_arm_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=np.float64)
    delta = target_arm_q - current_arm_q
    max_delta = float(np.max(np.abs(delta)))
    max_delta_joint = int(np.argmax(np.abs(delta)))

    velocity_limit = (
        float(trajectory_velocity_limit)
        if trajectory_velocity_limit is not None and trajectory_velocity_limit > 0
        else float(cfg.arm_velocity_limit)
    )
    logger_mp.info(
        f"{label}: current->target max joint delta is {max_delta:.4f} rad "
        f"on joint_{max_delta_joint}; trajectory velocity limit is {velocity_limit:.4f} rad/s."
    )

    tau = np.zeros_like(target_arm_q)
    start_time = time.perf_counter()
    next_log_time = start_time
    motion_frequency = max(float(cfg.slow_move_frequency), 1.0)
    trajectory_time = max_delta / max(velocity_limit, 1e-6) if max_delta > 0 else 0.0
    steps = max(1, int(np.ceil(trajectory_time * motion_frequency)))
    start_arm_q = current_arm_q.copy()
    last_arm_cmd_q = current_arm_q.copy()
    last_tau = np.zeros_like(target_arm_q)
    tracking_pause_ratio = max(0.0, min(1.0, float(cfg.slow_move_tracking_pause_ratio)))
    if wait_for_clearance_on_block:
        tracking_pause_ratio = max(0.0, min(1.0, float(cfg.lower_block_tracking_ratio)))
    tracking_pause_limit = (
        cfg.arm_tracking_error_limit * tracking_pause_ratio
        if cfg.arm_tracking_error_limit > 0
        else float("inf")
    )
    tracking_wait_timeout_s = (
        float(cfg.lower_block_wait_timeout_s)
        if wait_for_clearance_on_block
        else float(cfg.slow_move_tracking_wait_timeout_s)
    )

    logger_mp.info(
        f"Easing to {label} over nominal {trajectory_time:.1f}s ({steps} steps); "
        f"slow move frequency is {motion_frequency:.1f} Hz; "
        f"tracking pause limit is {tracking_pause_limit:.4f} rad; "
        f"tracking wait timeout is {tracking_wait_timeout_s:.1f}s."
    )
    for step_idx in range(1, steps + 1):
        if allow_q_interrupt and key_listener is not None and key_listener.shutdown_requested:
            logger_mp.info(f"Movement to {label} interrupted by q.")
            return False

        tracking_wait_started_at = None
        while step_idx > 1:
            current_arm_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=np.float64)
            tracking_err = np.abs(last_arm_cmd_q - current_arm_q)
            max_tracking_err = float(np.max(tracking_err))
            max_tracking_joint = int(np.argmax(tracking_err))

            if cfg.arm_tracking_error_limit > 0 and max_tracking_err > cfg.arm_tracking_error_limit:
                logger_mp.info(
                    f"Stopping {label}: tracking error {max_tracking_err:.4f} rad on joint_{max_tracking_joint} "
                    f"exceeds arm_tracking_error_limit {cfg.arm_tracking_error_limit:.4f} rad."
                )
                hold_current_arm_pose(arm_ctrl, arm_ik)
                return False

            if max_tracking_err <= tracking_pause_limit:
                break

            now = time.perf_counter()
            if tracking_wait_started_at is None:
                tracking_wait_started_at = now
            elif (
                tracking_wait_timeout_s > 0
                and now - tracking_wait_started_at > tracking_wait_timeout_s
            ):
                if wait_for_clearance_on_block and key_listener is not None and cfg.lower_block_wait_for_continue:
                    current_arm_q = hold_current_arm_pose(arm_ctrl, arm_ik)
                    logger_mp.info(
                        f"Possible obstacle while moving to {label}: tracking stayed above "
                        f"{tracking_pause_limit:.4f} rad for {tracking_wait_timeout_s:.1f}s. "
                        "Holding current pose. Clear the table/obstacle, then press 'c' to continue lowering."
                    )
                    key_listener.wait_for_manual_key("c")
                    logger_mp.info(f"Continue requested; resuming movement to {label} from current pose.")
                    return move_arm_to_pose_slowly(
                        cfg,
                        arm_ctrl,
                        arm_ik,
                        target_arm_q,
                        label,
                        key_listener=key_listener,
                        allow_q_interrupt=False,
                        trajectory_velocity_limit=trajectory_velocity_limit,
                        wait_for_clearance_on_block=wait_for_clearance_on_block,
                    )

                logger_mp.info(
                    f"Stopping {label}: tracking stayed above pause limit "
                    f"{tracking_pause_limit:.4f} rad for {tracking_wait_timeout_s:.1f}s."
                )
                hold_current_arm_pose(arm_ctrl, arm_ik)
                return False

            elapsed = now - start_time
            if now >= next_log_time:
                err = np.abs(target_arm_q - current_arm_q)
                max_err = float(np.max(err))
                logger_mp.info(
                    f"Waiting for {label} tracking: max target error {max_err:.4f} rad, "
                    f"max tracking error {max_tracking_err:.4f} rad on joint_{max_tracking_joint} after {elapsed:.1f}s."
                )
                next_log_time = now + 2.0

            arm_ctrl.ctrl_dual_arm(last_arm_cmd_q, last_tau)
            time.sleep(1.0 / motion_frequency)

            if allow_q_interrupt and key_listener is not None and key_listener.shutdown_requested:
                logger_mp.info(f"Movement to {label} interrupted by q.")
                return False

        alpha = _smoothstep(step_idx / steps)
        arm_cmd_q = start_arm_q + delta * alpha
        tau = compute_slow_move_tau(cfg, arm_ik, arm_cmd_q, alpha)
        arm_ctrl.ctrl_dual_arm(arm_cmd_q, tau)
        last_arm_cmd_q = arm_cmd_q.copy()
        last_tau = tau.copy()

        current_arm_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=np.float64)
        tracking_err = np.abs(arm_cmd_q - current_arm_q)
        max_tracking_err = float(np.max(tracking_err))
        max_tracking_joint = int(np.argmax(tracking_err))
        if cfg.arm_tracking_error_limit > 0 and max_tracking_err > cfg.arm_tracking_error_limit:
            logger_mp.info(
                f"Stopping {label}: tracking error {max_tracking_err:.4f} rad on joint_{max_tracking_joint} "
                f"exceeds arm_tracking_error_limit {cfg.arm_tracking_error_limit:.4f} rad."
            )
            hold_current_arm_pose(arm_ctrl, arm_ik)
            return False

        err = np.abs(target_arm_q - current_arm_q)
        max_err = float(np.max(err))

        now = time.perf_counter()
        elapsed = now - start_time
        if now >= next_log_time:
            logger_mp.info(
                f"Moving to {label}: max target error {max_err:.4f} rad, "
                f"max tracking error {max_tracking_err:.4f} rad after {elapsed:.1f}s."
            )
            next_log_time = now + 2.0

        time.sleep(1.0 / motion_frequency)

    tau = compute_slow_move_tau(cfg, arm_ik, target_arm_q, 1.0)
    arm_ctrl.ctrl_dual_arm(target_arm_q, tau)
    while True:
        if allow_q_interrupt and key_listener is not None and key_listener.shutdown_requested:
            logger_mp.info(f"Movement to {label} interrupted by q.")
            return False

        current_arm_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=np.float64)
        err = np.abs(target_arm_q - current_arm_q)
        max_err = float(np.max(err))
        if max_err <= cfg.init_tolerance:
            logger_mp.info(f"Reached {label}; max joint error {max_err:.4f} rad.")
            return True

        now = time.perf_counter()
        elapsed = now - start_time
        if cfg.init_timeout_s > 0 and elapsed > cfg.init_timeout_s:
            logger_mp.info(
                f"Timed out while moving to {label}; max joint error is still {max_err:.4f} rad. "
                "Evaluation will not start."
            )
            return False
        if now >= next_log_time:
            logger_mp.info(f"Settling at {label}: max joint error {max_err:.4f} rad after {elapsed:.1f}s.")
            next_log_time = now + 2.0

        time.sleep(0.05)


def move_ee_to_pose_slowly(
    cfg: EvalRealConfig,
    ee_shared_mem: dict[str, Any],
    ee_dof: int,
    target_ee_state: np.ndarray,
    label: str,
    key_listener: KeyCommandListener | None = None,
    velocity_limit: float | None = None,
) -> bool:
    if ee_dof <= 0 or not target_ee_state.size:
        return True

    target_ee_state = np.asarray(target_ee_state, dtype=np.float64)
    if target_ee_state.shape[0] != 2 * ee_dof:
        raise ValueError(f"{label} target ee state length {target_ee_state.shape[0]} != {2 * ee_dof}.")

    with ee_shared_mem["lock"]:
        current_ee_state = np.array(ee_shared_mem["state"][:], dtype=np.float64)
    if current_ee_state.shape[0] != 2 * ee_dof:
        logger_mp.warning(f"Skipping {label}: current ee state length {current_ee_state.shape[0]} != {2 * ee_dof}.")
        return True

    delta = target_ee_state - current_ee_state
    max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
    velocity_limit = max(float(cfg.init_ee_velocity_limit if velocity_limit is None else velocity_limit), 1e-6)
    motion_frequency = max(float(cfg.slow_move_frequency), 1.0)
    trajectory_time = max_delta / velocity_limit if max_delta > 0 else 0.0
    steps = max(1, int(np.ceil(trajectory_time * motion_frequency)))
    logger_mp.info(
        f"{label}: current->target max finger delta is {max_delta:.4f} rad; "
        f"moving over nominal {trajectory_time:.1f}s ({steps} steps)."
    )

    for step_idx in range(1, steps + 1):
        if key_listener is not None and key_listener.shutdown_requested:
            logger_mp.info(f"Movement to {label} interrupted by q.")
            return False

        alpha = _smoothstep(step_idx / steps)
        ee_cmd = current_ee_state + delta * alpha
        left_cmd = ee_cmd[:ee_dof]
        right_cmd = ee_cmd[ee_dof:]
        if isinstance(ee_shared_mem["left"], SynchronizedArray):
            ee_shared_mem["left"][:] = to_list(left_cmd)
            ee_shared_mem["right"][:] = to_list(right_cmd)
        elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
            ee_shared_mem["left"].value = to_scalar(left_cmd)
            ee_shared_mem["right"].value = to_scalar(right_cmd)

        time.sleep(1.0 / motion_frequency)

    with ee_shared_mem["lock"]:
        final_ee_state = np.array(ee_shared_mem["state"][:], dtype=np.float64)
    final_error = target_ee_state - final_ee_state
    max_final_error = float(np.max(np.abs(final_error))) if final_error.size else 0.0
    if max_final_error > float(cfg.ee_tracking_tolerance):
        logger_mp.warning(
            f"{label}: final finger tracking error {max_final_error:.4f} rad exceeds "
            f"ee_tracking_tolerance {cfg.ee_tracking_tolerance:.4f} rad; "
            "hand command may not be reaching the controller."
        )
        return False

    logger_mp.info(f"Reached {label}; max finger error {max_final_error:.4f} rad.")
    return True


def run_shutdown_sequence(
    cfg: EvalRealConfig,
    arm_ctrl,
    arm_ik,
    ready_arm_pose: np.ndarray,
    lowered_arm_pose: np.ndarray,
    key_listener: KeyCommandListener | None = None,
    ee_shared_mem: dict[str, Any] | None = None,
    ee_dof: int = 0,
) -> bool:
    if cfg.open_ee_on_quit and ee_shared_mem is not None and ee_dof > 0:
        logger_mp.info("Shutdown sequence: opening hands before arm retreat.")
        open_ok = move_ee_to_pose_slowly(
            cfg,
            ee_shared_mem,
            ee_dof,
            np.zeros(2 * ee_dof, dtype=np.float64),
            "shutdown open hands",
            key_listener=None,
            velocity_limit=cfg.shutdown_ee_open_velocity_limit,
        )
        if not open_ok:
            logger_mp.info("Shutdown hand opening did not complete; continuing arm shutdown for safety.")

    logger_mp.info("Shutdown sequence: moving back to ready arm pose before lowering.")
    ready_ok = move_arm_to_pose_slowly(
        cfg,
        arm_ctrl,
        arm_ik,
        ready_arm_pose,
        "ready arm pose",
        key_listener=None,
        allow_q_interrupt=False,
        trajectory_velocity_limit=cfg.ready_arm_velocity_limit,
    )
    if not ready_ok:
        return False

    if cfg.lower_to_startup_on_quit:
        logger_mp.info("Shutdown sequence: lowering back to startup arm pose.")
        return move_arm_to_pose_slowly(
            cfg,
            arm_ctrl,
            arm_ik,
            lowered_arm_pose,
            "startup/lowered arm pose",
            key_listener=key_listener,
            allow_q_interrupt=False,
            wait_for_clearance_on_block=True,
        )

    return True


def warmup_policy_from_dataset(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    device: torch.device,
    runtime_thermal_input_adapter: RuntimeThermalInputAdapter | None = None,
) -> None:
    warmup_steps = int(cfg.policy_warmup_steps)
    if warmup_steps <= 0:
        return

    from_idx = dataset.meta.episodes["dataset_from_index"][0]
    step = dataset[from_idx]
    observation = extract_observation(step)
    observation = filter_observation_camera_features(cfg, observation)
    observation = ensure_runtime_thermal_input(cfg, observation, runtime_thermal_input_adapter)
    task = step["task"]

    logger_mp.info(f"Warming up policy for {warmup_steps} step(s) before connecting to robot.")
    start_time = time.perf_counter()
    for _ in range(warmup_steps):
        _ = predict_action(
            observation,
            policy,
            device,
            preprocessor,
            postprocessor,
            policy.config.use_amp,
            task,
            use_dataset=True,
            robot_type=None,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    logger_mp.info(f"Policy warmup finished in {elapsed:.2f}s.")

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()


def warmup_policy_from_real_observation(
    cfg: EvalRealConfig,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    device: torch.device,
    task: str,
    arm_ctrl,
    ee_shared_mem: dict[str, Any],
    arm_dof: int,
    ee_dof: int,
    tv_img_array,
    wrist_img_array,
    thermal_img_array,
    tv_img_shape,
    wrist_img_shape,
    thermal_img_shape,
    is_binocular,
    has_wrist_cam,
    has_thermal_cam,
    runtime_thermal_input_adapter: RuntimeThermalInputAdapter | None = None,
) -> None:
    warmup_steps = int(cfg.real_policy_warmup_steps)
    if warmup_steps <= 0:
        return

    logger_mp.info(f"Warming up policy for {warmup_steps} real-observation step(s); actions will be discarded.")
    start_time = time.perf_counter()
    for warmup_idx in range(warmup_steps):
        observation, current_arm_q = process_images_and_observations(
            tv_img_array,
            wrist_img_array,
            tv_img_shape,
            wrist_img_shape,
            is_binocular,
            has_wrist_cam,
            arm_ctrl,
            thermal_img_array=thermal_img_array,
            thermal_img_shape=thermal_img_shape,
            has_thermal_cam=has_thermal_cam,
        )
        observation = filter_observation_camera_features(cfg, observation)
        observation = ensure_runtime_thermal_input(cfg, observation, runtime_thermal_input_adapter)
        left_ee_state = right_ee_state = np.array([])
        if cfg.ee:
            with ee_shared_mem["lock"]:
                full_state = np.array(ee_shared_mem["state"][:])
                left_ee_state = full_state[:ee_dof]
                right_ee_state = full_state[ee_dof:]
        observation["observation.state"] = torch.from_numpy(
            np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
        ).float()

        _ = predict_action(
            observation,
            policy,
            device,
            preprocessor,
            postprocessor,
            policy.config.use_amp,
            task,
            use_dataset=cfg.use_dataset,
            robot_type=None,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    logger_mp.info(f"Real-observation policy warmup finished in {elapsed:.2f}s.")

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()


def eval_policy(
    cfg: EvalRealAsyncConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    runtime_thermal_input_adapter: RuntimeThermalInputAdapter | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")
    if not cfg.send_real_robot:
        logger_mp.warning("--send_real_robot=false, skipping image client and robot interface setup.")
        return

    validate_camera_feature_selection(cfg)
    logger_mp.info(
        f"Selected camera features for async real eval: {_selected_camera_feature_names(cfg)}; "
        f"thermal_input_type={_thermal_input_type(cfg)!r}."
    )

    if cfg.visualization:
        rerun_logger = RerunLogger()

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    image_info = None
    key_listener = KeyCommandListener()
    arm_ctrl = arm_ik = None
    ee_shared_mem = None
    ee_dof = 0
    action_log_file = None
    action_log_path = None
    ready_arm_pose = lowered_arm_pose = None
    shutdown_done = False
    async_stop_event: threading.Event | None = None
    async_inference_thread: threading.Thread | None = None

    def shutdown_robot() -> bool:
        return run_shutdown_sequence(
            cfg,
            arm_ctrl,
            arm_ik,
            ready_arm_pose,
            lowered_arm_pose,
            key_listener=key_listener,
            ee_shared_mem=ee_shared_mem,
            ee_dof=ee_dof,
        )

    try:
        # --- Setup Phase ---
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        # Unpack interfaces for convenience
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )
        (
            tv_img_array,
            wrist_img_array,
            thermal_img_array,
            tv_img_shape,
            wrist_img_shape,
            thermal_img_shape,
            is_binocular,
            has_wrist_cam,
            has_thermal_cam,
        ) = (
            image_info[key]
            for key in [
                "tv_img_array",
                "wrist_img_array",
                "thermal_img_array",
                "tv_img_shape",
                "wrist_img_shape",
                "thermal_img_shape",
                "is_binocular",
                "has_wrist_cam",
                "has_thermal_cam",
            ]
        )

        # Get initial pose from the first step of the dataset
        from_idx = dataset.meta.episodes["dataset_from_index"][0]
        step = dataset[from_idx]
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()
        init_ee_pose = step["observation.state"][arm_dof : arm_dof + 2 * ee_dof].cpu().numpy()
        expected_policy_dim = arm_dof + (2 * ee_dof if cfg.ee else 0)
        compressed_dex3_action_dim = arm_dof + 2
        supports_compressed_dex3_action = cfg.ee == "dex3" and ee_dof == 7
        policy_state_dim = _policy_feature_dim(getattr(cfg.policy, "input_features", None), "observation.state")
        policy_action_dim = _policy_feature_dim(getattr(cfg.policy, "output_features", None), "action")
        logger_mp.info(
            f"Policy dimension check: expected real-robot state/action dim {expected_policy_dim}; "
            f"policy state dim {policy_state_dim}; policy action dim {policy_action_dim}."
        )
        if policy_state_dim is not None and policy_state_dim != expected_policy_dim:
            raise RuntimeError(
                "Policy observation.state dimension does not match enabled robot interfaces: "
                f"policy expects {policy_state_dim}, but arm={arm_dof}, ee='{cfg.ee}' gives "
                f"{expected_policy_dim}. Use --ee=dex3 for a 28D hand policy, or an arm-only checkpoint "
                "for --ee=''."
            )
        valid_policy_action_dims = {expected_policy_dim}
        if supports_compressed_dex3_action:
            valid_policy_action_dims.add(compressed_dex3_action_dim)
        if policy_action_dim is not None and policy_action_dim not in valid_policy_action_dims:
            raise RuntimeError(
                "Policy action dimension does not match enabled robot interfaces: "
                f"policy outputs {policy_action_dim}, but arm={arm_dof}, ee='{cfg.ee}' gives "
                f"valid dimensions {sorted(valid_policy_action_dims)}."
            )
        if policy_action_dim == compressed_dex3_action_dim and supports_compressed_dex3_action:
            logger_mp.info(
                "Using compressed Dex3 policy actions: expanding the two grip scalars "
                "to fixed left/right 7-DoF grasp targets before robot control."
            )

        key_listener.start()
        logger_mp.info(
            "Keyboard controls: r=move to dataset first frame, s=start policy, "
            "q=open hands, retreat and lower, c=continue lowering after obstacle clearance."
        )

        if hasattr(arm_ctrl, "arm_velocity_limit"):
            arm_ctrl.arm_velocity_limit = float(cfg.arm_low_level_velocity_limit)
            logger_mp.info(
                f"Low-level arm velocity backstop set to {arm_ctrl.arm_velocity_limit:.4f} rad/s; "
                f"trajectory speed is {cfg.arm_velocity_limit:.4f} rad/s; "
                f"tracking error limit is {cfg.arm_tracking_error_limit:.4f} rad; "
                f"slow move tau scale is {cfg.slow_move_tau_scale:.2f}."
            )

        warmup_policy_from_real_observation(
            cfg,
            policy,
            preprocessor,
            postprocessor,
            get_safe_torch_device(policy.config.device),
            step["task"],
            arm_ctrl,
            ee_shared_mem,
            arm_dof,
            ee_dof,
            tv_img_array,
            wrist_img_array,
            thermal_img_array,
            tv_img_shape,
            wrist_img_shape,
            thermal_img_shape,
            is_binocular,
            has_wrist_cam,
            has_thermal_cam,
            runtime_thermal_input_adapter,
        )

        lowered_arm_pose = hold_current_arm_pose(arm_ctrl, arm_ik)
        ready_arm_pose = get_ready_arm_pose(cfg, arm_ik, lowered_arm_pose)
        logger_mp.info(f"Startup/lowered arm pose captured:\n{lowered_arm_pose}")
        logger_mp.info(f"Ready arm pose target:\n{ready_arm_pose}")

        if cfg.auto_ready_pose:
            if not move_arm_to_pose_slowly(
                cfg,
                arm_ctrl,
                arm_ik,
                ready_arm_pose,
                "ready arm pose",
                key_listener=key_listener,
                trajectory_velocity_limit=cfg.ready_arm_velocity_limit,
            ):
                logger_mp.info("Ready arm pose move did not complete; starting shutdown sequence.")
                shutdown_done = shutdown_robot()
                return

        idx = 0

        logger_mp.info("Ready. Press 'r' to slowly move to the dataset first frame, or 'q' to open hands and lower.")
        key = key_listener.wait_for("r", "q")
        if key == "q":
            shutdown_done = shutdown_robot()
            return

        save_head_camera_snapshot(tv_img_array)

        # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
        logger_mp.info("Preparing to move robot to dataset starting pose...")
        if not move_arm_to_pose_slowly(
            cfg,
            arm_ctrl,
            arm_ik,
            init_arm_pose,
            "dataset first frame",
            key_listener=key_listener,
        ):
            logger_mp.info("Dataset first frame move did not complete; starting shutdown sequence.")
            shutdown_done = shutdown_robot()
            return
        if cfg.ee and not move_ee_to_pose_slowly(
            cfg,
            ee_shared_mem,
            ee_dof,
            init_ee_pose,
            "dataset first frame hands",
            key_listener=key_listener,
        ):
            logger_mp.info("Dataset first frame hand move did not complete; starting shutdown sequence.")
            shutdown_done = shutdown_robot()
            return

        logger_mp.info("At dataset first frame. Press 's' to start policy, or 'q' to open hands and lower.")
        key = key_listener.wait_for("s", "q")
        if key == "q":
            shutdown_done = shutdown_robot()
            return

        # --- Run Main Loop ---
        device = get_safe_torch_device(policy.config.device)
        actions_per_chunk = resolve_async_actions_per_chunk(cfg, policy)
        aggregate_fn = get_async_aggregate_function(cfg.async_aggregate_fn)
        logger_mp.info(
            f"Starting async closed-loop chunk control at {cfg.frequency} Hz; "
            f"actions_per_chunk={actions_per_chunk}, "
            f"chunk_size_threshold={cfg.async_chunk_size_threshold:.2f}, "
            f"aggregate_fn={cfg.async_aggregate_fn}."
        )
        action_log_file, action_log_path = open_action_log(cfg)
        if action_log_file is not None:
            dataset_action = step["action"].cpu().numpy() if "action" in step else np.array([])
            dataset_state = step["observation.state"].cpu().numpy()
            write_action_log(
                action_log_file,
                {
                    "type": "metadata",
                    "created_time_s": time.time(),
                    "repo_id": cfg.repo_id,
                    "root": cfg.root,
                    "policy_path": str(getattr(cfg.policy, "path", "") or ""),
                    "task": step["task"],
                    "selected_camera_features": _selected_camera_feature_names(cfg),
                    "thermal_input_type": _thermal_input_type(cfg),
                    "frequency": float(cfg.frequency),
                    "arm_dof": int(arm_dof),
                    "ee_dof": int(ee_dof),
                    "max_policy_arm_delta": float(cfg.max_policy_arm_delta),
                    "max_policy_ee_delta": float(cfg.max_policy_ee_delta),
                    "limit_policy_delta_from_last_command": bool(cfg.limit_policy_delta_from_last_command),
                    "ee_tracking_tolerance": float(cfg.ee_tracking_tolerance),
                    "async_actions_per_chunk": int(actions_per_chunk),
                    "async_requested_actions_per_chunk": int(cfg.async_actions_per_chunk),
                    "async_chunk_size_threshold": float(cfg.async_chunk_size_threshold),
                    "async_aggregate_fn": str(cfg.async_aggregate_fn),
                    "async_action_queue_wait_timeout_s": float(cfg.async_action_queue_wait_timeout_s),
                    "init_arm_pose": _json_array(init_arm_pose),
                    "init_ee_pose": _json_array(init_ee_pose),
                    "dataset_action": _json_array(dataset_action),
                    "dataset_action_minus_state_abs_max": _max_abs(dataset_action - dataset_state)
                    if dataset_action.shape == dataset_state.shape
                    else None,
                    "dataset_arm_action_minus_state_abs_max": _max_abs(
                        dataset_action[:arm_dof] - dataset_state[:arm_dof]
                    )
                    if dataset_action.shape == dataset_state.shape
                    else None,
                    "dataset_ee_action_minus_state_abs_max": _max_abs(
                        dataset_action[arm_dof : arm_dof + 2 * ee_dof]
                        - dataset_state[arm_dof : arm_dof + 2 * ee_dof]
                    )
                    if dataset_action.shape == dataset_state.shape and ee_dof > 0
                    else None,
                },
            )
        next_policy_log_time = time.perf_counter()
        action_log_every_n = max(1, int(cfg.action_log_every_n))
        prev_raw_action_np = None
        prev_sent_action_np = None
        async_stop_event = threading.Event()
        async_observation_queue = Queue(maxsize=1)
        async_action_queue = AsyncActionQueue(aggregate_fn, actions_per_chunk)
        async_must_go = threading.Event()
        async_must_go.set()
        async_error: dict[str, Exception | None] = {"exc": None}
        async_stats: dict[str, Any] = {
            "chunks": 0,
            "last_chunk_size": 0,
            "last_accepted_actions": 0,
            "last_queue_size": 0,
            "last_inference_elapsed_s": None,
            "last_observation_timestep": None,
        }

        def async_inference_worker() -> None:
            logger_mp.info("Async inference worker started.")
            queue_timeout = max(0.001, float(cfg.async_observation_queue_timeout_s))
            while not async_stop_event.is_set():
                try:
                    timed_observation = async_observation_queue.get(timeout=queue_timeout)
                except Empty:
                    continue

                if async_stop_event.is_set():
                    break

                inference_start_time = time.perf_counter()
                try:
                    actions = predict_action_chunk_async(
                        timed_observation.observation,
                        policy,
                        device,
                        preprocessor,
                        postprocessor,
                        policy.config.use_amp,
                        step["task"],
                        actions_per_chunk,
                        use_dataset=cfg.use_dataset,
                        robot_type=None,
                    )
                    inference_elapsed = time.perf_counter() - inference_start_time
                    if async_stop_event.is_set():
                        break
                    if cfg.policy_inference_timeout_s > 0 and inference_elapsed > cfg.policy_inference_timeout_s:
                        raise TimeoutError(
                            f"Policy chunk inference took {inference_elapsed:.2f}s, exceeding "
                            f"policy_inference_timeout_s {cfg.policy_inference_timeout_s:.2f}s."
                        )

                    timed_actions = [
                        AsyncTimedAction(
                            timestamp=timed_observation.timestamp + action_idx / float(cfg.frequency),
                            timestep=timed_observation.timestep + action_idx,
                            action=action,
                            observation_timestep=timed_observation.timestep,
                            inference_elapsed_s=inference_elapsed,
                        )
                        for action_idx, action in enumerate(actions)
                    ]
                    accepted_actions, queue_size = async_action_queue.merge(timed_actions)
                    async_must_go.set()
                    async_stats.update(
                        {
                            "chunks": int(async_stats["chunks"]) + 1,
                            "last_chunk_size": len(timed_actions),
                            "last_accepted_actions": accepted_actions,
                            "last_queue_size": queue_size,
                            "last_inference_elapsed_s": inference_elapsed,
                            "last_observation_timestep": timed_observation.timestep,
                        }
                    )
                except Exception as exc:
                    async_error["exc"] = exc
                    async_stop_event.set()
                    key_listener.shutdown_requested = True
                    logger_mp.warning(f"Async inference worker failed: {exc}")
                    return

            logger_mp.info("Async inference worker stopped.")

        async_inference_thread = threading.Thread(
            target=async_inference_worker,
            name="s1_async_inference",
            daemon=True,
        )
        async_inference_thread.start()

        no_action_started_at = None
        next_wait_log_time = time.perf_counter()
        while True:
            if async_error["exc"] is not None:
                raise RuntimeError("Async inference worker failed.") from async_error["exc"]
            if key_listener.shutdown_requested:
                logger_mp.info("q pressed; stopping policy and starting shutdown sequence.")
                async_stop_event.set()
                shutdown_done = shutdown_robot()
                return
            loop_start_time = time.perf_counter()
            # 1. Get Observations
            observation, current_arm_q, left_ee_state, right_ee_state, state_tensor = collect_real_observation(
                cfg,
                arm_ctrl,
                ee_shared_mem,
                arm_dof,
                ee_dof,
                tv_img_array,
                wrist_img_array,
                thermal_img_array,
                tv_img_shape,
                wrist_img_shape,
                thermal_img_shape,
                is_binocular,
                has_wrist_cam,
                has_thermal_cam,
                runtime_thermal_input_adapter,
            )
            save_runtime_thermal_input_preview(
                cfg,
                observation,
                runtime_thermal_input_adapter,
                label="on_s_first_frame",
            )
            if key_listener.shutdown_requested:
                logger_mp.info("q pressed; stopping async control and starting shutdown sequence.")
                async_stop_event.set()
                shutdown_done = shutdown_robot()
                return

            if async_action_queue.ready_to_send_observation(cfg.async_chunk_size_threshold):
                must_go = async_must_go.is_set() and async_action_queue.qsize() == 0
                timed_observation = AsyncTimedObservation(
                    timestamp=time.time(),
                    timestep=async_action_queue.latest_timestep_for_observation(),
                    observation=clone_observation_for_async(observation),
                    must_go=must_go,
                )
                if enqueue_latest_observation(async_observation_queue, timed_observation) and must_go:
                    async_must_go.clear()

            # 2. Execute the next locally queued action. The inference thread refills this queue.
            timed_action = async_action_queue.pop_next()
            if timed_action is None:
                now = time.perf_counter()
                if no_action_started_at is None:
                    no_action_started_at = now
                    logger_mp.info("Waiting for the next async action chunk; holding current arm pose.")
                wait_elapsed = now - no_action_started_at
                hold_tau = np.asarray(arm_ik.solve_tau(current_arm_q), dtype=np.float64)
                arm_ctrl.ctrl_dual_arm(current_arm_q, hold_tau)

                if now >= next_wait_log_time:
                    logger_mp.info(
                        f"Async action queue empty for {wait_elapsed:.2f}s; "
                        f"latest_action={async_action_queue.latest_timestep_for_observation()}, "
                        f"chunks_received={async_stats['chunks']}."
                    )
                    next_wait_log_time = now + 2.0

                if (
                    cfg.async_action_queue_wait_timeout_s > 0
                    and wait_elapsed > cfg.async_action_queue_wait_timeout_s
                ):
                    logger_mp.info(
                        f"No async action became available within "
                        f"{cfg.async_action_queue_wait_timeout_s:.2f}s; starting shutdown sequence."
                    )
                    async_stop_event.set()
                    shutdown_done = shutdown_robot()
                    return

                time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
                continue

            no_action_started_at = None
            action_result = execute_robot_action(
                cfg,
                timed_action.action,
                current_arm_q,
                left_ee_state,
                right_ee_state,
                arm_dof,
                ee_dof,
                arm_ctrl,
                arm_ik,
                ee_shared_mem,
                prev_sent_action_np,
            )

            raw_action_np = action_result["raw_action_np"]
            action_np = action_result["action_np"]
            raw_arm_action = action_result["raw_arm_action"]
            arm_action = action_result["arm_action"]
            tau = action_result["tau"]
            raw_arm_delta = action_result["raw_arm_delta"]
            clipped_arm_delta = action_result["clipped_arm_delta"]
            raw_left_ee_action = action_result["raw_left_ee_action"]
            raw_right_ee_action = action_result["raw_right_ee_action"]
            left_ee_action = action_result["left_ee_action"]
            right_ee_action = action_result["right_ee_action"]
            raw_left_delta = action_result["raw_left_delta"]
            raw_right_delta = action_result["raw_right_delta"]
            sent_left_delta = action_result["sent_left_delta"]
            sent_right_delta = action_result["sent_right_delta"]

            now = time.perf_counter()
            raw_action_step_delta = (
                raw_action_np - prev_raw_action_np if prev_raw_action_np is not None else np.array([])
            )
            sent_action_step_delta = (
                action_np - prev_sent_action_np if prev_sent_action_np is not None else np.array([])
            )
            queue_size_after_pop = async_action_queue.qsize()
            if action_log_file is not None and idx % action_log_every_n == 0:
                record = {
                    "type": "policy_step",
                    "idx": int(idx),
                    "time_s": time.time(),
                    "action_timestep": int(timed_action.timestep),
                    "observation_timestep": int(timed_action.observation_timestep),
                    "action_queue_size": int(queue_size_after_pop),
                    "async_chunks_received": int(async_stats["chunks"]),
                    "async_last_chunk_size": int(async_stats["last_chunk_size"]),
                    "async_last_accepted_actions": int(async_stats["last_accepted_actions"]),
                    "inference_elapsed_s": round(float(timed_action.inference_elapsed_s), 6),
                    "loop_elapsed_s": round(float(now - loop_start_time), 6),
                    "raw_action": _json_array(raw_action_np),
                    "sent_action": _json_array(action_np),
                    "current_arm_q": _json_array(current_arm_q),
                    "raw_arm_action": _json_array(raw_arm_action),
                    "sent_arm_action": _json_array(arm_action),
                    "tau": _json_array(tau),
                    "raw_arm_delta": _json_array(raw_arm_delta),
                    "sent_arm_delta": _json_array(clipped_arm_delta),
                    "raw_arm_delta_abs_max": _max_abs(raw_arm_delta),
                    "raw_arm_delta_abs_argmax": _argmax_abs(raw_arm_delta),
                    "sent_arm_delta_abs_max": _max_abs(clipped_arm_delta),
                    "sent_arm_delta_abs_argmax": _argmax_abs(clipped_arm_delta),
                    "arm_was_clipped": bool(_max_abs(raw_arm_delta - clipped_arm_delta) > 1e-6),
                    "raw_action_step_delta_abs_max": _max_abs(raw_action_step_delta),
                    "raw_action_step_delta_abs_argmax": _argmax_abs(raw_action_step_delta),
                    "sent_action_step_delta_abs_max": _max_abs(sent_action_step_delta),
                    "sent_action_step_delta_abs_argmax": _argmax_abs(sent_action_step_delta),
                }
                if cfg.ee:
                    record.update(
                        {
                            "current_left_ee": _json_array(left_ee_state),
                            "current_right_ee": _json_array(right_ee_state),
                            "raw_left_ee_action": _json_array(raw_left_ee_action),
                            "raw_right_ee_action": _json_array(raw_right_ee_action),
                            "sent_left_ee_action": _json_array(left_ee_action),
                            "sent_right_ee_action": _json_array(right_ee_action),
                            "raw_left_ee_delta": _json_array(raw_left_delta),
                            "raw_right_ee_delta": _json_array(raw_right_delta),
                            "sent_left_ee_delta": _json_array(sent_left_delta),
                            "sent_right_ee_delta": _json_array(sent_right_delta),
                            "raw_ee_delta_abs_max": max(_max_abs(raw_left_delta), _max_abs(raw_right_delta)),
                            "sent_ee_delta_abs_max": max(_max_abs(sent_left_delta), _max_abs(sent_right_delta)),
                            "ee_was_clipped": bool(
                                max(
                                    _max_abs(raw_left_delta - sent_left_delta),
                                    _max_abs(raw_right_delta - sent_right_delta),
                                )
                                > 1e-6
                            ),
                        }
                    )
                write_action_log(action_log_file, record)
            prev_raw_action_np = raw_action_np.copy()
            prev_sent_action_np = action_np.copy()

            if now >= next_policy_log_time:
                msg = (
                    f"Policy step {idx}: action #{timed_action.timestep}, "
                    f"chunk inference {timed_action.inference_elapsed_s:.3f}s, "
                    f"queue {queue_size_after_pop}/{async_action_queue.action_chunk_size}, "
                    f"raw arm delta max {float(np.max(np.abs(raw_arm_delta))):.4f} rad, "
                    f"sent arm delta max {float(np.max(np.abs(clipped_arm_delta))):.4f} rad"
                )
                if cfg.ee:
                    msg += (
                        f", raw ee delta max "
                        f"{max(float(np.max(np.abs(raw_left_delta))), float(np.max(np.abs(raw_right_delta)))):.4f} rad, "
                        f"sent ee delta max "
                        f"{max(float(np.max(np.abs(sent_left_delta))), float(np.max(np.abs(sent_right_delta)))):.4f} rad"
                    )
                logger_mp.info(msg)
                next_policy_log_time = now + 2.0

            if cfg.visualization:
                visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)
            idx += 1
            # Maintain frequency
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt received; starting shutdown sequence.")
        key_listener.shutdown_requested = True
    except Exception as e:
        logger_mp.info(f"An error occurred: {e}")
        key_listener.shutdown_requested = True
    finally:
        if async_stop_event is not None:
            async_stop_event.set()
        if async_inference_thread is not None:
            async_inference_thread.join(timeout=1.0)
        if (
            not shutdown_done
            and arm_ctrl is not None
            and arm_ik is not None
            and ready_arm_pose is not None
            and lowered_arm_pose is not None
            and key_listener.shutdown_requested
        ):
            shutdown_robot()
        if action_log_file is not None:
            logger_mp.info(f"Action log saved to {action_log_path}")
            action_log_file.close()
        key_listener.stop()
        if image_info:
            cleanup_resources(image_info)


@parser.wrap()
def eval_main(cfg: EvalRealAsyncConfig):
    if cfg.send_real_robot and getattr(cfg.policy, "compile_model", False):
        logging.warning(
            "Disabling torch.compile for real-robot eval. "
            "The checkpoint requested compile_model=True, which can block the first policy step for autotuning."
        )
        cfg.policy.compile_model = False

    configure_policy_for_thermal_input_type(cfg)
    validate_thermal_input_selection(cfg)
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset_feature_names = _policy_dataset_feature_names(cfg)
    logging.info(
        f"Loading dataset repo_id={cfg.repo_id!r}, root={cfg.root or None!r}, "
        f"feature_names={dataset_feature_names!r}."
    )
    dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.root or None, feature_names=dataset_feature_names)
    episode_count = len(dataset.meta.episodes["dataset_from_index"])
    logging.info(
        f"Dataset loaded: root={dataset.root}, frames={len(dataset)}, "
        f"episodes={episode_count}."
    )

    logging.info("Making policy.")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    runtime_thermal_input_adapter = make_runtime_thermal_input_adapter(cfg) if cfg.send_real_robot else None

    if cfg.send_real_robot:
        warmup_policy_from_dataset(
            cfg,
            dataset,
            policy,
            preprocessor,
            postprocessor,
            device,
            runtime_thermal_input_adapter,
        )

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(cfg, dataset, policy, preprocessor, postprocessor, runtime_thermal_input_adapter)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
