"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import torch
import logging
import importlib
import json
import shlex
import select
import sys
import termios
import threading
import tty

import cv2
import numpy as np
import pinocchio as pin
from pprint import pformat
from dataclasses import asdict
from pathlib import Path
from torch import nn
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Callable

_LEROBOT_SRC = Path(__file__).resolve().parents[1] / "lerobot" / "src"
if _LEROBOT_SRC.exists() and str(_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SRC))

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
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
    clear_cached_eval_actions,
    cleanup_resources,
    extract_observation,
    predict_action,
    to_list,
    to_scalar,
    EvalRealConfig,
)
try:
    from astribot_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
except ModuleNotFoundError as exc:
    if exc.name != "rerun":
        raise
    RerunLogger = None
    visualization_data = None

import logging_mp
from astribot_lerobot.eval_robot.utils.logging_mp_compat import patch_logging_mp

patch_logging_mp(logging_mp)
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def require_rerun_visualization():
    if RerunLogger is None or visualization_data is None:
        raise ImportError(
            "Rerun visualization requires the `rerun` package. Install it with "
            "`python -m pip install rerun-sdk`, or run with `--visualization=false`."
        )
    return RerunLogger, visualization_data


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
        if key not in {"q", "r", "s", "c", "j", "b"}:
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
            if self.shutdown_requested and "q" in allowed_keys:
                return "q"
            key = self.consume(*allowed_keys)
            if key is not None:
                return key
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

    logger_mp.info(
        "No ready_arm_pose or ready_lift_m provided; using the current arm pose as ready pose. "
        "Pass --ready_arm_pose or --ready_lift_m for an explicit pre-policy retreat."
    )
    return current_arm_q.copy()


def hold_current_arm_pose(arm_ctrl, arm_ik):
    current_arm_q = require_finite_vector("current arm q", arm_ctrl.get_current_dual_arm_q())
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


def require_finite_vector(name: str, values, expected_dim: int | None = None) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise RuntimeError(f"{name} must be a 1D vector, got shape {vector.shape}.")
    if expected_dim is not None and vector.shape[0] != expected_dim:
        raise RuntimeError(f"{name} length {vector.shape[0]} does not match expected {expected_dim}.")
    if not np.all(np.isfinite(vector)):
        bad_indices = np.where(~np.isfinite(vector))[0].tolist()
        raise RuntimeError(f"{name} contains non-finite value(s) at indices {bad_indices}.")
    return vector


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


def _attention_map_enabled(cfg: EvalRealConfig) -> bool:
    return bool(getattr(cfg, "save_attention_maps", False))


def _attention_map_raw_interval_s(cfg: EvalRealConfig) -> float:
    return float(getattr(cfg, "attention_map_interval_s", 10.0))


def _attention_map_video_mode(cfg: EvalRealConfig) -> bool:
    return _attention_map_raw_interval_s(cfg) < 0.0


def _attention_video_frame_interval_s(cfg: EvalRealConfig) -> float:
    raw_interval_s = _attention_map_raw_interval_s(cfg)
    return abs(raw_interval_s) if raw_interval_s < 0.0 else 0.0


def _attention_capture_interval_s(cfg: EvalRealConfig) -> float:
    raw_interval_s = _attention_map_raw_interval_s(cfg)
    return 1.0 if raw_interval_s < 0.0 else max(0.0, raw_interval_s)


def _attention_video_fps(cfg: EvalRealConfig) -> float:
    interval_s = _attention_video_frame_interval_s(cfg)
    if interval_s <= 0.0:
        return 1.0
    target_fps = 1.0 / interval_s
    loop_fps = float(getattr(cfg, "frequency", 30.0) or 30.0)
    return float(np.clip(min(target_fps, loop_fps), 0.1, 60.0))


def _attention_overlay_video_fps(cfg: EvalRealConfig) -> float:
    interval_s = _attention_capture_interval_s(cfg)
    if interval_s <= 0.0:
        return 1.0
    return float(np.clip(1.0 / interval_s, 0.1, 30.0))


def _action_log_base_dir(cfg: EvalRealConfig) -> Path:
    path_text = str(getattr(cfg, "action_log_path", "eval_action_logs")).strip()
    if not path_text or path_text.lower() in {"false", "none", "off", "0"}:
        path = Path("eval_action_logs")
    else:
        path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.parent if path.suffix == ".jsonl" else path


def _save_run_command_files(cfg: EvalRealConfig, artifact_dir: Path) -> None:
    saved_dir = getattr(cfg, "_run_command_saved_dir", "")
    if saved_dir == str(artifact_dir):
        return

    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        argv = list(sys.argv)
        command = shlex.join([sys.executable, *argv])
        metadata = {
            "created_time_s": time.time(),
            "created_time_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "cwd": str(Path.cwd()),
            "python_executable": sys.executable,
            "argv": argv,
            "command": command,
        }
        (artifact_dir / "run_command.txt").write_text(
            f"{command}\n",
            encoding="utf-8",
        )
        (artifact_dir / "run_command.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        setattr(cfg, "_run_command_saved_dir", str(artifact_dir))
    except Exception as exc:
        logger_mp.warning("Could not save run command files to %s: %s", artifact_dir, exc)


def ensure_eval_artifact_dir(cfg: EvalRealConfig) -> Path:
    path_text = getattr(cfg, "_attention_map_dir", "")
    if path_text:
        artifact_dir = Path(path_text)
        _save_run_command_files(cfg, artifact_dir)
        return artifact_dir

    artifact_dir = _action_log_base_dir(cfg) / f"attention_maps_{time.strftime('%Y%m%d_%H%M%S')}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    setattr(cfg, "_attention_map_dir", str(artifact_dir))
    _save_run_command_files(cfg, artifact_dir)
    if not hasattr(cfg, "_attention_map_capture_count"):
        initial_count = 0 if bool(getattr(cfg, "attention_map_warmup", False)) else 1
        setattr(cfg, "_attention_map_capture_count", initial_count)
    return artifact_dir


def prepare_attention_map_dir(cfg: EvalRealConfig) -> Path | None:
    if not _attention_map_enabled(cfg):
        return None
    attention_dir = ensure_eval_artifact_dir(cfg)
    logger_mp.info(f"Writing policy attention artifacts to {attention_dir}")
    return attention_dir


def get_attention_map_dir(cfg: EvalRealConfig) -> Path | None:
    path_text = getattr(cfg, "_attention_map_dir", "")
    if not path_text:
        return prepare_attention_map_dir(cfg)
    return Path(path_text)


def _attention_video_state(cfg: EvalRealConfig, attention_dir: Path) -> dict[str, Any]:
    state = getattr(cfg, "_attention_video_state", None)
    if isinstance(state, dict):
        return state
    state = {
        "dir": str(attention_dir),
        "frame_video_fps": _attention_video_fps(cfg),
        "attention_video_fps": _attention_overlay_video_fps(cfg),
        "writers": {},
        "videos": {},
    }
    setattr(cfg, "_attention_video_state", state)
    return state


def _to_video_bgr_frame(frame_bgr: np.ndarray) -> np.ndarray | None:
    frame = np.asarray(frame_bgr)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3:
        return None
    if frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    if frame.shape[-1] != 3:
        return None
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _append_attention_video_frame(
    cfg: EvalRealConfig,
    attention_dir: Path,
    video_name: str,
    frame_bgr: np.ndarray,
    *,
    fps: float | None = None,
) -> dict[str, Any] | None:
    frame = _to_video_bgr_frame(frame_bgr)
    if frame is None:
        return None

    state = _attention_video_state(cfg, attention_dir)
    key = _safe_stem(video_name)
    writers = state["writers"]
    videos = state["videos"]
    height, width = int(frame.shape[0]), int(frame.shape[1])

    if key not in writers:
        path = attention_dir / f"{key}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_fps = float(fps if fps is not None else state["frame_video_fps"])
        writer = cv2.VideoWriter(
            str(path),
            fourcc,
            video_fps,
            (width, height),
        )
        if not writer.isOpened():
            fallback_path = attention_dir / f"{key}.avi"
            writer = cv2.VideoWriter(
                str(fallback_path),
                cv2.VideoWriter_fourcc(*"XVID"),
                video_fps,
                (width, height),
            )
            path = fallback_path
        if not writer.isOpened():
            logger_mp.warning(f"Could not open attention video writer for {path}.")
            return None
        writers[key] = writer
        videos[key] = {
            "path": str(path),
            "fps": video_fps,
            "size": [height, width],
            "frame_count": 0,
            "first_time_s": time.time(),
            "last_time_s": None,
        }

    info = videos[key]
    video_h, video_w = (int(value) for value in info["size"])
    if frame.shape[:2] != (video_h, video_w):
        frame = cv2.resize(frame, (video_w, video_h), interpolation=cv2.INTER_AREA)

    writers[key].write(frame)
    info["frame_count"] = int(info.get("frame_count", 0)) + 1
    info["last_time_s"] = time.time()
    return {
        "path": info["path"],
        "fps": info["fps"],
        "size": info["size"],
        "frame_count": info["frame_count"],
    }


def finalize_attention_videos(cfg: EvalRealConfig) -> dict[str, Any] | None:
    state = getattr(cfg, "_attention_video_state", None)
    if not isinstance(state, dict):
        return None

    for writer in list(state.get("writers", {}).values()):
        writer.release()
    videos = dict(state.get("videos", {}))
    summary = {
        "created_time_s": time.time(),
        "attention_map_interval_s": _attention_map_raw_interval_s(cfg),
        "capture_interval_s": _attention_capture_interval_s(cfg),
        "video_frame_interval_s": _attention_video_frame_interval_s(cfg),
        "frame_video_fps": float(state.get("frame_video_fps", _attention_video_fps(cfg))),
        "attention_video_fps": float(
            state.get("attention_video_fps", _attention_overlay_video_fps(cfg))
        ),
        "videos": videos,
    }
    summary_path = Path(state["dir"]) / "attention_videos.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    setattr(cfg, "_attention_video_state", None)
    logger_mp.info(f"Saved attention videos summary to {summary_path}: {list(videos)}")
    return summary


def pop_policy_attention_map(policy: PreTrainedPolicy) -> dict[str, Any] | None:
    if hasattr(policy, "pop_last_attention_map"):
        return policy.pop_last_attention_map()
    model = getattr(policy, "model", None)
    if model is not None and hasattr(model, "pop_last_attention_map"):
        return model.pop_last_attention_map()
    return None


def _safe_stem(text: str) -> str:
    cleaned = []
    for char in str(text):
        cleaned.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(cleaned).strip("_") or "attention"


def _to_numpy(value, dtype=np.float32) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _infer_token_grid(token_count: int) -> tuple[int, int] | None:
    side = int(round(float(token_count) ** 0.5))
    if side > 0 and side * side == token_count:
        return side, side
    return None


def _infer_patch_token_grid(token_count: int) -> tuple[tuple[int, int], slice] | None:
    grid = _infer_token_grid(token_count)
    if grid is not None:
        return grid, slice(0, token_count)

    side = int(np.floor(float(token_count) ** 0.5))
    if side <= 0:
        return None

    patch_count = side * side
    # DINO-style encoders often prepend CLS/register tokens before square patch tokens.
    return (side, side), slice(token_count - patch_count, token_count)


def _infer_factor_token_grid(
    token_count: int,
    image_shape: tuple[int, int, int] | None = None,
) -> tuple[int, int] | None:
    if token_count <= 0:
        return None
    grid = _infer_token_grid(token_count)
    if grid is not None:
        return grid

    candidates: list[tuple[int, int]] = []
    for height in range(1, int(np.floor(float(token_count) ** 0.5)) + 1):
        if token_count % height != 0:
            continue
        width = token_count // height
        candidates.append((height, width))
        if height != width:
            candidates.append((width, height))
    if not candidates:
        return None

    if image_shape is not None:
        image_h, image_w = int(image_shape[0]), int(image_shape[1])
        if image_h > 0 and image_w > 0:
            target_aspect = float(image_w) / float(image_h)
            return min(
                candidates,
                key=lambda item: (
                    abs((float(item[1]) / float(item[0])) - target_aspect),
                    abs(item[0] - item[1]),
                ),
            )

    return min(
        candidates,
        key=lambda item: (
            abs(item[0] - item[1]),
            0 if item[1] >= item[0] else 1,
        ),
    )


def _is_attention_image_segment(segment: dict[str, Any]) -> bool:
    kind = str(segment.get("kind", ""))
    return kind == "image" or kind.startswith("image_")


def _write_heatmap_png(
    values: np.ndarray,
    path: Path,
    min_width: int = 320,
    min_height: int = 96,
    value_range: tuple[float, float] | None = None,
    apply_colormap: bool = True,
    square_cells: bool = False,
) -> bool:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.size == 0 or not np.any(np.isfinite(matrix)):
        return False
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if value_range is None:
        lo = float(matrix.min())
        hi = float(matrix.max())
    else:
        lo, hi = (float(value) for value in value_range)
        if hi <= lo:
            raise ValueError(f"value_range must be increasing, got {value_range}.")
    if hi > lo:
        matrix = np.clip((matrix - lo) / (hi - lo), 0.0, 1.0)
    else:
        matrix = np.zeros_like(matrix)
    image = np.clip(matrix * 255.0, 0, 255).astype(np.uint8)
    scale_w = max(1, int(np.ceil(min_width / max(1, image.shape[1]))))
    scale_h = max(1, int(np.ceil(min_height / max(1, image.shape[0]))))
    if square_cells:
        scale = max(scale_w, scale_h)
        scale_w = scale_h = scale
    image = cv2.resize(
        image,
        (image.shape[1] * scale_w, image.shape[0] * scale_h),
        interpolation=cv2.INTER_NEAREST,
    )
    if apply_colormap:
        colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
        image = cv2.applyColorMap(image, colormap)
    return bool(cv2.imwrite(str(path), image))


def _observation_image_to_rgb_uint8(image) -> np.ndarray | None:
    if image is None:
        return None
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim != 3:
        return None
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[:, :, :3]
    elif image.shape[-1] != 3:
        return None

    image = image.astype(np.float32, copy=False)
    finite = np.isfinite(image)
    if not finite.any():
        return None
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(image.min())
    max_value = float(image.max())
    if min_value >= -1.0 and max_value <= 1.0:
        image = (image + 1.0) * 0.5 if min_value < 0.0 else image
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _get_observation_image(observation: dict[str, Any] | None, image_name: str):
    if not observation:
        return None
    candidates = [image_name]
    if image_name.startswith("images."):
        candidates.append(f"observation.{image_name}")
    elif not image_name.startswith("observation.images."):
        candidates.append(f"observation.images.{image_name}")
    for candidate in candidates:
        if candidate in observation:
            return observation[candidate]
    return None


def _append_unique_feature_name(target: list[str], value: Any) -> None:
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _append_unique_feature_name(target, item)
        return
    text = str(value or "").strip()
    if text and text not in target:
        target.append(text)


def _attention_image_feature_candidates(
    cfg: EvalRealConfig,
    prefix_token_layout: list[dict[str, Any]],
) -> dict[str, list[str]]:
    policy_cfg = getattr(cfg, "policy", None)
    head_candidates: list[str] = []
    thermal_candidates: list[str] = []

    _append_unique_feature_name(
        head_candidates,
        getattr(policy_cfg, "thermal_fusion_head_rgb_feature", None),
    )
    _append_unique_feature_name(
        head_candidates,
        getattr(policy_cfg, "rgb_thermal_mix_rgb_feature", None),
    )
    for segment in prefix_token_layout:
        _append_unique_feature_name(head_candidates, segment.get("head_rgb_feature"))
    _append_unique_feature_name(head_candidates, "observation.images.cam_left_high")

    _append_unique_feature_name(
        thermal_candidates,
        getattr(policy_cfg, "rgb_thermal_mix_thermal_feature", None),
    )
    _append_unique_feature_name(
        thermal_candidates,
        getattr(policy_cfg, "thermal_image_features", None),
    )
    for segment in prefix_token_layout:
        _append_unique_feature_name(thermal_candidates, segment.get("thermal_features"))
    _append_unique_feature_name(thermal_candidates, "observation.images.cam_" + "ther" + "mal")

    return {
        "rgb_head": list(
            dict.fromkeys(_raw_feature_name(cfg, name) for name in head_candidates)
        ),
        "thermal": list(
            dict.fromkeys(_raw_feature_name(cfg, name) for name in thermal_candidates)
        ),
    }


def _write_observation_image_png(image_rgb: np.ndarray, path: Path) -> bool:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return bool(cv2.imwrite(str(path), image_bgr))


def _save_attention_frame_images(
    cfg: EvalRealConfig,
    attention_dir: Path,
    stem: str,
    prefix_token_layout: list[dict[str, Any]],
    observation: dict[str, Any] | None,
) -> dict[str, dict[str, str]]:
    if not bool(getattr(cfg, "save_attention_images", False)):
        return {}

    image_paths: dict[str, dict[str, str]] = {}
    for role, candidates in _attention_image_feature_candidates(
        cfg,
        prefix_token_layout,
    ).items():
        for feature_name in candidates:
            image_rgb = _observation_image_to_rgb_uint8(
                _get_observation_image(observation, feature_name)
            )
            if image_rgb is None:
                continue
            if _attention_map_video_mode(cfg):
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                video_info = _append_attention_video_frame(
                    cfg,
                    attention_dir,
                    f"{role}_{_safe_stem(feature_name)}",
                    image_bgr,
                )
                if video_info is None:
                    continue
                image_paths[role] = {
                    "feature": feature_name,
                    **video_info,
                }
                break
            image_path = (
                attention_dir
                / f"{stem}_{role}_{_safe_stem(feature_name)}.png"
            )
            if _write_observation_image_png(image_rgb, image_path):
                image_paths[role] = {
                    "feature": feature_name,
                    "path": str(image_path),
                }
                break
    return image_paths


def save_manual_camera_snapshot(
    cfg: EvalRealConfig,
    observation: dict[str, Any] | None,
    *,
    label: str,
    step_idx: int | None = None,
) -> dict[str, Any] | None:
    artifact_dir = ensure_eval_artifact_dir(cfg)
    snapshot_count = int(getattr(cfg, "_manual_camera_snapshot_count", 0)) + 1
    setattr(cfg, "_manual_camera_snapshot_count", snapshot_count)

    stem_parts = [
        "manual_snapshot",
        f"{snapshot_count:04d}",
        _safe_stem(label),
    ]
    if step_idx is not None:
        stem_parts.append(f"step_{int(step_idx):06d}")
    stem = "_".join(stem_parts)

    metadata: dict[str, Any] = {
        "created_time_s": time.time(),
        "label": label,
        "step_idx": int(step_idx) if step_idx is not None else None,
        "saved": {},
        "missing": {},
    }
    candidates_by_role = _attention_image_feature_candidates(cfg, [])
    for role, candidates in candidates_by_role.items():
        role_saved = False
        for feature_name in candidates:
            image_rgb = _observation_image_to_rgb_uint8(
                _get_observation_image(observation, feature_name)
            )
            if image_rgb is None:
                continue
            image_path = artifact_dir / f"{stem}_{role}_{_safe_stem(feature_name)}.png"
            if not _write_observation_image_png(image_rgb, image_path):
                logger_mp.warning("Manual snapshot PNG write failed for %s at %s.", feature_name, image_path)
                continue
            metadata["saved"][role] = {
                "feature": feature_name,
                "path": str(image_path),
                "shape": list(image_rgb.shape),
                "dtype": str(image_rgb.dtype),
            }
            role_saved = True
            break
        if not role_saved:
            metadata["missing"][role] = list(candidates)

    if not metadata["saved"]:
        logger_mp.warning(
            "Manual snapshot skipped: no camera images found. Available keys: %s.",
            sorted(observation.keys()) if observation else [],
        )
        return None

    metadata_path = artifact_dir / f"{stem}_metadata.json"
    metadata["metadata_path"] = str(metadata_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger_mp.info(
        "Saved manual camera snapshot #%04d to %s: %s",
        snapshot_count,
        artifact_dir,
        {
            role: item["path"]
            for role, item in metadata["saved"].items()
        },
    )
    return metadata


def _camera_name_candidates(name: str) -> set[str]:
    name = str(name).strip()
    candidates = {name}
    if name.startswith("images."):
        candidates.add(f"observation.{name}")
    elif not name.startswith("observation.images."):
        candidates.add(f"observation.images.{name}")
    return {candidate for candidate in candidates if candidate}


def _attention_map_camera_matches(cfg: EvalRealConfig, image_name: str) -> bool:
    target = str(getattr(cfg, "attention_map_camera", "observation.images.cam_left_high") or "").strip()
    if target.lower() in {"*", "all"}:
        return True
    if not target:
        return False
    return bool(_camera_name_candidates(target) & _camera_name_candidates(image_name))


def _normalize_heatmap(values: np.ndarray) -> np.ndarray:
    heat = np.asarray(values, dtype=np.float32)
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)
    if heat.size == 0:
        return heat
    lo = float(np.percentile(heat, 5.0))
    hi = float(np.percentile(heat, 99.0))
    if hi <= lo:
        lo = float(heat.min())
        hi = float(heat.max())
    return np.clip((heat - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(heat)


def _valid_patch_crop(matrix: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    grid_h, grid_w = matrix.shape
    longest = float(max(image_h, image_w))
    valid_h = max(1, min(grid_h, int(round(grid_h * image_h / longest))))
    valid_w = max(1, min(grid_w, int(round(grid_w * image_w / longest))))
    top = max(0, (grid_h - valid_h) // 2)
    left = max(0, (grid_w - valid_w) // 2)
    return matrix[top : top + valid_h, left : left + valid_w]


def _edge_suppression_mask(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    edge_dist = np.minimum.reduce([xx, yy, width - 1 - xx, height - 1 - yy]).astype(np.float32)
    ramp = max(8.0, 0.10 * float(max(height, width)))
    return 0.30 + 0.70 * np.clip(edge_dist / ramp, 0.0, 1.0)


def _aggregate_image_attention(
    layers: list[dict[str, Any]],
    *,
    start: int,
    end: int,
    token_count: int,
    image_shape: tuple[int, int, int],
) -> np.ndarray | None:
    patch_grid = _infer_patch_token_grid(token_count)
    if patch_grid is None:
        return None
    grid, patch_slice = patch_grid

    crops = []
    layer_weights = []
    num_layers = max(1, len(layers))
    for layer_pos, layer in enumerate(layers):
        weights = _to_numpy(layer["action_to_prefix"])[start:end]
        if weights.size != token_count:
            continue
        patch_weights = weights[patch_slice]
        if patch_weights.size != grid[0] * grid[1]:
            continue
        matrix = patch_weights.reshape(grid)
        crop = _normalize_heatmap(_valid_patch_crop(matrix, image_shape))
        if crop.size == 0:
            continue
        if min(crop.shape) >= 3:
            crop = cv2.GaussianBlur(crop, (3, 3), 0)
        position = layer_pos / max(1, num_layers - 1)
        # Middle/late layers tend to carry more visual grounding; final layers are often action-format heavy.
        layer_weight = 0.15 + float(np.exp(-((position - 0.62) ** 2) / (2 * 0.22**2)))
        crops.append(crop)
        layer_weights.append(layer_weight)

    if not crops:
        return None
    crop_stack = np.stack(crops, axis=0)
    weights = np.asarray(layer_weights, dtype=np.float32)
    heat = np.average(crop_stack, axis=0, weights=weights)
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    heat = cv2.resize(heat, (image_w, image_h), interpolation=cv2.INTER_CUBIC)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(1.0, min(image_h, image_w) / 96.0))
    heat = heat * _edge_suppression_mask(image_h, image_w)
    return _normalize_heatmap(heat)


def _attention_overlay_bgr(
    image_rgb: np.ndarray,
    attention_map: np.ndarray,
    *,
    alpha: float = 0.55,
) -> np.ndarray | None:
    heat = np.asarray(attention_map, dtype=np.float32)
    if heat.size == 0 or not np.any(np.isfinite(heat)):
        return None
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(heat.min())
    hi = float(heat.max())
    heat = (heat - lo) / (hi - lo) if hi > lo else np.zeros_like(heat)

    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    color_bgr = cv2.applyColorMap(heat_u8, colormap)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, color_bgr, alpha, 0.0)


def _write_attention_overlay_png(
    image_rgb: np.ndarray,
    attention_map: np.ndarray,
    path: Path,
    *,
    alpha: float = 0.55,
) -> bool:
    overlay = _attention_overlay_bgr(image_rgb, attention_map, alpha=alpha)
    if overlay is None:
        return False
    return bool(cv2.imwrite(str(path), overlay))


def _coefficient_summary_from_scalar(value: Any) -> dict[str, Any]:
    scalar = round(float(value), 6)
    return {"mean": scalar, "min": scalar, "max": scalar, "shape": [1], "values": [scalar]}


def _patch_gate_matrix(
    summary: dict[str, Any] | None,
    image_shape: tuple[int, int, int] | None = None,
    *,
    prefer_factor_grid: bool = False,
) -> np.ndarray | None:
    if not isinstance(summary, dict):
        return None
    values = np.asarray(summary.get("values", []), dtype=np.float32)
    shape = summary.get("shape", [])
    if values.size == 0 or not isinstance(shape, list) or len(shape) < 1:
        return None
    try:
        token_count = int(shape[-1])
    except (TypeError, ValueError):
        return None
    if token_count <= 0 or values.size % token_count != 0:
        return None

    token_values = values.reshape(-1, token_count)[0]
    if prefer_factor_grid:
        grid = _infer_factor_token_grid(token_count, image_shape)
        if grid is None or token_values.size != grid[0] * grid[1]:
            return None
        return token_values.reshape(grid)

    patch_grid = _infer_patch_token_grid(token_count)
    if patch_grid is None:
        return None
    grid, patch_slice = patch_grid
    patch_values = token_values[patch_slice]
    if patch_values.size != grid[0] * grid[1]:
        return None
    return patch_values.reshape(grid)


def _select_patch_gate_matrix(
    gate: dict[str, Any],
    keys: tuple[str, ...],
    image_shape: tuple[int, int, int] | None,
) -> tuple[str | None, np.ndarray | None]:
    for key in keys:
        matrix = _patch_gate_matrix(
            gate.get(key),
            image_shape,
            prefer_factor_grid=key.startswith("small_patch_"),
        )
        if matrix is not None:
            return key, matrix
    return None, None


def _save_patch_gate_maps(
    attention_dir: Path,
    stem: str,
    prefix_token_layout: list[dict[str, Any]],
    observation: dict[str, Any] | None,
) -> dict[str, dict[str, str]]:
    map_paths = {}
    for segment in prefix_token_layout:
        if str(segment.get("kind", "")) not in {
            "image_patchgatevit_twogrey",
            "image_patchsinglegateresvit_twogrey_residual",
            "image_smallpatchgatetworesvit_cold_residual",
            "image_smallpatchsinglegatetworesvit_cold_residual",
            "image_smallpatchgateresvit_twogrey_residual",
            "image_smallpatchsinglegateresvit_twogrey_residual",
        }:
            continue
        gate = segment.get("gate") or {}
        image_name = str(segment.get("name", "thermal"))
        image_rgb = _observation_image_to_rgb_uint8(
            _get_observation_image(observation, image_name)
        )
        image_shape = image_rgb.shape if image_rgb is not None else None
        cold_key, cold_matrix = _select_patch_gate_matrix(
            gate,
            ("small_patch_cold_weight", "cold_weight", "cold_alpha"),
            image_shape,
        )
        hot_key, hot_matrix = _select_patch_gate_matrix(
            gate,
            ("small_patch_hot_weight", "hot_weight", "hot_beta"),
            image_shape,
        )
        if cold_matrix is None and hot_matrix is not None:
            cold_matrix = 1.0 - hot_matrix
        if hot_matrix is None and cold_matrix is not None:
            hot_matrix = 1.0 - cold_matrix
        if (
            cold_matrix is None
            or hot_matrix is None
            or cold_matrix.shape != hot_matrix.shape
        ):
            continue

        mode_stem = (
            "smallpatchgatevit"
            if (cold_key or hot_key or "").startswith("small_patch_")
            else "patchgatevit"
        )
        if (
            str(segment.get("kind", ""))
            == "image_smallpatchgatetworesvit_cold_residual"
        ):
            mode_stem = "smallpatchgatetworesvit"
        elif (
            str(segment.get("kind", ""))
            == "image_smallpatchsinglegatetworesvit_cold_residual"
        ):
            mode_stem = "smallpatchsinglegatetworesvit"
        elif str(segment.get("kind", "")) == "image_patchsinglegateresvit_twogrey_residual":
            mode_stem = "patchsinglegateresvit"
        elif str(segment.get("kind", "")) == "image_smallpatchgateresvit_twogrey_residual":
            mode_stem = "smallpatchgateresvit"
        elif str(segment.get("kind", "")) == "image_smallpatchsinglegateresvit_twogrey_residual":
            mode_stem = "smallpatchsinglegateresvit"

        cold_path = (
            attention_dir
            / f"{stem}_{_safe_stem(image_name)}_{mode_stem}_cold_weight_gray.png"
        )
        hot_path = (
            attention_dir
            / f"{stem}_{_safe_stem(image_name)}_{mode_stem}_hot_weight_gray.png"
        )
        saved_paths = {}
        for name, matrix, path in (
            ("cold_weight_gray", cold_matrix, cold_path),
            ("hot_weight_gray", hot_matrix, hot_path),
        ):
            if _write_heatmap_png(
                matrix,
                path,
                min_height=320,
                value_range=(0.0, 1.0),
                apply_colormap=False,
                square_cells=True,
            ):
                saved_paths[name] = str(path)
        if saved_paths:
            map_paths[image_name] = saved_paths
    return map_paths


def _extract_twogrey_coefficient_records(prefix_token_layout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for segment in prefix_token_layout:
        kind = str(segment.get("kind", ""))
        gate_mode = {
            "image_gatevit_twogrey": "GateViT",
            "image_gateactionvit_twogrey_cold": "GateActionViT",
            "image_doublegatevit_twogrey": "DoubleGateViT",
            "image_gatevitmix_twogrey": "GateViTMix",
            "image_gateresvit_twogrey_residual": "GateResViT",
            "image_gateresandvit_twogrey_residual": "GateResandViT",
            "image_gatetworesvit_cold_residual": "GateTwoResViT",
            "image_gateoneresvit_grey_residual": "GateOneResViT",
            "image_doublegatetworesvit_cold_residual": "DoubleGateTwoResViT",
            "image_patchsinglegateresvit_twogrey_residual": "PatchSingleGateResViT",
            "image_smallpatchgatetworesvit_cold_residual": "SmallPatchGateTwoResViT",
            "image_smallpatchsinglegatetworesvit_cold_residual": "SmallPatchSingleGateTwoResViT",
            "image_smallpatchgateresvit_twogrey_residual": "SmallPatchGateResViT",
            "image_smallpatchsinglegateresvit_twogrey_residual": "SmallPatchSingleGateResViT",
            "image_gateresvit3_twogrey_residual": "GateResViT3",
        }.get(kind)
        if gate_mode is not None:
            gate = segment.get("gate") or {}
            records.append(
                {
                    "mode": gate_mode,
                    "image_feature": segment.get("name"),
                    "head_rgb_feature": segment.get("head_rgb_feature"),
                    "thermal_features": segment.get("thermal_features", []),
                    "prefix_attention_scope": segment.get("prefix_attention_scope"),
                    "action_attention_bias_scope": segment.get(
                        "action_attention_bias_scope"
                    ),
                    "final_cold_coefficient": gate.get("cold_alpha"),
                    "final_hot_coefficient": gate.get("hot_beta"),
                    "final_thermal_coefficient": gate.get("thermal_alpha"),
                    "thermal_weight": gate.get("thermal_weight"),
                    "cold_weight_before_confidence": gate.get("cold_weight"),
                    "hot_weight_before_confidence": gate.get("hot_weight"),
                    "cold_multiplier": gate.get("cold_multiplier"),
                    "hot_multiplier": gate.get("hot_multiplier"),
                    "thermal_confidence": gate.get("thermal_confidence"),
                    "effective_thermal_confidence": gate.get("effective_thermal_confidence"),
                    "thermal_evidence": gate.get("thermal_evidence"),
                    "confidence_logit": gate.get("confidence_logit"),
                    "learned_confidence_logit": gate.get("learned_confidence_logit"),
                    "evidence_logit": gate.get("evidence_logit"),
                    "thermal_stats": gate.get("thermal_stats"),
                    "head_mix_context_scale": gate.get("head_mix_context_scale"),
                    "head_mix_context_rms": gate.get("head_mix_context_rms"),
                    "head_mix_delta_rms": gate.get("head_mix_delta_rms"),
                    "head_mix_valid_count": gate.get("head_mix_valid_count"),
                    "cold_attention_bias": gate.get("cold_attention_bias"),
                    "hot_attention_bias": gate.get("hot_attention_bias"),
                    "attention_bias_strength": gate.get("attention_bias_strength"),
                    "text_cold_weight": gate.get("text_cold_weight"),
                    "text_hot_weight": gate.get("text_hot_weight"),
                    "rgb_cold_alignment_weight": gate.get(
                        "rgb_cold_alignment_weight"
                    ),
                    "rgb_hot_alignment_weight": gate.get(
                        "rgb_hot_alignment_weight"
                    ),
                    "rgb_gate_strength": gate.get("rgb_gate_strength"),
                    "text_logits": gate.get("text_logits"),
                    "rgb_logits": gate.get("rgb_logits"),
                    "logits": gate.get("logits"),
                    "patch_relevance": gate.get("patch_relevance"),
                    "cold_residual_scale": gate.get("cold_residual_scale"),
                    "hot_residual_scale": gate.get("hot_residual_scale"),
                    "thermal_residual_scale": gate.get("thermal_residual_scale"),
                    "temperature_logits": gate.get("temperature_logits"),
                    "relevance_logits": gate.get("relevance_logits"),
                    "global_text_logits": gate.get("global_text_logits"),
                    "cold_evidence": gate.get("cold_evidence"),
                    "hot_evidence": gate.get("hot_evidence"),
                    "cold_valid_count": gate.get("cold_valid_count"),
                    "hot_valid_count": gate.get("hot_valid_count"),
                    "text_valid_count": gate.get("text_valid_count"),
                    "token_count": int(segment.get("token_count", 0)),
                }
            )
        elif kind == "image_resvit_twogrey_residual":
            records.append(
                {
                    "mode": "ResViT",
                    "image_feature": segment.get("name"),
                    "thermal_features": segment.get("thermal_features", []),
                    "final_cold_coefficient": _coefficient_summary_from_scalar(
                        segment.get("thermal_cold_alpha", 1.0)
                    ),
                    "final_hot_coefficient": _coefficient_summary_from_scalar(
                        segment.get("thermal_hot_beta", 1.0)
                    ),
                    "token_count": int(segment.get("token_count", 0)),
                }
            )
    return records


def _save_attention_map_capture(
    cfg: EvalRealConfig,
    policy: PreTrainedPolicy,
    *,
    label: str,
    step_idx: int | None = None,
    inference_elapsed: float | None = None,
    observation: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    attention_dir = get_attention_map_dir(cfg)
    if attention_dir is None:
        return None

    capture = pop_policy_attention_map(policy)
    if not capture or not capture.get("layers"):
        return None

    capture_count = int(getattr(cfg, "_attention_map_capture_count", 0))
    setattr(cfg, "_attention_map_capture_count", capture_count + 1)
    stem_parts = [f"{capture_count:04d}", _safe_stem(label)]
    if step_idx is not None:
        stem_parts.append(f"step_{int(step_idx):06d}")
    stem = "_".join(stem_parts)

    layers = sorted(capture["layers"], key=lambda item: int(item["layer"]))
    action_to_prefix = np.stack([_to_numpy(item["action_to_prefix"]) for item in layers], axis=0)
    overlay_paths = {}
    prefix_token_layout = capture.get("prefix_token_layout", [])
    attention_image_paths = (
        {}
        if _attention_map_video_mode(cfg)
        else _save_attention_frame_images(
            cfg,
            attention_dir,
            stem,
            prefix_token_layout,
            observation,
        )
    )
    image_segment_name_counts = {}
    for segment in prefix_token_layout:
        if not _is_attention_image_segment(segment):
            continue
        image_name = str(segment.get("name", "image"))
        image_segment_name_counts[image_name] = image_segment_name_counts.get(image_name, 0) + 1
    last_layer_prefix = action_to_prefix[-1]
    for segment in prefix_token_layout:
        if not _is_attention_image_segment(segment):
            continue
        image_name = str(segment.get("name", "image"))
        if not _attention_map_camera_matches(cfg, image_name):
            continue
        start = int(segment["start"])
        end = int(segment["end"])
        if last_layer_prefix[start:end].size == 0:
            continue

        image_rgb = _observation_image_to_rgb_uint8(_get_observation_image(observation, image_name))
        if image_rgb is not None:
            image_attention = _aggregate_image_attention(
                layers,
                start=start,
                end=end,
                token_count=int(segment.get("token_count", end - start)),
                image_shape=image_rgb.shape,
            )
            if image_attention is not None:
                image_stem = _safe_stem(image_name)
                overlay_key = image_name
                if image_segment_name_counts.get(image_name, 0) > 1:
                    segment_suffix = _safe_stem(
                        str(segment.get("stream") or segment.get("kind") or "image")
                    )
                    image_stem = f"{image_stem}_{segment_suffix}"
                    overlay_key = f"{image_name}:{segment_suffix}"
                if _attention_map_video_mode(cfg):
                    overlay_bgr = _attention_overlay_bgr(image_rgb, image_attention)
                    if overlay_bgr is None:
                        continue
                    video_info = _append_attention_video_frame(
                        cfg,
                        attention_dir,
                        f"attention_overlay_{image_stem}",
                        overlay_bgr,
                        fps=_attention_overlay_video_fps(cfg),
                    )
                    if video_info is not None:
                        overlay_paths[overlay_key] = video_info
                    continue
                overlay_path = attention_dir / f"{stem}_{image_stem}_overlay.png"
                if _write_attention_overlay_png(image_rgb, image_attention, overlay_path):
                    overlay_paths[overlay_key] = str(overlay_path)

    twogrey_coefficient_records = _extract_twogrey_coefficient_records(prefix_token_layout)
    patch_gate_map_paths = _save_patch_gate_maps(
        attention_dir,
        stem,
        prefix_token_layout,
        observation,
    )
    twogrey_coefficients_path = None
    if twogrey_coefficient_records:
        payload = {
            "label": label,
            "step_idx": int(step_idx) if step_idx is not None else None,
            "time_s": time.time(),
            "inference_elapsed_s": (
                round(float(inference_elapsed), 6) if inference_elapsed is not None else None
            ),
            "denoise_step": int(capture.get("denoise_step", -1)),
            "num_inference_steps": int(capture.get("num_inference_steps", -1)),
            "attention_map_video_mode": bool(_attention_map_video_mode(cfg)),
            "attention_capture_interval_s": float(_attention_capture_interval_s(cfg)),
            "attention_video_frame_interval_s": float(_attention_video_frame_interval_s(cfg)),
            "attention_image_paths": attention_image_paths,
            "patch_gate_map_paths": patch_gate_map_paths,
            "records": twogrey_coefficient_records,
        }
        coefficient_path = attention_dir / f"{stem}_twogrey_coefficients.json"
        coefficient_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        twogrey_coefficients_path = str(coefficient_path)

    metadata = {
        "label": label,
        "step_idx": int(step_idx) if step_idx is not None else None,
        "time_s": time.time(),
        "inference_elapsed_s": round(float(inference_elapsed), 6) if inference_elapsed is not None else None,
        "prefix_len": int(capture.get("prefix_len", action_to_prefix.shape[1])),
        "suffix_len": int(capture.get("suffix_len", 0)),
        "total_len": int(capture.get("total_len", action_to_prefix.shape[1])),
        "denoise_step": int(capture.get("denoise_step", -1)),
        "num_inference_steps": int(capture.get("num_inference_steps", -1)),
        "layers": [int(item["layer"]) for item in layers],
        "attention_map_camera": str(getattr(cfg, "attention_map_camera", "")),
        "attention_map_video_mode": bool(_attention_map_video_mode(cfg)),
        "attention_capture_interval_s": float(_attention_capture_interval_s(cfg)),
        "attention_video_frame_interval_s": float(_attention_video_frame_interval_s(cfg)),
        "attention_image_paths": attention_image_paths,
        "overlay_paths": overlay_paths,
        "patch_gate_map_paths": patch_gate_map_paths,
        "twogrey_coefficients_path": twogrey_coefficients_path,
    }
    if attention_image_paths:
        logger_mp.info(f"Saved attention-frame images {label!r}: {attention_image_paths}")
    if overlay_paths:
        logger_mp.info(f"Saved PI05 attention overlay {label!r}: {list(overlay_paths.values())}")
    else:
        logger_mp.warning(
            f"PI05 attention capture {label!r} had no overlay for camera "
            f"{getattr(cfg, 'attention_map_camera', '')!r}."
        )
    if twogrey_coefficients_path is not None:
        logger_mp.info(f"Saved image preprocessing coefficients {label!r}: {twogrey_coefficients_path}")
    if patch_gate_map_paths:
        logger_mp.info(f"Saved PatchGateViT patch maps {label!r}: {patch_gate_map_paths}")
    return metadata


def save_attention_map_capture(
    cfg: EvalRealConfig,
    policy: PreTrainedPolicy,
    *,
    label: str,
    step_idx: int | None = None,
    inference_elapsed: float | None = None,
    observation: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        return _save_attention_map_capture(
            cfg,
            policy,
            label=label,
            step_idx=step_idx,
            inference_elapsed=inference_elapsed,
            observation=observation,
        )
    except Exception as exc:
        logger_mp.warning(f"PI05 attention-map saving failed and will be skipped: {exc}")
        return None


def _resolve_repo_relative_path(path_text: str) -> str:
    path = Path(str(path_text)).expanduser()
    if path.is_absolute() or path.exists():
        return str(path)
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / path
    return str(candidate) if candidate.exists() else str(path)


def _thermal_input_type(cfg: EvalRealConfig) -> str:
    return "false"

    thermal_input_type = normalize_thermal_input_type(getattr(cfg, "thermal_input_type", "false"))
    if thermal_input_type == "false":
        policy_cfg = getattr(cfg, "policy", None)
        for attr_name in ("thermal_input_type", "thermal_encoder_channel"):
            policy_value = (
                str(getattr(policy_cfg, attr_name, "") or "")
                .strip()
                .strip("\"'“”‘’")
                .lower()
            )
            if attr_name == "thermal_input_type" and policy_value in {
                "grey",
                "twogrey",
                "twoblack",
                "twomatchingblack",
                "twofixmatchingblack",
                "matching",
                "anythermal",
                "resthermal",
            }:
                thermal_input_type = policy_value
                break
            if attr_name == "thermal_encoder_channel" and policy_value in {
                "gateoneresvit",
                "gate_one_res_vit",
                "gate_oneres_vit",
                "gate-one-res-vit",
                "gate1resvit",
                "gate_1res_vit",
            }:
                thermal_input_type = "grey"
                break
            if attr_name == "thermal_encoder_channel" and policy_value in {
                "gatevit",
                "gate_vit",
                "gate-vit",
                "gateactionvit",
                "gate_action_vit",
                "gateaction_vit",
                "gate-action-vit",
                "doublegatevit",
                "double_gate_vit",
                "doublegate_vit",
                "double-gate-vit",
                "doublegatetworesvit",
                "double_gate_two_res_vit",
                "doublegate_two_res_vit",
                "double-gate-two-res-vit",
                "doublegate2resvit",
                "double_gate_2res_vit",
                "patchgatevit",
                "patch_gate_vit",
                "patchgate_vit",
                "patch-gate-vit",
                "patchsinglegateresvit",
                "patch_single_gate_res_vit",
                "patchsingle_gate_res_vit",
                "patch-single-gate-res-vit",
                "patchsingleresvit",
                "patch_single_res_vit",
                "patch-single-res-vit",
                "smallpatchgatevit",
                "small_patch_gate_vit",
                "smallpatch_gate_vit",
                "small-patch-gate-vit",
                "smallpatchvit",
                "small_patch_vit",
                "small-patch-vit",
                "hardsmallpatchgatevit",
                "hard_small_patch_gate_vit",
                "hardsmall_patch_gate_vit",
                "hard-small-patch-gate-vit",
                "hardsmallpatchvit",
                "hard_small_patch_vit",
                "hard-small-patch-vit",
                "smallpatchgatetworesvit",
                "small_patch_gate_two_res_vit",
                "smallpatch_gate_two_res_vit",
                "small-patch-gate-two-res-vit",
                "smallpatchgate2resvit",
                "small_patch_gate_2res_vit",
                "smallpatchsinglegatetworesvit",
                "small_patch_single_gate_two_res_vit",
                "smallpatch_single_gate_two_res_vit",
                "small-patch-single-gate-two-res-vit",
                "smallpatchsinglegate2resvit",
                "small_patch_single_gate_2res_vit",
                "smallpatchgateresvit",
                "small_patch_gate_res_vit",
                "smallpatch_gate_res_vit",
                "small-patch-gate-res-vit",
                "smallpatchresvit",
                "small_patch_res_vit",
                "small-patch-res-vit",
                "smallpatchsinglegateresvit",
                "small_patch_single_gate_res_vit",
                "smallpatch_single_gate_res_vit",
                "small-patch-single-gate-res-vit",
                "smallpatchsingleresvit",
                "small_patch_single_res_vit",
                "small-patch-single-res-vit",
                "gatevitmix",
                "gate_vit_mix",
                "gatevit_mix",
                "gate-vit-mix",
                "gateresvit",
                "gate_resvit",
                "gate_res_vit",
                "gate-resvit",
                "gateresandvit",
                "gate_res_and_vit",
                "gate_resand_vit",
                "gate-res-and-vit",
                "gateresvitandvit",
                "gate_res_vit_and_vit",
                "gateresplusvit",
                "gate_res_plus_vit",
                "gatetworesvit",
                "gate_two_res_vit",
                "gate_twores_vit",
                "gate-two-res-vit",
                "gate2resvit",
                "gate_2res_vit",
                "gateresvit3",
                "gate_resvit3",
                "gate_res_vit3",
                "gate-resvit3",
                "gateresvit_3",
            }:
                thermal_input_type = "twogrey"
                break
            if policy_value in {"anythermal", "resthermal"}:
                thermal_input_type = policy_value
                break
    setattr(cfg, "thermal_input_type", thermal_input_type)
    return thermal_input_type


def _policy_is_pi05(cfg: EvalRealConfig) -> bool:
    policy_cfg = getattr(cfg, "policy", None)
    if policy_cfg is None:
        return False
    policy_type = str(getattr(policy_cfg, "type", "") or "").lower()
    policy_class = type(policy_cfg).__name__.lower()
    return policy_class == "pi05config" or policy_type == "pi05"


def configure_policy_for_thermal_input_type(cfg: EvalRealConfig) -> None:
    mode = _thermal_input_type(cfg)
    policy_cfg = getattr(cfg, "policy", None)
    if policy_cfg is None:
        return

    apply_thermal_input_type = getattr(policy_cfg, "apply_thermal_input_type", None)
    if callable(apply_thermal_input_type):
        apply_thermal_input_type(mode)

    if not hasattr(policy_cfg, "mix_rgb_thermal"):
        return
    if mode in {"matching", "twomatchingblack"} and hasattr(
        policy_cfg,
        "thermal_matching_preload",
    ):
        # RuntimeThermalInputAdapter owns MINIMA during real-robot eval. Avoid
        # loading a duplicate frozen matcher inside the policy.
        policy_cfg.thermal_matching_preload = False
    if (
        mode in {"matching", "twomatchingblack"}
        and getattr(cfg, "runtime_rgb_thermal_mix_match_every_n_frames", None) is None
    ):
        cfg.runtime_rgb_thermal_mix_match_every_n_frames = int(
            getattr(policy_cfg, "thermal_matching_match_every_n_frames", 10)
        )

    _, thermal_key, output_key = _runtime_rgb_thermal_mix_keys(cfg)
    input_features = getattr(policy_cfg, "input_features", None)
    if (
        mode
        in {
            "grey",
            "twogrey",
            "twoblack",
            "twomatchingblack",
            "twofixmatchingblack",
            "matching",
            "anythermal",
            "resthermal",
        }
        and isinstance(input_features, dict)
        and output_key in input_features
    ):
        if thermal_key not in input_features:
            input_features[thermal_key] = input_features[output_key]
        del input_features[output_key]
    if mode in {"twogrey", "twoblack", "twomatchingblack", "twofixmatchingblack"}:
        rewrite_twogrey_input_features = getattr(policy_cfg, "rewrite_twogrey_input_features", None)
        if callable(rewrite_twogrey_input_features):
            rewrite_twogrey_input_features()

    if mode == "mixing":
        if _policy_is_pi05(cfg):
            policy_cfg.mix_rgb_thermal = True
            policy_cfg.rgb_thermal_mix_source = "precomputed"
            policy_cfg.rgb_thermal_mix_use_precomputed = True
        else:
            policy_cfg.mix_rgb_thermal = False
    elif mode in {
        "false",
        "grey",
        "twogrey",
        "twoblack",
        "twomatchingblack",
        "twofixmatchingblack",
        "matching",
        "anythermal",
        "resthermal",
    }:
        policy_cfg.mix_rgb_thermal = False


def _policy_needs_runtime_rgb_thermal_mix(cfg: EvalRealConfig) -> bool:
    return (
        _thermal_input_type(cfg) == "mixing"
        and _policy_is_pi05(cfg)
        and bool(getattr(cfg, "runtime_rgb_thermal_mix", True))
    )


def _thermal_input_needs_raw_thermal(cfg: EvalRealConfig) -> bool:
    if _thermal_input_type(cfg) in {
        "grey",
        "twogrey",
        "twoblack",
        "twomatchingblack",
        "twofixmatchingblack",
        "mixing",
        "matching",
        "anythermal",
        "resthermal",
    }:
        return True
    _, thermal_key, _ = _runtime_rgb_thermal_mix_keys(cfg)
    return thermal_key in set(_selected_camera_feature_names(cfg))


def validate_thermal_input_selection(cfg: EvalRealConfig) -> None:
    mode = _thermal_input_type(cfg)
    if mode in {"mixing", "matching", "twomatchingblack", "twofixmatchingblack"} and not bool(
        getattr(cfg, "runtime_rgb_thermal_mix", True)
    ):
        raise ValueError(f"Unsupported image preprocessing mode: {mode!r}.")
    if mode == "false":
        policy_features = _policy_input_feature_names(cfg)
        _, _, output_key = _runtime_rgb_thermal_mix_keys(cfg)
        if output_key in policy_features:
            raise ValueError(
                "This checkpoint expects an additional camera preprocessing output, but this eval path "
                "only supports the configured RGB camera features."
            )
        return

    if mode in {
        "grey",
        "twogrey",
        "twoblack",
        "twomatchingblack",
        "twofixmatchingblack",
        "matching",
        "anythermal",
        "resthermal",
    }:
        _, thermal_key, output_key = _runtime_rgb_thermal_mix_keys(cfg)
        policy_features = _policy_input_feature_names(cfg)
        if output_key in policy_features and thermal_key not in policy_features:
            raise ValueError(
                f"Unsupported image preprocessing mode: {mode!r}."
            )
    if mode in {
        "grey",
        "twogrey",
        "twoblack",
        "twomatchingblack",
        "twofixmatchingblack",
        "mixing",
        "matching",
        "anythermal",
        "resthermal",
    }:
        return

    raise ValueError(f"Unsupported image preprocessing mode: {mode!r}.")


def _runtime_rgb_thermal_mix_keys(cfg: EvalRealConfig) -> tuple[str, str, str]:
    policy_cfg = getattr(cfg, "policy", None)
    rgb_key = getattr(policy_cfg, "rgb_thermal_mix_rgb_feature", "observation.images.cam_left_high")
    thermal_key = getattr(
        policy_cfg,
        "rgb_thermal_mix_thermal_feature",
        "observation.images.cam_" + "ther" + "mal",
    )
    output_key = getattr(
        policy_cfg,
        "rgb_thermal_mix_precomputed_feature",
        "observation.images.cam_rgb_thermal_mixing",
    )
    return (
        _raw_feature_name(cfg, rgb_key),
        _raw_feature_name(cfg, thermal_key),
        _raw_feature_name(cfg, output_key),
    )


def _runtime_twogrey_output_keys(cfg: EvalRealConfig) -> tuple[str, str]:
    policy_cfg = getattr(cfg, "policy", None)
    _, thermal_key, _ = _runtime_rgb_thermal_mix_keys(cfg)
    feature_names_fn = getattr(policy_cfg, "thermal_twogrey_feature_names", None)
    if callable(feature_names_fn):
        cold_key, hot_key = feature_names_fn(thermal_key)
    else:
        cold_key, hot_key = f"{thermal_key}_cold", f"{thermal_key}_hot"
    return _raw_feature_name(cfg, cold_key), _raw_feature_name(cfg, hot_key)


def _runtime_matching_interval(cfg: EvalRealConfig) -> int:
    value = getattr(cfg, "runtime_rgb_thermal_mix_match_every_n_frames", None)
    if value is None:
        value = getattr(
            getattr(cfg, "policy", None),
            "thermal_matching_match_every_n_frames",
            10,
        )
    value = int(value)
    if value <= 0:
        raise ValueError(
            "Runtime image preprocessing interval must be positive or omitted."
        )
    return value


def _is_channel_first_image(array: np.ndarray) -> bool:
    return bool(array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4})


def _image_value_to_rgb_uint8(value) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if _is_channel_first_image(image):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim != 3:
        return None
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[:, :, :3]
    elif image.shape[-1] != 3:
        return None

    image = image.astype(np.float32, copy=False)
    if not np.isfinite(image).any():
        return None
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(image.min())
    max_value = float(image.max())
    if min_value >= -1.0 and max_value <= 1.0:
        image = (image + 1.0) * 0.5 if min_value < 0.0 else image
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _format_rgb_uint8_like_reference(image_rgb: np.ndarray, reference):
    image = np.asarray(image_rgb, dtype=np.uint8)
    reference_array = reference.detach().cpu().numpy() if isinstance(reference, torch.Tensor) else np.asarray(reference)
    channel_first = _is_channel_first_image(reference_array)
    if channel_first:
        image = np.transpose(image, (2, 0, 1))

    if isinstance(reference, torch.Tensor):
        output = torch.from_numpy(np.ascontiguousarray(image))
        if reference.dtype.is_floating_point:
            output = output.to(dtype=reference.dtype) / 255.0
            ref_min = float(torch.nan_to_num(reference.detach()).min().cpu()) if reference.numel() else 0.0
            if ref_min < 0.0:
                output = output * 2.0 - 1.0
        else:
            output = output.to(dtype=reference.dtype)
        return output

    if np.issubdtype(reference_array.dtype, np.floating):
        output = image.astype(reference_array.dtype, copy=False) / 255.0
        if reference_array.size and float(np.nanmin(reference_array)) < 0.0:
            output = output * 2.0 - 1.0
        return output
    return image.astype(reference_array.dtype, copy=False)


class RuntimeThermalInputAdapter:
    def __init__(self, cfg: EvalRealConfig):
        self.mode = _thermal_input_type(cfg)
        self.rgb_key, self.thermal_key, self.output_key = _runtime_rgb_thermal_mix_keys(cfg)
        if self.mode == "matching" or not _policy_needs_runtime_rgb_thermal_mix(cfg):
            self.output_key = self.thermal_key
        self.output_keys = (self.output_key,)
        if self.mode in {
            "grey",
            "twogrey",
            "twoblack",
            "twomatchingblack",
            "twofixmatchingblack",
        }:
            policy_cfg = getattr(cfg, "policy", None)
            self.grey_background_roi = tuple(
                getattr(policy_cfg, "thermal_grey_background_roi", (0, 0, 64, 64))
            )
            self.grey_background_value = float(
                getattr(policy_cfg, "thermal_grey_background_value", 80.0)
            )
            self.black_background_outlier_threshold = float(
                getattr(policy_cfg, "thermal_black_background_outlier_threshold", 12.0)
            )
            self.black_background_outlier_ratio = float(
                getattr(policy_cfg, "thermal_black_background_outlier_ratio", 2.0)
            )
            if self.mode in {
                "twogrey",
                "twoblack",
                "twomatchingblack",
                "twofixmatchingblack",
            }:
                self.output_keys = _runtime_twogrey_output_keys(cfg)
            self._logged_first_frame = False
            if self.mode not in {"twomatchingblack", "twofixmatchingblack"}:
                return

        if self.mode == "twofixmatchingblack":
            from lerobot.policies.pi05.thermal_matching import (
                rescale_homography,
                warp_thermal_batch,
            )

            policy_cfg = getattr(cfg, "policy", None)
            policy_thermal_key = getattr(
                policy_cfg,
                "rgb_thermal_mix_thermal_feature",
                self.thermal_key,
            )
            homographies = getattr(
                policy_cfg,
                "thermal_fixed_matching_homographies",
                {},
            )
            source_sizes = getattr(
                policy_cfg,
                "thermal_fixed_matching_source_sizes",
                {},
            )
            output_sizes = getattr(
                policy_cfg,
                "thermal_fixed_matching_output_sizes",
                {},
            )
            lookup_key = (
                policy_thermal_key
                if policy_thermal_key in homographies
                else self.thermal_key
            )
            if (
                lookup_key not in homographies
                or lookup_key not in source_sizes
                or lookup_key not in output_sizes
            ):
                raise ValueError(
                    "Selected image preprocessing mode requires checkpoint data "
                    f"for {policy_thermal_key!r}. Available: "
                    f"{sorted(homographies)}."
                )
            self._fixed_homography = np.asarray(
                homographies[lookup_key],
                dtype=np.float64,
            )
            self._fixed_source_size = tuple(int(value) for value in source_sizes[lookup_key])
            self._fixed_output_size = tuple(int(value) for value in output_sizes[lookup_key])
            self._rescale_homography = rescale_homography
            self._warp_thermal_batch = warp_thermal_batch
            self.outer_padding = max(
                0,
                int(getattr(policy_cfg, "thermal_matching_outer_padding", 30)),
            )
            self.policy_cuda_stream = None
            self.frame_index = 0
            logger_mp.info(
                "Loaded checkpoint-fixed image transform: rgb=%s, auxiliary=%s, "
                "outputs=%s, stored_source_size=%s, stored_output_size=%s, "
                "outer_padding=%d.",
                self.rgb_key,
                self.thermal_key,
                self.output_keys,
                self._fixed_source_size,
                self._fixed_output_size,
                self.outer_padding,
            )
            return

        if self.mode in {"matching", "twomatchingblack"}:
            from lerobot.policies.pi05.thermal_matching import (
                MinimaThermalMatcher,
                warp_thermal_batch,
            )

            policy_cfg = getattr(cfg, "policy", None)
            minima_root = _resolve_repo_relative_path(
                getattr(
                    cfg,
                    "runtime_rgb_thermal_mix_minima_root",
                    getattr(policy_cfg, "thermal_matching_minima_root", "./MINIMA"),
                )
            )
            checkpoint = str(
                getattr(
                    cfg,
                    "runtime_rgb_thermal_mix_ckpt",
                    getattr(
                        policy_cfg,
                        "thermal_matching_checkpoint",
                        "./weights/minima_lightglue.pth",
                    ),
                )
            )
            self.fast_matcher = MinimaThermalMatcher(
                minima_root=minima_root,
                checkpoint=checkpoint,
                ransac_reproj_threshold=float(
                    getattr(cfg, "runtime_rgb_thermal_mix_ransac_reproj_threshold", 5.0)
                ),
                min_matches=int(getattr(cfg, "runtime_rgb_thermal_mix_min_matches", 4)),
                min_inliers=int(getattr(cfg, "runtime_rgb_thermal_mix_min_inliers", 4)),
                match_every_n_frames=_runtime_matching_interval(cfg),
                cache_size=int(getattr(policy_cfg, "thermal_matching_cache_size", 32768)),
                outer_padding=max(
                    0,
                    int(getattr(cfg, "runtime_rgb_thermal_mix_outer_padding", 30)),
                ),
            )
            self.match_every_n_frames = self.fast_matcher.match_every_n_frames
            self._warp_thermal_batch = warp_thermal_batch
            self._async_homography: np.ndarray | None = None
            self._async_match_lock = threading.Lock()
            self._async_match_pending = False
            self._async_match_result = None
            self._async_match_thread: threading.Thread | None = None
            self._async_update_log_count = 0
            self._realtime_schedule_started = False
            self._matching_cuda_stream = None
            self.policy_cuda_stream = None
            matcher_owner = getattr(self.fast_matcher.matcher, "__self__", None)
            matcher_device = getattr(matcher_owner, "device", None)
            if (
                isinstance(matcher_device, torch.device)
                and matcher_device.type == "cuda"
                and torch.cuda.is_available()
            ):
                least_priority, greatest_priority = torch.cuda.get_stream_priority_range()
                self._matching_cuda_stream = torch.cuda.Stream(
                    device=matcher_device,
                    priority=least_priority,
                )
                configured_policy_device = getattr(policy_cfg, "device", None)
                policy_stream_device = (
                    torch.device(configured_policy_device)
                    if configured_policy_device is not None
                    else matcher_device
                )
                self.policy_cuda_stream = torch.cuda.Stream(
                    device=policy_stream_device,
                    priority=greatest_priority,
                )
            self.frame_index = 0
            self._fallback_log_count = 0
            self._logged_first_frame = False
            logger_mp.info(
                "Loaded fast MINIMA runtime adapter: mode=%s, rgb=%s, auxiliary=%s, "
                "outputs=%s, outer_padding=%d.",
                self.mode,
                self.rgb_key,
                self.thermal_key,
                self.output_keys,
                self.fast_matcher.outer_padding,
            )
            return

        merge_module = importlib.import_module(
            "astribot_lerobot.utils.match_and_merge_rgb_and_" + "thermo" + "graphy"
        )
        estimate_thermal_to_color_homography = merge_module.estimate_thermal_to_color_homography
        load_matcher = merge_module.load_matcher
        make_mixing_image = merge_module.make_mixing_image

        self.alpha = float(getattr(cfg, "runtime_rgb_thermal_mix_alpha", 0.7))
        self.boundary_feather = int(getattr(cfg, "runtime_rgb_thermal_mix_boundary_feather", 50))
        self.thermal_border_crop = int(getattr(cfg, "runtime_rgb_thermal_mix_thermal_border_crop", 30))
        self.match_every_n_frames = _runtime_matching_interval(cfg)
        self.ransac_reproj_threshold = float(
            getattr(cfg, "runtime_rgb_thermal_mix_ransac_reproj_threshold", 5.0)
        )
        self.min_matches = int(getattr(cfg, "runtime_rgb_thermal_mix_min_matches", 4))
        self.min_inliers = int(getattr(cfg, "runtime_rgb_thermal_mix_min_inliers", 4))
        self.homography_center_ratio = float(getattr(cfg, "runtime_rgb_thermal_mix_homography_center_ratio", 1.0))
        self.center_weight_strength = float(getattr(cfg, "runtime_rgb_thermal_mix_center_weight_strength", 0.0))
        self.center_weight_sigma = float(getattr(cfg, "runtime_rgb_thermal_mix_center_weight_sigma", 0.6))
        self.previous_homography = None
        self.frame_index = 0
        self._fallback_log_count = 0
        self._logged_first_frame = False
        self._estimate_thermal_to_color_homography = estimate_thermal_to_color_homography
        self._make_mixing_image = make_mixing_image

        minima_root = _resolve_repo_relative_path(getattr(cfg, "runtime_rgb_thermal_mix_minima_root", "./MINIMA"))
        args = SimpleNamespace(
            method="sp_lg",
            minima_root=minima_root,
            ckpt=str(getattr(cfg, "runtime_rgb_thermal_mix_ckpt", "./weights/minima_lightglue.pth")),
        )
        logger_mp.info(
            "Loading runtime image adapter with MINIMA sp_lg: mode=%s, rgb=%s, auxiliary=%s, output=%s, "
            "minima_root=%s, alpha=%.3f, boundary_feather=%d, border_crop=%d.",
            self.mode,
            self.rgb_key,
            self.thermal_key,
            self.output_key,
            minima_root,
            self.alpha,
            self.boundary_feather,
            self.thermal_border_crop,
        )
        self.matcher = load_matcher(args, use_path=False)

    def _add_grey_to_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.thermal_key not in observation:
            raise RuntimeError(
                f"Selected image preprocessing mode requires {self.thermal_key!r}. "
                f"Available keys: {sorted(observation.keys())}."
            )
        thermal_value = observation[self.thermal_key]
        thermal_image = _image_value_to_rgb_uint8(thermal_value)
        if thermal_image is None:
            raise RuntimeError(
                "Thermal grayscale conversion received an unsupported image shape: "
                f"{self.thermal_key}={getattr(thermal_value, 'shape', None)}."
            )

        from lerobot.policies.pi05.thermal_utils import thermal_to_background_normalized_grey_array

        grey_rgb = thermal_to_background_normalized_grey_array(
            thermal_image,
            background_roi=self.grey_background_roi,
            target_background=self.grey_background_value,
        )
        observation = dict(observation)
        observation[self.thermal_key] = _format_rgb_uint8_like_reference(grey_rgb, thermal_value)
        if not self._logged_first_frame:
            logger_mp.info(
                "Runtime image preprocessing mode=grey replaced %s with ROI-normalized 3-channel grayscale "
                "image: ROI=%s, target_background=%.1f.",
                self.thermal_key,
                self.grey_background_roi,
                self.grey_background_value,
            )
            self._logged_first_frame = True
        return observation

    def _add_twogrey_to_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.thermal_key not in observation:
            raise RuntimeError(
                f"Selected image preprocessing mode {self.mode!r} requires {self.thermal_key!r}. "
                f"Available keys: {sorted(observation.keys())}."
            )
        thermal_value = observation[self.thermal_key]
        thermal_image = _image_value_to_rgb_uint8(thermal_value)
        if thermal_image is None:
            raise RuntimeError(
                "Thermal two-grey conversion received an unsupported image shape: "
                f"{self.thermal_key}={getattr(thermal_value, 'shape', None)}."
            )

        if self.mode == "twoblack":
            from lerobot.policies.pi05.thermal_utils import (
                thermal_to_two_black_background_normalized_grey_array,
            )

            cold_rgb, hot_rgb = thermal_to_two_black_background_normalized_grey_array(
                thermal_image,
                background_roi=self.grey_background_roi,
                target_background=self.grey_background_value,
                outlier_threshold=self.black_background_outlier_threshold,
                outlier_ratio=self.black_background_outlier_ratio,
            )
        else:
            from lerobot.policies.pi05.thermal_utils import (
                thermal_to_two_background_normalized_grey_array,
            )

            cold_rgb, hot_rgb = thermal_to_two_background_normalized_grey_array(
                thermal_image,
                background_roi=self.grey_background_roi,
                target_background=self.grey_background_value,
            )
        cold_key, hot_key = self.output_keys
        observation = dict(observation)
        observation[cold_key] = _format_rgb_uint8_like_reference(cold_rgb, thermal_value)
        observation[hot_key] = _format_rgb_uint8_like_reference(hot_rgb, thermal_value)
        if not self._logged_first_frame:
            logger_mp.info(
                "Runtime image preprocessing mode=%s split %s into cold=%s and hot=%s "
                "using ROI=%s, boundary=%.1f.",
                self.mode,
                self.thermal_key,
                cold_key,
                hot_key,
                self.grey_background_roi,
                self.grey_background_value,
            )
            self._logged_first_frame = True
        return observation

    def _add_fixed_matching_to_observation(
        self,
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        missing_keys = [key for key in (self.rgb_key, self.thermal_key) if key not in observation]
        if missing_keys:
            raise RuntimeError(
                "Selected image preprocessing mode is missing source "
                f"key(s) {missing_keys}. Available keys: {sorted(observation.keys())}."
            )

        rgb_value = observation[self.rgb_key]
        thermal_value = observation[self.thermal_key]
        rgb_image = _image_value_to_rgb_uint8(rgb_value)
        thermal_image = _image_value_to_rgb_uint8(thermal_value)
        if rgb_image is None or thermal_image is None:
            raise RuntimeError(
                "Runtime fixed thermal matching received an unsupported image shape: "
                f"{self.rgb_key}={getattr(rgb_value, 'shape', None)}, "
                f"{self.thermal_key}={getattr(thermal_value, 'shape', None)}."
            )

        thermal_tensor = (
            torch.from_numpy(np.ascontiguousarray(thermal_image))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(dtype=torch.float32)
            / 255.0
        )
        current_source_size = (int(thermal_image.shape[0]), int(thermal_image.shape[1]))
        current_output_size = (int(rgb_image.shape[0]), int(rgb_image.shape[1]))
        homography = self._rescale_homography(
            self._fixed_homography,
            stored_source_size=self._fixed_source_size,
            current_source_size=current_source_size,
            stored_output_size=self._fixed_output_size,
            current_output_size=current_output_size,
        )
        aligned, valid_alpha = self._warp_thermal_batch(
            thermal_tensor,
            homography[None],
            output_size=current_output_size,
            outer_padding=self.outer_padding,
            feature_name=self.thermal_key,
        )

        from lerobot.policies.pi05.thermal_utils import (
            estimate_thermal_black_background_tensor,
            thermal_to_two_black_background_normalized_grey_tensor,
        )

        background = estimate_thermal_black_background_tensor(
            thermal_tensor,
            feature_name=self.thermal_key,
            background_roi=self.grey_background_roi,
            outlier_threshold=self.black_background_outlier_threshold,
            outlier_ratio=self.black_background_outlier_ratio,
        )
        cold, hot = thermal_to_two_black_background_normalized_grey_tensor(
            aligned,
            feature_name=self.thermal_key,
            background_roi=self.grey_background_roi,
            target_background=self.grey_background_value,
            outlier_threshold=self.black_background_outlier_threshold,
            outlier_ratio=self.black_background_outlier_ratio,
            background=background,
            valid_alpha=valid_alpha,
        )
        cold_rgb = (
            cold[0].permute(1, 2, 0).mul(255.0).round().to(torch.uint8).cpu().numpy()
        )
        hot_rgb = (
            hot[0].permute(1, 2, 0).mul(255.0).round().to(torch.uint8).cpu().numpy()
        )
        cold_key, hot_key = self.output_keys
        observation = dict(observation)
        observation[cold_key] = _format_rgb_uint8_like_reference(cold_rgb, thermal_value)
        observation[hot_key] = _format_rgb_uint8_like_reference(hot_rgb, thermal_value)
        if not self._logged_first_frame:
            logger_mp.info(
                "Runtime image preprocessing applied the checkpoint "
                "homography at frame=%d: outputs=%s.",
                self.frame_index,
                self.output_keys,
            )
            self._logged_first_frame = True
        self.frame_index += 1
        return observation

    def _run_async_homography_update(
        self,
        rgb_image: np.ndarray,
        thermal_image: np.ndarray,
        scheduled_frame: int,
    ) -> None:
        started = time.perf_counter()
        result = None
        try:
            if self._matching_cuda_stream is None:
                homography, metrics = self.fast_matcher.estimate_homography(
                    rgb_image,
                    thermal_image,
                )
            else:
                with torch.cuda.stream(self._matching_cuda_stream):
                    homography, metrics = self.fast_matcher.estimate_homography(
                        rgb_image,
                        thermal_image,
                    )
            result = (
                homography,
                metrics,
                int(scheduled_frame),
                (time.perf_counter() - started) * 1000.0,
            )
        except Exception as error:
            logger_mp.warning(
                "Asynchronous MINIMA update failed at frame=%d; retaining the previous "
                "homography: %s",
                scheduled_frame,
                error,
            )
        finally:
            with self._async_match_lock:
                self._async_match_result = result
                self._async_match_pending = False

    def _submit_async_homography_update(
        self,
        rgb_image: np.ndarray,
        thermal_image: np.ndarray,
    ) -> bool:
        with self._async_match_lock:
            if self._async_match_pending or self._async_match_result is not None:
                return False
            self._async_match_pending = True
        thread = threading.Thread(
            target=self._run_async_homography_update,
            args=(
                np.ascontiguousarray(rgb_image).copy(),
                np.ascontiguousarray(thermal_image).copy(),
                int(self.frame_index),
            ),
            name="minima-homography-update",
            daemon=True,
        )
        self._async_match_thread = thread
        thread.start()
        return True

    def _poll_async_homography_update(self) -> bool:
        with self._async_match_lock:
            result = self._async_match_result
            self._async_match_result = None
        if result is None:
            return False

        homography, metrics, scheduled_frame, elapsed_ms = result
        self._async_homography = homography
        if self._async_update_log_count < 5:
            logger_mp.info(
                "Applied asynchronous MINIMA homography: scheduled_frame=%d, "
                "applied_frame=%d, elapsed_ms=%.1f, raw_matches=%s, inliers=%s, status=%s.",
                scheduled_frame,
                self.frame_index,
                elapsed_ms,
                metrics.get("raw_matches"),
                metrics.get("inliers"),
                metrics.get("fallback_reason") or "matched",
            )
            self._async_update_log_count += 1
        return True

    def start_realtime_schedule(self) -> None:
        """Start the formal control-loop interval without redoing the initial match."""
        if self.mode not in {"matching", "twomatchingblack"}:
            return
        self.frame_index = 0
        self._realtime_schedule_started = True
        logger_mp.info(
            "Started real-time MINIMA schedule at frame=0 with interval=%d; "
            "the precomputed startup homography is retained.",
            self.match_every_n_frames,
        )

    def prepare_live_startup_match(self) -> None:
        """Discard a dataset-warmup H so the live camera pair supplies startup H."""
        if self.mode not in {"matching", "twomatchingblack"}:
            return
        self._async_homography = None
        self.frame_index = 0
        self._realtime_schedule_started = False
        logger_mp.info(
            "Cleared the dataset-warmup MINIMA homography; the first live "
            "observation will provide the real-time startup homography."
        )

    def _add_fast_matching_to_observation(
        self,
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        missing_keys = [key for key in (self.rgb_key, self.thermal_key) if key not in observation]
        if missing_keys:
            raise RuntimeError(
                f"Runtime image preprocessing mode={self.mode!r} is missing source key(s) "
                f"{missing_keys}. Available keys: {sorted(observation.keys())}."
            )

        rgb_value = observation[self.rgb_key]
        thermal_value = observation[self.thermal_key]
        rgb_image = _image_value_to_rgb_uint8(rgb_value)
        thermal_image = _image_value_to_rgb_uint8(thermal_value)
        if rgb_image is None or thermal_image is None:
            raise RuntimeError(
                f"Runtime image preprocessing mode={self.mode!r} received an unsupported image shape: "
                f"{self.rgb_key}={getattr(rgb_value, 'shape', None)}, "
                f"{self.thermal_key}={getattr(thermal_value, 'shape', None)}."
            )

        rgb_tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb_image))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(dtype=torch.float32)
            / 255.0
        )
        thermal_tensor = (
            torch.from_numpy(np.ascontiguousarray(thermal_image))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(dtype=torch.float32)
            / 255.0
        )
        applied_async_update = self._poll_async_homography_update()
        initial_match = self._async_homography is None
        if initial_match:
            match_started = time.perf_counter()
            self._async_homography, initial_metrics = self.fast_matcher.estimate_homography(
                rgb_image,
                thermal_image,
            )
            logger_mp.info(
                "Completed initial synchronous MINIMA homography before real-time reuse: "
                "elapsed_ms=%.1f, raw_matches=%s, inliers=%s, status=%s.",
                (time.perf_counter() - match_started) * 1000.0,
                initial_metrics.get("raw_matches"),
                initial_metrics.get("inliers"),
                initial_metrics.get("fallback_reason") or "matched",
            )
        scheduled_async_update = False
        if (
            not initial_match
            and self._realtime_schedule_started
            and self.frame_index > 0
            and self.frame_index % self.match_every_n_frames == 0
        ):
            scheduled_async_update = self._submit_async_homography_update(
                rgb_image,
                thermal_image,
            )

        aligned, valid_alpha = self._warp_thermal_batch(
            thermal_tensor,
            np.asarray(self._async_homography)[None],
            output_size=(int(rgb_tensor.shape[-2]), int(rgb_tensor.shape[-1])),
            outer_padding=self.fast_matcher.outer_padding,
            feature_name=self.thermal_key,
        )

        observation = dict(observation)
        if self.mode == "matching":
            output = (aligned * valid_alpha).clamp(0.0, 1.0)
            output_rgb = (
                output[0]
                .permute(1, 2, 0)
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            observation[self.thermal_key] = _format_rgb_uint8_like_reference(
                output_rgb,
                thermal_value,
            )
        else:
            from lerobot.policies.pi05.thermal_utils import (
                estimate_thermal_black_background_tensor,
                thermal_to_two_black_background_normalized_grey_tensor,
            )

            background = estimate_thermal_black_background_tensor(
                thermal_tensor,
                feature_name=self.thermal_key,
                background_roi=self.grey_background_roi,
                outlier_threshold=self.black_background_outlier_threshold,
                outlier_ratio=self.black_background_outlier_ratio,
            )
            cold, hot = thermal_to_two_black_background_normalized_grey_tensor(
                aligned,
                feature_name=self.thermal_key,
                background_roi=self.grey_background_roi,
                target_background=self.grey_background_value,
                outlier_threshold=self.black_background_outlier_threshold,
                outlier_ratio=self.black_background_outlier_ratio,
                background=background,
                valid_alpha=valid_alpha,
            )
            cold_rgb = (
                cold[0]
                .permute(1, 2, 0)
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            hot_rgb = (
                hot[0]
                .permute(1, 2, 0)
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            cold_key, hot_key = self.output_keys
            observation[cold_key] = _format_rgb_uint8_like_reference(cold_rgb, thermal_value)
            observation[hot_key] = _format_rgb_uint8_like_reference(hot_rgb, thermal_value)

        if not self._logged_first_frame:
            logger_mp.info(
                "Runtime image preprocessing mode=%s aligned frame=%d: matches=%d, cache_hits=%d, "
                "outputs=%s, match_every_n_frames=%d, async=true.",
                self.mode,
                self.frame_index,
                int(initial_match),
                int(not initial_match),
                self.output_keys,
                self.match_every_n_frames,
            )
            self._logged_first_frame = True
        elif scheduled_async_update or applied_async_update:
            logger_mp.debug(
                "Runtime MINIMA async state: frame=%d, scheduled=%s, applied=%s.",
                self.frame_index,
                scheduled_async_update,
                applied_async_update,
            )
        self.frame_index += 1
        return observation

    def _fallback_bgr(self, color_bgr: np.ndarray, thermal_bgr: np.ndarray, metrics: dict[str, Any]) -> np.ndarray:
        if self.mode == "matching":
            metrics.setdefault("fallback_reason", "thermal_frame")
            return cv2.resize(
                thermal_bgr,
                (color_bgr.shape[1], color_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        metrics.setdefault("fallback_reason", "color_frame")
        return color_bgr

    def add_to_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.mode == "false":
            return observation
        if self.mode == "grey":
            return self._add_grey_to_observation(observation)
        if self.mode in {"twogrey", "twoblack"}:
            return self._add_twogrey_to_observation(observation)
        if self.mode == "twofixmatchingblack":
            return self._add_fixed_matching_to_observation(observation)
        if self.mode in {"matching", "twomatchingblack"}:
            return self._add_fast_matching_to_observation(observation)

        missing_keys = [key for key in (self.rgb_key, self.thermal_key) if key not in observation]
        if missing_keys:
            raise RuntimeError(
                f"Runtime image preprocessing mode={self.mode!r} cannot create "
                f"{self.output_key!r}; missing source key(s): {missing_keys}. "
                f"Available keys: {sorted(observation.keys())}."
            )

        rgb_value = observation[self.rgb_key]
        thermal_value = observation[self.thermal_key]
        rgb_image = _image_value_to_rgb_uint8(rgb_value)
        thermal_image = _image_value_to_rgb_uint8(thermal_value)
        if rgb_image is None or thermal_image is None:
            raise RuntimeError(
                f"Runtime image preprocessing mode={self.mode!r} received an unsupported image shape: "
                f"{self.rgb_key}={getattr(rgb_value, 'shape', None)}, "
                f"{self.thermal_key}={getattr(thermal_value, 'shape', None)}."
            )

        color_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        thermal_bgr = cv2.cvtColor(thermal_image, cv2.COLOR_RGB2BGR)
        metrics: dict[str, Any] = {}
        output_bgr = None
        try:
            reuse_previous = (
                self.match_every_n_frames > 1
                and self.previous_homography is not None
                and self.frame_index % self.match_every_n_frames != 0
            )
            alpha = 1.0 if self.mode == "matching" else self.alpha
            if reuse_previous:
                output_bgr, _, _, _ = self._make_mixing_image(
                    color_bgr,
                    thermal_bgr,
                    self.previous_homography,
                    alpha,
                    True,
                    self.boundary_feather,
                    self.thermal_border_crop,
                )
                metrics = {"fallback_reason": "reused_homography"}
            else:
                homography, metrics = self._estimate_thermal_to_color_homography(
                    self.matcher,
                    color_bgr,
                    thermal_bgr,
                    self.ransac_reproj_threshold,
                    self.min_matches,
                    self.min_inliers,
                    self.homography_center_ratio,
                    True,
                    self.center_weight_strength,
                    self.center_weight_sigma,
                )
                fallback_reason = ""
                if homography is None and self.previous_homography is not None:
                    homography = self.previous_homography
                    fallback_reason = "previous_homography"
                if homography is not None:
                    output_bgr, _, _, _ = self._make_mixing_image(
                        color_bgr,
                        thermal_bgr,
                        homography,
                        alpha,
                        True,
                        self.boundary_feather,
                        self.thermal_border_crop,
                    )
                    if fallback_reason == "":
                        self.previous_homography = homography
                    metrics["fallback_reason"] = fallback_reason
        except Exception as exc:
            metrics = {"fallback_reason": f"error:{exc}"}
            output_bgr = None

        if output_bgr is None:
            output_bgr = self._fallback_bgr(color_bgr, thermal_bgr, metrics)

        fallback_reason = str(metrics.get("fallback_reason") or "")
        if not self._logged_first_frame or (fallback_reason and self._fallback_log_count < 5):
            log_fn = (
                logger_mp.warning
                if fallback_reason and fallback_reason != "reused_homography"
                else logger_mp.info
            )
            log_fn(
                "Runtime image preprocessing mode=%s frame=%d status=%s raw_matches=%s inliers=%s output=%s.",
                self.mode,
                self.frame_index,
                fallback_reason or "matched",
                metrics.get("raw_matches"),
                metrics.get("inliers"),
                self.output_key,
            )
            self._logged_first_frame = True
            if fallback_reason:
                self._fallback_log_count += 1

        reference = rgb_value if self.mode == "mixing" else thermal_value
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        observation = dict(observation)
        observation[self.output_key] = _format_rgb_uint8_like_reference(output_rgb, reference)
        self.frame_index += 1
        return observation


def make_runtime_thermal_input_adapter(cfg: EvalRealConfig) -> RuntimeThermalInputAdapter | None:
    if _thermal_input_type(cfg) in {"false", "anythermal", "resthermal"}:
        return None
    return RuntimeThermalInputAdapter(cfg)


def make_runtime_rgb_thermal_mixer(cfg: EvalRealConfig) -> RuntimeThermalInputAdapter | None:
    return make_runtime_thermal_input_adapter(cfg)


def ensure_runtime_thermal_input(
    cfg: EvalRealConfig,
    observation: dict[str, Any],
    adapter: RuntimeThermalInputAdapter | None,
) -> dict[str, Any]:
    if adapter is None:
        return observation
    return adapter.add_to_observation(observation)


def _runtime_policy_stream_context(
    adapter: RuntimeThermalInputAdapter | None,
):
    stream = getattr(adapter, "policy_cuda_stream", None)
    return torch.cuda.stream(stream) if stream is not None else nullcontext()


def ensure_runtime_rgb_thermal_mix_feature(
    cfg: EvalRealConfig,
    observation: dict[str, Any],
    mixer: RuntimeThermalInputAdapter | None,
) -> dict[str, Any]:
    return ensure_runtime_thermal_input(cfg, observation, mixer)


def _has_cli_task(cfg: EvalRealConfig) -> bool:
    return bool(str(getattr(cfg, "task", "") or "").strip())


def resolve_eval_task(cfg: EvalRealConfig, step: dict[str, Any]) -> str:
    if _has_cli_task(cfg):
        return str(cfg.task)
    task = step.get("task", "")
    return "" if task is None else str(task)


def _policy_input_feature_names(cfg: EvalRealConfig) -> set[str]:
    features = getattr(getattr(cfg, "policy", None), "input_features", None)
    if not features:
        return set()
    return set(features.keys())


def _policy_output_feature_names(cfg: EvalRealConfig) -> set[str]:
    features = getattr(getattr(cfg, "policy", None), "output_features", None)
    if not features:
        return set()
    return set(features.keys())


def _policy_dataset_feature_names(cfg: EvalRealConfig) -> list[str]:
    reverse_rename_map = {renamed_key: raw_key for raw_key, renamed_key in cfg.rename_map.items()}
    feature_names = [
        reverse_rename_map.get(feature_name, feature_name)
        for feature_name in sorted(_policy_input_feature_names(cfg) | _policy_output_feature_names(cfg))
    ]
    thermal_input_type = _thermal_input_type(cfg)
    if thermal_input_type == "mixing":
        rgb_key, thermal_key, output_key = _runtime_rgb_thermal_mix_keys(cfg)
        feature_names = [feature_name for feature_name in feature_names if feature_name != output_key]
        feature_names.extend([rgb_key, thermal_key])
    elif thermal_input_type in {"matching", "twomatchingblack", "twofixmatchingblack"}:
        rgb_key, thermal_key, output_key = _runtime_rgb_thermal_mix_keys(cfg)
        derived_twogrey_features = (
            _runtime_twogrey_output_keys(cfg)
            if thermal_input_type in {"twomatchingblack", "twofixmatchingblack"}
            else ()
        )
        feature_names = [
            feature_name
            for feature_name in feature_names
            if feature_name != output_key and feature_name not in derived_twogrey_features
        ]
        feature_names.extend([rgb_key, thermal_key])
    elif thermal_input_type in {"grey", "twogrey", "twoblack", "anythermal", "resthermal"}:
        _, thermal_key, output_key = _runtime_rgb_thermal_mix_keys(cfg)
        derived_twogrey_features = (
            _runtime_twogrey_output_keys(cfg)
            if thermal_input_type in {"twogrey", "twoblack"}
            else ()
        )
        feature_names = [
            feature_name
            for feature_name in feature_names
            if feature_name != output_key and feature_name not in derived_twogrey_features
        ]
        feature_names.append(thermal_key)
    return list(dict.fromkeys(feature_names))


def _raw_feature_name(cfg: EvalRealConfig, feature_name: str) -> str:
    reverse_rename_map = {renamed_key: raw_key for raw_key, renamed_key in cfg.rename_map.items()}
    return reverse_rename_map.get(feature_name, feature_name)


def _normalize_camera_feature_name(name: str) -> str:
    name = str(name).strip()
    if not name:
        return ""
    if name.startswith("observation.images."):
        return name
    if name.startswith("images."):
        return f"observation.{name}"
    return f"observation.images.{name}"


def _cli_camera_feature_names(cfg: EvalRealConfig) -> list[str]:
    camera_names: list[str] = []
    for feature_name in getattr(cfg, "feature_names", []) or []:
        feature_name = str(feature_name).strip()
        if (
            feature_name.startswith("observation.images.")
            or feature_name.startswith("images.")
            or feature_name.startswith("cam_")
        ):
            camera_names.append(_raw_feature_name(cfg, _normalize_camera_feature_name(feature_name)))
    for feature_name in getattr(cfg, "camera_features", []) or []:
        camera_names.append(_raw_feature_name(cfg, _normalize_camera_feature_name(feature_name)))
    return [name for name in dict.fromkeys(camera_names) if name]


def _has_cli_camera_feature_selection(cfg: EvalRealConfig) -> bool:
    return bool(getattr(cfg, "feature_names", []) or getattr(cfg, "camera_features", []))


def _policy_camera_feature_names(cfg: EvalRealConfig) -> list[str]:
    feature_names = [
        _raw_feature_name(cfg, feature_name)
        for feature_name in sorted(_policy_input_feature_names(cfg))
        if feature_name.startswith("observation.images.")
    ]
    thermal_input_type = _thermal_input_type(cfg)
    rgb_key, thermal_key, output_key = _runtime_rgb_thermal_mix_keys(cfg)
    if thermal_input_type in {
        "grey",
        "twogrey",
        "twoblack",
        "anythermal",
        "resthermal",
    }:
        derived_twogrey_features = (
            _runtime_twogrey_output_keys(cfg)
            if thermal_input_type in {"twogrey", "twoblack"}
            else ()
        )
        feature_names = [
            feature_name
            for feature_name in feature_names
            if feature_name != output_key and feature_name not in derived_twogrey_features
        ]
        feature_names.append(thermal_key)
    elif thermal_input_type in {
        "mixing",
        "matching",
        "twomatchingblack",
        "twofixmatchingblack",
    }:
        rgb_key, thermal_key, output_key = _runtime_rgb_thermal_mix_keys(cfg)
        derived_twogrey_features = (
            _runtime_twogrey_output_keys(cfg)
            if thermal_input_type in {"twomatchingblack", "twofixmatchingblack"}
            else ()
        )
        feature_names = [
            feature_name
            for feature_name in feature_names
            if feature_name != output_key and feature_name not in derived_twogrey_features
        ]
        feature_names.extend([rgb_key, thermal_key])
    return list(dict.fromkeys(feature_names))


def _selected_camera_feature_names(cfg: EvalRealConfig) -> list[str]:
    if _has_cli_camera_feature_selection(cfg):
        feature_names = _cli_camera_feature_names(cfg)
    else:
        feature_names = _policy_camera_feature_names(cfg)

    thermal_input_type = _thermal_input_type(cfg)
    rgb_key, thermal_key, output_key = _runtime_rgb_thermal_mix_keys(cfg)
    if thermal_input_type in {
        "grey",
        "twogrey",
        "twoblack",
        "anythermal",
        "resthermal",
    }:
        derived_twogrey_features = (
            _runtime_twogrey_output_keys(cfg)
            if thermal_input_type in {"twogrey", "twoblack"}
            else ()
        )
        feature_names = [
            feature_name
            for feature_name in feature_names
            if feature_name != output_key and feature_name not in derived_twogrey_features
        ]
        feature_names.append(thermal_key)
    elif thermal_input_type in {
        "mixing",
        "matching",
        "twomatchingblack",
        "twofixmatchingblack",
    }:
        derived_twogrey_features = (
            _runtime_twogrey_output_keys(cfg)
            if thermal_input_type in {"twomatchingblack", "twofixmatchingblack"}
            else ()
        )
        feature_names = [
            feature_name
            for feature_name in feature_names
            if feature_name != output_key and feature_name not in derived_twogrey_features
        ]
        feature_names.extend([rgb_key, thermal_key])
    return list(dict.fromkeys(feature_names))


def validate_camera_feature_selection(cfg: EvalRealConfig) -> None:
    validate_thermal_input_selection(cfg)
    required = set(_policy_camera_feature_names(cfg))
    selected = set(_selected_camera_feature_names(cfg))
    missing = sorted(required - selected)
    if missing:
        raise RuntimeError(
            "Camera feature selection is missing checkpoint-required camera input(s): "
            f"{missing}. Selected camera features: {sorted(selected)}. "
            "Add them to --camera_features or --feature_names before running the real robot."
        )

    extra = sorted(selected - required)
    if extra:
        logger_mp.info(
            "Camera feature selection includes camera(s) not used by this checkpoint; "
            f"they will be ignored by policy validation: {extra}."
        )


def log_groot_camera_processor_alignment(
    cfg: EvalRealConfig,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
) -> None:
    if str(getattr(getattr(cfg, "policy", None), "type", "")).lower() != "groot":
        return

    pack_step = next(
        (
            step
            for step in getattr(preprocessor, "steps", ())
            if getattr(type(step), "_registry_name", None) == "groot_n1_7_pack_inputs_v1"
            or type(step).__name__ == "GrootN17PackInputsStep"
        ),
        None,
    )
    if pack_step is None:
        return

    selected = set(_selected_camera_feature_names(cfg))
    video_modality_keys = getattr(pack_step, "video_modality_keys", None)
    if not video_modality_keys:
        logger_mp.info(
            "GR00T processor has no explicit video_modality_keys; it will feed selected cameras in "
            f"alphabetical feature order: {sorted(selected)}."
        )
        return

    consumed = {f"observation.images.{key}" for key in video_modality_keys}
    used_for_mix = set()
    if bool(getattr(pack_step, "mix_rgb_thermal", False)):
        for attr_name in ("rgb_thermal_mix_rgb_feature", "rgb_thermal_mix_thermal_feature"):
            feature_name = getattr(pack_step, attr_name, None)
            if isinstance(feature_name, str):
                used_for_mix.add(feature_name)

    dropped = sorted(selected - consumed - used_for_mix)
    mix_only = sorted((selected & used_for_mix) - consumed)
    logger_mp.info(
        "GR00T processor video alignment: selected=%s, VLM_consumed=%s, mix_only=%s, dropped=%s.",
        sorted(selected),
        sorted(selected & consumed),
        mix_only,
        dropped,
    )
    if dropped:
        logger_mp.warning(
            "Some selected camera features are not consumed by the serialized GR00T processor: %s. "
            "This is safe only if the checkpoint was trained with the same processor layout.",
            dropped,
        )


def filter_observation_camera_features(cfg: EvalRealConfig, observation: dict[str, Any]) -> dict[str, Any]:
    selected = set(_selected_camera_feature_names(cfg))
    if not selected:
        return observation
    return {
        key: value
        for key, value in observation.items()
        if not key.startswith("observation.images.") or key in selected
    }


def _needs_head_camera_snapshot(cfg: EvalRealConfig) -> bool:
    selected = set(_selected_camera_feature_names(cfg))
    return bool(
        {
            "observation.images.cam_left_high",
            "observation.images.cam_right_high",
            "observation.images.head",
        }
        & selected
    )


def _selected_wrist_camera_sides(cfg: EvalRealConfig) -> tuple[bool, bool]:
    selected = set(_selected_camera_feature_names(cfg))
    return (
        bool({"observation.images.cam_left_wrist", "observation.images.left"} & selected),
        bool({"observation.images.cam_right_wrist", "observation.images.right"} & selected),
    )


def _selected_thermal_camera(cfg: EvalRealConfig) -> bool:
    return _thermal_input_needs_raw_thermal(cfg)


def _policy_requires_wrist_cameras(cfg: EvalRealConfig) -> bool:
    feature_names = set(_selected_camera_feature_names(cfg))
    return bool(
        {
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
            "observation.images.left",
            "observation.images.right",
        }
        & feature_names
    )


def save_camera_snapshot(
    image_source,
    label: str,
    camera_name: str,
    wait_timeout_s: float,
    blank_mean_threshold: float,
    save_debug_image: bool = False,
) -> Path | None:
    wait_timeout_s = max(float(wait_timeout_s), 0.0)
    blank_mean_threshold = max(float(blank_mean_threshold), 0.0)
    deadline = time.perf_counter() + wait_timeout_s
    image = np.asarray(image_source.copy())
    while image.size and float(np.mean(image)) < blank_mean_threshold and time.perf_counter() < deadline:
        time.sleep(0.05)
        image = np.asarray(image_source.copy())

    if image.size == 0:
        logger_mp.warning(f"{camera_name} snapshot skipped: empty image buffer.")
        return None
    if image.ndim != 3 or image.shape[2] != 3:
        logger_mp.warning(f"{camera_name} snapshot skipped: unexpected image shape {image.shape}.")
        return None

    mean_pixel = float(np.mean(image))
    if mean_pixel < blank_mean_threshold:
        logger_mp.warning(
            f"{camera_name} frame mean pixel value is {mean_pixel:.2f}; the frame may be blank."
        )
        return None

    if not save_debug_image:
        logger_mp.info(f"{camera_name} frame is ready with shape {image.shape}; debug image saving is disabled.")
        return Path(label)

    path = Path.cwd() / label
    image_for_write = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), image_for_write)
    if not ok:
        logger_mp.warning(f"Failed to save {camera_name} snapshot to {path}.")
        return None

    logger_mp.info(f"Saved {camera_name} snapshot to {path} with shape {image.shape}.")
    return path


def save_head_camera_snapshot(
    tv_img_array,
    label: str = "debug_head_camera_on_r.png",
    wait_timeout_s: float = 5.0,
    blank_mean_threshold: float = 1.0,
    save_debug_image: bool = False,
) -> Path | None:
    return save_camera_snapshot(
        tv_img_array,
        label,
        "Head camera",
        wait_timeout_s,
        blank_mean_threshold,
        save_debug_image=save_debug_image,
    )


def save_wrist_camera_snapshots(
    wrist_img_array,
    wrist_img_shape,
    wait_timeout_s: float = 5.0,
    blank_mean_threshold: float = 1.0,
    require_left: bool = True,
    require_right: bool = True,
    save_debug_image: bool = False,
) -> bool:
    if not require_left and not require_right:
        return True
    if wrist_img_array is None or wrist_img_shape is None:
        logger_mp.warning("Wrist camera snapshot skipped: wrist image buffer is not available.")
        return False

    half_width = int(wrist_img_shape[1]) // 2
    left_path = None
    right_path = None
    if require_left:
        left_path = save_camera_snapshot(
            wrist_img_array[:, :half_width],
            "debug_left_wrist_camera_on_r.png",
            "Left wrist camera",
            wait_timeout_s,
            blank_mean_threshold,
            save_debug_image=save_debug_image,
        )
    if require_right:
        right_path = save_camera_snapshot(
            wrist_img_array[:, half_width:],
            "debug_right_wrist_camera_on_r.png",
            "Right wrist camera",
            wait_timeout_s,
            blank_mean_threshold,
            save_debug_image=save_debug_image,
        )
    return (not require_left or left_path is not None) and (not require_right or right_path is not None)


def save_thermal_camera_snapshot(
    thermal_img_array,
    label: str = "debug_optional_camera_on_r.png",
    wait_timeout_s: float = 5.0,
    blank_mean_threshold: float = 1.0,
    save_debug_image: bool = False,
) -> Path | None:
    if thermal_img_array is None:
        logger_mp.warning("Optional camera snapshot skipped: image buffer is not available.")
        return None
    return save_camera_snapshot(
        thermal_img_array,
        label,
        "Optional camera",
        wait_timeout_s,
        blank_mean_threshold,
        save_debug_image=save_debug_image,
    )


def _camera_feature_file_name(feature_name: str) -> str:
    return feature_name.replace("observation.images.", "").replace(".", "_").replace("/", "_")


def _observation_image_to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        image = value.detach().cpu().numpy()
    else:
        image = np.asarray(value)

    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in {1, 3} and image.shape[-1] not in {1, 3}:
        image = np.transpose(image, (1, 2, 0))
    return np.ascontiguousarray(image)


def _image_for_png(image: np.ndarray) -> np.ndarray | None:
    if image.ndim == 2:
        png_image = image
    elif image.ndim == 3 and image.shape[2] == 1:
        png_image = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 3:
        png_image = image
    else:
        return None

    if png_image.dtype == np.uint8:
        return png_image
    if np.issubdtype(png_image.dtype, np.floating):
        max_value = float(np.nanmax(png_image)) if png_image.size else 0.0
        if max_value <= 1.0:
            png_image = png_image * 255.0
    return np.clip(png_image, 0, 255).astype(np.uint8)


def save_runtime_thermal_input_preview(
    cfg: EvalRealConfig,
    observation: dict[str, Any],
    adapter: RuntimeThermalInputAdapter | None,
    label: str,
) -> dict[str, Any] | None:
    mode = _thermal_input_type(cfg)
    if bool(getattr(cfg, "_thermal_input_preview_saved", False)):
        return None

    if adapter is not None:
        feature_names = list(getattr(adapter, "output_keys", (adapter.output_key,)))
    else:
        _, thermal_key, output_key = _runtime_rgb_thermal_mix_keys(cfg)
        feature_names = [output_key if _policy_needs_runtime_rgb_thermal_mix(cfg) else thermal_key]

    artifact_dir = ensure_eval_artifact_dir(cfg)
    saved_metadata = []
    missing_features = []
    for feature_name in feature_names:
        if feature_name not in observation:
            missing_features.append(feature_name)
            continue

        image = _observation_image_to_numpy(observation[feature_name])
        if len(feature_names) == 1:
            stem = f"thermal_input_{mode}_{_safe_stem(label)}"
        else:
            stem = f"thermal_input_{mode}_{_safe_stem(feature_name.split('.')[-1])}_{_safe_stem(label)}"
        npy_path = artifact_dir / f"{stem}.npy"
        png_path = artifact_dir / f"{stem}.png"
        metadata_path = artifact_dir / f"{stem}.json"

        np.save(npy_path, image)
        png_image = _image_for_png(image)
        png_ok = False
        if png_image is not None:
            if png_image.ndim == 3 and png_image.shape[2] == 3:
                png_image = cv2.cvtColor(png_image, cv2.COLOR_RGB2BGR)
            png_ok = bool(cv2.imwrite(str(png_path), png_image))
        if not png_ok:
            logger_mp.warning("Image preprocessing preview PNG could not be written; raw array saved to %s.", npy_path)

        metadata = {
            "label": label,
            "image_input_mode": mode,
            "feature": feature_name,
            "time_s": time.time(),
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "mean": float(np.mean(image)) if image.size else None,
            "min": float(np.min(image)) if image.size else None,
            "max": float(np.max(image)) if image.size else None,
            "npy_path": str(npy_path),
            "png_path": str(png_path) if png_ok else None,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        saved_metadata.append(metadata)

    if not saved_metadata:
        if mode == "false":
            return None
        logger_mp.warning(
            "Image preprocessing preview skipped: feature(s) %s are not available after runtime adapter. "
            "Available keys: %s.",
            missing_features or feature_names,
            sorted(observation.keys()),
        )
        return None

    if missing_features:
        logger_mp.warning(
            "Image preprocessing preview saved partially; missing feature(s): %s. Available keys: %s.",
            missing_features,
            sorted(observation.keys()),
        )

    metadata = saved_metadata[0]
    if len(saved_metadata) > 1:
        summary_path = artifact_dir / f"thermal_input_{mode}_{_safe_stem(label)}.json"
        metadata = {
            "label": label,
            "image_input_mode": mode,
            "time_s": time.time(),
            "features": saved_metadata,
        }
        summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    setattr(cfg, "_thermal_input_preview_saved", True)
    logger_mp.info(
        "Saved image preprocessing preview for mode=%s feature=%s to %s",
        mode,
        [item["feature"] for item in saved_metadata],
        [item["png_path"] or item["npy_path"] for item in saved_metadata],
    )
    return metadata


def save_camera_feature_snapshot_on_s(cfg: EvalRealConfig, observation: dict[str, Any]) -> None:
    if not cfg.save_camera_features_on_s:
        return

    selected_features = _selected_camera_feature_names(cfg)
    if _policy_needs_runtime_rgb_thermal_mix(cfg):
        _, _, output_key = _runtime_rgb_thermal_mix_keys(cfg)
        if output_key in observation and output_key not in selected_features:
            selected_features.append(output_key)
    if not selected_features:
        return

    snapshot_dir = Path(cfg.camera_feature_snapshot_dir)
    if not snapshot_dir.is_absolute():
        snapshot_dir = Path.cwd() / snapshot_dir
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    metadata: dict[str, Any] = {
        "created_time_s": time.time(),
        "selected_camera_features": selected_features,
        "saved": [],
        "missing": [],
    }

    for feature_name in selected_features:
        if feature_name not in observation:
            metadata["missing"].append(feature_name)
            logger_mp.warning(f"Start-frame snapshot missing selected camera feature: {feature_name}")
            continue

        image = _observation_image_to_numpy(observation[feature_name])
        file_stem = f"on_s_{timestamp}_{_camera_feature_file_name(feature_name)}"
        npy_path = snapshot_dir / f"{file_stem}.npy"
        png_path = snapshot_dir / f"{file_stem}.png"
        np.save(npy_path, image)

        png_image = _image_for_png(image)
        if png_image is not None and png_image.ndim == 3 and png_image.shape[2] == 3:
            png_image = cv2.cvtColor(png_image, cv2.COLOR_RGB2BGR)
        png_ok = bool(png_image is not None and cv2.imwrite(str(png_path), png_image))
        if not png_ok:
            logger_mp.warning(f"Could not write PNG for {feature_name}; raw array saved to {npy_path}.")

        metadata["saved"].append(
            {
                "feature": feature_name,
                "shape": list(image.shape),
                "dtype": str(image.dtype),
                "mean": float(np.mean(image)) if image.size else None,
                "min": float(np.min(image)) if image.size else None,
                "max": float(np.max(image)) if image.size else None,
                "npy_path": str(npy_path),
                "png_path": str(png_path) if png_ok else None,
            }
        )

    metadata_path = snapshot_dir / f"on_s_{timestamp}_camera_features.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger_mp.info(f"Saved selected camera feature start-frame snapshot metadata to {metadata_path}")


def enable_required_camera_inputs(cfg: EvalRealConfig) -> None:
    if _policy_requires_wrist_cameras(cfg) and not cfg.use_wrist_cameras:
        cfg.use_wrist_cameras = True
        logger_mp.info("Selected camera features include wrist cameras; enabling --use_wrist_cameras automatically.")


def _policy_required_real_observation_keys(cfg: EvalRealConfig) -> list[str]:
    feature_names = _policy_input_feature_names(cfg)
    if not feature_names:
        return []

    reverse_rename_map = {renamed_key: raw_key for raw_key, renamed_key in cfg.rename_map.items()}
    required_keys = [
        reverse_rename_map.get(feature_name, feature_name)
        for feature_name in sorted(feature_names)
        if feature_name.startswith("observation.")
    ]
    return list(dict.fromkeys(required_keys))


def validate_real_observation_keys(cfg: EvalRealConfig, observation: dict[str, Any], context: str) -> None:
    required_keys = _policy_required_real_observation_keys(cfg)
    missing_keys = [key for key in required_keys if key not in observation]
    if not missing_keys:
        return

    hint = ""
    if any("wrist" in key or key.endswith(".left") or key.endswith(".right") for key in missing_keys):
        hint = (
            " This policy expects left/right wrist camera inputs; make sure the wrist camera streams "
            "are running and --use_wrist_cameras is enabled."
        )
    raise RuntimeError(
        f"{context}: real observation is missing policy input key(s): {missing_keys}. "
        f"Available keys: {sorted(observation.keys())}.{hint}"
    )


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


def _consume_manual_snapshot_requests(
    key_listener: KeyCommandListener | None,
    snapshot_callback: Callable[[str], None] | None,
    label: str,
) -> None:
    if key_listener is None or snapshot_callback is None:
        return
    if key_listener.shutdown_requested:
        return
    while key_listener.consume("j") is not None:
        if key_listener.shutdown_requested:
            return
        snapshot_callback(label)


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
    settle_tolerance: float | None = None,
    snapshot_callback: Callable[[str], None] | None = None,
) -> bool:
    target_arm_q = require_finite_vector("target arm q", target_arm_q)
    current_arm_q = require_finite_vector("current arm q", arm_ctrl.get_current_dual_arm_q(), target_arm_q.shape[0])
    settle_tolerance = float(cfg.init_tolerance if settle_tolerance is None else settle_tolerance)
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
        _consume_manual_snapshot_requests(key_listener, snapshot_callback, f"{label}_moving")
        if allow_q_interrupt and key_listener is not None and key_listener.shutdown_requested:
            logger_mp.info(f"Movement to {label} interrupted by q.")
            return False

        tracking_wait_started_at = None
        while step_idx > 1:
            _consume_manual_snapshot_requests(key_listener, snapshot_callback, f"{label}_tracking_wait")
            current_arm_q = require_finite_vector("current arm q", arm_ctrl.get_current_dual_arm_q(), target_arm_q.shape[0])
            tracking_err = np.abs(last_arm_cmd_q - current_arm_q)
            max_tracking_err = float(np.max(tracking_err))
            max_tracking_joint = int(np.argmax(tracking_err))

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
                        snapshot_callback=snapshot_callback,
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

        current_arm_q = require_finite_vector("current arm q", arm_ctrl.get_current_dual_arm_q(), target_arm_q.shape[0])
        tracking_err = np.abs(arm_cmd_q - current_arm_q)
        max_tracking_err = float(np.max(tracking_err))
        max_tracking_joint = int(np.argmax(tracking_err))
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
        _consume_manual_snapshot_requests(key_listener, snapshot_callback, f"{label}_settling")
        if allow_q_interrupt and key_listener is not None and key_listener.shutdown_requested:
            logger_mp.info(f"Movement to {label} interrupted by q.")
            return False

        current_arm_q = require_finite_vector("current arm q", arm_ctrl.get_current_dual_arm_q(), target_arm_q.shape[0])
        err = np.abs(target_arm_q - current_arm_q)
        max_err = float(np.max(err))
        if max_err <= settle_tolerance:
            logger_mp.info(f"Reached {label}; max joint error {max_err:.4f} rad.")
            return True

        now = time.perf_counter()
        elapsed = now - start_time
        if cfg.init_timeout_s > 0 and elapsed > cfg.init_timeout_s:
            logger_mp.info(
                f"Timed out while moving to {label}; max joint error is still {max_err:.4f} rad. "
                f"Settle tolerance is {settle_tolerance:.4f} rad. Evaluation will not start."
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
    snapshot_callback: Callable[[str], None] | None = None,
) -> bool:
    if ee_dof <= 0 or not target_ee_state.size:
        return True

    target_ee_state = require_finite_vector("target ee state", target_ee_state)
    if target_ee_state.shape[0] != 2 * ee_dof:
        raise ValueError(f"{label} target ee state length {target_ee_state.shape[0]} != {2 * ee_dof}.")

    with ee_shared_mem["lock"]:
        current_ee_state = require_finite_vector(
            "current ee state",
            ee_shared_mem["state"][:],
            2 * ee_dof,
        )
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
        _consume_manual_snapshot_requests(key_listener, snapshot_callback, f"{label}_moving")
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
        final_ee_state = require_finite_vector("final ee state", ee_shared_mem["state"][:], 2 * ee_dof)
    final_error = target_ee_state - final_ee_state
    max_final_error = float(np.max(np.abs(final_error))) if final_error.size else 0.0
    if max_final_error > float(cfg.ee_tracking_tolerance):
        logger_mp.warning(
            f"{label}: final finger tracking error {max_final_error:.4f} rad exceeds "
            f"ee_tracking_tolerance {cfg.ee_tracking_tolerance:.4f} rad; "
            "hand command may not be reaching the controller."
        )
        return bool(not cfg.abort_on_ee_tracking_error)

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

    step = dataset[_first_eval_dataset_index(dataset)]
    observation = extract_observation(step)
    observation = ensure_runtime_thermal_input(cfg, observation, runtime_thermal_input_adapter)
    task = resolve_eval_task(cfg, step)

    logger_mp.info(f"Warming up policy for {warmup_steps} step(s) before connecting to robot.")
    start_time = time.perf_counter()
    for _ in range(warmup_steps):
        with _runtime_policy_stream_context(runtime_thermal_input_adapter):
            _ = predict_action(
                observation,
                policy,
                device,
                preprocessor,
                postprocessor,
                policy.config.use_amp,
                task,
                use_dataset=True,
                robot_type=cfg.ee,
            )
    policy_stream = getattr(runtime_thermal_input_adapter, "policy_cuda_stream", None)
    if policy_stream is not None:
        policy_stream.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    logger_mp.info(f"Policy warmup finished in {elapsed:.2f}s.")

    policy.reset()
    clear_cached_eval_actions(policy)
    preprocessor.reset()
    postprocessor.reset()


def _dataset_action_dim(step: dict[str, Any]) -> int | None:
    if "action" not in step:
        return None
    action = step["action"]
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    action = np.asarray(action)
    if action.size == 0:
        return None
    return int(action.reshape(-1).shape[0])


def _first_eval_dataset_index(dataset: LeRobotDataset) -> int:
    return 0 if getattr(dataset, "episodes", None) is not None else int(dataset.meta.episodes["dataset_from_index"][0])


def _sample_eval_dataset_episodes(cfg: EvalRealConfig) -> list[int] | None:
    ratio = float(getattr(cfg, "dataset_sample_ratio", 1.0))
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"dataset_sample_ratio must be in (0, 1], got {ratio}.")
    if ratio >= 1.0:
        return None

    total_episodes = None
    if cfg.root:
        info_path = Path(cfg.root).expanduser() / "meta" / "info.json"
        if info_path.is_file():
            with info_path.open("r", encoding="utf-8") as f:
                total_episodes = int(json.load(f)["total_episodes"])
    if total_episodes is None:
        meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.root or None)
        total_episodes = int(meta.total_episodes)
    if total_episodes <= 0:
        raise RuntimeError("Cannot sample eval dataset episodes because the dataset has no episodes.")

    sample_count = max(1, int(np.ceil(total_episodes * ratio)))
    rng = np.random.default_rng(int(getattr(cfg, "dataset_sample_seed", 0)))
    episodes = sorted(int(ep) for ep in rng.choice(total_episodes, size=sample_count, replace=False))
    logging.info(
        f"Using eval dataset episode sample: ratio={ratio:.4f}, seed={cfg.dataset_sample_seed}, "
        f"episodes={sample_count}/{total_episodes}, selected={episodes}."
    )
    return episodes


def _validate_eval_test_action(
    cfg: EvalRealConfig,
    action: torch.Tensor,
    step: dict[str, Any],
    step_idx: int,
    inference_elapsed: float,
) -> np.ndarray:
    if cfg.policy_inference_timeout_s > 0 and inference_elapsed > cfg.policy_inference_timeout_s:
        raise RuntimeError(
            f"Dry-run policy step {step_idx} took {inference_elapsed:.2f}s, exceeding "
            f"policy_inference_timeout_s {cfg.policy_inference_timeout_s:.2f}s."
        )

    action_np = np.asarray(action.detach().cpu().numpy()).reshape(-1)
    action_np = require_finite_vector(f"dry-run policy action step {step_idx}", action_np)

    dataset_action_dim = _dataset_action_dim(step)
    policy_action_dim = _policy_feature_dim(getattr(cfg.policy, "output_features", None), "action")
    expected_dims = {dim for dim in (dataset_action_dim, policy_action_dim) if dim is not None}
    if expected_dims and action_np.shape[0] not in expected_dims:
        raise RuntimeError(
            f"Dry-run policy action dim {action_np.shape[0]} does not match expected dim(s) "
            f"{sorted(expected_dims)} from dataset/policy metadata."
        )
    return action_np


def run_policy_dataset_dry_run(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    device: torch.device,
    runtime_thermal_input_adapter: RuntimeThermalInputAdapter | None = None,
) -> None:
    """Safely test policy inference without connecting to or commanding the robot."""
    dry_run_steps = max(1, int(cfg.policy_warmup_steps))
    if len(dataset) <= 0:
        raise RuntimeError("Dataset dry-run requires at least one episode.")
    episode_starts = (
        list(range(min(dry_run_steps, len(dataset))))
        if getattr(dataset, "episodes", None) is not None
        else list(dataset.meta.episodes["dataset_from_index"])
    )
    if not episode_starts:
        raise RuntimeError("Dataset dry-run requires at least one episode.")

    logger_mp.warning(
        "--send_real_robot=false: running dataset-only policy dry-run. "
        "No image client, robot interface, arm command, or hand command will be created."
    )
    logger_mp.info(
        f"Dry-run policy type={getattr(cfg.policy, 'type', None)!r}; "
        f"steps={dry_run_steps}; "
        f"selected camera features={_selected_camera_feature_names(cfg)}."
    )
    validate_camera_feature_selection(cfg)

    policy.reset()
    clear_cached_eval_actions(policy)
    preprocessor.reset()
    postprocessor.reset()

    for step_idx in range(dry_run_steps):
        dataset_idx = int(episode_starts[step_idx % len(episode_starts)])
        step = dataset[dataset_idx]
        observation = extract_observation(step)
        observation = filter_observation_camera_features(cfg, observation)
        observation = ensure_runtime_thermal_input(cfg, observation, runtime_thermal_input_adapter)
        validate_real_observation_keys(cfg, observation, f"Dry-run dataset step {step_idx}")
        task = resolve_eval_task(cfg, step)

        inference_start = time.perf_counter()
        capture_attention_map = (
            _attention_map_enabled(cfg)
            and bool(getattr(cfg, "attention_map_warmup", False))
            and step_idx == 0
        )
        with _runtime_policy_stream_context(runtime_thermal_input_adapter):
            action = predict_action(
                observation,
                policy,
                device,
                preprocessor,
                postprocessor,
                policy.config.use_amp,
                task,
                use_dataset=True,
                robot_type=cfg.ee,
                capture_attention_map=capture_attention_map,
            )
        if capture_attention_map:
            save_attention_map_capture(
                cfg,
                policy,
                label="dataset_dry_run",
                step_idx=step_idx,
                observation=observation,
            )
        policy_stream = getattr(runtime_thermal_input_adapter, "policy_cuda_stream", None)
        if policy_stream is not None:
            policy_stream.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_elapsed = time.perf_counter() - inference_start
        action_np = _validate_eval_test_action(cfg, action, step, step_idx, inference_elapsed)
        logger_mp.info(
            f"Dry-run step {step_idx}: dataset_idx={dataset_idx}, inference={inference_elapsed:.3f}s, "
            f"action_dim={action_np.shape[0]}, action_abs_max={_max_abs(action_np):.4f}."
        )

    policy.reset()
    clear_cached_eval_actions(policy)
    preprocessor.reset()
    postprocessor.reset()
    logger_mp.info("Dataset-only policy dry-run completed successfully.")


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
        current_arm_q = require_finite_vector("current arm q", current_arm_q, arm_dof)
        left_ee_state = right_ee_state = np.array([])
        if cfg.ee:
            with ee_shared_mem["lock"]:
                full_state = require_finite_vector("current ee state", ee_shared_mem["state"][:], 2 * ee_dof)
                left_ee_state = full_state[:ee_dof]
                right_ee_state = full_state[ee_dof:]
        observation["observation.state"] = torch.from_numpy(
            np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
        ).float()
        validate_real_observation_keys(cfg, observation, "Real-observation warmup")

        capture_attention_map = (
            _attention_map_enabled(cfg)
            and bool(getattr(cfg, "attention_map_warmup", False))
            and warmup_idx == warmup_steps - 1
        )
        with _runtime_policy_stream_context(runtime_thermal_input_adapter):
            _ = predict_action(
                observation,
                policy,
                device,
                preprocessor,
                postprocessor,
                policy.config.use_amp,
                task,
                use_dataset=cfg.use_dataset,
                robot_type=cfg.ee,
                capture_attention_map=capture_attention_map,
            )
        if capture_attention_map:
            save_attention_map_capture(
                cfg,
                policy,
                label="real_warmup",
                step_idx=warmup_idx,
                observation=observation,
            )
    policy_stream = getattr(runtime_thermal_input_adapter, "policy_cuda_stream", None)
    if policy_stream is not None:
        policy_stream.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    logger_mp.info(f"Real-observation policy warmup finished in {elapsed:.2f}s.")

    policy.reset()
    clear_cached_eval_actions(policy)
    preprocessor.reset()
    postprocessor.reset()


def eval_policy(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    runtime_thermal_input_adapter: RuntimeThermalInputAdapter | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")
    if not cfg.send_real_robot:
        run_policy_dataset_dry_run(
            cfg,
            dataset,
            policy,
            preprocessor,
            postprocessor,
            get_safe_torch_device(policy.config.device),
            runtime_thermal_input_adapter,
        )
        return

    enable_required_camera_inputs(cfg)
    validate_camera_feature_selection(cfg)
    logger_mp.info(
        f"Selected camera features for real eval: {_selected_camera_feature_names(cfg)}."
    )

    if cfg.visualization:
        rerun_logger_cls, _ = require_rerun_visualization()
        rerun_logger = rerun_logger_cls()

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        clear_cached_eval_actions(policy)
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
        if _policy_requires_wrist_cameras(cfg) and not image_info.get("has_wrist_cam", False):
            raise RuntimeError(
                "Selected camera features require wrist cameras, but the image client did not enable wrist buffers. "
                "Use --use_wrist_cameras=true or keep automatic camera enabling active."
            )
        if _selected_thermal_camera(cfg) and not image_info.get("has_thermal_cam", False):
            raise RuntimeError(
                "Selected camera features include an unsupported input. Use head/left/right RGB camera features."
            )
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
        step = dataset[_first_eval_dataset_index(dataset)]
        eval_task = resolve_eval_task(cfg, step)
        task_source = "--task" if _has_cli_task(cfg) else "dataset first frame"
        logger_mp.info(f"Policy task prompt ({task_source}): {eval_task!r}")
        init_arm_pose = require_finite_vector(
            "dataset first frame arm pose",
            step["observation.state"][:arm_dof].cpu().numpy(),
            arm_dof,
        )
        init_ee_pose = require_finite_vector(
            "dataset first frame ee pose",
            step["observation.state"][arm_dof : arm_dof + 2 * ee_dof].cpu().numpy(),
            2 * ee_dof,
        )
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
        if str(getattr(cfg.policy, "type", "")).lower() == "groot":
            logger_mp.info(
                "GR00T action execution cadence: policy n_action_steps=%s, chunk_size=%s, "
                "resolved queued action steps=%s. For more reactive real-robot eval, try "
                "--policy.n_action_steps=1.",
                getattr(cfg.policy, "n_action_steps", None),
                getattr(cfg.policy, "chunk_size", None),
                getattr(policy, "_action_queue_steps", None),
            )

        key_listener.start()
        logger_mp.info(
            "Keyboard controls: r=move to dataset first frame, s=start policy, "
            "j=save current camera snapshot, b=back to dataset first frame after s, "
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

        policy_device = get_safe_torch_device(policy.config.device)

        def warmup_real_policy(label: str) -> None:
            logger_mp.info("Preparing real-observation policy warmup (%s).", label)
            if runtime_thermal_input_adapter is not None:
                runtime_thermal_input_adapter.prepare_live_startup_match()
            warmup_policy_from_real_observation(
                cfg,
                policy,
                preprocessor,
                postprocessor,
                policy_device,
                eval_task,
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

        warmup_real_policy("initial")

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
        full_state = None

        def capture_manual_snapshot(label: str) -> None:
            try:
                snapshot_observation, _ = process_images_and_observations(
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
                snapshot_observation = filter_observation_camera_features(cfg, snapshot_observation)
                save_manual_camera_snapshot(
                    cfg,
                    snapshot_observation,
                    label=label,
                    step_idx=idx,
                )
            except Exception as exc:
                logger_mp.warning("Manual snapshot failed for %s: %s", label, exc)

        def drain_pending_key(key: str) -> None:
            while key_listener.consume(key) is not None:
                pass

        def move_to_dataset_first_frame(label: str) -> bool:
            logger_mp.info(f"{label}: preparing to move robot to dataset starting pose...")
            if not move_arm_to_pose_slowly(
                cfg,
                arm_ctrl,
                arm_ik,
                init_arm_pose,
                "dataset first frame",
                key_listener=key_listener,
                settle_tolerance=cfg.dataset_start_tolerance,
                snapshot_callback=capture_manual_snapshot,
            ):
                logger_mp.info(f"{label}: dataset first frame arm move did not complete.")
                return False
            if cfg.ee and not move_ee_to_pose_slowly(
                cfg,
                ee_shared_mem,
                ee_dof,
                init_ee_pose,
                "dataset first frame hands",
                key_listener=key_listener,
                snapshot_callback=capture_manual_snapshot,
            ):
                logger_mp.info(f"{label}: dataset first frame hand move did not complete.")
                return False
            return True

        def wait_for_policy_start(snapshot_label: str) -> bool:
            logger_mp.info(
                "At dataset first frame. Press 's' to start policy, 'j' to save a current camera snapshot, "
                "'b' is ignored until policy is running, or 'q' to open hands and lower."
            )
            while True:
                key = key_listener.wait_for("s", "q", "j", "b")
                if key == "q":
                    return False
                if key == "j":
                    capture_manual_snapshot(snapshot_label)
                    continue
                if key == "b":
                    logger_mp.info("b pressed before policy start; already at dataset first frame.")
                    continue
                break

            drain_pending_key("b")
            start_observation, _ = process_images_and_observations(
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
            start_observation = filter_observation_camera_features(cfg, start_observation)
            start_observation = ensure_runtime_thermal_input(cfg, start_observation, runtime_thermal_input_adapter)
            save_runtime_thermal_input_preview(
                cfg,
                start_observation,
                runtime_thermal_input_adapter,
                label="on_s_first_frame",
            )
            save_camera_feature_snapshot_on_s(cfg, start_observation)
            if runtime_thermal_input_adapter is not None:
                runtime_thermal_input_adapter.start_realtime_schedule()

            policy.reset()
            clear_cached_eval_actions(policy)
            preprocessor.reset()
            postprocessor.reset()
            logger_mp.info("Policy and processors reset immediately before starting/restarting the evaluation loop.")
            return True

        logger_mp.info(
            "Ready. Press 'r' to slowly move to the dataset first frame, "
            "'j' to save a current camera snapshot, or 'q' to open hands and lower."
        )
        while True:
            key = key_listener.wait_for("r", "q", "j", "b")
            if key == "q":
                shutdown_done = shutdown_robot()
                return
            if key == "j":
                capture_manual_snapshot("pre_r")
                continue
            if key == "b":
                logger_mp.info("b pressed before policy start; ignoring.")
                continue
            break

        if _needs_head_camera_snapshot(cfg):
            head_snapshot_path = save_head_camera_snapshot(
                tv_img_array,
                wait_timeout_s=cfg.head_camera_wait_timeout_s,
                blank_mean_threshold=cfg.head_camera_blank_mean_threshold,
                save_debug_image=cfg.save_camera_debug_on_r,
            )
            if cfg.require_head_camera_on_r and head_snapshot_path is None:
                logger_mp.error(
                    "Head camera did not produce a valid frame after "
                    f"{cfg.head_camera_wait_timeout_s:.1f}s; refusing to continue blind policy execution."
                )
                key_listener.shutdown_requested = True
                shutdown_done = shutdown_robot()
                return
        else:
            logger_mp.info("No selected head camera features; skipping head camera snapshot requirement.")

        require_left_wrist, require_right_wrist = _selected_wrist_camera_sides(cfg)
        if require_left_wrist or require_right_wrist:
            wrists_ready = save_wrist_camera_snapshots(
                wrist_img_array,
                wrist_img_shape,
                wait_timeout_s=cfg.wrist_camera_wait_timeout_s,
                blank_mean_threshold=cfg.wrist_camera_blank_mean_threshold,
                require_left=require_left_wrist,
                require_right=require_right_wrist,
                save_debug_image=cfg.save_camera_debug_on_r,
            )
            if cfg.require_wrist_cameras_on_r and not wrists_ready:
                wrist_labels = [
                    label
                    for label, required in [
                        ("left wrist", require_left_wrist),
                        ("right wrist", require_right_wrist),
                    ]
                    if required
                ]
                logger_mp.error(
                    f"Selected wrist camera(s) {wrist_labels} did not produce valid frames after "
                    f"{cfg.wrist_camera_wait_timeout_s:.1f}s; refusing to continue blind policy execution."
                )
                key_listener.shutdown_requested = True
                shutdown_done = shutdown_robot()
                return

        if _selected_thermal_camera(cfg):
            thermal_wait_timeout_s = float(getattr(cfg, "thermal_camera_wait_timeout_s", 5.0))
            thermal_blank_mean_threshold = float(getattr(cfg, "thermal_camera_blank_mean_threshold", 1.0))
            thermal_snapshot_path = save_thermal_camera_snapshot(
                thermal_img_array,
                wait_timeout_s=thermal_wait_timeout_s,
                blank_mean_threshold=thermal_blank_mean_threshold,
                save_debug_image=cfg.save_camera_debug_on_r,
            )
            if bool(getattr(cfg, "require_thermal_camera_on_r", True)) and thermal_snapshot_path is None:
                logger_mp.error(
                    "Selected optional camera did not produce a valid frame after "
                    f"{thermal_wait_timeout_s:.1f}s; refusing to continue blind policy execution."
                )
                key_listener.shutdown_requested = True
                shutdown_done = shutdown_robot()
                return

        # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
        if not move_to_dataset_first_frame("Initial start"):
            logger_mp.info("Dataset first frame move did not complete; starting shutdown sequence.")
            shutdown_done = shutdown_robot()
            return
        if not wait_for_policy_start("pre_start"):
            shutdown_done = shutdown_robot()
            return

        # --- Run Main Loop ---
        logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
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
                    "task": eval_task,
                    "task_source": task_source,
                    "selected_camera_features": _selected_camera_feature_names(cfg),
                    "image_input_mode": _thermal_input_type(cfg),
                    "dataset_task": step["task"],
                    "frequency": float(cfg.frequency),
                    "arm_dof": int(arm_dof),
                    "ee_dof": int(ee_dof),
                    "max_policy_arm_delta": float(cfg.max_policy_arm_delta),
                    "max_policy_ee_delta": float(cfg.max_policy_ee_delta),
                    "limit_policy_delta_from_last_command": bool(cfg.limit_policy_delta_from_last_command),
                    "ee_tracking_tolerance": float(cfg.ee_tracking_tolerance),
                    "abort_on_ee_tracking_error": bool(cfg.abort_on_ee_tracking_error),
                    "save_attention_maps": bool(cfg.save_attention_maps),
                    "save_attention_images": bool(
                        getattr(cfg, "save_attention_images", False)
                    ),
                    "attention_map_interval_s": float(cfg.attention_map_interval_s),
                    "attention_map_video_mode": bool(_attention_map_video_mode(cfg)),
                    "attention_capture_interval_s": float(_attention_capture_interval_s(cfg)),
                    "attention_video_frame_interval_s": float(
                        _attention_video_frame_interval_s(cfg)
                    ),
                    "attention_video_fps": float(_attention_video_fps(cfg)),
                    "attention_overlay_video_fps": float(_attention_overlay_video_fps(cfg)),
                    "attention_map_dir": getattr(cfg, "_attention_map_dir", None),
                    "policy_action_dim": int(policy_action_dim) if policy_action_dim is not None else None,
                    "supports_compressed_dex3_action": bool(supports_compressed_dex3_action),
                    "compressed_dex3_action_dim": int(compressed_dex3_action_dim),
                    "groot_action_queue_steps": int(getattr(policy, "_action_queue_steps", 0) or 0)
                    if str(getattr(cfg.policy, "type", "")).lower() == "groot"
                    else None,
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
        attention_interval_s = _attention_capture_interval_s(cfg)
        attention_capture_enabled = _attention_map_enabled(cfg) and attention_interval_s > 0.0
        attention_video_frame_interval_s = _attention_video_frame_interval_s(cfg)
        attention_video_frame_enabled = (
            attention_capture_enabled
            and _attention_map_video_mode(cfg)
            and bool(getattr(cfg, "save_attention_images", False))
            and attention_video_frame_interval_s > 0.0
        )
        if attention_capture_enabled and _attention_map_video_mode(cfg):
            logger_mp.info(
                f"Attention video mode enabled: camera frame interval "
                f"{attention_video_frame_interval_s:.3f}s, frame-video fps "
                f"{_attention_video_fps(cfg):.3f}; attention capture interval "
                f"{attention_interval_s:.3f}s, attention-video fps "
                f"{_attention_overlay_video_fps(cfg):.3f}."
            )
        next_attention_map_time = time.perf_counter()
        next_attention_video_frame_time = next_attention_map_time
        attention_capture_pending = False
        prev_raw_action_np = None
        prev_sent_action_np = None

        def handle_back_request(trigger: str) -> bool:
            nonlocal prev_raw_action_np
            nonlocal prev_sent_action_np
            nonlocal next_policy_log_time
            nonlocal next_attention_map_time
            nonlocal next_attention_video_frame_time
            nonlocal attention_capture_pending

            drain_pending_key("b")
            logger_mp.info("b pressed (%s); pausing policy and returning to dataset first frame.", trigger)
            if action_log_file is not None:
                write_action_log(
                    action_log_file,
                    {
                        "type": "control_event",
                        "event": "back_requested",
                        "idx": int(idx),
                        "time_s": time.time(),
                        "trigger": trigger,
                    },
                )

            clear_cached_eval_actions(policy)
            prev_raw_action_np = None
            prev_sent_action_np = None
            attention_capture_pending = False

            if not move_to_dataset_first_frame("Back requested"):
                return False
            warmup_real_policy("back")
            if key_listener.shutdown_requested:
                return False
            if not wait_for_policy_start("pre_restart"):
                return False

            now = time.perf_counter()
            next_policy_log_time = now
            next_attention_map_time = now
            next_attention_video_frame_time = now
            attention_capture_pending = False
            prev_raw_action_np = None
            prev_sent_action_np = None
            if action_log_file is not None:
                write_action_log(
                    action_log_file,
                    {
                        "type": "control_event",
                        "event": "back_restarted",
                        "idx": int(idx),
                        "time_s": time.time(),
                        "trigger": trigger,
                    },
                )
            logger_mp.info("Back complete; policy restarted from dataset first frame.")
            return True

        while True:
            if key_listener.shutdown_requested:
                logger_mp.info("q pressed; stopping policy and starting shutdown sequence.")
                shutdown_done = shutdown_robot()
                return
            loop_start_time = time.perf_counter()
            # 1. Get Observations
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
            current_arm_q = require_finite_vector("current arm q", current_arm_q, arm_dof)
            left_ee_state = right_ee_state = np.array([])
            if cfg.ee:
                with ee_shared_mem["lock"]:
                    full_state = require_finite_vector("current ee state", ee_shared_mem["state"][:], 2 * ee_dof)
                    left_ee_state = full_state[:ee_dof]
                    right_ee_state = full_state[ee_dof:]
            state_tensor = torch.from_numpy(
                np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
            ).float()
            observation["observation.state"] = state_tensor
            validate_real_observation_keys(cfg, observation, "Policy step observation")
            if key_listener.shutdown_requested:
                logger_mp.info("q pressed; stopping before manual snapshot or policy inference.")
                shutdown_done = shutdown_robot()
                return
            if key_listener.consume("b") is not None:
                if not handle_back_request("before_policy_inference"):
                    shutdown_done = shutdown_robot()
                    return
                continue
            while key_listener.consume("j") is not None:
                try:
                    save_manual_camera_snapshot(
                        cfg,
                        observation,
                        label="policy_step",
                        step_idx=idx,
                    )
                except Exception as exc:
                    logger_mp.warning("Manual snapshot failed at policy step %s: %s", idx, exc)
                if key_listener.shutdown_requested:
                    logger_mp.info("q pressed during manual snapshot handling; starting shutdown sequence.")
                    shutdown_done = shutdown_robot()
                    return
            if attention_video_frame_enabled and loop_start_time >= next_attention_video_frame_time:
                attention_dir = get_attention_map_dir(cfg)
                if attention_dir is not None:
                    _save_attention_frame_images(
                        cfg,
                        attention_dir,
                        f"video_frame_step_{int(idx):06d}",
                        [],
                        observation,
                    )
                while next_attention_video_frame_time <= loop_start_time:
                    next_attention_video_frame_time += attention_video_frame_interval_s
            if key_listener.shutdown_requested:
                logger_mp.info("q pressed; stopping before policy inference and starting shutdown sequence.")
                shutdown_done = shutdown_robot()
                return

            # 2. Get Action from Policy
            inference_start_time = time.perf_counter()
            if attention_capture_enabled and inference_start_time >= next_attention_map_time:
                attention_capture_pending = True
            capture_attention_map = attention_capture_pending
            policy_cuda_stream = getattr(
                runtime_thermal_input_adapter,
                "policy_cuda_stream",
                None,
            )
            stream_context = (
                torch.cuda.stream(policy_cuda_stream)
                if policy_cuda_stream is not None
                else nullcontext()
            )
            with stream_context:
                action = predict_action(
                    observation,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    preprocessor,
                    postprocessor,
                    policy.config.use_amp,
                    eval_task,
                    use_dataset=cfg.use_dataset,
                    robot_type=cfg.ee,
                    capture_attention_map=capture_attention_map,
                )
            if policy_cuda_stream is not None:
                # MINIMA runs on the least-priority CUDA stream. Synchronizing
                # the high-priority policy stream here keeps latency accounting
                # correct and ensures the action is ready before control.
                policy_cuda_stream.synchronize()
            inference_elapsed = time.perf_counter() - inference_start_time
            attention_metadata = None
            if capture_attention_map:
                attention_metadata = save_attention_map_capture(
                    cfg,
                    policy,
                    label="policy_step",
                    step_idx=idx,
                    inference_elapsed=inference_elapsed,
                    observation=observation,
                )
                if attention_metadata is not None:
                    attention_capture_pending = False
                    next_attention_map_time = inference_start_time + attention_interval_s
            if key_listener.shutdown_requested:
                logger_mp.info("q pressed during policy inference; skipping action and starting shutdown sequence.")
                shutdown_done = shutdown_robot()
                return
            if cfg.policy_inference_timeout_s > 0 and inference_elapsed > cfg.policy_inference_timeout_s:
                logger_mp.info(
                    f"Policy inference took {inference_elapsed:.2f}s, exceeding "
                    f"policy_inference_timeout_s {cfg.policy_inference_timeout_s:.2f}s; "
                    "skipping stale action and starting shutdown sequence."
                )
                shutdown_done = shutdown_robot()
                return
            if key_listener.consume("b") is not None:
                if not handle_back_request("after_policy_inference"):
                    shutdown_done = shutdown_robot()
                    return
                continue

            policy_action_np = np.asarray(action.cpu().numpy()).reshape(-1)
            compressed_dex3_action_np = None
            if supports_compressed_dex3_action and policy_action_np.shape[0] == compressed_dex3_action_dim:
                compressed_dex3_action_np = policy_action_np.copy()
                policy_action_np = expand_dex3_action(policy_action_np, arm_dof=arm_dof)
            raw_action_np = require_finite_vector("policy action", policy_action_np, expected_policy_dim)
            action_np = raw_action_np.copy()
            # 3. Execute Action
            raw_arm_action = raw_action_np[:arm_dof].copy()
            raw_arm_delta = raw_arm_action - current_arm_q
            arm_action = raw_arm_action
            if cfg.max_policy_arm_delta > 0:
                arm_delta_limit = float(cfg.max_policy_arm_delta)
                arm_action = current_arm_q + np.clip(
                    arm_action - current_arm_q,
                    -arm_delta_limit,
                    arm_delta_limit,
                )
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
                    arm_action = current_arm_q + np.clip(
                        arm_action - current_arm_q,
                        -arm_delta_limit,
                        arm_delta_limit,
                    )
                action_np[:arm_dof] = arm_action
            arm_action = require_finite_vector("sent arm action", arm_action, arm_dof)
            tau = require_finite_vector("arm tau", arm_ik.solve_tau(arm_action), arm_dof)
            arm_ctrl.ctrl_dual_arm(arm_action, tau)

            raw_left_ee_action = raw_right_ee_action = np.array([])
            left_ee_action = right_ee_action = np.array([])
            raw_left_delta = raw_right_delta = np.array([])
            sent_left_delta = sent_right_delta = np.array([])
            if cfg.ee:
                ee_action_start_idx = arm_dof
                raw_left_ee_action = raw_action_np[ee_action_start_idx : ee_action_start_idx + ee_dof].copy()
                raw_right_ee_action = raw_action_np[
                    ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof
                ].copy()
                left_ee_action = raw_left_ee_action
                right_ee_action = raw_right_ee_action
                if cfg.max_policy_ee_delta > 0:
                    ee_delta_limit = float(cfg.max_policy_ee_delta)
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
                    if (
                        cfg.limit_policy_delta_from_last_command
                        and prev_sent_action_np is not None
                        and prev_sent_action_np.shape[0] >= ee_action_start_idx + 2 * ee_dof
                    ):
                        prev_left_ee_action = prev_sent_action_np[
                            ee_action_start_idx : ee_action_start_idx + ee_dof
                        ]
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
                # logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")
                left_ee_action = require_finite_vector("sent left ee action", left_ee_action, ee_dof)
                right_ee_action = require_finite_vector("sent right ee action", right_ee_action, ee_dof)

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

            now = time.perf_counter()
            clipped_arm_delta = arm_action - current_arm_q
            raw_action_step_delta = (
                raw_action_np - prev_raw_action_np if prev_raw_action_np is not None else np.array([])
            )
            sent_action_step_delta = (
                action_np - prev_sent_action_np if prev_sent_action_np is not None else np.array([])
            )
            if action_log_file is not None and idx % action_log_every_n == 0:
                record = {
                    "type": "policy_step",
                    "idx": int(idx),
                    "time_s": time.time(),
                    "inference_elapsed_s": round(float(inference_elapsed), 6),
                    "loop_elapsed_s": round(float(now - loop_start_time), 6),
                    "attention_map": attention_metadata,
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
                if compressed_dex3_action_np is not None:
                    record.update(
                        {
                            "raw_compressed_dex3_action": _json_array(compressed_dex3_action_np),
                            "raw_compressed_dex3_left_grip": float(compressed_dex3_action_np[arm_dof]),
                            "raw_compressed_dex3_right_grip": float(compressed_dex3_action_np[arm_dof + 1]),
                        }
                    )
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
                    f"Policy step {idx}: inference {inference_elapsed:.3f}s, "
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
                _, visualization_data_fn = require_rerun_visualization()
                visualization_data_fn(idx, observation, state_tensor.numpy(), action_np, rerun_logger)
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
        if not shutdown_done and arm_ctrl is not None and arm_ik is not None and key_listener.shutdown_requested:
            if ready_arm_pose is not None and lowered_arm_pose is not None:
                shutdown_robot()
            else:
                try:
                    logger_mp.info("Shutdown fallback: holding current arm pose; ready pose was not initialized.")
                    hold_current_arm_pose(arm_ctrl, arm_ik)
                except Exception as hold_exc:
                    logger_mp.warning(f"Shutdown fallback hold failed: {hold_exc}")
        if action_log_file is not None:
            logger_mp.info(f"Action log saved to {action_log_path}")
            action_log_file.close()
        finalize_attention_videos(cfg)
        key_listener.stop()
        if image_info:
            cleanup_resources(image_info)


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    if (cfg.send_real_robot or cfg.save_attention_maps) and getattr(cfg.policy, "compile_model", False):
        logging.warning(
            "Disabling torch.compile for eval. "
            "The checkpoint requested compile_model=True, which can block the first policy step for autotuning "
            "or recompile when attention-map capture is enabled."
        )
        cfg.policy.compile_model = False

    configure_policy_for_thermal_input_type(cfg)
    validate_thermal_input_selection(cfg)
    logging.info(pformat(asdict(cfg)))
    prepare_attention_map_dir(cfg)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset_feature_names = _policy_dataset_feature_names(cfg)
    logging.info(
        f"Loading dataset repo_id={cfg.repo_id!r}, root={cfg.root or None!r}, "
        f"feature_names={dataset_feature_names!r}."
    )
    dataset_episodes = _sample_eval_dataset_episodes(cfg)
    dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        root=cfg.root or None,
        episodes=dataset_episodes,
        feature_names=dataset_feature_names,
        video_backend="pyav",
    )
    episode_count = dataset.num_episodes
    logging.info(
        f"Dataset loaded: root={dataset.root}, frames={len(dataset)}, "
        f"episodes={episode_count}, sampled_episodes={dataset_episodes if dataset_episodes is not None else 'all'}."
    )

    logging.info("Making policy.")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": cfg.policy.device},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    if cfg.policy.type in {"pi0", "pi05"}:
        preprocessor_overrides["tokenizer_processor"] = {
            "prefer_local_files": cfg.tokenizer_prefer_local_files,
            "local_files_only": cfg.tokenizer_local_files_only,
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides=preprocessor_overrides,
    )
    log_groot_camera_processor_alignment(cfg, preprocessor)
    runtime_thermal_input_adapter = make_runtime_thermal_input_adapter(cfg)

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
