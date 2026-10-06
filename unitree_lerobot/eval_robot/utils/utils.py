import numpy as np
import torch
from typing import Any
from contextlib import nullcontext
from copy import copy
import logging
from dataclasses import dataclass, field
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION


import logging_mp
from unitree_lerobot.eval_robot.utils.logging_mp_compat import patch_logging_mp

patch_logging_mp(logging_mp)
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

THERMAL_INPUT_TYPES = {
    "false",
    "grey",
    "twogrey",
    "twoblack",
    "twomatchingblack",
    "twofixmatchingblack",
    "mixing",
    "matching",
    "anythermal",
    "resthermal",
}


def normalize_thermal_input_type(value: str | bool | None) -> str:
    """Normalize a legacy eval-side image input selector."""
    if isinstance(value, bool):
        if value is False:
            return "false"
        raise ValueError("image input mode true is ambiguous; use false or a named mode.")
    text = str(value or "false").strip().strip("\"'").lower()
    if text in {"", "0", "false", "none", "off", "no", "disable", "disabled"}:
        return "false"
    if text in {"grey", "gray"}:
        return "grey"
    if text in {"twogrey", "two_grey", "two-grey", "twogray", "two_gray", "two-gray"}:
        return "twogrey"
    if text in {"twoblack", "two_black", "two-black"}:
        return "twoblack"
    if text in {
        "twomatchingblack",
        "two_matching_black",
        "two-matching-black",
        "twomatchedblack",
        "two_matched_black",
        "two-matched-black",
    }:
        return "twomatchingblack"
    if text in {
        "twofixmatchingblack",
        "two_fix_matching_black",
        "two-fix-matching-black",
        "twofixedmatchingblack",
        "two_fixed_matching_black",
        "two-fixed-matching-black",
    }:
        return "twofixmatchingblack"
    if text in {"mix", "mixing"}:
        return "mixing"
    if text in {"match", "matched", "matching"}:
        return "matching"
    if text in {"anythermal", "any_thermal"}:
        return "anythermal"
    if text in {"resthermal", "res_thermal", "residual_thermal"}:
        return "resthermal"
    raise ValueError(f"Unsupported image input mode: {value!r}.")


def _feature_dim(feature: Any) -> int | None:
    if feature is None:
        return None
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, dict):
        shape = feature.get("shape")
    if not shape:
        return None
    return int(shape[-1])


def _postprocessor_action_dim(
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
) -> int | None:
    """Return the action dimension expected by the first action normalizer/unnormalizer."""
    for step in getattr(postprocessor, "steps", ()):
        features = getattr(step, "features", None)
        if isinstance(features, dict):
            dim = _feature_dim(features.get(ACTION))
            if dim is not None:
                return dim

        tensor_stats = getattr(step, "_tensor_stats", None)
        if isinstance(tensor_stats, dict) and ACTION in tensor_stats:
            action_stats = tensor_stats[ACTION]
            if isinstance(action_stats, dict) and action_stats:
                first_stat = next(iter(action_stats.values()))
                if hasattr(first_stat, "shape") and len(first_stat.shape) > 0:
                    return int(first_stat.shape[-1])
    return None


def _maybe_trim_dex3_action_for_postprocessor(
    action: PolicyAction,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    robot_type: str | None,
) -> PolicyAction:
    """Adapt padded PI0.5 outputs to checkpoints trained with compressed Dex3 grip scalars.

    Old Unitree PI0.5 runs could be trained with ``--dataset.compress_dex3_actions=true``:
    the policy/postprocessor action stats are arm_dof + 2, while real Dex3 state/action space is
    arm_dof + 14.  Some migrated PI0.5 checkpoints can still expose the internal padded action
    width before unnormalization.  Keep the trained arm+left_grip+right_grip prefix before the
    postprocessor; eval_g1.py expands the unnormalized arm+2-grip action back to full Dex3
    commands before sending to the robot.
    """
    if str(robot_type or "").lower() != "dex3" or not hasattr(action, "shape") or action.ndim == 0:
        return action

    expected_dim = _postprocessor_action_dim(postprocessor)
    if expected_dim is None or expected_dim < 3:
        return action

    action_dim = int(action.shape[-1])
    if action_dim <= expected_dim:
        return action

    trimmed = action[..., :expected_dim]
    if not getattr(postprocessor, "_unitree_logged_dex3_auto_trim", False):
        logger_mp.info(
            "Detected compressed Dex3 action postprocessor: trimming raw policy action "
            f"{tuple(action.shape)} -> {tuple(trimmed.shape)} before unnormalization."
        )
        setattr(postprocessor, "_unitree_logged_dex3_auto_trim", True)
    return trimmed


def _is_groot_relative_action_policy(policy: PreTrainedPolicy) -> bool:
    config = getattr(policy, "config", None)
    return str(getattr(config, "type", "")).lower() == "groot" and bool(
        getattr(config, "use_relative_actions", False)
    )


def clear_cached_eval_actions(policy: PreTrainedPolicy | None) -> None:
    if policy is not None:
        setattr(policy, "_unitree_eval_decoded_action_queue", [])


def extract_observation(step: dict):
    observation = {}

    for key, value in step.items():
        if key.startswith("observation.images."):
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] in [1, 3]:
                value = np.transpose(value, (2, 0, 1))
            observation[key] = value

        elif key == "observation.state":
            observation[key] = value

    return observation


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    use_dataset: bool | None = False,
    robot_type: str | None = None,
    capture_attention_map: bool = False,
):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        use_external_groot_queue = _is_groot_relative_action_policy(policy) and not use_dataset
        if use_external_groot_queue:
            cached_actions = getattr(policy, "_unitree_eval_decoded_action_queue", [])
            if cached_actions:
                action = cached_actions.pop(0)
                setattr(policy, "_unitree_eval_decoded_action_queue", cached_actions)
                return action.to("cpu")

        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if not use_dataset:
                # Skip non-tensor observations (like task strings)
                if not hasattr(observation[name], "unsqueeze"):
                    continue
                if "images" in name:
                    observation[name] = observation[name].type(torch.float32) / 255
                    observation[name] = observation[name].permute(2, 0, 1).contiguous()

            observation[name] = observation[name].unsqueeze(0).to(device)

        observation["task"] = task if task else ""
        observation["robot_type"] = robot_type if robot_type else ""

        observation = preprocessor(observation)

        # Compute the next action with the policy
        # based on the current observation
        setattr(policy, "_image_input_log_context", "dataset" if use_dataset else "real")
        previous_capture_attention_map = getattr(policy, "_unitree_capture_attention_map", False)
        setattr(policy, "_unitree_capture_attention_map", bool(capture_attention_map))
        if _is_groot_relative_action_policy(policy):
            try:
                action = policy.predict_action_chunk(observation)
                action = postprocessor(action)
                if use_external_groot_queue:
                    actions_per_chunk = int(
                        getattr(
                            policy,
                            "_action_queue_steps",
                            getattr(getattr(policy, "config", None), "n_action_steps", action.shape[1]),
                        )
                    )
                    action_chunk = action.squeeze(0)[: max(1, actions_per_chunk)].to("cpu")
                    cached_actions = [action_chunk[idx].clone() for idx in range(1, action_chunk.shape[0])]
                    setattr(policy, "_unitree_eval_decoded_action_queue", cached_actions)
                    return action_chunk[0]
            finally:
                setattr(policy, "_unitree_capture_attention_map", previous_capture_attention_map)
        else:
            try:
                action = policy.select_action(observation)
                action = _maybe_trim_dex3_action_for_postprocessor(action, postprocessor, robot_type)
                action = postprocessor(action)
            finally:
                setattr(policy, "_unitree_capture_attention_map", previous_capture_attention_map)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def reset_policy(policy: PreTrainedPolicy):
    policy.reset()
    clear_cached_eval_actions(policy)


def cleanup_resources(image_info: dict[str, Any]):
    """Safely close and unlink shared memory resources."""
    image_client = image_info.get("image_client")
    if image_client is not None and hasattr(image_client, "stop"):
        image_client.stop()

    logger_mp.info("Cleaning up shared memory resources.")
    for shm in image_info["shm_resources"]:
        if shm:
            shm.close()
            shm.unlink()


def to_list(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().ravel().tolist()
    if isinstance(x, np.ndarray):
        return x.ravel().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def to_scalar(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return float(x.detach().cpu().ravel()[0].item())
    if isinstance(x, np.ndarray):
        return float(x.ravel()[0])
    if isinstance(x, (list, tuple)):
        return float(x[0])
    return float(x)


@dataclass
class EvalRealConfig:
    repo_id: str
    policy: PreTrainedConfig | None = None

    root: str = ""
    dataset_sample_ratio: float = 1.0
    dataset_sample_seed: int = 0
    task: str = ""
    feature_names: list[str] = field(default_factory=list)
    camera_features: list[str] = field(default_factory=list)
    episodes: int = 0
    frequency: float = 30.0
    arm_velocity_limit: float = 0.25
    ready_arm_velocity_limit: float = 0.25
    arm_low_level_velocity_limit: float = 8.0
    arm_tracking_error_limit: float = 0.35
    slow_move_tau_scale: float = 1.0
    slow_move_tau_limit: float = 8.0
    slow_move_frequency: float = 50.0
    slow_move_tracking_pause_ratio: float = 0.98
    slow_move_tracking_wait_timeout_s: float = 5.0
    lower_block_wait_for_continue: bool = True
    lower_block_tracking_ratio: float = 0.75
    lower_block_wait_timeout_s: float = 1.0
    init_tolerance: float = 0.03
    dataset_start_tolerance: float = 0.08
    init_timeout_s: float = 600.0
    require_step_confirm: bool = True
    auto_ready_pose: bool = True
    ready_arm_pose: str = ""
    ready_lift_m: float = 0.0
    ready_max_joint_delta: float = 0.8
    lower_to_startup_on_quit: bool = True
    open_ee_on_quit: bool = True
    max_policy_arm_delta: float = 0.080
    max_policy_ee_delta: float = 10.0
    limit_policy_delta_from_last_command: bool = True
    ee_tracking_tolerance: float = 5.0
    abort_on_ee_tracking_error: bool = False
    require_head_camera_on_r: bool = True
    save_camera_debug_on_r: bool = False
    head_camera_wait_timeout_s: float = 5.0
    head_camera_blank_mean_threshold: float = 1.0
    require_wrist_cameras_on_r: bool = True
    wrist_camera_wait_timeout_s: float = 5.0
    wrist_camera_blank_mean_threshold: float = 1.0
    save_camera_features_on_s: bool = True
    camera_feature_snapshot_dir: str = "tmp"
    image_transport: str = "teleimager"
    image_server_address: str = "192.168.123.164"
    dds_network_interface: str = ""
    legacy_image_port: int = 5555
    head_camera_image_height: int = 480
    head_camera_image_width: int = 640
    wrist_camera_image_height: int = 480
    wrist_camera_image_width: int = 640
    use_wrist_cameras: bool = False
    head_camera_zmq_port: int = 55555
    left_wrist_camera_zmq_port: int = 55556
    right_wrist_camera_zmq_port: int = 55557
    init_ee_velocity_limit: float = 25.0
    shutdown_ee_open_velocity_limit: float = 50.0
    policy_warmup_steps: int = 1
    real_policy_warmup_steps: int = 1
    policy_inference_timeout_s: float = 5.0
    tokenizer_prefer_local_files: bool = True
    tokenizer_local_files_only: bool = False
    action_log_path: str = "eval_action_logs"
    action_log_every_n: int = 1
    save_attention_maps: bool = False
    save_attention_images: bool = False
    attention_map_interval_s: float = 10.0
    attention_map_warmup: bool = False
    attention_map_camera: str = "observation.images.cam_left_high"

    # Basic control parameters
    arm: str = "Astribot_S1"  # Astribot_S1, G1_29, G1_23
    ee: str = ""  # astribot, dex3, dex1, inspire1, brainco

    # Mode flags
    motion: bool = False
    headless: bool = False
    visualization: bool = False
    send_real_robot: bool = False
    use_dataset: bool = False

    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logging.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
