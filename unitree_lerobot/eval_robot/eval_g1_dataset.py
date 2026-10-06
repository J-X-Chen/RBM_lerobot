"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import torch
import tqdm
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from pprint import pformat
from typing import Any
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
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

from unitree_lerobot.eval_robot.utils.utils import (
    extract_observation,
    predict_action,
    to_list,
    to_scalar,
    EvalRealConfig,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data


import logging_mp
from unitree_lerobot.eval_robot.utils.logging_mp_compat import patch_logging_mp

patch_logging_mp(logging_mp)
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


ARM_DOF_BY_NAME = {
    "G1_29": 14,
    "G1_23": 14,
}

EE_DOF_BY_NAME = {
    "dex3": 7,
    "dex1": 1,
    "inspire1": 6,
    "brainco": 6,
}


def get_episode_bounds(dataset: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    if episode_index < 0 or episode_index >= len(dataset.meta.episodes):
        raise ValueError(f"Episode {episode_index} is out of range. Available: 0..{len(dataset.meta.episodes) - 1}")

    episode = dataset.meta.episodes[episode_index]
    return int(episode["dataset_from_index"]), int(episode["dataset_to_index"])


def get_configured_dofs(cfg: EvalRealConfig, action_dim: int) -> tuple[int, int]:
    arm_dof = int(ARM_DOF_BY_NAME.get(cfg.arm, min(14, action_dim)))
    ee_dof = int(EE_DOF_BY_NAME.get(str(cfg.ee).lower(), 0)) if cfg.ee else 0
    if action_dim < arm_dof:
        raise ValueError(f"Action dimension {action_dim} is smaller than arm_dof={arm_dof}.")
    return arm_dof, ee_dof


def _format_top_step_dims(step_abs: np.ndarray, start_dim: int, count: int = 3) -> str:
    if step_abs.size == 0:
        return "none"
    per_dim_p95 = np.percentile(step_abs, 95, axis=0)
    top_local = np.argsort(per_dim_p95)[-count:][::-1]
    return ", ".join(f"dim{start_dim + idx + 1}:p95={per_dim_p95[idx]:.4f}" for idx in top_local)


def _max_abs(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def log_action_diagnostics(
    ground_truth_actions: np.ndarray,
    predicted_actions: np.ndarray,
    arm_dof: int,
    ee_dof: int,
) -> None:
    segments = [("arm", 0, arm_dof)]
    if ee_dof > 0:
        segments.extend(
            [
                ("left_ee", arm_dof, arm_dof + ee_dof),
                ("right_ee", arm_dof + ee_dof, arm_dof + 2 * ee_dof),
            ]
        )

    for label, start, end in segments:
        end = min(end, ground_truth_actions.shape[1], predicted_actions.shape[1])
        if end <= start:
            continue

        gt = ground_truth_actions[:, start:end]
        pred = predicted_actions[:, start:end]
        rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))

        gt_step_abs = np.abs(np.diff(gt, axis=0))
        pred_step_abs = np.abs(np.diff(pred, axis=0))
        if pred_step_abs.size == 0:
            logger_mp.info(f"{label}: rmse={rmse:.4f}; not enough frames for step-delta diagnostics.")
            continue

        logger_mp.info(
            f"{label}: rmse={rmse:.4f}; "
            f"pred step |delta| mean/p95/max="
            f"{float(np.mean(pred_step_abs)):.4f}/"
            f"{float(np.percentile(pred_step_abs, 95)):.4f}/"
            f"{float(np.max(pred_step_abs)):.4f}; "
            f"gt step |delta| mean/p95/max="
            f"{float(np.mean(gt_step_abs)):.4f}/"
            f"{float(np.percentile(gt_step_abs, 95)):.4f}/"
            f"{float(np.max(gt_step_abs)):.4f}; "
            f"top jitter dims: {_format_top_step_dims(pred_step_abs, start)}"
        )


def eval_policy(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    # init pose  dataset.meta.episodes["dataset_from_index"][episode_index]
    from_idx, to_idx = get_episode_bounds(dataset, int(cfg.episodes))
    step = dataset[from_idx]
    first_action_dim = int(step["action"].numel())
    arm_dof, ee_dof = get_configured_dofs(cfg, first_action_dim)
    logger_mp.info(
        f"Evaluating episode {cfg.episodes}: frames [{from_idx}, {to_idx}), "
        f"action_dim={first_action_dim}, arm_dof={arm_dof}, ee_dof={ee_dof if cfg.ee else 0}."
    )

    ground_truth_actions = []
    predicted_actions = []

    if cfg.send_real_robot:
        from unitree_lerobot.eval_robot.make_robot import setup_robot_interface

        robot_interface = setup_robot_interface(cfg)
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()
        if not cfg.ee:
            logger_mp.info("End-effector disabled by --ee=''; policy hand outputs will be ignored.")

    # ===============init robot=====================
    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
    if user_input.lower() == "s":
        prev_sent_action_np = None
        next_robot_log_time = time.perf_counter()
        if cfg.send_real_robot:
            # Initialize robot to starting pose
            logger_mp.info("Initializing robot to starting pose...")
            tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
            robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)

            time.sleep(1)

        for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
            loop_start_time = time.perf_counter()

            step = dataset[step_idx]
            observation = extract_observation(step)

            action = predict_action(
                observation,
                policy,
                get_safe_torch_device(policy.config.device),
                preprocessor,
                postprocessor,
                policy.config.use_amp,
                step["task"],
                use_dataset=True,
                robot_type=None,
            )
            action_np = action.cpu().numpy()
            if action_np.shape[0] < arm_dof:
                raise ValueError(f"Policy action dimension {action_np.shape[0]} is smaller than arm_dof={arm_dof}.")

            ground_truth_actions.append(step["action"].numpy())
            predicted_actions.append(action_np)

            if cfg.send_real_robot:
                # Execute Action
                sent_action_np = action_np.copy()
                current_arm_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=np.float64)
                raw_arm_action = action_np[:arm_dof].copy()
                arm_action = raw_arm_action.copy()
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
                sent_action_np[:arm_dof] = arm_action

                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)
                # logger_mp.info(f"Arm Action: {arm_action}")

                raw_ee_delta_abs_max = 0.0
                sent_ee_delta_abs_max = 0.0
                if cfg.ee:
                    ee_action_start_idx = arm_dof
                    raw_left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof].copy()
                    raw_right_ee_action = action_np[
                        ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof
                    ].copy()
                    left_ee_action = raw_left_ee_action.copy()
                    right_ee_action = raw_right_ee_action.copy()
                    # logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")

                    with ee_shared_mem["lock"]:
                        full_ee_state = np.array(ee_shared_mem["state"][:], dtype=np.float64)
                    left_ee_state = full_ee_state[:ee_dof]
                    right_ee_state = full_ee_state[ee_dof : 2 * ee_dof]

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

                    sent_action_np[ee_action_start_idx : ee_action_start_idx + ee_dof] = left_ee_action
                    sent_action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof] = right_ee_action

                    if isinstance(ee_shared_mem["left"], SynchronizedArray):
                        ee_shared_mem["left"][:] = to_list(left_ee_action)
                        ee_shared_mem["right"][:] = to_list(right_ee_action)
                    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                        ee_shared_mem["left"].value = to_scalar(left_ee_action)
                        ee_shared_mem["right"].value = to_scalar(right_ee_action)

                    raw_ee_delta_abs_max = max(
                        _max_abs(raw_left_ee_action - left_ee_state),
                        _max_abs(raw_right_ee_action - right_ee_state),
                    )
                    sent_ee_delta_abs_max = max(
                        _max_abs(left_ee_action - left_ee_state),
                        _max_abs(right_ee_action - right_ee_state),
                    )

                raw_arm_delta_abs_max = _max_abs(raw_arm_action - current_arm_q)
                sent_arm_delta_abs_max = _max_abs(arm_action - current_arm_q)
                now = time.perf_counter()
                if now >= next_robot_log_time:
                    msg = (
                        f"Robot command step {step_idx - from_idx}: "
                        f"raw arm delta max {raw_arm_delta_abs_max:.4f} rad, "
                        f"sent arm delta max {sent_arm_delta_abs_max:.4f} rad"
                    )
                    if cfg.ee:
                        msg += (
                            f", raw ee delta max {raw_ee_delta_abs_max:.4f} rad, "
                            f"sent ee delta max {sent_ee_delta_abs_max:.4f} rad"
                        )
                    logger_mp.info(msg)
                    next_robot_log_time = now + 2.0
                prev_sent_action_np = sent_action_np.copy()

            if cfg.visualization:
                visualization_data(step_idx, observation, observation["observation.state"], action_np, rerun_logger)

            # Maintain frequency
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))

        ground_truth_actions = np.array(ground_truth_actions)
        predicted_actions = np.array(predicted_actions)
        log_action_diagnostics(ground_truth_actions, predicted_actions, arm_dof, ee_dof if cfg.ee else 0)

        plot_end_dim = arm_dof if not cfg.ee else ground_truth_actions.shape[1]
        plot_title_suffix = "Arm Only" if not cfg.ee else "All Actions"
        ground_truth_plot = ground_truth_actions[:, :plot_end_dim]
        predicted_plot = predicted_actions[:, :plot_end_dim]

        # Get the number of timesteps and action dimensions
        n_dims = ground_truth_plot.shape[1]

        # Create a figure with subplots for each action dimension
        fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4 * n_dims), sharex=True)
        axes = np.atleast_1d(axes)
        fig.suptitle(f"Ground Truth vs Predicted Actions ({plot_title_suffix})")

        # Plot each dimension
        for i in range(n_dims):
            ax = axes[i]

            ax.plot(ground_truth_plot[:, i], label="Ground Truth", color="blue")
            ax.plot(predicted_plot[:, i], label="Predicted", color="red", linestyle="--")
            ax.set_ylabel(f"Dim {i + 1}")
            ax.legend()

        # Set common x-label
        axes[-1].set_xlabel("Timestep")

        plt.tight_layout()
        # plt.show()

        time.sleep(1)
        plt.savefig("figure.png")
        plt.close(fig)


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.root)

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

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(cfg, dataset, policy, preprocessor, postprocessor)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
