"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import numpy as np

from multiprocessing.sharedctypes import SynchronizedArray
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from unitree_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from unitree_lerobot.eval_robot.eval_g1 import (
    KeyCommandListener,
    get_ready_arm_pose,
    hold_current_arm_pose,
    move_arm_to_pose_slowly,
    move_ee_to_pose_slowly,
    run_shutdown_sequence,
)
from unitree_lerobot.eval_robot.utils.utils import cleanup_resources, EvalRealConfig

from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from unitree_lerobot.eval_robot.utils.utils import to_list, to_scalar

import logging_mp
from unitree_lerobot.eval_robot.utils.logging_mp_compat import patch_logging_mp

patch_logging_mp(logging_mp)
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


def get_episode_bounds(dataset: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    if episode_index < 0 or episode_index >= len(dataset.meta.episodes):
        raise ValueError(f"Episode {episode_index} is out of range. Available: 0..{len(dataset.meta.episodes) - 1}")

    episode = dataset.meta.episodes[episode_index]
    return int(episode["dataset_from_index"]), int(episode["dataset_to_index"])


@parser.wrap()
def replay_main(cfg: EvalRealConfig):
    logger_mp.info(f"Arguments: {cfg}")

    if cfg.frequency <= 0:
        raise ValueError(f"frequency must be positive, got {cfg.frequency}")

    if cfg.visualization:
        rerun_logger = RerunLogger()
    visualization_enabled = bool(cfg.visualization)

    dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.root, episodes=[cfg.episodes])
    actions = dataset.hf_dataset.select_columns("action")
    from_idx, to_idx = get_episode_bounds(dataset, int(cfg.episodes))
    episode_num_frames = to_idx - from_idx
    logger_mp.info(
        f"Replay episode {cfg.episodes}: frames [{from_idx}, {to_idx}), "
        f"{episode_num_frames} frame(s), {episode_num_frames / cfg.frequency:.2f}s at {cfg.frequency} Hz."
    )

    try:
        user_input = input(
            "This will connect to the real robot, enter debug mode, and start low-level hold control. "
            "Enter 'c' to connect:"
        )
    except EOFError:
        logger_mp.info("Replay canceled before connecting to robot: no input received.")
        return
    if user_input.lower() != "c":
        logger_mp.info("Replay canceled before connecting to robot.")
        return

    image_info = None
    key_listener = KeyCommandListener()
    arm_ctrl = arm_ik = None
    ee_shared_mem = None
    ee_dof = 0
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
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        """The main control and evaluation loop."""
        # Unpack interfaces for convenience
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )
        tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam = (
            image_info[key]
            for key in [
                "tv_img_array",
                "wrist_img_array",
                "tv_img_shape",
                "wrist_img_shape",
                "is_binocular",
                "has_wrist_cam",
            ]
        )

        expected_action_dim = arm_dof + (2 * ee_dof if cfg.ee else 0)
        first_action_dim = int(actions[from_idx]["action"].numel())
        if first_action_dim < expected_action_dim:
            raise ValueError(
                f"Action dimension {first_action_dim} is too small for arm_dof={arm_dof}, "
                f"ee_dof={ee_dof}, expected at least {expected_action_dim}."
            )

        step = dataset.hf_dataset[from_idx]
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()
        init_ee_pose = step["observation.state"][arm_dof : arm_dof + 2 * ee_dof].cpu().numpy()

        key_listener.start()
        logger_mp.info(
            "Keyboard controls: r=slowly move to dataset first frame, s=start replay, "
            "q=open hands, retreat and lower, c=exit after replay."
        )

        if hasattr(arm_ctrl, "arm_velocity_limit"):
            arm_ctrl.arm_velocity_limit = float(cfg.arm_low_level_velocity_limit)
            logger_mp.info(
                f"Low-level arm velocity backstop set to {arm_ctrl.arm_velocity_limit:.4f} rad/s; "
                f"trajectory speed is {cfg.arm_velocity_limit:.4f} rad/s; "
                f"tracking error limit is {cfg.arm_tracking_error_limit:.4f} rad."
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

        logger_mp.info("Ready. Press 'r' to slowly move to dataset first frame, or 'q' to shutdown.")
        key = key_listener.wait_for("r", "q")
        if key == "q":
            shutdown_done = shutdown_robot()
            return

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

        logger_mp.info("At dataset first frame. Press 's' to start replay, or 'q' to shutdown.")
        key = key_listener.wait_for("s", "q")
        if key == "q":
            shutdown_done = shutdown_robot()
            return

        logger_mp.info(f"Starting replay loop at {cfg.frequency} Hz.")
        for idx, dataset_idx in enumerate(range(from_idx, to_idx)):
            if key_listener.shutdown_requested:
                logger_mp.info("q pressed; stopping replay and starting shutdown sequence.")
                shutdown_done = shutdown_robot()
                return

            loop_start_time = time.perf_counter()

            left_ee_state = right_ee_state = np.array([])
            action_np = actions[dataset_idx]["action"].numpy()

            # exec action
            arm_action = action_np[:arm_dof]
            tau = arm_ik.solve_tau(arm_action)
            arm_ctrl.ctrl_dual_arm(arm_action, tau)
            logger_mp.info(f"arm_action {arm_action}, tau {tau}")

            if cfg.ee:
                ee_action_start_idx = arm_dof
                left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]
                logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")

                with ee_shared_mem["lock"]:
                    full_state = np.array(ee_shared_mem["state"][:])
                    left_ee_state = full_state[:ee_dof]
                    right_ee_state = full_state[ee_dof:]

                if isinstance(ee_shared_mem["left"], SynchronizedArray):
                    ee_shared_mem["left"][:] = to_list(left_ee_action)
                    ee_shared_mem["right"][:] = to_list(right_ee_action)
                elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                    ee_shared_mem["left"].value = to_scalar(left_ee_action)
                    ee_shared_mem["right"].value = to_scalar(right_ee_action)

            if visualization_enabled:
                observation, current_arm_q = process_images_and_observations(
                    tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam, arm_ctrl
                )
                state = np.concatenate((current_arm_q, left_ee_state, right_ee_state))

                try:
                    visualization_data(idx, observation, state, action_np, rerun_logger)
                except Exception as exc:
                    visualization_enabled = False
                    logger_mp.warning(f"Visualization failed and will be disabled for this replay: {exc}")

            # Maintain frequency
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))

        logger_mp.info("Replay finished. Press 'q' to run shutdown sequence, or 'c' to exit while holding final pose.")
        key = key_listener.wait_for("q", "c")
        if key == "q":
            shutdown_done = shutdown_robot()
    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt received; starting shutdown sequence.")
        key_listener.shutdown_requested = True
    except Exception as e:
        logger_mp.info(f"An error occurred: {e}")
        key_listener.shutdown_requested = True
    finally:
        if (
            not shutdown_done
            and arm_ctrl is not None
            and arm_ik is not None
            and ready_arm_pose is not None
            and lowered_arm_pose is not None
            and key_listener.shutdown_requested
        ):
            shutdown_robot()
        key_listener.stop()
        if image_info:
            cleanup_resources(image_info)


if __name__ == "__main__":
    replay_main()
