"""PI05/DM05 offline evaluation or synchronous dual-Piper control.

Uses the shared project lerobot and the supplied Piper SDK, never Unitree SDK.
"""

import argparse
import itertools
import json
import shlex
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "mask_operation/src", ROOT / "piper_sdk", ROOT / "lerobot" / "src"):
    sys.path.insert(0, str(path))

import numpy as np
from aloha_lifecycle import TerminalCommands, slow_move
from aloha_policy import AlohaPolicy
from robot_control.piper_bimanual import ArmConfig, PiperBimanual


def offline(args, runner):
    from lerobot.datasets.cobot_magic import validate_cobot_magic_features
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        "local/aloha_cup_selection_red", root=args.dataset_root, episodes=[args.episode], video_backend="pyav"
    )
    validate_cobot_magic_features(dataset.meta.features)
    runner.reset()
    errors = []
    with args.log.open("w") as log:
        for i in range(min(args.steps, len(dataset))):
            item = dataset[i]
            action = runner.action(item, args.task or item["task"])
            target = item["action"].numpy()
            errors.append(np.square(action - target))
            log.write(json.dumps({"frame": i, "action": action.tolist(), "target": target.tolist()}) + "\n")
    if not errors:
        raise ValueError("No dataset frames selected")
    mse = np.mean(errors, axis=0)
    print(
        json.dumps(
            {
                "frames": len(errors),
                "mse_per_joint": mse.tolist(),
                "left_mse": float(mse[:7].mean()),
                "right_mse": float(mse[7:].mean()),
            },
            indent=2,
        )
    )


def load_alignment_target(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        "local/aloha_cup_selection_red", root=args.dataset_root, episodes=[args.episode], video_backend="pyav"
    )
    # Numeric observation only: no video decoding is needed for alignment.
    goal = np.asarray(dataset.hf_dataset[0]["observation.state"], dtype=np.float64)
    if goal.shape != (14,) or not np.isfinite(goal).all():
        raise ValueError("Alignment requires finite 14D dataset state")
    return goal


def live(args, runner):
    from aloha_cameras import AlohaCameras

    if args.robot_host:
        from robot_control.remote_piper import RemotePiper

        robot = RemotePiper(
            args.robot_host,
            args.robot_port,
            execute=args.execute,
            joint_speed=args.joint_speed,
            gripper_speed=args.gripper_speed,
            active_arm=args.active_arm,
            command_timeout=args.command_timeout,
            camera_timeout=args.camera_timeout,
        )
    else:
        if not args.hardware_config:
            raise ValueError("Specify --robot-host for network control, or --hardware-config for local CAN")
        hardware = json.loads(args.hardware_config.read_text())
        robot = PiperBimanual(
            ArmConfig(**hardware["left"]),
            ArmConfig(**hardware["right"]),
            execute=args.execute,
            joint_speed=args.joint_speed,
            gripper_speed=args.gripper_speed,
            active_arm=args.active_arm,
            command_timeout=args.command_timeout,
        )
    cameras = AlohaCameras(
        args.image_server_address or args.robot_host or "127.0.0.1",
        {
            "head": args.head_camera_port,
            "left": args.left_camera_port,
            "right": args.right_camera_port,
        },
        max_age=args.camera_timeout,
    )
    task = args.task
    if not task:
        # Use the recorded task; no made-up prompt and no need to read videos.
        import pyarrow.parquet as pq

        tasks = pq.read_table(args.dataset_root / "meta/tasks.parquet").to_pydict()["task"]
        if len(tasks) != 1:
            raise ValueError("Multiple dataset tasks: specify --task")
        task = tasks[0]

    def move(goal, keys, log, phase):
        return slow_move(
            robot,
            cameras,
            goal,
            keys,
            frequency=args.frequency,
            joint_speed=min(args.joint_speed, args.return_joint_speed),
            gripper_speed=min(args.gripper_speed, args.return_gripper_speed),
            timeout=args.return_timeout,
            log=log,
            phase=phase,
        )

    async_runner = None
    try:
        robot.connect()
        if getattr(args, "adaptive_tracking", False) and args.execute:
            if not args.robot_host or not robot.info.get("smooth_control") or "accepted_target" not in robot.info.get("capabilities", []):
                raise RuntimeError("Adaptive tracking requires updated remote service with --smooth-control")
        cameras.connect()
        with TerminalCommands() as keys, args.log.open("w") as log:
            state = robot.read_state()
            print("Current left-7/right-7 state:", state.tolist())
            runner.warmup({"observation.state": state, **cameras.snapshot()}, task, args.policy_warmup_steps)
            if getattr(args, "async_prefetch", False):
                from aloha_async import PrefetchPolicy
                async_runner = PrefetchPolicy(runner, args.prefetch_steps, args.async_timeout)
                runner = async_runner
            first_frame = load_alignment_target(args) if args.execute else None
            home = None

            def start_control():
                nonlocal home
                if not robot.enabled:
                    robot.enable()
                    home = robot.anchor.copy()  # sampled immediately before the first enable

            def ready():
                print("r: slowly align episode first frame; s: run policy; q/Ctrl+C: return and exit")
                while not keys.quit:
                    key = keys.poll()
                    if keys.quit:
                        return False
                    if key == "s":
                        start_control()
                        return True
                    if key == "r":
                        # Load/validate before enabling, while the operator explicitly requests alignment.
                        goal = first_frame.copy()
                        robot.prepare_target(goal)
                        start_control()
                        if not move(goal, keys, log, "alignment"):
                            return False
                        runner.reset()
                        print("Alignment complete. Press s to start, q to return and exit.")
                    if robot.enabled:
                        cameras.snapshot()
                        robot.read_state()
                        robot.send(robot.last, 1 / args.frequency)  # hold and renew watchdog while waiting
                    time.sleep(1 / args.frequency)
                return False

            keys.poll()
            running = not keys.quit
            if args.execute:
                if not sys.stdin.isatty():
                    raise RuntimeError("Live execution requires a terminal for start/stop commands")
                running = ready() if running else False
            else:
                print("Read-only preview: q/Ctrl+C exits; r/s do not enable or move the arms.")
            from aloha_tracking import TrackingPacer
            pacer = TrackingPacer() if getattr(args, "adaptive_tracking", False) and args.execute else None
            tracking_target = None
            steps = itertools.count() if args.steps is None else range(args.steps)
            for step in steps if running else ():
                started = time.monotonic()
                key = keys.poll()
                if keys.quit:
                    break
                if key == "r" and args.execute:
                    tracking_target = None
                    if pacer is not None:
                        pacer.reset()
                    if not move(first_frame.copy(), keys, log, "alignment"):
                        break
                    runner.reset()
                    if not ready():
                        break
                    started = time.monotonic()
                state = robot.read_state()
                images = cameras.snapshot()
                if pacer is not None and tracking_target is not None:
                    joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
                    lag = float(np.max(np.abs(tracking_target[joints] - state[joints])))
                    was_waiting = pacer.started is not None
                    if pacer.wait(lag, time.monotonic()):
                        if not was_waiting and async_runner is not None:
                            async_runner.suspend_prediction()
                        command = robot.send(tracking_target, 1 / args.frequency)
                        log.write(json.dumps(dict(step=step, phase="tracking_wait", lag_rad=lag,
                            time_unix_s=time.time(), state=state.tolist(), command=command.tolist(),
                            action=tracking_target.tolist(), queued_actions=len(runner.queue))) + "\n")
                        log.flush()
                        time.sleep(max(0, 1 / args.frequency - (time.monotonic() - started)))
                        continue
                prediction_started = time.monotonic()
                action = runner.action({"observation.state": state, **images}, task)
                prediction_s = time.monotonic() - prediction_started
                # Discard a predicted action when shutdown was requested during inference.
                key = keys.poll()
                if keys.quit:
                    break
                if key == "r" and args.execute:
                    tracking_target = None
                    if pacer is not None:
                        pacer.reset()
                    if not move(first_frame.copy(), keys, log, "alignment"):
                        break
                    runner.reset()
                    if not ready():
                        break
                    continue
                cameras.snapshot()
                if time.monotonic() - started > robot.command_timeout:
                    raise RuntimeError("Inference exceeded command timeout")
                waiting = action is None
                if waiting:
                    # Only hold the last acknowledged output while a bounded prediction is pending.
                    action = tracking_target.copy() if tracking_target is not None else (robot.last.copy() if robot.last is not None else state.copy())
                command = robot.send(action, 1 / args.frequency)
                if pacer is not None and not waiting:
                    tracking_target = robot.accepted_target.copy()
                log.write(
                    json.dumps(
                        {
                            "step": step,
                            "phase": "async_wait" if waiting else "policy",
                            "async": getattr(runner, "status", None),
                            "time_unix_s": time.time(),
                            "prediction_s": prediction_s,
                            "control_step_s": time.monotonic() - started,
                            "queued_actions": len(runner.queue),
                            "state": state.tolist(),
                            "action": action.tolist(),
                            "command": command.tolist(),
                            "execute": args.execute,
                        }
                    )
                    + "\n"
                )
                log.flush()
                time.sleep(max(0, 1 / args.frequency - (time.monotonic() - started)))
            # Only normal completion/q/first SIGINT reach this point. Faults skip return entirely.
            if async_runner is not None:
                async_runner.close()
                async_runner = None
            if args.execute and robot.enabled and home is not None:
                if keys.abort:
                    raise RuntimeError("Repeated stop request: stopping without return")
                keys.returning = True
                goal = home.copy()
                goal[[6, 13]] = robot.read_state()[[6, 13]]  # do not open a loaded gripper on exit
                print("Returning arms to startup pose, holding gripper openings. q/Ctrl+C again stops immediately.")
                move(goal, keys, log, "return")
                keys.poll()
                if keys.abort:
                    raise RuntimeError("Repeated stop request: stopping instead of holding")
                robot.finish_hold()
                print("Return completed; holding position. Keep the service running for the next session.")
    finally:
        try:
            robot.close()
        finally:
            cameras.close()
            if async_runner is not None:
                async_runner.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy.path", dest="checkpoint", required=True)
    parser.add_argument("--mode", choices=("dataset", "live"), default="dataset")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "lerobotv3_datasets/aloha_cup_selection_red")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None,
                        help="Control-cycle limit; omitted: live runs until q/Ctrl+C or fault, dataset evaluates 100 frames")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-action-steps", type=int)
    parser.add_argument("--policy-warmup-steps", type=int, default=5)
    parser.add_argument("--async-prefetch", action="store_true", help="Experimental small-overlap background inference")
    parser.add_argument("--adaptive-tracking", action="store_true", help="Pause queue advancement when accepted joint target lag exceeds 0.075 rad")
    parser.add_argument("--prefetch-steps", type=int, default=5)
    parser.add_argument("--async-timeout", type=float, default=2.0)
    parser.add_argument("--task")
    parser.add_argument("--hardware-config", type=Path)
    parser.add_argument("--robot-host", help="Robot machine running agilex_image_service")
    parser.add_argument("--robot-port", type=int, default=6200)
    parser.add_argument("--image-server-address", help="Defaults to robot-host")
    parser.add_argument("--head-camera-port", type=int, default=55555)
    parser.add_argument("--left-camera-port", type=int, default=55556)
    parser.add_argument("--right-camera-port", type=int, default=55557)
    parser.add_argument("--camera-timeout", type=float, default=2.0)
    parser.add_argument("--execute", action="store_true", help="Enable real CAN motion (live mode only)")
    parser.add_argument("--active-arm", choices=("left", "right", "both"), default="both")
    parser.add_argument("--frequency", type=float, default=30)
    parser.add_argument("--joint-speed", type=float, default=0.3, help="Maximum commanded joint speed, rad/s")
    parser.add_argument("--gripper-speed", type=float, default=0.02, help="Maximum commanded opening speed, m/s")
    parser.add_argument("--command-timeout", type=float, default=5.0)
    parser.add_argument("--return-joint-speed", type=float, default=0.15)
    parser.add_argument("--return-gripper-speed", type=float, default=0.01)
    parser.add_argument("--return-timeout", type=float, default=60.0)
    parser.add_argument("--log", type=Path, default=Path("eval_action_logs/aloha.jsonl"))
    parser.add_argument("--robot-mask", action="store_true", help="Render robot silhouette from measured state and fill RGB")
    parser.add_argument("--mask-fill-rgb", type=int, nargs=3, default=[0, 255, 0], metavar=("R", "G", "B"))
    parser.add_argument("--mask-assets-root", type=Path, default=Path.home()/"code_file/cloak")
    parser.add_argument("--mask-config", type=Path)
    parser.add_argument("--allow-candidate-mask", action="store_true")
    args = parser.parse_args(argv)
    if (args.steps is not None and args.steps <= 0) or not np.isfinite(args.frequency) or args.frequency < 1 or args.episode < 0:
        parser.error("steps/frequency must be positive (frequency >=1); episode must be nonnegative")
    if args.mode == "dataset" and args.steps is None:
        args.steps = 100
    for name in (
        "joint_speed",
        "gripper_speed",
        "command_timeout",
        "camera_timeout",
        "return_joint_speed",
        "return_gripper_speed",
        "return_timeout",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name + " must be finite and positive")
    if args.policy_warmup_steps < 0:
        parser.error("policy-warmup-steps must be nonnegative")
    if args.prefetch_steps < 1 or not np.isfinite(args.async_timeout) or args.async_timeout <= 0:
        parser.error("prefetch-steps and async-timeout must be positive")
    if args.async_prefetch and args.mode != "live":
        parser.error("async-prefetch is for live mode only")
    if args.mode == "dataset" and args.execute:
        parser.error("Dataset observations must never be used to control real hardware")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    from cloak_mask.runtime import RobotMask
    mask = RobotMask("aloha", assets_root=args.mask_assets_root, config=args.mask_config,
        fill_rgb=args.mask_fill_rgb, allow_candidate=args.allow_candidate_mask) if args.robot_mask else None
    runner = AlohaPolicy(args.checkpoint, args.device, args.n_action_steps)
    runner.robot_mask = mask
    # Keep the effective CLI and checkpoint contract beside action logs, like the S1 evaluator.
    metadata = {
        "command": shlex.join([sys.executable, sys.argv[0], *(sys.argv[1:] if argv is None else argv)]),
        "cwd": str(Path.cwd()),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "policy_type": runner.policy.config.type,
        "action_feature_names": runner.policy.config.action_feature_names,
        "image_keys": list(runner.policy.config.image_features),
        "n_action_steps": runner.n_action_steps,
    }
    args.log.with_suffix(".run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    try:
        if args.mode == "dataset":
            offline(args, runner)
        else:
            live(args, runner)
    finally:
        if mask is not None:
            mask.close()


if __name__ == "__main__":
    main()
