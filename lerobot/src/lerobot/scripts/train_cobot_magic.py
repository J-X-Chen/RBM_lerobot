"""Launch PI05/DM05 fine-tuning on local Cobot Magic v3 recordings.

Example: python -m lerobot.scripts.train_cobot_magic --policy pi05 --check-data
Additional --key=value arguments are forwarded to lerobot_train.
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from lerobot.datasets.cobot_magic import IMAGE_KEYS, validate_cobot_magic_features


def training_command(args: argparse.Namespace, extra: list[str]) -> list[str]:
    checkpoint = args.pretrained_path or f"lerobot/{args.policy}_base"
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        "--dataset.repo_id=local/aloha_cup_selection_red",
        f"--dataset.root={args.dataset_root.resolve()}",
        "--dataset.video_backend=pyav",
        "--dataset.compress_dex3_actions=false",
        "--dataset.astribot_action_view=full",
        f"--policy.path={checkpoint}",
        "--policy.device=cuda",
        "--policy.dtype=bfloat16",
        "--policy.push_to_hub=false",
        "--policy.use_relative_actions=false",
        "--policy.chunk_size=50",
        "--policy.n_action_steps=50",
        "--policy.compile_model=false",
        f"--output_dir={args.output_dir.resolve()}",
        "--steps=30000",
        "--batch_size=1",
        "--num_workers=4",
        "--save_freq=5000",
        "--log_freq=100",
        "--env_eval_freq=0",
        "--wandb.enable=false",
    ]
    if args.policy == "pi05":
        command += ["--policy.gradient_checkpointing=true"]
    else:
        command += [
            f"--policy.image_keys={json.dumps(IMAGE_KEYS)}",
            "--policy.add_state=true",
            "--policy.vlm_gradient_checkpointing=true",
            "--policy.ae_gradient_checkpointing=true",
        ]
    # Replace defaults, rather than passing duplicate flags to draccus.
    overrides = {arg.split("=", 1)[0] for arg in extra}
    return [arg for arg in command if arg.split("=", 1)[0] not in overrides] + extra


def check_data(root: Path, policy_type: str) -> None:
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy_config
    from lerobot.scripts.lerobot_train import apply_robot_dataset_views

    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="local/aloha_cup_selection_red", root=str(root), video_backend="pyav"),
        policy=make_policy_config(policy_type, device="cpu"),
    )
    dataset, _ = apply_robot_dataset_views(config, make_dataset(config))
    # Exercise every episode boundary: catches missing videos, timestamp offsets,
    # action padding and parquet-file transitions without scanning every image.
    for episode in dataset.meta.episodes:
        for index in (int(episode["dataset_from_index"]), int(episode["dataset_to_index"]) - 1):
            sample = dataset[index]
            assert sample["observation.state"].shape == (14,)
            assert sample["action"].shape == (50, 14)
            assert sample["task"].strip()
            for key in IMAGE_KEYS:
                assert sample[key].shape[0] == 3, (key, sample[key].shape)
    print(
        json.dumps(
            {
                "policy": policy_type,
                "episodes": dataset.num_episodes,
                "frames": len(dataset),
                "cameras": IMAGE_KEYS,
                "numeric_stats": {
                    key: {name: value.tolist() for name, value in dataset.meta.stats[key].items()}
                    for key in ("observation.state", "action")
                },
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("pi05", "dm05"), required=True)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("lerobotv3_datasets/aloha_cup_selection_red")
    )
    parser.add_argument("--pretrained-path", help="Complete LeRobot checkpoint directory or Hub ID")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--check-data", action="store_true", help="Read all episode boundaries without loading weights"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the training command without launching")
    args, extra = parser.parse_known_args()
    if any(not arg.startswith("--") or "=" not in arg for arg in extra):
        parser.error("Forwarded training options must use --key=value syntax.")
    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    if info.get("robot_type") != "cobot_magic" or info.get("codebase_version") != "v3.0":
        parser.error("Expected a Cobot Magic LeRobot v3.0 dataset.")
    validate_cobot_magic_features(info["features"])
    if any(key not in info["features"] for key in IMAGE_KEYS):
        parser.error("This recipe requires head, left and right cameras.")
    if args.check_data:
        if extra:
            parser.error("--check-data does not accept forwarded training options.")
        check_data(args.dataset_root, args.policy)
        return
    args.output_dir = args.output_dir or Path(f"outputs/train/cobot_magic_{args.policy}")
    command = training_command(args, extra)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
