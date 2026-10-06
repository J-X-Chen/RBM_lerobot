"""
Merge local LeRobot v3 datasets, including datasets with thermal video features.

Example:
    python astribot_lerobot/utils/merge_lerobotv3_datasets.py \
        /path/to/dataset_a \
        /path/to/dataset_b \
        --output-dir /path/to/dataset_merged \
        --output-repo-id J-X-Chen/dataset_merged
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


LEROBOT_SRC = Path(__file__).resolve().parents[1] / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets import LeRobotDataset, merge_datasets  # noqa: E402


def _default_repo_ids(roots: list[Path]) -> list[str]:
    counts: dict[str, int] = {}
    repo_ids = []
    for root in roots:
        name = root.name or "dataset"
        counts[name] = counts.get(name, 0) + 1
        suffix = f"_{counts[name] - 1}" if counts[name] > 1 else ""
        repo_ids.append(f"local/{name}{suffix}")
    return repo_ids


def _parse_repo_ids(raw: str | None, roots: list[Path]) -> list[str]:
    if raw is None:
        return _default_repo_ids(roots)

    try:
        repo_ids = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError('--repo-ids must be a JSON list, e.g. \'["local/a", "local/b"]\'.') from exc

    if not isinstance(repo_ids, list) or not all(isinstance(item, str) for item in repo_ids):
        raise ValueError("--repo-ids must be a JSON list of strings.")
    if len(repo_ids) != len(roots):
        raise ValueError(f"--repo-ids length must match input roots: {len(repo_ids)} != {len(roots)}.")
    return repo_ids


def merge_local_lerobotv3_datasets(
    roots: list[Path],
    output_dir: Path,
    output_repo_id: str,
    repo_ids: list[str] | None = None,
    concatenate_videos: bool = False,
    concatenate_data: bool = False,
) -> LeRobotDataset:
    if len(roots) < 2:
        raise ValueError("Please provide at least two input dataset roots.")
    for root in roots:
        if not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Not a LeRobot v3 dataset root: {root}")

    source_repo_ids = repo_ids if repo_ids is not None else _default_repo_ids(roots)
    datasets = [
        LeRobotDataset(repo_id=repo_id, root=root)
        for repo_id, root in zip(source_repo_ids, roots, strict=True)
    ]
    return merge_datasets(
        datasets=datasets,
        output_repo_id=output_repo_id,
        output_dir=output_dir,
        concatenate_videos=concatenate_videos,
        concatenate_data=concatenate_data,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge local LeRobot v3 datasets.")
    parser.add_argument("roots", nargs="+", type=Path, help="input LeRobot v3 dataset roots")
    parser.add_argument("--output-dir", required=True, type=Path, help="merged dataset output root")
    parser.add_argument(
        "--output-repo-id",
        default=None,
        help="merged dataset repo id, default: local/<output-dir-name>",
    )
    parser.add_argument(
        "--repo-ids",
        default=None,
        help='optional JSON list of repo ids for the input roots, e.g. \'["local/a", "local/b"]\'',
    )
    parser.add_argument(
        "--concatenate-videos",
        action="store_true",
        help="pack source videos into larger shard files instead of keeping one output file per source file",
    )
    parser.add_argument(
        "--concatenate-data",
        action="store_true",
        help="pack source parquet data into larger shard files instead of keeping one output file per source file",
    )
    args = parser.parse_args()

    roots = [root.expanduser().resolve() for root in args.roots]
    output_dir = args.output_dir.expanduser().resolve()
    output_repo_id = args.output_repo_id or f"local/{output_dir.name}"
    repo_ids = _parse_repo_ids(args.repo_ids, roots)

    merged_dataset = merge_local_lerobotv3_datasets(
        roots=roots,
        output_dir=output_dir,
        output_repo_id=output_repo_id,
        repo_ids=repo_ids,
        concatenate_videos=args.concatenate_videos,
        concatenate_data=args.concatenate_data,
    )

    print(f"Merged dataset saved to: {merged_dataset.root}")
    print(f"Repo id: {merged_dataset.repo_id}")
    print(f"Episodes: {merged_dataset.meta.total_episodes}")
    print(f"Frames: {merged_dataset.meta.total_frames}")
    print(f"Features: {sorted(merged_dataset.meta.features)}")


if __name__ == "__main__":
    main()
