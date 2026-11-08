"""Utilities for renaming episode folders.

Example:
    python astribot_lerobot/utils/sort_and_rename_folders.py --data-dir $HOME/datasets/astribot_task
"""

import re
import uuid
from pathlib import Path


EPISODE_DIR_PATTERN = re.compile(r"^episode_(\d+)$")


def sort_and_rename_episode_folders(data_dir: Path | str) -> tuple[int, list[tuple[str, str]]]:
    """Sort episode_XXXX folders by numeric id and rename them to episode_0000...

    Non-episode directories are ignored. The two-phase temporary rename avoids
    collisions such as episode_0001 -> episode_0000.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise RuntimeError(f"Dataset path does not exist: {data_dir}")

    episodes = []
    for child in data_dir.iterdir():
        if not child.is_dir():
            continue
        match = EPISODE_DIR_PATTERN.match(child.name)
        if match:
            episodes.append((int(match.group(1)), child))

    episodes.sort(key=lambda item: item[0])
    if not episodes:
        raise RuntimeError(f"No episode_XXXX folders found under: {data_dir}")

    target_paths = [data_dir / f"episode_{idx:04d}" for idx in range(len(episodes))]
    mapping_preview = [
        (source_path.name, target_path.name)
        for (_, source_path), target_path in zip(episodes, target_paths)
    ]

    if all(source_name == target_name for source_name, target_name in mapping_preview):
        return 0, mapping_preview

    temp_prefix = f"__tmp_episode_sort_{uuid.uuid4().hex}_"
    temp_records = []

    try:
        for idx, (_, source_path) in enumerate(episodes):
            temp_path = data_dir / f"{temp_prefix}{idx:04d}"
            if temp_path.exists():
                raise RuntimeError(f"Temporary folder already exists: {temp_path}")
            source_path.rename(temp_path)
            temp_records.append((source_path, temp_path))

        for target_path in target_paths:
            if target_path.exists():
                raise RuntimeError(f"Target folder already exists: {target_path}")

        for (_, temp_path), target_path in zip(temp_records, target_paths):
            temp_path.rename(target_path)

    except Exception:
        for original_path, temp_path in reversed(temp_records):
            if temp_path.exists() and not original_path.exists():
                temp_path.rename(original_path)
        raise

    changed_count = sum(1 for source_name, target_name in mapping_preview if source_name != target_name)
    return changed_count, mapping_preview


def sort_and_rename_folders(data_dir: Path) -> None:
    """CLI-compatible wrapper around sort_and_rename_episode_folders."""
    changed_count, mapping = sort_and_rename_episode_folders(data_dir)
    print(f"Episode folders sorted. Total: {len(mapping)}, renamed: {changed_count}.")
    for source_name, target_name in mapping:
        if source_name != target_name:
            print(f"{source_name} -> {target_name}")


def main() -> None:
    try:
        import tyro
    except ImportError:
        import argparse

        parser = argparse.ArgumentParser(description="Sort and rename episode_XXXX folders.")
        parser.add_argument("--data-dir", required=True, type=Path)
        args = parser.parse_args()
        sort_and_rename_folders(args.data_dir)
    else:
        tyro.cli(sort_and_rename_folders)


if __name__ == "__main__":
    main()
