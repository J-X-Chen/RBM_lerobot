#!/usr/bin/env python3
"""Interactively rename every task in a local LeRobot v3 dataset.

The script updates task text in ``meta/tasks.parquet`` and
``meta/episodes/**/*.parquet``. If multiple old tasks are renamed to the same
text, it also remaps ``task_index`` in ``data/**/*.parquet`` and refreshes the
task-index statistics.

Example:
    python astribot_lerobot/utils/change_lerobotv3_instruction.py /path/to/dataset

For every task, enter the replacement as a JSON string with ASCII double
quotes, for example: ``"Pick up the bottle."``. Enter the original task in
double quotes to keep it unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


class DatasetFormatError(ValueError):
    """Raised when a directory is not a consistent LeRobot v3 dataset."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetFormatError(f"无法读取 JSON 文件: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetFormatError(f"JSON 顶层必须是对象: {path}")
    return value


def _task_list(value: Any, *, context: str) -> list[str]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise DatasetFormatError(f"{context} 的 tasks 必须是字符串列表，实际为 {type(value).__name__}")
    tasks = list(value)
    if not tasks or not all(isinstance(task, str) and task for task in tasks):
        raise DatasetFormatError(f"{context} 的 tasks 必须是非空字符串列表")
    if len(tasks) != len(set(tasks)):
        raise DatasetFormatError(f"{context} 的 tasks 含有重复文本: {tasks}")
    return tasks


def _load_task_table(tasks_path: Path) -> tuple[pd.DataFrame, list[str], dict[int, str]]:
    try:
        tasks_df = pd.read_parquet(tasks_path)
    except Exception as exc:
        raise DatasetFormatError(f"无法读取 {tasks_path}: {exc}") from exc

    if "task_index" not in tasks_df.columns:
        raise DatasetFormatError(f"{tasks_path} 缺少 task_index 列")
    task_names = tasks_df.index.tolist()
    if not task_names or not all(isinstance(task, str) and task for task in task_names):
        raise DatasetFormatError(f"{tasks_path} 的索引必须是非空 task 字符串")
    if len(task_names) != len(set(task_names)):
        raise DatasetFormatError(f"{tasks_path} 包含重复 task 文本")

    try:
        task_indices = [int(value) for value in tasks_df["task_index"].tolist()]
    except (TypeError, ValueError) as exc:
        raise DatasetFormatError(f"{tasks_path} 的 task_index 必须是整数") from exc

    expected_indices = list(range(len(task_names)))
    if task_indices != expected_indices:
        raise DatasetFormatError(
            f"{tasks_path} 的行顺序必须与连续 task_index 对应；"
            f"期望 {expected_indices}，实际 {task_indices}"
        )
    return tasks_df, task_names, dict(zip(task_indices, task_names, strict=True))


def validate_lerobot_v3_dataset(root: Path) -> dict[str, Any]:
    """Validate task-related LeRobot v3 files and return their loaded metadata."""
    if not root.is_dir():
        raise DatasetFormatError(f"数据集目录不存在: {root}")

    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.parquet"
    episodes_dir = root / "meta" / "episodes"
    data_dir = root / "data"
    for path in (info_path, tasks_path):
        if not path.is_file():
            raise DatasetFormatError(f"缺少 LeRobot v3 文件: {path}")
    for path in (episodes_dir, data_dir):
        if not path.is_dir():
            raise DatasetFormatError(f"缺少 LeRobot v3 目录: {path}")

    info = _read_json(info_path)
    version = str(info.get("codebase_version", ""))
    if not (version.startswith("v3.") or version.startswith("3.")):
        raise DatasetFormatError(
            f"仅支持 LeRobot v3 数据集，info.json 中 codebase_version={version!r}"
        )

    tasks_df, task_names, task_by_index = _load_task_table(tasks_path)
    if int(info.get("total_tasks", -1)) != len(task_names):
        raise DatasetFormatError(
            f"info.json total_tasks={info.get('total_tasks')}，但 tasks.parquet 有 {len(task_names)} 个 task"
        )

    episode_files = sorted(episodes_dir.rglob("*.parquet"))
    data_files = sorted(data_dir.rglob("*.parquet"))
    if not episode_files:
        raise DatasetFormatError(f"没有找到 episode 元数据: {episodes_dir}/**/*.parquet")
    if not data_files:
        raise DatasetFormatError(f"没有找到逐帧数据: {data_dir}/**/*.parquet")

    episode_tasks: dict[int, list[str]] = {}
    known_tasks = set(task_names)
    for path in episode_files:
        try:
            frame = pd.read_parquet(path, columns=["episode_index", "tasks"])
        except Exception as exc:
            raise DatasetFormatError(f"无法读取 {path} 的 episode_index/tasks: {exc}") from exc
        for row_index, row in frame.iterrows():
            episode_index = int(row["episode_index"])
            if episode_index in episode_tasks:
                raise DatasetFormatError(f"episode_index={episode_index} 在 episode 元数据中重复")
            tasks = _task_list(row["tasks"], context=f"{path} 第 {row_index} 行")
            unknown = set(tasks) - known_tasks
            if unknown:
                raise DatasetFormatError(
                    f"episode_index={episode_index} 引用了 tasks.parquet 中不存在的 task: {sorted(unknown)}"
                )
            episode_tasks[episode_index] = tasks

    data_tasks_by_episode: dict[int, set[str]] = defaultdict(set)
    total_frames = 0
    for path in data_files:
        try:
            frame = pd.read_parquet(path, columns=["episode_index", "task_index"])
        except Exception as exc:
            raise DatasetFormatError(f"无法读取 {path} 的 episode_index/task_index: {exc}") from exc
        if frame[["episode_index", "task_index"]].isnull().any().any():
            raise DatasetFormatError(f"{path} 的 episode_index/task_index 含有空值")
        total_frames += len(frame)
        for (episode_index, task_index), _count in frame.value_counts(
            ["episode_index", "task_index"]
        ).items():
            episode_index = int(episode_index)
            task_index = int(task_index)
            if task_index not in task_by_index:
                raise DatasetFormatError(f"{path} 引用了不存在的 task_index={task_index}")
            data_tasks_by_episode[episode_index].add(task_by_index[task_index])

    episode_indices = set(episode_tasks)
    data_episode_indices = set(data_tasks_by_episode)
    if episode_indices != data_episode_indices:
        raise DatasetFormatError(
            "episode 元数据与逐帧数据的 episode_index 不一致；"
            f"仅元数据中存在={sorted(episode_indices - data_episode_indices)}，"
            f"仅逐帧数据中存在={sorted(data_episode_indices - episode_indices)}"
        )

    for episode_index, tasks in episode_tasks.items():
        frame_tasks = data_tasks_by_episode[episode_index]
        if set(tasks) != frame_tasks:
            raise DatasetFormatError(
                f"episode_index={episode_index} 的 tasks={tasks}，"
                f"但逐帧 task_index 对应文本={sorted(frame_tasks)}"
            )

    if int(info.get("total_episodes", -1)) != len(episode_tasks):
        raise DatasetFormatError(
            f"info.json total_episodes={info.get('total_episodes')}，"
            f"但 episode 元数据有 {len(episode_tasks)} 条"
        )
    if int(info.get("total_frames", -1)) != total_frames:
        raise DatasetFormatError(
            f"info.json total_frames={info.get('total_frames')}，但逐帧数据有 {total_frames} 行"
        )

    return {
        "info": info,
        "tasks_df": tasks_df,
        "task_names": task_names,
        "task_by_index": task_by_index,
        "episode_tasks": episode_tasks,
        "episode_files": episode_files,
        "data_files": data_files,
    }


def _prompt_quoted_task(old_task: str, position: int, total: int) -> str:
    print(f"\n[{position}/{total}] 当前 task:")
    print(json.dumps(old_task, ensure_ascii=False))
    while True:
        try:
            raw = input('请输入新的 task（必须使用英文双引号，例如 "fghij"）: ').strip()
        except (EOFError, KeyboardInterrupt) as exc:
            print("\n已取消；尚未修改任何文件。")
            raise SystemExit(130) from exc

        if len(raw) < 2 or not raw.startswith('"') or not raw.endswith('"'):
            print('输入无效：必须在文本两端使用英文双引号 "，请重新输入。')
            continue
        try:
            new_task = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"输入无效：不是合法的双引号 JSON 字符串（{exc.msg}），请重新输入。")
            continue
        if not isinstance(new_task, str):
            print("输入无效：双引号内必须是字符串，请重新输入。")
            continue
        if not new_task or not new_task.strip():
            print("输入无效：task 不能为空，请重新输入。")
            continue
        if new_task != new_task.strip():
            print("输入无效：task 首尾不能包含空白字符，请重新输入。")
            continue
        return new_task


def _build_new_task_table(
    task_names: list[str], rename_by_task: dict[str, str]
) -> tuple[pd.DataFrame, dict[int, int], dict[int, str]]:
    new_task_names: list[str] = []
    new_index_by_name: dict[str, int] = {}
    old_index_to_new_index: dict[int, int] = {}

    for old_index, old_task in enumerate(task_names):
        new_task = rename_by_task[old_task]
        if new_task not in new_index_by_name:
            new_index_by_name[new_task] = len(new_task_names)
            new_task_names.append(new_task)
        old_index_to_new_index[old_index] = new_index_by_name[new_task]

    new_tasks_df = pd.DataFrame(
        {"task_index": range(len(new_task_names))},
        index=pd.Index(new_task_names, name="task"),
    )
    new_task_by_index = dict(enumerate(new_task_names))
    return new_tasks_df, old_index_to_new_index, new_task_by_index


def _counter_quantile(counter: Counter[int], quantile: float) -> float:
    count = sum(counter.values())
    if count <= 0:
        raise ValueError("无法计算空 task_index 的统计值")

    position = quantile * (count - 1)
    lower_rank = int(np.floor(position))
    upper_rank = int(np.ceil(position))

    def value_at_rank(rank: int) -> int:
        seen = 0
        for value, value_count in sorted(counter.items()):
            seen += value_count
            if rank < seen:
                return value
        raise RuntimeError("task_index quantile rank 越界")

    lower = value_at_rank(lower_rank)
    upper = value_at_rank(upper_rank)
    return float(lower + (upper - lower) * (position - lower_rank))


def _task_index_stats(counter: Counter[int]) -> dict[str, list[int | float]]:
    count = sum(counter.values())
    if count <= 0:
        raise ValueError("无法计算空 task_index 的统计值")
    mean = sum(value * value_count for value, value_count in counter.items()) / count
    variance = sum((value - mean) ** 2 * value_count for value, value_count in counter.items()) / count
    stats: dict[str, list[int | float]] = {
        "min": [min(counter)],
        "max": [max(counter)],
        "mean": [float(mean)],
        "std": [float(np.sqrt(variance))],
        "count": [count],
    }
    stats.update({key: [_counter_quantile(counter, value)] for key, value in QUANTILES.items()})
    return stats


def _rewrite_data_file(
    source: Path,
    destination: Path,
    old_index_to_new_index: dict[int, int],
    global_counter: Counter[int],
    episode_counters: dict[int, Counter[int]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    parquet_file = pq.ParquetFile(source)
    schema = parquet_file.schema_arrow
    column_index = schema.get_field_index("task_index")
    if column_index < 0:
        raise DatasetFormatError(f"{source} 缺少 task_index 列")
    task_field = schema.field(column_index)

    with pq.ParquetWriter(destination, schema) as writer:
        for row_group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_index)
            old_indices = table.column("task_index").combine_chunks().to_numpy(zero_copy_only=False)
            episode_indices = table.column("episode_index").combine_chunks().to_numpy(zero_copy_only=False)
            try:
                new_indices = np.fromiter(
                    (old_index_to_new_index[int(value)] for value in old_indices),
                    dtype=old_indices.dtype,
                    count=len(old_indices),
                )
            except KeyError as exc:
                raise DatasetFormatError(f"{source} 引用了未知 task_index={exc.args[0]}") from exc

            new_column = pa.array(new_indices, type=task_field.type)
            table = table.set_column(column_index, task_field, new_column)
            writer.write_table(table)

            if len(new_indices):
                values, counts = np.unique(new_indices, return_counts=True)
                global_counter.update(
                    {int(value): int(count) for value, count in zip(values, counts, strict=True)}
                )
                pairs = np.column_stack((episode_indices, new_indices))
                unique_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)
                for (episode_index, task_index), count in zip(unique_pairs, pair_counts, strict=True):
                    episode_counters[int(episode_index)][int(task_index)] += int(count)


def _rewrite_episode_file(
    source: Path,
    destination: Path,
    episode_new_tasks: dict[int, list[str]],
    episode_counters: dict[int, Counter[int]] | None,
) -> None:
    frame = pd.read_parquet(source)
    frame["tasks"] = frame["episode_index"].apply(lambda value: episode_new_tasks[int(value)])

    if episode_counters is not None:
        stats_by_episode = {
            int(episode_index): _task_index_stats(episode_counters[int(episode_index)])
            for episode_index in frame["episode_index"]
        }
        for stat_name in ("min", "max", "mean", "std", "count", *QUANTILES):
            column = f"stats/task_index/{stat_name}"
            if column not in frame.columns:
                continue
            values = []
            for row_index, episode_index_value in frame["episode_index"].items():
                old_value = np.asarray(frame.at[row_index, column])
                dtype = old_value.dtype if old_value.size else None
                value = stats_by_episode[int(episode_index_value)][stat_name][0]
                values.append(np.asarray([value], dtype=dtype))
            # Construct the complete object column at once. Assigning a one-element
            # ndarray through DataFrame.at can silently unwrap it into a scalar.
            frame[column] = pd.Series(values, index=frame.index, dtype=object)

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=4)
        file.write("\n")


def _commit_with_rollback(root: Path, staging_root: Path, relative_paths: list[Path]) -> None:
    backup_root = staging_root / "original"
    replaced: list[tuple[Path, Path]] = []
    try:
        for relative_path in relative_paths:
            target = root / relative_path
            staged = staging_root / "new" / relative_path
            backup = backup_root / relative_path
            backup.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, backup)
            try:
                os.replace(staged, target)
            except Exception:
                os.replace(backup, target)
                raise
            replaced.append((target, backup))
    except Exception:
        for target, backup in reversed(replaced):
            if backup.exists():
                if target.exists():
                    target.unlink()
                os.replace(backup, target)
        raise


def change_tasks_interactively(root: Path) -> None:
    root = root.expanduser().resolve()
    metadata = validate_lerobot_v3_dataset(root)
    task_names: list[str] = metadata["task_names"]

    print(f"已确认 LeRobot v3 数据集: {root}")
    print(f"共发现 {len(task_names)} 个 task。若不想修改某项，请输入带双引号的原文本。")
    rename_by_task = {
        old_task: _prompt_quoted_task(old_task, position, len(task_names))
        for position, old_task in enumerate(task_names, start=1)
    }

    if all(rename_by_task[task] == task for task in task_names):
        print("所有 task 均保持不变，未写入任何文件。")
        return

    new_tasks_df, old_index_to_new_index, new_task_by_index = _build_new_task_table(
        task_names, rename_by_task
    )
    indices_changed = any(old_index != new_index for old_index, new_index in old_index_to_new_index.items())

    episode_new_tasks = {
        episode_index: list(dict.fromkeys(rename_by_task[task] for task in tasks))
        for episode_index, tasks in metadata["episode_tasks"].items()
    }

    staging_root = Path(tempfile.mkdtemp(prefix=".change_tasks_staging_", dir=root))
    relative_paths: list[Path] = []
    try:
        staged_new_root = staging_root / "new"

        tasks_relative = Path("meta/tasks.parquet")
        staged_tasks_path = staged_new_root / tasks_relative
        staged_tasks_path.parent.mkdir(parents=True, exist_ok=True)
        new_tasks_df.to_parquet(staged_tasks_path)
        relative_paths.append(tasks_relative)

        info = dict(metadata["info"])
        info["total_tasks"] = len(new_tasks_df)
        info_relative = Path("meta/info.json")
        _write_json(staged_new_root / info_relative, info)
        relative_paths.append(info_relative)

        global_counter: Counter[int] = Counter()
        episode_counters: dict[int, Counter[int]] = defaultdict(Counter)
        if indices_changed:
            for source in metadata["data_files"]:
                relative_path = source.relative_to(root)
                _rewrite_data_file(
                    source,
                    staged_new_root / relative_path,
                    old_index_to_new_index,
                    global_counter,
                    episode_counters,
                )
                relative_paths.append(relative_path)

            for episode_index, tasks in episode_new_tasks.items():
                frame_task_names = {
                    new_task_by_index[task_index] for task_index in episode_counters[episode_index]
                }
                if set(tasks) != frame_task_names:
                    raise DatasetFormatError(
                        f"重命名后 episode_index={episode_index} 的 episode tasks={tasks}，"
                        f"逐帧 tasks={sorted(frame_task_names)}，拒绝写入"
                    )

        for source in metadata["episode_files"]:
            relative_path = source.relative_to(root)
            _rewrite_episode_file(
                source,
                staged_new_root / relative_path,
                episode_new_tasks,
                episode_counters if indices_changed else None,
            )
            relative_paths.append(relative_path)

        stats_path = root / "meta" / "stats.json"
        if indices_changed and stats_path.is_file():
            stats = _read_json(stats_path)
            stats["task_index"] = _task_index_stats(global_counter)
            stats_relative = Path("meta/stats.json")
            _write_json(staged_new_root / stats_relative, stats)
            relative_paths.append(stats_relative)

        _commit_with_rollback(root, staging_root, relative_paths)
        try:
            validated = validate_lerobot_v3_dataset(root)
            actual_tasks = validated["task_names"]
            expected_tasks = new_tasks_df.index.tolist()
            if actual_tasks != expected_tasks:
                raise DatasetFormatError(
                    f"写入后 tasks.parquet 不符合预期: {actual_tasks} != {expected_tasks}"
                )
        except Exception:
            backup_root = staging_root / "original"
            for relative_path in reversed(relative_paths):
                backup = backup_root / relative_path
                target = root / relative_path
                if backup.exists():
                    if target.exists():
                        target.unlink()
                    os.replace(backup, target)
            raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    print("\n修改完成，并已通过 tasks/episodes/data 一致性校验。")
    print(f"task 数量: {len(task_names)} -> {len(new_tasks_df)}")
    for old_task in task_names:
        print(
            f"  {json.dumps(old_task, ensure_ascii=False)} -> "
            f"{json.dumps(rename_by_task[old_task], ensure_ascii=False)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="逐个交互修改本地 LeRobot v3 数据集中的所有 task 文本。"
    )
    parser.add_argument("dataset_root", type=Path, help="本地 LeRobot v3 数据集根目录")
    args = parser.parse_args()

    try:
        change_tasks_interactively(args.dataset_root)
    except DatasetFormatError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
