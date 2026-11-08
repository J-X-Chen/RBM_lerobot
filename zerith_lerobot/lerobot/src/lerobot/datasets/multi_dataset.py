#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from bisect import bisect_right
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import datasets
import pandas as pd
import torch
import torch.utils

from lerobot.utils.constants import HF_LEROBOT_HOME

from .compute_stats import aggregate_stats
from .feature_utils import get_hf_features_from_features
from .lerobot_dataset import LeRobotDataset
from .video_utils import VideoFrame

logger = logging.getLogger(__name__)

_VISUAL_DTYPES = {"image", "video"}
_DEPTH_INFO_KEYS = (
    "depth_unit",
    "video.depth_min",
    "video.depth_max",
    "video.shift",
    "video.use_log",
)


def _is_depth_feature(feature: dict) -> bool:
    info = feature.get("info") or {}
    video_info = feature.get("video_info") or {}
    return bool(
        info.get("is_depth_map")
        or info.get("video.is_depth_map")
        or video_info.get("video.is_depth_map")
    )


def _features_compatible_for_runtime_merge(features_a: dict[str, dict], features_b: dict[str, dict]) -> bool:
    """Check tensor-layout compatibility while ignoring harmless video encoding metadata."""
    if set(features_a) != set(features_b):
        return False

    for key, feature_a in features_a.items():
        feature_b = features_b[key]
        if feature_a.get("dtype") != feature_b.get("dtype"):
            return False
        if tuple(feature_a.get("shape", ())) != tuple(feature_b.get("shape", ())):
            return False
        if feature_a.get("names") != feature_b.get("names"):
            return False

        if feature_a.get("dtype") not in _VISUAL_DTYPES:
            if feature_a != feature_b:
                return False
            continue

        if _is_depth_feature(feature_a) != _is_depth_feature(feature_b):
            return False
        if _is_depth_feature(feature_a):
            info_a = feature_a.get("info") or {}
            info_b = feature_b.get("info") or {}
            for info_key in _DEPTH_INFO_KEYS:
                if info_a.get(info_key) != info_b.get(info_key):
                    return False

    return True


class MergedLeRobotDataset(torch.utils.data.Dataset):
    """Runtime view that concatenates already-instantiated ``LeRobotDataset`` objects.

    Unlike :class:`MultiLeRobotDataset`, this accepts arbitrary local roots. It keeps
    one reader per source dataset and builds a merged metadata view so the training
    sampler and policy initialization see one continuous dataset.
    """

    def __init__(
        self,
        datasets: list[LeRobotDataset],
        repo_id: str | None = None,
    ) -> None:
        super().__init__()
        if not datasets:
            raise ValueError("MergedLeRobotDataset requires at least one dataset.")

        self._datasets = datasets
        self.repo_id = repo_id or "+".join(dict.fromkeys(ds.repo_id for ds in datasets))
        self.root = [ds.root for ds in datasets]
        self.delta_timestamps = datasets[0].delta_timestamps
        self.tolerance_s = datasets[0].tolerance_s
        self.image_transforms = datasets[0].image_transforms
        self.episodes = None
        self._hf_dataset = None

        self._validate_compatible_datasets()
        self._frame_offsets = self._build_frame_offsets()
        self._episode_sources: list[tuple[int, int]] = []
        self._source_episode_to_global: list[dict[int, int]] = []
        self._local_task_to_global: list[dict[int, int]] = []
        self.meta = self._build_merged_meta()
        self._validate_merged_episode_alignment()

    def _validate_compatible_datasets(self) -> None:
        reference = self._datasets[0]
        ref_features = reference.meta.features
        ref_robot_type = reference.meta.robot_type

        for dataset in self._datasets[1:]:
            if dataset.meta.fps != reference.meta.fps:
                raise ValueError(
                    "All merged datasets must have the same fps. "
                    f"Got {reference.root}: {reference.meta.fps}, {dataset.root}: {dataset.meta.fps}."
                )

            if dataset.meta.robot_type != ref_robot_type:
                raise ValueError(
                    "All merged datasets must have the same robot_type to avoid action/state "
                    f"misalignment. Got {reference.root}: {ref_robot_type!r}, "
                    f"{dataset.root}: {dataset.meta.robot_type!r}."
                )

            if not _features_compatible_for_runtime_merge(ref_features, dataset.meta.features):
                raise ValueError(
                    "All merged datasets must have the same selected feature schema to avoid "
                    "action/state/camera misalignment. "
                    f"{reference.root} features: {ref_features}. "
                    f"{dataset.root} features: {dataset.meta.features}. "
                    "Video codec/encoding metadata is ignored, but feature names, shapes, dtypes, "
                    "and action/state dimension names must match."
                )

    def _build_frame_offsets(self) -> list[int]:
        offsets = [0]
        total = 0
        for dataset in self._datasets:
            total += dataset.num_frames
            offsets.append(total)
        return offsets

    @staticmethod
    def _selected_episodes(dataset: LeRobotDataset) -> list[int]:
        if dataset.episodes is None:
            return list(range(dataset.meta.total_episodes))
        return sorted(int(ep) for ep in dataset.episodes)

    def _selected_episode_rows(self, dataset: LeRobotDataset) -> list[dict]:
        selected_episodes = self._selected_episodes(dataset)
        episode_frame = dataset.meta.episodes.to_pandas()
        if episode_frame["episode_index"].duplicated().any():
            duplicates = sorted(
                int(ep) for ep in episode_frame.loc[episode_frame["episode_index"].duplicated(), "episode_index"]
            )
            raise ValueError(f"Dataset {dataset.root} has duplicate episode_index rows: {duplicates}.")

        rows_by_episode = {
            int(row["episode_index"]): row.to_dict()
            for _, row in episode_frame.iterrows()
        }
        missing = [ep for ep in selected_episodes if ep not in rows_by_episode]
        if missing:
            raise ValueError(f"Dataset {dataset.root} is missing metadata rows for episodes: {missing}.")
        return [rows_by_episode[episode_index] for episode_index in selected_episodes]

    def _build_merged_meta(self):
        merged_meta = deepcopy(self._datasets[0].meta)
        merged_meta.repo_id = self.repo_id
        merged_meta.root = Path("__runtime_merged_lerobot_dataset__")
        merged_meta.info = deepcopy(self._datasets[0].meta.info)

        task_to_global_index: dict[str, int] = {}
        episode_rows: list[dict] = []
        frame_offset = 0

        for dataset_index, dataset in enumerate(self._datasets):
            local_task_to_global: dict[int, int] = {}
            for task_name, task_row in dataset.meta.tasks.iterrows():
                if task_name not in task_to_global_index:
                    task_to_global_index[task_name] = len(task_to_global_index)
                local_task_to_global[int(task_row["task_index"])] = task_to_global_index[task_name]
            self._local_task_to_global.append(local_task_to_global)

            selected_episodes = self._selected_episodes(dataset)
            source_to_global: dict[int, int] = {}
            self._source_episode_to_global.append(source_to_global)
            if not selected_episodes:
                continue

            child_frame_count = 0
            for row in self._selected_episode_rows(dataset):
                source_episode = int(row["episode_index"])
                global_episode = len(episode_rows)
                source_to_global[source_episode] = global_episode
                self._episode_sources.append((dataset_index, source_episode))

                episode_length = int(row["dataset_to_index"]) - int(row["dataset_from_index"])
                row["episode_index"] = global_episode
                row["dataset_from_index"] = frame_offset
                row["dataset_to_index"] = frame_offset + episode_length
                if "task_index" in row:
                    row["task_index"] = local_task_to_global[int(row["task_index"])]
                episode_rows.append(row)
                frame_offset += episode_length
                child_frame_count += episode_length

            if child_frame_count != dataset.num_frames:
                raise ValueError(
                    "Selected episode metadata length does not match loaded frame count for "
                    f"{dataset.root}: metadata={child_frame_count}, loaded={dataset.num_frames}."
                )

        if not episode_rows:
            raise ValueError("Merged datasets contain no selected episodes.")

        merged_meta.info.total_episodes = len(episode_rows)
        merged_meta.info.total_frames = frame_offset
        merged_meta.info.total_tasks = len(task_to_global_index)
        merged_meta.info.splits = {"train": f"0:{len(episode_rows)}"}
        merged_meta.tasks = pd.DataFrame(
            {"task_index": list(task_to_global_index.values())},
            index=pd.Index(list(task_to_global_index.keys()), name="task"),
        )
        merged_meta.episodes = datasets.Dataset.from_pandas(
            pd.DataFrame(episode_rows), preserve_index=False
        )

        stats_list = [dataset.meta.stats for dataset in self._datasets if dataset.meta.stats is not None]
        merged_meta.stats = aggregate_stats(stats_list) if len(stats_list) == len(self._datasets) else None
        return merged_meta

    def _validate_merged_episode_alignment(self) -> None:
        if len(self._episode_sources) != self.meta.total_episodes:
            raise ValueError(
                "Merged episode source mapping is inconsistent: "
                f"{len(self._episode_sources)} sources for {self.meta.total_episodes} episodes."
            )

        previous_end = 0
        for expected_episode_index, row in enumerate(self.meta.episodes):
            episode_index = int(row["episode_index"])
            start = int(row["dataset_from_index"])
            end = int(row["dataset_to_index"])
            if episode_index != expected_episode_index:
                raise ValueError(
                    "Merged episode metadata is not contiguous: "
                    f"row {expected_episode_index} has episode_index={episode_index}."
                )
            if start != previous_end:
                raise ValueError(
                    "Merged episode frame ranges are not contiguous: "
                    f"episode {episode_index} starts at {start}, expected {previous_end}."
                )
            if end <= start:
                raise ValueError(
                    f"Merged episode {episode_index} has invalid frame range [{start}, {end})."
                )
            previous_end = end

        if previous_end != self.num_frames:
            raise ValueError(
                "Merged episode frame ranges do not match dataset length: "
                f"metadata ends at {previous_end}, dataset has {self.num_frames} frames."
            )

    def set_image_transforms(self, image_transforms: Callable | None) -> None:
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self.image_transforms = image_transforms
        for dataset in self._datasets:
            dataset.set_image_transforms(image_transforms)

    def clear_image_transforms(self) -> None:
        self.set_image_transforms(None)

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def video(self) -> bool:
        return len(self.meta.video_keys) > 0

    @property
    def camera_keys(self) -> list[str]:
        return self.meta.camera_keys

    @property
    def video_frame_keys(self) -> list[str]:
        return self.meta.video_keys

    @property
    def num_frames(self) -> int:
        return self._frame_offsets[-1]

    @property
    def num_episodes(self) -> int:
        return self.meta.total_episodes

    @property
    def absolute_to_relative_idx(self) -> None:
        return None

    @property
    def hf_dataset(self) -> datasets.Dataset:
        if self._hf_dataset is None:
            self._hf_dataset = datasets.concatenate_datasets(
                [dataset.hf_dataset for dataset in self._datasets]
            )
        return self._hf_dataset

    def _locate_frame(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")

        dataset_index = bisect_right(self._frame_offsets, idx) - 1
        local_index = idx - self._frame_offsets[dataset_index]
        return dataset_index, local_index

    def __len__(self) -> int:
        return self.num_frames

    @staticmethod
    def _scalar_like(value, replacement: int):
        if isinstance(value, torch.Tensor):
            return torch.as_tensor(replacement, dtype=value.dtype, device=value.device)
        return replacement

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        dataset_index, local_index = self._locate_frame(idx)
        item = dict(self._datasets[dataset_index][local_index])

        if "episode_index" in item:
            local_episode_value = item["episode_index"]
            local_episode = int(
                local_episode_value.item()
                if isinstance(local_episode_value, torch.Tensor)
                else local_episode_value
            )
            global_episode = self._source_episode_to_global[dataset_index].get(local_episode, local_episode)
            item["episode_index"] = self._scalar_like(item["episode_index"], global_episode)
        if "task_index" in item:
            local_task_value = item["task_index"]
            local_task = int(
                local_task_value.item() if isinstance(local_task_value, torch.Tensor) else local_task_value
            )
            global_task = self._local_task_to_global[dataset_index].get(local_task, local_task)
            item["task_index"] = self._scalar_like(item["task_index"], global_task)
        if "index" in item:
            item["index"] = self._scalar_like(item["index"], idx)
        return item

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        dataset_index, source_episode = self._episode_sources[ep_idx]
        reader = self._datasets[dataset_index]._ensure_reader()  # noqa: SLF001
        return reader._query_videos(query_timestamps, source_episode)  # noqa: SLF001

    def select_columns(self, column_names: str | list[str]):
        return self.hf_dataset.select_columns(column_names)

    def get_raw_item(self, idx: int) -> dict:
        return self.hf_dataset[idx]

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository ID: '{self.repo_id}',\n"
            f"  Number of datasets: {len(self._datasets)},\n"
            f"  Number of selected episodes: {self.num_episodes},\n"
            f"  Number of selected samples: {self.num_frames},\n"
            f"  Features: {feature_keys},\n"
            f")"
        )


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self._datasets = [
            LeRobotDataset(
                repo_id,
                root=self.root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
            )
            for repo_id in repo_ids
        ]

        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            if extra_keys:
                logger.warning(
                    f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                    "other datasets."
                )
                self.disabled_features.update(extra_keys)

        self.delta_timestamps = delta_timestamps
        # TODO(rcadene, aliberts): We should not perform this aggregation for datasets
        # with multiple robots of different ranges. Instead we should have one normalization
        # per robot.
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])
        self.set_image_transforms(image_transforms)

    def set_image_transforms(self, image_transforms: Callable | None) -> None:
        """Replace the transform for this dataset and its children."""
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self.image_transforms = image_transforms
        for dataset in getattr(self, "_datasets", []):
            dataset.set_image_transforms(self.image_transforms)

    def clear_image_transforms(self) -> None:
        """Remove the transform from this dataset and its children."""
        self.set_image_transforms(None)

    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info.fps

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return len(self._datasets[0].meta.video_keys) > 0

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update(
                {
                    k: v
                    for k, v in get_hf_features_from_features(dataset.features).items()
                    if k not in self.disabled_features
                }
            )
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image | VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
