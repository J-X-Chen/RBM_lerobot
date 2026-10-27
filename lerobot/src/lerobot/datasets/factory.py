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
import ast
import json
import logging
import math
from pathlib import Path
from typing import Literal, cast

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_IMAGE, OBS_PREFIX, OBS_STATE, REWARD

from .dataset_metadata import LeRobotDatasetMetadata
from .feature_selection import filter_dataset_metadata_features
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MergedLeRobotDataset, MultiLeRobotDataset
from .storage import DEFAULT_STORAGE_FORMAT, is_bucket_root, load_dataset_metadata
from .streaming_dataset import DEFAULT_STREAMING_SEED, StreamingLeRobotDataset
from .utils import resolve_episode_indices

logger = logging.getLogger(__name__)


def resolve_delta_timestamps(
    cfg: PreTrainedConfig | RewardModelConfig,
    ds_meta: LeRobotDatasetMetadata,
    rename_map: dict[str, str] | None = None,
) -> dict[str, list[float]] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the config.

    Args:
        cfg (PreTrainedConfig | RewardModelConfig): The config to read delta_indices from. Both
            ``PreTrainedConfig`` and concrete ``RewardModelConfig`` subclasses expose the
            ``{observation,action,reward}_delta_indices`` properties used below.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list[float]] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    # Only policies that opt into modality-specific history (currently Pi05 with MEM)
    # define these; everything else falls back to the shared observation indices.
    explicit_image_indices = getattr(cfg, "image_observation_delta_indices", None)
    image_indices = (
        explicit_image_indices if explicit_image_indices is not None else cfg.observation_delta_indices
    )
    explicit_state_indices = getattr(cfg, "state_observation_delta_indices", None)
    state_indices = (
        explicit_state_indices if explicit_state_indices is not None else cfg.observation_delta_indices
    )

    delta_timestamps: dict[str, list[float]] = {}
    matched_image_keys: list[str] = []
    for key in ds_meta.features:
        policy_key = (rename_map or {}).get(key, key)
        if policy_key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if policy_key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        # `OBS_IMAGE` matches both the `observation.image` and `observation.images.<cam>`
        # conventions; matching `OBS_IMAGES` alone would silently give singular-key
        # datasets no image history at all.
        if policy_key.startswith(OBS_IMAGE):
            indices = image_indices
            matched_image_keys.append(key)
        elif policy_key == OBS_STATE:
            indices = state_indices
        else:
            indices = cfg.observation_delta_indices if policy_key.startswith(OBS_PREFIX) else None
        if indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in indices]

    # A policy asking for an image history that no dataset key can supply would train
    # on single frames without any error, so fail instead of degrading silently.
    if explicit_image_indices is not None and len(explicit_image_indices) > 1 and not matched_image_keys:
        raise ValueError(
            f"{type(cfg).__name__} requests {len(explicit_image_indices)} history frames per camera, but no "
            f"dataset feature maps to an image key. Dataset features: {sorted(ds_meta.features)}. "
            "Image keys must be named `observation.image*` after applying `--rename_map`."
        )

    if len(delta_timestamps) == 0:
        return None

    return delta_timestamps


def _coerce_dataset_root_list(parsed: object) -> list[Path]:
    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ValueError("dataset.root list must be a non-empty list.")
    if not all(isinstance(item, (str, Path)) for item in parsed):
        raise ValueError("dataset.root list must contain only paths as strings.")
    return [Path(item) for item in parsed]


def _parse_dataset_root_list(text: str) -> list[Path]:
    parse_errors: list[Exception] = []
    for parser in (json.loads, ast.literal_eval):
        try:
            return _coerce_dataset_root_list(parser(text))
        except (json.JSONDecodeError, SyntaxError, ValueError) as exc:
            parse_errors.append(exc)

    inner = text[1:-1].strip() if text.endswith("]") else ""
    if inner:
        parts = [part.strip().strip("'\"") for part in inner.split(",")]
        parts = [part for part in parts if part]
        if parts:
            return [Path(part) for part in parts]

    raise ValueError(
        "dataset.root looks like a list but could not be parsed. "
        "Use --dataset.root='[\"/path/a\", \"/path/b\"]' or --dataset.root='[/path/a,/path/b]'."
    ) from parse_errors[-1]


def _normalize_dataset_roots(root: str | Path | list[str | Path] | None) -> list[str | Path | None]:
    if root is None:
        return [None]
    if isinstance(root, (list, tuple)):
        if not root:
            raise ValueError("dataset.root list must not be empty.")
        return [Path(item) for item in root]
    if isinstance(root, Path):
        return [root]

    text = str(root).strip()
    if text.startswith("["):
        return _parse_dataset_root_list(text)
    return [root]


def _normalize_dataset_repo_ids(repo_id: str | list[str], num_roots: int) -> list[str]:
    if isinstance(repo_id, str):
        return [repo_id] * num_roots
    repo_ids = list(repo_id)
    if len(repo_ids) != num_roots:
        raise ValueError(
            f"dataset.repo_id and dataset.root must have the same length when both are lists; "
            f"got {len(repo_ids)} repo ids and {num_roots} roots."
        )
    return repo_ids


def _make_image_transforms(cfg: TrainPipelineConfig) -> ImageTransforms | None:
    return (
        ImageTransforms(cfg.dataset.image_transforms)
        if cfg.dataset.image_transforms.enable and cfg.dataset.image_transforms.backend == "dataloader"
        else None
    )


def _make_single_dataset(
    cfg: TrainPipelineConfig,
    *,
    repo_id: str,
    root: str | Path | None,
    episodes: list[int] | None,
    image_transforms: ImageTransforms | None,
) -> LeRobotDataset | StreamingLeRobotDataset:
    if isinstance(repo_id, str):
        root, repo_type_name = root, cfg.dataset.repo_type
        if cfg.dataset.streaming and is_bucket_root(repo_id, root):
            # Streaming reads default-format buckets directly, so a bucket URI root selects
            # bucket mode instead of the object-store localization used by other formats.
            root, repo_type_name = None, "bucket"
        repo_type = cast(Literal["dataset", "bucket"], repo_type_name)
        # Storage-aware loader: same as LeRobotDatasetMetadata(...), plus support
        # for datasets whose root is an object-store URI (e.g. ``hf://``).
        ds_meta = load_dataset_metadata(
            repo_id,
            root=root,
            revision=cfg.dataset.revision,
            repo_type=repo_type,
        )
        filter_dataset_metadata_features(ds_meta, cfg.dataset.feature_names)
        delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, ds_meta, cfg.rename_map)
        episodes = resolve_episode_indices(episodes, ds_meta.total_episodes, cfg.dataset.exclude_episodes)
        if cfg.dataset.streaming and ds_meta.storage_format != DEFAULT_STORAGE_FORMAT:
            raise ValueError(
                f"dataset.streaming=True is not supported for storage_format="
                f"{ds_meta.storage_format!r}: StreamingLeRobotDataset only reads the default "
                f"{DEFAULT_STORAGE_FORMAT!r} layout. Note that some formats (e.g. 'lance') "
                "support remote map-style access without streaming mode."
            )
        if not cfg.dataset.streaming:
            if repo_type == "bucket" and ds_meta.storage_format == DEFAULT_STORAGE_FORMAT:
                raise ValueError(
                    f"repo_type='bucket' is streaming-only for the default {DEFAULT_STORAGE_FORMAT!r} "
                    "storage format: set dataset.streaming=true to train from an HF Storage Bucket."
                )
            dataset = LeRobotDataset(
                repo_id,
                root=root,
                episodes=episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                return_uint8=True,
                depth_output_unit=cfg.dataset.depth_output_unit,
                tolerance_s=cfg.tolerance_s,
                repo_type=repo_type,
                video_decoder_cache_size=cfg.dataset.video_decoder_cache_size,
                feature_names=cfg.dataset.feature_names,
            )
        else:
            if cfg.dataset.feature_names is not None:
                raise NotImplementedError("dataset.feature_names is not supported with streaming datasets.")
            dataset = StreamingLeRobotDataset(
                repo_id,
                root=root,
                episodes=episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                seed=cfg.seed if cfg.seed is not None else DEFAULT_STREAMING_SEED,
                max_num_shards=max(1, cfg.num_workers),
                tolerance_s=cfg.tolerance_s,
                return_uint8=True,
                depth_output_unit=cfg.dataset.depth_output_unit,
                video_backend=cfg.dataset.video_backend,
                episode_pool_size=cfg.dataset.streaming_episode_pool_size,
                sampling_strategy=cfg.dataset.streaming_sampling_strategy,
                prefetch_episodes=cfg.dataset.streaming_prefetch_episodes,
                byte_budget_gb=cfg.dataset.streaming_byte_budget_gb,
                decode_threads=cfg.dataset.streaming_decode_threads,
                decoded_queue_size=cfg.dataset.streaming_decoded_queue_size,
                video_decoder_cache_size=cfg.dataset.video_decoder_cache_size,
                native_http_connections=cfg.dataset.streaming_native_http_connections,
                native_http_subranges=cfg.dataset.streaming_native_http_subranges,
                repeat=True,
                repo_type=repo_type,
            )
    return dataset


def _make_dataset_raw(
    cfg: TrainPipelineConfig,
    *,
    image_transforms: ImageTransforms | None,
    episodes_by_root: list[list[int] | None] | None = None,
) -> LeRobotDataset | StreamingLeRobotDataset | MergedLeRobotDataset | MultiLeRobotDataset:
    roots = _normalize_dataset_roots(cfg.dataset.root)
    repo_ids = _normalize_dataset_repo_ids(cfg.dataset.repo_id, len(roots))

    if len(roots) == 1:
        episodes = episodes_by_root[0] if episodes_by_root is not None else cfg.dataset.episodes
        return _make_single_dataset(
            cfg,
            repo_id=repo_ids[0],
            root=roots[0],
            episodes=episodes,
            image_transforms=image_transforms,
        )

    if cfg.dataset.streaming:
        raise NotImplementedError("dataset.root as a list is not supported with streaming datasets.")
    if episodes_by_root is not None and len(episodes_by_root) != len(roots):
        raise ValueError(
            f"episodes_by_root must have one entry per dataset root, got {len(episodes_by_root)} "
            f"for {len(roots)} roots."
        )

    datasets_to_merge = []
    for index, (repo_id, root) in enumerate(zip(repo_ids, roots, strict=True)):
        episodes = episodes_by_root[index] if episodes_by_root is not None else cfg.dataset.episodes
        datasets_to_merge.append(
            _make_single_dataset(
                cfg,
                repo_id=repo_id,
                root=root,
                episodes=episodes,
                image_transforms=image_transforms,
            )
        )

    logging.info(
        "Loaded %d local dataset roots for runtime training merge: %s",
        len(datasets_to_merge),
        [str(root) for root in roots],
    )
    return MergedLeRobotDataset(datasets_to_merge, repo_id="+".join(dict.fromkeys(repo_ids)))


def _apply_imagenet_stats(dataset: LeRobotDataset | MultiLeRobotDataset | MergedLeRobotDataset) -> None:
    if dataset.meta.stats is None:
        return
    for key in dataset.meta.camera_keys:
        if key in dataset.meta.depth_keys:
            continue
        dataset.meta.stats.setdefault(key, {})
        for stats_type, stats in IMAGENET_STATS.items():
            dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)


def make_dataset(
    cfg: TrainPipelineConfig,
) -> LeRobotDataset | StreamingLeRobotDataset | MultiLeRobotDataset | MergedLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: For unsupported combinations such as multiple streaming roots.

    Returns:
        LeRobotDataset | StreamingLeRobotDataset | MultiLeRobotDataset | MergedLeRobotDataset
    """
    dataset = _make_dataset_raw(cfg, image_transforms=_make_image_transforms(cfg))

    if cfg.dataset.use_imagenet_stats:
        _apply_imagenet_stats(dataset)

    return dataset


def make_train_eval_datasets(
    cfg: TrainPipelineConfig,
) -> tuple[LeRobotDataset | MultiLeRobotDataset | MergedLeRobotDataset, LeRobotDataset | None]:
    """Create train and optional eval datasets by splitting episodes based on eval_split.

    The last ceil(n_episodes * eval_split) episodes per task are held out for evaluation.
    If eval_split == 0.0, returns (full_dataset, None).
    """
    full_dataset = make_dataset(cfg)

    if cfg.dataset.eval_split == 0.0:
        return full_dataset, None

    if isinstance(full_dataset, MergedLeRobotDataset):
        raise NotImplementedError(
            "dataset.eval_split is not supported when dataset.root contains multiple roots. "
            "Set --dataset.eval_split=0.0 for multi-root training."
        )

    base_episodes = (
        full_dataset.episodes if full_dataset.episodes is not None else list(range(full_dataset.num_episodes))
    )

    episode_tasks = full_dataset.meta.episodes["tasks"]
    task_to_episodes: dict[str, list[int]] = {}
    for ep_idx in base_episodes:
        task_key = episode_tasks[ep_idx][0] if episode_tasks[ep_idx] else ""
        task_to_episodes.setdefault(task_key, []).append(ep_idx)

    train_episodes, eval_episodes = [], []
    for eps in task_to_episodes.values():
        n_eval = math.ceil(len(eps) * cfg.dataset.eval_split)
        train_episodes.extend(eps[: len(eps) - n_eval])
        eval_episodes.extend(eps[len(eps) - n_eval :])

    if not train_episodes:
        raise ValueError(
            f"eval_split={cfg.dataset.eval_split} leaves 0 training episodes from {len(base_episodes)} total."
        )

    logger.info(
        f"Train/eval split: {len(train_episodes)} train, {len(eval_episodes)} eval "
        f"(eval_split={cfg.dataset.eval_split}, {len(task_to_episodes)} tasks)"
    )

    delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, full_dataset.meta, cfg.rename_map)

    # On the "gpu" backend the workers only decode; lerobot-train augments each batch on the policy device.
    image_transforms_cfg = cfg.dataset.image_transforms
    train_image_transforms = (
        ImageTransforms(image_transforms_cfg)
        if image_transforms_cfg.enable and image_transforms_cfg.backend == "dataloader"
        else None
    )

    train_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=train_image_transforms,
        depth_output_unit=cfg.dataset.depth_output_unit,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
        repo_type=cfg.dataset.repo_type,
        video_decoder_cache_size=cfg.dataset.video_decoder_cache_size,
        feature_names=cfg.dataset.feature_names,
    )

    eval_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=eval_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        depth_output_unit=cfg.dataset.depth_output_unit,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
        repo_type=cfg.dataset.repo_type,
        video_decoder_cache_size=cfg.dataset.video_decoder_cache_size,
        feature_names=cfg.dataset.feature_names,
    )

    if cfg.dataset.use_imagenet_stats:
        for ds in (train_dataset, eval_dataset):
            for key in ds.meta.camera_keys:
                if key in ds.meta.depth_keys:
                    continue
                ds.meta.stats.setdefault(key, {})
                for stats_type, stats in IMAGENET_STATS.items():
                    ds.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return train_dataset, eval_dataset
