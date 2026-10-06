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
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_PREFIX, REWARD

from .dataset_metadata import LeRobotDatasetMetadata
from .feature_selection import filter_dataset_metadata_features
from .io_utils import write_info, write_stats
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MergedLeRobotDataset, MultiLeRobotDataset
from .streaming_dataset import StreamingLeRobotDataset
from .video_utils import clear_video_decoder_cache, get_video_duration_in_s, get_video_info


_VIDEO_EPISODE_METADATA_SUFFIXES = ("chunk_index", "file_index", "from_timestamp", "to_timestamp")


def get_modal_rgb_color(image: torch.Tensor) -> tuple[int, int, int]:
    """Compatibility wrapper for the PI0.5 thermal background estimator."""
    from lerobot.policies.pi05.thermal_utils import get_modal_rgb_color as _get_modal_rgb_color

    return _get_modal_rgb_color(image)


def _get_trainable_config(cfg: TrainPipelineConfig):
    return getattr(cfg, "trainable_config", getattr(cfg, "policy", None))


def _get_rgb_thermal_align_gt_feature(policy) -> str:
    return getattr(
        policy,
        "rgb_thermal_align_gt_feature",
        "observation.images.cam_rgb_thermal_mixing_GT",
    )


def validate_rgb_thermal_mix_feature_names(
    cfg: TrainPipelineConfig,
    available_features: dict[str, dict] | None = None,
) -> None:
    """Validate selected dataset features for PI0.5 pixel-level mixing."""
    policy = _get_trainable_config(cfg)
    uses_pixel_mix = policy is not None and getattr(policy, "mix_rgb_thermal", False)
    uses_align_fusion = policy is not None and getattr(policy, "rgb_thermal_align_fusion", False)
    if not uses_pixel_mix and not uses_align_fusion:
        return

    if cfg.dataset.feature_names is None and available_features is None:
        raise ValueError(
            "PI05 RGB/thermal features require dataset.feature_names to explicitly include "
            "the RGB and thermal features, unless dataset metadata is available for validation."
        )

    selected_features = (
        set(available_features or {})
        if cfg.dataset.feature_names is None
        else set(cfg.dataset.feature_names)
    )
    source = getattr(policy, "rgb_thermal_mix_source", "online")
    precomputed_feature = getattr(
        policy,
        "rgb_thermal_mix_precomputed_feature",
        "observation.images.cam_rgb_thermal_mixing",
    )
    align_gt_feature = _get_rgb_thermal_align_gt_feature(policy)
    use_precomputed = getattr(policy, "rgb_thermal_mix_use_precomputed", False)
    if uses_align_fusion:
        required_features = {
            getattr(policy, "thermal_fusion_head_rgb_feature", policy.rgb_thermal_mix_rgb_feature),
            policy.rgb_thermal_mix_thermal_feature,
            align_gt_feature,
        }
        if uses_pixel_mix and (source == "precomputed" or use_precomputed):
            required_features.add(precomputed_feature)
    elif source == "precomputed" or use_precomputed:
        required_features = {
            policy.rgb_thermal_mix_rgb_feature,
            policy.rgb_thermal_mix_thermal_feature,
            precomputed_feature,
        }
    else:
        required_features = {
            policy.rgb_thermal_mix_rgb_feature,
            policy.rgb_thermal_mix_thermal_feature,
        }

    if available_features is not None:
        missing_available = required_features.difference(available_features)
        if missing_available:
            raise ValueError(
                "PI05 RGB/thermal features require dataset metadata to contain: "
                f"{sorted(required_features)}. Missing from metadata: {sorted(missing_available)}. "
                f"Available features: {sorted(available_features)}."
            )

    missing_features = required_features.difference(selected_features)
    if missing_features:
        raise ValueError(
            "PI05 RGB/thermal features are missing required entries in dataset.feature_names: "
            f"{sorted(missing_features)}. Received: {cfg.dataset.feature_names}."
        )


def _precomputed_mix_layout_exists(
    ds_meta: LeRobotDatasetMetadata,
    precomputed_feature: str,
    layout_feature: str,
) -> bool:
    if ds_meta.video_path is None or ds_meta.episodes is None:
        return False

    required_columns = [
        f"videos/{layout_feature}/{suffix}" for suffix in _VIDEO_EPISODE_METADATA_SUFFIXES
    ]
    if any(column not in ds_meta.episodes.column_names for column in required_columns):
        return False

    required_video_ranges: dict[object, float] = {}
    for episode_index in range(ds_meta.total_episodes):
        episode = ds_meta.episodes[episode_index]
        chunk_index = int(episode[f"videos/{layout_feature}/chunk_index"])
        file_index = int(episode[f"videos/{layout_feature}/file_index"])
        path = (
            ds_meta.root
            / ds_meta.video_path.format(
                video_key=precomputed_feature,
                chunk_index=chunk_index,
                file_index=file_index,
            )
        )
        required_to_timestamp = float(episode[f"videos/{layout_feature}/to_timestamp"])
        required_video_ranges[path] = max(required_video_ranges.get(path, 0.0), required_to_timestamp)

    for path, required_duration_s in required_video_ranges.items():
        if not path.is_file():
            return False
        if get_video_duration_in_s(path) + 0.05 < required_duration_s:
            logging.warning(
                "Precomputed RGB/thermal mix candidate %s is too short for %s layout: "
                "duration=%.3fs, required_to_timestamp=%.3fs.",
                path,
                layout_feature,
                get_video_duration_in_s(path),
                required_duration_s,
            )
            return False
    return True


def _copy_precomputed_mix_episode_columns(
    ds_meta: LeRobotDatasetMetadata,
    *,
    source_feature: str,
    precomputed_feature: str,
) -> None:
    for suffix in _VIDEO_EPISODE_METADATA_SUFFIXES:
        source_column = f"videos/{source_feature}/{suffix}"
        target_column = f"videos/{precomputed_feature}/{suffix}"
        if target_column not in ds_meta.episodes.column_names:
            ds_meta.episodes = ds_meta.episodes.add_column(target_column, ds_meta.episodes[source_column])

    episodes_dir = ds_meta.root / "meta" / "episodes"
    for parquet_path in sorted(episodes_dir.glob("*/*.parquet")):
        episode_frame = pd.read_parquet(parquet_path)
        changed = False
        for suffix in _VIDEO_EPISODE_METADATA_SUFFIXES:
            source_column = f"videos/{source_feature}/{suffix}"
            target_column = f"videos/{precomputed_feature}/{suffix}"
            if target_column not in episode_frame.columns:
                episode_frame[target_column] = episode_frame[source_column]
                changed = True
        if changed:
            episode_frame.to_parquet(parquet_path, index=False)


def register_precomputed_rgb_thermal_mix_feature_if_present(
    cfg: TrainPipelineConfig,
    ds_meta: LeRobotDatasetMetadata,
) -> None:
    """Register a raw mixed-video folder as a dataset feature when possible."""
    policy = _get_trainable_config(cfg)
    uses_pixel_mix = policy is not None and getattr(policy, "mix_rgb_thermal", False)
    uses_align_fusion = policy is not None and getattr(policy, "rgb_thermal_align_fusion", False)
    if not uses_pixel_mix and not uses_align_fusion:
        return

    source = getattr(policy, "rgb_thermal_mix_source", "auto")
    if source == "online" and not uses_align_fusion:
        return

    precomputed_features = []
    if uses_pixel_mix and source != "online":
        precomputed_features.append(
            getattr(
                policy,
                "rgb_thermal_mix_precomputed_feature",
                "observation.images.cam_rgb_thermal_mixing",
            )
        )
    if uses_align_fusion:
        precomputed_features.append(_get_rgb_thermal_align_gt_feature(policy))

    for precomputed_feature in dict.fromkeys(precomputed_features):
        if precomputed_feature in ds_meta.features:
            continue

        precomputed_dir = ds_meta.root / "videos" / precomputed_feature
        if not precomputed_dir.is_dir():
            if source == "precomputed" and precomputed_feature != _get_rgb_thermal_align_gt_feature(policy):
                raise ValueError(
                    "policy.rgb_thermal_mix_source=precomputed requires either dataset metadata or "
                    f"a video folder for {precomputed_feature!r}; missing: {precomputed_dir}"
                )
            continue

        rgb_feature = policy.rgb_thermal_mix_rgb_feature
        thermal_feature = policy.rgb_thermal_mix_thermal_feature
        layout_candidates = [rgb_feature, thermal_feature]
        layout_feature = next(
            (
                candidate
                for candidate in layout_candidates
                if _precomputed_mix_layout_exists(ds_meta, precomputed_feature, candidate)
            ),
            None,
        )
        if layout_feature is None:
            message = (
                f"Found {precomputed_dir}, but it does not mirror the chunk/file layout of "
                f"{thermal_feature!r} or {rgb_feature!r}; cannot register it as a LeRobot video feature."
            )
            if source == "precomputed" and precomputed_feature != _get_rgb_thermal_align_gt_feature(policy):
                raise ValueError(message)
            logging.warning("%s Falling back to online RGB/thermal mixing.", message)
            continue

        feature_template_key = rgb_feature if rgb_feature in ds_meta.features else layout_feature
        ds_meta.info.features[precomputed_feature] = deepcopy(ds_meta.features[feature_template_key])
        ds_meta.info.features[precomputed_feature]["dtype"] = "video"
        first_video_path = ds_meta.root / ds_meta.video_path.format(
            video_key=precomputed_feature,
            chunk_index=0,
            file_index=0,
        )
        if first_video_path.is_file():
            ds_meta.info.features[precomputed_feature]["info"] = {
                **(ds_meta.info.features[precomputed_feature].get("info") or {}),
                **get_video_info(first_video_path),
            }

        if ds_meta.stats is not None:
            stats_template = ds_meta.stats.get(feature_template_key) or ds_meta.stats.get(layout_feature)
            if stats_template is not None:
                ds_meta.stats[precomputed_feature] = deepcopy(stats_template)

        _copy_precomputed_mix_episode_columns(
            ds_meta,
            source_feature=layout_feature,
            precomputed_feature=precomputed_feature,
        )
        write_info(ds_meta.info, ds_meta.root)
        if ds_meta.stats is not None and precomputed_feature in ds_meta.stats:
            write_stats(ds_meta.stats, ds_meta.root)

        logging.info(
            "Registered precomputed RGB/thermal feature %s from %s using %s episode video layout.",
            precomputed_feature,
            precomputed_dir,
            layout_feature,
        )
    return


def prepare_rgb_thermal_mix_dataset_features(
    cfg: TrainPipelineConfig,
    ds_meta: LeRobotDatasetMetadata,
) -> None:
    """Select the PI0.5 RGB/thermal mix source and add helper video features."""
    policy = _get_trainable_config(cfg)
    uses_pixel_mix = policy is not None and getattr(policy, "mix_rgb_thermal", False)
    uses_align_fusion = policy is not None and getattr(policy, "rgb_thermal_align_fusion", False)
    if not uses_pixel_mix and not uses_align_fusion:
        return

    policy.rgb_thermal_mix_use_precomputed = False
    source = getattr(policy, "rgb_thermal_mix_source", "auto")
    mix_precomputed_feature = getattr(
        policy,
        "rgb_thermal_mix_precomputed_feature",
        "observation.images.cam_rgb_thermal_mixing",
    )
    align_gt_feature = _get_rgb_thermal_align_gt_feature(policy)
    if source not in {"auto", "precomputed", "online"}:
        raise ValueError("rgb_thermal_mix_source must be one of: auto, precomputed, online")

    register_precomputed_rgb_thermal_mix_feature_if_present(cfg, ds_meta)

    mix_precomputed_meta = ds_meta.features.get(mix_precomputed_feature)
    has_mix_precomputed = (
        mix_precomputed_meta is not None and mix_precomputed_meta.get("dtype") in {"video", "image"}
    )
    if uses_pixel_mix and source == "precomputed" and not has_mix_precomputed:
        raise ValueError(
            "policy.rgb_thermal_mix_source=precomputed requires dataset metadata to contain "
            f"{mix_precomputed_feature!r} as an image/video feature. "
            f"Available features: {sorted(ds_meta.features)}"
        )

    align_gt_meta = ds_meta.features.get(align_gt_feature)
    has_align_gt = align_gt_meta is not None and align_gt_meta.get("dtype") in {"video", "image"}
    if uses_align_fusion and not has_align_gt:
        raise ValueError(
            "policy.rgb_thermal_align_fusion=true requires a precomputed RGB/thermal GT feature "
            f"{align_gt_feature!r}. Available features: {sorted(ds_meta.features)}"
        )

    if uses_pixel_mix:
        use_precomputed = source == "precomputed" or (source == "auto" and has_mix_precomputed)
        if use_precomputed:
            policy.rgb_thermal_mix_use_precomputed = True
            if cfg.dataset.feature_names is not None and mix_precomputed_feature not in cfg.dataset.feature_names:
                cfg.dataset.feature_names = [*cfg.dataset.feature_names, mix_precomputed_feature]
            logging.info(
                "PI05 RGB/thermal mix will use precomputed mixed feature %s when building batches.",
                mix_precomputed_feature,
            )
        else:
            logging.info("PI05 RGB/thermal mix will use online image blending.")

    if uses_align_fusion:
        if cfg.dataset.feature_names is not None and align_gt_feature not in cfg.dataset.feature_names:
            cfg.dataset.feature_names = [*cfg.dataset.feature_names, align_gt_feature]
        logging.info(
            "PI05 RGB/thermal alignment decoder will use GT feature %s.",
            align_gt_feature,
        )

    validate_rgb_thermal_mix_feature_names(cfg, available_features=ds_meta.features)


def resolve_rgb_thermal_mix_fill_color(
    cfg: TrainPipelineConfig,
    dataset: LeRobotDataset | MultiLeRobotDataset,
) -> None:
    """Resolve and persist the modal background color of each selected episode."""
    policy = _get_trainable_config(cfg)
    if not getattr(policy, "mix_rgb_thermal", False):
        return
    if getattr(policy, "rgb_thermal_mix_use_precomputed", False):
        return
    if policy.rgb_thermal_mix_episode_fill_colors:
        return
    if hasattr(dataset, "_ensure_reader"):
        reader = dataset._ensure_reader()  # noqa: SLF001
    elif hasattr(dataset, "_query_videos"):
        reader = dataset
    else:
        raise TypeError("Per-episode RGB/thermal fill colors require a LeRobotDataset-like object.")

    thermal_feature = policy.rgb_thermal_mix_thermal_feature
    selected_episodes = getattr(dataset, "episodes", None)
    episode_indices = (
        selected_episodes
        if selected_episodes is not None
        else list(range(dataset.meta.total_episodes))
    )
    resolved: dict[int, tuple[int, int, int]] = {}
    try:
        for episode_index in episode_indices:
            first_thermal_frame = reader._query_videos(  # noqa: SLF001
                {thermal_feature: [0.0]}, episode_index
            )[thermal_feature]
            resolved[episode_index] = get_modal_rgb_color(first_thermal_frame)
    finally:
        # Parent-process TorchCodec handles must not leak into forked workers.
        clear_video_decoder_cache()

    if not resolved:
        raise ValueError("Could not resolve an RGB/thermal fill color from the selected episodes.")
    fallback = policy.rgb_thermal_mix_fill_color or next(iter(resolved.values()))
    policy.rgb_thermal_mix_fill_color = fallback
    policy.rgb_thermal_mix_episode_fill_colors = [
        resolved.get(episode_index, fallback)
        for episode_index in range(dataset.meta.total_episodes)
    ]
    logging.info(
        "Resolved %d PI05 per-episode RGB/thermal fill colors; inference fallback: RGB%s",
        len(resolved),
        policy.rgb_thermal_mix_fill_color,
    )


def resolve_delta_timestamps(
    cfg: PreTrainedConfig | RewardModelConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the config.

    Args:
        cfg (PreTrainedConfig | RewardModelConfig): The config to read delta_indices from. Both
            ``PreTrainedConfig`` and concrete ``RewardModelConfig`` subclasses expose the
            ``{observation,action,reward}_delta_indices`` properties used below.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

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
    return ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None


def _make_single_dataset(
    cfg: TrainPipelineConfig,
    *,
    repo_id: str,
    root: str | Path | None,
    episodes: list[int] | None,
    image_transforms: ImageTransforms | None,
) -> LeRobotDataset | StreamingLeRobotDataset:
    ds_meta = LeRobotDatasetMetadata(repo_id, root=root, revision=cfg.dataset.revision)
    prepare_rgb_thermal_mix_dataset_features(cfg, ds_meta)
    filter_dataset_metadata_features(ds_meta, cfg.dataset.feature_names)
    delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, ds_meta)
    if not cfg.dataset.streaming:
        return LeRobotDataset(
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
            feature_names=cfg.dataset.feature_names,
        )

    if cfg.dataset.feature_names is not None:
        raise NotImplementedError("`dataset.feature_names` is not supported with streaming datasets yet.")
    return StreamingLeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=cfg.dataset.revision,
        max_num_shards=cfg.num_workers,
        tolerance_s=cfg.tolerance_s,
        return_uint8=True,
    )


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
        episodes = (
            episodes_by_root[index]
            if episodes_by_root is not None
            else cfg.dataset.episodes
        )
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
    resolve_rgb_thermal_mix_fill_color(cfg, dataset)

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

    logging.info(
        f"Train/eval split: {len(train_episodes)} train, {len(eval_episodes)} eval "
        f"(eval_split={cfg.dataset.eval_split}, {len(task_to_episodes)} tasks)"
    )

    delta_timestamps = resolve_delta_timestamps(cfg.trainable_config, full_dataset.meta)

    train_image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    train_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=train_image_transforms,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
        feature_names=cfg.dataset.feature_names,
    )

    eval_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=eval_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=None,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
        feature_names=cfg.dataset.feature_names,
    )

    if cfg.dataset.use_imagenet_stats:
        for ds in (train_dataset, eval_dataset):
            for key in ds.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    ds.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return train_dataset, eval_dataset
