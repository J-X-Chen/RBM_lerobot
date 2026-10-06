#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Fast, cached MINIMA alignment for PI0.5 thermal training inputs.

MINIMA still estimates each selected homography one image pair at a time, but
decoded images remain on the accelerator for the expensive full-resolution
warp. Homographies are cached by episode/frame bucket so repeated epochs and
nearby frames do not rerun the matcher.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from .thermal_utils import image_to_bchw


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "MINIMA" / "load_model.py").is_file():
            return parent
    return Path.cwd().resolve()


def _resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    path = Path(str(path)).expanduser()
    if path.is_absolute():
        return path
    candidates = []
    if base is not None:
        candidates.append(base / path)
    candidates.extend((_project_root() / path, Path.cwd() / path))
    return next(
        (candidate.resolve() for candidate in candidates if candidate.exists()),
        candidates[0].resolve(),
    )


def _metadata_values(value: Any, batch_size: int) -> list[int | None]:
    if value is None:
        return [None] * batch_size
    tensor = torch.as_tensor(value).detach().reshape(-1).cpu()
    if tensor.numel() != batch_size:
        return [None] * batch_size
    return [int(item) for item in tensor.tolist()]


def _to_rgb_uint8_batch(images: Tensor, feature_name: str) -> np.ndarray:
    source_dtype = images.dtype
    images = image_to_bchw(images, feature_name).detach().to(dtype=torch.float32)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    else:
        images = images[:, :3]
    if images.numel():
        minimum = float(images.min().item())
        maximum = float(images.max().item())
        if minimum < 0.0 and maximum <= 1.0:
            images = (images + 1.0) * 0.5
        elif source_dtype == torch.uint8 or (minimum >= 0.0 and maximum > 1.0):
            images = images / 255.0
    return (
        images.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .cpu()
        .numpy()
    )


def _resize_homography(
    source_height: int,
    source_width: int,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    scale_x = (output_width - 1) / max(source_width - 1, 1)
    scale_y = (output_height - 1) / max(source_height - 1, 1)
    return np.asarray(
        ((scale_x, 0.0, 0.0), (0.0, scale_y, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def rescale_homography(
    homography: Tensor | np.ndarray,
    *,
    stored_source_size: tuple[int, int] | list[int],
    current_source_size: tuple[int, int] | list[int],
    stored_output_size: tuple[int, int] | list[int],
    current_output_size: tuple[int, int] | list[int],
) -> np.ndarray:
    """Adapt a stored source-to-output homography to different image sizes."""
    stored_source_height, stored_source_width = (int(value) for value in stored_source_size)
    current_source_height, current_source_width = (int(value) for value in current_source_size)
    stored_output_height, stored_output_width = (int(value) for value in stored_output_size)
    current_output_height, current_output_width = (int(value) for value in current_output_size)
    sizes = (
        stored_source_height,
        stored_source_width,
        current_source_height,
        current_source_width,
        stored_output_height,
        stored_output_width,
        current_output_height,
        current_output_width,
    )
    if any(value <= 0 for value in sizes):
        raise ValueError(f"Homography image sizes must be positive, got {sizes}.")

    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"homography must have shape (3, 3), got {matrix.shape}.")
    source_current_to_stored = np.diag(
        (
            (stored_source_width - 1) / max(current_source_width - 1, 1),
            (stored_source_height - 1) / max(current_source_height - 1, 1),
            1.0,
        )
    )
    output_stored_to_current = np.diag(
        (
            (current_output_width - 1) / max(stored_output_width - 1, 1),
            (current_output_height - 1) / max(stored_output_height - 1, 1),
            1.0,
        )
    )
    return output_stored_to_current @ matrix @ source_current_to_stored


def robust_average_homographies(
    homographies: list[np.ndarray],
    *,
    source_size: tuple[int, int],
) -> tuple[np.ndarray, dict[str, float]]:
    """Average homographies through their projected grid and reject spatial outliers."""
    import cv2

    if not homographies:
        raise ValueError("At least one homography is required.")
    source_height, source_width = (int(value) for value in source_size)
    if source_height <= 0 or source_width <= 0:
        raise ValueError(f"source_size must be positive, got {source_size}.")

    normalized = []
    for homography in homographies:
        matrix = np.asarray(homography, dtype=np.float64)
        if (
            matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
            or abs(float(np.linalg.det(matrix))) < 1e-10
        ):
            continue
        scale = float(matrix[2, 2])
        if abs(scale) < 1e-10:
            scale = float(np.linalg.norm(matrix))
        if abs(scale) < 1e-10:
            continue
        normalized.append(matrix / scale)
    if not normalized:
        raise ValueError("No finite, invertible homographies were provided.")

    x_values = np.linspace(0.0, max(source_width - 1, 0), 3)
    y_values = np.linspace(0.0, max(source_height - 1, 0), 3)
    source_points = np.asarray(
        [(x, y) for y in y_values for x in x_values],
        dtype=np.float64,
    )
    source_homogeneous = np.concatenate(
        [source_points, np.ones((source_points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    projected = []
    valid_matrices = []
    for matrix in normalized:
        destination = source_homogeneous @ matrix.T
        denominator = destination[:, 2:3]
        if np.any(np.abs(denominator) < 1e-8):
            continue
        destination = destination[:, :2] / denominator
        if np.isfinite(destination).all():
            valid_matrices.append(matrix)
            projected.append(destination)
    if not projected:
        raise ValueError("Homographies did not produce finite projected points.")

    projected_array = np.stack(projected)
    median_projection = np.median(projected_array, axis=0)
    candidate_errors = np.median(
        np.linalg.norm(projected_array - median_projection[None], axis=-1),
        axis=1,
    )
    median_error = float(np.median(candidate_errors))
    mad = float(np.median(np.abs(candidate_errors - median_error)))
    rejection_radius = median_error + max(2.5 * 1.4826 * mad, 2.0)
    keep = candidate_errors <= rejection_radius
    if not bool(np.any(keep)):
        keep[int(np.argmin(candidate_errors))] = True

    consensus_projection = np.mean(projected_array[keep], axis=0)
    consensus, _ = cv2.findHomography(
        source_points.astype(np.float32),
        consensus_projection.astype(np.float32),
        method=0,
    )
    if (
        consensus is None
        or not np.isfinite(consensus).all()
        or abs(float(np.linalg.det(consensus))) < 1e-10
    ):
        consensus = np.mean(np.stack(valid_matrices)[keep], axis=0)
    consensus = np.asarray(consensus, dtype=np.float64)
    consensus_scale = float(consensus[2, 2])
    if abs(consensus_scale) < 1e-10:
        consensus_scale = float(np.linalg.norm(consensus))
    if abs(consensus_scale) < 1e-10:
        raise ValueError("Robust homography consensus has zero scale.")
    consensus /= consensus_scale
    return consensus, {
        "candidate_count": float(len(projected)),
        "accepted_count": float(int(keep.sum())),
        "median_projection_error_px": median_error,
        "max_accepted_projection_error_px": float(candidate_errors[keep].max()),
    }


def warp_thermal_batch(
    thermal_images: Tensor,
    homographies: Tensor | np.ndarray,
    *,
    output_size: tuple[int, int],
    outer_padding: int,
    feature_name: str,
) -> tuple[Tensor, Tensor]:
    """Warp source-to-output homographies and return ``(image, alpha)``.

    The original thermal pixels are never attenuated. Replicated pixels are
    added *outside* the source image and fade from the edge value to zero alpha.
    Areas not covered by the padded source therefore have alpha zero.
    """
    thermal = image_to_bchw(thermal_images, feature_name)
    reference_dtype = thermal.dtype
    work = thermal.to(dtype=torch.float32)
    batch_size, _, source_height, source_width = work.shape
    output_height, output_width = (int(value) for value in output_size)
    padding = max(0, int(outer_padding))

    homography = torch.as_tensor(homographies, device=work.device, dtype=torch.float32)
    if homography.shape != (batch_size, 3, 3):
        raise ValueError(
            f"homographies must have shape {(batch_size, 3, 3)}, got {tuple(homography.shape)}."
        )

    # Geometry kernels must stay in FP32 even when the policy runs under bf16 autocast.
    with torch.autocast(device_type=work.device.type, enabled=False):
        work = work.to(dtype=torch.float32)
        homography = homography.to(dtype=torch.float32)

        if padding:
            work = F.pad(work, (padding, padding, padding, padding), mode="replicate")
            padded_height = source_height + 2 * padding
            padded_width = source_width + 2 * padding
            rows = torch.arange(padded_height, device=work.device, dtype=torch.float32)
            columns = torch.arange(padded_width, device=work.device, dtype=torch.float32)
            row_distance = torch.minimum(rows, padded_height - 1 - rows)
            column_distance = torch.minimum(columns, padded_width - 1 - columns)
            alpha_source = (
                torch.minimum(row_distance[:, None], column_distance[None, :])
                .div(float(padding))
                .clamp(0.0, 1.0)
            )
            alpha_source[
                padding : padding + source_height,
                padding : padding + source_width,
            ] = 1.0
            padded_to_original = homography.new_tensor(
                (
                    (1.0, 0.0, -float(padding)),
                    (0.0, 1.0, -float(padding)),
                    (0.0, 0.0, 1.0),
                )
            )
            homography = torch.matmul(homography, padded_to_original)
        else:
            padded_height, padded_width = source_height, source_width
            alpha_source = work.new_ones((source_height, source_width))

        y, x = torch.meshgrid(
            torch.arange(output_height, device=work.device, dtype=torch.float32),
            torch.arange(output_width, device=work.device, dtype=torch.float32),
            indexing="ij",
        )
        destination = torch.stack((x, y, torch.ones_like(x)), dim=-1).reshape(1, -1, 3)
        inverse = torch.linalg.inv(homography)
        source = torch.bmm(
            destination.expand(batch_size, -1, -1),
            inverse.transpose(1, 2),
        )
        denominator = source[..., 2:3]
        denominator = torch.where(
            denominator.abs() < 1e-8,
            torch.full_like(denominator, 1e-8),
            denominator,
        )
        source_xy = source[..., :2] / denominator
        grid_x = source_xy[..., 0] * (2.0 / max(padded_width - 1, 1)) - 1.0
        grid_y = source_xy[..., 1] * (2.0 / max(padded_height - 1, 1)) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
            batch_size,
            output_height,
            output_width,
            2,
        )

        aligned = F.grid_sample(
            work,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        alpha = F.grid_sample(
            alpha_source.reshape(1, 1, padded_height, padded_width).expand(batch_size, -1, -1, -1),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).clamp(0.0, 1.0)
    return aligned.to(dtype=reference_dtype), alpha


class MinimaThermalMatcher:
    """Preloaded MINIMA matcher with frame-bucket homography caching."""

    def __init__(
        self,
        *,
        minima_root: str | Path,
        checkpoint: str | Path,
        ransac_reproj_threshold: float = 5.0,
        min_matches: int = 4,
        min_inliers: int = 4,
        match_every_n_frames: int = 10,
        cache_size: int = 32768,
        outer_padding: int = 30,
    ) -> None:
        project_root = _project_root()
        project_root_text = str(project_root)
        if project_root_text not in sys.path:
            sys.path.insert(0, project_root_text)

        from unitree_lerobot.utils.match_and_merge_rgb_and_thermography import (
            estimate_thermal_to_color_homography,
            load_matcher,
        )

        minima_root = _resolve_path(minima_root)
        checkpoint = _resolve_path(checkpoint, base=minima_root)
        if not minima_root.is_dir():
            raise FileNotFoundError(f"MINIMA directory does not exist: {minima_root}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"MINIMA checkpoint does not exist: {checkpoint}")

        args = SimpleNamespace(method="sp_lg", minima_root=str(minima_root), ckpt=str(checkpoint))
        grad_enabled = torch.is_grad_enabled()
        started = time.perf_counter()
        try:
            self.matcher = load_matcher(args, use_path=False)
        finally:
            # Older MINIMA wrappers changed this process-global state in __init__.
            torch.set_grad_enabled(grad_enabled)
        self.load_ms = (time.perf_counter() - started) * 1000.0
        self._estimate = estimate_thermal_to_color_homography
        self.ransac_reproj_threshold = float(ransac_reproj_threshold)
        self.min_matches = int(min_matches)
        self.min_inliers = int(min_inliers)
        self.match_every_n_frames = max(1, int(match_every_n_frames))
        self.cache_size = max(1, int(cache_size))
        self.outer_padding = max(0, int(outer_padding))
        self.cache: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self.match_calls = 0
        self.cache_hits = 0
        logging.info(
            "Loaded MINIMA thermal matcher in %.1f ms: cache=%d, match_every_n_frames=%d, "
            "outer_padding=%d.",
            self.load_ms,
            self.cache_size,
            self.match_every_n_frames,
            self.outer_padding,
        )

    def _cache_key(
        self,
        *,
        dataset_index: int | None,
        episode_index: int | None,
        frame_index: int | None,
        sample_index: int | None,
    ) -> tuple[Any, ...] | None:
        if episode_index is not None and frame_index is not None:
            return (
                "episode",
                dataset_index,
                episode_index,
                frame_index // self.match_every_n_frames,
            )
        if sample_index is not None:
            return ("sample", dataset_index, sample_index)
        return None

    def _remember(self, key: tuple[Any, ...], homography: np.ndarray) -> None:
        self.cache[key] = np.asarray(homography, dtype=np.float64)
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def estimate_homography(
        self,
        rgb_image: np.ndarray,
        thermal_image: np.ndarray,
        *,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Estimate one thermal-to-RGB homography without warping either image."""
        import cv2

        rgb_image = np.asarray(rgb_image, dtype=np.uint8)
        thermal_image = np.asarray(thermal_image, dtype=np.uint8)
        if rgb_image.ndim != 3 or rgb_image.shape[-1] != 3:
            raise ValueError(f"rgb_image must be HWC RGB uint8, got {rgb_image.shape}.")
        if thermal_image.ndim != 3 or thermal_image.shape[-1] != 3:
            raise ValueError(
                f"thermal_image must be HWC RGB uint8, got {thermal_image.shape}."
            )
        if output_size is None:
            output_size = (int(rgb_image.shape[0]), int(rgb_image.shape[1]))

        color_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        thermal_bgr = cv2.cvtColor(thermal_image, cv2.COLOR_RGB2BGR)
        metrics: dict[str, Any] = {}
        homography = None
        try:
            autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.inference_mode(), torch.autocast(device_type=autocast_device, enabled=False):
                homography, metrics = self._estimate(
                    self.matcher,
                    color_bgr,
                    thermal_bgr,
                    self.ransac_reproj_threshold,
                    self.min_matches,
                    self.min_inliers,
                    1.0,
                    True,
                    0.0,
                    0.6,
                )
            self.match_calls += 1
        except Exception as error:
            metrics = {"fallback_reason": f"error:{error}"}
            logging.warning("MINIMA matching failed; using resize fallback: %s", error)

        if (
            homography is None
            or not np.isfinite(homography).all()
            or abs(float(np.linalg.det(homography))) < 1e-10
        ):
            metrics.setdefault("fallback_reason", "resize")
            homography = _resize_homography(
                int(thermal_image.shape[0]),
                int(thermal_image.shape[1]),
                int(output_size[0]),
                int(output_size[1]),
            )
        else:
            metrics.setdefault("fallback_reason", "")
        return np.asarray(homography, dtype=np.float64), metrics

    def align_batch(
        self,
        rgb_images: Tensor,
        thermal_images: Tensor,
        *,
        rgb_feature_name: str,
        thermal_feature_name: str,
        sample_indices: Any = None,
        episode_indices: Any = None,
        frame_indices: Any = None,
        dataset_indices: Any = None,
    ) -> tuple[Tensor, Tensor, dict[str, int]]:
        rgb = image_to_bchw(rgb_images, rgb_feature_name)
        thermal = image_to_bchw(thermal_images, thermal_feature_name)
        if rgb.shape[0] != thermal.shape[0]:
            raise ValueError(
                "RGB and thermal matching batches must have the same size, "
                f"got {rgb.shape[0]} and {thermal.shape[0]}."
            )
        batch_size = int(rgb.shape[0])
        output_size = (int(rgb.shape[-2]), int(rgb.shape[-1]))
        sample_values = _metadata_values(sample_indices, batch_size)
        episode_values = _metadata_values(episode_indices, batch_size)
        frame_values = _metadata_values(frame_indices, batch_size)
        dataset_values = _metadata_values(dataset_indices, batch_size)

        keys = []
        for index in range(batch_size):
            base_key = self._cache_key(
                dataset_index=dataset_values[index],
                episode_index=episode_values[index],
                frame_index=frame_values[index],
                sample_index=sample_values[index],
            )
            keys.append(
                (thermal_feature_name, *base_key) if base_key is not None else None
            )
        homographies: list[np.ndarray | None] = [None] * batch_size
        missing_positions = []
        representatives = []
        cache_hits = 0
        for index, key in enumerate(keys):
            if key is not None and key in self.cache:
                homographies[index] = self.cache[key]
                self.cache.move_to_end(key)
                self.cache_hits += 1
                cache_hits += 1
            else:
                missing_positions.append(index)

        if missing_positions:
            grouped_positions: dict[tuple[Any, ...], list[int]] = {}
            for batch_index in missing_positions:
                key = keys[batch_index]
                group_key = key if key is not None else ("uncached_batch_position", batch_index)
                if group_key not in grouped_positions:
                    representatives.append(batch_index)
                    grouped_positions[group_key] = []
                grouped_positions[group_key].append(batch_index)

            rgb_u8 = _to_rgb_uint8_batch(rgb[representatives], rgb_feature_name)
            thermal_u8 = _to_rgb_uint8_batch(thermal[representatives], thermal_feature_name)
            for local_index, representative in enumerate(representatives):
                homography, _ = self.estimate_homography(
                    rgb_u8[local_index],
                    thermal_u8[local_index],
                    output_size=output_size,
                )
                key = keys[representative]
                group_key = (
                    key if key is not None else ("uncached_batch_position", representative)
                )
                group = grouped_positions[group_key]
                for batch_index in group:
                    homographies[batch_index] = homography
                cache_hits += len(group) - 1
                self.cache_hits += len(group) - 1
                if key is not None:
                    self._remember(key, homography)

        homography_tensor = torch.as_tensor(
            np.stack(homographies),
            device=thermal.device,
            dtype=torch.float32,
        )
        aligned, alpha = warp_thermal_batch(
            thermal,
            homography_tensor,
            output_size=output_size,
            outer_padding=self.outer_padding,
            feature_name=thermal_feature_name,
        )
        return aligned, alpha, {
            "matches": len(representatives),
            "cache_hits": cache_hits,
        }


_FIXED_MATCHING_CONFIG_FIELDS = (
    "thermal_fixed_matching_homographies",
    "thermal_fixed_matching_source_sizes",
    "thermal_fixed_matching_output_sizes",
    "thermal_fixed_matching_dataset_roots",
    "thermal_fixed_matching_sample_indices",
    "thermal_fixed_matching_sample_timestamps",
    "thermal_fixed_matching_diagnostics",
)


def fixed_matching_config_payload(config: Any) -> dict[str, Any]:
    """Return the checkpoint-serializable fixed-matching fields."""
    return {
        field_name: getattr(config, field_name)
        for field_name in _FIXED_MATCHING_CONFIG_FIELDS
    }


def apply_fixed_matching_config_payload(config: Any, payload: dict[str, Any]) -> None:
    """Apply fixed-matching fields received from another training process."""
    for field_name in _FIXED_MATCHING_CONFIG_FIELDS:
        if field_name not in payload:
            raise ValueError(f"Fixed-matching payload is missing {field_name!r}.")
        setattr(config, field_name, payload[field_name])
    validate = getattr(config, "_validate_thermal_fixed_matching_config", None)
    if callable(validate):
        validate()


def _normalized_dataset_roots(dataset: Any) -> list[str]:
    roots = getattr(dataset, "root", None)
    if roots is None:
        return []
    if not isinstance(roots, (list, tuple)):
        roots = [roots]
    normalized = []
    for root in roots:
        path = Path(root).expanduser()
        normalized.append(str(path.resolve() if path.exists() else path.absolute()))
    return normalized


def _sample_image_to_rgb_uint8(value: Any, feature_name: str) -> np.ndarray:
    image = torch.as_tensor(value)
    while image.ndim > 3:
        image = image[-1]
    if image.ndim != 3:
        raise ValueError(
            f"Sampled image {feature_name!r} must have three dimensions, got {tuple(image.shape)}."
        )
    return _to_rgb_uint8_batch(image.unsqueeze(0), feature_name)[0]


def _sample_scalar(item: dict[str, Any], key: str, default: float) -> float:
    value = item.get(key)
    if value is None:
        return float(default)
    tensor = torch.as_tensor(value).detach().reshape(-1)
    return float(tensor[-1].cpu().item()) if tensor.numel() else float(default)


def prepare_fixed_thermal_matching(
    config: Any,
    dataset: Any,
) -> dict[str, Any]:
    """Estimate and store one robust MINIMA homography for the training dataset."""
    if getattr(config, "thermal_input_type", None) != "twofixmatchingblack":
        return fixed_matching_config_payload(config)
    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        raise TypeError("twofixmatchingblack requires a finite random-access training dataset.")
    dataset_length = int(len(dataset))
    if dataset_length <= 0:
        raise ValueError("Cannot estimate fixed thermal matching from an empty dataset.")

    thermal_features = list(getattr(config, "thermal_image_features", []))
    if not thermal_features:
        raise ValueError("twofixmatchingblack requires at least one thermal image feature.")
    rgb_feature = str(config.rgb_thermal_mix_rgb_feature)
    dataset_roots = _normalized_dataset_roots(dataset)
    existing = getattr(config, "thermal_fixed_matching_homographies", {})
    stored_roots = list(getattr(config, "thermal_fixed_matching_dataset_roots", []))
    if all(feature_name in existing for feature_name in thermal_features) and (
        not stored_roots or stored_roots == dataset_roots
    ):
        logging.info(
            "Reusing checkpoint fixed thermal homographies for dataset roots: %s.",
            dataset_roots,
        )
        config.thermal_fixed_matching_dataset_roots = dataset_roots
        return fixed_matching_config_payload(config)

    sample_count = min(int(config.thermal_fixed_matching_num_samples), dataset_length)
    min_valid = int(config.thermal_fixed_matching_min_valid_samples)
    if sample_count < min_valid:
        raise ValueError(
            "The training dataset is too small for twofixmatchingblack: "
            f"sample_count={sample_count}, required_valid={min_valid}."
        )
    rng = np.random.default_rng(int(config.thermal_fixed_matching_seed))
    sample_indices = [int(value) for value in rng.choice(dataset_length, sample_count, replace=False)]

    matcher = MinimaThermalMatcher(
        minima_root=config.thermal_matching_minima_root,
        checkpoint=config.thermal_matching_checkpoint,
        ransac_reproj_threshold=config.thermal_matching_ransac_reproj_threshold,
        min_matches=config.thermal_matching_min_matches,
        min_inliers=config.thermal_matching_min_inliers,
        match_every_n_frames=config.thermal_matching_match_every_n_frames,
        cache_size=config.thermal_matching_cache_size,
        outer_padding=config.thermal_matching_outer_padding,
    )
    candidates: dict[str, list[np.ndarray]] = {feature_name: [] for feature_name in thermal_features}
    source_sizes: dict[str, tuple[int, int]] = {}
    output_sizes: dict[str, tuple[int, int]] = {}
    failed_counts = dict.fromkeys(thermal_features, 0)
    sample_timestamps = []

    original_transforms = getattr(dataset, "image_transforms", None)
    clear_transforms = getattr(dataset, "clear_image_transforms", None)
    set_transforms = getattr(dataset, "set_image_transforms", None)
    if callable(clear_transforms):
        clear_transforms()
    try:
        for sample_index in sample_indices:
            item = dataset[sample_index]
            if rgb_feature not in item:
                raise ValueError(
                    f"Sample {sample_index} is missing head RGB feature {rgb_feature!r}."
                )
            rgb_image = _sample_image_to_rgb_uint8(item[rgb_feature], rgb_feature)
            output_size = (int(rgb_image.shape[0]), int(rgb_image.shape[1]))
            sample_timestamps.append(_sample_scalar(item, "timestamp", sample_index))
            for feature_name in thermal_features:
                if feature_name not in item:
                    raise ValueError(
                        f"Sample {sample_index} is missing thermal feature {feature_name!r}."
                    )
                thermal_image = _sample_image_to_rgb_uint8(item[feature_name], feature_name)
                source_size = (int(thermal_image.shape[0]), int(thermal_image.shape[1]))
                previous_source_size = source_sizes.setdefault(feature_name, source_size)
                previous_output_size = output_sizes.setdefault(feature_name, output_size)
                if source_size != previous_source_size or output_size != previous_output_size:
                    raise ValueError(
                        "twofixmatchingblack sampled inconsistent image sizes for "
                        f"{feature_name!r}: source={source_size}/{previous_source_size}, "
                        f"output={output_size}/{previous_output_size}."
                    )
                homography, metrics = matcher.estimate_homography(
                    rgb_image,
                    thermal_image,
                    output_size=output_size,
                )
                if metrics.get("fallback_reason"):
                    failed_counts[feature_name] += 1
                    continue
                candidates[feature_name].append(homography)
    finally:
        if callable(set_transforms):
            set_transforms(original_transforms)

    homographies = {}
    diagnostics = {}
    for feature_name in thermal_features:
        valid_count = len(candidates[feature_name])
        if valid_count < min_valid:
            raise RuntimeError(
                "MINIMA did not produce enough valid homographies for "
                f"{feature_name!r}: valid={valid_count}, required={min_valid}, "
                f"sampled={sample_count}, failed={failed_counts[feature_name]}."
            )
        consensus, feature_diagnostics = robust_average_homographies(
            candidates[feature_name],
            source_size=source_sizes[feature_name],
        )
        required_consensus = min(3, min_valid)
        if int(feature_diagnostics["accepted_count"]) < required_consensus:
            raise RuntimeError(
                "MINIMA fixed matching candidates did not form a stable consensus for "
                f"{feature_name!r}: accepted={int(feature_diagnostics['accepted_count'])}, "
                f"required={required_consensus}."
            )
        feature_diagnostics["sampled_count"] = float(sample_count)
        feature_diagnostics["failed_count"] = float(failed_counts[feature_name])
        homographies[feature_name] = consensus.tolist()
        diagnostics[feature_name] = feature_diagnostics
        logging.info(
            "Estimated fixed MINIMA homography for %s from %d/%d valid random frames: "
            "accepted=%d, median_projection_error=%.2f px.",
            feature_name,
            valid_count,
            sample_count,
            int(feature_diagnostics["accepted_count"]),
            feature_diagnostics["median_projection_error_px"],
        )

    config.thermal_fixed_matching_homographies = homographies
    config.thermal_fixed_matching_source_sizes = {
        key: list(value) for key, value in source_sizes.items()
    }
    config.thermal_fixed_matching_output_sizes = {
        key: list(value) for key, value in output_sizes.items()
    }
    config.thermal_fixed_matching_dataset_roots = dataset_roots
    config.thermal_fixed_matching_sample_indices = sample_indices
    config.thermal_fixed_matching_sample_timestamps = sample_timestamps
    config.thermal_fixed_matching_diagnostics = diagnostics
    validate = getattr(config, "_validate_thermal_fixed_matching_config", None)
    if callable(validate):
        validate()
    return fixed_matching_config_payload(config)
