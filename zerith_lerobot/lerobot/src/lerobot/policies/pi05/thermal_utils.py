#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Thermal-image adapters used by the PI0.5 integration."""

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torchvision.models import resnet18


def _disable_torch_compile(func):
    compiler_disable = getattr(getattr(torch, "compiler", None), "disable", None)
    if callable(compiler_disable):
        return compiler_disable(func)

    dynamo_disable = getattr(getattr(torch, "_dynamo", None), "disable", None)
    if callable(dynamo_disable):
        return dynamo_disable(func)

    return func


ANYTHERMAL_DINOV2_MEAN = (0.48145466, 0.4578275, 0.40821073)
ANYTHERMAL_DINOV2_STD = (0.26862954, 0.26130258, 0.27577711)
DINOV2_IMAGENET_MEAN = (0.485, 0.456, 0.406)
DINOV2_IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_ANYTHERMAL_CHECKPOINT_PATH = (
    "pretrained_checkpoints/backbone/AnyThermal_full/model20.pth"
)
DEFAULT_RGB_DINOV2_CHECKPOINT_PATH = (
    "pretrained_checkpoints/backbone/Dinov2/dinov2_vitb14_pretrain.pth"
)
DEFAULT_DINOV2_REPO_PATH = "dinov2"
DEFAULT_THERMAL_GREY_BACKGROUND_ROI = (0, 0, 64, 64)
DEFAULT_THERMAL_GREY_BACKGROUND_VALUE = 80.0
DEFAULT_THERMAL_BLACK_BACKGROUND_OUTLIER_THRESHOLD = 12.0
DEFAULT_THERMAL_BLACK_BACKGROUND_OUTLIER_RATIO = 2.0


def _project_root() -> Path:
    """Return the repository root that holds project-local checkpoints."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pretrained_checkpoints").is_dir() or (parent / "dinov2" / "hubconf.py").is_file():
            return parent
    return Path.cwd().resolve()


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(str(path)).expanduser()
    if path.is_absolute():
        return path
    return _project_root() / path


def image_to_bchw(images: Tensor, feature_name: str) -> Tensor:
    """Convert a four-dimensional image batch to ``[B, C, H, W]``."""
    if images.ndim != 4:
        raise ValueError(
            f"Image feature {feature_name} must have shape [B,C,H,W] or [B,H,W,C], "
            f"got {tuple(images.shape)}."
        )
    if images.shape[1] in {1, 3, 4}:
        return images
    if images.shape[-1] in {1, 3, 4}:
        return images.permute(0, 3, 1, 2)
    raise ValueError(
        f"Cannot determine the channel dimension for image feature {feature_name}: "
        f"{tuple(images.shape)}."
    )


def rgb_to_three_black_dominance_tensor(
    images: Tensor,
    *,
    feature_name: str,
    input_range: str = "auto",
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert RGB into red/green/blue black-importance views.

    Each view measures how much its selected channel exceeds the other two
    channels. Strong channel-specific evidence maps to black, while neutral
    pixels (including white/grey background) map to white. Outputs are
    three-channel tensors that preserve the input's layout, numeric range,
    device, and dtype.
    """
    if images.ndim != 4:
        raise ValueError(
            f"RGB three-black input {feature_name} must have shape [B,C,H,W] or "
            f"[B,H,W,C], got {tuple(images.shape)}."
        )
    if input_range not in {"auto", "minus_one_to_one", "zero_to_one", "uint8"}:
        raise ValueError(
            "input_range must be one of: auto, minus_one_to_one, zero_to_one, uint8"
        )

    channels_first = images.shape[1] in {3, 4}
    images_bchw = image_to_bchw(images, feature_name)
    if images_bchw.shape[1] not in {3, 4}:
        raise ValueError(
            f"RGB three-black input {feature_name} requires three or four channels, "
            f"got {images_bchw.shape[1]}."
        )

    rgb = images_bchw[:, :3].to(dtype=torch.float32)
    resolved_range = input_range
    if resolved_range == "auto":
        if not images_bchw.is_floating_point():
            resolved_range = "uint8"
        else:
            min_value = float(rgb.detach().amin().item())
            max_value = float(rgb.detach().amax().item())
            if min_value < -0.05:
                resolved_range = "minus_one_to_one"
            elif max_value <= 1.0 + 1e-5:
                resolved_range = "zero_to_one"
            else:
                resolved_range = "uint8"

    if resolved_range == "minus_one_to_one":
        rgb_unit = (rgb + 1.0) * 0.5
    elif resolved_range == "zero_to_one":
        rgb_unit = rgb
    else:
        rgb_unit = rgb / 255.0
    rgb_unit = rgb_unit.clamp(0.0, 1.0)

    views = []
    for channel_index in range(3):
        other_indices = [index for index in range(3) if index != channel_index]
        other_max = rgb_unit[:, other_indices].amax(dim=1, keepdim=True)
        color_evidence = (rgb_unit[:, channel_index : channel_index + 1] - other_max).clamp(
            0.0, 1.0
        )
        black_importance = 1.0 - color_evidence
        view = black_importance.repeat(1, 3, 1, 1)
        if resolved_range == "minus_one_to_one":
            view = view * 2.0 - 1.0
        elif resolved_range == "uint8":
            view = torch.round(view * 255.0)
        view = view.to(dtype=images.dtype)
        if not channels_first:
            view = view.permute(0, 2, 3, 1).contiguous()
        views.append(view)

    return views[0], views[1], views[2]


def _clip_background_roi(
    background_roi: tuple[int, int, int, int] | list[int],
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    if len(background_roi) != 4:
        raise ValueError(
            "thermal grey background ROI must be (top, left, height, width), "
            f"got {background_roi!r}."
        )
    top, left, roi_h, roi_w = (int(value) for value in background_roi)
    if roi_h <= 0 or roi_w <= 0:
        raise ValueError(
            "thermal grey background ROI height and width must be positive, "
            f"got {background_roi!r}."
        )
    if height <= 0 or width <= 0:
        raise ValueError(f"thermal grey image size must be positive, got {(height, width)}.")

    top = max(0, min(top, height - 1))
    left = max(0, min(left, width - 1))
    bottom = max(top + 1, min(top + roi_h, height))
    right = max(left + 1, min(left + roi_w, width))
    return top, left, bottom, right


def _background_normalized_thermal_gray_tensor(
    images: Tensor,
    *,
    feature_name: str,
    background_roi: tuple[int, int, int, int] | list[int],
    target_background: float,
) -> tuple[Tensor, bool, bool]:
    if images.ndim != 4:
        raise ValueError(
            f"Thermal grey input {feature_name} must have shape [B,C,H,W] or [B,H,W,C], "
            f"got {tuple(images.shape)}."
        )
    channels_first = images.shape[1] in {1, 3, 4}
    images_bchw = image_to_bchw(images, feature_name)
    _, channels, height, width = images_bchw.shape
    if channels < 1:
        raise ValueError(f"Thermal grey input {feature_name} has no channel dimension.")
    top, left, bottom, right = _clip_background_roi(background_roi, int(height), int(width))

    input_is_unit_float = images_bchw.is_floating_point()
    gray = images_bchw[:, :1].to(dtype=torch.float32)
    if input_is_unit_float:
        gray = gray * 255.0

    roi = gray[:, :, top:bottom, left:right].flatten(start_dim=1)
    bg_median = torch.quantile(roi, 0.5, dim=1).view(-1, 1, 1, 1)
    gray = torch.clamp(gray - bg_median + float(target_background), 0.0, 255.0)
    return gray, input_is_unit_float, channels_first


def _finish_grey_rgb_tensor(
    gray: Tensor,
    *,
    reference: Tensor,
    input_is_unit_float: bool,
    channels_first: bool,
) -> Tensor:
    grey_rgb = gray.repeat(1, 3, 1, 1)
    if input_is_unit_float:
        grey_rgb = grey_rgb / 255.0
        grey_rgb = grey_rgb.to(dtype=reference.dtype)
    else:
        grey_rgb = torch.round(grey_rgb).to(dtype=reference.dtype)

    if channels_first:
        return grey_rgb.contiguous()
    return grey_rgb.permute(0, 2, 3, 1).contiguous()


def thermal_to_background_normalized_grey_tensor(
    images: Tensor,
    *,
    feature_name: str,
    background_roi: tuple[int, int, int, int] | list[int] = DEFAULT_THERMAL_GREY_BACKGROUND_ROI,
    target_background: float = DEFAULT_THERMAL_GREY_BACKGROUND_VALUE,
) -> Tensor:
    """Convert thermal images to 3-channel grey with fixed background brightness.

    The thermal source is treated as one channel: if a decoded dataset/video frame
    has three channels, channel 0 is used rather than applying RGB grayscale
    weights. The fixed ROI median is shifted to ``target_background`` in uint8
    space:

    ``gray = clip(gray - bg_median + target_background, 0, 255)``.
    """
    gray, input_is_unit_float, channels_first = _background_normalized_thermal_gray_tensor(
        images,
        feature_name=feature_name,
        background_roi=background_roi,
        target_background=target_background,
    )
    return _finish_grey_rgb_tensor(
        gray,
        reference=images,
        input_is_unit_float=input_is_unit_float,
        channels_first=channels_first,
    )


def thermal_to_two_background_normalized_grey_tensor(
    images: Tensor,
    *,
    feature_name: str,
    background_roi: tuple[int, int, int, int] | list[int] = DEFAULT_THERMAL_GREY_BACKGROUND_ROI,
    target_background: float = DEFAULT_THERMAL_GREY_BACKGROUND_VALUE,
) -> tuple[Tensor, Tensor]:
    """Split ROI-normalized thermal grey into cold and hot 3-channel images.

    ``target_background`` is the shared boundary. Values below it are retained
    in the cold image and clipped upward in the hot image; values above it are
    retained in the hot image and clipped downward in the cold image.
    """
    gray, input_is_unit_float, channels_first = _background_normalized_thermal_gray_tensor(
        images,
        feature_name=feature_name,
        background_roi=background_roi,
        target_background=target_background,
    )
    boundary = torch.as_tensor(
        float(target_background),
        dtype=gray.dtype,
        device=gray.device,
    ).clamp(0.0, 255.0)
    cold_gray = torch.minimum(gray, boundary)
    hot_gray = torch.maximum(gray, boundary)
    cold_rgb = _finish_grey_rgb_tensor(
        cold_gray,
        reference=images,
        input_is_unit_float=input_is_unit_float,
        channels_first=channels_first,
    )
    hot_rgb = _finish_grey_rgb_tensor(
        hot_gray,
        reference=images,
        input_is_unit_float=input_is_unit_float,
        channels_first=channels_first,
    )
    return cold_rgb, hot_rgb


def twogrey_patch_evidence(
    cold_images: Tensor,
    hot_images: Tensor,
    *,
    target_background: float,
    token_count: int,
    black_importance: bool = False,
) -> tuple[Tensor, Tensor]:
    """Pool cold/hot importance onto a ViT token grid.

    ``twogrey`` measures deviations below/above its fixed grey boundary.
    ``twoblack`` maps both kinds of evidence to black, so their evidence is the
    distance from the shared white background instead.
    """
    if token_count <= 0:
        raise ValueError(f"token_count must be positive, got {token_count}.")
    cold = image_to_bchw(cold_images, "cold_images").detach().to(dtype=torch.float32)
    hot = image_to_bchw(hot_images, "hot_images").detach().to(dtype=torch.float32)
    if cold.shape != hot.shape:
        raise ValueError(
            "cold_images and hot_images must have the same shape, "
            f"got {tuple(cold.shape)} and {tuple(hot.shape)}."
        )

    cold_gray = cold[:, :3].mean(dim=1, keepdim=True)
    hot_gray = hot[:, :3].mean(dim=1, keepdim=True)
    # resize_with_pad_torch pads both streams with -1. A true important pixel
    # can reach -1 only while its counterpart remains at the background value.
    valid_pixels = ~((cold_gray <= -1.0 + 1e-5) & (hot_gray <= -1.0 + 1e-5))
    if black_importance:
        cold_evidence = ((1.0 - cold_gray).clamp_min(0.0) * 0.5).clamp(0.0, 1.0)
        hot_evidence = ((1.0 - hot_gray).clamp_min(0.0) * 0.5).clamp(0.0, 1.0)
    else:
        boundary = 2.0 * float(target_background) / 255.0 - 1.0
        boundary_tensor = cold_gray.new_tensor(boundary)
        cold_scale = max(boundary + 1.0, 1e-6)
        hot_scale = max(1.0 - boundary, 1e-6)
        cold_evidence = (
            (boundary_tensor - cold_gray).clamp_min(0.0) / cold_scale
        ).clamp(0.0, 1.0)
        hot_evidence = (
            (hot_gray - boundary_tensor).clamp_min(0.0) / hot_scale
        ).clamp(0.0, 1.0)
    cold_evidence = cold_evidence * valid_pixels.to(dtype=cold_evidence.dtype)
    hot_evidence = hot_evidence * valid_pixels.to(dtype=hot_evidence.dtype)

    grid_side = math.isqrt(token_count)
    patch_count = grid_side * grid_side
    prefix_count = token_count - patch_count
    if patch_count <= 0:
        raise ValueError(f"Could not infer a patch grid from token_count={token_count}.")

    def pool(values: Tensor) -> Tensor:
        patch_values = F.adaptive_avg_pool2d(values, (grid_side, grid_side)).flatten(start_dim=1)
        if prefix_count <= 0:
            return patch_values
        global_value = values.flatten(start_dim=1).mean(dim=1, keepdim=True)
        return torch.cat([global_value.expand(-1, prefix_count), patch_values], dim=1)

    return pool(cold_evidence), pool(hot_evidence)


def _three_top_background_rois(
    background_roi: tuple[int, int, int, int] | list[int],
    height: int,
    width: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """Build equal-size top-left, top-center, and top-right background ROIs.

    For this robust estimator, ``background_roi`` is interpreted as
    ``(top, side_margin, height, width)``. The second value positions the left
    ROI and is mirrored for the right ROI; the middle ROI is centered.
    """
    if len(background_roi) != 4:
        raise ValueError(
            "thermal black background ROI must be (top, side_margin, height, width), "
            f"got {background_roi!r}."
        )
    top, side_margin, roi_h, roi_w = (int(value) for value in background_roi)
    if side_margin < 0:
        raise ValueError(
            f"thermal black background ROI side_margin must be non-negative, got {side_margin}."
        )
    left_positions = (
        side_margin,
        max(0, (width - roi_w) // 2),
        max(0, width - side_margin - roi_w),
    )
    return tuple(
        _clip_background_roi((top, left, roi_h, roi_w), height, width)
        for left in left_positions
    )


def _robust_three_region_background_tensor(
    gray: Tensor,
    *,
    background_roi: tuple[int, int, int, int] | list[int],
    outlier_threshold: float,
    outlier_ratio: float,
) -> Tensor:
    """Estimate each batch item's background, rejecting one clear top-ROI outlier."""
    if outlier_threshold < 0:
        raise ValueError(
            "thermal black background outlier threshold must be non-negative, "
            f"got {outlier_threshold}."
        )
    if outlier_ratio < 1:
        raise ValueError(
            "thermal black background outlier ratio must be at least 1, "
            f"got {outlier_ratio}."
        )

    _, _, height, width = gray.shape
    boxes = _three_top_background_rois(background_roi, int(height), int(width))
    regions = [
        gray[:, :, top:bottom, left:right].flatten(start_dim=1)
        for top, left, bottom, right in boxes
    ]

    def median(values: Tensor) -> Tensor:
        sorted_values = torch.sort(values, dim=1).values
        midpoint = sorted_values.shape[1] // 2
        if sorted_values.shape[1] % 2:
            return sorted_values[:, midpoint]
        return 0.5 * (
            sorted_values[:, midpoint - 1] + sorted_values[:, midpoint]
        )

    medians = torch.stack(
        [median(region) for region in regions],
        dim=1,
    )

    pairs = ((0, 1), (0, 2), (1, 2))
    pair_differences = torch.stack(
        [(medians[:, first] - medians[:, second]).abs() for first, second in pairs],
        dim=1,
    )
    closest_pair = pair_differences.argmin(dim=1)
    pair_centers = torch.stack(
        [0.5 * (medians[:, first] + medians[:, second]) for first, second in pairs],
        dim=1,
    )
    outlier_indices = torch.tensor((2, 1, 0), device=gray.device)
    batch_indices = torch.arange(gray.shape[0], device=gray.device)
    selected_pair_difference = pair_differences[batch_indices, closest_pair]
    selected_pair_center = pair_centers[batch_indices, closest_pair]
    selected_outlier = outlier_indices[closest_pair]
    outlier_distance = (
        medians[batch_indices, selected_outlier] - selected_pair_center
    ).abs()
    reject_outlier = (outlier_distance >= float(outlier_threshold)) & (
        outlier_distance
        >= float(outlier_ratio) * selected_pair_difference.clamp_min(1e-6)
    )

    all_background = median(torch.cat(regions, dim=1))
    pair_backgrounds = torch.stack(
        [
            median(torch.cat((regions[first], regions[second]), dim=1))
            for first, second in pairs
        ],
        dim=1,
    )
    selected_pair_background = pair_backgrounds[batch_indices, closest_pair]
    return torch.where(reject_outlier, selected_pair_background, all_background).view(
        -1, 1, 1, 1
    )


def estimate_thermal_black_background_tensor(
    images: Tensor,
    *,
    feature_name: str,
    background_roi: tuple[int, int, int, int] | list[int] = DEFAULT_THERMAL_GREY_BACKGROUND_ROI,
    outlier_threshold: float = DEFAULT_THERMAL_BLACK_BACKGROUND_OUTLIER_THRESHOLD,
    outlier_ratio: float = DEFAULT_THERMAL_BLACK_BACKGROUND_OUTLIER_RATIO,
) -> Tensor:
    """Estimate robust per-frame thermal backgrounds in uint8 pixel units."""
    images_bchw = image_to_bchw(images, feature_name)
    if images_bchw.shape[1] < 1:
        raise ValueError(f"Thermal black input {feature_name} has no channel dimension.")
    gray = images_bchw[:, :1].to(dtype=torch.float32)
    if images_bchw.is_floating_point():
        gray = torch.round(gray * 255.0)
    return _robust_three_region_background_tensor(
        gray,
        background_roi=background_roi,
        outlier_threshold=float(outlier_threshold),
        outlier_ratio=float(outlier_ratio),
    )


def thermal_to_two_black_background_normalized_grey_tensor(
    images: Tensor,
    *,
    feature_name: str,
    background_roi: tuple[int, int, int, int] | list[int] = DEFAULT_THERMAL_GREY_BACKGROUND_ROI,
    target_background: float = DEFAULT_THERMAL_GREY_BACKGROUND_VALUE,
    outlier_threshold: float = DEFAULT_THERMAL_BLACK_BACKGROUND_OUTLIER_THRESHOLD,
    outlier_ratio: float = DEFAULT_THERMAL_BLACK_BACKGROUND_OUTLIER_RATIO,
    background: Tensor | None = None,
    valid_alpha: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Split thermal input into cold/hot views where stronger evidence is darker.

    The background is estimated from the top-left, top-center, and top-right
    regions. One region is ignored only when it clearly disagrees with the
    closest pair. Both output streams map the shared background to white and
    independently map their strongest representable deviation to black.
    """
    if images.ndim != 4:
        raise ValueError(
            f"Thermal black input {feature_name} must have shape [B,C,H,W] or [B,H,W,C], "
            f"got {tuple(images.shape)}."
        )
    channels_first = images.shape[1] in {1, 3, 4}
    images_bchw = image_to_bchw(images, feature_name)
    if images_bchw.shape[1] < 1:
        raise ValueError(f"Thermal black input {feature_name} has no channel dimension.")

    input_is_unit_float = images_bchw.is_floating_point()
    gray = images_bchw[:, :1].to(dtype=torch.float32)
    if input_is_unit_float:
        # LeRobot decodes video frames to unit floats from uint8 pixels. Restore
        # those pixel values exactly so training and NumPy eval preprocessing agree.
        gray = torch.round(gray * 255.0)
    if background is None:
        background = _robust_three_region_background_tensor(
            gray,
            background_roi=background_roi,
            outlier_threshold=float(outlier_threshold),
            outlier_ratio=float(outlier_ratio),
        )
    else:
        background = torch.as_tensor(background, dtype=gray.dtype, device=gray.device)
        if background.numel() != gray.shape[0]:
            raise ValueError(
                "Thermal background must contain one value per image, "
                f"got {background.numel()} for batch size {gray.shape[0]}."
            )
        background = background.reshape(-1, 1, 1, 1)
    boundary_value = float(int(min(max(float(target_background), 0.0), 255.0)))
    boundary = torch.as_tensor(boundary_value, dtype=gray.dtype, device=gray.device)
    # Match decoded uint8 visualization/eval semantics before remapping.
    normalized = torch.trunc(
        torch.clamp(gray - background + float(target_background), 0.0, 255.0)
    )
    cold_gray = torch.minimum(normalized, boundary)
    hot_gray = torch.maximum(normalized, boundary)

    if boundary_value > 0:
        cold_gray = torch.round(cold_gray * (255.0 / boundary_value))
    else:
        cold_gray = torch.full_like(cold_gray, 255.0)
    if boundary_value < 255:
        hot_gray = torch.round(
            (255.0 - hot_gray) * (255.0 / (255.0 - boundary_value))
        )
    else:
        hot_gray = torch.full_like(hot_gray, 255.0)

    if valid_alpha is not None:
        alpha = image_to_bchw(valid_alpha, f"{feature_name} valid_alpha")
        if alpha.shape[0] != gray.shape[0] or alpha.shape[-2:] != gray.shape[-2:]:
            raise ValueError(
                "Thermal valid_alpha must match the image batch and spatial shape, "
                f"got {tuple(alpha.shape)} for {tuple(gray.shape)}."
            )
        alpha = alpha[:, :1].to(dtype=gray.dtype, device=gray.device).clamp(0.0, 1.0)
        # The missing target area is white. The source-side outward padding
        # supplies the only transition band; original thermal pixels stay intact.
        cold_gray = cold_gray * alpha + 255.0 * (1.0 - alpha)
        hot_gray = hot_gray * alpha + 255.0 * (1.0 - alpha)

    return (
        _finish_grey_rgb_tensor(
            cold_gray,
            reference=images,
            input_is_unit_float=input_is_unit_float,
            channels_first=channels_first,
        ),
        _finish_grey_rgb_tensor(
            hot_gray,
            reference=images,
            input_is_unit_float=input_is_unit_float,
            channels_first=channels_first,
        ),
    )


def _background_normalized_thermal_gray_array(
    image: np.ndarray,
    *,
    background_roi: tuple[int, int, int, int] | list[int],
    target_background: float,
) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        gray = array[0]
    elif array.ndim == 3 and array.shape[-1] in {1, 3, 4}:
        gray = array[..., 0]
    else:
        raise ValueError(f"Thermal grey input must be HWC, CHW, or HW, got {array.shape}.")

    gray_f32 = gray.astype(np.float32, copy=False)
    if np.issubdtype(gray.dtype, np.floating):
        gray_f32 = gray_f32 * 255.0
    top, left, bottom, right = _clip_background_roi(background_roi, gray_f32.shape[0], gray_f32.shape[1])
    bg_median = float(np.median(gray_f32[top:bottom, left:right]))
    return np.clip(gray_f32 - bg_median + float(target_background), 0, 255).astype(np.uint8)


def thermal_to_background_normalized_grey_array(
    image: np.ndarray,
    *,
    background_roi: tuple[int, int, int, int] | list[int] = DEFAULT_THERMAL_GREY_BACKGROUND_ROI,
    target_background: float = DEFAULT_THERMAL_GREY_BACKGROUND_VALUE,
) -> np.ndarray:
    """Numpy variant for eval-time camera frames; returns HWC uint8 RGB grey."""
    gray_u8 = _background_normalized_thermal_gray_array(
        image,
        background_roi=background_roi,
        target_background=target_background,
    )
    return np.repeat(gray_u8[..., None], 3, axis=-1)


def thermal_to_two_background_normalized_grey_array(
    image: np.ndarray,
    *,
    background_roi: tuple[int, int, int, int] | list[int] = DEFAULT_THERMAL_GREY_BACKGROUND_ROI,
    target_background: float = DEFAULT_THERMAL_GREY_BACKGROUND_VALUE,
) -> tuple[np.ndarray, np.ndarray]:
    """Numpy variant for eval-time camera frames; returns cold/hot HWC uint8 RGB grey images."""
    gray_u8 = _background_normalized_thermal_gray_array(
        image,
        background_roi=background_roi,
        target_background=target_background,
    )
    boundary = np.uint8(np.clip(float(target_background), 0, 255))
    cold_gray = np.minimum(gray_u8, boundary)
    hot_gray = np.maximum(gray_u8, boundary)
    return (
        np.repeat(cold_gray[..., None], 3, axis=-1),
        np.repeat(hot_gray[..., None], 3, axis=-1),
    )


def thermal_to_two_black_background_normalized_grey_array(
    image: np.ndarray,
    *,
    background_roi: tuple[int, int, int, int] | list[int] = DEFAULT_THERMAL_GREY_BACKGROUND_ROI,
    target_background: float = DEFAULT_THERMAL_GREY_BACKGROUND_VALUE,
    outlier_threshold: float = DEFAULT_THERMAL_BLACK_BACKGROUND_OUTLIER_THRESHOLD,
    outlier_ratio: float = DEFAULT_THERMAL_BLACK_BACKGROUND_OUTLIER_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy eval-time variant of the training ``twoblack`` transform."""
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        gray = array[0]
    elif array.ndim == 3 and array.shape[-1] in {1, 3, 4}:
        gray = array[..., 0]
    else:
        raise ValueError(f"Thermal black input must be HWC, CHW, or HW, got {array.shape}.")

    gray_f32 = gray.astype(np.float32, copy=False)
    if np.issubdtype(gray.dtype, np.floating):
        gray_f32 = gray_f32 * 255.0
    if outlier_threshold < 0:
        raise ValueError(
            "thermal black background outlier threshold must be non-negative, "
            f"got {outlier_threshold}."
        )
    if outlier_ratio < 1:
        raise ValueError(
            "thermal black background outlier ratio must be at least 1, "
            f"got {outlier_ratio}."
        )

    boxes = _three_top_background_rois(
        background_roi,
        gray_f32.shape[0],
        gray_f32.shape[1],
    )
    regions = [
        gray_f32[top:bottom, left:right].reshape(-1)
        for top, left, bottom, right in boxes
    ]
    medians = [float(np.median(region)) for region in regions]
    pairs = ((0, 1), (0, 2), (1, 2))
    pair = min(pairs, key=lambda indices: abs(medians[indices[0]] - medians[indices[1]]))
    outlier_index = next(index for index in range(3) if index not in pair)
    pair_difference = abs(medians[pair[0]] - medians[pair[1]])
    pair_center = 0.5 * (medians[pair[0]] + medians[pair[1]])
    outlier_distance = abs(medians[outlier_index] - pair_center)
    reject_outlier = (
        outlier_distance >= float(outlier_threshold)
        and outlier_distance >= float(outlier_ratio) * max(pair_difference, 1e-6)
    )
    used_indices = pair if reject_outlier else (0, 1, 2)
    background = float(np.median(np.concatenate([regions[index] for index in used_indices])))

    boundary = int(np.uint8(np.clip(float(target_background), 0, 255)))
    normalized = np.clip(
        gray_f32 - background + float(target_background),
        0,
        255,
    ).astype(np.uint8)
    cold_gray = np.minimum(normalized, boundary)
    hot_gray = np.maximum(normalized, boundary)
    if boundary > 0:
        cold_gray = np.rint(cold_gray.astype(np.float32) * (255.0 / boundary)).astype(
            np.uint8
        )
    else:
        cold_gray = np.full_like(cold_gray, 255)
    if boundary < 255:
        hot_gray = np.rint(
            (255.0 - hot_gray.astype(np.float32)) * (255.0 / (255 - boundary))
        ).astype(np.uint8)
    else:
        hot_gray = np.full_like(hot_gray, 255)
    return (
        np.repeat(cold_gray[..., None], 3, axis=-1),
        np.repeat(hot_gray[..., None], 3, axis=-1),
    )


def mix_rgb_and_thermal_images(
    rgb: Tensor,
    thermal: Tensor,
    *,
    rgb_feature: str,
    thermal_feature: str,
    horizontal_offset: int,
    vertical_offset: int,
    thermal_width: int,
    thermal_weight: float,
    fill_color: tuple[int, int, int] | Tensor,
) -> Tensor:
    """Align thermal pixels to the RGB canvas and return a blended BCHW tensor."""
    rgb_bchw = image_to_bchw(rgb, rgb_feature).to(dtype=torch.float32)
    thermal_bchw = image_to_bchw(thermal, thermal_feature).to(
        device=rgb_bchw.device, dtype=torch.float32
    )

    if rgb_bchw.shape[0] != thermal_bchw.shape[0]:
        raise ValueError(
            "RGB and thermal batch sizes must match, "
            f"got {rgb_bchw.shape[0]} and {thermal_bchw.shape[0]}."
        )
    if rgb_bchw.shape[1] != 3 or thermal_bchw.shape[1] != 3:
        raise ValueError(
            "RGB/thermal mixing requires three-channel images, "
            f"got RGB={rgb_bchw.shape[1]} and thermal={thermal_bchw.shape[1]}."
        )
    if thermal_width <= 0:
        raise ValueError(f"thermal_width must be positive, got {thermal_width}.")

    _, channels, output_height, output_width = rgb_bchw.shape
    if not -output_width <= horizontal_offset <= thermal_width:
        raise ValueError(
            "horizontal_offset must be in "
            f"[-{output_width}, {thermal_width}], got {horizontal_offset}."
        )
    if not -output_height <= vertical_offset <= output_height:
        raise ValueError(
            f"vertical_offset must be in [-{output_height}, {output_height}], "
            f"got {vertical_offset}."
        )

    resized_thermal = F.interpolate(
        thermal_bchw,
        size=(output_height, thermal_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0.0, 1.0)

    fill = torch.as_tensor(fill_color, dtype=torch.float32, device=rgb_bchw.device)
    if fill.ndim == 1:
        fill = fill.unsqueeze(0)
    if fill.ndim != 2 or fill.shape[1] != channels or fill.shape[0] not in {
        1,
        rgb_bchw.shape[0],
    }:
        raise ValueError(
            "fill_color must have shape [3] or [B,3], "
            f"got {tuple(fill.shape)} for batch size {rgb_bchw.shape[0]}."
        )
    fill = fill / 255.0
    thermal_canvas = fill.view(fill.shape[0], channels, 1, 1).expand_as(rgb_bchw).clone()

    thermal_x = -horizontal_offset
    thermal_y = -vertical_offset
    dst_x0 = max(thermal_x, 0)
    dst_y0 = max(thermal_y, 0)
    dst_x1 = min(thermal_x + thermal_width, output_width)
    dst_y1 = min(thermal_y + output_height, output_height)

    if dst_x0 < dst_x1 and dst_y0 < dst_y1:
        src_x0 = dst_x0 - thermal_x
        src_y0 = dst_y0 - thermal_y
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        thermal_canvas[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = resized_thermal[
            :, :, src_y0:src_y1, src_x0:src_x1
        ]

    return thermal_weight * thermal_canvas + (1.0 - thermal_weight) * rgb_bchw


def get_modal_rgb_color(image: Tensor) -> tuple[int, int, int]:
    """Return the most frequent RGB triplet in a CHW or HWC image."""
    if image.ndim != 3:
        raise ValueError(
            "The thermal image used to resolve rgb_thermal_mix_fill_color must be "
            f"three-dimensional, got shape {tuple(image.shape)}."
        )
    if image.shape[0] == 3:
        image_chw = image
    elif image.shape[-1] == 3:
        image_chw = image.permute(2, 0, 1)
    else:
        raise ValueError(
            "RGB/thermal mixing requires a three-channel thermal image, "
            f"got shape {tuple(image.shape)}."
        )

    image_chw = image_chw.detach().to(device="cpu")
    if image_chw.is_floating_point():
        if image_chw.min() < 0 or image_chw.max() > 1:
            raise ValueError(
                "Floating-point thermal images must be in [0, 1] to resolve the modal fill color."
            )
        image_u8 = torch.round(image_chw * 255).to(torch.int64)
    else:
        image_u8 = image_chw.to(torch.int64).clamp(0, 255)

    packed_rgb = (image_u8[0] * (256**2) + image_u8[1] * 256 + image_u8[2]).flatten()
    modal_value = int(torch.mode(packed_rgb).values.item())
    return (
        (modal_value >> 16) & 0xFF,
        (modal_value >> 8) & 0xFF,
        modal_value & 0xFF,
    )


def _make_group_norm(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ThermalResNet18Encoder(nn.Module):
    """ResNet18 thermal encoder that emits PI05-compatible image tokens."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        output_dim: int,
        token_grid: tuple[int, int],
        dropout: float = 0.0,
    ):
        super().__init__()
        del hidden_dim
        backbone = resnet18(weights=None, norm_layer=_make_group_norm)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        feature_dim = 512
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(token_grid)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.output_proj = nn.Linear(feature_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, token_grid[0] * token_grid[1], output_dim)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, images: Tensor) -> Tensor:
        images = image_to_bchw(images, "thermal_encoder_input")
        features = self.pool(self.backbone(images))
        tokens = self.feature_norm(features.flatten(2).transpose(1, 2))
        tokens = self.output_proj(tokens)
        tokens = self.output_norm(
            tokens + self.position_embedding.to(dtype=tokens.dtype, device=tokens.device)
        )
        return self.dropout(tokens)


ThermalEncoder = ThermalResNet18Encoder


class AnyThermalEncoder(nn.Module):
    """AnyThermal DINOv2 backbone adapter that emits PI05-compatible image tokens."""

    def __init__(
        self,
        *,
        output_dim: int,
        model_type: str = "auto",
        checkpoint_path: str | None = DEFAULT_ANYTHERMAL_CHECKPOINT_PATH,
        dinov2_repo_path: str | None = None,
        freeze_backbone: bool = True,
        include_cls_token: bool = True,
        include_register_tokens: bool = True,
        input_range: str = "minus_one_to_one",
        dropout: float = 0.0,
    ):
        super().__init__()
        checkpoint = self._load_checkpoint_payload(checkpoint_path)
        checkpoint_model_type = checkpoint.get("student_model_type") if checkpoint is not None else None
        if model_type == "auto":
            model_type = checkpoint_model_type or "dinov2_vitb14"
        elif checkpoint_model_type is not None and checkpoint_model_type != model_type:
            logging.warning(
                "AnyThermal checkpoint reports student_model_type=%s but config requested %s. "
                "Using the configured model type.",
                checkpoint_model_type,
                model_type,
            )

        self.model_type = model_type
        self.include_cls_token = include_cls_token
        self.include_register_tokens = include_register_tokens
        self.freeze_backbone = freeze_backbone
        if input_range not in {"minus_one_to_one", "zero_to_one", "auto"}:
            raise ValueError("input_range must be one of: minus_one_to_one, zero_to_one, auto")
        self.input_range = input_range
        self.backbone = self._load_dinov2_backbone(model_type, dinov2_repo_path)
        self.patch_size = int(getattr(self.backbone, "patch_size", 14))
        if self.patch_size != 14:
            logging.warning(
                "AnyThermal was trained with DINOv2 patch_size=14, but %s reports patch_size=%s.",
                model_type,
                self.patch_size,
            )
        self.num_register_tokens = int(getattr(self.backbone, "num_register_tokens", 0))
        self.embed_dim = self._resolve_backbone_embed_dim()

        if checkpoint is not None:
            self._load_anythermal_backbone_state(checkpoint)

        self.feature_norm = nn.LayerNorm(self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

    @staticmethod
    def _default_checkpoint_path() -> Path:
        return _resolve_project_path(DEFAULT_ANYTHERMAL_CHECKPOINT_PATH)

    @classmethod
    def _resolve_checkpoint_path(cls, checkpoint_path: str | None) -> Path:
        if checkpoint_path is None:
            return cls._default_checkpoint_path()
        return _resolve_project_path(checkpoint_path)

    @classmethod
    def _load_checkpoint_payload(cls, checkpoint_path: str | None) -> dict | None:
        path = cls._resolve_checkpoint_path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                "AnyThermal checkpoint not found: "
                f"{path}. Put model20.pth under pretrained_checkpoints/backbone/AnyThermal_full "
                "or pass --policy.thermal_anythermal_checkpoint_path=/path/to/model20.pth."
            )
        return torch.load(path, map_location="cpu", weights_only=True)

    @staticmethod
    def _candidate_dinov2_repo_paths(repo_path: str | None = None) -> list[Path]:
        paths: list[Path] = []
        if repo_path is not None:
            paths.append(_resolve_project_path(repo_path))

        paths.extend(
            [
                _resolve_project_path(DEFAULT_DINOV2_REPO_PATH),
                _resolve_project_path("pretrained_checkpoints/Dinov2/dinov2"),
                _resolve_project_path("pretrained_checkpoints/dinov2"),
            ]
        )

        unique_paths: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            normalized = str(path)
            if normalized not in seen:
                unique_paths.append(path)
                seen.add(normalized)
        return unique_paths

    @classmethod
    def _load_dinov2_backbone(cls, model_type: str, repo_path: str | None = None) -> nn.Module:
        searched_paths = cls._candidate_dinov2_repo_paths(repo_path)
        for path in searched_paths:
            if (path / "hubconf.py").is_file():
                try:
                    return torch.hub.load(
                        str(path),
                        model_type,
                        source="local",
                        pretrained=False,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Found a local DINOv2 repo but failed to load the backbone from "
                        f"{path}. Check that this is facebookresearch/dinov2 and that "
                        f"thermal_anythermal_model_type={model_type!r} is valid."
                    ) from exc

        searched = "\n".join(f"  - {path}" for path in searched_paths)
        expected_path = _resolve_project_path(DEFAULT_DINOV2_REPO_PATH)
        raise FileNotFoundError(
            "DINOv2 repo was not found in the project. Put facebookresearch/dinov2 at:\n"
            f"  {expected_path}\n"
            "or pass --policy.thermal_anythermal_dinov2_repo_path=/path/to/dinov2.\n"
            "Searched paths:\n"
            f"{searched}"
        )

    def _resolve_backbone_embed_dim(self) -> int:
        embed_dim = getattr(self.backbone, "embed_dim", None)
        if embed_dim is not None:
            return int(embed_dim)
        norm = getattr(self.backbone, "norm", None)
        normalized_shape = getattr(norm, "normalized_shape", None)
        if normalized_shape:
            return int(normalized_shape[0])
        raise ValueError(f"Could not infer DINOv2 embed_dim for AnyThermal model {self.model_type!r}.")

    def _load_anythermal_backbone_state(self, checkpoint: dict) -> None:
        state = checkpoint.get("student_model_state_dict", checkpoint)
        if "backbone_model_state_dict" in state:
            state = state["backbone_model_state_dict"]
        try:
            self.backbone.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load AnyThermal backbone weights. Check that "
                f"thermal_anythermal_model_type={self.model_type!r} matches the checkpoint."
            ) from exc
        logging.info("Loaded AnyThermal backbone checkpoint for %s.", self.model_type)

    def _preprocess_for_dinov2(self, images: Tensor) -> Tensor:
        images = image_to_bchw(images, "anythermal_encoder_input").to(dtype=torch.float32)
        if images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1)
        elif images.shape[1] == 4:
            images = images[:, :3]
        elif images.shape[1] != 3:
            raise ValueError(
                "AnyThermal requires one-, three-, or four-channel images, "
                f"got {images.shape[1]} channels."
            )

        if self.input_range == "minus_one_to_one":
            images = (images + 1.0) * 0.5
        elif self.input_range == "auto" and images.detach().amin() < -0.05:
            images = (images + 1.0) * 0.5
        images = images.clamp(0.0, 1.0)

        height, width = images.shape[-2:]
        new_height = max(self.patch_size, (height // self.patch_size) * self.patch_size)
        new_width = max(self.patch_size, (width // self.patch_size) * self.patch_size)
        if (new_height, new_width) != (height, width):
            images = F.interpolate(
                images,
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        mean = torch.tensor(ANYTHERMAL_DINOV2_MEAN, device=images.device, dtype=images.dtype).view(
            1, 3, 1, 1
        )
        std = torch.tensor(ANYTHERMAL_DINOV2_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        return (images - mean) / std

    def _tokens_from_forward_features(self, images: Tensor) -> Tensor | None:
        forward_features = getattr(self.backbone, "forward_features", None)
        if not callable(forward_features):
            return None
        features = forward_features(images)
        if not isinstance(features, dict):
            return None

        tokens = []
        cls_token = features.get("x_norm_clstoken")
        if self.include_cls_token and cls_token is not None:
            tokens.append(cls_token[:, None])
        reg_tokens = features.get("x_norm_regtokens")
        if self.include_register_tokens and reg_tokens is not None and reg_tokens.numel() > 0:
            tokens.append(reg_tokens)
        patch_tokens = features.get("x_norm_patchtokens")
        if patch_tokens is not None:
            tokens.append(patch_tokens)
        if not tokens:
            return None
        return torch.cat(tokens, dim=1)

    def _tokens_from_final_block_hook(self, images: Tensor) -> Tensor:
        blocks = getattr(self.backbone, "blocks", None)
        if not blocks:
            raise RuntimeError("DINOv2 backbone does not expose forward_features or transformer blocks.")

        captured: dict[str, Tensor] = {}

        def hook_fn(_module, _inputs, output):
            captured["tokens"] = output[0] if isinstance(output, tuple) else output

        handle = blocks[-1].register_forward_hook(hook_fn)
        try:
            _ = self.backbone(images)
        finally:
            handle.remove()

        raw_tokens = captured.get("tokens")
        if raw_tokens is None:
            raise RuntimeError("Failed to capture AnyThermal DINOv2 final block tokens.")
        norm = getattr(self.backbone, "norm", None)
        if callable(norm):
            raw_tokens = norm(raw_tokens)

        tokens = []
        if self.include_cls_token:
            tokens.append(raw_tokens[:, :1])
        patch_start = 1
        register_end = patch_start + self.num_register_tokens
        if self.include_register_tokens and self.num_register_tokens > 0:
            tokens.append(raw_tokens[:, patch_start:register_end])
        tokens.append(raw_tokens[:, register_end:])
        return torch.cat(tokens, dim=1)

    def _extract_tokens(self, images: Tensor) -> Tensor:
        tokens = self._tokens_from_forward_features(images)
        if tokens is not None:
            return tokens
        return self._tokens_from_final_block_hook(images)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        images = self._preprocess_for_dinov2(images)
        if self.freeze_backbone:
            with torch.no_grad():
                tokens = self._extract_tokens(images)
        else:
            tokens = self._extract_tokens(images)
        tokens = self.feature_norm(tokens)
        tokens = self.output_proj(tokens)
        tokens = self.output_norm(tokens)
        return self.dropout(tokens)


class DINOv2RGBEncoder(AnyThermalEncoder):
    """RGB DINOv2 backbone adapter that emits PI05-compatible image tokens."""

    def __init__(
        self,
        *,
        output_dim: int,
        model_type: str = "dinov2_vitb14",
        checkpoint_path: str | None = DEFAULT_RGB_DINOV2_CHECKPOINT_PATH,
        dinov2_repo_path: str | None = None,
        freeze_backbone: bool = True,
        include_cls_token: bool = False,
        include_register_tokens: bool = False,
        input_range: str = "minus_one_to_one",
        dropout: float = 0.0,
    ):
        nn.Module.__init__(self)
        if not str(model_type or "").strip():
            raise ValueError("model_type must be non-empty")

        self.model_type = model_type
        self.include_cls_token = include_cls_token
        self.include_register_tokens = include_register_tokens
        self.freeze_backbone = freeze_backbone
        if input_range not in {"minus_one_to_one", "zero_to_one", "auto"}:
            raise ValueError("input_range must be one of: minus_one_to_one, zero_to_one, auto")
        self.input_range = input_range

        self.backbone = self._load_dinov2_backbone(model_type, dinov2_repo_path)
        self.patch_size = int(getattr(self.backbone, "patch_size", 14))
        if self.patch_size != 14:
            logging.warning(
                "RGB DINOv2 default checkpoint expects patch_size=14, but %s reports patch_size=%s.",
                model_type,
                self.patch_size,
            )
        self.num_register_tokens = int(getattr(self.backbone, "num_register_tokens", 0))
        self.embed_dim = self._resolve_backbone_embed_dim()
        self._load_rgb_dinov2_backbone_state(checkpoint_path)

        self.feature_norm = nn.LayerNorm(self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

    @classmethod
    def _resolve_rgb_checkpoint_path(cls, checkpoint_path: str | None) -> Path:
        if checkpoint_path is None:
            checkpoint_path = DEFAULT_RGB_DINOV2_CHECKPOINT_PATH
        return _resolve_project_path(checkpoint_path)

    @classmethod
    def _load_rgb_checkpoint_state_dict(cls, checkpoint_path: str | None) -> dict:
        path = cls._resolve_rgb_checkpoint_path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                "RGB DINOv2 checkpoint not found: "
                f"{path}. Put dinov2_vitb14_pretrain.pth under "
                "pretrained_checkpoints/backbone/Dinov2 or pass "
                "--policy.rgb_dinov2_checkpoint_path=/path/to/dinov2_vitb14_pretrain.pth."
            )
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError(f"RGB DINOv2 checkpoint must be a state dict, got {type(state).__name__}.")
        for key in ("model", "state_dict", "teacher", "student"):
            nested = state.get(key)
            if isinstance(nested, dict):
                state = nested
                break
        return {key.removeprefix("module."): value for key, value in state.items()}

    def _load_rgb_dinov2_backbone_state(self, checkpoint_path: str | None) -> None:
        state = self._load_rgb_checkpoint_state_dict(checkpoint_path)
        try:
            self.backbone.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load RGB DINOv2 backbone weights. Check that "
                f"rgb_dinov2_model_type={self.model_type!r} matches the checkpoint."
            ) from exc
        logging.info("Loaded RGB DINOv2 backbone checkpoint for %s.", self.model_type)

    def _preprocess_for_dinov2(self, images: Tensor) -> Tensor:
        images = image_to_bchw(images, "rgb_dinov2_encoder_input").to(dtype=torch.float32)
        if images.shape[1] == 4:
            images = images[:, :3]
        elif images.shape[1] != 3:
            raise ValueError(f"RGB DINOv2 requires three- or four-channel images, got {images.shape[1]}.")

        if self.input_range == "minus_one_to_one":
            images = (images + 1.0) * 0.5
        elif self.input_range == "auto" and images.detach().amin() < -0.05:
            images = (images + 1.0) * 0.5
        images = images.clamp(0.0, 1.0)

        height, width = images.shape[-2:]
        new_height = max(self.patch_size, (height // self.patch_size) * self.patch_size)
        new_width = max(self.patch_size, (width // self.patch_size) * self.patch_size)
        if (new_height, new_width) != (height, width):
            images = F.interpolate(
                images,
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        mean = torch.tensor(DINOV2_IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(
            1, 3, 1, 1
        )
        std = torch.tensor(DINOV2_IMAGENET_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        return (images - mean) / std


class AnyThermalResidualFusion(AnyThermalEncoder):
    """Inject AnyThermal patch context as a zero-initialized residual on RGB tokens."""

    def __init__(
        self,
        *,
        rgb_dim: int,
        adapter_dim: int = 512,
        decoder_hidden_dim: int = 1024,
        num_heads: int = 8,
        model_type: str = "auto",
        checkpoint_path: str | None = DEFAULT_ANYTHERMAL_CHECKPOINT_PATH,
        dinov2_repo_path: str | None = None,
        freeze_backbone: bool = True,
        input_range: str = "minus_one_to_one",
        dropout: float = 0.0,
    ):
        nn.Module.__init__(self)
        if adapter_dim <= 0:
            raise ValueError("adapter_dim must be positive")
        if decoder_hidden_dim <= 0:
            raise ValueError("decoder_hidden_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if adapter_dim % num_heads != 0:
            raise ValueError(f"num_heads must divide adapter_dim ({adapter_dim}), got {num_heads}.")

        checkpoint = self._load_checkpoint_payload(checkpoint_path)
        checkpoint_model_type = checkpoint.get("student_model_type") if checkpoint is not None else None
        if model_type == "auto":
            model_type = checkpoint_model_type or "dinov2_vitb14"
        elif checkpoint_model_type is not None and checkpoint_model_type != model_type:
            logging.warning(
                "AnyThermal checkpoint reports student_model_type=%s but config requested %s. "
                "Using the configured model type.",
                checkpoint_model_type,
                model_type,
            )

        self.model_type = model_type
        self.include_cls_token = False
        self.include_register_tokens = False
        self.freeze_backbone = freeze_backbone
        if input_range not in {"minus_one_to_one", "zero_to_one", "auto"}:
            raise ValueError("input_range must be one of: minus_one_to_one, zero_to_one, auto")
        self.input_range = input_range
        self.backbone = self._load_dinov2_backbone(model_type, dinov2_repo_path)
        self.patch_size = int(getattr(self.backbone, "patch_size", 14))
        if self.patch_size != 14:
            logging.warning(
                "AnyThermal was trained with DINOv2 patch_size=14, but %s reports patch_size=%s.",
                model_type,
                self.patch_size,
            )
        self.num_register_tokens = int(getattr(self.backbone, "num_register_tokens", 0))
        self.embed_dim = self._resolve_backbone_embed_dim()

        if checkpoint is not None:
            self._load_anythermal_backbone_state(checkpoint)

        self.thermal_norm = nn.LayerNorm(self.embed_dim)
        self.thermal_adapter = nn.Linear(self.embed_dim, adapter_dim)
        self.rgb_norm = nn.LayerNorm(rgb_dim)
        self.rgb_adapter = nn.Linear(rgb_dim, adapter_dim)
        self.cross_attn = nn.MultiheadAttention(
            adapter_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(adapter_dim)
        self.residual_decoder = nn.Sequential(
            nn.Linear(adapter_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, rgb_dim),
        )
        nn.init.zeros_(self.residual_decoder[-1].weight)
        nn.init.zeros_(self.residual_decoder[-1].bias)

        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

    def extract_thermal_patch_tokens(self, images: Tensor) -> Tensor:
        images = self._preprocess_for_dinov2(images)
        if self.freeze_backbone:
            with torch.no_grad():
                return self._extract_tokens(images)
        return self._extract_tokens(images)

    @_disable_torch_compile
    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        thermal_images: Tensor,
        rgb_token_mask: Tensor,
        thermal_image_mask: Tensor,
    ) -> Tensor:
        thermal_tokens = self.extract_thermal_patch_tokens(thermal_images)
        if thermal_tokens.shape[0] != rgb_tokens.shape[0]:
            raise ValueError(
                "RGB and thermal batch sizes must match, "
                f"got {rgb_tokens.shape[0]} and {thermal_tokens.shape[0]}."
            )

        batch_size, num_rgb_tokens = rgb_tokens.shape[:2]
        num_thermal_tokens = thermal_tokens.shape[1]
        rgb_token_mask = rgb_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        if rgb_token_mask.shape != (batch_size, num_rgb_tokens):
            raise ValueError(
                "rgb_token_mask must have shape [B, N_rgb], "
                f"got {tuple(rgb_token_mask.shape)} for RGB tokens {tuple(rgb_tokens.shape)}."
            )

        thermal_image_mask = thermal_image_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        if thermal_image_mask.ndim != 1 or thermal_image_mask.shape[0] != batch_size:
            raise ValueError(
                "thermal_image_mask must have shape [B], "
                f"got {tuple(thermal_image_mask.shape)} for batch size {batch_size}."
            )
        thermal_token_mask = thermal_image_mask[:, None].expand(batch_size, num_thermal_tokens)

        module_dtype = self.rgb_norm.weight.dtype
        rgb_query = self.rgb_adapter(self.rgb_norm(rgb_tokens.to(dtype=module_dtype)))
        thermal_tokens = thermal_tokens.to(device=rgb_tokens.device, dtype=module_dtype)
        thermal_kv = self.thermal_adapter(self.thermal_norm(thermal_tokens))

        key_padding_mask = ~thermal_token_mask
        all_thermal_padded = key_padding_mask.all(dim=1)
        if all_thermal_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_thermal_padded, 0] = False

        context, _ = self.cross_attn(
            query=rgb_query,
            key=thermal_kv,
            value=thermal_kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        thermal_valid = thermal_token_mask.any(dim=1)
        context = context.masked_fill(~thermal_valid[:, None, None], 0.0)
        residual = self.residual_decoder(self.context_norm(context))
        residual_mask = rgb_token_mask & thermal_valid[:, None]
        residual = residual.masked_fill(~residual_mask[:, :, None], 0.0)
        return rgb_tokens + residual.to(dtype=rgb_tokens.dtype)


class ResViTAttentionFusion(nn.Module):
    """Thermo-VL-style dual-attention residual fusion for projected ViT tokens."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int = 8,
        merge_hidden_dim: int | None = None,
        residual_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError(f"num_heads must divide embed_dim ({embed_dim}), got {num_heads}.")
        if merge_hidden_dim is None:
            merge_hidden_dim = embed_dim
        if residual_hidden_dim is None:
            residual_hidden_dim = embed_dim * 2
        if merge_hidden_dim <= 0:
            raise ValueError("merge_hidden_dim must be positive")
        if residual_hidden_dim <= 0:
            raise ValueError("residual_hidden_dim must be positive")

        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.thermal_norm = nn.LayerNorm(embed_dim)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.rgb_cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.merge_norm = nn.LayerNorm(embed_dim * 3)
        self.merge_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, merge_hidden_dim),
            nn.GELU(),
            nn.Linear(merge_hidden_dim, embed_dim),
        )
        self.refined_norm = nn.LayerNorm(embed_dim)
        self.residual_norm = nn.LayerNorm(embed_dim)
        self.residual_mlp = nn.Sequential(
            nn.Linear(embed_dim, residual_hidden_dim),
            nn.GELU(),
            nn.Linear(residual_hidden_dim, embed_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 1),
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

    @staticmethod
    def _validate_token_shape(name: str, tokens: Tensor, embed_dim: int) -> None:
        if tokens.ndim != 3 or tokens.shape[-1] != embed_dim:
            raise ValueError(
                f"{name} must have shape [B, N, {embed_dim}], got {tuple(tokens.shape)}."
            )

    @staticmethod
    def _validate_mask_shape(name: str, mask: Tensor, expected_shape: tuple[int, int]) -> None:
        if mask.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(mask.shape)}."
            )

    @staticmethod
    def _safe_cross_attention(
        attn: nn.MultiheadAttention,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if key.shape[1] == 0:
            valid = torch.zeros(query.shape[0], dtype=torch.bool, device=query.device)
            return torch.zeros_like(query), valid

        valid = key_mask.any(dim=1)
        # MultiheadAttention cannot consume a row with every key masked. Make
        # key 0 temporarily visible for those rows, then zero their outputs.
        # This tensor-only form stays inside torch.compile graphs.
        safe_key_mask = key_mask.clone()
        safe_key_mask[:, 0] = safe_key_mask[:, 0] | ~valid

        context, _ = attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=~safe_key_mask,
            need_weights=False,
        )
        context = context.masked_fill(~valid[:, None, None], 0.0)
        return context, valid

    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        thermal_tokens: Tensor,
        text_tokens: Tensor,
        rgb_token_mask: Tensor,
        thermal_token_mask: Tensor,
        text_token_mask: Tensor,
    ) -> Tensor:
        embed_dim = self.rgb_norm.normalized_shape[0]
        self._validate_token_shape("rgb_tokens", rgb_tokens, embed_dim)
        self._validate_token_shape("thermal_tokens", thermal_tokens, embed_dim)
        self._validate_token_shape("text_tokens", text_tokens, embed_dim)
        if rgb_tokens.shape[:2] != thermal_tokens.shape[:2]:
            raise ValueError(
                "ResViTAttention requires aligned RGB and thermal token grids, "
                f"got rgb={tuple(rgb_tokens.shape)} and thermal={tuple(thermal_tokens.shape)}."
            )
        if text_tokens.shape[0] != rgb_tokens.shape[0]:
            raise ValueError(
                "RGB, thermal, and text batch sizes must match, "
                f"got rgb={rgb_tokens.shape[0]}, thermal={thermal_tokens.shape[0]}, "
                f"text={text_tokens.shape[0]}."
            )

        batch_size, num_rgb_tokens = rgb_tokens.shape[:2]
        num_text_tokens = text_tokens.shape[1]
        rgb_token_mask = rgb_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        thermal_token_mask = thermal_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        text_token_mask = text_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        self._validate_mask_shape("rgb_token_mask", rgb_token_mask, (batch_size, num_rgb_tokens))
        self._validate_mask_shape(
            "thermal_token_mask",
            thermal_token_mask,
            (batch_size, num_rgb_tokens),
        )
        self._validate_mask_shape("text_token_mask", text_token_mask, (batch_size, num_text_tokens))

        module_dtype = self.rgb_norm.weight.dtype
        rgb_in = rgb_tokens.to(dtype=module_dtype)
        thermal_in = thermal_tokens.to(device=rgb_tokens.device, dtype=module_dtype)
        text_in = text_tokens.to(device=rgb_tokens.device, dtype=module_dtype)

        rgb_norm = self.rgb_norm(rgb_in)
        thermal_norm = self.thermal_norm(thermal_in)
        text_norm = self.text_norm(text_in)

        thermal_from_rgb, rgb_valid = self._safe_cross_attention(
            self.rgb_cross_attn,
            query=thermal_norm,
            key=rgb_norm,
            value=rgb_norm,
            key_mask=rgb_token_mask,
        )
        thermal_from_text, text_valid = self._safe_cross_attention(
            self.text_cross_attn,
            query=thermal_norm,
            key=text_norm,
            value=text_norm,
            key_mask=text_token_mask,
        )

        merged = self.merge_mlp(
            self.merge_norm(torch.cat([thermal_norm, thermal_from_rgb, thermal_from_text], dim=-1))
        )
        refined_thermal = self.refined_norm(thermal_in + self.dropout(merged))

        residual = self.residual_mlp(self.residual_norm(refined_thermal))
        gate = torch.sigmoid(self.gate(torch.cat([rgb_norm, refined_thermal], dim=-1)))
        residual = gate * residual

        residual_mask = thermal_token_mask & rgb_token_mask & rgb_valid[:, None] & text_valid[:, None]
        residual = residual.masked_fill(~residual_mask[:, :, None], 0.0)
        return rgb_tokens + residual.to(dtype=rgb_tokens.dtype)


class TwoGreyGateResViTFusion(nn.Module):
    """Text-conditioned scalar gates for projected cold/hot thermal residuals."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int = 8,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        temperature: float = 1.0,
        gate_init_std: float = 0.02,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError(f"num_heads must divide embed_dim ({embed_dim}), got {num_heads}.")
        if hidden_dim is None:
            hidden_dim = max(128, embed_dim // 4)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if gate_init_std < 0:
            raise ValueError("gate_init_std must be non-negative")

        self.temperature = float(temperature)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.thermal_norm = nn.LayerNorm(embed_dim)
        self.text_to_thermal_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        gate_input_dim = embed_dim * 5
        second_hidden_dim = max(64, hidden_dim // 2)
        self.gate_input_norm = nn.LayerNorm(gate_input_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden_dim, 2),
        )
        nn.init.normal_(self.gate_mlp[-1].weight, mean=0.0, std=gate_init_std)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    @staticmethod
    def _validate_token_shape(name: str, tokens: Tensor, embed_dim: int) -> None:
        if tokens.ndim != 3 or tokens.shape[-1] != embed_dim:
            raise ValueError(
                f"{name} must have shape [B, N, {embed_dim}], got {tuple(tokens.shape)}."
            )

    @staticmethod
    def _validate_mask_shape(name: str, mask: Tensor, expected_shape: tuple[int, int]) -> None:
        if mask.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(mask.shape)}."
            )

    @staticmethod
    def _masked_mean(tokens: Tensor, mask: Tensor) -> Tensor:
        mask = mask.to(device=tokens.device, dtype=torch.bool)
        masked = tokens.masked_fill(~mask[:, :, None], 0.0)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=tokens.dtype)
        return masked.sum(dim=1) / denom

    def _text_to_thermal_context(
        self,
        *,
        text_tokens: Tensor,
        thermal_tokens: Tensor,
        thermal_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        context, thermal_valid = ResViTAttentionFusion._safe_cross_attention(
            self.text_to_thermal_attn,
            query=text_tokens,
            key=thermal_tokens,
            value=thermal_tokens,
            key_mask=thermal_token_mask,
        )
        return context, thermal_valid

    def _compute_text_gate(
        self,
        *,
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        text_tokens: Tensor,
        cold_token_mask: Tensor,
        hot_token_mask: Tensor,
        text_token_mask: Tensor,
        mode_name: str,
    ) -> dict[str, Tensor]:
        embed_dim = self.text_norm.normalized_shape[0]
        self._validate_token_shape("cold_tokens", cold_tokens, embed_dim)
        self._validate_token_shape("hot_tokens", hot_tokens, embed_dim)
        self._validate_token_shape("text_tokens", text_tokens, embed_dim)
        if cold_tokens.shape != hot_tokens.shape:
            raise ValueError(
                f"{mode_name} requires aligned cold and hot thermal token grids, "
                f"got cold={tuple(cold_tokens.shape)} and hot={tuple(hot_tokens.shape)}."
            )
        if text_tokens.shape[0] != cold_tokens.shape[0]:
            raise ValueError(
                f"{mode_name} cold thermal, hot thermal, and text batch sizes must match, "
                f"got thermal={cold_tokens.shape[0]}, text={text_tokens.shape[0]}."
            )

        batch_size, num_thermal_tokens = cold_tokens.shape[:2]
        num_text_tokens = text_tokens.shape[1]
        cold_token_mask = cold_token_mask.to(device=cold_tokens.device, dtype=torch.bool)
        hot_token_mask = hot_token_mask.to(device=cold_tokens.device, dtype=torch.bool)
        text_token_mask = text_token_mask.to(device=cold_tokens.device, dtype=torch.bool)
        self._validate_mask_shape(
            "cold_token_mask", cold_token_mask, (batch_size, num_thermal_tokens)
        )
        self._validate_mask_shape(
            "hot_token_mask", hot_token_mask, (batch_size, num_thermal_tokens)
        )
        self._validate_mask_shape("text_token_mask", text_token_mask, (batch_size, num_text_tokens))

        module_dtype = self.text_norm.weight.dtype
        cold_in = cold_tokens.to(dtype=module_dtype)
        hot_in = hot_tokens.to(device=cold_tokens.device, dtype=module_dtype)
        text_in = text_tokens.to(device=cold_tokens.device, dtype=module_dtype)

        text_norm = self.text_norm(text_in)
        cold_norm = self.thermal_norm(cold_in)
        hot_norm = self.thermal_norm(hot_in)
        cold_context, cold_valid = self._text_to_thermal_context(
            text_tokens=text_norm,
            thermal_tokens=cold_norm,
            thermal_token_mask=cold_token_mask,
        )
        hot_context, hot_valid = self._text_to_thermal_context(
            text_tokens=text_norm,
            thermal_tokens=hot_norm,
            thermal_token_mask=hot_token_mask,
        )

        text_valid = text_token_mask.any(dim=1)
        text_pool = self._masked_mean(text_norm, text_token_mask)
        cold_pool = self._masked_mean(cold_context, text_token_mask)
        hot_pool = self._masked_mean(hot_context, text_token_mask)
        gate_input = torch.cat(
            [
                text_pool,
                cold_pool,
                hot_pool,
                cold_pool - text_pool,
                hot_pool - text_pool,
            ],
            dim=-1,
        )
        logits = self.gate_mlp(self.gate_input_norm(gate_input))
        multipliers = 2.0 * torch.softmax(logits / self.temperature, dim=-1)

        return {
            "cold_in": cold_in,
            "hot_in": hot_in,
            "cold_token_mask": cold_token_mask,
            "hot_token_mask": hot_token_mask,
            "text_valid": text_valid,
            "cold_valid": cold_valid,
            "hot_valid": hot_valid,
            "logits": logits,
            "multipliers": multipliers,
        }

    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        text_tokens: Tensor,
        rgb_token_mask: Tensor,
        cold_token_mask: Tensor,
        hot_token_mask: Tensor,
        text_token_mask: Tensor,
        base_cold_alpha: float = 1.0,
        base_hot_beta: float = 1.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        embed_dim = self.text_norm.normalized_shape[0]
        self._validate_token_shape("rgb_tokens", rgb_tokens, embed_dim)
        if rgb_tokens.shape != cold_tokens.shape or rgb_tokens.shape != hot_tokens.shape:
            raise ValueError(
                "GateResViT requires aligned RGB, cold thermal, and hot thermal token grids, "
                f"got rgb={tuple(rgb_tokens.shape)}, cold={tuple(cold_tokens.shape)}, "
                f"hot={tuple(hot_tokens.shape)}."
            )

        batch_size, num_rgb_tokens = rgb_tokens.shape[:2]
        rgb_token_mask = rgb_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        self._validate_mask_shape("rgb_token_mask", rgb_token_mask, (batch_size, num_rgb_tokens))
        gate_state = self._compute_text_gate(
            cold_tokens=cold_tokens,
            hot_tokens=hot_tokens,
            text_tokens=text_tokens,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
            mode_name="GateResViT",
        )

        module_dtype = self.text_norm.weight.dtype
        rgb_in = rgb_tokens.to(dtype=module_dtype)
        cold_in = gate_state["cold_in"]
        hot_in = gate_state["hot_in"]
        cold_token_mask = gate_state["cold_token_mask"]
        hot_token_mask = gate_state["hot_token_mask"]
        text_valid = gate_state["text_valid"]
        cold_valid = gate_state["cold_valid"]
        hot_valid = gate_state["hot_valid"]
        logits = gate_state["logits"]
        multipliers = gate_state["multipliers"]
        valid_gate = text_valid.to(dtype=module_dtype)
        cold_alpha = (
            float(base_cold_alpha)
            * multipliers[:, 0]
            * cold_valid.to(dtype=module_dtype)
            * valid_gate
        )
        hot_beta = (
            float(base_hot_beta)
            * multipliers[:, 1]
            * hot_valid.to(dtype=module_dtype)
            * valid_gate
        )

        cold_residual = cold_in.masked_fill(~cold_token_mask[:, :, None], 0.0)
        hot_residual = hot_in.masked_fill(~hot_token_mask[:, :, None], 0.0)
        fused = (
            rgb_in
            + cold_alpha[:, None, None] * cold_residual
            + hot_beta[:, None, None] * hot_residual
        )
        fused = torch.where(rgb_token_mask[:, :, None], fused, rgb_in)
        gate_info = {
            "cold_alpha": cold_alpha.detach(),
            "hot_beta": hot_beta.detach(),
            "cold_multiplier": multipliers[:, 0].detach(),
            "hot_multiplier": multipliers[:, 1].detach(),
            "logits": logits.detach(),
            "cold_valid": cold_valid.detach(),
            "hot_valid": hot_valid.detach(),
            "text_valid": text_valid.detach(),
        }
        return fused.to(dtype=rgb_tokens.dtype), gate_info


class TwoGreyGateTwoResViTFusion(TwoGreyGateResViTFusion):
    """Text-conditioned scalar gates emitted as separate cold/hot RGB residual streams."""

    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        text_tokens: Tensor,
        rgb_token_mask: Tensor,
        cold_token_mask: Tensor,
        hot_token_mask: Tensor,
        text_token_mask: Tensor,
        base_cold_alpha: float = 1.0,
        base_hot_beta: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        embed_dim = self.text_norm.normalized_shape[0]
        self._validate_token_shape("rgb_tokens", rgb_tokens, embed_dim)
        if rgb_tokens.shape != cold_tokens.shape or rgb_tokens.shape != hot_tokens.shape:
            raise ValueError(
                "GateTwoResViT requires aligned RGB, cold thermal, and hot thermal token grids, "
                f"got rgb={tuple(rgb_tokens.shape)}, cold={tuple(cold_tokens.shape)}, "
                f"hot={tuple(hot_tokens.shape)}."
            )

        batch_size, num_rgb_tokens = rgb_tokens.shape[:2]
        rgb_token_mask = rgb_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        self._validate_mask_shape("rgb_token_mask", rgb_token_mask, (batch_size, num_rgb_tokens))
        gate_state = self._compute_text_gate(
            cold_tokens=cold_tokens,
            hot_tokens=hot_tokens,
            text_tokens=text_tokens,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
            mode_name="GateTwoResViT",
        )

        module_dtype = self.text_norm.weight.dtype
        rgb_in = rgb_tokens.to(dtype=module_dtype)
        cold_in = gate_state["cold_in"]
        hot_in = gate_state["hot_in"]
        cold_token_mask = gate_state["cold_token_mask"]
        hot_token_mask = gate_state["hot_token_mask"]
        text_valid = gate_state["text_valid"]
        cold_valid = gate_state["cold_valid"]
        hot_valid = gate_state["hot_valid"]
        logits = gate_state["logits"]
        multipliers = gate_state["multipliers"]
        valid_gate = text_valid.to(dtype=module_dtype)
        cold_alpha = (
            float(base_cold_alpha)
            * multipliers[:, 0]
            * cold_valid.to(dtype=module_dtype)
            * valid_gate
        )
        hot_beta = (
            float(base_hot_beta)
            * multipliers[:, 1]
            * hot_valid.to(dtype=module_dtype)
            * valid_gate
        )

        cold_output_mask = rgb_token_mask & cold_token_mask
        hot_output_mask = rgb_token_mask & hot_token_mask
        cold_residual = cold_in.masked_fill(~cold_token_mask[:, :, None], 0.0)
        hot_residual = hot_in.masked_fill(~hot_token_mask[:, :, None], 0.0)
        cold_out = rgb_in + cold_alpha[:, None, None] * cold_residual
        hot_out = rgb_in + hot_beta[:, None, None] * hot_residual
        cold_out = cold_out.masked_fill(~cold_output_mask[:, :, None], 0.0)
        hot_out = hot_out.masked_fill(~hot_output_mask[:, :, None], 0.0)

        gate_info = {
            "cold_alpha": cold_alpha.detach(),
            "hot_beta": hot_beta.detach(),
            "cold_multiplier": multipliers[:, 0].detach(),
            "hot_multiplier": multipliers[:, 1].detach(),
            "cold_residual_scale": cold_alpha.detach(),
            "hot_residual_scale": hot_beta.detach(),
            "logits": logits.detach(),
            "cold_valid": cold_valid.detach(),
            "hot_valid": hot_valid.detach(),
            "text_valid": text_valid.detach(),
            "cold_residual_valid": cold_output_mask.any(dim=1).detach(),
            "hot_residual_valid": hot_output_mask.any(dim=1).detach(),
        }
        return (
            cold_out.to(dtype=rgb_tokens.dtype),
            cold_output_mask,
            hot_out.to(dtype=rgb_tokens.dtype),
            hot_output_mask,
            gate_info,
        )


class OneGreyGateOneResViTFusion(nn.Module):
    """Text-conditioned scalar gate for one projected thermal residual on head RGB."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int = 8,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        temperature: float = 1.0,
        gate_init_std: float = 0.02,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError(f"num_heads must divide embed_dim ({embed_dim}), got {num_heads}.")
        if hidden_dim is None:
            hidden_dim = max(128, embed_dim // 4)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if gate_init_std < 0:
            raise ValueError("gate_init_std must be non-negative")

        self.temperature = float(temperature)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.thermal_norm = nn.LayerNorm(embed_dim)
        self.text_to_thermal_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        gate_input_dim = embed_dim * 3
        second_hidden_dim = max(64, hidden_dim // 2)
        self.gate_input_norm = nn.LayerNorm(gate_input_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden_dim, 1),
        )
        nn.init.normal_(self.gate_mlp[-1].weight, mean=0.0, std=gate_init_std)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        thermal_tokens: Tensor,
        text_tokens: Tensor,
        rgb_token_mask: Tensor,
        thermal_token_mask: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        embed_dim = self.text_norm.normalized_shape[0]
        TwoGreyGateResViTFusion._validate_token_shape("rgb_tokens", rgb_tokens, embed_dim)
        TwoGreyGateResViTFusion._validate_token_shape(
            "thermal_tokens",
            thermal_tokens,
            embed_dim,
        )
        TwoGreyGateResViTFusion._validate_token_shape("text_tokens", text_tokens, embed_dim)
        if rgb_tokens.shape != thermal_tokens.shape:
            raise ValueError(
                "GateOneResViT requires aligned RGB and thermal projected token grids, "
                f"got rgb={tuple(rgb_tokens.shape)} and thermal={tuple(thermal_tokens.shape)}."
            )
        if text_tokens.shape[0] != rgb_tokens.shape[0]:
            raise ValueError(
                "GateOneResViT RGB, thermal, and text batch sizes must match, "
                f"got rgb={rgb_tokens.shape[0]}, thermal={thermal_tokens.shape[0]}, "
                f"text={text_tokens.shape[0]}."
            )

        batch_size, num_tokens = rgb_tokens.shape[:2]
        num_text_tokens = text_tokens.shape[1]
        rgb_token_mask = rgb_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        thermal_token_mask = thermal_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        text_token_mask = text_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        TwoGreyGateResViTFusion._validate_mask_shape(
            "rgb_token_mask",
            rgb_token_mask,
            (batch_size, num_tokens),
        )
        TwoGreyGateResViTFusion._validate_mask_shape(
            "thermal_token_mask",
            thermal_token_mask,
            (batch_size, num_tokens),
        )
        TwoGreyGateResViTFusion._validate_mask_shape(
            "text_token_mask",
            text_token_mask,
            (batch_size, num_text_tokens),
        )

        module_dtype = self.text_norm.weight.dtype
        rgb_in = rgb_tokens.to(dtype=module_dtype)
        thermal_in = thermal_tokens.to(device=rgb_tokens.device, dtype=module_dtype)
        text_in = text_tokens.to(device=rgb_tokens.device, dtype=module_dtype)

        text_norm = self.text_norm(text_in)
        thermal_norm = self.thermal_norm(thermal_in)
        thermal_context, thermal_valid = ResViTAttentionFusion._safe_cross_attention(
            self.text_to_thermal_attn,
            query=text_norm,
            key=thermal_norm,
            value=thermal_norm,
            key_mask=thermal_token_mask,
        )
        text_valid = text_token_mask.any(dim=1)
        text_pool = TwoGreyGateResViTFusion._masked_mean(text_norm, text_token_mask)
        thermal_pool = TwoGreyGateResViTFusion._masked_mean(thermal_context, text_token_mask)
        gate_input = torch.cat(
            [
                text_pool,
                thermal_pool,
                thermal_pool - text_pool,
            ],
            dim=-1,
        )
        logits = self.gate_mlp(self.gate_input_norm(gate_input)).squeeze(-1)
        alpha = torch.sigmoid(logits / self.temperature)
        alpha = alpha * (text_valid & thermal_valid).to(dtype=module_dtype)

        output_mask = rgb_token_mask
        thermal_residual = thermal_in.masked_fill(~thermal_token_mask[:, :, None], 0.0)
        fused = rgb_in + alpha[:, None, None] * thermal_residual
        fused = torch.where(rgb_token_mask[:, :, None], fused, rgb_in)

        gate_info = {
            "thermal_alpha": alpha.detach(),
            "thermal_weight": alpha.detach(),
            "thermal_residual_scale": alpha.detach(),
            "logits": logits.detach(),
            "thermal_valid": thermal_valid.detach(),
            "text_valid": text_valid.detach(),
            "residual_valid": output_mask.any(dim=1).detach(),
        }
        return fused.to(dtype=rgb_tokens.dtype), output_mask, gate_info


class RGBThreeBlackGateThreeResViTFusion(nn.Module):
    """Emit red/green/blue text-gated residual streams on one base RGB token grid."""

    _COLOR_NAMES = ("red", "green", "blue")

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int = 8,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        temperature: float = 1.0,
        gate_init_std: float = 0.02,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError(f"num_heads must divide embed_dim ({embed_dim}), got {num_heads}.")
        if hidden_dim is None:
            hidden_dim = max(128, embed_dim // 4)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if gate_init_std < 0:
            raise ValueError("gate_init_std must be non-negative")

        self.temperature = float(temperature)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.color_norm = nn.LayerNorm(embed_dim)
        self.text_to_color_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        gate_input_dim = embed_dim * 7
        second_hidden_dim = max(64, hidden_dim // 2)
        self.gate_input_norm = nn.LayerNorm(gate_input_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden_dim, 3),
        )
        nn.init.normal_(self.gate_mlp[-1].weight, mean=0.0, std=gate_init_std)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        red_tokens: Tensor,
        green_tokens: Tensor,
        blue_tokens: Tensor,
        text_tokens: Tensor,
        rgb_token_mask: Tensor,
        red_token_mask: Tensor,
        green_token_mask: Tensor,
        blue_token_mask: Tensor,
        text_token_mask: Tensor,
        base_red_alpha: float = 1.0,
        base_green_alpha: float = 1.0,
        base_blue_alpha: float = 1.0,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        dict[str, Tensor],
    ]:
        embed_dim = self.text_norm.normalized_shape[0]
        TwoGreyGateResViTFusion._validate_token_shape("rgb_tokens", rgb_tokens, embed_dim)
        TwoGreyGateResViTFusion._validate_token_shape("text_tokens", text_tokens, embed_dim)
        color_tokens = (red_tokens, green_tokens, blue_tokens)
        color_masks = (red_token_mask, green_token_mask, blue_token_mask)
        for color_name, tokens in zip(self._COLOR_NAMES, color_tokens, strict=True):
            TwoGreyGateResViTFusion._validate_token_shape(
                f"{color_name}_tokens", tokens, embed_dim
            )
            if tokens.shape != rgb_tokens.shape:
                raise ValueError(
                    "RGB GateTwoResViT requires aligned original/red/green/blue token grids, "
                    f"got rgb={tuple(rgb_tokens.shape)} and "
                    f"{color_name}={tuple(tokens.shape)}."
                )
        if text_tokens.shape[0] != rgb_tokens.shape[0]:
            raise ValueError(
                "RGB GateTwoResViT image and text batch sizes must match, "
                f"got rgb={rgb_tokens.shape[0]} and text={text_tokens.shape[0]}."
            )

        batch_size, num_rgb_tokens = rgb_tokens.shape[:2]
        num_text_tokens = text_tokens.shape[1]
        rgb_token_mask = rgb_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        text_token_mask = text_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        TwoGreyGateResViTFusion._validate_mask_shape(
            "rgb_token_mask", rgb_token_mask, (batch_size, num_rgb_tokens)
        )
        TwoGreyGateResViTFusion._validate_mask_shape(
            "text_token_mask", text_token_mask, (batch_size, num_text_tokens)
        )

        module_dtype = self.text_norm.weight.dtype
        rgb_in = rgb_tokens.to(dtype=module_dtype)
        text_in = text_tokens.to(device=rgb_tokens.device, dtype=module_dtype)
        text_norm = self.text_norm(text_in)
        text_valid = text_token_mask.any(dim=1)
        text_pool = TwoGreyGateResViTFusion._masked_mean(text_norm, text_token_mask)

        color_inputs = []
        normalized_color_masks = []
        color_validity = []
        color_pools = []
        for color_name, tokens, token_mask in zip(
            self._COLOR_NAMES,
            color_tokens,
            color_masks,
            strict=True,
        ):
            token_mask = token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
            TwoGreyGateResViTFusion._validate_mask_shape(
                f"{color_name}_token_mask", token_mask, (batch_size, num_rgb_tokens)
            )
            color_in = tokens.to(device=rgb_tokens.device, dtype=module_dtype)
            color_norm = self.color_norm(color_in)
            color_context, color_valid = ResViTAttentionFusion._safe_cross_attention(
                self.text_to_color_attn,
                query=text_norm,
                key=color_norm,
                value=color_norm,
                key_mask=token_mask,
            )
            color_inputs.append(color_in)
            normalized_color_masks.append(token_mask)
            color_validity.append(color_valid)
            color_pools.append(
                TwoGreyGateResViTFusion._masked_mean(color_context, text_token_mask)
            )

        gate_input = torch.cat(
            [
                text_pool,
                *color_pools,
                *(color_pool - text_pool for color_pool in color_pools),
            ],
            dim=-1,
        )
        logits = self.gate_mlp(self.gate_input_norm(gate_input))
        multipliers = 3.0 * torch.softmax(logits / self.temperature, dim=-1)
        base_alphas = (base_red_alpha, base_green_alpha, base_blue_alpha)

        output_tokens = []
        output_masks = []
        gate_info: dict[str, Tensor] = {
            "logits": logits.detach(),
            "text_valid": text_valid.detach(),
        }
        for color_index, (
            color_name,
            color_in,
            color_mask,
            color_valid,
            base_alpha,
        ) in enumerate(
            zip(
                self._COLOR_NAMES,
                color_inputs,
                normalized_color_masks,
                color_validity,
                base_alphas,
                strict=True,
            )
        ):
            alpha = (
                float(base_alpha)
                * multipliers[:, color_index]
                * color_valid.to(dtype=module_dtype)
                * text_valid.to(dtype=module_dtype)
            )
            output_mask = rgb_token_mask & color_mask
            residual = color_in.masked_fill(~color_mask[:, :, None], 0.0)
            output = rgb_in + alpha[:, None, None] * residual
            output = output.masked_fill(~output_mask[:, :, None], 0.0)
            output_tokens.append(output.to(dtype=rgb_tokens.dtype))
            output_masks.append(output_mask)
            gate_info[f"{color_name}_alpha"] = alpha.detach()
            gate_info[f"{color_name}_multiplier"] = multipliers[:, color_index].detach()
            gate_info[f"{color_name}_residual_scale"] = alpha.detach()
            gate_info[f"{color_name}_valid"] = color_valid.detach()
            gate_info[f"{color_name}_residual_valid"] = output_mask.any(dim=1).detach()

        return (
            output_tokens[0],
            output_masks[0],
            output_tokens[1],
            output_masks[1],
            output_tokens[2],
            output_masks[2],
            gate_info,
        )


class TwoGreyGateViTFusion(TwoGreyGateResViTFusion):
    """Text-conditioned cold/hot mixture returned as one independent thermal stream."""

    def forward(
        self,
        *,
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        text_tokens: Tensor,
        cold_token_mask: Tensor,
        hot_token_mask: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        gate_state = self._compute_text_gate(
            cold_tokens=cold_tokens,
            hot_tokens=hot_tokens,
            text_tokens=text_tokens,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
            mode_name="GateViT",
        )

        cold_in = gate_state["cold_in"]
        hot_in = gate_state["hot_in"]
        cold_token_mask = gate_state["cold_token_mask"]
        hot_token_mask = gate_state["hot_token_mask"]
        text_valid = gate_state["text_valid"]
        cold_valid = gate_state["cold_valid"]
        hot_valid = gate_state["hot_valid"]
        logits = gate_state["logits"]
        multipliers = gate_state["multipliers"]

        raw_weights = multipliers * 0.5
        available = torch.stack([cold_valid, hot_valid], dim=-1) & text_valid[:, None]
        weights = raw_weights * available.to(dtype=raw_weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        cold_weight = weights[:, 0]
        hot_weight = weights[:, 1]

        cold_stream = cold_in.masked_fill(~cold_token_mask[:, :, None], 0.0)
        hot_stream = hot_in.masked_fill(~hot_token_mask[:, :, None], 0.0)
        fused = (
            cold_weight[:, None, None] * cold_stream
            + hot_weight[:, None, None] * hot_stream
        )
        fused_token_mask = (
            (cold_token_mask & cold_valid[:, None])
            | (hot_token_mask & hot_valid[:, None])
        ) & text_valid[:, None]
        fused = fused.masked_fill(~fused_token_mask[:, :, None], 0.0)

        gate_info = {
            "cold_alpha": cold_weight.detach(),
            "hot_beta": hot_weight.detach(),
            "cold_weight": cold_weight.detach(),
            "hot_weight": hot_weight.detach(),
            "cold_multiplier": multipliers[:, 0].detach(),
            "hot_multiplier": multipliers[:, 1].detach(),
            "logits": logits.detach(),
            "cold_valid": cold_valid.detach(),
            "hot_valid": hot_valid.detach(),
            "text_valid": text_valid.detach(),
        }
        return fused.to(dtype=cold_tokens.dtype), fused_token_mask, gate_info


class TwoGreyPatchGateViTFusion(nn.Module):
    """Lightweight text/RGB-conditioned cold/hot/relevance gate for every patch."""

    def __init__(
        self,
        *,
        embed_dim: int,
        gate_dim: int = 256,
        num_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        temperature: float = 1.0,
        gate_init_std: float = 0.01,
        evidence_strength: float = 1.0,
        evidence_floor: float = 0.05,
        detach_head_rgb: bool = True,
    ):
        super().__init__()
        if embed_dim <= 0 or gate_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embed_dim, gate_dim, and hidden_dim must be positive")
        if num_heads <= 0 or gate_dim % num_heads != 0:
            raise ValueError(f"num_heads must be positive and divide gate_dim ({gate_dim})")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(gate_init_std) or gate_init_std < 0:
            raise ValueError("gate_init_std must be finite and non-negative")
        if not math.isfinite(evidence_strength) or evidence_strength < 0:
            raise ValueError("evidence_strength must be finite and non-negative")
        if not math.isfinite(evidence_floor) or not 0 < evidence_floor <= 1:
            raise ValueError("evidence_floor must be finite and in (0, 1]")

        self.embed_dim = int(embed_dim)
        self.gate_dim = int(gate_dim)
        self.temperature = float(temperature)
        self.evidence_strength = float(evidence_strength)
        self.evidence_floor = float(evidence_floor)
        self.detach_head_rgb = bool(detach_head_rgb)

        self.image_norm = nn.LayerNorm(embed_dim)
        self.image_adapter = nn.Linear(embed_dim, gate_dim)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.text_adapter = nn.Linear(embed_dim, gate_dim)
        self.query_norm = nn.LayerNorm(gate_dim)
        self.text_cross_attn = nn.MultiheadAttention(
            gate_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.rgb_cross_attn = nn.MultiheadAttention(
            gate_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        gate_input_dim = gate_dim * 6
        second_hidden_dim = max(64, hidden_dim // 2)
        self.gate_input_norm = nn.LayerNorm(gate_input_dim)
        self.patch_gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden_dim, 3),
        )
        self.global_text_gate = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Linear(gate_dim, second_hidden_dim),
            nn.GELU(),
            nn.Linear(second_hidden_dim, 2),
        )
        nn.init.normal_(self.patch_gate_mlp[-1].weight, mean=0.0, std=gate_init_std)
        nn.init.zeros_(self.patch_gate_mlp[-1].bias)
        nn.init.normal_(self.global_text_gate[-1].weight, mean=0.0, std=gate_init_std)
        nn.init.zeros_(self.global_text_gate[-1].bias)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _validate_tokens(name: str, tokens: Tensor, embed_dim: int) -> None:
        if tokens.ndim != 3 or tokens.shape[-1] != embed_dim:
            raise ValueError(
                f"{name} must have shape [B, N, {embed_dim}], got {tuple(tokens.shape)}."
            )

    @staticmethod
    def _validate_mask(name: str, mask: Tensor, shape: tuple[int, int]) -> Tensor:
        mask = mask.to(dtype=torch.bool)
        if mask.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(mask.shape)}.")
        return mask

    @staticmethod
    def _masked_mean(tokens: Tensor, mask: Tensor) -> Tensor:
        masked = tokens.masked_fill(~mask[:, :, None], 0.0)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=tokens.dtype)
        return masked.sum(dim=1) / denom

    @staticmethod
    def _safe_context(
        attention: nn.MultiheadAttention,
        *,
        query: Tensor,
        context: Tensor,
        context_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return ResViTAttentionFusion._safe_cross_attention(
            attention,
            query=query,
            key=context,
            value=context,
            key_mask=context_mask,
        )

    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        text_tokens: Tensor,
        rgb_token_mask: Tensor,
        cold_token_mask: Tensor,
        hot_token_mask: Tensor,
        text_token_mask: Tensor,
        cold_evidence: Tensor | None = None,
        hot_evidence: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        self._validate_tokens("rgb_tokens", rgb_tokens, self.embed_dim)
        self._validate_tokens("cold_tokens", cold_tokens, self.embed_dim)
        self._validate_tokens("hot_tokens", hot_tokens, self.embed_dim)
        self._validate_tokens("text_tokens", text_tokens, self.embed_dim)
        if cold_tokens.shape != hot_tokens.shape:
            raise ValueError(
                "PatchGateViT requires aligned cold/hot token grids, "
                f"got {tuple(cold_tokens.shape)} and {tuple(hot_tokens.shape)}."
            )
        if not (
            rgb_tokens.shape[0] == cold_tokens.shape[0] == text_tokens.shape[0]
        ):
            raise ValueError("PatchGateViT RGB, thermal, and text batch sizes must match.")

        batch_size, token_count = cold_tokens.shape[:2]
        rgb_token_mask = self._validate_mask(
            "rgb_token_mask",
            rgb_token_mask.to(device=cold_tokens.device),
            tuple(rgb_tokens.shape[:2]),
        )
        cold_token_mask = self._validate_mask(
            "cold_token_mask",
            cold_token_mask.to(device=cold_tokens.device),
            (batch_size, token_count),
        )
        hot_token_mask = self._validate_mask(
            "hot_token_mask",
            hot_token_mask.to(device=cold_tokens.device),
            (batch_size, token_count),
        )
        text_token_mask = self._validate_mask(
            "text_token_mask",
            text_token_mask.to(device=cold_tokens.device),
            tuple(text_tokens.shape[:2]),
        )

        module_dtype = self.image_norm.weight.dtype
        cold_in = cold_tokens.to(dtype=module_dtype)
        hot_in = hot_tokens.to(device=cold_tokens.device, dtype=module_dtype)
        rgb_in = rgb_tokens.to(device=cold_tokens.device, dtype=module_dtype)
        if self.detach_head_rgb:
            rgb_in = rgb_in.detach()
        text_in = text_tokens.to(device=cold_tokens.device, dtype=module_dtype)

        cold_small = self.image_adapter(self.image_norm(cold_in))
        hot_small = self.image_adapter(self.image_norm(hot_in))
        rgb_small = self.image_adapter(self.image_norm(rgb_in))
        text_small = self.text_adapter(self.text_norm(text_in))
        pair_query = self.query_norm(0.5 * (cold_small + hot_small))
        text_context, text_valid = self._safe_context(
            self.text_cross_attn,
            query=pair_query,
            context=text_small,
            context_mask=text_token_mask,
        )
        rgb_context, rgb_valid = self._safe_context(
            self.rgb_cross_attn,
            query=pair_query,
            context=rgb_small,
            context_mask=rgb_token_mask,
        )

        gate_input = torch.cat(
            [
                cold_small,
                hot_small,
                cold_small - hot_small,
                cold_small * hot_small,
                text_context,
                rgb_context,
            ],
            dim=-1,
        )
        patch_logits = self.patch_gate_mlp(self.gate_input_norm(gate_input))
        text_pool = self._masked_mean(text_small, text_token_mask)
        global_text_logits = self.global_text_gate(text_pool)
        global_text_logits = global_text_logits * text_valid[:, None].to(
            dtype=global_text_logits.dtype
        )
        temperature_logits = patch_logits[:, :, :2] + global_text_logits[:, None, :]

        evidence_shape = (batch_size, token_count)
        if cold_evidence is None:
            cold_evidence = temperature_logits.new_zeros(evidence_shape)
        else:
            cold_evidence = cold_evidence.to(
                device=temperature_logits.device,
                dtype=temperature_logits.dtype,
            )
        if hot_evidence is None:
            hot_evidence = temperature_logits.new_zeros(evidence_shape)
        else:
            hot_evidence = hot_evidence.to(
                device=temperature_logits.device,
                dtype=temperature_logits.dtype,
            )
        if cold_evidence.shape != evidence_shape or hot_evidence.shape != evidence_shape:
            raise ValueError(
                "PatchGateViT evidence must match thermal tokens [B, N], "
                f"got cold={tuple(cold_evidence.shape)}, hot={tuple(hot_evidence.shape)}, "
                f"expected={evidence_shape}."
            )
        cold_evidence = cold_evidence.clamp(0.0, 1.0)
        hot_evidence = hot_evidence.clamp(0.0, 1.0)
        evidence = torch.stack([cold_evidence, hot_evidence], dim=-1)
        evidence_prior = torch.log(evidence + self.evidence_floor)
        temperature_logits = temperature_logits + self.evidence_strength * evidence_prior

        available = torch.stack([cold_token_mask, hot_token_mask], dim=-1)
        masked_logits = temperature_logits.masked_fill(~available, -1e4)
        temperature_weights = torch.softmax(masked_logits / self.temperature, dim=-1)
        temperature_weights = temperature_weights * available.to(dtype=temperature_weights.dtype)
        temperature_weights = temperature_weights / temperature_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        cold_weight = temperature_weights[:, :, 0]
        hot_weight = temperature_weights[:, :, 1]

        # 2 * sigmoid(0) == 1 gives an identity-strength start while retaining
        # a healthy derivative. The model can suppress irrelevant patches
        # toward zero or amplify useful thermal patches toward two.
        relevance_logits = patch_logits[:, :, 2]
        relevance = 2.0 * torch.sigmoid(relevance_logits)
        fused_token_mask = cold_token_mask | hot_token_mask
        relevance = relevance * fused_token_mask.to(dtype=relevance.dtype)
        cold_stream = cold_in.masked_fill(~cold_token_mask[:, :, None], 0.0)
        hot_stream = hot_in.masked_fill(~hot_token_mask[:, :, None], 0.0)
        fused = relevance[:, :, None] * (
            cold_weight[:, :, None] * cold_stream
            + hot_weight[:, :, None] * hot_stream
        )
        fused = fused.masked_fill(~fused_token_mask[:, :, None], 0.0)

        gate_info = {
            "cold_alpha": cold_weight.detach(),
            "hot_beta": hot_weight.detach(),
            "cold_weight": cold_weight.detach(),
            "hot_weight": hot_weight.detach(),
            "patch_relevance": relevance.detach(),
            "temperature_logits": temperature_logits.detach(),
            "relevance_logits": relevance_logits.detach(),
            "global_text_logits": global_text_logits.detach(),
            "cold_evidence": cold_evidence.detach(),
            "hot_evidence": hot_evidence.detach(),
            "cold_valid": cold_token_mask.any(dim=1).detach(),
            "hot_valid": hot_token_mask.any(dim=1).detach(),
            "text_valid": text_valid.detach(),
            "rgb_valid": rgb_valid.detach(),
        }
        return fused.to(dtype=cold_tokens.dtype), fused_token_mask, gate_info


class TwoGreyPatchSingleGateResViTFusion(TwoGreyPatchGateViTFusion):
    """Full patch text-conditioned cold/hot residuals fused into one head-RGB stream."""

    def __init__(self, **kwargs):
        gate_init_std_value = kwargs.get("gate_init_std", 0.01)
        gate_init_std = 0.01 if gate_init_std_value is None else float(gate_init_std_value)
        super().__init__(**kwargs)
        self.rgb_cross_attn = None
        hidden_dim = int(self.patch_gate_mlp[0].out_features)
        second_hidden_dim = int(self.patch_gate_mlp[3].out_features)
        dropout = 0.0
        for module in self.patch_gate_mlp:
            if isinstance(module, nn.Dropout):
                dropout = float(module.p)
                break
        gate_input_dim = self.gate_dim * 5
        self.gate_input_norm = nn.LayerNorm(gate_input_dim)
        self.patch_gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden_dim, 3),
        )
        nn.init.normal_(self.patch_gate_mlp[-1].weight, mean=0.0, std=gate_init_std)
        nn.init.zeros_(self.patch_gate_mlp[-1].bias)

    @_disable_torch_compile
    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        text_tokens: Tensor,
        rgb_token_mask: Tensor,
        cold_token_mask: Tensor,
        hot_token_mask: Tensor,
        text_token_mask: Tensor,
        cold_evidence: Tensor | None = None,
        hot_evidence: Tensor | None = None,
        base_cold_alpha: float = 1.0,
        base_hot_beta: float = 1.0,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        self._validate_tokens("rgb_tokens", rgb_tokens, self.embed_dim)
        self._validate_tokens("cold_tokens", cold_tokens, self.embed_dim)
        self._validate_tokens("hot_tokens", hot_tokens, self.embed_dim)
        self._validate_tokens("text_tokens", text_tokens, self.embed_dim)
        if rgb_tokens.shape != cold_tokens.shape or rgb_tokens.shape != hot_tokens.shape:
            raise ValueError(
                "PatchSingleGateResViT requires aligned projected head RGB, cold, and hot "
                "token grids, got "
                f"rgb={tuple(rgb_tokens.shape)}, cold={tuple(cold_tokens.shape)}, "
                f"hot={tuple(hot_tokens.shape)}."
            )
        if text_tokens.shape[0] != cold_tokens.shape[0]:
            raise ValueError("PatchSingleGateResViT thermal and text batch sizes must match.")

        batch_size, token_count = cold_tokens.shape[:2]
        rgb_token_mask = self._validate_mask(
            "rgb_token_mask",
            rgb_token_mask.to(device=cold_tokens.device),
            tuple(rgb_tokens.shape[:2]),
        )
        cold_token_mask = self._validate_mask(
            "cold_token_mask",
            cold_token_mask.to(device=cold_tokens.device),
            (batch_size, token_count),
        )
        hot_token_mask = self._validate_mask(
            "hot_token_mask",
            hot_token_mask.to(device=cold_tokens.device),
            (batch_size, token_count),
        )
        text_token_mask = self._validate_mask(
            "text_token_mask",
            text_token_mask.to(device=cold_tokens.device),
            tuple(text_tokens.shape[:2]),
        )

        module_dtype = self.image_norm.weight.dtype
        rgb_in = rgb_tokens.to(device=cold_tokens.device, dtype=module_dtype)
        cold_in = cold_tokens.to(device=cold_tokens.device, dtype=module_dtype)
        hot_in = hot_tokens.to(device=cold_tokens.device, dtype=module_dtype)
        text_in = text_tokens.to(device=cold_tokens.device, dtype=module_dtype)

        cold_small = self.image_adapter(self.image_norm(cold_in))
        hot_small = self.image_adapter(self.image_norm(hot_in))
        text_small = self.text_adapter(self.text_norm(text_in))
        pair_query = self.query_norm(0.5 * (cold_small + hot_small))
        text_context, text_valid = self._safe_context(
            self.text_cross_attn,
            query=pair_query,
            context=text_small,
            context_mask=text_token_mask,
        )

        gate_input = torch.cat(
            [
                cold_small,
                hot_small,
                cold_small - hot_small,
                cold_small * hot_small,
                text_context,
            ],
            dim=-1,
        )
        patch_logits = self.patch_gate_mlp(self.gate_input_norm(gate_input))
        text_pool = self._masked_mean(text_small, text_token_mask)
        global_text_logits = self.global_text_gate(text_pool)
        global_text_logits = global_text_logits * text_valid[:, None].to(
            dtype=global_text_logits.dtype
        )
        temperature_logits = patch_logits[:, :, :2] + global_text_logits[:, None, :]

        evidence_shape = (batch_size, token_count)
        if cold_evidence is None:
            cold_evidence = temperature_logits.new_zeros(evidence_shape)
        else:
            cold_evidence = cold_evidence.to(
                device=temperature_logits.device,
                dtype=temperature_logits.dtype,
            )
        if hot_evidence is None:
            hot_evidence = temperature_logits.new_zeros(evidence_shape)
        else:
            hot_evidence = hot_evidence.to(
                device=temperature_logits.device,
                dtype=temperature_logits.dtype,
            )
        if cold_evidence.shape != evidence_shape or hot_evidence.shape != evidence_shape:
            raise ValueError(
                "PatchSingleGateResViT evidence must match thermal tokens [B, N], "
                f"got cold={tuple(cold_evidence.shape)}, hot={tuple(hot_evidence.shape)}, "
                f"expected={evidence_shape}."
            )
        cold_evidence = cold_evidence.clamp(0.0, 1.0)
        hot_evidence = hot_evidence.clamp(0.0, 1.0)
        evidence = torch.stack([cold_evidence, hot_evidence], dim=-1)
        evidence_prior = torch.log(evidence + self.evidence_floor)
        temperature_logits = temperature_logits + self.evidence_strength * evidence_prior

        available = torch.stack([cold_token_mask, hot_token_mask], dim=-1)
        masked_logits = temperature_logits.masked_fill(~available, -1e4)
        temperature_weights = torch.softmax(masked_logits / self.temperature, dim=-1)
        temperature_weights = temperature_weights * available.to(dtype=temperature_weights.dtype)
        temperature_weights = temperature_weights / temperature_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        cold_weight = temperature_weights[:, :, 0]
        hot_weight = temperature_weights[:, :, 1]
        relevance_logits = patch_logits[:, :, 2]
        relevance = 2.0 * torch.sigmoid(relevance_logits)
        fused_token_mask = cold_token_mask | hot_token_mask
        relevance = relevance * fused_token_mask.to(dtype=relevance.dtype)

        cold_scale = float(base_cold_alpha) * cold_weight * relevance
        hot_scale = float(base_hot_beta) * hot_weight * relevance
        cold_residual = cold_in.masked_fill(~cold_token_mask[:, :, None], 0.0)
        hot_residual = hot_in.masked_fill(~hot_token_mask[:, :, None], 0.0)
        fused = rgb_in + cold_scale[:, :, None] * cold_residual + hot_scale[:, :, None] * hot_residual
        fused = torch.where(rgb_token_mask[:, :, None], fused, rgb_in)

        gate_info = {
            "cold_alpha": cold_scale,
            "hot_beta": hot_scale,
            "cold_weight": cold_weight,
            "hot_weight": hot_weight,
            "cold_residual_scale": cold_scale,
            "hot_residual_scale": hot_scale,
            "patch_relevance": relevance,
            "temperature_logits": temperature_logits,
            "relevance_logits": relevance_logits,
            "global_text_logits": global_text_logits,
            "cold_evidence": cold_evidence,
            "hot_evidence": hot_evidence,
            "cold_valid": cold_token_mask.any(dim=1),
            "hot_valid": hot_token_mask.any(dim=1),
            "text_valid": text_valid,
            "rgb_valid": rgb_token_mask.any(dim=1),
            "cold_residual_valid": (rgb_token_mask & cold_token_mask).any(dim=1),
            "hot_residual_valid": (rgb_token_mask & hot_token_mask).any(dim=1),
        }
        return fused.to(dtype=rgb_tokens.dtype), rgb_token_mask, gate_info


class TwoGreySmallPatchGateViTFusion(TwoGreyPatchGateViTFusion):
    """PatchGateViT with a coarse square gate grid shared across nearby ViT patches."""

    def __init__(self, *, gate_grid: tuple[int, int] = (4, 4), **kwargs):
        super().__init__(**kwargs)
        gate_rows, gate_cols = (int(value) for value in gate_grid)
        if gate_rows <= 0 or gate_cols <= 0:
            raise ValueError(f"gate_grid entries must be positive, got {gate_grid}.")
        self.gate_grid = (gate_rows, gate_cols)

    @staticmethod
    def _infer_spatial_grid(token_count: int, target_grid: tuple[int, int]) -> tuple[int, int]:
        root = int(round(math.sqrt(token_count)))
        if root * root == token_count:
            return root, root

        target_rows, target_cols = target_grid
        target_ratio = target_cols / max(target_rows, 1)
        candidates = []
        for rows in range(1, int(math.sqrt(token_count)) + 1):
            if token_count % rows != 0:
                continue
            cols = token_count // rows
            candidates.append((rows, cols))
            if rows != cols:
                candidates.append((cols, rows))
        if not candidates:
            return 1, token_count
        return min(candidates, key=lambda shape: abs((shape[1] / shape[0]) - target_ratio))

    @staticmethod
    def _pool_token_grid(values: Tensor, grid_shape: tuple[int, int], target_grid: tuple[int, int]) -> Tensor:
        batch_size, token_count, channels = values.shape
        rows, cols = grid_shape
        pooled = F.adaptive_avg_pool2d(
            values.transpose(1, 2).reshape(batch_size, channels, rows, cols),
            target_grid,
        )
        return pooled.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _pool_mask_grid(mask: Tensor, grid_shape: tuple[int, int], target_grid: tuple[int, int]) -> Tensor:
        batch_size, token_count = mask.shape
        rows, cols = grid_shape
        pooled = F.adaptive_max_pool2d(mask.to(dtype=torch.float32).reshape(batch_size, 1, rows, cols), target_grid)
        return pooled.flatten(1).to(dtype=torch.bool)

    @staticmethod
    def _pool_evidence_grid(
        evidence: Tensor,
        mask: Tensor,
        grid_shape: tuple[int, int],
        target_grid: tuple[int, int],
    ) -> Tensor:
        batch_size, token_count = evidence.shape
        rows, cols = grid_shape
        evidence_grid = evidence.reshape(batch_size, 1, rows, cols)
        mask_grid = mask.to(device=evidence.device, dtype=evidence.dtype).reshape(batch_size, 1, rows, cols)
        pooled_evidence = F.adaptive_avg_pool2d(evidence_grid * mask_grid, target_grid)
        pooled_mask = F.adaptive_avg_pool2d(mask_grid, target_grid).clamp_min(1e-6)
        return (pooled_evidence / pooled_mask).flatten(1).clamp(0.0, 1.0)

    @staticmethod
    def _expand_grid_values(
        values: Tensor,
        *,
        source_grid: tuple[int, int],
        target_grid: tuple[int, int],
    ) -> Tensor:
        batch_size, coarse_count = values.shape[:2]
        channels = 1 if values.ndim == 2 else int(values.shape[2])
        rows, cols = source_grid
        target_rows, target_cols = target_grid
        grid_values = values.reshape(batch_size, rows, cols, channels).permute(0, 3, 1, 2)
        expanded = F.interpolate(grid_values, size=(target_rows, target_cols), mode="nearest")
        expanded = expanded.permute(0, 2, 3, 1).reshape(batch_size, target_rows * target_cols, channels)
        return expanded.squeeze(-1) if values.ndim == 2 else expanded

    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        text_tokens: Tensor,
        rgb_token_mask: Tensor,
        cold_token_mask: Tensor,
        hot_token_mask: Tensor,
        text_token_mask: Tensor,
        cold_evidence: Tensor | None = None,
        hot_evidence: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        self._validate_tokens("rgb_tokens", rgb_tokens, self.embed_dim)
        self._validate_tokens("cold_tokens", cold_tokens, self.embed_dim)
        self._validate_tokens("hot_tokens", hot_tokens, self.embed_dim)
        self._validate_tokens("text_tokens", text_tokens, self.embed_dim)
        if cold_tokens.shape != hot_tokens.shape:
            raise ValueError(
                "SmallPatchGateViT requires aligned cold/hot token grids, "
                f"got {tuple(cold_tokens.shape)} and {tuple(hot_tokens.shape)}."
            )
        if not (
            rgb_tokens.shape[0] == cold_tokens.shape[0] == text_tokens.shape[0]
        ):
            raise ValueError("SmallPatchGateViT RGB, thermal, and text batch sizes must match.")

        batch_size, token_count = cold_tokens.shape[:2]
        grid_shape = self._infer_spatial_grid(token_count, self.gate_grid)
        if grid_shape[0] * grid_shape[1] != token_count:
            raise ValueError(
                "SmallPatchGateViT could not map thermal tokens to a 2D grid: "
                f"token_count={token_count}, inferred_grid={grid_shape}."
            )

        rgb_token_mask = self._validate_mask(
            "rgb_token_mask",
            rgb_token_mask.to(device=cold_tokens.device),
            tuple(rgb_tokens.shape[:2]),
        )
        cold_token_mask = self._validate_mask(
            "cold_token_mask",
            cold_token_mask.to(device=cold_tokens.device),
            (batch_size, token_count),
        )
        hot_token_mask = self._validate_mask(
            "hot_token_mask",
            hot_token_mask.to(device=cold_tokens.device),
            (batch_size, token_count),
        )
        text_token_mask = self._validate_mask(
            "text_token_mask",
            text_token_mask.to(device=cold_tokens.device),
            tuple(text_tokens.shape[:2]),
        )

        module_dtype = self.image_norm.weight.dtype
        cold_in = cold_tokens.to(dtype=module_dtype)
        hot_in = hot_tokens.to(device=cold_tokens.device, dtype=module_dtype)
        rgb_in = rgb_tokens.to(device=cold_tokens.device, dtype=module_dtype)
        if self.detach_head_rgb:
            rgb_in = rgb_in.detach()
        text_in = text_tokens.to(device=cold_tokens.device, dtype=module_dtype)

        cold_small = self.image_adapter(self.image_norm(cold_in))
        hot_small = self.image_adapter(self.image_norm(hot_in))
        rgb_small = self.image_adapter(self.image_norm(rgb_in))
        text_small = self.text_adapter(self.text_norm(text_in))

        coarse_cold = self._pool_token_grid(cold_small, grid_shape, self.gate_grid)
        coarse_hot = self._pool_token_grid(hot_small, grid_shape, self.gate_grid)
        coarse_cold_mask = self._pool_mask_grid(cold_token_mask, grid_shape, self.gate_grid)
        coarse_hot_mask = self._pool_mask_grid(hot_token_mask, grid_shape, self.gate_grid)
        coarse_query = self.query_norm(0.5 * (coarse_cold + coarse_hot))
        text_context, text_valid = self._safe_context(
            self.text_cross_attn,
            query=coarse_query,
            context=text_small,
            context_mask=text_token_mask,
        )
        rgb_context, rgb_valid = self._safe_context(
            self.rgb_cross_attn,
            query=coarse_query,
            context=rgb_small,
            context_mask=rgb_token_mask,
        )

        gate_input = torch.cat(
            [
                coarse_cold,
                coarse_hot,
                coarse_cold - coarse_hot,
                coarse_cold * coarse_hot,
                text_context,
                rgb_context,
            ],
            dim=-1,
        )
        coarse_patch_logits = self.patch_gate_mlp(self.gate_input_norm(gate_input))
        text_pool = self._masked_mean(text_small, text_token_mask)
        global_text_logits = self.global_text_gate(text_pool)
        global_text_logits = global_text_logits * text_valid[:, None].to(
            dtype=global_text_logits.dtype
        )
        coarse_temperature_logits = coarse_patch_logits[:, :, :2] + global_text_logits[:, None, :]

        evidence_shape = (batch_size, token_count)
        if cold_evidence is None:
            cold_evidence = coarse_temperature_logits.new_zeros(evidence_shape)
        else:
            cold_evidence = cold_evidence.to(
                device=coarse_temperature_logits.device,
                dtype=coarse_temperature_logits.dtype,
            )
        if hot_evidence is None:
            hot_evidence = coarse_temperature_logits.new_zeros(evidence_shape)
        else:
            hot_evidence = hot_evidence.to(
                device=coarse_temperature_logits.device,
                dtype=coarse_temperature_logits.dtype,
            )
        if cold_evidence.shape != evidence_shape or hot_evidence.shape != evidence_shape:
            raise ValueError(
                "SmallPatchGateViT evidence must match thermal tokens [B, N], "
                f"got cold={tuple(cold_evidence.shape)}, hot={tuple(hot_evidence.shape)}, "
                f"expected={evidence_shape}."
            )
        cold_evidence = cold_evidence.clamp(0.0, 1.0)
        hot_evidence = hot_evidence.clamp(0.0, 1.0)
        coarse_cold_evidence = self._pool_evidence_grid(
            cold_evidence, cold_token_mask, grid_shape, self.gate_grid
        )
        coarse_hot_evidence = self._pool_evidence_grid(
            hot_evidence, hot_token_mask, grid_shape, self.gate_grid
        )
        coarse_evidence = torch.stack([coarse_cold_evidence, coarse_hot_evidence], dim=-1)
        evidence_prior = torch.log(coarse_evidence + self.evidence_floor)
        coarse_temperature_logits = coarse_temperature_logits + self.evidence_strength * evidence_prior

        coarse_available = torch.stack([coarse_cold_mask, coarse_hot_mask], dim=-1)
        masked_logits = coarse_temperature_logits.masked_fill(~coarse_available, -1e4)
        coarse_temperature_weights = torch.softmax(masked_logits / self.temperature, dim=-1)
        coarse_temperature_weights = coarse_temperature_weights * coarse_available.to(
            dtype=coarse_temperature_weights.dtype
        )
        coarse_temperature_weights = coarse_temperature_weights / coarse_temperature_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        coarse_cold_weight = coarse_temperature_weights[:, :, 0]
        coarse_hot_weight = coarse_temperature_weights[:, :, 1]
        coarse_relevance_logits = coarse_patch_logits[:, :, 2]
        coarse_relevance = 2.0 * torch.sigmoid(coarse_relevance_logits)
        coarse_fused_mask = coarse_cold_mask | coarse_hot_mask
        coarse_relevance = coarse_relevance * coarse_fused_mask.to(dtype=coarse_relevance.dtype)

        cold_weight = self._expand_grid_values(
            coarse_cold_weight,
            source_grid=self.gate_grid,
            target_grid=grid_shape,
        )
        hot_weight = self._expand_grid_values(
            coarse_hot_weight,
            source_grid=self.gate_grid,
            target_grid=grid_shape,
        )
        relevance = self._expand_grid_values(
            coarse_relevance,
            source_grid=self.gate_grid,
            target_grid=grid_shape,
        )
        temperature_logits = self._expand_grid_values(
            coarse_temperature_logits,
            source_grid=self.gate_grid,
            target_grid=grid_shape,
        )
        relevance_logits = self._expand_grid_values(
            coarse_relevance_logits,
            source_grid=self.gate_grid,
            target_grid=grid_shape,
        )

        fused_token_mask = cold_token_mask | hot_token_mask
        relevance = relevance * fused_token_mask.to(dtype=relevance.dtype)
        cold_stream = cold_in.masked_fill(~cold_token_mask[:, :, None], 0.0)
        hot_stream = hot_in.masked_fill(~hot_token_mask[:, :, None], 0.0)
        fused = relevance[:, :, None] * (
            cold_weight[:, :, None] * cold_stream
            + hot_weight[:, :, None] * hot_stream
        )
        fused = fused.masked_fill(~fused_token_mask[:, :, None], 0.0)

        gate_info = {
            "cold_alpha": cold_weight,
            "hot_beta": hot_weight,
            "cold_weight": cold_weight,
            "hot_weight": hot_weight,
            "patch_relevance": relevance,
            "temperature_logits": temperature_logits,
            "relevance_logits": relevance_logits,
            "global_text_logits": global_text_logits,
            "cold_evidence": cold_evidence,
            "hot_evidence": hot_evidence,
            "small_patch_cold_weight": coarse_cold_weight,
            "small_patch_hot_weight": coarse_hot_weight,
            "small_patch_relevance": coarse_relevance,
            "small_patch_temperature_logits": coarse_temperature_logits,
            "small_patch_relevance_logits": coarse_relevance_logits,
            "small_patch_cold_evidence": coarse_cold_evidence,
            "small_patch_hot_evidence": coarse_hot_evidence,
            "cold_valid": cold_token_mask.any(dim=1),
            "hot_valid": hot_token_mask.any(dim=1),
            "text_valid": text_valid,
            "rgb_valid": rgb_valid,
        }
        return fused.to(dtype=cold_tokens.dtype), fused_token_mask, gate_info


class TwoGreyHardSmallPatchGateViTFusion(TwoGreySmallPatchGateViTFusion):
    """Straight-through hard cold/hot routing on a fixed 4x4 gate grid.

    The forward value at every thermal patch is taken from exactly one of the
    aligned cold/hot tokens. Backpropagation follows the corresponding soft
    mixture so the router and both token branches receive gradients in a single
    training stage. Raw thermal evidence and the relevance head do not affect
    this routing mode.
    """

    def __init__(self, **kwargs):
        requested_grid = tuple(kwargs.pop("gate_grid", (4, 4)))
        if requested_grid != (4, 4):
            raise ValueError(
                "HardSmallPatchGateViT requires gate_grid=(4, 4), "
                f"got {requested_grid}."
            )
        # HardSmallPatchGateViT intentionally learns routing only from the
        # token/text/RGB gate path. Equal zero evidence priors cancel between
        # cold and hot, but setting the strength to zero also makes that intent
        # explicit in checkpoints and logs.
        kwargs["evidence_strength"] = 0.0
        super().__init__(gate_grid=(4, 4), **kwargs)

    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        text_tokens: Tensor,
        rgb_token_mask: Tensor,
        cold_token_mask: Tensor,
        hot_token_mask: Tensor,
        text_token_mask: Tensor,
        cold_evidence: Tensor | None = None,
        hot_evidence: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        # Reuse the tested coarse gate to obtain differentiable cold/hot
        # probabilities, while deliberately omitting raw thermal evidence.
        _, fused_token_mask, gate_info = super().forward(
            rgb_tokens=rgb_tokens,
            cold_tokens=cold_tokens,
            hot_tokens=hot_tokens,
            text_tokens=text_tokens,
            rgb_token_mask=rgb_token_mask,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
            cold_evidence=None,
            hot_evidence=None,
        )

        soft_cold_weight = gate_info["cold_weight"]
        soft_hot_weight = gate_info["hot_weight"]
        soft_weights = torch.stack([soft_cold_weight, soft_hot_weight], dim=-1)
        available = torch.stack(
            [
                cold_token_mask.to(device=soft_weights.device, dtype=torch.bool),
                hot_token_mask.to(device=soft_weights.device, dtype=torch.bool),
            ],
            dim=-1,
        )
        hard_indices = soft_weights.argmax(dim=-1)
        hard_weights = F.one_hot(hard_indices, num_classes=2).to(dtype=soft_weights.dtype)
        hard_weights = hard_weights * available.to(dtype=hard_weights.dtype)
        hard_cold_weight = hard_weights[..., 0]
        hard_hot_weight = hard_weights[..., 1]

        route_dtype = soft_weights.dtype
        cold_stream = cold_tokens.to(dtype=route_dtype).masked_fill(
            ~available[..., 0, None], 0.0
        )
        hot_stream = hot_tokens.to(device=cold_tokens.device, dtype=route_dtype).masked_fill(
            ~available[..., 1, None], 0.0
        )
        hard_fused = (
            hard_cold_weight[..., None] * cold_stream
            + hard_hot_weight[..., None] * hot_stream
        )
        soft_fused = (
            soft_cold_weight[..., None] * cold_stream
            + soft_hot_weight[..., None] * hot_stream
        )
        # Forward: hard_fused. Backward: soft_fused. Applying the straight-
        # through surrogate at the fused-token level also sends gradients to
        # an unselected cold/hot projector according to its soft probability.
        fused = hard_fused.detach() + (soft_fused - soft_fused.detach())
        fused = fused.masked_fill(~fused_token_mask[..., None], 0.0)

        coarse_soft_cold = gate_info["small_patch_cold_weight"]
        coarse_soft_hot = gate_info["small_patch_hot_weight"]
        coarse_soft_weights = torch.stack([coarse_soft_cold, coarse_soft_hot], dim=-1)
        coarse_hard_weights = F.one_hot(
            coarse_soft_weights.argmax(dim=-1), num_classes=2
        ).to(dtype=coarse_soft_weights.dtype)
        coarse_available = (coarse_soft_weights.sum(dim=-1, keepdim=True) > 0).to(
            dtype=coarse_hard_weights.dtype
        )
        coarse_hard_weights = coarse_hard_weights * coarse_available

        valid_strength = fused_token_mask.to(dtype=soft_weights.dtype)
        gate_info = dict(gate_info)
        gate_info.update(
            {
                "cold_alpha": hard_cold_weight,
                "hot_beta": hard_hot_weight,
                "cold_weight": hard_cold_weight,
                "hot_weight": hard_hot_weight,
                "hard_cold_weight": hard_cold_weight,
                "hard_hot_weight": hard_hot_weight,
                "soft_cold_weight": soft_cold_weight,
                "soft_hot_weight": soft_hot_weight,
                "patch_relevance": valid_strength,
                "routing_margin": (soft_cold_weight - soft_hot_weight).abs(),
                "small_patch_cold_weight": coarse_hard_weights[..., 0],
                "small_patch_hot_weight": coarse_hard_weights[..., 1],
                "small_patch_hard_cold_weight": coarse_hard_weights[..., 0],
                "small_patch_hard_hot_weight": coarse_hard_weights[..., 1],
                "small_patch_soft_cold_weight": coarse_soft_cold,
                "small_patch_soft_hot_weight": coarse_soft_hot,
                "small_patch_relevance": (coarse_soft_weights.sum(dim=-1) > 0).to(
                    dtype=soft_weights.dtype
                ),
            }
        )
        return fused.to(dtype=cold_tokens.dtype), fused_token_mask, gate_info


class TwoGreySmallPatchGateTwoResViTFusion(TwoGreySmallPatchGateViTFusion):
    """Coarse patch-gated cold/hot residuals emitted as two RGB-like streams."""

    @_disable_torch_compile
    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        text_tokens: Tensor,
        rgb_token_mask: Tensor,
        cold_token_mask: Tensor,
        hot_token_mask: Tensor,
        text_token_mask: Tensor,
        cold_evidence: Tensor | None = None,
        hot_evidence: Tensor | None = None,
        base_cold_alpha: float = 1.0,
        base_hot_beta: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        if rgb_tokens.shape != cold_tokens.shape or rgb_tokens.shape != hot_tokens.shape:
            raise ValueError(
                "SmallPatchGateTwoResViT requires aligned projected head RGB, cold, and hot "
                "token grids, got "
                f"rgb={tuple(rgb_tokens.shape)}, cold={tuple(cold_tokens.shape)}, "
                f"hot={tuple(hot_tokens.shape)}."
            )

        _, _, gate_info = super().forward(
            rgb_tokens=rgb_tokens,
            cold_tokens=cold_tokens,
            hot_tokens=hot_tokens,
            text_tokens=text_tokens,
            rgb_token_mask=rgb_token_mask,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
            cold_evidence=cold_evidence,
            hot_evidence=hot_evidence,
        )

        module_dtype = self.image_norm.weight.dtype
        rgb_token_mask = rgb_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        cold_token_mask = cold_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        hot_token_mask = hot_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)

        rgb_in = rgb_tokens.to(dtype=module_dtype)
        cold_in = cold_tokens.to(device=rgb_tokens.device, dtype=module_dtype)
        hot_in = hot_tokens.to(device=rgb_tokens.device, dtype=module_dtype)
        cold_weight = gate_info["cold_weight"].to(device=rgb_tokens.device, dtype=module_dtype)
        hot_weight = gate_info["hot_weight"].to(device=rgb_tokens.device, dtype=module_dtype)
        relevance = gate_info["patch_relevance"].to(device=rgb_tokens.device, dtype=module_dtype)

        cold_scale = float(base_cold_alpha) * cold_weight * relevance
        hot_scale = float(base_hot_beta) * hot_weight * relevance
        cold_output_mask = rgb_token_mask & cold_token_mask
        hot_output_mask = rgb_token_mask & hot_token_mask

        cold_residual = cold_in.masked_fill(~cold_token_mask[:, :, None], 0.0)
        hot_residual = hot_in.masked_fill(~hot_token_mask[:, :, None], 0.0)
        cold_out = rgb_in + cold_scale[:, :, None] * cold_residual
        hot_out = rgb_in + hot_scale[:, :, None] * hot_residual
        cold_out = cold_out.masked_fill(~cold_output_mask[:, :, None], 0.0)
        hot_out = hot_out.masked_fill(~hot_output_mask[:, :, None], 0.0)

        gate_info = dict(gate_info)
        gate_info.update(
            {
                "cold_alpha": cold_scale,
                "hot_beta": hot_scale,
                "cold_residual_scale": cold_scale,
                "hot_residual_scale": hot_scale,
                "cold_residual_valid": cold_output_mask.any(dim=1),
                "hot_residual_valid": hot_output_mask.any(dim=1),
            }
        )
        return (
            cold_out.to(dtype=rgb_tokens.dtype),
            cold_output_mask,
            hot_out.to(dtype=rgb_tokens.dtype),
            hot_output_mask,
            gate_info,
        )


class TwoGreySmallPatchGateResViTFusion(TwoGreySmallPatchGateViTFusion):
    """Coarse patch-gated cold/hot residuals fused back into one head-RGB stream."""

    def _apply_single_rgb_residual(
        self,
        *,
        rgb_tokens: Tensor,
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        rgb_token_mask: Tensor,
        cold_token_mask: Tensor,
        hot_token_mask: Tensor,
        gate_info: dict[str, Tensor],
        base_cold_alpha: float,
        base_hot_beta: float,
        mode_name: str,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if rgb_tokens.shape != cold_tokens.shape or rgb_tokens.shape != hot_tokens.shape:
            raise ValueError(
                f"{mode_name} requires aligned projected head RGB, cold, and hot token grids, "
                f"got rgb={tuple(rgb_tokens.shape)}, cold={tuple(cold_tokens.shape)}, "
                f"hot={tuple(hot_tokens.shape)}."
            )

        module_dtype = self.image_norm.weight.dtype
        rgb_token_mask = rgb_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        cold_token_mask = cold_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        hot_token_mask = hot_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)

        rgb_in = rgb_tokens.to(dtype=module_dtype)
        cold_in = cold_tokens.to(device=rgb_tokens.device, dtype=module_dtype)
        hot_in = hot_tokens.to(device=rgb_tokens.device, dtype=module_dtype)
        cold_weight = gate_info["cold_weight"].to(device=rgb_tokens.device, dtype=module_dtype)
        hot_weight = gate_info["hot_weight"].to(device=rgb_tokens.device, dtype=module_dtype)
        relevance = gate_info["patch_relevance"].to(device=rgb_tokens.device, dtype=module_dtype)

        cold_scale = float(base_cold_alpha) * cold_weight * relevance
        hot_scale = float(base_hot_beta) * hot_weight * relevance
        output_mask = rgb_token_mask

        cold_residual = cold_in.masked_fill(~cold_token_mask[:, :, None], 0.0)
        hot_residual = hot_in.masked_fill(~hot_token_mask[:, :, None], 0.0)
        fused = rgb_in + cold_scale[:, :, None] * cold_residual + hot_scale[:, :, None] * hot_residual
        fused = torch.where(output_mask[:, :, None], fused, rgb_in)

        gate_info = dict(gate_info)
        gate_info.update(
            {
                "cold_alpha": cold_scale,
                "hot_beta": hot_scale,
                "cold_residual_scale": cold_scale,
                "hot_residual_scale": hot_scale,
                "cold_residual_valid": (rgb_token_mask & cold_token_mask).any(dim=1),
                "hot_residual_valid": (rgb_token_mask & hot_token_mask).any(dim=1),
            }
        )
        return fused.to(dtype=rgb_tokens.dtype), output_mask, gate_info

    @_disable_torch_compile
    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        text_tokens: Tensor,
        rgb_token_mask: Tensor,
        cold_token_mask: Tensor,
        hot_token_mask: Tensor,
        text_token_mask: Tensor,
        cold_evidence: Tensor | None = None,
        hot_evidence: Tensor | None = None,
        base_cold_alpha: float = 1.0,
        base_hot_beta: float = 1.0,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        _, _, gate_info = super().forward(
            rgb_tokens=rgb_tokens,
            cold_tokens=cold_tokens,
            hot_tokens=hot_tokens,
            text_tokens=text_tokens,
            rgb_token_mask=rgb_token_mask,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
            cold_evidence=cold_evidence,
            hot_evidence=hot_evidence,
        )
        return self._apply_single_rgb_residual(
            rgb_tokens=rgb_tokens,
            cold_tokens=cold_tokens,
            hot_tokens=hot_tokens,
            rgb_token_mask=rgb_token_mask,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            gate_info=gate_info,
            base_cold_alpha=base_cold_alpha,
            base_hot_beta=base_hot_beta,
            mode_name="SmallPatchGateResViT",
        )


class TwoGreySmallPatchSingleGateResViTFusion(TwoGreySmallPatchGateResViTFusion):
    """SmallPatchGateResViT variant whose patch gate only cross-attends to text."""

    def __init__(self, **kwargs):
        gate_init_std_value = kwargs.get("gate_init_std", 0.02)
        gate_init_std = 0.02 if gate_init_std_value is None else float(gate_init_std_value)
        super().__init__(**kwargs)
        self.rgb_cross_attn = None
        gate_dim = self.query_norm.normalized_shape[0]
        hidden_dim = int(self.patch_gate_mlp[0].out_features)
        second_hidden_dim = int(self.patch_gate_mlp[3].out_features)
        dropout = 0.0
        for module in self.patch_gate_mlp:
            if isinstance(module, nn.Dropout):
                dropout = float(module.p)
                break
        gate_input_dim = gate_dim * 5
        self.gate_input_norm = nn.LayerNorm(gate_input_dim)
        self.patch_gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden_dim, 3),
        )
        nn.init.normal_(self.patch_gate_mlp[-1].weight, mean=0.0, std=gate_init_std)
        nn.init.zeros_(self.patch_gate_mlp[-1].bias)

    @_disable_torch_compile
    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        text_tokens: Tensor,
        rgb_token_mask: Tensor,
        cold_token_mask: Tensor,
        hot_token_mask: Tensor,
        text_token_mask: Tensor,
        cold_evidence: Tensor | None = None,
        hot_evidence: Tensor | None = None,
        base_cold_alpha: float = 1.0,
        base_hot_beta: float = 1.0,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        self._validate_tokens("rgb_tokens", rgb_tokens, self.embed_dim)
        self._validate_tokens("cold_tokens", cold_tokens, self.embed_dim)
        self._validate_tokens("hot_tokens", hot_tokens, self.embed_dim)
        self._validate_tokens("text_tokens", text_tokens, self.embed_dim)
        if rgb_tokens.shape != cold_tokens.shape or rgb_tokens.shape != hot_tokens.shape:
            raise ValueError(
                "SmallPatchSingleGateResViT requires aligned projected head RGB, cold, and hot "
                "token grids, got "
                f"rgb={tuple(rgb_tokens.shape)}, cold={tuple(cold_tokens.shape)}, "
                f"hot={tuple(hot_tokens.shape)}."
            )
        if text_tokens.shape[0] != cold_tokens.shape[0]:
            raise ValueError("SmallPatchSingleGateResViT thermal and text batch sizes must match.")

        batch_size, token_count = cold_tokens.shape[:2]
        grid_shape = self._infer_spatial_grid(token_count, self.gate_grid)
        if grid_shape[0] * grid_shape[1] != token_count:
            raise ValueError(
                "SmallPatchSingleGateResViT could not map thermal tokens to a 2D grid: "
                f"token_count={token_count}, inferred_grid={grid_shape}."
            )

        rgb_token_mask = self._validate_mask(
            "rgb_token_mask",
            rgb_token_mask.to(device=cold_tokens.device),
            tuple(rgb_tokens.shape[:2]),
        )
        cold_token_mask = self._validate_mask(
            "cold_token_mask",
            cold_token_mask.to(device=cold_tokens.device),
            (batch_size, token_count),
        )
        hot_token_mask = self._validate_mask(
            "hot_token_mask",
            hot_token_mask.to(device=cold_tokens.device),
            (batch_size, token_count),
        )
        text_token_mask = self._validate_mask(
            "text_token_mask",
            text_token_mask.to(device=cold_tokens.device),
            tuple(text_tokens.shape[:2]),
        )

        module_dtype = self.image_norm.weight.dtype
        cold_in = cold_tokens.to(dtype=module_dtype)
        hot_in = hot_tokens.to(device=cold_tokens.device, dtype=module_dtype)
        text_in = text_tokens.to(device=cold_tokens.device, dtype=module_dtype)

        cold_small = self.image_adapter(self.image_norm(cold_in))
        hot_small = self.image_adapter(self.image_norm(hot_in))
        text_small = self.text_adapter(self.text_norm(text_in))

        coarse_cold = self._pool_token_grid(cold_small, grid_shape, self.gate_grid)
        coarse_hot = self._pool_token_grid(hot_small, grid_shape, self.gate_grid)
        coarse_cold_mask = self._pool_mask_grid(cold_token_mask, grid_shape, self.gate_grid)
        coarse_hot_mask = self._pool_mask_grid(hot_token_mask, grid_shape, self.gate_grid)
        coarse_query = self.query_norm(0.5 * (coarse_cold + coarse_hot))
        text_context, text_valid = self._safe_context(
            self.text_cross_attn,
            query=coarse_query,
            context=text_small,
            context_mask=text_token_mask,
        )

        gate_input = torch.cat(
            [
                coarse_cold,
                coarse_hot,
                coarse_cold - coarse_hot,
                coarse_cold * coarse_hot,
                text_context,
            ],
            dim=-1,
        )
        coarse_patch_logits = self.patch_gate_mlp(self.gate_input_norm(gate_input))
        text_pool = self._masked_mean(text_small, text_token_mask)
        global_text_logits = self.global_text_gate(text_pool)
        global_text_logits = global_text_logits * text_valid[:, None].to(
            dtype=global_text_logits.dtype
        )
        coarse_temperature_logits = coarse_patch_logits[:, :, :2] + global_text_logits[:, None, :]

        evidence_shape = (batch_size, token_count)
        if cold_evidence is None:
            cold_evidence = coarse_temperature_logits.new_zeros(evidence_shape)
        else:
            cold_evidence = cold_evidence.to(
                device=coarse_temperature_logits.device,
                dtype=coarse_temperature_logits.dtype,
            )
        if hot_evidence is None:
            hot_evidence = coarse_temperature_logits.new_zeros(evidence_shape)
        else:
            hot_evidence = hot_evidence.to(
                device=coarse_temperature_logits.device,
                dtype=coarse_temperature_logits.dtype,
            )
        if cold_evidence.shape != evidence_shape or hot_evidence.shape != evidence_shape:
            raise ValueError(
                "SmallPatchSingleGateResViT evidence must match thermal tokens [B, N], "
                f"got cold={tuple(cold_evidence.shape)}, hot={tuple(hot_evidence.shape)}, "
                f"expected={evidence_shape}."
            )
        cold_evidence = cold_evidence.clamp(0.0, 1.0)
        hot_evidence = hot_evidence.clamp(0.0, 1.0)
        coarse_cold_evidence = self._pool_evidence_grid(
            cold_evidence, cold_token_mask, grid_shape, self.gate_grid
        )
        coarse_hot_evidence = self._pool_evidence_grid(
            hot_evidence, hot_token_mask, grid_shape, self.gate_grid
        )
        coarse_evidence = torch.stack([coarse_cold_evidence, coarse_hot_evidence], dim=-1)
        evidence_prior = torch.log(coarse_evidence + self.evidence_floor)
        coarse_temperature_logits = coarse_temperature_logits + self.evidence_strength * evidence_prior

        coarse_available = torch.stack([coarse_cold_mask, coarse_hot_mask], dim=-1)
        masked_logits = coarse_temperature_logits.masked_fill(~coarse_available, -1e4)
        coarse_temperature_weights = torch.softmax(masked_logits / self.temperature, dim=-1)
        coarse_temperature_weights = coarse_temperature_weights * coarse_available.to(
            dtype=coarse_temperature_weights.dtype
        )
        coarse_temperature_weights = coarse_temperature_weights / coarse_temperature_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        coarse_cold_weight = coarse_temperature_weights[:, :, 0]
        coarse_hot_weight = coarse_temperature_weights[:, :, 1]
        coarse_relevance_logits = coarse_patch_logits[:, :, 2]
        coarse_relevance = 2.0 * torch.sigmoid(coarse_relevance_logits)
        coarse_fused_mask = coarse_cold_mask | coarse_hot_mask
        coarse_relevance = coarse_relevance * coarse_fused_mask.to(dtype=coarse_relevance.dtype)

        cold_weight = self._expand_grid_values(
            coarse_cold_weight,
            source_grid=self.gate_grid,
            target_grid=grid_shape,
        )
        hot_weight = self._expand_grid_values(
            coarse_hot_weight,
            source_grid=self.gate_grid,
            target_grid=grid_shape,
        )
        relevance = self._expand_grid_values(
            coarse_relevance,
            source_grid=self.gate_grid,
            target_grid=grid_shape,
        )
        temperature_logits = self._expand_grid_values(
            coarse_temperature_logits,
            source_grid=self.gate_grid,
            target_grid=grid_shape,
        )
        relevance_logits = self._expand_grid_values(
            coarse_relevance_logits,
            source_grid=self.gate_grid,
            target_grid=grid_shape,
        )

        fused_token_mask = cold_token_mask | hot_token_mask
        relevance = relevance * fused_token_mask.to(dtype=relevance.dtype)
        gate_info = {
            "cold_alpha": cold_weight,
            "hot_beta": hot_weight,
            "cold_weight": cold_weight,
            "hot_weight": hot_weight,
            "patch_relevance": relevance,
            "temperature_logits": temperature_logits,
            "relevance_logits": relevance_logits,
            "global_text_logits": global_text_logits,
            "cold_evidence": cold_evidence,
            "hot_evidence": hot_evidence,
            "small_patch_cold_weight": coarse_cold_weight,
            "small_patch_hot_weight": coarse_hot_weight,
            "small_patch_relevance": coarse_relevance,
            "small_patch_temperature_logits": coarse_temperature_logits,
            "small_patch_relevance_logits": coarse_relevance_logits,
            "small_patch_cold_evidence": coarse_cold_evidence,
            "small_patch_hot_evidence": coarse_hot_evidence,
            "cold_valid": cold_token_mask.any(dim=1),
            "hot_valid": hot_token_mask.any(dim=1),
            "text_valid": text_valid,
            "rgb_valid": rgb_token_mask.any(dim=1),
        }
        return self._apply_single_rgb_residual(
            rgb_tokens=rgb_tokens,
            cold_tokens=cold_tokens,
            hot_tokens=hot_tokens,
            rgb_token_mask=rgb_token_mask,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            gate_info=gate_info,
            base_cold_alpha=base_cold_alpha,
            base_hot_beta=base_hot_beta,
            mode_name="SmallPatchSingleGateResViT",
        )


class TwoGreySmallPatchSingleGateTwoResViTFusion(TwoGreySmallPatchSingleGateResViTFusion):
    """Text-only coarse patch gates emitted as separate cold/hot RGB residual streams."""

    @_disable_torch_compile
    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        text_tokens: Tensor,
        rgb_token_mask: Tensor,
        cold_token_mask: Tensor,
        hot_token_mask: Tensor,
        text_token_mask: Tensor,
        cold_evidence: Tensor | None = None,
        hot_evidence: Tensor | None = None,
        base_cold_alpha: float = 1.0,
        base_hot_beta: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        _, _, gate_info = super().forward(
            rgb_tokens=rgb_tokens,
            cold_tokens=cold_tokens,
            hot_tokens=hot_tokens,
            text_tokens=text_tokens,
            rgb_token_mask=rgb_token_mask,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
            cold_evidence=cold_evidence,
            hot_evidence=hot_evidence,
            base_cold_alpha=base_cold_alpha,
            base_hot_beta=base_hot_beta,
        )

        module_dtype = self.image_norm.weight.dtype
        rgb_token_mask = rgb_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        cold_token_mask = cold_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        hot_token_mask = hot_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)

        rgb_in = rgb_tokens.to(dtype=module_dtype)
        cold_in = cold_tokens.to(device=rgb_tokens.device, dtype=module_dtype)
        hot_in = hot_tokens.to(device=rgb_tokens.device, dtype=module_dtype)
        cold_scale = gate_info["cold_residual_scale"].to(
            device=rgb_tokens.device,
            dtype=module_dtype,
        )
        hot_scale = gate_info["hot_residual_scale"].to(
            device=rgb_tokens.device,
            dtype=module_dtype,
        )
        cold_output_mask = rgb_token_mask & cold_token_mask
        hot_output_mask = rgb_token_mask & hot_token_mask

        cold_residual = cold_in.masked_fill(~cold_token_mask[:, :, None], 0.0)
        hot_residual = hot_in.masked_fill(~hot_token_mask[:, :, None], 0.0)
        cold_out = rgb_in + cold_scale[:, :, None] * cold_residual
        hot_out = rgb_in + hot_scale[:, :, None] * hot_residual
        cold_out = cold_out.masked_fill(~cold_output_mask[:, :, None], 0.0)
        hot_out = hot_out.masked_fill(~hot_output_mask[:, :, None], 0.0)

        gate_info = dict(gate_info)
        gate_info.update(
            {
                "cold_residual_valid": cold_output_mask.any(dim=1),
                "hot_residual_valid": hot_output_mask.any(dim=1),
            }
        )
        return (
            cold_out.to(dtype=rgb_tokens.dtype),
            cold_output_mask,
            hot_out.to(dtype=rgb_tokens.dtype),
            hot_output_mask,
            gate_info,
        )


class TwoGreyGateActionViTFusion(TwoGreyGateResViTFusion):
    """Keep cold/hot streams separate and gate only action attention to them."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int = 8,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        temperature: float = 1.0,
        gate_init_std: float = 0.02,
        bias_epsilon: float = 0.1,
        bias_max_strength: float = 1.0,
        bias_init_strength: float = 0.25,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            temperature=temperature,
            gate_init_std=gate_init_std,
        )
        if not 0 <= bias_epsilon < 1:
            raise ValueError("bias_epsilon must be in [0, 1)")
        if not math.isfinite(bias_max_strength) or bias_max_strength <= 0:
            raise ValueError("bias_max_strength must be finite and positive")
        if not 0 <= bias_init_strength <= bias_max_strength:
            raise ValueError("bias_init_strength must be between 0 and bias_max_strength")

        self.bias_epsilon = float(bias_epsilon)
        self.bias_max_strength = float(bias_max_strength)
        init_ratio = float(bias_init_strength) / self.bias_max_strength
        init_ratio = min(max(init_ratio, 1e-6), 1.0 - 1e-6)
        self.raw_bias_strength = nn.Parameter(
            torch.tensor(math.log(init_ratio / (1.0 - init_ratio)), dtype=torch.float32)
        )

    def forward(
        self,
        *,
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        text_tokens: Tensor,
        cold_token_mask: Tensor,
        hot_token_mask: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        gate_state = self._compute_text_gate(
            cold_tokens=cold_tokens,
            hot_tokens=hot_tokens,
            text_tokens=text_tokens,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
            mode_name="GateActionViT",
        )
        return self._build_action_attention_output(
            gate_state=gate_state,
            cold_tokens=cold_tokens,
            hot_tokens=hot_tokens,
        )

    def _build_action_attention_output(
        self,
        *,
        gate_state: dict[str, Tensor],
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        extra_gate_info: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        cold_token_mask = gate_state["cold_token_mask"]
        hot_token_mask = gate_state["hot_token_mask"]
        text_valid = gate_state["text_valid"]
        cold_valid = gate_state["cold_valid"]
        hot_valid = gate_state["hot_valid"]
        logits = gate_state["logits"]
        multipliers = gate_state["multipliers"]

        raw_weights = multipliers * 0.5
        available = torch.stack([cold_valid, hot_valid], dim=-1)
        masked_weights = raw_weights * available.to(dtype=raw_weights.dtype)
        available_count = available.sum(dim=-1, keepdim=True)
        fallback_weights = available.to(dtype=raw_weights.dtype) / available_count.clamp_min(1)
        fallback_weights = torch.where(
            available_count > 0,
            fallback_weights,
            torch.full_like(fallback_weights, 0.5),
        )
        normalized_weights = masked_weights / masked_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        use_text_gate = text_valid[:, None] & (available_count > 0)
        weights = torch.where(use_text_gate, normalized_weights, fallback_weights)

        # A centered log-prior changes only the cold/hot preference. Equal gates
        # produce exactly zero bias and preserve the original ViT+twogrey path.
        safe_weights = (
            self.bias_epsilon * 0.5 + (1.0 - self.bias_epsilon) * weights
        ).clamp_min(1e-6)
        centered_log_prior = torch.log(safe_weights)
        centered_log_prior = centered_log_prior - centered_log_prior.mean(
            dim=-1, keepdim=True
        )
        bias_strength = self.bias_max_strength * torch.sigmoid(self.raw_bias_strength)
        pair_valid = text_valid & cold_valid & hot_valid
        attention_bias = (
            bias_strength * centered_log_prior * pair_valid[:, None].to(centered_log_prior.dtype)
        )

        cold_weight = weights[:, 0]
        hot_weight = weights[:, 1]
        gate_info = {
            "cold_alpha": cold_weight.detach(),
            "hot_beta": hot_weight.detach(),
            "cold_weight": cold_weight.detach(),
            "hot_weight": hot_weight.detach(),
            "cold_multiplier": multipliers[:, 0].detach(),
            "hot_multiplier": multipliers[:, 1].detach(),
            "cold_attention_bias": attention_bias[:, 0].detach(),
            "hot_attention_bias": attention_bias[:, 1].detach(),
            "attention_bias_strength": bias_strength.expand_as(cold_weight).detach(),
            "logits": logits.detach(),
            "cold_valid": cold_valid.detach(),
            "hot_valid": hot_valid.detach(),
            "text_valid": text_valid.detach(),
        }
        if extra_gate_info:
            gate_info.update(extra_gate_info)
        return (
            cold_tokens,
            cold_token_mask,
            hot_tokens,
            hot_token_mask,
            attention_bias,
            gate_info,
        )


class TwoGreyDoubleGateViTFusion(TwoGreyGateViTFusion):
    """Merge cold/hot using independent text and head-RGB alignment gates."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int = 8,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        temperature: float = 1.0,
        gate_init_std: float = 0.02,
        rgb_gate_max_strength: float = 1.0,
        rgb_gate_init_strength: float = 0.5,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            temperature=temperature,
            gate_init_std=gate_init_std,
        )
        if not math.isfinite(rgb_gate_max_strength) or rgb_gate_max_strength <= 0:
            raise ValueError("rgb_gate_max_strength must be finite and positive")
        if not 0 <= rgb_gate_init_strength <= rgb_gate_max_strength:
            raise ValueError(
                "rgb_gate_init_strength must be between 0 and rgb_gate_max_strength"
            )
        if hidden_dim is None:
            hidden_dim = max(128, embed_dim // 4)

        self.rgb_gate_max_strength = float(rgb_gate_max_strength)
        rgb_init_ratio = float(rgb_gate_init_strength) / self.rgb_gate_max_strength
        rgb_init_ratio = min(max(rgb_init_ratio, 1e-6), 1.0 - 1e-6)
        self.raw_rgb_gate_strength = nn.Parameter(
            torch.tensor(
                math.log(rgb_init_ratio / (1.0 - rgb_init_ratio)),
                dtype=torch.float32,
            )
        )
        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.rgb_to_thermal_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        gate_input_dim = embed_dim * 5
        second_hidden_dim = max(64, hidden_dim // 2)
        self.rgb_gate_input_norm = nn.LayerNorm(gate_input_dim)
        self.rgb_gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden_dim, 2),
        )
        nn.init.normal_(self.rgb_gate_mlp[-1].weight, mean=0.0, std=gate_init_std)
        nn.init.zeros_(self.rgb_gate_mlp[-1].bias)

    def _compute_rgb_gate(
        self,
        *,
        rgb_tokens: Tensor,
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        rgb_token_mask: Tensor,
        cold_token_mask: Tensor,
        hot_token_mask: Tensor,
    ) -> dict[str, Tensor]:
        embed_dim = self.rgb_norm.normalized_shape[0]
        self._validate_token_shape("rgb_tokens", rgb_tokens, embed_dim)
        self._validate_token_shape("cold_tokens", cold_tokens, embed_dim)
        self._validate_token_shape("hot_tokens", hot_tokens, embed_dim)
        if rgb_tokens.shape[0] != cold_tokens.shape[0] or cold_tokens.shape != hot_tokens.shape:
            raise ValueError(
                "DoubleGateViT requires matching RGB/cold/hot batch sizes and aligned "
                "cold/hot token grids."
            )

        batch_size, num_rgb_tokens = rgb_tokens.shape[:2]
        num_thermal_tokens = cold_tokens.shape[1]
        rgb_token_mask = rgb_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        cold_token_mask = cold_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        hot_token_mask = hot_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        self._validate_mask_shape(
            "rgb_token_mask",
            rgb_token_mask,
            (batch_size, num_rgb_tokens),
        )
        self._validate_mask_shape(
            "cold_token_mask",
            cold_token_mask,
            (batch_size, num_thermal_tokens),
        )
        self._validate_mask_shape(
            "hot_token_mask",
            hot_token_mask,
            (batch_size, num_thermal_tokens),
        )

        module_dtype = self.rgb_norm.weight.dtype
        rgb_norm = self.rgb_norm(rgb_tokens.to(dtype=module_dtype))
        cold_norm = self.thermal_norm(cold_tokens.to(dtype=module_dtype))
        hot_norm = self.thermal_norm(hot_tokens.to(dtype=module_dtype))
        cold_context, cold_valid = ResViTAttentionFusion._safe_cross_attention(
            self.rgb_to_thermal_attn,
            query=rgb_norm,
            key=cold_norm,
            value=cold_norm,
            key_mask=cold_token_mask,
        )
        hot_context, hot_valid = ResViTAttentionFusion._safe_cross_attention(
            self.rgb_to_thermal_attn,
            query=rgb_norm,
            key=hot_norm,
            value=hot_norm,
            key_mask=hot_token_mask,
        )

        rgb_valid = rgb_token_mask.any(dim=1)
        rgb_pool = self._masked_mean(rgb_norm, rgb_token_mask)
        cold_pool = self._masked_mean(cold_context, rgb_token_mask)
        hot_pool = self._masked_mean(hot_context, rgb_token_mask)
        gate_input = torch.cat(
            [
                rgb_pool,
                cold_pool,
                hot_pool,
                cold_pool - rgb_pool,
                hot_pool - rgb_pool,
            ],
            dim=-1,
        )
        logits = self.rgb_gate_mlp(self.rgb_gate_input_norm(gate_input))
        valid = rgb_valid & cold_valid & hot_valid
        return {
            "logits": logits,
            "valid_logits": logits * valid[:, None].to(dtype=logits.dtype),
            "rgb_valid": rgb_valid,
            "cold_valid": cold_valid,
            "hot_valid": hot_valid,
        }

    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        text_tokens: Tensor,
        rgb_token_mask: Tensor,
        cold_token_mask: Tensor,
        hot_token_mask: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        text_gate_state = self._compute_text_gate(
            cold_tokens=cold_tokens,
            hot_tokens=hot_tokens,
            text_tokens=text_tokens,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
            mode_name="DoubleGateViT",
        )
        rgb_gate_state = self._compute_rgb_gate(
            rgb_tokens=rgb_tokens,
            cold_tokens=cold_tokens,
            hot_tokens=hot_tokens,
            rgb_token_mask=rgb_token_mask,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
        )

        rgb_gate_strength = self.rgb_gate_max_strength * torch.sigmoid(
            self.raw_rgb_gate_strength
        )
        text_logits = text_gate_state["logits"]
        rgb_logits = rgb_gate_state["logits"]
        combined_logits = text_logits + rgb_gate_strength * rgb_gate_state["valid_logits"]
        text_gate_state["logits"] = combined_logits
        text_gate_state["multipliers"] = 2.0 * torch.softmax(
            combined_logits / self.temperature,
            dim=-1,
        )

        text_weights = torch.softmax(text_logits / self.temperature, dim=-1)
        rgb_weights = torch.softmax(rgb_logits / self.temperature, dim=-1)
        multipliers = text_gate_state["multipliers"]
        cold_token_mask = text_gate_state["cold_token_mask"]
        hot_token_mask = text_gate_state["hot_token_mask"]
        text_valid = text_gate_state["text_valid"]
        cold_valid = text_gate_state["cold_valid"]
        hot_valid = text_gate_state["hot_valid"]
        raw_weights = multipliers * 0.5
        available = torch.stack([cold_valid, hot_valid], dim=-1) & text_valid[:, None]
        weights = raw_weights * available.to(dtype=raw_weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        cold_weight = weights[:, 0]
        hot_weight = weights[:, 1]

        cold_stream = text_gate_state["cold_in"].masked_fill(
            ~cold_token_mask[:, :, None],
            0.0,
        )
        hot_stream = text_gate_state["hot_in"].masked_fill(
            ~hot_token_mask[:, :, None],
            0.0,
        )
        fused = (
            cold_weight[:, None, None] * cold_stream
            + hot_weight[:, None, None] * hot_stream
        )
        fused_token_mask = (
            (cold_token_mask & cold_valid[:, None])
            | (hot_token_mask & hot_valid[:, None])
        ) & text_valid[:, None]
        fused = fused.masked_fill(~fused_token_mask[:, :, None], 0.0)

        gate_info = {
            "cold_alpha": cold_weight.detach(),
            "hot_beta": hot_weight.detach(),
            "cold_weight": cold_weight.detach(),
            "hot_weight": hot_weight.detach(),
            "cold_multiplier": multipliers[:, 0].detach(),
            "hot_multiplier": multipliers[:, 1].detach(),
            "logits": combined_logits.detach(),
            "text_logits": text_logits.detach(),
            "rgb_logits": rgb_logits.detach(),
            "text_cold_weight": text_weights[:, 0].detach(),
            "text_hot_weight": text_weights[:, 1].detach(),
            "rgb_cold_alignment_weight": rgb_weights[:, 0].detach(),
            "rgb_hot_alignment_weight": rgb_weights[:, 1].detach(),
            "rgb_gate_strength": rgb_gate_strength.expand(text_logits.shape[0]).detach(),
            "rgb_valid": rgb_gate_state["rgb_valid"].detach(),
            "rgb_cold_valid": rgb_gate_state["cold_valid"].detach(),
            "rgb_hot_valid": rgb_gate_state["hot_valid"].detach(),
            "cold_valid": cold_valid.detach(),
            "hot_valid": hot_valid.detach(),
            "text_valid": text_valid.detach(),
        }
        return fused.to(dtype=cold_tokens.dtype), fused_token_mask, gate_info


class TwoGreyDoubleGateTwoResViTFusion(TwoGreyDoubleGateViTFusion):
    """Text/RGB-conditioned scalar gates emitted as separate RGB residual streams."""

    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        text_tokens: Tensor,
        rgb_token_mask: Tensor,
        cold_token_mask: Tensor,
        hot_token_mask: Tensor,
        text_token_mask: Tensor,
        base_cold_alpha: float = 1.0,
        base_hot_beta: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        embed_dim = self.text_norm.normalized_shape[0]
        self._validate_token_shape("rgb_tokens", rgb_tokens, embed_dim)
        if rgb_tokens.shape != cold_tokens.shape or rgb_tokens.shape != hot_tokens.shape:
            raise ValueError(
                "DoubleGateTwoResViT requires aligned RGB, cold thermal, and hot thermal "
                "projected token grids, got "
                f"rgb={tuple(rgb_tokens.shape)}, cold={tuple(cold_tokens.shape)}, "
                f"hot={tuple(hot_tokens.shape)}."
            )

        batch_size, num_rgb_tokens = rgb_tokens.shape[:2]
        rgb_token_mask = rgb_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        self._validate_mask_shape("rgb_token_mask", rgb_token_mask, (batch_size, num_rgb_tokens))
        text_gate_state = self._compute_text_gate(
            cold_tokens=cold_tokens,
            hot_tokens=hot_tokens,
            text_tokens=text_tokens,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
            mode_name="DoubleGateTwoResViT",
        )
        rgb_gate_state = self._compute_rgb_gate(
            rgb_tokens=rgb_tokens,
            cold_tokens=cold_tokens,
            hot_tokens=hot_tokens,
            rgb_token_mask=rgb_token_mask,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
        )

        rgb_gate_strength = self.rgb_gate_max_strength * torch.sigmoid(
            self.raw_rgb_gate_strength
        )
        text_logits = text_gate_state["logits"]
        rgb_logits = rgb_gate_state["logits"]
        combined_logits = text_logits + rgb_gate_strength * rgb_gate_state["valid_logits"]
        multipliers = 2.0 * torch.softmax(combined_logits / self.temperature, dim=-1)
        text_weights = torch.softmax(text_logits / self.temperature, dim=-1)
        rgb_weights = torch.softmax(rgb_logits / self.temperature, dim=-1)
        combined_weights = torch.softmax(combined_logits / self.temperature, dim=-1)

        module_dtype = self.text_norm.weight.dtype
        rgb_in = rgb_tokens.to(dtype=module_dtype)
        cold_in = text_gate_state["cold_in"]
        hot_in = text_gate_state["hot_in"]
        cold_token_mask = text_gate_state["cold_token_mask"]
        hot_token_mask = text_gate_state["hot_token_mask"]
        text_valid = text_gate_state["text_valid"]
        cold_valid = text_gate_state["cold_valid"] & rgb_gate_state["cold_valid"]
        hot_valid = text_gate_state["hot_valid"] & rgb_gate_state["hot_valid"]
        rgb_valid = rgb_gate_state["rgb_valid"]
        valid_gate = (text_valid & rgb_valid).to(dtype=module_dtype)
        cold_alpha = (
            float(base_cold_alpha)
            * multipliers[:, 0]
            * cold_valid.to(dtype=module_dtype)
            * valid_gate
        )
        hot_beta = (
            float(base_hot_beta)
            * multipliers[:, 1]
            * hot_valid.to(dtype=module_dtype)
            * valid_gate
        )

        cold_output_mask = rgb_token_mask & cold_token_mask
        hot_output_mask = rgb_token_mask & hot_token_mask
        cold_residual = cold_in.masked_fill(~cold_token_mask[:, :, None], 0.0)
        hot_residual = hot_in.masked_fill(~hot_token_mask[:, :, None], 0.0)
        cold_out = rgb_in + cold_alpha[:, None, None] * cold_residual
        hot_out = rgb_in + hot_beta[:, None, None] * hot_residual
        cold_out = cold_out.masked_fill(~cold_output_mask[:, :, None], 0.0)
        hot_out = hot_out.masked_fill(~hot_output_mask[:, :, None], 0.0)

        gate_info = {
            "cold_alpha": cold_alpha.detach(),
            "hot_beta": hot_beta.detach(),
            "cold_weight": combined_weights[:, 0].detach(),
            "hot_weight": combined_weights[:, 1].detach(),
            "cold_multiplier": multipliers[:, 0].detach(),
            "hot_multiplier": multipliers[:, 1].detach(),
            "cold_residual_scale": cold_alpha.detach(),
            "hot_residual_scale": hot_beta.detach(),
            "logits": combined_logits.detach(),
            "text_logits": text_logits.detach(),
            "rgb_logits": rgb_logits.detach(),
            "text_cold_weight": text_weights[:, 0].detach(),
            "text_hot_weight": text_weights[:, 1].detach(),
            "rgb_cold_alignment_weight": rgb_weights[:, 0].detach(),
            "rgb_hot_alignment_weight": rgb_weights[:, 1].detach(),
            "rgb_gate_strength": rgb_gate_strength.expand(text_logits.shape[0]).detach(),
            "rgb_valid": rgb_valid.detach(),
            "rgb_cold_valid": rgb_gate_state["cold_valid"].detach(),
            "rgb_hot_valid": rgb_gate_state["hot_valid"].detach(),
            "cold_valid": cold_valid.detach(),
            "hot_valid": hot_valid.detach(),
            "text_valid": text_valid.detach(),
            "cold_residual_valid": cold_output_mask.any(dim=1).detach(),
            "hot_residual_valid": hot_output_mask.any(dim=1).detach(),
        }
        return (
            cold_out.to(dtype=rgb_tokens.dtype),
            cold_output_mask,
            hot_out.to(dtype=rgb_tokens.dtype),
            hot_output_mask,
            gate_info,
        )


class GateViTMixFusion(nn.Module):
    """Head-RGB-conditioned thermal stream with no thermal residual on RGB."""

    def __init__(
        self,
        *,
        embed_dim: int,
        mix_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        context_scale: float = 0.5,
        detach_head_rgb: bool = True,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if mix_dim <= 0:
            raise ValueError("mix_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if mix_dim % num_heads != 0:
            raise ValueError(f"num_heads must divide mix_dim ({mix_dim}), got {num_heads}.")
        if not math.isfinite(context_scale) or context_scale < 0:
            raise ValueError("context_scale must be finite and non-negative")

        self.context_scale = float(context_scale)
        self.detach_head_rgb = bool(detach_head_rgb)
        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.thermal_norm = nn.LayerNorm(embed_dim)
        self.rgb_to_mix = nn.Linear(embed_dim, mix_dim)
        self.thermal_to_mix = nn.Linear(embed_dim, mix_dim)
        self.thermal_from_rgb_attn = nn.MultiheadAttention(
            mix_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.rgb_from_thermal_attn = nn.MultiheadAttention(
            mix_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.thermal_mix_norm = nn.LayerNorm(mix_dim)
        self.context_norm = nn.LayerNorm(mix_dim * 2)
        self.context_projector = nn.Sequential(
            nn.Linear(mix_dim * 2, mix_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mix_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _validate_token_shape(name: str, tokens: Tensor, embed_dim: int) -> None:
        if tokens.ndim != 3 or tokens.shape[-1] != embed_dim:
            raise ValueError(
                f"{name} must have shape [B, N, {embed_dim}], got {tuple(tokens.shape)}."
            )

    @staticmethod
    def _validate_mask_shape(name: str, mask: Tensor, expected_shape: tuple[int, int]) -> None:
        if mask.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(mask.shape)}."
            )

    @staticmethod
    def _masked_rms(values: Tensor, mask: Tensor) -> Tensor:
        squared = values.to(dtype=torch.float32).square().mean(dim=-1)
        masked = squared.masked_fill(~mask, 0.0)
        denom = mask.sum(dim=1).clamp_min(1).to(dtype=masked.dtype)
        return torch.sqrt(masked.sum(dim=1) / denom)

    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        thermal_tokens: Tensor,
        rgb_token_mask: Tensor,
        thermal_token_mask: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        embed_dim = self.rgb_norm.normalized_shape[0]
        self._validate_token_shape("rgb_tokens", rgb_tokens, embed_dim)
        self._validate_token_shape("thermal_tokens", thermal_tokens, embed_dim)
        if rgb_tokens.shape != thermal_tokens.shape:
            raise ValueError(
                "GateViTMix requires aligned head RGB and thermal token grids, "
                f"got rgb={tuple(rgb_tokens.shape)} and thermal={tuple(thermal_tokens.shape)}."
            )

        batch_size, num_tokens = rgb_tokens.shape[:2]
        rgb_token_mask = rgb_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        thermal_token_mask = thermal_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        self._validate_mask_shape("rgb_token_mask", rgb_token_mask, (batch_size, num_tokens))
        self._validate_mask_shape(
            "thermal_token_mask",
            thermal_token_mask,
            (batch_size, num_tokens),
        )

        module_dtype = self.rgb_norm.weight.dtype
        rgb_source = rgb_tokens.detach() if self.detach_head_rgb else rgb_tokens
        rgb_in = rgb_source.to(dtype=module_dtype)
        thermal_in = thermal_tokens.to(device=rgb_tokens.device, dtype=module_dtype)
        rgb_mix = self.rgb_to_mix(self.rgb_norm(rgb_in))
        thermal_mix = self.thermal_to_mix(self.thermal_norm(thermal_in))

        thermal_from_rgb, rgb_valid = ResViTAttentionFusion._safe_cross_attention(
            self.thermal_from_rgb_attn,
            query=thermal_mix,
            key=rgb_mix,
            value=rgb_mix,
            key_mask=rgb_token_mask,
        )
        adjusted_thermal_mix = self.thermal_mix_norm(
            thermal_mix + self.dropout(thermal_from_rgb)
        )
        adjusted_thermal_mix = adjusted_thermal_mix.masked_fill(
            ~thermal_token_mask[:, :, None],
            0.0,
        )
        rgb_from_thermal, thermal_valid = ResViTAttentionFusion._safe_cross_attention(
            self.rgb_from_thermal_attn,
            query=rgb_mix,
            key=adjusted_thermal_mix,
            value=adjusted_thermal_mix,
            key_mask=thermal_token_mask,
        )

        context = self.context_projector(
            self.context_norm(torch.cat([thermal_from_rgb, rgb_from_thermal], dim=-1))
        )
        mix_mask = (
            rgb_token_mask
            & thermal_token_mask
            & rgb_valid[:, None]
            & thermal_valid[:, None]
        )
        context = context.masked_fill(~mix_mask[:, :, None], 0.0)
        mixed_candidate = (
            thermal_in
            + self.context_scale * self.dropout(context)
        )
        mixed_thermal = torch.where(
            mix_mask[:, :, None],
            mixed_candidate,
            thermal_in,
        )

        delta = mixed_thermal - thermal_in
        mix_info = {
            "head_mix_context_scale": torch.full(
                (batch_size,),
                self.context_scale,
                device=rgb_tokens.device,
                dtype=torch.float32,
            ),
            "head_mix_context_rms": self._masked_rms(context, mix_mask).detach(),
            "head_mix_delta_rms": self._masked_rms(delta, mix_mask).detach(),
            "head_mix_valid": mix_mask.any(dim=1).detach(),
        }
        return mixed_thermal.to(dtype=thermal_tokens.dtype), mix_info


class TwoGreyGateResViT3Fusion(TwoGreyGateResViTFusion):
    """Text-gated cold/hot residuals scaled by thermal-statistics confidence."""

    stat_dim = 16

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int = 8,
        hidden_dim: int | None = None,
        stat_hidden_dim: int = 64,
        residual_hidden_dim: int | None = None,
        dropout: float = 0.0,
        temperature: float = 1.0,
        gate_init_std: float = 0.02,
        confidence_bias_init: float = -3.0,
        evidence_gain: float = 48.0,
        evidence_threshold: float = 0.08,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            temperature=temperature,
            gate_init_std=gate_init_std,
        )
        if stat_hidden_dim <= 0:
            raise ValueError("stat_hidden_dim must be positive")
        if residual_hidden_dim is None:
            residual_hidden_dim = max(128, embed_dim // 4)
        if residual_hidden_dim <= 0:
            raise ValueError("residual_hidden_dim must be positive")
        if evidence_gain <= 0:
            raise ValueError("evidence_gain must be positive")
        if evidence_threshold < 0:
            raise ValueError("evidence_threshold must be non-negative")
        self.evidence_gain = float(evidence_gain)
        self.evidence_threshold = float(evidence_threshold)

        self.stats_norm = nn.LayerNorm(self.stat_dim)
        self.stats_mlp = nn.Sequential(
            nn.Linear(self.stat_dim, stat_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(stat_hidden_dim, max(16, stat_hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, stat_hidden_dim // 2), 1),
        )
        nn.init.zeros_(self.stats_mlp[-1].weight)
        nn.init.constant_(self.stats_mlp[-1].bias, float(confidence_bias_init))

        self.residual_norm = nn.LayerNorm(embed_dim)
        self.residual_adapter = nn.Sequential(
            nn.Linear(embed_dim, residual_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(residual_hidden_dim, embed_dim),
        )
        nn.init.zeros_(self.residual_adapter[-1].weight)
        nn.init.zeros_(self.residual_adapter[-1].bias)

    @staticmethod
    def _grey_zero_one(images: Tensor, *, input_range: str = "minus_one_to_one") -> Tensor:
        images = image_to_bchw(images, "gateresvit3_thermal_stats").to(dtype=torch.float32)
        if images.shape[1] == 1:
            grey = images[:, 0]
        else:
            grey = images[:, :3].mean(dim=1)
        if input_range == "zero_to_one":
            return grey.clamp(0.0, 1.0)
        if input_range == "uint8":
            return (grey / 255.0).clamp(0.0, 1.0)
        if input_range != "minus_one_to_one":
            raise ValueError(
                "GateResViT3 thermal stats input_range must be 'minus_one_to_one', "
                f"'zero_to_one', or 'uint8'. Got {input_range!r}."
            )
        return ((grey + 1.0) * 0.5).clamp(0.0, 1.0)

    @staticmethod
    def _center_crop(values: Tensor) -> Tensor:
        height, width = values.shape[-2:]
        top = height // 4
        bottom = height - top
        left = width // 4
        right = width - left
        if bottom <= top or right <= left:
            return values
        return values[:, top:bottom, left:right]

    @classmethod
    def compute_thermal_stats(
        cls,
        *,
        cold_images: Tensor,
        hot_images: Tensor,
        background_value: float,
        input_range: str = "minus_one_to_one",
        black_importance: bool = False,
    ) -> Tensor:
        cold = cls._grey_zero_one(cold_images, input_range=input_range)
        hot = cls._grey_zero_one(hot_images, input_range=input_range)
        boundary = float(background_value) / 255.0
        eps = 1.0 / 255.0

        if black_importance:
            cold_delta = (1.0 - cold).clamp_min(0.0)
            hot_delta = (1.0 - hot).clamp_min(0.0)
        else:
            cold_delta = (boundary - cold).clamp_min(0.0)
            hot_delta = (hot - boundary).clamp_min(0.0)
        cold_center = cls._center_crop(cold_delta)
        hot_center = cls._center_crop(hot_delta)
        total_delta = cold_delta + hot_delta

        def flat(values: Tensor) -> Tensor:
            return values.flatten(start_dim=1)

        cold_flat = flat(cold_delta)
        hot_flat = flat(hot_delta)
        cold_center_flat = flat(cold_center)
        hot_center_flat = flat(hot_center)
        total_flat = flat(total_delta)

        cold_mean = cold_flat.mean(dim=1)
        hot_mean = hot_flat.mean(dim=1)
        cold_std = cold_flat.std(dim=1, unbiased=False)
        hot_std = hot_flat.std(dim=1, unbiased=False)
        cold_max = cold_flat.max(dim=1).values
        hot_max = hot_flat.max(dim=1).values
        cold_area = (cold_flat > eps).to(dtype=torch.float32).mean(dim=1)
        hot_area = (hot_flat > eps).to(dtype=torch.float32).mean(dim=1)
        cold_center_mean = cold_center_flat.mean(dim=1)
        hot_center_mean = hot_center_flat.mean(dim=1)
        cold_center_area = (cold_center_flat > eps).to(dtype=torch.float32).mean(dim=1)
        hot_center_area = (hot_center_flat > eps).to(dtype=torch.float32).mean(dim=1)
        total_mean = total_flat.mean(dim=1)
        total_std = total_flat.std(dim=1, unbiased=False)
        max_delta = torch.maximum(cold_max, hot_max)
        balance = cold_mean - hot_mean

        return torch.stack(
            [
                cold_mean,
                hot_mean,
                cold_std,
                hot_std,
                cold_max,
                hot_max,
                cold_area,
                hot_area,
                cold_center_mean,
                hot_center_mean,
                cold_center_area,
                hot_center_area,
                total_mean,
                total_std,
                max_delta,
                balance,
            ],
            dim=1,
        )

    def _thermal_evidence_logit(self, stats: Tensor) -> tuple[Tensor, Tensor]:
        cold_area = stats[:, 6]
        hot_area = stats[:, 7]
        cold_center_mean = stats[:, 8]
        hot_center_mean = stats[:, 9]
        total_mean = stats[:, 12]
        total_std = stats[:, 13]
        max_delta = stats[:, 14]

        area = (cold_area + hot_area).clamp(0.0, 1.0)
        center_mean = (cold_center_mean + hot_center_mean).clamp_min(0.0)
        evidence = torch.maximum(4.0 * total_mean, 2.0 * total_std)
        evidence = torch.maximum(evidence, 2.0 * max_delta * torch.sqrt(area.clamp_min(1e-6)))
        evidence = torch.maximum(evidence, 4.0 * center_mean)
        evidence = evidence.clamp(0.0, 1.0)
        logit = self.evidence_gain * (evidence - self.evidence_threshold)
        return evidence, logit

    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        cold_tokens: Tensor,
        hot_tokens: Tensor,
        text_tokens: Tensor,
        rgb_token_mask: Tensor,
        cold_token_mask: Tensor,
        hot_token_mask: Tensor,
        text_token_mask: Tensor,
        thermal_stats: Tensor,
        base_cold_alpha: float = 1.0,
        base_hot_beta: float = 1.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        embed_dim = self.text_norm.normalized_shape[0]
        self._validate_token_shape("rgb_tokens", rgb_tokens, embed_dim)
        self._validate_token_shape("cold_tokens", cold_tokens, embed_dim)
        self._validate_token_shape("hot_tokens", hot_tokens, embed_dim)
        self._validate_token_shape("text_tokens", text_tokens, embed_dim)
        if rgb_tokens.shape != cold_tokens.shape or rgb_tokens.shape != hot_tokens.shape:
            raise ValueError(
                "GateResViT3 requires aligned RGB, cold thermal, and hot thermal token grids, "
                f"got rgb={tuple(rgb_tokens.shape)}, cold={tuple(cold_tokens.shape)}, "
                f"hot={tuple(hot_tokens.shape)}."
            )
        if thermal_stats.shape != (rgb_tokens.shape[0], self.stat_dim):
            raise ValueError(
                f"thermal_stats must have shape [{rgb_tokens.shape[0]}, {self.stat_dim}], "
                f"got {tuple(thermal_stats.shape)}."
            )

        batch_size, num_rgb_tokens = rgb_tokens.shape[:2]
        num_text_tokens = text_tokens.shape[1]
        rgb_token_mask = rgb_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        cold_token_mask = cold_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        hot_token_mask = hot_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        text_token_mask = text_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        self._validate_mask_shape("rgb_token_mask", rgb_token_mask, (batch_size, num_rgb_tokens))
        self._validate_mask_shape("cold_token_mask", cold_token_mask, (batch_size, num_rgb_tokens))
        self._validate_mask_shape("hot_token_mask", hot_token_mask, (batch_size, num_rgb_tokens))
        self._validate_mask_shape("text_token_mask", text_token_mask, (batch_size, num_text_tokens))

        module_dtype = self.text_norm.weight.dtype
        rgb_in = rgb_tokens.to(dtype=module_dtype)
        cold_in = cold_tokens.to(device=rgb_tokens.device, dtype=module_dtype)
        hot_in = hot_tokens.to(device=rgb_tokens.device, dtype=module_dtype)
        text_in = text_tokens.to(device=rgb_tokens.device, dtype=module_dtype)
        stats_in = thermal_stats.to(device=rgb_tokens.device, dtype=module_dtype)

        text_norm = self.text_norm(text_in)
        cold_norm = self.thermal_norm(cold_in)
        hot_norm = self.thermal_norm(hot_in)
        cold_context, cold_valid = self._text_to_thermal_context(
            text_tokens=text_norm,
            thermal_tokens=cold_norm,
            thermal_token_mask=cold_token_mask,
        )
        hot_context, hot_valid = self._text_to_thermal_context(
            text_tokens=text_norm,
            thermal_tokens=hot_norm,
            thermal_token_mask=hot_token_mask,
        )

        text_valid = text_token_mask.any(dim=1)
        text_pool = self._masked_mean(text_norm, text_token_mask)
        cold_pool = self._masked_mean(cold_context, text_token_mask)
        hot_pool = self._masked_mean(hot_context, text_token_mask)
        gate_input = torch.cat(
            [
                text_pool,
                cold_pool,
                hot_pool,
                cold_pool - text_pool,
                hot_pool - text_pool,
            ],
            dim=-1,
        )
        logits = self.gate_mlp(self.gate_input_norm(gate_input))
        multipliers = 2.0 * torch.softmax(logits / self.temperature, dim=-1)

        learned_confidence_logit = self.stats_mlp(self.stats_norm(stats_in)).squeeze(-1)
        thermal_evidence, evidence_logit = self._thermal_evidence_logit(stats_in)
        confidence_logit = learned_confidence_logit + evidence_logit
        thermal_confidence = torch.sigmoid(confidence_logit)
        valid_gate = text_valid.to(dtype=module_dtype)
        residual_valid = ((cold_valid | hot_valid) & text_valid).to(dtype=module_dtype)
        effective_thermal_confidence = thermal_confidence * residual_valid
        cold_weight = (
            float(base_cold_alpha)
            * multipliers[:, 0]
            * cold_valid.to(dtype=module_dtype)
            * valid_gate
        )
        hot_weight = (
            float(base_hot_beta)
            * multipliers[:, 1]
            * hot_valid.to(dtype=module_dtype)
            * valid_gate
        )
        cold_alpha = effective_thermal_confidence * cold_weight
        hot_beta = effective_thermal_confidence * hot_weight

        cold_residual = cold_in.masked_fill(~cold_token_mask[:, :, None], 0.0)
        hot_residual = hot_in.masked_fill(~hot_token_mask[:, :, None], 0.0)
        thermal_mix = (
            cold_weight[:, None, None] * cold_residual
            + hot_weight[:, None, None] * hot_residual
        )
        adapted = thermal_mix + self.residual_adapter(self.residual_norm(thermal_mix))
        fused = rgb_in + effective_thermal_confidence[:, None, None] * adapted
        fused = torch.where(rgb_token_mask[:, :, None], fused, rgb_in)
        gate_info = {
            "cold_alpha": cold_alpha.detach(),
            "hot_beta": hot_beta.detach(),
            "cold_weight": cold_weight.detach(),
            "hot_weight": hot_weight.detach(),
            "cold_multiplier": multipliers[:, 0].detach(),
            "hot_multiplier": multipliers[:, 1].detach(),
            "thermal_confidence": thermal_confidence.detach(),
            "effective_thermal_confidence": effective_thermal_confidence.detach(),
            "confidence_logit": confidence_logit.detach(),
            "learned_confidence_logit": learned_confidence_logit.detach(),
            "thermal_evidence": thermal_evidence.detach(),
            "evidence_logit": evidence_logit.detach(),
            "thermal_stats": thermal_stats.detach(),
            "logits": logits.detach(),
            "cold_valid": cold_valid.detach(),
            "hot_valid": hot_valid.detach(),
            "text_valid": text_valid.detach(),
        }
        return fused.to(dtype=rgb_tokens.dtype), gate_info


class RGBThermalTokenAligner(nn.Module):
    """Align thermal ViT tokens onto the head-RGB ViT token grid."""

    def __init__(self, token_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.rgb_norm = nn.LayerNorm(token_dim)
        self.thermal_norm = nn.LayerNorm(token_dim)
        self.cross_attn = nn.MultiheadAttention(
            token_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(token_dim * 2),
            nn.Linear(token_dim * 2, token_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        rgb_tokens: Tensor,
        thermal_tokens: Tensor,
        rgb_token_mask: Tensor,
        thermal_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        module_dtype = self.rgb_norm.weight.dtype
        rgb_in = rgb_tokens.to(dtype=module_dtype)
        thermal_in = thermal_tokens.to(dtype=module_dtype)

        key_padding_mask = ~thermal_token_mask
        all_thermal_padded = key_padding_mask.all(dim=1)
        if all_thermal_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_thermal_padded, 0] = False

        context, _ = self.cross_attn(
            query=self.rgb_norm(rgb_in),
            key=self.thermal_norm(thermal_in),
            value=self.thermal_norm(thermal_in),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        thermal_valid = thermal_token_mask.any(dim=1)
        context = context.masked_fill(~thermal_valid[:, None, None], 0.0)

        gate = self.gate(torch.cat([rgb_in, context], dim=-1))
        fused_tokens = self.out_norm(rgb_in + gate * context)
        fused_tokens = torch.where(thermal_valid[:, None, None], fused_tokens, rgb_in)
        fused_tokens = fused_tokens.masked_fill(~rgb_token_mask[:, :, None], 0.0)
        return fused_tokens, rgb_token_mask


class RGBThermalFusionDecoder(nn.Module):
    """Decode aligned RGB-thermal raw ViT tokens into a fused RGB image."""

    def __init__(
        self,
        token_dim: int,
        output_size: tuple[int, int],
        hidden_dim: int | None = None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(64, min(256, token_dim // 4))
        mid_dim = max(32, hidden_dim // 2)
        small_dim = max(16, hidden_dim // 4)

        self.output_size = tuple(int(dim) for dim in output_size)
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, mid_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_dim, small_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(small_dim, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    @staticmethod
    def _grid_shape(num_tokens: int) -> tuple[int, int]:
        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(
                "RGB-thermal fusion decoder expects a square ViT token grid, "
                f"got {num_tokens} tokens."
            )
        return side, side

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected token shape [B, N, D], got {tuple(tokens.shape)}")
        batch_size, num_tokens, _ = tokens.shape
        grid_h, grid_w = self._grid_shape(int(num_tokens))

        x = self.token_norm(tokens)
        x = self.token_proj(x)
        x = x.transpose(1, 2).reshape(batch_size, -1, grid_h, grid_w)
        x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)
        return self.decoder(x)


class ThermalHeadFusion(nn.Module):
    """Enrich thermal-query tokens by cross-attending to head-RGB tokens."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.thermal_norm = nn.LayerNorm(embed_dim)
        self.head_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        thermal_tokens: Tensor,
        head_tokens: Tensor,
        thermal_token_mask: Tensor,
        head_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        module_dtype = self.thermal_norm.weight.dtype
        thermal_in = thermal_tokens.to(dtype=module_dtype)
        head_in = head_tokens.to(dtype=module_dtype)

        key_padding_mask = ~head_token_mask
        all_head_padded = key_padding_mask.all(dim=1)
        if all_head_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_head_padded, 0] = False

        context, _ = self.cross_attn(
            query=self.thermal_norm(thermal_in),
            key=self.head_norm(head_in),
            value=self.head_norm(head_in),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        gate = self.gate(torch.cat([thermal_in, context], dim=-1))
        fused_tokens = self.out_norm(thermal_in + gate * context)

        fused_mask = thermal_token_mask & head_token_mask.any(dim=1)[:, None]
        fused_tokens = fused_tokens.masked_fill(~fused_mask[:, :, None], 0.0)
        return fused_tokens, fused_mask


def save_pi05_image_input_snapshots(
    context: str,
    image_keys: list[str],
    images: list[Tensor],
    img_masks: list[Tensor],
) -> None:
    """Best-effort dump of the first preprocessed image in each PI0.5 camera slot."""
    try:
        import cv2
        import numpy as np

        snapshot_dir = Path.cwd() / "tmp" / "pi05_inputs"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_id = time.time_ns()
        metadata = {"created_time_s": time.time(), "context": context, "saved": []}

        for slot_idx, (key, img, mask) in enumerate(zip(image_keys, images, img_masks, strict=True)):
            normalized = img[0].detach().to("cpu", dtype=torch.float32).contiguous()
            normalized_np = normalized.numpy()
            image_hwc = (
                np.transpose(normalized_np, (1, 2, 0))
                if normalized_np.ndim == 3 and normalized_np.shape[0] in {1, 3}
                else normalized_np
            )
            png_image = np.clip((image_hwc + 1.0) * 127.5, 0, 255).astype(np.uint8)
            if png_image.ndim == 3 and png_image.shape[2] == 1:
                png_image = png_image[:, :, 0]
            elif png_image.ndim == 3 and png_image.shape[2] == 3:
                png_image = cv2.cvtColor(png_image, cv2.COLOR_RGB2BGR)

            safe_key = key.replace("observation.images.", "").replace(".", "_").replace("/", "_")
            file_stem = f"pi05_input_{timestamp}_{unique_id}_{context}_slot{slot_idx}_{safe_key}"
            npy_path = snapshot_dir / f"{file_stem}.npy"
            png_path = snapshot_dir / f"{file_stem}.png"
            np.save(npy_path, normalized_np)
            png_ok = bool(cv2.imwrite(str(png_path), png_image))
            metadata["saved"].append(
                {
                    "slot": slot_idx,
                    "feature": key,
                    "shape": list(normalized_np.shape),
                    "dtype": str(normalized.dtype),
                    "mask_true": int(mask.sum().item()),
                    "min": float(normalized.min().item()),
                    "mean": float(normalized.mean().item()),
                    "max": float(normalized.max().item()),
                    "npy_path": str(npy_path),
                    "png_path": str(png_path) if png_ok else None,
                }
            )

        metadata_path = snapshot_dir / f"pi05_input_{timestamp}_{unique_id}_{context}_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logging.info("Saved PI05 image input snapshot metadata to %s", metadata_path)
    except Exception as exc:
        logging.warning("Failed to save PI05 image input snapshots: %s", exc)
