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

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

THERMOACT_CHANNEL = "thermoact"
DEFAULT_THERMOACT_THERMAL_FEATURE = "observation.images.cam_thermal"
DEFAULT_THERMOACT_TEMPERATURE_MIN_C = 20.0
DEFAULT_THERMOACT_TEMPERATURE_MAX_C = 35.0

_INFERNO_RGB_LUT: Tensor | None = None
_INFERNO_RGB_LUT_BY_DEVICE: dict[str, Tensor] = {}


def _build_inferno_rgb_lut() -> Tensor:
    """Return OpenCV's INFERNO colormap as RGB float values in [0, 1]."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "ThermoAct thermal preprocessing requires opencv-python-headless for COLORMAP_INFERNO."
        ) from exc

    grayscale = np.arange(256, dtype=np.uint8).reshape(256, 1)
    bgr = cv2.applyColorMap(grayscale, cv2.COLORMAP_INFERNO).reshape(256, 3)
    rgb = bgr[:, ::-1].copy()
    return torch.from_numpy(rgb).to(dtype=torch.float32).div(255.0)


def inferno_rgb_lut(device: torch.device | str | None = None) -> Tensor:
    """Return the INFERNO lookup table on the requested device."""
    global _INFERNO_RGB_LUT
    if _INFERNO_RGB_LUT is None:
        _INFERNO_RGB_LUT = _build_inferno_rgb_lut()

    target_device = torch.device("cpu") if device is None else torch.device(device)
    cache_key = str(target_device)
    if cache_key not in _INFERNO_RGB_LUT_BY_DEVICE:
        _INFERNO_RGB_LUT_BY_DEVICE[cache_key] = _INFERNO_RGB_LUT.to(device=target_device)
    return _INFERNO_RGB_LUT_BY_DEVICE[cache_key]


def _thermal_scalar_plane(thermal: Tensor, *, batched: bool = False) -> tuple[Tensor, str]:
    """Extract one thermal channel while remembering the original image layout."""
    if thermal.ndim < 2:
        raise ValueError(
            f"ThermoAct thermal input must have at least 2 dimensions, got {thermal.shape}."
        )

    if thermal.ndim == 2:
        return thermal, "no_channel"

    channel_sizes = {1, 3, 4}
    if thermal.ndim == 3:
        if batched:
            return thermal, "no_channel"
        if thermal.shape[0] in channel_sizes:
            return thermal[0], "channels_first"
        if thermal.shape[-1] in channel_sizes:
            return thermal[..., 0], "channels_last"
        return thermal, "no_channel"

    if thermal.shape[1] in channel_sizes:
        return thermal[:, 0], "channels_first"
    if thermal.shape[-1] in channel_sizes:
        return thermal[..., 0], "channels_last"

    return thermal, "no_channel"


def _restore_pseudo_rgb_layout(pseudo_rgb: Tensor, layout: str) -> Tensor:
    """Restore pseudo-RGB to the layout expected by the existing PI0 image path."""
    if layout == "channels_last":
        return pseudo_rgb
    if pseudo_rgb.ndim == 3:
        return pseudo_rgb.movedim(-1, 0)
    return pseudo_rgb.movedim(-1, -3)


def thermoact_thermal_to_pseudo_rgb(
    thermal: Tensor,
    *,
    temperature_min_c: float = DEFAULT_THERMOACT_TEMPERATURE_MIN_C,
    temperature_max_c: float = DEFAULT_THERMOACT_TEMPERATURE_MAX_C,
    batched: bool = False,
) -> Tensor:
    """Convert raw thermal Celsius frames to ThermoAct-style INFERNO pseudo-RGB.

    Pipeline:
      raw thermal -> 20-35 C normalization -> 8-bit grayscale -> INFERNO pseudo-color.

    The output is a 3-channel float32 image in [0, 1], preserving channel-first or
    channel-last layout when the input had an explicit channel dimension.
    """
    if temperature_max_c <= temperature_min_c:
        raise ValueError(
            "ThermoAct temperature_max_c must be greater than temperature_min_c, "
            f"got {temperature_min_c} and {temperature_max_c}."
        )

    thermal_plane, layout = _thermal_scalar_plane(thermal, batched=batched)
    normalized = (thermal_plane.to(dtype=torch.float32) - temperature_min_c) / (
        temperature_max_c - temperature_min_c
    )
    grayscale_indices = torch.round(normalized.clamp(0.0, 1.0) * 255.0).to(
        dtype=torch.long
    )
    pseudo_rgb = inferno_rgb_lut(device=thermal.device)[grayscale_indices]
    return _restore_pseudo_rgb_layout(pseudo_rgb, layout)
