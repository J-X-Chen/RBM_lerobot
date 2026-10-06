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

"""Runtime compression for fixed-shape Dex3 grasp actions."""

from copy import deepcopy
from typing import TypeVar

import numpy as np
import torch

from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.utils.constants import ACTION


DEX3_DOF = 7
DEX3_DUAL_ARM_DOF = 14
DEX3_FULL_ACTION_DIM = 28
DEX3_COMPRESSED_ACTION_DIM = 16

# Dex3 order: thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1.
DEX3_LEFT_CLOSED_Q = (0.0, 0.75, 1.35, -1.15, -1.35, -1.15, -1.35)
DEX3_RIGHT_CLOSED_Q = (0.0, -0.75, -1.35, 1.15, 1.35, 1.15, 1.35)

ArrayLike = TypeVar("ArrayLike", np.ndarray, torch.Tensor)


def _template_like(values: tuple[float, ...], value: ArrayLike) -> ArrayLike:
    if isinstance(value, torch.Tensor):
        return torch.as_tensor(values, dtype=value.dtype, device=value.device)
    return np.asarray(values, dtype=value.dtype)


def _concatenate(values: list[ArrayLike], reference: ArrayLike) -> ArrayLike:
    if isinstance(reference, torch.Tensor):
        return torch.cat(values, dim=-1)
    return np.concatenate(values, axis=-1)


def _clip01(value: ArrayLike) -> ArrayLike:
    if isinstance(value, torch.Tensor):
        return value.clamp(0.0, 1.0)
    return np.clip(value, 0.0, 1.0)


def compress_dex3_action(
    action: ArrayLike, *, arm_dof: int = DEX3_DUAL_ARM_DOF, ignore_hands: bool = False
) -> ArrayLike:
    """Project each fixed-shape 7-DoF Dex3 grasp onto one grip scalar.

    When ``ignore_hands`` is enabled, preserve the 14D arm + 2D grip policy interface but
    replace both grip targets with exact zeros. This avoids sparse hand commands producing
    degenerate quantile ranges and extreme normalized action values.
    """
    expected_dim = arm_dof + 2 * DEX3_DOF
    if action.shape[-1] != expected_dim:
        raise ValueError(f"Expected a {expected_dim}D Dex3 action, got shape {tuple(action.shape)}.")

    if ignore_hands:
        ignored_grips = (
            torch.zeros_like(action[..., :2])
            if isinstance(action, torch.Tensor)
            else np.zeros_like(action[..., :2])
        )
        return _concatenate([action[..., :arm_dof], ignored_grips], action)

    left = action[..., arm_dof : arm_dof + DEX3_DOF]
    right = action[..., arm_dof + DEX3_DOF : expected_dim]
    left_closed = _template_like(DEX3_LEFT_CLOSED_Q, action)
    right_closed = _template_like(DEX3_RIGHT_CLOSED_Q, action)
    left_grip = (left * left_closed).sum(axis=-1, keepdims=True) / (left_closed * left_closed).sum()
    right_grip = (right * right_closed).sum(axis=-1, keepdims=True) / (
        right_closed * right_closed
    ).sum()
    return _concatenate([action[..., :arm_dof], _clip01(left_grip), _clip01(right_grip)], action)


def expand_dex3_action(action: ArrayLike, *, arm_dof: int = DEX3_DUAL_ARM_DOF) -> ArrayLike:
    """Expand two grip scalars into fixed-shape left/right Dex3 joint targets."""
    expected_dim = arm_dof + 2
    if action.shape[-1] != expected_dim:
        raise ValueError(
            f"Expected a {expected_dim}D arm + compressed Dex3 action, got shape {tuple(action.shape)}."
        )

    left_closed = _template_like(DEX3_LEFT_CLOSED_Q, action)
    right_closed = _template_like(DEX3_RIGHT_CLOSED_Q, action)
    left = _clip01(action[..., arm_dof : arm_dof + 1]) * left_closed
    right = _clip01(action[..., arm_dof + 1 : arm_dof + 2]) * right_closed
    return _concatenate([action[..., :arm_dof], left, right], action)


class Dex3ActionCompressedDataset(torch.utils.data.Dataset):
    """Runtime view of a dataset with 28D Dex3 actions compressed to 16D."""

    def __init__(self, dataset, *, ignore_hands: bool = False):
        self.dataset = dataset
        self.ignore_hands = ignore_hands

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item[ACTION] = compress_dex3_action(item[ACTION], ignore_hands=self.ignore_hands)
        return item

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def enable_dex3_action_compression(
    dataset, *, ignore_hands: bool = False
) -> Dex3ActionCompressedDataset:
    """Create a compressed runtime view and update in-memory metadata."""
    feature = dataset.meta.features.get(ACTION)
    if feature is None:
        raise ValueError("Dex3 action compression requires the dataset 'action' feature.")
    if tuple(feature["shape"]) != (DEX3_FULL_ACTION_DIM,):
        raise ValueError(
            "Dex3 action compression currently requires a 28D action "
            f"(14 arm + 7 left hand + 7 right hand), got {feature['shape']}."
        )

    robot_type = str(getattr(dataset.meta, "robot_type", "")).lower()
    if "dex3" not in robot_type:
        raise ValueError(
            "Dex3 action compression was requested for a non-Dex3 dataset "
            f"(robot_type={getattr(dataset.meta, 'robot_type', None)!r})."
        )

    raw_action_column = dataset.hf_dataset.with_format(
        "numpy", columns=[ACTION], output_all_columns=False
    )[ACTION]
    compressed_actions = compress_dex3_action(
        np.stack(list(raw_action_column)), ignore_hands=ignore_hands
    )

    compressed_feature = deepcopy(feature)
    compressed_feature["shape"] = [DEX3_COMPRESSED_ACTION_DIM]
    names = compressed_feature.get("names")
    if isinstance(names, list):
        flat_names = names[0] if len(names) == 1 and isinstance(names[0], list) else names
        if len(flat_names) == DEX3_FULL_ACTION_DIM:
            compressed_feature["names"] = [
                *flat_names[:DEX3_DUAL_ARM_DOF],
                "kLeftHandGrip",
                "kRightHandGrip",
            ]
    dataset.meta.info.features[ACTION] = compressed_feature
    compressed_stats = get_feature_stats(compressed_actions, axis=0, keepdims=False)
    if ignore_hands:
        # Histogram-based quantiles can return tiny nonzero interpolation artifacts for an
        # all-zero feature. Store the exact constant statistics so normalization and
        # unnormalization preserve an exact zero grip target.
        for stat_name, stat_value in compressed_stats.items():
            if stat_name != "count":
                stat_value[..., -2:] = 0
    dataset.meta.stats[ACTION] = compressed_stats
    return Dex3ActionCompressedDataset(dataset, ignore_hands=ignore_hands)
