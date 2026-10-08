#!/usr/bin/env python

"""Runtime action/state views for Astribot S1 whole-body datasets."""

from copy import deepcopy
from typing import TypeVar

import numpy as np
import torch

from lerobot.utils.constants import ACTION, OBS_STATE


ASTRIBOT_S1_FULL_DIM = 25
ASTRIBOT_S1_ARM_GRIPPER_DIM = 16

# Dataset order:
# chassis(3), torso(4), left arm(7), left gripper(1), right arm(7), right gripper(1), head(2).
ASTRIBOT_S1_ARM_GRIPPER_INDICES = tuple(range(7, 15)) + tuple(range(15, 23))

ArrayLike = TypeVar("ArrayLike", np.ndarray, torch.Tensor)


def select_astribot_s1_arm_gripper(value: ArrayLike) -> ArrayLike:
    """Keep only left/right arm joints and left/right gripper values."""
    if value.shape[-1] != ASTRIBOT_S1_FULL_DIM:
        raise ValueError(
            f"Expected a {ASTRIBOT_S1_FULL_DIM}D Astribot S1 vector, got shape {tuple(value.shape)}."
        )
    if isinstance(value, torch.Tensor):
        indices = torch.as_tensor(ASTRIBOT_S1_ARM_GRIPPER_INDICES, device=value.device)
        return value.index_select(dim=-1, index=indices)
    return value[..., list(ASTRIBOT_S1_ARM_GRIPPER_INDICES)]


def expand_astribot_s1_arm_gripper(value: ArrayLike, full_reference: ArrayLike) -> ArrayLike:
    """Insert a 16D arms+grippers vector into a 25D whole-body reference pose."""
    if value.shape[-1] != ASTRIBOT_S1_ARM_GRIPPER_DIM:
        raise ValueError(
            f"Expected a {ASTRIBOT_S1_ARM_GRIPPER_DIM}D Astribot S1 arm/gripper vector, "
            f"got shape {tuple(value.shape)}."
        )
    if full_reference.shape[-1] != ASTRIBOT_S1_FULL_DIM:
        raise ValueError(
            f"Expected a {ASTRIBOT_S1_FULL_DIM}D Astribot S1 reference vector, "
            f"got shape {tuple(full_reference.shape)}."
        )

    if isinstance(full_reference, torch.Tensor):
        expanded = full_reference.clone()
        indices = torch.as_tensor(ASTRIBOT_S1_ARM_GRIPPER_INDICES, device=expanded.device)
        expanded.index_copy_(-1, indices, value.to(device=expanded.device, dtype=expanded.dtype))
        return expanded

    expanded = np.array(full_reference, copy=True)
    expanded[..., list(ASTRIBOT_S1_ARM_GRIPPER_INDICES)] = np.asarray(value, dtype=expanded.dtype)
    return expanded


def _slice_names(names):
    if not isinstance(names, list):
        return names
    nested = len(names) == 1 and isinstance(names[0], list)
    flat_names = names[0] if nested else names
    if len(flat_names) != ASTRIBOT_S1_FULL_DIM:
        return names
    sliced = [flat_names[i] for i in ASTRIBOT_S1_ARM_GRIPPER_INDICES]
    return [sliced] if nested else sliced


def _slice_stat_value(value):
    if isinstance(value, torch.Tensor):
        if value.shape and value.shape[-1] == ASTRIBOT_S1_FULL_DIM:
            indices = torch.as_tensor(ASTRIBOT_S1_ARM_GRIPPER_INDICES, device=value.device)
            return value.index_select(dim=-1, index=indices)
        return value
    if isinstance(value, np.ndarray):
        if value.shape and value.shape[-1] == ASTRIBOT_S1_FULL_DIM:
            return value[..., list(ASTRIBOT_S1_ARM_GRIPPER_INDICES)]
        return value
    if isinstance(value, list):
        if len(value) == ASTRIBOT_S1_FULL_DIM:
            return [value[i] for i in ASTRIBOT_S1_ARM_GRIPPER_INDICES]
        if len(value) == 1 and isinstance(value[0], list) and len(value[0]) == ASTRIBOT_S1_FULL_DIM:
            return [[value[0][i] for i in ASTRIBOT_S1_ARM_GRIPPER_INDICES]]
    return deepcopy(value)


def _slice_stats(stats: dict) -> dict:
    return {key: _slice_stat_value(value) if key != "count" else deepcopy(value) for key, value in stats.items()}


def _update_feature_metadata(dataset, key: str) -> None:
    feature = dataset.meta.features.get(key)
    if feature is None:
        raise ValueError(f"Astribot S1 arm/gripper filtering requires the dataset '{key}' feature.")
    if tuple(feature.get("shape", ())) != (ASTRIBOT_S1_FULL_DIM,):
        raise ValueError(
            f"Astribot S1 arm/gripper filtering requires a {ASTRIBOT_S1_FULL_DIM}D '{key}' feature, "
            f"got {feature.get('shape')}."
        )

    filtered_feature = deepcopy(feature)
    filtered_feature["shape"] = [ASTRIBOT_S1_ARM_GRIPPER_DIM]
    filtered_feature["names"] = _slice_names(filtered_feature.get("names"))
    dataset.meta.info.features[key] = filtered_feature

    if key in dataset.meta.stats:
        dataset.meta.stats[key] = _slice_stats(dataset.meta.stats[key])


class AstribotS1ArmGripperDataset(torch.utils.data.Dataset):
    """Runtime view of a 25D Astribot S1 dataset as 16D arms + grippers."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item[ACTION] = select_astribot_s1_arm_gripper(item[ACTION])
        item[OBS_STATE] = select_astribot_s1_arm_gripper(item[OBS_STATE])
        return item

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def enable_astribot_s1_arm_gripper_filter(dataset) -> AstribotS1ArmGripperDataset:
    """Create a runtime view and update in-memory metadata/stats to 16D."""
    _update_feature_metadata(dataset, ACTION)
    _update_feature_metadata(dataset, OBS_STATE)
    return AstribotS1ArmGripperDataset(dataset)
