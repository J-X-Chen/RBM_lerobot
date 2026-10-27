#!/usr/bin/env python

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.utils.astribot_action import (
    ASTRIBOT_S1_ACTION_VIEW_ARM_GRIPPER,
    ASTRIBOT_S1_ACTION_VIEW_AUTO,
    ASTRIBOT_S1_ACTION_VIEW_FULL,
    ASTRIBOT_S1_ARM_GRIPPER_DIM,
    ASTRIBOT_S1_ARM_GRIPPER_INDICES,
    ASTRIBOT_S1_FULL_DIM,
    enable_astribot_s1_arm_gripper_filter,
    expand_astribot_s1_arm_gripper,
    normalize_astribot_s1_action_view,
    select_astribot_s1_arm_gripper,
)
from lerobot.utils.constants import ACTION, OBS_STATE


@pytest.mark.parametrize("array_type", [np.asarray, torch.as_tensor])
def test_astribot_s1_arm_gripper_view_round_trip(array_type):
    full = np.arange(2 * ASTRIBOT_S1_FULL_DIM, dtype=np.float32).reshape(2, ASTRIBOT_S1_FULL_DIM)
    reference = full + 1000.0

    selected = select_astribot_s1_arm_gripper(array_type(full))
    expanded = expand_astribot_s1_arm_gripper(selected, array_type(reference))

    assert selected.shape == (2, ASTRIBOT_S1_ARM_GRIPPER_DIM)
    np.testing.assert_array_equal(np.asarray(selected), full[:, list(ASTRIBOT_S1_ARM_GRIPPER_INDICES)])
    expanded_np = np.asarray(expanded)
    np.testing.assert_array_equal(expanded_np[:, list(ASTRIBOT_S1_ARM_GRIPPER_INDICES)], np.asarray(selected))
    locked_indices = sorted(set(range(ASTRIBOT_S1_FULL_DIM)).difference(ASTRIBOT_S1_ARM_GRIPPER_INDICES))
    np.testing.assert_array_equal(expanded_np[:, locked_indices], reference[:, locked_indices])


def test_astribot_s1_arm_gripper_filter_updates_metadata_and_items():
    raw = np.arange(ASTRIBOT_S1_FULL_DIM, dtype=np.float32)

    class FakeMeta:
        def __init__(self):
            features = {
                ACTION: {"shape": [ASTRIBOT_S1_FULL_DIM], "names": [[f"action_{i}" for i in range(25)]]},
                OBS_STATE: {"shape": [ASTRIBOT_S1_FULL_DIM], "names": [f"state_{i}" for i in range(25)]},
            }
            self.info = SimpleNamespace(features=features)
            self.stats = {
                ACTION: {"mean": raw.copy(), "count": np.array([1])},
                OBS_STATE: {"mean": raw.copy(), "count": np.array([1])},
            }

        @property
        def features(self):
            return self.info.features

    class FakeDataset:
        def __init__(self):
            self.meta = FakeMeta()

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {ACTION: torch.as_tensor(raw), OBS_STATE: torch.as_tensor(raw + 100)}

    dataset = enable_astribot_s1_arm_gripper_filter(FakeDataset())

    assert dataset.meta.features[ACTION]["shape"] == [ASTRIBOT_S1_ARM_GRIPPER_DIM]
    assert dataset.meta.features[OBS_STATE]["shape"] == [ASTRIBOT_S1_ARM_GRIPPER_DIM]
    assert len(dataset.meta.features[ACTION]["names"][0]) == ASTRIBOT_S1_ARM_GRIPPER_DIM
    np.testing.assert_array_equal(
        dataset.meta.stats[ACTION]["mean"],
        raw[list(ASTRIBOT_S1_ARM_GRIPPER_INDICES)],
    )
    np.testing.assert_array_equal(
        dataset[0][ACTION].numpy(),
        raw[list(ASTRIBOT_S1_ARM_GRIPPER_INDICES)],
    )


def test_astribot_s1_arm_gripper_filter_accepts_already_16d_dataset():
    raw = np.arange(ASTRIBOT_S1_ARM_GRIPPER_DIM, dtype=np.float32)

    class FakeMeta:
        def __init__(self):
            features = {
                ACTION: {"shape": [ASTRIBOT_S1_ARM_GRIPPER_DIM], "names": [f"action_{i}" for i in range(16)]},
                OBS_STATE: {
                    "shape": [ASTRIBOT_S1_ARM_GRIPPER_DIM],
                    "names": [f"state_{i}" for i in range(16)],
                },
            }
            self.info = SimpleNamespace(features=features)
            self.stats = {ACTION: {"mean": raw.copy()}, OBS_STATE: {"mean": raw.copy()}}

        @property
        def features(self):
            return self.info.features

    class FakeDataset:
        def __init__(self):
            self.meta = FakeMeta()

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {ACTION: torch.as_tensor(raw), OBS_STATE: torch.as_tensor(raw + 100)}

    raw_dataset = FakeDataset()
    dataset = enable_astribot_s1_arm_gripper_filter(raw_dataset)

    assert dataset is raw_dataset
    assert dataset.meta.features[ACTION]["shape"] == [ASTRIBOT_S1_ARM_GRIPPER_DIM]
    np.testing.assert_array_equal(dataset[0][ACTION].numpy(), raw)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("full", ASTRIBOT_S1_ACTION_VIEW_FULL),
        ("25d", ASTRIBOT_S1_ACTION_VIEW_FULL),
        ("arms_grippers", ASTRIBOT_S1_ACTION_VIEW_ARM_GRIPPER),
        ("16d", ASTRIBOT_S1_ACTION_VIEW_ARM_GRIPPER),
        ("auto", ASTRIBOT_S1_ACTION_VIEW_AUTO),
        (True, ASTRIBOT_S1_ACTION_VIEW_ARM_GRIPPER),
        (False, ASTRIBOT_S1_ACTION_VIEW_FULL),
    ],
)
def test_normalize_astribot_s1_action_view(value, expected):
    assert normalize_astribot_s1_action_view(value) == expected
