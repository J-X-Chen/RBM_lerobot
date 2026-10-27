#!/usr/bin/env python

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.utils.constants import ACTION
from lerobot.utils.dex3_action import (
    DEX3_LEFT_CLOSED_Q,
    DEX3_RIGHT_CLOSED_Q,
    compress_dex3_action,
    enable_dex3_action_compression,
    expand_dex3_action,
)


@pytest.mark.parametrize("array_type", [np.asarray, torch.as_tensor])
def test_compression_round_trip(array_type):
    action = np.zeros((3, 28), dtype=np.float32)
    action[:, :14] = np.arange(14, dtype=np.float32)
    action[:, 14:21] = np.asarray(DEX3_LEFT_CLOSED_Q) * 0.25
    action[:, 21:28] = np.asarray(DEX3_RIGHT_CLOSED_Q) * 0.75
    action = array_type(action)

    compressed = compress_dex3_action(action)
    expanded = expand_dex3_action(compressed)

    assert compressed.shape == (3, 16)
    np.testing.assert_allclose(np.asarray(compressed[..., -2:]), [[0.25, 0.75]] * 3, atol=1e-6)
    np.testing.assert_allclose(np.asarray(expanded), np.asarray(action), atol=1e-6)


def test_expansion_clips_grip_range():
    compressed = np.zeros(16, dtype=np.float32)
    compressed[-2:] = [-0.2, 1.2]

    expanded = expand_dex3_action(compressed)

    np.testing.assert_allclose(expanded[14:21], 0.0)
    np.testing.assert_allclose(expanded[21:28], DEX3_RIGHT_CLOSED_Q)


def test_compression_rejects_wrong_dimension():
    with pytest.raises(ValueError, match="28D"):
        compress_dex3_action(np.zeros(16, dtype=np.float32))


@pytest.mark.parametrize("array_type", [np.asarray, torch.as_tensor])
def test_compression_can_ignore_hands_while_preserving_16d_interface(array_type):
    action = np.zeros((3, 28), dtype=np.float32)
    action[:, :14] = np.arange(14, dtype=np.float32)
    action[:, 14:21] = np.asarray(DEX3_LEFT_CLOSED_Q)
    action[:, 21:28] = np.asarray(DEX3_RIGHT_CLOSED_Q)

    compressed = compress_dex3_action(array_type(action), ignore_hands=True)

    assert compressed.shape == (3, 16)
    np.testing.assert_allclose(np.asarray(compressed[:, :14]), action[:, :14])
    np.testing.assert_array_equal(np.asarray(compressed[:, -2:]), 0.0)


def test_ignored_sparse_hands_do_not_explode_under_quantile_normalization():
    action = np.zeros((1_000, 28), dtype=np.float32)
    action[0, 14:21] = np.asarray(DEX3_LEFT_CLOSED_Q)
    action[1, 21:28] = np.asarray(DEX3_RIGHT_CLOSED_Q)
    compressed = compress_dex3_action(action, ignore_hands=True)
    stats = get_feature_stats(compressed, axis=0, keepdims=False)

    denom = stats["q99"] - stats["q01"]
    denom = np.where(denom == 0, 1e-8, denom)
    normalized = 2.0 * (compressed - stats["q01"]) / denom - 1.0

    np.testing.assert_array_equal(compressed[:, -2:], 0.0)
    assert np.isfinite(normalized).all()
    assert np.abs(normalized[:, -2:]).max() < 2.0


def test_enable_compression_with_ignored_hands_updates_data_and_stats():
    raw_actions = np.zeros((3, 28), dtype=np.float32)
    raw_actions[0, 14:21] = np.asarray(DEX3_LEFT_CLOSED_Q)
    raw_actions[1, 21:28] = np.asarray(DEX3_RIGHT_CLOSED_Q)

    class FakeHfDataset(dict):
        def with_format(self, *args, **kwargs):
            return self

    class FakeMeta:
        def __init__(self):
            action_feature = {
                "shape": [28],
                "names": [[f"joint_{index}" for index in range(28)]],
            }
            self.info = SimpleNamespace(features={ACTION: action_feature})
            self.robot_type = "Unitree_G1_Dex3"
            self.stats = {}

        @property
        def features(self):
            return self.info.features

    class FakeDataset:
        def __init__(self):
            self.meta = FakeMeta()
            self.hf_dataset = FakeHfDataset({ACTION: raw_actions})

        def __len__(self):
            return len(raw_actions)

        def __getitem__(self, index):
            return {ACTION: raw_actions[index].copy()}

    dataset = enable_dex3_action_compression(FakeDataset(), ignore_hands=True)

    assert dataset.meta.features[ACTION]["shape"] == [16]
    np.testing.assert_array_equal(dataset[0][ACTION][-2:], 0.0)
    for stat_name, stat_value in dataset.meta.stats[ACTION].items():
        if stat_name != "count":
            np.testing.assert_array_equal(stat_value[-2:], 0.0)
