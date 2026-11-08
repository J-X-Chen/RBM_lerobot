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
import draccus
import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.utils.astribot_action import (
    ASTRIBOT_S1_ACTION_VIEW_ARM_GRIPPER,
    ASTRIBOT_S1_ACTION_VIEW_AUTO,
    ASTRIBOT_S1_ACTION_VIEW_FULL,
)


def test_dataset_config_valid():
    DatasetConfig(repo_id="user/repo", episodes=[0, 1, 2])


def test_dataset_config_negative_episodes():
    with pytest.raises(ValueError, match="non-negative"):
        DatasetConfig(repo_id="user/repo", episodes=[0, -1, 2])


def test_dataset_config_duplicate_episodes():
    with pytest.raises(ValueError, match="duplicates"):
        DatasetConfig(repo_id="user/repo", episodes=[0, 1, 1, 2])


def test_dataset_config_none_episodes_ok():
    DatasetConfig(repo_id="user/repo", episodes=None)


def test_dataset_config_empty_episodes_ok():
    DatasetConfig(repo_id="user/repo", episodes=[])


@pytest.mark.parametrize("mode", [True, False, "ignore", "IGNORE"])
def test_dataset_config_accepts_dex3_action_compression_modes(mode):
    config = DatasetConfig(repo_id="user/repo", compress_dex3_actions=mode)

    assert config.compress_dex3_actions == (mode.lower() if isinstance(mode, str) else mode)


def test_dataset_config_rejects_unknown_dex3_action_compression_mode():
    with pytest.raises(ValueError, match="true, false, 'auto', or 'ignore'"):
        DatasetConfig(repo_id="user/repo", compress_dex3_actions="unknown")


@pytest.mark.parametrize(
    ("cli_value", "expected"),
    [("true", True), ("false", False), ("ignore", "ignore")],
)
def test_dataset_config_parses_dex3_action_compression_modes_from_cli(cli_value, expected):
    config = draccus.parse(
        DatasetConfig,
        args=["--repo_id=user/repo", f"--compress_dex3_actions={cli_value}"],
    )

    assert config.compress_dex3_actions == expected


@pytest.mark.parametrize(
    ("view", "expected"),
    [
        ("full", ASTRIBOT_S1_ACTION_VIEW_FULL),
        ("25d", ASTRIBOT_S1_ACTION_VIEW_FULL),
        ("arms_grippers", ASTRIBOT_S1_ACTION_VIEW_ARM_GRIPPER),
        ("16d", ASTRIBOT_S1_ACTION_VIEW_ARM_GRIPPER),
        ("auto", ASTRIBOT_S1_ACTION_VIEW_AUTO),
    ],
)
def test_dataset_config_accepts_astribot_action_view(view, expected):
    config = DatasetConfig(repo_id="user/repo", astribot_action_view=view)

    assert config.astribot_action_view == expected


@pytest.mark.parametrize(
    ("legacy_value", "expected"),
    [
        (True, ASTRIBOT_S1_ACTION_VIEW_ARM_GRIPPER),
        ("ignore", ASTRIBOT_S1_ACTION_VIEW_ARM_GRIPPER),
        ("auto", ASTRIBOT_S1_ACTION_VIEW_AUTO),
        (False, ASTRIBOT_S1_ACTION_VIEW_FULL),
    ],
)
def test_dataset_config_maps_legacy_astribot_compression_to_action_view(legacy_value, expected):
    config = DatasetConfig(repo_id="user/repo", compress_astribot_actions=legacy_value)

    assert config.astribot_action_view == expected


def test_dataset_config_rejects_unknown_astribot_action_view():
    with pytest.raises(ValueError, match="astribot_action_view"):
        DatasetConfig(repo_id="user/repo", astribot_action_view="unknown")
