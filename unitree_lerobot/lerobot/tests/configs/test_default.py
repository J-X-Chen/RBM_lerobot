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
    with pytest.raises(ValueError, match="true, false, or 'ignore'"):
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
