# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import pytest

import lerobot.configs.train as tc
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


class _FakeHTTPError(tc.HfHubHTTPError):
    """HfHubHTTPError that can be raised without a real HTTP response object."""

    def __init__(self):
        pass


def test_replace_cli_flag_removes_existing_output_without_prompt(tmp_path, monkeypatch):
    output_dir = tmp_path / "existing-output"
    output_dir.mkdir()
    (output_dir / "old-checkpoint.txt").write_text("old")
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: pytest.fail("replace=true must not prompt for input"),
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/dataset"),
        policy=PI05Config(push_to_hub=False),
        output_dir=output_dir,
        replace=True,
    )

    cfg.validate()

    assert not output_dir.exists()


def test_replace_is_available_from_cli():
    cfg = tc.draccus.parse(
        TrainPipelineConfig,
        args=["--dataset.repo_id", "user/dataset", "--replace=true"],
    )

    assert cfg.replace is True


def test_from_pretrained_falls_back_to_latest_checkpoint_config(tmp_path, monkeypatch):
    """A Hub repo with no root train_config.json (an interrupted run that only pushed
    checkpoints/) resolves via the latest checkpoint's config."""
    # A real train_config.json written by save_pretrained, to be returned by the fallback.
    parsed = tc.draccus.parse(TrainPipelineConfig, args=["--dataset.repo_id", "u/d"])
    cfg_file = tmp_path / "train_config.json"
    parsed._save_pretrained(tmp_path)
    assert cfg_file.is_file()

    calls = []

    def fake_hf_hub_download(filename=None, **kwargs):
        calls.append(filename)
        if filename == "train_config.json":
            raise _FakeHTTPError()  # no root config
        if filename == "checkpoints/000010/pretrained_model/train_config.json":
            return str(cfg_file)
        raise AssertionError(f"unexpected filename {filename}")

    monkeypatch.setattr(tc, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(
        tc, "find_latest_hub_checkpoint", lambda repo_id, token=None, revision=None: "checkpoints/000010"
    )

    loaded = TrainPipelineConfig.from_pretrained("user/interrupted-run")
    assert loaded.dataset.repo_id == "u/d"
    # Tried the root config first, then fell back to the latest checkpoint's config.
    assert calls == ["train_config.json", "checkpoints/000010/pretrained_model/train_config.json"]


def test_from_pretrained_raises_when_no_root_config_and_no_checkpoints(monkeypatch):
    """No root config AND no checkpoints → a clear FileNotFoundError, not the raw HTTP error."""

    def fake_hf_hub_download(filename=None, **kwargs):
        raise _FakeHTTPError()

    monkeypatch.setattr(tc, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(tc, "find_latest_hub_checkpoint", lambda repo_id, token=None, revision=None: None)

    with pytest.raises(FileNotFoundError, match="train_config.json not found"):
        TrainPipelineConfig.from_pretrained("user/empty-repo")
