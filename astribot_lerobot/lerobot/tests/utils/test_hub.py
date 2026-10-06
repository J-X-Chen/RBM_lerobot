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

from unittest.mock import MagicMock

from lerobot.utils.hub import find_latest_hub_checkpoint
from lerobot.utils.local_assets import resolve_local_model_path


def _patch_list_files(monkeypatch, files):
    api = MagicMock()
    api.list_repo_files.return_value = files
    # HfApi is imported into lerobot.utils.hub at module load, so patch it there.
    monkeypatch.setattr("lerobot.utils.hub.HfApi", lambda *a, **k: api)
    return api


def test_find_latest_hub_checkpoint_picks_highest_step(monkeypatch):
    _patch_list_files(
        monkeypatch,
        [
            "README.md",
            "checkpoints/000500/pretrained_model/model.safetensors",
            "checkpoints/000500/training_state/training_step.json",
            "checkpoints/020000/pretrained_model/model.safetensors",
            "checkpoints/001000/training_state/training_step.json",
        ],
    )
    # Numeric max, not lexicographic — "020000" beats "001000"/"000500".
    assert find_latest_hub_checkpoint("u/run") == "checkpoints/020000"


def test_find_latest_hub_checkpoint_ignores_non_step_entries(monkeypatch):
    _patch_list_files(
        monkeypatch,
        ["checkpoints/last/pretrained_model/model.safetensors", "config.json"],
    )
    # "last" (a symlink target name) is not a numeric step → no resolvable checkpoint.
    assert find_latest_hub_checkpoint("u/run") is None


def test_find_latest_hub_checkpoint_none_when_no_checkpoints(monkeypatch):
    _patch_list_files(monkeypatch, ["config.json", "model.safetensors"])
    assert find_latest_hub_checkpoint("u/run") is None


def test_resolve_local_model_path_accepts_nested_download_layout(tmp_path, monkeypatch):
    model_dir = tmp_path / "base_models" / "pi05_base" / "pi05_base"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").touch()
    monkeypatch.setenv("LEROBOT_LOCAL_MODEL_ROOT", str(tmp_path / "base_models"))

    assert resolve_local_model_path("lerobot/pi05_base") == model_dir.resolve()


def test_resolve_local_model_path_ignores_incomplete_bundle(tmp_path, monkeypatch):
    model_dir = tmp_path / "base_models" / "pi05_base"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").touch()
    monkeypatch.setenv("LEROBOT_LOCAL_MODEL_ROOT", str(tmp_path / "base_models"))

    assert resolve_local_model_path("lerobot/pi05_base") is None


def test_resolve_local_model_path_supports_tokenizer_bundle(tmp_path, monkeypatch):
    tokenizer_dir = tmp_path / "base_models" / "paligemma-3b-pt-224"
    tokenizer_dir.mkdir(parents=True)
    required_files = ("config.json", "tokenizer_config.json", "tokenizer.model")
    for filename in required_files:
        (tokenizer_dir / filename).touch()
    monkeypatch.setenv("LEROBOT_LOCAL_MODEL_ROOT", str(tmp_path / "base_models"))

    assert (
        resolve_local_model_path(
            "google/paligemma-3b-pt-224",
            required_files=required_files,
        )
        == tokenizer_dir.resolve()
    )
