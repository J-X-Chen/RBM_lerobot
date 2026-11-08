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

"""Resolve project-local model bundles without contacting a remote registry."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path


LOCAL_MODEL_ROOT_ENV = "LEROBOT_LOCAL_MODEL_ROOT"
DEFAULT_LOCAL_MODEL_DIRS = ("base_models", "pretrained_models")


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        normalized = path.expanduser().resolve(strict=False)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _search_roots() -> list[Path]:
    configured_root = os.getenv(LOCAL_MODEL_ROOT_ENV)
    if configured_root:
        return _unique_paths([Path(configured_root)])

    # Training is commonly launched from either the project root or the nested
    # LeRobot checkout. Search both that working-directory ancestry and the
    # installed source ancestry so editable installs behave consistently.
    roots: list[Path] = []
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for parent in (start, *start.parents):
            roots.extend(parent / dirname for dirname in DEFAULT_LOCAL_MODEL_DIRS)

    return _unique_paths(roots)


def _contains_required_files(path: Path, required_files: tuple[str, ...]) -> bool:
    return path.is_dir() and all((path / filename).is_file() for filename in required_files)


def resolve_local_model_path(
    model_name_or_path: str | Path,
    *,
    required_files: tuple[str, ...] = ("model.safetensors",),
) -> Path | None:
    """Return a complete local model directory when one is available.

    Hub-style identifiers such as ``lerobot/pi05_base`` are searched below
    project-local ``base_models`` and ``pretrained_models`` directories. One
    repeated basename level is accepted as some download tools materialize
    ``base_models/pi05_base/pi05_base``.

    This function performs filesystem checks only and never calls the Hub.
    """

    requested = Path(model_name_or_path).expanduser()
    direct_candidates = (requested, requested / requested.name)
    for candidate in direct_candidates:
        if _contains_required_files(candidate, required_files):
            return candidate.resolve()

    model_id = str(model_name_or_path).strip().rstrip("/")
    basename = model_id.rsplit("/", 1)[-1]
    aliases = tuple(dict.fromkeys((basename, model_id.replace("/", "--"))))
    for root in _search_roots():
        for alias in aliases:
            candidate = root / alias
            for resolved_candidate in (candidate, candidate / basename):
                if _contains_required_files(resolved_candidate, required_files):
                    return resolved_candidate.resolve()

    return None
