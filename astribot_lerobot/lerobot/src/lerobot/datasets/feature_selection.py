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

"""Runtime feature selection without changing dataset files on disk."""

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.utils.constants import DEFAULT_FEATURES


def filter_dataset_metadata_features(
    meta: LeRobotDatasetMetadata,
    feature_names: list[str] | None,
) -> None:
    """Restrict in-memory metadata to selected features plus bookkeeping fields."""
    if feature_names is None:
        return

    selected = set(feature_names)
    available = set(meta.features)
    missing = selected - available
    if missing:
        raise ValueError(
            "Selected dataset feature(s) are not present in metadata: "
            f"{sorted(missing)}. Available features: {sorted(available)}"
        )

    keep = {key for key in DEFAULT_FEATURES if key in available}
    keep.update(selected)
    ordered_features = {
        key: meta.features[key]
        for key in [*DEFAULT_FEATURES, *feature_names]
        if key in keep and key in meta.features
    }

    meta.info.features = ordered_features
    if meta.stats is not None:
        meta.stats = {key: value for key, value in meta.stats.items() if key in keep}
