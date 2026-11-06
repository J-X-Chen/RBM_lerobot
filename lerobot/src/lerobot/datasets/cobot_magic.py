"""Training contract for the 14-DoF Cobot Magic recordings.

Keep recorded joint units and order. Only normalization metadata is replaced,
in memory, using the selected training episodes (never held-out episodes).
"""

import logging
from copy import deepcopy
from typing import Any

import numpy as np

JOINT_NAMES = [
    f"{side}.{joint}"
    for side in ("left", "right")
    for joint in (*[f"joint{i}" for i in range(1, 7)], "gripper")
]
IMAGE_KEYS = [f"observation.images.{camera}" for camera in ("head", "left", "right")]


def is_cobot_magic(dataset: Any) -> bool:
    return getattr(getattr(dataset, "meta", None), "robot_type", None) == "cobot_magic"


def validate_cobot_magic_features(features: dict) -> None:
    for key in ("observation.state", "action"):
        feature = features.get(key, {})
        if tuple(feature.get("shape", ())) != (14,):
            raise ValueError(f"Cobot Magic requires {key} with shape (14,).")
        if feature.get("names") != JOINT_NAMES:
            raise ValueError(f"Cobot Magic {key} joint order must be {JOINT_NAMES}.")


def prepare_cobot_magic_dataset(dataset: Any, eval_dataset: Any | None = None) -> None:
    """Validate the contract and compute exact numeric stats without decoding videos.

    Old exported statistics can contain averaged episode quantiles, which are not
    global quantiles. Reading only two numeric columns is inexpensive for these
    recordings. The underlying HF table already respects episode selection.
    """
    validate_cobot_magic_features(dataset.meta.features)
    if eval_dataset is not None:
        validate_cobot_magic_features(eval_dataset.meta.features)
    table = dataset.hf_dataset.select_columns(["observation.state", "action"]).with_format("numpy")
    stats = deepcopy(dataset.meta.stats)
    for key in ("observation.state", "action"):
        values = np.asarray(table[key], dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 14 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Cobot Magic {key} must contain finite, nonempty 14D vectors.")
        stats[key] = {
            "min": values.min(axis=0),
            "max": values.max(axis=0),
            "mean": values.mean(axis=0),
            "std": values.std(axis=0),
            "count": np.array([len(values)]),
        }
        for q in (0.01, 0.10, 0.50, 0.90, 0.99):
            stats[key][f"q{round(q * 100):02d}"] = np.quantile(values, q, axis=0)
    dataset.meta.stats = stats
    if eval_dataset is not None:
        eval_dataset.meta.stats = deepcopy(stats)
    logging.info(
        "Cobot Magic: retained 14D joint targets; exact numeric statistics from %d training frames.",
        len(table),
    )
