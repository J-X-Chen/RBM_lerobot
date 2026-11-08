"""Zerith H1 data contract and lossless removal of constant action columns."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch

NAMES = [f"left_arm_{i}" for i in range(1, 8)] + ["left_gripper"] + [
    f"right_arm_{i}" for i in range(1, 8)
] + ["right_gripper", "waist_lift", "waist_pitch", "waist_yaw", "head_yaw",
     "head_pitch", "base_linear_speed", "base_angular_speed"]
CAMERAS = ["observation.images.cam_high", "observation.images.cam_left_wrist",
           "observation.images.cam_right_wrist"]


@dataclass
class ZerithActionLayout:
    names: list[str]
    indices: list[int]
    reference: list[float]

    @classmethod
    def fit(cls, actions, names=NAMES):
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[1] != 23 or len(actions) == 0:
            raise ValueError("Expected nonempty [frames, 23] Zerith actions")
        if list(names) != NAMES or not np.isfinite(actions).all():
            raise ValueError("Invalid Zerith joint names or nonfinite actions")
        indices = np.flatnonzero(np.ptp(actions, axis=0) != 0).tolist()
        if not indices:
            raise ValueError("All actions are constant; no trainable action remains")
        # Keep the left gripper in the policy interface even when its target is constant.
        indices = sorted(set(indices) | {NAMES.index("left_gripper")})
        return cls(list(names), indices, actions[0].tolist())

    def select(self, value):
        if value.shape[-1] != len(self.names):
            raise ValueError("Expected full 23D action")
        return value[..., self.indices]

    def expand(self, value):
        if value.shape[-1] != len(self.indices):
            raise ValueError("Policy action dimension does not match saved Zerith layout")
        if isinstance(value, torch.Tensor):
            result = value.new_tensor(self.reference).expand(*value.shape[:-1], 23).clone()
        else:
            value = np.asarray(value)
            result = np.broadcast_to(np.asarray(self.reference, dtype=value.dtype), (*value.shape[:-1], 23)).copy()
        result[..., self.indices] = value
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        layout = cls(**json.loads(Path(path).read_text()))
        if (layout.names != NAMES or len(layout.reference) != 23 or not layout.indices
                or layout.indices != sorted(set(layout.indices))
                or any(i < 0 or i >= 23 for i in layout.indices)
                or not np.isfinite(layout.reference).all()):
            raise ValueError("Invalid Zerith action layout")
        return layout


def numeric_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"min": values.min(0), "max": values.max(0), "mean": values.mean(0),
            "std": values.std(0), "count": np.array([len(values)]),
            **{f"q{q:02d}": np.quantile(values, q / 100, axis=0) for q in (1, 10, 50, 90, 99)}}


class ZerithDataset(torch.utils.data.Dataset):
    """23D state, variable action; only in-memory metadata is changed."""
    def __init__(self, dataset, layout, stats):
        self.dataset = dataset
        self.layout = layout
        self.meta = deepcopy(dataset.meta)
        self.meta.info.features["action"]["shape"] = [len(layout.indices)]
        self.meta.info.features["action"]["names"] = [layout.names[i] for i in layout.indices]
        self.meta.stats.update(deepcopy(stats))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        item["action"] = self.layout.select(item["action"])
        return item

    def __getitems__(self, indices):
        # DataLoader batch fetching must not bypass the action projection.
        return [self[index] for index in indices]

    def __getattr__(self, name):
        # Avoid recursive access during DataLoader worker unpickling.
        if name in {"dataset", "meta", "layout"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)


def _arrays(dataset):
    if dataset.meta.robot_type != "zerith_h1_pro":
        raise ValueError("Zerith filtering requires robot_type=zerith_h1_pro")
    for key in ("action", "observation.state"):
        feature = dataset.meta.features[key]
        if list(feature["shape"]) != [23] or feature["names"] != NAMES:
            raise ValueError(f"Unexpected Zerith {key} layout")
    table = dataset.hf_dataset.with_format("numpy")
    actions = np.asarray(table["action"], dtype=np.float32)
    states = np.asarray(table["observation.state"], dtype=np.float32)
    if not np.isfinite(states).all():
        raise ValueError("Nonfinite Zerith state")
    return actions, states


def prepare_zerith_datasets(dataset, eval_dataset=None, layout=None):
    actions, states = _arrays(dataset)
    layout = layout or ZerithActionLayout.fit(actions)
    dropped = sorted(set(range(23)) - set(layout.indices))
    if not np.isfinite(actions).all() or not np.all(actions[:, dropped] == np.asarray(layout.reference)[dropped]):
        raise ValueError("Actions differ from saved constant values")
    stats = {"action": numeric_stats(layout.select(actions)), "observation.state": numeric_stats(states)}
    if eval_dataset is not None:
        eval_actions, _ = _arrays(eval_dataset)
        if (not np.isfinite(eval_actions).all()
                or not np.all(eval_actions[:, dropped] == np.asarray(layout.reference)[dropped])):
            raise ValueError("Held-out actions vary in columns constant in training; use zerith_action_view=full")
        eval_dataset = ZerithDataset(eval_dataset, layout, stats)
    return ZerithDataset(dataset, layout, stats), eval_dataset, layout
