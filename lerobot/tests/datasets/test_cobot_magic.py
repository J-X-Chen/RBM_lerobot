"""Cobot numeric contract, episode isolation, and existing robot routing."""

from copy import deepcopy
from types import SimpleNamespace

import datasets
import numpy as np
import pytest

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.cobot_magic import JOINT_NAMES, prepare_cobot_magic_dataset
from lerobot.scripts.lerobot_train import apply_robot_dataset_views


def dataset(values, robot="cobot_magic"):
    features = {key: {"shape": (14,), "names": JOINT_NAMES.copy()} for key in ("action", "observation.state")}
    return SimpleNamespace(
        meta=SimpleNamespace(robot_type=robot, features=features, stats={"camera": {"mean": [0.5]}}),
        hf_dataset=datasets.Dataset.from_dict(dict.fromkeys(features, values)),
    )


def test_exact_stats_and_validation_use_training_frames_only():
    values = np.repeat(np.array([0.0, 0.0, 0.0, 40.0])[:, None], 14, axis=1)
    train, valid = dataset(values), dataset(values + 1000)
    features = deepcopy(train.meta.features)
    prepare_cobot_magic_dataset(train, valid)
    np.testing.assert_allclose(train.meta.stats["action"]["q99"], 38.8)
    np.testing.assert_allclose(valid.meta.stats["action"]["q99"], 38.8)
    assert train.meta.features == features
    assert train.meta.stats["camera"] == {"mean": [0.5]}
    assert train.meta.stats is not valid.meta.stats


@pytest.mark.parametrize("error", ["names", "shape", "nan"])
def test_invalid_contract_fails(error):
    ds = dataset(np.zeros((3, 14)))
    if error == "nan":
        ds = dataset(np.full((3, 14), np.nan))
    else:
        ds.meta.features["action"][error] = []
    with pytest.raises(ValueError):
        prepare_cobot_magic_dataset(ds)


def test_cobot_default_routing_keeps_native_actions():
    ds = dataset(np.zeros((3, 14)))
    cfg = SimpleNamespace(dataset=DatasetConfig(repo_id="local/test"), policy=None)
    train, valid = apply_robot_dataset_views(cfg, ds, ds)
    assert train is ds and valid is ds
    assert train.meta.features["action"]["shape"] == (14,)


@pytest.mark.parametrize(
    "options",
    [
        {"compress_dex3_actions": True},
        {"compress_dex3_actions": "ignore"},
        {"astribot_action_view": "arms_grippers"},
        {"streaming": True},
    ],
)
def test_cobot_rejects_other_robot_compression(options):
    cfg = SimpleNamespace(dataset=DatasetConfig(repo_id="local/test", **options), policy=None)
    with pytest.raises((ValueError, NotImplementedError)):
        apply_robot_dataset_views(cfg, dataset(np.zeros((3, 14))))


def test_exact_stats_respect_real_episode_selection(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "data"
    features = {
        key: {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES}
        for key in ("observation.state", "action")
    }
    recorded = LeRobotDataset.create(
        "local/test", fps=30, features=features, root=root, robot_type="cobot_magic", use_videos=False
    )
    for value in (0.0, 100.0):
        for i in range(3):
            recorded.add_frame(
                {**{key: np.full(14, value + i, dtype=np.float32) for key in features}, "task": "pick"}
            )
        recorded.save_episode()
    recorded.finalize()
    selected = LeRobotDataset("local/test", root=root, episodes=[1], video_backend="pyav")
    prepare_cobot_magic_dataset(selected)
    assert selected.meta.stats["action"]["count"].tolist() == [3]
    np.testing.assert_allclose(selected.meta.stats["action"]["mean"], 101.0)
