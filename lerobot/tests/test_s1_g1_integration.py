"""Regression tests for shared S1/G1 training and unchanged evaluation clients."""

import importlib
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.astribot_action import enable_astribot_s1_arm_gripper_filter
from lerobot.utils.dex3_action import enable_dex3_action_compression


def make_numeric_dataset(root, dim, robot_type, frames=6):
    ds = LeRobotDataset.create(
        repo_id="local/integration",
        root=root,
        fps=10,
        robot_type=robot_type,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": (dim,),
                "names": [f"joint_{i}" for i in range(dim)],
            },
            "observation.environment_state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
            "action": {"dtype": "float32", "shape": (dim,), "names": [f"joint_{i}" for i in range(dim)]},
        },
        use_videos=False,
    )
    for episode in range(2):
        for index in range(frames):
            values = np.arange(dim, dtype=np.float32) * 0.01 + index * 0.001
            ds.add_frame(
                {
                    "observation.state": values.copy(),
                    "action": values.copy(),
                    "observation.environment_state": np.array([index, episode], dtype=np.float32),
                    "task": "move the arms",
                }
            )
        ds.save_episode()
    ds.finalize()
    return root


@pytest.mark.parametrize(
    ("dim", "robot_type", "convert", "state_dim"),
    [
        (25, "astribot_s1", enable_astribot_s1_arm_gripper_filter, 16),
        (28, "unitree_g1_dex3", enable_dex3_action_compression, 28),
    ],
)
def test_real_dataset_dataloader_applies_action_views(tmp_path, dim, robot_type, convert, state_dim):
    root = make_numeric_dataset(tmp_path / "dataset", dim, robot_type)
    source = LeRobotDataset("local/integration", root=root, video_backend="pyav")
    transformed = convert(source)
    batch = next(iter(DataLoader(transformed, batch_size=2)))
    assert batch["action"].shape == (2, 16)
    assert batch["observation.state"].shape == (2, state_dim)
    assert tuple(transformed.meta.features["action"]["shape"]) == (16,)
    # Runtime views must leave the recorded schema and parquet columns intact.
    original = LeRobotDataset("local/integration", root=root, video_backend="pyav")
    assert original[0]["action"].shape == (dim,)
    assert tuple(original.meta.features["action"]["shape"]) == (dim,)


def test_multi_root_feature_selection_and_episode_indices(tmp_path):
    roots = [make_numeric_dataset(tmp_path / f"ds{i}", 25, "astribot_s1") for i in range(2)]
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="local/integration",
            root=[str(p) for p in roots],
            feature_names=["observation.state", "action"],
            use_imagenet_stats=False,
            video_backend="pyav",
        ),
        policy=ACTConfig(device="cpu", chunk_size=2, n_action_steps=2),
    )
    ds = make_dataset(cfg)
    assert len(ds) == 24
    assert ds.num_episodes == 4
    assert "observation.environment_state" not in ds[0]
    assert ds[12]["episode_index"].item() == 2
    transformed = enable_astribot_s1_arm_gripper_filter(ds)
    assert next(iter(DataLoader(transformed, batch_size=2)))["action"].shape == (2, 2, 16)


@pytest.mark.parametrize("module_name", ["pi0", "pi05"])
@pytest.mark.parametrize("flag", ["_eval_capture_attention_map", "_unitree_capture_attention_map"])
def test_eval_attention_flag_aliases(module_name, flag):
    pytest.importorskip("transformers")
    module = importlib.import_module(f"lerobot.policies.{module_name}.modeling_{module_name}")
    policy_cls = module.PI0Policy if module_name == "pi0" else module.PI05Policy
    policy = policy_cls.__new__(policy_cls)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        n_action_steps=1, rtc_config=None, use_visual_memory=False, use_proprioceptive_memory=False
    )
    policy._action_queue = deque()
    setattr(policy, flag, True)
    captured = {}
    if module_name == "pi05":

        def predict(batch, **kwargs):
            captured.update(kwargs)
            return torch.ones(1, 1, 16)

        policy.predict_action_chunk = predict
        assert policy.select_action({}).shape == (1, 16)
    else:
        # PI0 resolves the flags inside predict_action_chunk.
        policy._preprocess_images = lambda batch: ([], [])
        policy.prepare_state = lambda batch: torch.zeros(1, 25)
        policy.config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))}
        policy.config.image_features = {}

        def sample(*args, **kwargs):
            captured.update(kwargs)
            return torch.ones(1, 1, 16)

        policy.model = SimpleNamespace(sample_actions=sample)
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        policy.predict_action_chunk(
            {OBS_LANGUAGE_TOKENS: torch.ones(1, 2), OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2)}
        )
    assert captured["capture_attention_map"] is True


def test_pi05_missing_camera_keeps_checkpoint_order_and_legacy_padding():
    pytest.importorskip("transformers")
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    policy = PI05Policy.__new__(PI05Policy)
    nn.Module.__init__(policy)
    policy.anchor = nn.Parameter(torch.zeros(()))
    policy.config = PI05Config(
        device="cpu",
        image_resolution=(8, 8),
        legacy_resize_padding=True,
        input_features={
            "observation.images.left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 8)),
            "observation.images.right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 8)),
        },
    )
    images, masks = policy._preprocess_images({"observation.images.right": torch.full((1, 3, 4, 8), 0.5)})
    assert not masks[0].any() and masks[1].all()
    assert images[0].shape == images[1].shape == (1, 3, 8, 8)
    assert images[0].unique().tolist() == [-1.0]
    assert images[1].min() == -3.0


def test_pi05_words_ignore_and_memory_processor_roundtrip():
    pytest.importorskip("transformers")
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep

    step = Pi05PrepareStateTokenizerProcessorStep(
        words_ignore=["iced", "with ice"], include_state_in_prompt=False
    )
    assert step._remove_ignored_words("Pick ICED water, with ice.") == "Pick water,."
    restored = Pi05PrepareStateTokenizerProcessorStep(**step.get_config())
    assert not restored.include_state_in_prompt
    assert restored.words_ignore == step.words_ignore


def test_groot_embodiments_and_words_ignore():
    pytest.importorskip("transformers")
    from lerobot.policies.groot.processor_groot import N1_7_EMBODIMENT_MAPPING
    from lerobot.policies.groot.utils import prepare_n1_7_language_batch

    assert N1_7_EMBODIMENT_MAPPING["astribot_s1"] == 25
    assert N1_7_EMBODIMENT_MAPPING["real_g1_relative_eef_relative_joints"] == 25
    assert prepare_n1_7_language_batch(
        "Pick iced water", 1, formalize_language=False, words_ignore=["iced"]
    ) == ["Pick water"]


def test_legacy_types_are_the_same_objects():
    from lerobot import lerobot_types, types

    assert types.TransitionKey is lerobot_types.TransitionKey


def test_legacy_device_helper_import():
    from lerobot.utils.device_utils import get_safe_torch_device
    from lerobot.utils.utils import get_safe_torch_device as legacy_helper

    assert legacy_helper is get_safe_torch_device


@pytest.mark.parametrize("policy_type", ["pi0", "pi05"])
def test_old_nonthermal_checkpoint_config_loads(tmp_path, policy_type):
    import json

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy_config

    config = make_policy_config(policy_type, device="cpu", push_to_hub=False)
    config.save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    saved.update(thermal_encoder_channel=False, thermal_image_features=["observation.images.cam_thermal"])
    if policy_type == "pi0":
        saved.update(thermoact_temperature_min_c=20, thermoact_temperature_max_c=35)
    else:
        saved.update(
            thermal_input_type="false",
            thermal_encoder_dropout=0.0,
            freeze_non_thermal_input=False,
            use_thermal_encoder=None,
            share_rgb_thermal_vit=None,
            rgb_thermal_align_fusion=False,
        )
    path.write_text(json.dumps(saved))
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.type == policy_type
    assert restored.chunk_size == config.chunk_size
    saved["thermal_encoder_channel"] = "vit"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="removed S1/G1 thermal"):
        PreTrainedConfig.from_pretrained(tmp_path)


def test_old_training_config_with_inactive_thermal_fields_loads(tmp_path):
    import json

    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="local/s1"), policy=PI05Config(device="cpu"))
    cfg.save_pretrained(tmp_path)
    path = tmp_path / "train_config.json"
    saved = json.loads(path.read_text())
    saved["thermal_input_type"] = None
    saved["policy"].update(thermal_encoder_channel=False, thermal_encoder_dropout=0.0)
    path.write_text(json.dumps(saved))
    restored = TrainPipelineConfig.from_pretrained(tmp_path)
    assert restored.policy.type == "pi05"
    assert restored.dataset.repo_id == "local/s1"
