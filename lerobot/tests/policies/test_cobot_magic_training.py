"""Both policies train, save and reload with Cobot Magic's 14D/three-camera contract.

Only model sizes and tokenizer vocabulary are reduced; processors, image encoders,
losses, gradient updates and checkpoint IO run normally on CPU.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.cobot_magic import IMAGE_KEYS, JOINT_NAMES
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_train import apply_robot_dataset_views
from tests.policies.dm05.test_dm05_integration import _save_tiny_real_processor, _tiny_core_config


def record(root):
    features = {
        key: {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES}
        for key in ("observation.state", "action")
    }
    features.update(
        {
            key: {"dtype": "image", "shape": (16, 24, 3), "names": ["height", "width", "channels"]}
            for key in IMAGE_KEYS
        }
    )
    ds = LeRobotDataset.create(
        "local/test", fps=30, features=features, root=root, robot_type="cobot_magic", use_videos=False
    )
    for i in range(4):
        ds.add_frame(
            {
                "observation.state": np.full(14, i * 0.1, dtype=np.float32),
                "action": np.full(14, i * 0.2, dtype=np.float32),
                "task": "pick",
                **{key: np.full((16, 24, 3), i * 20, dtype=np.uint8) for key in IMAGE_KEYS},
            }
        )
    ds.save_episode()
    ds.finalize()


def tiny_pi05(monkeypatch, tokenizer):
    from lerobot.policies.pi05 import modeling_pi05 as model
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    monkeypatch.setattr(
        model,
        "get_gemma_config",
        lambda _: model.GemmaConfig(width=32, depth=1, mlp_dim=64, num_heads=8, num_kv_heads=1, head_dim=16),
    )
    original = model.PaliGemmaForConditionalGenerationWithPiGemma

    def tiny_vlm(config):
        config.text_config.vocab_size = 128
        config._vocab_size = 128
        config.image_token_index = 127
        config.vision_config.hidden_size = 32
        config.vision_config.intermediate_size = 64
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 2
        config.vision_config.patch_size = 8
        config.vision_config.projection_dim = 32
        config.projection_dim = 32
        return original(config)

    monkeypatch.setattr(model, "PaliGemmaForConditionalGenerationWithPiGemma", tiny_vlm)
    return PI05Config(
        device="cpu",
        dtype=torch.float32,
        chunk_size=2,
        n_action_steps=2,
        num_inference_steps=1,
        image_resolution=(16, 16),
        text_tokenizer_name=str(tokenizer),
        tokenizer_max_length=48,
    )


@pytest.mark.parametrize("policy_type", ["pi05", "dm05"])
def test_cobot_training_roundtrip(tmp_path, monkeypatch, policy_type):
    root = Path(os.environ["COBOT_DATASET_ROOT"]) if "COBOT_DATASET_ROOT" in os.environ else tmp_path / "data"
    if "COBOT_DATASET_ROOT" not in os.environ:
        record(root)
    processor = tmp_path / "processor"
    _save_tiny_real_processor(processor)
    if policy_type == "pi05":
        config = tiny_pi05(monkeypatch, processor)
    else:
        from lerobot.policies.dm05.configuration_dm05 import DM05Config

        core = _tiny_core_config()
        core.action_dim = 32
        config = DM05Config(
            core_config=core.to_dict(),
            processor_name_or_path=str(processor),
            device="cpu",
            dtype=torch.float32,
            chunk_size=2,
            n_action_steps=2,
            diffusion_steps=1,
            image_keys=IMAGE_KEYS,
            vlm_gradient_checkpointing=False,
            ae_gradient_checkpointing=False,
        )
    # A base checkpoint may describe a different robot and camera vocabulary.
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(32,)),
        "observation.images.base_camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
    }
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="local/test", root=str(root), video_backend="pyav"), policy=config
    )
    dataset, _ = apply_robot_dataset_views(cfg, make_dataset(cfg))
    policy = make_policy(config, ds_meta=dataset.meta)
    assert policy.config.action_feature_names == JOINT_NAMES
    assert set(policy.config.image_features) == set(IMAGE_KEYS)
    pre, post = make_pre_post_processors(policy.config, dataset_stats=dataset.meta.stats)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    processed = pre(batch)
    loss, _ = policy(processed)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    optimizer.step()
    policy.eval()
    noise = torch.zeros(1, 2, 32)
    with torch.no_grad():
        expected = policy.predict_action_chunk(processed, noise=noise)
    checkpoint = tmp_path / "checkpoint"
    policy.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint, config_filename="policy_preprocessor.json")
    post.save_pretrained(checkpoint, config_filename="policy_postprocessor.json")
    saved = PreTrainedConfig.from_pretrained(checkpoint)
    saved.pretrained_path = checkpoint
    reloaded = make_policy(saved, ds_meta=dataset.meta, preserve_checkpoint_features=True)
    eval_pre, eval_post = make_pre_post_processors(reloaded.config, pretrained_path=checkpoint)
    with torch.no_grad():
        actual = reloaded.predict_action_chunk(eval_pre(batch), noise=noise)
    torch.testing.assert_close(actual, expected)
    decoded = eval_post(actual)
    assert decoded.shape == (1, 2, 14) and torch.isfinite(decoded).all()
