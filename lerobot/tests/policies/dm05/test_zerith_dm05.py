"""Tiny DM05 backward/save/reload with Zerith 23D state and 19D action."""


import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs import PreTrainedConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.dm05.configuration_dm05 import DM05Config
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_train import apply_robot_dataset_views
from lerobot.utils.zerith_action import CAMERAS, NAMES, ZerithActionLayout
from tests.policies.dm05.test_dm05_integration import _save_tiny_real_processor, _tiny_core_config


def _record_dataset(root, dim, robot):
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (dim,),
            "names": NAMES.copy(),
        },
        "action": {"dtype": "float32", "shape": (dim,), "names": NAMES.copy()},
        **{
            key: {"dtype": "image", "shape": (16, 24, 3), "names": ["height", "width", "channels"]}
            for key in CAMERAS
        },
    }
    ds = LeRobotDataset.create(
        "local/test", fps=10, features=features, root=root, robot_type=robot, use_videos=False
    )
    for frame in range(4):
        values = np.arange(dim, dtype=np.float32) * 0.01 + frame * 0.001
        action = values.copy()
        action[[7, 19, 20, 21, 22]] = 0
        ds.add_frame(
            {
                "observation.state": values,
                "action": action,
                "task": "pick",
                **{key: np.full((16, 24, 3), frame * 20, dtype=np.uint8) for key in CAMERAS},
            }
        )
    ds.save_episode()
    ds.finalize()


def test_dm05_zerith_training_and_eval_roundtrip(tmp_path):
    robot, dim, state_dim, action_dim = "zerith_h1_pro", 23, 23, 19
    root = tmp_path / "data"
    _record_dataset(root, dim, robot)
    processor_path = tmp_path / "processor"
    _save_tiny_real_processor(processor_path)
    core = _tiny_core_config()
    core.action_dim = 32
    config = DM05Config(
        core_config=core.to_dict(),
        processor_name_or_path=str(processor_path),
        device="cpu",
        dtype=torch.float32,
        chunk_size=2,
        n_action_steps=2,
        diffusion_steps=1,
        vlm_gradient_checkpointing=False,
        ae_gradient_checkpointing=False,
        freeze_vlm_embedding=False,
        llm_attn_implementation="eager",
        vision_attn_implementation="eager",
        action_attn_implementation="eager",
    )
    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="local/test",
            root=str(root),
            video_backend="pyav",
            use_imagenet_stats=False,
            zerith_action_view="auto",
        ),
        policy=config,
    )
    raw = make_dataset(train_cfg)
    dataset, held_out = apply_robot_dataset_views(train_cfg, raw, make_dataset(train_cfg))
    assert held_out.meta.features["action"]["shape"] == dataset.meta.features["action"]["shape"]
    policy = make_policy(config, ds_meta=dataset.meta)
    assert tuple(policy.config.input_features["observation.state"].shape) == (state_dim,)
    assert tuple(policy.config.output_features["action"].shape) == (action_dim,)
    pre, post = make_pre_post_processors(policy.config, dataset_stats=dataset.meta.stats)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    processed = pre(batch)
    loss, _ = policy(processed)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    optimizer.step()
    policy.eval()
    noise = torch.zeros(2, 2, 32)
    with torch.no_grad():
        expected = policy.predict_action_chunk(processed, noise=noise)
    checkpoint = tmp_path / "checkpoint"
    policy.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint, config_filename="policy_preprocessor.json")
    post.save_pretrained(checkpoint, config_filename="policy_postprocessor.json")
    dataset.layout.save(checkpoint / "zerith_action_layout.json")
    saved = PreTrainedConfig.from_pretrained(checkpoint)
    saved.pretrained_path = checkpoint
    # eval_robot reads the full-body reference recording, not the transformed training view.
    reloaded = make_policy(saved, ds_meta=raw.meta, preserve_checkpoint_features=True)
    assert tuple(reloaded.config.input_features["observation.state"].shape) == (state_dim,)
    assert tuple(reloaded.config.output_features["action"].shape) == (action_dim,)
    eval_pre, eval_post = make_pre_post_processors(reloaded.config, pretrained_path=checkpoint)
    with torch.no_grad():
        actual = reloaded.predict_action_chunk(eval_pre(batch), noise=noise)
        decoded = eval_post(actual)
    torch.testing.assert_close(actual, expected)
    assert decoded.shape == (2, 2, action_dim) and torch.isfinite(decoded).all()
    full = ZerithActionLayout.load(checkpoint / "zerith_action_layout.json").expand(decoded)
    assert full.shape == (2, 2, 23)
    assert torch.count_nonzero(full[..., 19:]) == 0
    assert 7 in dataset.layout.indices  # Keep the constant left gripper trainable.
    assert processed["action"].shape[-1] == 19 and torch.isfinite(processed["action"]).all()
