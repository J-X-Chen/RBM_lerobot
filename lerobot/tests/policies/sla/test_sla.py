"""SLA (pi0.5 + spatial alignment) unit tests.

The model is shrunk the same way as ``tests/policies/test_cobot_magic_training.py`` and the 1B-param
OmniVGGT teacher is replaced by a deterministic stub, so these run on CPU without downloads.
"""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from lerobot.configs import PreTrainedConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_policy, make_pre_post_processors
from lerobot.policies.sla.alignment import (
    AlignProjector,
    make_align_mask,
    pool_vggt_features,
    preprocess_images_for_vggt,
    resize_with_pad_valid_mask,
)
from lerobot.policies.sla.configuration_sla import SLAConfig
from lerobot.policies.sla.processor_sla import (
    OBS_LANGUAGE_ATTENTION_MASK_WITHOUT_STATE,
    OBS_LANGUAGE_TOKENS_WITHOUT_STATE,
)
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from tests.policies.dm05.test_dm05_integration import _save_tiny_real_processor

CAMERAS = ["observation.images.head", "observation.images.left"]
VGGT_DIM = 8


def test_policy_is_registered():
    assert get_policy_class("sla").__name__ == "SLAPolicy"
    assert isinstance(PreTrainedConfig.get_choice_class("sla")(), SLAConfig)


def test_resize_with_pad_mask_matches_openpi_geometry():
    mask = resize_with_pad_valid_mask(480, 640, 224, 224, "cpu")
    # 640x480 -> 224x168, padded with 28 rows on top and bottom.
    assert mask[:28].sum() == 0 and mask[196:].sum() == 0
    assert mask[28:196].all()
    token_mask = make_align_mask([torch.ones(2, dtype=torch.bool)], [mask[None].expand(2, -1, -1)], 256, 16)
    rows = token_mask[0].view(16, 16).all(dim=1)
    assert rows.tolist() == [False, False] + [True] * 12 + [False, False]


def test_missing_camera_tokens_are_masked():
    mask = make_align_mask([torch.tensor([True, False]), torch.tensor([True, True])], None, 4, 2)
    assert mask.tolist() == [[True] * 8, [False] * 4 + [True] * 4]


def test_vggt_preprocessing_and_pooling_shapes():
    images = preprocess_images_for_vggt([torch.rand(2, 3, 224, 224)] * 3)
    assert images.shape == (2, 3, 3, 518, 518)
    pooled = pool_vggt_features(torch.randn(2, 3, 37 * 37, 16), (37, 37), (518, 518), 256)
    assert pooled.shape == (2, 3 * 256, 16)


def test_align_projector_loss():
    torch.manual_seed(0)
    projector = AlignProjector(llm_dim=8, vggt_dim=4)
    hidden = torch.randn(2, 6, 8)
    with torch.no_grad():
        target = projector.project(hidden)
    mask = torch.ones(2, 6, dtype=torch.bool)
    torch.testing.assert_close(projector(hidden, target, mask), torch.zeros(2), atol=1e-5, rtol=0)
    # Masked tokens do not contribute.
    corrupted = target.clone()
    corrupted[:, 3:] = -corrupted[:, 3:]
    mask[:, 3:] = False
    torch.testing.assert_close(projector(hidden, corrupted, mask), torch.zeros(2), atol=1e-5, rtol=0)
    mask[:, 3:] = True
    assert (projector(hidden, corrupted, mask) > 0.9).all()


class StubTeacher(nn.Module):
    """Stands in for OmniVGGT: deterministic features derived from the input pixels."""

    patch_size = 14

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 2 * VGGT_DIM)
        self.calls = 0

    def forward(self, images, *, feature_layer=-1):
        self.calls += 1
        bsize, num_images, _, height, width = images.shape
        patches = nn.functional.avg_pool2d(images.flatten(0, 1), self.patch_size)  # [BN, 3, 37, 37]
        tokens = self.proj(patches.flatten(2).transpose(1, 2)).view(bsize, num_images, -1, 2 * VGGT_DIM)
        special = torch.zeros(bsize, num_images, 5, 2 * VGGT_DIM)
        return [torch.cat([special, tokens], dim=2)], 5


def _record_dataset(root):
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
        "action": {"dtype": "float32", "shape": (6,), "names": [f"j{i}" for i in range(6)]},
    }
    features.update(
        {
            key: {"dtype": "image", "shape": (12, 16, 3), "names": ["height", "width", "channels"]}
            for key in CAMERAS
        }
    )
    ds = LeRobotDataset.create("local/sla_test", fps=30, features=features, root=root, use_videos=False)
    rng = np.random.default_rng(0)
    for i in range(4):
        ds.add_frame(
            {
                "observation.state": np.full(6, i * 0.1, dtype=np.float32),
                "action": np.full(6, i * 0.2, dtype=np.float32),
                "task": "pick",
                **{key: rng.integers(0, 255, (12, 16, 3), dtype=np.uint8) for key in CAMERAS},
            }
        )
    ds.save_episode()
    ds.finalize()


def _tiny_sla_config(monkeypatch, tokenizer, **overrides) -> SLAConfig:
    from lerobot.policies.pi05 import modeling_pi05 as model

    monkeypatch.setattr(
        model,
        "get_gemma_config",
        lambda _: model.GemmaConfig(width=32, depth=2, mlp_dim=64, num_heads=8, num_kv_heads=1, head_dim=16),
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
    kwargs = {
        "device": "cpu",
        "dtype": torch.float32,
        "chunk_size": 2,
        "n_action_steps": 2,
        "num_inference_steps": 1,
        "image_resolution": (16, 16),
        "text_tokenizer_name": str(tokenizer),
        "tokenizer_max_length": 48,
        "vla_layers_align": 0,
        "vggt_dim": VGGT_DIM,
    }
    kwargs.update(overrides)
    return SLAConfig(**kwargs)


@pytest.fixture
def tiny_setup(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _record_dataset(root)
    tokenizer = tmp_path / "tokenizer"
    _save_tiny_real_processor(tokenizer)
    config = _tiny_sla_config(monkeypatch, tokenizer)
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="local/sla_test", root=str(root)), policy=config)
    dataset = make_dataset(cfg)
    if isinstance(dataset, tuple):
        dataset = dataset[0]
    policy = make_policy(config, ds_meta=dataset.meta)
    teacher = StubTeacher()
    policy.model.__dict__["_vggt_teacher"] = teacher
    pre, post = make_pre_post_processors(policy.config, dataset_stats=dataset.meta.stats)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    return policy, teacher, pre, post, batch, dataset


def test_processor_emits_stateless_prompt(tiny_setup):
    _, _, pre, _, batch, _ = tiny_setup
    processed = pre(batch)
    assert processed[OBS_LANGUAGE_TOKENS_WITHOUT_STATE].shape == processed[OBS_LANGUAGE_TOKENS].shape
    with_state = processed[OBS_LANGUAGE_ATTENTION_MASK].sum(-1)
    without_state = processed[OBS_LANGUAGE_ATTENTION_MASK_WITHOUT_STATE].sum(-1)
    assert (without_state < with_state).all()


def test_training_step_trains_projector_and_vlm(tiny_setup):
    policy, teacher, pre, _, batch, _ = tiny_setup
    policy.train()
    loss, info = policy(pre(batch))
    assert torch.isfinite(loss) and teacher.calls == 1
    assert {"action_loss", "align_loss"} <= info.keys()
    assert loss.item() == pytest.approx(info["action_loss"] + 0.5 * info["align_loss"], rel=1e-5)
    loss.backward()
    projector_grad = policy.model.align_projector.fc1.weight.grad
    assert projector_grad is not None and projector_grad.abs().sum() > 0
    vlm_layer0 = policy.model.paligemma_with_expert.paligemma.model.language_model.layers[0]
    assert vlm_layer0.mlp.up_proj.weight.grad.abs().sum() > 0
    # The teacher is not a registered submodule: it never reaches optimizers or checkpoints.
    assert not any(name.startswith("model._vggt_teacher") for name, _ in policy.named_parameters())
    assert not any("_vggt_teacher" in key for key in policy.state_dict())


def test_eval_mode_skips_teacher_and_state_dropout(tiny_setup):
    policy, teacher, pre, _, batch, _ = tiny_setup
    policy.eval()
    processed = pre(batch)
    tokens, _ = policy._select_training_prompt(processed)
    assert torch.equal(tokens, processed[OBS_LANGUAGE_TOKENS])
    with torch.no_grad():
        _, info = policy(processed)
    assert teacher.calls == 0 and "align_loss" not in info


def test_state_dropout_swaps_prompts(tiny_setup):
    policy, _, pre, _, batch, _ = tiny_setup
    policy.train()
    policy.config.state_dropout_prob = 1.0
    processed = pre(batch)
    tokens, masks = policy._select_training_prompt(processed)
    assert torch.equal(tokens, processed[OBS_LANGUAGE_TOKENS_WITHOUT_STATE])
    assert torch.equal(masks, processed[OBS_LANGUAGE_ATTENTION_MASK_WITHOUT_STATE])


def test_checkpoint_roundtrip(tiny_setup, tmp_path):
    policy, _, pre, post, batch, dataset = tiny_setup
    policy.eval()
    processed = pre(batch)
    noise = torch.zeros(2, 2, 32)
    with torch.no_grad():
        expected = policy.predict_action_chunk(processed, noise=noise)
    checkpoint = tmp_path / "checkpoint"
    policy.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint, config_filename="policy_preprocessor.json")
    post.save_pretrained(checkpoint, config_filename="policy_postprocessor.json")
    saved = PreTrainedConfig.from_pretrained(checkpoint)
    assert saved.type == "sla"
    saved.pretrained_path = checkpoint
    reloaded = make_policy(saved, ds_meta=dataset.meta, preserve_checkpoint_features=True)
    torch.testing.assert_close(
        reloaded.model.align_projector.fc1.weight, policy.model.align_projector.fc1.weight
    )
    eval_pre, eval_post = make_pre_post_processors(reloaded.config, pretrained_path=checkpoint)
    with torch.no_grad():
        actual = reloaded.predict_action_chunk(eval_pre(batch), noise=noise)
    torch.testing.assert_close(actual, expected)
    assert eval_post(actual).shape == (2, 2, 6)


def test_pi05_checkpoint_initializes_projector(tiny_setup, tmp_path):
    """A pi0.5 checkpoint (no align projector) loads into SLA with a freshly initialized projector."""
    from safetensors.torch import save_file

    policy, _, _, _, _, _ = tiny_setup
    state = {
        key.removeprefix("model."): value.contiguous()
        for key, value in policy.state_dict().items()
        if ".align_projector." not in f".{key}" and "gemma_expert.lm_head" not in key
    }
    checkpoint = tmp_path / "pi05_like"
    checkpoint.mkdir()
    save_file(state, str(checkpoint / "model.safetensors"))
    loaded = type(policy).from_pretrained(checkpoint, config=policy.config)
    torch.testing.assert_close(loaded.model.action_out_proj.weight, policy.model.action_out_proj.weight)
