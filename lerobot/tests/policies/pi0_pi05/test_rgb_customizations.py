"""Nonthermal RGB customizations retained from the S1/G1 forks."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import make_policy_config, make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
from lerobot.policies.pi05.rgb_encoder import (
    DEFAULT_RGB_DINOV2_CHECKPOINT_PATH,
    DINOV2_IMAGENET_MEAN,
    DINOV2_IMAGENET_STD,
    DINOv2RGBEncoder,
)
from lerobot.policies.pi05.rgb_threeblack import (
    RGBThreeBlackGateThreeResViTFusion,
    rgb_to_three_black_dominance_tensor,
)
from lerobot.processor import PolicyProcessorPipeline

RGB_FEATURE = "observation.images.cam_left_high"
WRIST_LEFT_FEATURE = "observation.images.cam_left_wrist"
WRIST_RIGHT_FEATURE = "observation.images.cam_right_wrist"


def test_rgb_input_type_dinov2_alias_configures_pi05(tmp_path):
    policy = PI05Config(push_to_hub=False)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        policy=policy,
        rgb_input_type="'dinov2'",
        output_dir=tmp_path / "run",
    )

    cfg.validate()

    assert policy.rgb_input_type == "dinov2"
    assert policy.uses_dinov2_rgb_encoder
    assert policy.rgb_dinov2_checkpoint_path == DEFAULT_RGB_DINOV2_CHECKPOINT_PATH


def test_policy_rgb_gatetworesvit_enables_threeblack_preprocessing():
    config = PI05Config(
        push_to_hub=False,
        rgb_input_type="'GateTwoResViT'",
    )

    assert config.rgb_input_type == "gatetworesvit"
    assert config.rgb_threeblack_input_type == "withthreeblack"
    assert config.uses_gate_two_residual_rgb_vit
    assert not config.uses_dinov2_rgb_encoder


def test_policy_rgb_gatetworesvit_rejects_mismatched_language_embedding_dimension():
    with pytest.raises(ValueError, match="requires paligemma_variant='gemma_2b'"):
        PI05Config(
            push_to_hub=False,
            paligemma_variant="gemma_300m",
            rgb_input_type="GateTwoResViT",
        )


@pytest.mark.parametrize("policy_rgb_input_type", ["vit", "'GateTwoResViT'"])
def test_rgb_input_type_withthreeblack_alias_selects_rgb_gatetworesvit(
    tmp_path,
    policy_rgb_input_type,
):
    policy = PI05Config(
        push_to_hub=False,
        rgb_input_type=policy_rgb_input_type,
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        policy=policy,
        rgb_input_type="'withthreeblack'",
        output_dir=tmp_path / "run_rgb_threeblack",
    )

    cfg.validate()

    assert policy.rgb_input_type == "gatetworesvit"
    assert policy.rgb_threeblack_input_type == "withthreeblack"
    assert policy.uses_gate_two_residual_rgb_vit


def test_rgb_threeblack_dominance_views_map_selected_color_to_black():
    images = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    ).reshape(4, 3, 1, 1)

    red, green, blue = rgb_to_three_black_dominance_tensor(
        images,
        feature_name=RGB_FEATURE,
        input_range="zero_to_one",
    )

    torch.testing.assert_close(red[:, 0, 0, 0], torch.tensor([0.0, 1.0, 1.0, 1.0]))
    torch.testing.assert_close(green[:, 0, 0, 0], torch.tensor([1.0, 0.0, 1.0, 1.0]))
    torch.testing.assert_close(blue[:, 0, 0, 0], torch.tensor([1.0, 1.0, 0.0, 1.0]))
    torch.testing.assert_close(red[:, 0], red[:, 1])
    torch.testing.assert_close(red[:, 1], red[:, 2])


def test_rgb_gatetworesvit_replaces_only_head_camera_with_three_residual_slots():
    class DummyLanguageModel:
        def embed_language_tokens(self, tokens):
            return torch.zeros(tokens.shape[0], tokens.shape[1], 8)

    class DummyModel:
        _apply_checkpoint = PI05Pytorch._apply_checkpoint

        def __init__(self):
            self.config = SimpleNamespace(
                image_features={
                    RGB_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    WRIST_LEFT_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    WRIST_RIGHT_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                },
                uses_resvit_attention_encoder=False,
            )
            self.gradient_checkpointing_enabled = False
            self.training = False
            self.rgb_calls = 0
            self.rgb_threeblack_calls = 0
            self.rgb_encoder = None
            self.rgb_threeblack_gatetworesvit_fusion = object()
            self.rgb_threeblack_gatetworesvit_head_feature = RGB_FEATURE
            self.resvit_attention_fusion = None
            self.paligemma_with_expert = DummyLanguageModel()
            self._last_prefix_token_layout = []
            self._last_rgb_threeblack_gatetworesvit_gate_summary = None

        def _embed_rgb_image(self, image):
            self.rgb_calls += 1
            return torch.full((image.shape[0], 4, 8), 10.0 + self.rgb_calls)

        def _embed_rgb_threeblack_gatetworesvit_head_images(self, **kwargs):
            self.rgb_threeblack_calls += 1
            assert kwargs["text_tokens"].shape == (2, 3, 8)
            rgb_img = kwargs["rgb_img"]
            token_mask = kwargs["rgb_img_mask"][:, None].expand(rgb_img.shape[0], 4)
            self._last_rgb_threeblack_gatetworesvit_gate_summary = {
                "red_alpha": {"mean": 1.0},
                "green_alpha": {"mean": 1.0},
                "blue_alpha": {"mean": 1.0},
            }
            return (
                torch.ones(rgb_img.shape[0], 4, 8),
                token_mask,
                2.0 * torch.ones(rgb_img.shape[0], 4, 8),
                token_mask,
                3.0 * torch.ones(rgb_img.shape[0], 4, 8),
                token_mask,
            )

    model = DummyModel()
    prefix_embs, _, _ = PI05Pytorch.embed_prefix(
        model,
        images=[torch.zeros(2, 3, 8, 8) for _ in range(3)],
        img_masks=[torch.ones(2, dtype=torch.bool) for _ in range(3)],
        tokens=torch.zeros(2, 3, dtype=torch.long),
        masks=torch.ones(2, 3, dtype=torch.bool),
    )

    assert model.rgb_threeblack_calls == 1
    assert model.rgb_calls == 2
    image_layout = [item for item in model._last_prefix_token_layout if item["kind"].startswith("image")]
    assert [item["kind"] for item in image_layout] == [
        "image_rgb_gatetworesvit_red_residual",
        "image_rgb_gatetworesvit_green_residual",
        "image_rgb_gatetworesvit_blue_residual",
        "image",
        "image",
    ]
    assert [item["name"] for item in image_layout] == [
        RGB_FEATURE,
        RGB_FEATURE,
        RGB_FEATURE,
        WRIST_LEFT_FEATURE,
        WRIST_RIGHT_FEATURE,
    ]
    torch.testing.assert_close(prefix_embs[:, :4], torch.ones(2, 4, 8))
    torch.testing.assert_close(prefix_embs[:, 4:8], 2.0 * torch.ones(2, 4, 8))
    torch.testing.assert_close(prefix_embs[:, 8:12], 3.0 * torch.ones(2, 4, 8))
    assert image_layout[0]["gate"]["red_alpha"]["mean"] == 1.0
    assert image_layout[1]["gate"] is None
    assert image_layout[2]["gate"] is None


def test_rgb_threeblack_gatetworesvit_fusion_emits_three_rgb_residual_streams():
    class ZeroGate(nn.Module):
        def forward(self, gate_input):
            return torch.zeros(
                gate_input.shape[0],
                3,
                device=gate_input.device,
                dtype=gate_input.dtype,
            )

    fusion = RGBThreeBlackGateThreeResViTFusion(
        embed_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.0,
    )
    fusion.gate_mlp = ZeroGate()

    token_mask = torch.ones(1, 4, dtype=torch.bool)
    rgb_tokens = torch.full((1, 4, 4), 10.0)
    red_tokens = torch.ones(1, 4, 4)
    green_tokens = 2.0 * torch.ones(1, 4, 4)
    blue_tokens = 3.0 * torch.ones(1, 4, 4)
    (
        red_res,
        red_res_mask,
        green_res,
        green_res_mask,
        blue_res,
        blue_res_mask,
        gate_info,
    ) = fusion(
        rgb_tokens=rgb_tokens,
        red_tokens=red_tokens,
        green_tokens=green_tokens,
        blue_tokens=blue_tokens,
        text_tokens=torch.randn(1, 3, 4),
        rgb_token_mask=token_mask,
        red_token_mask=token_mask,
        green_token_mask=token_mask,
        blue_token_mask=token_mask,
        text_token_mask=torch.ones(1, 3, dtype=torch.bool),
        base_red_alpha=0.5,
        base_green_alpha=0.25,
        base_blue_alpha=0.1,
    )

    assert red_res_mask.all()
    assert green_res_mask.all()
    assert blue_res_mask.all()
    torch.testing.assert_close(gate_info["red_multiplier"], torch.tensor([1.0]))
    torch.testing.assert_close(gate_info["green_multiplier"], torch.tensor([1.0]))
    torch.testing.assert_close(gate_info["blue_multiplier"], torch.tensor([1.0]))
    torch.testing.assert_close(red_res, torch.full_like(rgb_tokens, 10.5))
    torch.testing.assert_close(green_res, torch.full_like(rgb_tokens, 10.5))
    torch.testing.assert_close(blue_res, torch.full_like(rgb_tokens, 10.3))


def test_rgb_dinov2_preprocess_matches_imagenet_normalization():
    encoder = DINOv2RGBEncoder.__new__(DINOv2RGBEncoder)
    nn.Module.__init__(encoder)
    encoder.input_range = "minus_one_to_one"
    encoder.patch_size = 14

    image = torch.linspace(-1.0, 1.0, 3 * 14 * 14).view(1, 3, 14, 14)
    processed = encoder._preprocess_for_dinov2(image)

    zero_to_one = (image + 1.0) * 0.5
    mean = torch.tensor(DINOV2_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(DINOV2_IMAGENET_STD).view(1, 3, 1, 1)
    expected = (zero_to_one - mean) / std
    torch.testing.assert_close(processed, expected)


def test_pi05_incompatible_pretrained_processors_fall_back(monkeypatch, caplog):
    policy_cfg = make_policy_config("pi05")
    dataset_stats = {"sentinel": {}}
    expected_processors = (object(), object())

    def raise_incompatible_processor(*args, **kwargs):
        raise ImportError("relative_actions_processor is not registered")

    monkeypatch.setattr(PolicyProcessorPipeline, "from_pretrained", raise_incompatible_processor)

    from lerobot.policies.pi05 import processor_pi05

    monkeypatch.setattr(
        processor_pi05,
        "make_pi05_pre_post_processors",
        lambda config, dataset_stats: expected_processors,
    )

    processors = make_pre_post_processors(
        policy_cfg,
        pretrained_path="lerobot/pi05_base",
        dataset_stats=dataset_stats,
    )

    assert processors is expected_processors
    assert "missing or incompatible policy processor files" in caplog.text
