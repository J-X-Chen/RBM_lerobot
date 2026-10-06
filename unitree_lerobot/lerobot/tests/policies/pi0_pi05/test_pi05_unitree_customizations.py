#!/usr/bin/env python

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import validate_rgb_thermal_mix_feature_names
from lerobot.datasets.feature_selection import filter_dataset_metadata_features
from lerobot.policies import make_policy_config, make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import (
    DEFAULT_ANYTHERMAL_CHECKPOINT_PATH,
    DEFAULT_RGB_DINOV2_CHECKPOINT_PATH,
    PI05Config,
)
from lerobot.policies.pi05.modeling_pi05 import (
    PI05Policy,
    PI05Pytorch,
    PaliGemmaWithExpertModel,
)
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.policies.pi05.thermal_cvae import ThermalCVAEEncoder
from lerobot.policies.pi05.thermal_matching import (
    prepare_fixed_thermal_matching,
    rescale_homography,
    robust_average_homographies,
    warp_thermal_batch,
)
from lerobot.policies.pi05.thermal_utils import (
    ANYTHERMAL_DINOV2_MEAN,
    ANYTHERMAL_DINOV2_STD,
    DINOV2_IMAGENET_MEAN,
    DINOV2_IMAGENET_STD,
    AnyThermalEncoder,
    AnyThermalResidualFusion,
    DINOv2RGBEncoder,
    GateViTMixFusion,
    OneGreyGateOneResViTFusion,
    RGBThreeBlackGateThreeResViTFusion,
    RGBThermalFusionDecoder,
    RGBThermalTokenAligner,
    ResViTAttentionFusion,
    ThermalEncoder,
    ThermalHeadFusion,
    TwoGreyDoubleGateViTFusion,
    TwoGreyDoubleGateTwoResViTFusion,
    TwoGreyGateActionViTFusion,
    TwoGreyGateTwoResViTFusion,
    TwoGreyGateViTFusion,
    TwoGreyGateResViTFusion,
    TwoGreyGateResViT3Fusion,
    TwoGreyHardSmallPatchGateViTFusion,
    TwoGreyPatchGateViTFusion,
    TwoGreyPatchSingleGateResViTFusion,
    TwoGreySmallPatchGateResViTFusion,
    TwoGreySmallPatchSingleGateResViTFusion,
    TwoGreySmallPatchSingleGateTwoResViTFusion,
    TwoGreySmallPatchGateTwoResViTFusion,
    TwoGreySmallPatchGateViTFusion,
    get_modal_rgb_color,
    mix_rgb_and_thermal_images,
    rgb_to_three_black_dominance_tensor,
    thermal_to_background_normalized_grey_array,
    thermal_to_background_normalized_grey_tensor,
    thermal_to_two_background_normalized_grey_array,
    thermal_to_two_background_normalized_grey_tensor,
    thermal_to_two_black_background_normalized_grey_array,
    thermal_to_two_black_background_normalized_grey_tensor,
    twogrey_patch_evidence,
)
from lerobot.policies.utils import validate_visual_features_consistency
from lerobot.processor import PolicyProcessorPipeline
from lerobot.scripts.lerobot_train import _stabilize_thermal_matching_training_io


RGB_FEATURE = "observation.images.cam_left_high"
WRIST_LEFT_FEATURE = "observation.images.cam_left_wrist"
WRIST_RIGHT_FEATURE = "observation.images.cam_right_wrist"
THERMAL_FEATURE = "observation.images.cam_thermal"
THERMAL_COLD_FEATURE = "observation.images.cam_thermal_cold"
THERMAL_HOT_FEATURE = "observation.images.cam_thermal_hot"
RGB_THERMAL_MIX_FEATURE = "observation.images.cam_rgb_thermal_mixing"
RGB_THERMAL_GT_FEATURE = "observation.images.cam_rgb_thermal_mixing_GT"


def test_freeze_non_thermal_input_freezes_paligemma_but_keeps_expert_trainable():
    model = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    nn.Module.__init__(model)
    model.freeze_vision_encoder = False
    model.train_expert_only = False
    model.freeze_non_thermal_input = True
    model.paligemma = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
    model.gemma_expert = nn.Linear(2, 2)

    model._set_requires_grad()
    model.train()

    assert all(not parameter.requires_grad for parameter in model.paligemma.parameters())
    assert all(parameter.requires_grad for parameter in model.gemma_expert.parameters())
    assert not model.paligemma.training
    assert model.gemma_expert.training


@pytest.mark.parametrize("has_dedicated_projector", [False, True])
def test_freeze_non_thermal_input_keeps_copied_thermal_vit_trainable(
    has_dedicated_projector,
):
    model = PI05Pytorch.__new__(PI05Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        freeze_non_thermal_input=True,
        freeze_vision_encoder=False,
        train_expert_only=False,
    )
    model.thermal_vision_tower = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
    model.thermal_multi_modal_projector = (
        nn.Linear(2, 2) if has_dedicated_projector else None
    )

    model.thermal_vision_tower.eval().requires_grad_(False)
    if model.thermal_multi_modal_projector is not None:
        model.thermal_multi_modal_projector.eval().requires_grad_(False)

    model._configure_thermal_vit_trainability()

    assert model.thermal_vision_tower.training
    assert all(parameter.requires_grad for parameter in model.thermal_vision_tower.parameters())
    if model.thermal_multi_modal_projector is not None:
        assert model.thermal_multi_modal_projector.training
        assert all(
            parameter.requires_grad
            for parameter in model.thermal_multi_modal_projector.parameters()
        )


def test_rgb_threeblack_color_vit_is_shared_across_colors_but_independent_from_original():
    class ModuleBox(nn.Module):
        pass

    class RecordingPaliGemma(nn.Module):
        def __init__(self):
            super().__init__()
            self.paligemma = ModuleBox()
            self.paligemma.model = ModuleBox()
            self.paligemma.model.vision_tower = nn.Linear(2, 2)
            self.paligemma.model.multi_modal_projector = nn.Linear(2, 2)
            self.calls = []

        def embed_image_with_modules(self, image, vision_tower, projector):
            self.calls.append((vision_tower, projector))
            return projector(vision_tower(image))

    model = PI05Pytorch.__new__(PI05Pytorch)
    nn.Module.__init__(model)
    model.paligemma_with_expert = RecordingPaliGemma()
    original_tower = model.paligemma_with_expert.paligemma.model.vision_tower
    original_projector = (
        model.paligemma_with_expert.paligemma.model.multi_modal_projector
    )
    model.rgb_threeblack_vision_tower = nn.Linear(2, 2)
    model.rgb_threeblack_multi_modal_projector = nn.Linear(2, 2)

    model.initialize_rgb_threeblack_vit_from_rgb()

    for original_parameter, color_parameter in zip(
        original_tower.parameters(),
        model.rgb_threeblack_vision_tower.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(color_parameter, original_parameter)
        assert color_parameter.data_ptr() != original_parameter.data_ptr()
    for original_parameter, color_parameter in zip(
        original_projector.parameters(),
        model.rgb_threeblack_multi_modal_projector.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(color_parameter, original_parameter)
        assert color_parameter.data_ptr() != original_parameter.data_ptr()

    color_input = torch.randn(1, 2)
    for _ in ("red", "green", "blue"):
        model._embed_rgb_threeblack_image(color_input)

    assert len(model.paligemma_with_expert.calls) == 3
    assert all(
        vision_tower is model.rgb_threeblack_vision_tower
        and projector is model.rgb_threeblack_multi_modal_projector
        for vision_tower, projector in model.paligemma_with_expert.calls
    )
    assert model.rgb_threeblack_vision_tower is not original_tower
    assert model.rgb_threeblack_multi_modal_projector is not original_projector


@pytest.mark.parametrize("conflicting_flag", ["train_expert_only", "freeze_vision_encoder"])
def test_freeze_non_thermal_input_rejects_thermal_freezing_flags(conflicting_flag):
    with pytest.raises(ValueError, match="also freezes the thermal ViT"):
        PI05Config(freeze_non_thermal_input=True, **{conflicting_flag: True})


def test_freeze_non_thermal_input_keeps_standalone_rgb_encoder_in_eval_mode():
    model = PI05Pytorch.__new__(PI05Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        uses_thermal_cvae_encoder=False,
        uses_anythermal_encoder=False,
        uses_dinov2_rgb_encoder=True,
        uses_resthermal_encoder=False,
        thermal_cvae_freeze_encoder=False,
        thermal_anythermal_freeze_backbone=False,
        rgb_dinov2_freeze_backbone=False,
        freeze_non_thermal_input=True,
        freeze_vision_encoder=False,
        train_expert_only=False,
    )
    model.rgb_encoder = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
    model.thermal_vision_tower = None
    model.thermal_multi_modal_projector = None
    model.thermal_encoder = None
    model.thermal_residual_fusion = None

    model.rgb_encoder.requires_grad_(False)
    model.train()

    assert not model.rgb_encoder.training
    assert all(not parameter.requires_grad for parameter in model.rgb_encoder.parameters())


def test_mix_rgb_and_thermal_images_applies_weight_offset_crop_and_fill():
    rgb = torch.full((1, 3, 4, 6), 0.2)
    thermal = torch.ones((1, 3, 4, 4))

    mixed = mix_rgb_and_thermal_images(
        rgb,
        thermal,
        rgb_feature=RGB_FEATURE,
        thermal_feature=THERMAL_FEATURE,
        horizontal_offset=-2,
        vertical_offset=1,
        thermal_width=4,
        thermal_weight=0.8,
        fill_color=(0, 0, 0),
    )

    expected = torch.full_like(rgb, 0.2 * 0.2)
    expected[:, :, :3, 2:6] = 0.8 * 1.0 + 0.2 * 0.2
    torch.testing.assert_close(mixed, expected)


def test_mix_rgb_and_thermal_images_accepts_channels_last_and_per_sample_fill():
    rgb = torch.zeros((2, 4, 6, 3))
    thermal = torch.ones((2, 4, 4, 3))
    fill_colors = torch.tensor([[255, 0, 0], [0, 255, 0]])

    mixed = mix_rgb_and_thermal_images(
        rgb,
        thermal,
        rgb_feature=RGB_FEATURE,
        thermal_feature=THERMAL_FEATURE,
        horizontal_offset=-6,
        vertical_offset=0,
        thermal_width=4,
        thermal_weight=0.8,
        fill_color=fill_colors,
    )

    expected = torch.zeros((2, 3, 4, 6))
    expected[0, 0] = 0.8
    expected[1, 1] = 0.8
    torch.testing.assert_close(mixed, expected)


def test_thermal_grey_background_normalization_uses_single_channel_and_roi_median():
    thermal_u8 = torch.tensor(
        [
            [
                [[100, 120, 200], [90, 80, 70]],
                [[255, 255, 255], [255, 255, 255]],
                [[0, 0, 0], [0, 0, 0]],
            ]
        ],
        dtype=torch.float32,
    )
    thermal = thermal_u8 / 255.0

    grey = thermal_to_background_normalized_grey_tensor(
        thermal,
        feature_name=THERMAL_FEATURE,
        background_roi=(0, 0, 1, 2),
        target_background=80,
    )

    expected_gray = torch.tensor([[[70, 90, 170], [60, 50, 40]]], dtype=torch.float32) / 255.0
    expected = expected_gray[:, None].repeat(1, 3, 1, 1)
    torch.testing.assert_close(grey, expected)

    image = np.transpose(thermal_u8[0].numpy().astype(np.uint8), (1, 2, 0))
    grey_np = thermal_to_background_normalized_grey_array(
        image,
        background_roi=(0, 0, 1, 2),
        target_background=80,
    )
    expected_np = np.repeat((expected_gray[0].numpy() * 255).astype(np.uint8)[..., None], 3, axis=-1)
    np.testing.assert_array_equal(grey_np, expected_np)


def test_thermal_twogrey_splits_background_normalized_grey_at_boundary():
    thermal_u8 = torch.tensor([[[[80, 80], [78, 81]]]], dtype=torch.float32)
    thermal = thermal_u8 / 255.0

    cold, hot = thermal_to_two_background_normalized_grey_tensor(
        thermal,
        feature_name=THERMAL_FEATURE,
        background_roi=(0, 0, 1, 2),
        target_background=80,
    )

    expected_cold_gray = torch.tensor([[[80, 80], [78, 80]]], dtype=torch.float32) / 255.0
    expected_hot_gray = torch.tensor([[[80, 80], [80, 81]]], dtype=torch.float32) / 255.0
    torch.testing.assert_close(cold, expected_cold_gray[:, None].repeat(1, 3, 1, 1))
    torch.testing.assert_close(hot, expected_hot_gray[:, None].repeat(1, 3, 1, 1))

    image = thermal_u8[0, 0].numpy().astype(np.uint8)
    cold_np, hot_np = thermal_to_two_background_normalized_grey_array(
        image,
        background_roi=(0, 0, 1, 2),
        target_background=80,
    )
    np.testing.assert_array_equal(
        cold_np,
        np.repeat((expected_cold_gray[0].numpy() * 255).astype(np.uint8)[..., None], 3, axis=-1),
    )
    np.testing.assert_array_equal(
        hot_np,
        np.repeat((expected_hot_gray[0].numpy() * 255).astype(np.uint8)[..., None], 3, axis=-1),
    )


def test_thermal_twoblack_rejects_top_roi_outlier_and_maps_importance_to_black():
    thermal_u8 = torch.full((1, 1, 4, 12), 100, dtype=torch.float32)
    thermal_u8[:, :, :2, :2] = 80
    thermal_u8[:, :, :2, 5:7] = 0
    thermal_u8[:, :, :2, 10:12] = 120
    thermal_u8[0, 0, 3, :6] = torch.tensor([20, 60, 100, 140, 200, 255])

    cold, hot = thermal_to_two_black_background_normalized_grey_tensor(
        thermal_u8 / 255.0,
        feature_name=THERMAL_FEATURE,
        background_roi=(0, 0, 2, 2),
        target_background=80,
        outlier_threshold=12,
        outlier_ratio=2,
    )

    expected_cold = torch.tensor([0, 128, 255, 255, 255, 255], dtype=torch.float32) / 255.0
    expected_hot = torch.tensor([255, 255, 255, 197, 109, 29], dtype=torch.float32) / 255.0
    torch.testing.assert_close(cold[0, 0, 3, :6], expected_cold)
    torch.testing.assert_close(hot[0, 0, 3, :6], expected_hot)
    torch.testing.assert_close(cold[:, 0], cold[:, 1])
    torch.testing.assert_close(cold[:, 1], cold[:, 2])
    torch.testing.assert_close(hot[:, 0], hot[:, 1])
    torch.testing.assert_close(hot[:, 1], hot[:, 2])

    cold_np, hot_np = thermal_to_two_black_background_normalized_grey_array(
        thermal_u8[0, 0].numpy().astype(np.uint8),
        background_roi=(0, 0, 2, 2),
        target_background=80,
        outlier_threshold=12,
        outlier_ratio=2,
    )
    np.testing.assert_array_equal(cold_np[3, :6, 0], (expected_cold * 255).numpy())
    np.testing.assert_array_equal(hot_np[3, :6, 0], (expected_hot * 255).numpy())


def test_thermal_encoder_and_head_fusion_shapes():
    encoder = ThermalEncoder(
        in_channels=3,
        hidden_dim=32,
        output_dim=64,
        token_grid=(2, 3),
    )
    thermal_tokens = encoder(torch.rand(2, 3, 32, 32))
    assert thermal_tokens.shape == (2, 6, 64)

    fusion = ThermalHeadFusion(embed_dim=64, num_heads=8)
    fused, mask = fusion(
        thermal_tokens=thermal_tokens,
        head_tokens=torch.rand(2, 4, 64),
        thermal_token_mask=torch.ones(2, 6, dtype=torch.bool),
        head_token_mask=torch.tensor([[True] * 4, [False] * 4]),
    )
    assert fused.shape == (2, 6, 64)
    assert mask[0].all()
    assert not mask[1].any()
    assert torch.count_nonzero(fused[1]) == 0


def test_thermal_cvae_encoder_channel_and_shape():
    config = PI05Config(thermal_encoder_channel="'cvae'")
    assert config.thermal_encoder_channel == "cvae"
    assert config.uses_thermal_cvae_encoder
    assert not config.thermal_cvae_freeze_encoder

    encoder = ThermalCVAEEncoder(
        latent_dim=8,
        hidden_dims=[4, 8],
        image_size=16,
        output_dim=32,
        token_grid_size=4,
    )
    tokens = encoder(
        torch.rand(2, 3, 16, 16) * 2.0 - 1.0,
        torch.rand(2, 3, 16, 16) * 2.0 - 1.0,
    )
    assert tokens.shape == (2, 16, 32)


@pytest.mark.parametrize("thermal_input_type", ["anythermal", "resthermal"])
def test_thermal_input_type_anythermal_alias_configures_pi05(tmp_path, thermal_input_type):
    policy = PI05Config(push_to_hub=False)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        policy=policy,
        thermal_input_type=thermal_input_type,
        output_dir=tmp_path / "run",
    )

    cfg.validate()

    assert policy.thermal_input_type == thermal_input_type
    assert policy.thermal_encoder_channel == thermal_input_type
    assert policy.uses_anythermal_encoder == (thermal_input_type == "anythermal")
    assert policy.uses_resthermal_encoder == (thermal_input_type == "resthermal")
    assert not policy.mix_rgb_thermal
    assert policy.thermal_anythermal_checkpoint_path == DEFAULT_ANYTHERMAL_CHECKPOINT_PATH


def test_thermal_input_type_grey_alias_configures_pi05_preprocessing(tmp_path):
    policy = PI05Config(
        push_to_hub=False,
        mix_rgb_thermal=True,
        rgb_thermal_mix_fill_color=(0, 0, 0),
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        policy=policy,
        thermal_input_type="'grey'",
        output_dir=tmp_path / "run_grey",
    )

    cfg.validate()

    assert policy.thermal_input_type == "grey"
    assert policy.thermal_encoder_channel is False
    assert not policy.mix_rgb_thermal
    assert policy.thermal_grey_background_roi == (0, 0, 64, 64)
    assert policy.thermal_grey_background_value == 80.0


def test_thermal_input_type_twogrey_alias_configures_shared_vit(tmp_path):
    policy = PI05Config(
        push_to_hub=False,
        mix_rgb_thermal=True,
        rgb_thermal_mix_fill_color=(0, 0, 0),
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        policy=policy,
        thermal_input_type="'twogrey'",
        output_dir=tmp_path / "run_twogrey",
    )

    cfg.validate()

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "shared_projector_vit"
    assert policy.uses_shared_projector_thermal_vit
    assert policy.uses_twogrey_thermal_input
    assert not policy.mix_rgb_thermal
    assert policy.thermal_twogrey_feature_names(THERMAL_FEATURE) == (
        THERMAL_COLD_FEATURE,
        THERMAL_HOT_FEATURE,
    )


def test_thermal_input_type_twoblack_alias_configures_shared_vit(tmp_path):
    policy = PI05Config(
        push_to_hub=False,
        thermal_encoder_channel="'ViT'",
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        policy=policy,
        thermal_input_type="'twoblack'",
        output_dir=tmp_path / "run_twoblack",
    )

    cfg.validate()

    assert policy.thermal_input_type == "twoblack"
    assert policy.thermal_encoder_channel == "shared_projector_vit"
    assert policy.uses_shared_projector_thermal_vit
    assert policy.uses_twogrey_thermal_input
    assert policy.thermal_black_background_outlier_threshold == 12.0
    assert policy.thermal_black_background_outlier_ratio == 2.0
    assert policy.thermal_twogrey_feature_names(THERMAL_FEATURE) == (
        THERMAL_COLD_FEATURE,
        THERMAL_HOT_FEATURE,
    )


def test_thermal_input_type_twomatchingblack_configures_patchgatevit(tmp_path):
    policy = PI05Config(
        push_to_hub=False,
        thermal_encoder_channel="'PatchGateViT'",
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        policy=policy,
        thermal_input_type="'twomatchingblack'",
        output_dir=tmp_path / "run_twomatchingblack",
    )

    cfg.validate()

    assert policy.thermal_input_type == "twomatchingblack"
    assert policy.thermal_encoder_channel == "patchgatevit"
    assert policy.uses_patch_gate_thermal_vit
    assert policy.uses_twogrey_thermal_input
    assert not policy.mix_rgb_thermal
    assert policy.thermal_matching_match_every_n_frames == 10
    assert policy.thermal_matching_outer_padding == 30


def test_thermal_input_type_twofixmatchingblack_configures_patchgatevit(tmp_path):
    policy = PI05Config(
        push_to_hub=False,
        thermal_encoder_channel="'PatchGateViT'",
        thermal_fixed_matching_num_samples=8,
        thermal_fixed_matching_min_valid_samples=3,
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        policy=policy,
        thermal_input_type="'twofixmatchingblack'",
        output_dir=tmp_path / "run_twofixmatchingblack",
    )

    cfg.validate()

    assert policy.thermal_input_type == "twofixmatchingblack"
    assert policy.thermal_encoder_channel == "patchgatevit"
    assert policy.uses_patch_gate_thermal_vit
    assert policy.uses_twogrey_thermal_input
    assert not policy.mix_rgb_thermal
    assert policy.thermal_fixed_matching_num_samples == 8
    assert policy.thermal_fixed_matching_min_valid_samples == 3


def test_thermal_matching_training_io_uses_pyav_and_lazy_preload(tmp_path):
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="'twomatchingblack'",
        thermal_encoder_channel="'PatchGateViT'",
        thermal_matching_preload=True,
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo", video_backend="torchcodec"),
        policy=policy,
        output_dir=tmp_path / "run_twomatchingblack",
        num_workers=2,
        persistent_workers=False,
    )
    cfg.validate()

    _stabilize_thermal_matching_training_io(cfg, log_changes=False)

    assert cfg.dataset.video_backend == "pyav"
    assert not policy.thermal_matching_preload
    assert cfg.persistent_workers


def test_fixed_matching_rescale_and_robust_average_reject_outlier():
    homographies = [
        np.asarray(((1, 0, 10), (0, 1, 4), (0, 0, 1)), dtype=np.float64),
        np.asarray(((1, 0, 11), (0, 1, 3), (0, 0, 1)), dtype=np.float64),
        np.asarray(((1, 0, 9), (0, 1, 5), (0, 0, 1)), dtype=np.float64),
        np.asarray(((1, 0, 180), (0, 1, -90), (0, 0, 1)), dtype=np.float64),
    ]

    consensus, diagnostics = robust_average_homographies(
        homographies,
        source_size=(48, 64),
    )
    assert consensus[0, 2] == pytest.approx(10, abs=1)
    assert consensus[1, 2] == pytest.approx(4, abs=1)
    assert diagnostics["accepted_count"] == 3

    resized = rescale_homography(
        consensus,
        stored_source_size=(48, 64),
        current_source_size=(96, 128),
        stored_output_size=(48, 64),
        current_output_size=(96, 128),
    )
    assert resized[0, 2] == pytest.approx(consensus[0, 2] * 127 / 63)
    assert resized[1, 2] == pytest.approx(consensus[1, 2] * 95 / 47)


def test_prepare_fixed_matching_samples_dataset_and_restores_transforms(
    monkeypatch,
    tmp_path,
):
    class DummyDataset:
        def __init__(self):
            self.root = tmp_path
            self.image_transforms = lambda image: image

        def __len__(self):
            return 8

        def __getitem__(self, index):
            return {
                RGB_FEATURE: torch.full((3, 12, 16), 120, dtype=torch.uint8),
                THERMAL_FEATURE: torch.full((3, 12, 16), 80, dtype=torch.uint8),
                "timestamp": torch.tensor(index / 30),
            }

        def clear_image_transforms(self):
            self.image_transforms = None

        def set_image_transforms(self, transforms):
            self.image_transforms = transforms

    class DummyMatcher:
        def __init__(self, **kwargs):
            self.calls = 0

        def estimate_homography(self, rgb_image, thermal_image, *, output_size):
            offsets = (10, 11, 9, 10, 12, 8, 10, 150)
            offset = offsets[self.calls]
            self.calls += 1
            return (
                np.asarray(((1, 0, offset), (0, 1, 4), (0, 0, 1)), dtype=np.float64),
                {"fallback_reason": ""},
            )

    from lerobot.policies.pi05 import thermal_matching

    monkeypatch.setattr(thermal_matching, "MinimaThermalMatcher", DummyMatcher)
    config = PI05Config(
        push_to_hub=False,
        thermal_input_type="twofixmatchingblack",
        thermal_encoder_channel="PatchGateViT",
        thermal_fixed_matching_num_samples=8,
        thermal_fixed_matching_min_valid_samples=4,
    )
    dataset = DummyDataset()
    transforms = dataset.image_transforms

    payload = prepare_fixed_thermal_matching(config, dataset)

    assert dataset.image_transforms is transforms
    assert len(config.thermal_fixed_matching_sample_indices) == 8
    assert len(config.thermal_fixed_matching_sample_timestamps) == 8
    assert config.thermal_fixed_matching_dataset_roots == [str(tmp_path.resolve())]
    assert config.thermal_fixed_matching_homographies[THERMAL_FEATURE][0][2] == pytest.approx(
        10,
        abs=2,
    )
    assert payload["thermal_fixed_matching_homographies"] == (
        config.thermal_fixed_matching_homographies
    )
    checkpoint_dir = tmp_path / "fixed_matching_checkpoint"
    checkpoint_dir.mkdir()
    config.save_pretrained(checkpoint_dir)
    restored = PreTrainedConfig.from_pretrained(checkpoint_dir)
    assert restored.thermal_fixed_matching_homographies == (
        config.thermal_fixed_matching_homographies
    )
    assert restored.thermal_fixed_matching_sample_timestamps == (
        config.thermal_fixed_matching_sample_timestamps
    )


def test_twomatchingblack_warp_preserves_source_and_fills_missing_white():
    thermal = torch.full((1, 1, 4, 4), 100.0 / 255.0)
    thermal[:, :, 2, 2] = 60.0 / 255.0
    background = torch.tensor([100.0]).reshape(1, 1, 1, 1)
    homography = torch.tensor(
        [[[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]]
    )

    aligned, alpha = warp_thermal_batch(
        thermal,
        homography,
        output_size=(8, 8),
        outer_padding=2,
        feature_name=THERMAL_FEATURE,
    )
    cold, hot = thermal_to_two_black_background_normalized_grey_tensor(
        aligned,
        feature_name=THERMAL_FEATURE,
        background_roi=(0, 0, 2, 2),
        target_background=80,
        background=background,
        valid_alpha=alpha,
    )

    torch.testing.assert_close(aligned[:, :, 2:6, 2:6], thermal)
    torch.testing.assert_close(alpha[:, :, 2:6, 2:6], torch.ones(1, 1, 4, 4))
    assert cold[0, 0, 4, 4] < 1.0
    assert hot[0, 0, 4, 4] == pytest.approx(1.0)
    assert cold[0, 0, 0, 0] == pytest.approx(1.0)
    assert hot[0, 0, 0, 0] == pytest.approx(1.0)


def test_twomatchingblack_warp_keeps_geometry_fp32_under_bfloat16_autocast():
    thermal = torch.rand(2, 3, 12, 16, dtype=torch.bfloat16)
    homography = torch.tensor(
        [
            [[1.0, 0.0, 1.5], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]],
            [[0.98, 0.02, 0.0], [-0.02, 0.98, 1.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        aligned, alpha = warp_thermal_batch(
            thermal,
            homography,
            output_size=(10, 14),
            outer_padding=3,
            feature_name=THERMAL_FEATURE,
        )

    assert aligned.shape == (2, 3, 10, 14)
    assert aligned.dtype == torch.bfloat16
    assert alpha.shape == (2, 1, 10, 14)
    assert alpha.dtype == torch.float32
    assert torch.isfinite(aligned.float()).all()
    assert torch.isfinite(alpha).all()


def test_preprocess_images_uses_checkpoint_fixed_matching_without_minima():
    class DummyPolicy(nn.Module):
        _preprocess_image_tensor = PI05Policy._preprocess_image_tensor

        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.matcher_called = False
            self.config = SimpleNamespace(
                thermal_input_type="twofixmatchingblack",
                rgb_thermal_mix_rgb_feature=RGB_FEATURE,
                thermal_image_features=[THERMAL_FEATURE],
                thermal_grey_background_roi=(0, 0, 2, 2),
                thermal_grey_background_value=80.0,
                thermal_black_background_outlier_threshold=12.0,
                thermal_black_background_outlier_ratio=2.0,
                thermal_fixed_matching_homographies={
                    THERMAL_FEATURE: [[7 / 3, 0, 0], [0, 7 / 3, 0], [0, 0, 1]]
                },
                thermal_fixed_matching_source_sizes={THERMAL_FEATURE: [4, 4]},
                thermal_fixed_matching_output_sizes={THERMAL_FEATURE: [8, 8]},
                thermal_matching_outer_padding=0,
                thermal_twogrey_feature_map=lambda: {
                    THERMAL_FEATURE: (THERMAL_COLD_FEATURE, THERMAL_HOT_FEATURE)
                },
                uses_gate_residual_thermal_vit3=False,
                mix_rgb_thermal=False,
                image_resolution=(8, 8),
                legacy_resize_padding=False,
                image_features={
                    RGB_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_COLD_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_HOT_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                },
            )
            self._logged_thermal_matching_input = False
            self._logged_thermal_twogrey_input = False
            self._logged_missing_image_features = False
            self._logged_image_input_summary = True
            self._logged_rgb_thermal_mix = False
            self._logged_thermal_grey_input = False

        def _get_thermal_matcher(self):
            self.matcher_called = True
            raise AssertionError("Fixed matching must not initialize MINIMA")

    policy = DummyPolicy()
    thermal = torch.full((2, 1, 4, 4), 80.0 / 255.0)
    thermal[:, :, 2, 2] = 40.0 / 255.0
    images, masks = PI05Policy._preprocess_images(
        policy,
        {
            RGB_FEATURE: torch.zeros(2, 3, 8, 8),
            THERMAL_FEATURE: thermal,
        },
    )

    assert not policy.matcher_called
    assert len(images) == 3
    assert all(image.shape == (2, 3, 8, 8) for image in images)
    assert all(mask.all() for mask in masks)


def test_thermal_input_type_twogrey_accepts_resvit_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        thermal_encoder_channel="'ResViT'",
        thermal_twogrey_resvit_cold_alpha=0.25,
        thermal_twogrey_resvit_hot_beta=0.75,
    )

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "resvit"
    assert policy.uses_residual_thermal_vit
    assert policy.thermal_twogrey_resvit_cold_alpha == 0.25
    assert policy.thermal_twogrey_resvit_hot_beta == 0.75


def test_thermal_input_type_twogrey_accepts_gateresvit_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        thermal_encoder_channel="'GateResViT'",
        thermal_twogrey_resvit_cold_alpha=0.25,
        thermal_twogrey_resvit_hot_beta=0.75,
        thermal_twogrey_gateresvit_hidden_dim=128,
    )

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "gateresvit"
    assert policy.uses_gate_residual_thermal_vit
    assert policy.thermal_twogrey_resvit_cold_alpha == 0.25
    assert policy.thermal_twogrey_resvit_hot_beta == 0.75
    assert policy.thermal_twogrey_gateresvit_hidden_dim == 128


@pytest.mark.parametrize("thermal_input_type", ["twogrey", "twofixmatchingblack"])
def test_thermal_input_type_accepts_gateresandvit_channel(thermal_input_type):
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type=thermal_input_type,
        thermal_encoder_channel="'GateResandViT'",
        thermal_twogrey_resvit_cold_alpha=0.25,
        thermal_twogrey_resvit_hot_beta=0.75,
        thermal_twogrey_gateresvit_hidden_dim=128,
    )

    assert policy.thermal_input_type == thermal_input_type
    assert policy.thermal_encoder_channel == "gateresandvit"
    assert policy.uses_gate_res_and_thermal_vit
    assert policy.uses_thermal_vit_encoder
    assert policy.uses_twogrey_thermal_input
    assert policy.thermal_twogrey_resvit_cold_alpha == 0.25
    assert policy.thermal_twogrey_resvit_hot_beta == 0.75
    assert policy.thermal_twogrey_gateresvit_hidden_dim == 128


def test_thermal_input_type_twogrey_accepts_gatetworesvit_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        thermal_encoder_channel="'GateTwoResViT'",
        thermal_twogrey_resvit_cold_alpha=0.25,
        thermal_twogrey_resvit_hot_beta=0.75,
        thermal_twogrey_gateresvit_hidden_dim=128,
    )

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "gatetworesvit"
    assert policy.uses_gate_two_residual_thermal_vit
    assert policy.uses_thermal_vit_encoder
    assert policy.thermal_twogrey_resvit_cold_alpha == 0.25
    assert policy.thermal_twogrey_resvit_hot_beta == 0.75
    assert policy.thermal_twogrey_gateresvit_hidden_dim == 128


def test_thermal_input_type_grey_accepts_gateoneresvit_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="grey",
        thermal_encoder_channel="'GateOneResViT'",
        thermal_grey_gateoneresvit_hidden_dim=128,
    )

    assert policy.thermal_input_type == "grey"
    assert policy.thermal_encoder_channel == "gateoneresvit"
    assert policy.uses_gate_one_residual_thermal_vit
    assert policy.uses_thermal_vit_encoder
    assert not policy.uses_twogrey_thermal_input
    assert policy.thermal_grey_gateoneresvit_hidden_dim == 128


def test_gateoneresvit_channel_auto_enables_grey_input_type():
    policy = PI05Config(push_to_hub=False, thermal_encoder_channel="'GateOneResViT'")

    assert policy.thermal_input_type == "grey"
    assert policy.thermal_encoder_channel == "gateoneresvit"


def test_gateoneresvit_rejects_two_stream_thermal_input_type():
    with pytest.raises(ValueError, match="GateOneResViT|thermal_input_type"):
        PI05Config(
            push_to_hub=False,
            thermal_input_type="twogrey",
            thermal_encoder_channel="'GateOneResViT'",
        )


def test_thermal_input_type_twogrey_accepts_gatevit_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        thermal_encoder_channel="'GateViT'",
        thermal_twogrey_gatevit_num_heads=4,
        thermal_twogrey_gatevit_hidden_dim=128,
        thermal_twogrey_gatevit_temperature=0.75,
    )

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "gatevit"
    assert policy.uses_gate_thermal_vit
    assert policy.uses_twogrey_thermal_input
    assert policy.thermal_twogrey_gatevit_num_heads == 4
    assert policy.thermal_twogrey_gatevit_hidden_dim == 128
    assert policy.thermal_twogrey_gatevit_temperature == 0.75


def test_thermal_input_type_twogrey_accepts_gateactionvit_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        thermal_encoder_channel="'GateActionViT'",
        thermal_twogrey_gateactionvit_num_heads=4,
        thermal_twogrey_gateactionvit_hidden_dim=128,
        thermal_twogrey_gateactionvit_temperature=0.75,
        thermal_twogrey_gateactionvit_bias_epsilon=0.2,
        thermal_twogrey_gateactionvit_bias_max_strength=1.5,
        thermal_twogrey_gateactionvit_bias_init_strength=0.3,
    )

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "gateactionvit"
    assert policy.uses_gate_action_thermal_vit
    assert policy.uses_twogrey_thermal_input
    assert policy.thermal_twogrey_gateactionvit_num_heads == 4
    assert policy.thermal_twogrey_gateactionvit_hidden_dim == 128
    assert policy.thermal_twogrey_gateactionvit_temperature == 0.75
    assert policy.thermal_twogrey_gateactionvit_bias_epsilon == 0.2
    assert policy.thermal_twogrey_gateactionvit_bias_max_strength == 1.5
    assert policy.thermal_twogrey_gateactionvit_bias_init_strength == 0.3


def test_thermal_input_type_twogrey_accepts_doublegatevit_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        thermal_encoder_channel="'DoubleGateViT'",
        thermal_twogrey_doublegatevit_num_heads=4,
        thermal_twogrey_doublegatevit_hidden_dim=128,
        thermal_twogrey_doublegatevit_temperature=0.75,
        thermal_twogrey_doublegatevit_rgb_gate_max_strength=1.5,
        thermal_twogrey_doublegatevit_rgb_gate_init_strength=0.4,
    )

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "doublegatevit"
    assert policy.uses_double_gate_thermal_vit
    assert policy.uses_twogrey_thermal_input
    assert policy.thermal_twogrey_doublegatevit_num_heads == 4
    assert policy.thermal_twogrey_doublegatevit_hidden_dim == 128
    assert policy.thermal_twogrey_doublegatevit_temperature == 0.75
    assert policy.thermal_twogrey_doublegatevit_rgb_gate_max_strength == 1.5
    assert policy.thermal_twogrey_doublegatevit_rgb_gate_init_strength == 0.4


def test_thermal_input_type_twogrey_accepts_doublegatetworesvit_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        thermal_encoder_channel="'DoubleGateTwoResViT'",
        thermal_twogrey_doublegatevit_num_heads=4,
        thermal_twogrey_doublegatevit_hidden_dim=128,
        thermal_twogrey_doublegatevit_temperature=0.75,
        thermal_twogrey_doublegatevit_rgb_gate_max_strength=1.5,
        thermal_twogrey_doublegatevit_rgb_gate_init_strength=0.4,
    )

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "doublegatetworesvit"
    assert policy.uses_double_gate_two_residual_thermal_vit
    assert policy.uses_twogrey_thermal_input
    assert policy.uses_thermal_vit_encoder
    assert policy.thermal_twogrey_doublegatevit_num_heads == 4
    assert policy.thermal_twogrey_doublegatevit_hidden_dim == 128
    assert policy.thermal_twogrey_doublegatevit_temperature == 0.75
    assert policy.thermal_twogrey_doublegatevit_rgb_gate_max_strength == 1.5
    assert policy.thermal_twogrey_doublegatevit_rgb_gate_init_strength == 0.4


def test_thermal_input_type_twogrey_accepts_patchgatevit_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        thermal_encoder_channel="'PatchGateViT'",
        thermal_twogrey_patchgatevit_gate_dim=128,
        thermal_twogrey_patchgatevit_num_heads=4,
        thermal_twogrey_patchgatevit_hidden_dim=96,
        thermal_twogrey_patchgatevit_temperature=0.75,
        thermal_twogrey_patchgatevit_evidence_strength=0.8,
        thermal_twogrey_patchgatevit_evidence_floor=0.1,
    )

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "patchgatevit"
    assert policy.uses_patch_gate_thermal_vit
    assert policy.uses_twogrey_thermal_input
    assert policy.thermal_twogrey_patchgatevit_gate_dim == 128
    assert policy.thermal_twogrey_patchgatevit_num_heads == 4
    assert policy.thermal_twogrey_patchgatevit_hidden_dim == 96
    assert policy.thermal_twogrey_patchgatevit_temperature == 0.75
    assert policy.thermal_twogrey_patchgatevit_evidence_strength == 0.8
    assert policy.thermal_twogrey_patchgatevit_evidence_floor == 0.1
    assert policy.thermal_twogrey_patchgatevit_detach_head_rgb


def test_thermal_input_type_twogrey_accepts_smallpatchgatevit_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        thermal_encoder_channel="'SmallPatchGateViT'",
        thermal_twogrey_patchgatevit_gate_dim=128,
        thermal_twogrey_patchgatevit_num_heads=4,
        thermal_twogrey_patchgatevit_hidden_dim=96,
    )

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "smallpatchgatevit"
    assert policy.uses_patch_gate_thermal_vit
    assert policy.uses_twogrey_thermal_input
    assert policy.uses_thermal_vit_encoder
    assert policy.thermal_twogrey_patchgatevit_gate_dim == 128
    assert policy.thermal_twogrey_smallpatchgatevit_gate_grid == (4, 4)

    alias_policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        thermal_encoder_channel="SmallPatchViT",
    )
    assert alias_policy.thermal_encoder_channel == "smallpatchgatevit"
    assert alias_policy.thermal_twogrey_smallpatchgatevit_gate_grid == (4, 4)


def test_thermal_input_type_twoblack_accepts_hardsmallpatchgatevit_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twoblack",
        thermal_encoder_channel="‘HardSmallPatchGateViT’",
    )

    assert policy.thermal_input_type == "twoblack"
    assert policy.thermal_encoder_channel == "hardsmallpatchgatevit"
    assert policy.uses_patch_gate_thermal_vit
    assert policy.uses_hard_small_patch_gate_thermal_vit
    assert policy.uses_twogrey_thermal_input
    assert policy.uses_thermal_vit_encoder
    assert policy.thermal_twogrey_smallpatchgatevit_gate_grid == (4, 4)


def test_hardsmallpatchgatevit_rejects_a_non_4x4_gate_grid():
    with pytest.raises(ValueError, match=r"requires .*\(4, 4\)"):
        PI05Config(
            push_to_hub=False,
            thermal_input_type="twoblack",
            thermal_encoder_channel="HardSmallPatchGateViT",
            thermal_twogrey_smallpatchgatevit_gate_grid=(2, 2),
        )


def test_thermal_input_type_twogrey_accepts_patchsinglegateresvit_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        thermal_encoder_channel="'PatchSingleGateResViT'",
        thermal_twogrey_patchgatevit_gate_dim=128,
        thermal_twogrey_patchgatevit_num_heads=4,
        thermal_twogrey_patchgatevit_hidden_dim=96,
    )

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "patchsinglegateresvit"
    assert policy.uses_patch_single_gate_residual_thermal_vit
    assert policy.uses_twogrey_thermal_input
    assert policy.uses_thermal_vit_encoder
    assert policy.thermal_twogrey_patchgatevit_gate_dim == 128


def test_thermal_input_type_twogrey_accepts_smallpatchgatetworesvit_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        thermal_encoder_channel="'SmallPatchGateTwoResViT'",
        thermal_twogrey_patchgatevit_gate_dim=128,
        thermal_twogrey_patchgatevit_num_heads=4,
        thermal_twogrey_patchgatevit_hidden_dim=96,
    )

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "smallpatchgatetworesvit"
    assert policy.uses_small_patch_gate_two_res_thermal_vit
    assert policy.uses_twogrey_thermal_input
    assert policy.uses_thermal_vit_encoder
    assert policy.thermal_twogrey_patchgatevit_gate_dim == 128
    assert policy.thermal_twogrey_smallpatchgatevit_gate_grid == (4, 4)


def test_thermal_input_type_twogrey_accepts_smallpatchsinglegatetworesvit_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        thermal_encoder_channel="'SmallPatchSingleGateTwoResViT'",
        thermal_twogrey_patchgatevit_gate_dim=128,
        thermal_twogrey_patchgatevit_num_heads=4,
        thermal_twogrey_patchgatevit_hidden_dim=96,
    )

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "smallpatchsinglegatetworesvit"
    assert policy.uses_small_patch_gate_two_res_thermal_vit
    assert policy.uses_small_patch_single_gate_two_res_thermal_vit
    assert policy.uses_twogrey_thermal_input
    assert policy.uses_thermal_vit_encoder
    assert policy.thermal_twogrey_patchgatevit_gate_dim == 128
    assert policy.thermal_twogrey_smallpatchgatevit_gate_grid == (4, 4)


def test_thermal_input_type_twogrey_accepts_smallpatchgateresvit_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        thermal_encoder_channel="'SmallPatchGateResViT'",
        thermal_twogrey_patchgatevit_gate_dim=128,
        thermal_twogrey_patchgatevit_num_heads=4,
        thermal_twogrey_patchgatevit_hidden_dim=96,
    )

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "smallpatchgateresvit"
    assert policy.uses_small_patch_gate_residual_thermal_vit
    assert not policy.uses_small_patch_single_gate_residual_thermal_vit
    assert policy.uses_twogrey_thermal_input
    assert policy.uses_thermal_vit_encoder
    assert policy.thermal_twogrey_patchgatevit_gate_dim == 128
    assert policy.thermal_twogrey_smallpatchgatevit_gate_grid == (4, 4)


def test_thermal_input_type_twogrey_accepts_smallpatchsinglegateresvit_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        thermal_encoder_channel="'SmallPatchSingleGateResViT'",
        thermal_twogrey_patchgatevit_gate_dim=128,
        thermal_twogrey_patchgatevit_num_heads=4,
        thermal_twogrey_patchgatevit_hidden_dim=96,
    )

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "smallpatchsinglegateresvit"
    assert policy.uses_small_patch_gate_residual_thermal_vit
    assert policy.uses_small_patch_single_gate_residual_thermal_vit
    assert policy.uses_twogrey_thermal_input
    assert policy.uses_thermal_vit_encoder
    assert policy.thermal_twogrey_patchgatevit_gate_dim == 128
    assert policy.thermal_twogrey_smallpatchgatevit_gate_grid == (4, 4)


@pytest.mark.parametrize(
    ("thermal_encoder_channel", "expected_channel"),
    [
        ("'PatchSingleGateResViT'", "patchsinglegateresvit"),
        ("'SmallPatchGateResViT'", "smallpatchgateresvit"),
        ("'SmallPatchSingleGateResViT'", "smallpatchsinglegateresvit"),
        ("'SmallPatchSingleGateTwoResViT'", "smallpatchsinglegatetworesvit"),
    ],
)
@pytest.mark.parametrize("thermal_input_type", ["twoblack", "twofixmatchingblack"])
def test_black_two_stream_input_types_accept_patch_res_channels(
    thermal_encoder_channel,
    expected_channel,
    thermal_input_type,
):
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type=thermal_input_type,
        thermal_encoder_channel=thermal_encoder_channel,
        thermal_image_features=[THERMAL_FEATURE],
        thermal_fusion_head_rgb_feature=RGB_FEATURE,
        input_features={
            RGB_FEATURE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
            THERMAL_FEATURE: PolicyFeature(type=FeatureType.VISUAL, shape=(1, 8, 8)),
        },
    )

    assert policy.thermal_input_type == thermal_input_type
    assert policy.thermal_encoder_channel == expected_channel
    assert (
        policy.uses_patch_single_gate_residual_thermal_vit
        or policy.uses_small_patch_gate_residual_thermal_vit
        or policy.uses_small_patch_gate_two_res_thermal_vit
    )
    assert policy.uses_twogrey_thermal_input
    policy.rewrite_twogrey_input_features()
    assert THERMAL_FEATURE not in policy.input_features
    assert THERMAL_COLD_FEATURE in policy.input_features
    assert THERMAL_HOT_FEATURE in policy.input_features


def test_thermal_input_type_twoblack_accepts_patchgatevit_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twoblack",
        thermal_encoder_channel="'PatchGateViT'",
    )

    assert policy.thermal_input_type == "twoblack"
    assert policy.thermal_encoder_channel == "patchgatevit"
    assert policy.uses_patch_gate_thermal_vit
    assert policy.uses_twogrey_thermal_input
    assert policy.thermal_black_background_outlier_threshold == 12.0
    assert policy.thermal_black_background_outlier_ratio == 2.0


def test_thermal_input_type_twogrey_accepts_gatevitmix_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        thermal_encoder_channel="'GateViTMix'",
        thermal_twogrey_gatevitmix_num_heads=4,
        thermal_twogrey_gatevitmix_hidden_dim=128,
        thermal_twogrey_gatevitmix_mix_dim=256,
        thermal_twogrey_gatevitmix_temperature=0.75,
        thermal_twogrey_gatevitmix_context_scale=0.25,
    )

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "gatevitmix"
    assert policy.uses_gate_thermal_vit_mix
    assert policy.uses_twogrey_thermal_input
    assert policy.thermal_twogrey_gatevitmix_num_heads == 4
    assert policy.thermal_twogrey_gatevitmix_hidden_dim == 128
    assert policy.thermal_twogrey_gatevitmix_mix_dim == 256
    assert policy.thermal_twogrey_gatevitmix_temperature == 0.75
    assert policy.thermal_twogrey_gatevitmix_context_scale == 0.25
    assert policy.thermal_twogrey_gatevitmix_detach_head_rgb


def test_thermal_input_type_twogrey_accepts_gateresvit3_channel():
    policy = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        thermal_encoder_channel="'GateResViT3'",
        thermal_twogrey_resvit_cold_alpha=0.25,
        thermal_twogrey_resvit_hot_beta=0.75,
        thermal_twogrey_gateresvit3_hidden_dim=128,
        thermal_twogrey_gateresvit3_stat_hidden_dim=32,
        thermal_twogrey_gateresvit3_residual_hidden_dim=64,
        thermal_twogrey_gateresvit3_evidence_gain=24.0,
        thermal_twogrey_gateresvit3_evidence_threshold=0.05,
    )

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "gateresvit3"
    assert policy.uses_gate_residual_thermal_vit3
    assert policy.uses_twogrey_thermal_input
    assert policy.thermal_twogrey_resvit_cold_alpha == 0.25
    assert policy.thermal_twogrey_resvit_hot_beta == 0.75
    assert policy.thermal_twogrey_gateresvit3_hidden_dim == 128
    assert policy.thermal_twogrey_gateresvit3_stat_hidden_dim == 32
    assert policy.thermal_twogrey_gateresvit3_residual_hidden_dim == 64
    assert policy.thermal_twogrey_gateresvit3_evidence_gain == 24.0
    assert policy.thermal_twogrey_gateresvit3_evidence_threshold == 0.05


def test_gateresvit_channel_auto_enables_twogrey_input_type():
    policy = PI05Config(push_to_hub=False, thermal_encoder_channel="'GateResViT'")

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "gateresvit"
    assert policy.uses_twogrey_thermal_input


def test_gateresandvit_channel_auto_enables_twogrey_input_type():
    policy = PI05Config(push_to_hub=False, thermal_encoder_channel="'GateResandViT'")

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "gateresandvit"
    assert policy.uses_gate_res_and_thermal_vit
    assert policy.uses_twogrey_thermal_input


def test_gatevit_channel_auto_enables_twogrey_input_type():
    policy = PI05Config(push_to_hub=False, thermal_encoder_channel="'GateViT'")

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "gatevit"
    assert policy.uses_twogrey_thermal_input


def test_gateactionvit_channel_auto_enables_twogrey_input_type():
    policy = PI05Config(push_to_hub=False, thermal_encoder_channel="'GateActionViT'")

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "gateactionvit"
    assert policy.uses_twogrey_thermal_input


def test_doublegatevit_channel_auto_enables_twogrey_input_type():
    policy = PI05Config(push_to_hub=False, thermal_encoder_channel="'DoubleGateViT'")

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "doublegatevit"
    assert policy.uses_twogrey_thermal_input


def test_doublegatetworesvit_channel_auto_enables_twogrey_input_type():
    policy = PI05Config(push_to_hub=False, thermal_encoder_channel="'DoubleGateTwoResViT'")

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "doublegatetworesvit"
    assert policy.uses_double_gate_two_residual_thermal_vit
    assert policy.uses_twogrey_thermal_input


def test_patchgatevit_channel_auto_enables_twogrey_input_type():
    policy = PI05Config(push_to_hub=False, thermal_encoder_channel="'PatchGateViT'")

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "patchgatevit"
    assert policy.uses_twogrey_thermal_input
    assert policy.thermal_twogrey_patchgatevit_evidence_strength == 1.0


@pytest.mark.parametrize(
    ("thermal_encoder_channel", "expected_channel"),
    [
        ("'PatchSingleGateResViT'", "patchsinglegateresvit"),
        ("'SmallPatchGateResViT'", "smallpatchgateresvit"),
        ("'SmallPatchSingleGateResViT'", "smallpatchsinglegateresvit"),
        ("'SmallPatchSingleGateTwoResViT'", "smallpatchsinglegatetworesvit"),
    ],
)
def test_patch_res_channels_auto_enable_twogrey_input_type(
    thermal_encoder_channel,
    expected_channel,
):
    policy = PI05Config(push_to_hub=False, thermal_encoder_channel=thermal_encoder_channel)

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == expected_channel
    assert (
        policy.uses_patch_single_gate_residual_thermal_vit
        or policy.uses_small_patch_gate_residual_thermal_vit
        or policy.uses_small_patch_gate_two_res_thermal_vit
    )
    assert policy.uses_twogrey_thermal_input


def test_patchgatevit_rejects_multiple_thermal_sources():
    with pytest.raises(ValueError, match="exactly one"):
        PI05Config(
            push_to_hub=False,
            thermal_encoder_channel="'PatchGateViT'",
            thermal_image_features=[
                THERMAL_FEATURE,
                "observation.images.cam_thermal_second",
            ],
        )


def test_gatevitmix_channel_auto_enables_twogrey_input_type():
    policy = PI05Config(push_to_hub=False, thermal_encoder_channel="'GateViTMix'")

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "gatevitmix"
    assert policy.uses_twogrey_thermal_input


def test_gateresvit3_channel_auto_enables_twogrey_input_type():
    policy = PI05Config(push_to_hub=False, thermal_encoder_channel="'GateResViT3'")

    assert policy.thermal_input_type == "twogrey"
    assert policy.thermal_encoder_channel == "gateresvit3"
    assert policy.uses_twogrey_thermal_input


def test_thermal_input_type_twogrey_rewrites_raw_thermal_feature_to_cold_hot():
    config = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        input_features={
            RGB_FEATURE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
            THERMAL_FEATURE: PolicyFeature(type=FeatureType.VISUAL, shape=(1, 8, 8)),
        },
    )

    config.set_dataset_feature_metadata({})

    assert THERMAL_FEATURE not in config.input_features
    assert config.input_features[THERMAL_COLD_FEATURE].shape == (3, 8, 8)
    assert config.input_features[THERMAL_HOT_FEATURE].shape == (3, 8, 8)
    assert config.thermal_model_image_features() == [THERMAL_COLD_FEATURE, THERMAL_HOT_FEATURE]


def test_thermal_input_type_twogrey_visual_consistency_accepts_raw_dataset_thermal():
    config = PI05Config(
        push_to_hub=False,
        thermal_input_type="twogrey",
        input_features={
            RGB_FEATURE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
            THERMAL_FEATURE: PolicyFeature(type=FeatureType.VISUAL, shape=(1, 8, 8)),
        },
    )
    dataset_features = {
        RGB_FEATURE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
        THERMAL_FEATURE: PolicyFeature(type=FeatureType.VISUAL, shape=(1, 8, 8)),
    }

    config.rewrite_twogrey_input_features(dataset_features)

    validate_visual_features_consistency(config, dataset_features)


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
        thermal_encoder_channel=False,
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
            thermal_encoder_channel=False,
        )


@pytest.mark.parametrize("policy_rgb_input_type", ["vit", "'GateTwoResViT'"])
def test_rgb_input_type_withthreeblack_alias_selects_rgb_gatetworesvit(
    tmp_path,
    policy_rgb_input_type,
):
    policy = PI05Config(
        push_to_hub=False,
        rgb_input_type=policy_rgb_input_type,
        thermal_encoder_channel=False,
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


@pytest.mark.parametrize(
    (
        "thermal_input_type",
        "expected_channel",
        "uses_dedicated",
        "uses_shared",
        "uses_residual",
        "uses_attention",
    ),
    [
        ("'vit'", "vit", True, False, False, False),
        ("'ViT'", "shared_projector_vit", False, True, False, False),
        ("'ResViT'", "resvit", False, False, True, False),
        ("'ResViTAttention'", "resvitattention", False, False, False, True),
    ],
)
def test_thermal_input_type_vit_variants_configure_projector_sharing(
    tmp_path,
    thermal_input_type,
    expected_channel,
    uses_dedicated,
    uses_shared,
    uses_residual,
    uses_attention,
):
    policy = PI05Config(push_to_hub=False)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        policy=policy,
        thermal_input_type=thermal_input_type,
        output_dir=tmp_path / f"run_{expected_channel}",
    )

    cfg.validate()

    assert policy.thermal_encoder_channel == expected_channel
    assert policy.uses_dedicated_thermal_vit == uses_dedicated
    assert policy.uses_shared_projector_thermal_vit == uses_shared
    assert policy.uses_residual_thermal_vit == uses_residual
    assert policy.uses_resvit_attention_encoder == uses_attention


@pytest.mark.parametrize("thermal_input_type", ["'ViT'", "'ResViT'"])
def test_shared_projector_vit_modes_reject_dinov2_rgb_tokens(thermal_input_type):
    with pytest.raises(ValueError, match="requires rgb_input_type='vit'"):
        PI05Config(rgb_input_type="'dinov2'", thermal_input_type=thermal_input_type)


def test_resvit_attention_accepts_dinov2_rgb_tokens():
    config = PI05Config(rgb_input_type="'dinov2'", thermal_input_type="'ResViTAttention'")

    assert config.rgb_input_type == "dinov2"
    assert config.thermal_encoder_channel == "resvitattention"
    assert config.uses_dinov2_rgb_encoder
    assert config.uses_resvit_attention_encoder


@pytest.mark.parametrize("rgb_input_type", ["vit", "'dinov2'"])
@pytest.mark.parametrize("thermal_input_type", [None, "anythermal", "resthermal", "ResViTAttention"])
def test_rgb_and_thermal_input_type_aliases_are_independent(
    tmp_path, rgb_input_type, thermal_input_type
):
    policy = PI05Config(push_to_hub=False)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        policy=policy,
        rgb_input_type=rgb_input_type,
        thermal_input_type=thermal_input_type,
        output_dir=tmp_path / f"run_{rgb_input_type.strip(chr(39))}_{thermal_input_type or 'none'}",
    )

    cfg.validate()

    expected_rgb_input_type = "dinov2" if "dinov2" in rgb_input_type else "vit"
    assert policy.rgb_input_type == expected_rgb_input_type
    assert policy.uses_dinov2_rgb_encoder == (expected_rgb_input_type == "dinov2")
    assert policy.uses_anythermal_encoder == (thermal_input_type == "anythermal")
    assert policy.uses_resthermal_encoder == (thermal_input_type == "resthermal")
    assert policy.uses_resvit_attention_encoder == (thermal_input_type == "ResViTAttention")


@pytest.mark.parametrize("thermal_input_type", [None, "anythermal", "resthermal"])
def test_dinov2_rgb_routes_head_and_wrist_cameras_through_same_encoder(thermal_input_type):
    class DummyLanguageModel:
        def embed_language_tokens(self, tokens):
            return torch.zeros(tokens.shape[0], tokens.shape[1], 8)

    class DummyThermalEncoder:
        def __init__(self):
            self.calls = 0

        def __call__(self, image):
            self.calls += 1
            return torch.zeros(image.shape[0], 5, 8)

    class DummyResidualFusion:
        def __init__(self):
            self.calls = 0

        def __call__(self, rgb_tokens, **kwargs):
            self.calls += 1
            return rgb_tokens

    class DummyModel:
        _apply_checkpoint = PI05Pytorch._apply_checkpoint
        _embed_resthermal_head_image = PI05Pytorch._embed_resthermal_head_image

        def __init__(self):
            image_features = {
                RGB_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                WRIST_LEFT_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                WRIST_RIGHT_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
            }
            if thermal_input_type is not None:
                image_features[THERMAL_FEATURE] = SimpleNamespace(shape=(1, 8, 8))

            self.config = SimpleNamespace(
                image_features=image_features,
                uses_thermal_resnet18_encoder=False,
                uses_anythermal_encoder=thermal_input_type == "anythermal",
                uses_thermal_cvae_encoder=False,
                uses_dedicated_thermal_vit=False,
                uses_shared_projector_thermal_vit=False,
                uses_residual_thermal_vit=False,
                uses_resvit_attention_encoder=False,
                thermal_cvae_source_feature=RGB_FEATURE,
            )
            self.gradient_checkpointing_enabled = False
            self.training = False
            self.rgb_calls = 0
            self.rgb_encoder = object()
            self.thermal_image_feature_set = (
                {THERMAL_FEATURE} if thermal_input_type in {"anythermal", "resthermal"} else set()
            )
            self.thermal_encoder = DummyThermalEncoder() if thermal_input_type == "anythermal" else None
            self.thermal_head_fusion = None
            self.rgb_thermal_align_aligner = None
            self.rgb_thermal_align_thermal_feature = None
            self.rgb_thermal_align_head_feature = RGB_FEATURE
            self.thermal_residual_fusion = (
                DummyResidualFusion() if thermal_input_type == "resthermal" else None
            )
            self.rgb_thermal_residual_head_feature = RGB_FEATURE
            self.rgb_thermal_residual_thermal_feature = (
                THERMAL_FEATURE if thermal_input_type == "resthermal" else None
            )
            self.resvit_attention_fusion = None
            self.rgb_thermal_resvit_attention_head_feature = RGB_FEATURE
            self.rgb_thermal_resvit_attention_thermal_feature = None
            self.thermal_vision_tower = None
            self.thermal_multi_modal_projector = None
            self.paligemma_with_expert = DummyLanguageModel()
            self._last_prefix_token_layout = []

        def _embed_rgb_image(self, image):
            self.rgb_calls += 1
            return torch.full((image.shape[0], 4, 8), float(self.rgb_calls))

    model = DummyModel()
    images = [
        torch.zeros(2, 3, 8, 8),
        torch.zeros(2, 3, 8, 8),
        torch.zeros(2, 3, 8, 8),
    ]
    if thermal_input_type is not None:
        images.append(torch.zeros(2, 1, 8, 8))
    image_masks = [torch.ones(2, dtype=torch.bool) for _ in images]

    PI05Pytorch.embed_prefix(
        model,
        images,
        image_masks,
        tokens=torch.zeros(2, 3, dtype=torch.long),
        masks=torch.ones(2, 3, dtype=torch.bool),
    )

    assert model.rgb_calls == 3
    if thermal_input_type == "anythermal":
        assert model.thermal_encoder.calls == 1
    elif thermal_input_type == "resthermal":
        assert model.thermal_residual_fusion.calls == 1

    image_layout_names = [
        item["name"] for item in model._last_prefix_token_layout if item["kind"].startswith("image")
    ]
    assert image_layout_names[:3] == [RGB_FEATURE, WRIST_LEFT_FEATURE, WRIST_RIGHT_FEATURE]


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
                uses_thermal_resnet18_encoder=False,
                uses_anythermal_encoder=False,
                uses_thermal_cvae_encoder=False,
                uses_dedicated_thermal_vit=False,
                uses_shared_projector_thermal_vit=False,
                uses_residual_thermal_vit=False,
                uses_resvit_attention_encoder=False,
                thermal_cvae_source_feature=RGB_FEATURE,
            )
            self.gradient_checkpointing_enabled = False
            self.training = False
            self.rgb_calls = 0
            self.rgb_threeblack_calls = 0
            self.rgb_encoder = None
            self.rgb_threeblack_gatetworesvit_fusion = object()
            self.rgb_threeblack_gatetworesvit_head_feature = RGB_FEATURE
            self.thermal_image_feature_set = set()
            self.thermal_encoder = None
            self.thermal_head_fusion = None
            self.rgb_thermal_align_aligner = None
            self.rgb_thermal_align_thermal_feature = None
            self.rgb_thermal_align_head_feature = RGB_FEATURE
            self.thermal_residual_fusion = None
            self.rgb_thermal_residual_head_feature = RGB_FEATURE
            self.rgb_thermal_residual_thermal_feature = None
            self.resvit_attention_fusion = None
            self.rgb_thermal_resvit_attention_head_feature = RGB_FEATURE
            self.rgb_thermal_resvit_attention_thermal_feature = None
            self.thermal_vision_tower = None
            self.thermal_multi_modal_projector = None
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
    image_layout = [
        item for item in model._last_prefix_token_layout if item["kind"].startswith("image")
    ]
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


@pytest.mark.parametrize("thermal_input_type", ["ViT", "ResViT", "ResViTAttention"])
def test_vit_projector_sharing_modes_route_thermal_tokens(thermal_input_type):
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
                    THERMAL_FEATURE: SimpleNamespace(shape=(1, 8, 8)),
                },
                uses_thermal_resnet18_encoder=False,
                uses_anythermal_encoder=False,
                uses_thermal_cvae_encoder=False,
                uses_dedicated_thermal_vit=False,
                uses_shared_projector_thermal_vit=thermal_input_type == "ViT",
                uses_residual_thermal_vit=thermal_input_type == "ResViT",
                uses_resvit_attention_encoder=thermal_input_type == "ResViTAttention",
                uses_twogrey_thermal_input=False,
                thermal_cvae_source_feature=RGB_FEATURE,
            )
            self.gradient_checkpointing_enabled = False
            self.training = False
            self.rgb_calls = 0
            self.shared_projector_thermal_calls = 0
            self.resvit_calls = 0
            self.resvit_attention_calls = 0
            self.rgb_encoder = None
            self.thermal_image_feature_set = {THERMAL_FEATURE}
            self.thermal_encoder = None
            self.thermal_head_fusion = None
            self.rgb_thermal_align_aligner = None
            self.rgb_thermal_align_thermal_feature = None
            self.rgb_thermal_align_head_feature = RGB_FEATURE
            self.thermal_residual_fusion = None
            self.rgb_thermal_residual_head_feature = RGB_FEATURE
            self.rgb_thermal_residual_thermal_feature = None
            self.rgb_thermal_resvit_head_feature = RGB_FEATURE
            self.rgb_thermal_resvit_thermal_feature = (
                THERMAL_FEATURE if thermal_input_type == "ResViT" else None
            )
            self.rgb_thermal_resvit_thermal_feature_set = (
                {THERMAL_FEATURE} if thermal_input_type == "ResViT" else set()
            )
            self.rgb_thermal_resvit_twogrey_cold_feature = None
            self.rgb_thermal_resvit_twogrey_hot_feature = None
            self.resvit_attention_fusion = (
                object() if thermal_input_type == "ResViTAttention" else None
            )
            self.rgb_thermal_resvit_attention_head_feature = RGB_FEATURE
            self.rgb_thermal_resvit_attention_thermal_feature = (
                THERMAL_FEATURE if thermal_input_type == "ResViTAttention" else None
            )
            self.thermal_vision_tower = object()
            self.thermal_multi_modal_projector = None
            self.paligemma_with_expert = DummyLanguageModel()
            self._last_prefix_token_layout = []

        def _embed_rgb_image(self, image):
            self.rgb_calls += 1
            return torch.zeros(image.shape[0], 4, 8)

        def _embed_shared_projector_thermal_vit_image(self, image):
            self.shared_projector_thermal_calls += 1
            return torch.zeros(image.shape[0], 4, 8)

        def _embed_resvit_head_image(self, **kwargs):
            self.resvit_calls += 1
            rgb_img = kwargs["rgb_img"]
            rgb_img_mask = kwargs["rgb_img_mask"]
            return torch.zeros(rgb_img.shape[0], 4, 8), rgb_img_mask[:, None].expand(rgb_img.shape[0], 4)

        def _embed_resvit_attention_head_image(self, **kwargs):
            self.resvit_attention_calls += 1
            rgb_img = kwargs["rgb_img"]
            rgb_img_mask = kwargs["rgb_img_mask"]
            return torch.zeros(rgb_img.shape[0], 4, 8), rgb_img_mask[:, None].expand(rgb_img.shape[0], 4)

    model = DummyModel()
    PI05Pytorch.embed_prefix(
        model,
        images=[
            torch.zeros(2, 3, 8, 8),
            torch.zeros(2, 3, 8, 8),
            torch.zeros(2, 1, 8, 8),
        ],
        img_masks=[torch.ones(2, dtype=torch.bool) for _ in range(3)],
        tokens=torch.zeros(2, 3, dtype=torch.long),
        masks=torch.ones(2, 3, dtype=torch.bool),
    )

    if thermal_input_type == "ViT":
        assert model.rgb_calls == 2
        assert model.shared_projector_thermal_calls == 1
        assert model.resvit_calls == 0
        assert model.resvit_attention_calls == 0
    elif thermal_input_type == "ResViT":
        assert model.rgb_calls == 1
        assert model.shared_projector_thermal_calls == 0
        assert model.resvit_calls == 1
        assert model.resvit_attention_calls == 0
    else:
        assert model.rgb_calls == 1
        assert model.shared_projector_thermal_calls == 0
        assert model.resvit_calls == 0
        assert model.resvit_attention_calls == 1

    image_layout_kinds = [
        item["kind"] for item in model._last_prefix_token_layout if item["kind"].startswith("image")
    ]
    assert ("image_resvit_residual" in image_layout_kinds) == (thermal_input_type == "ResViT")
    assert ("image_resvit_attention" in image_layout_kinds) == (
        thermal_input_type == "ResViTAttention"
    )


def test_resvit_adds_raw_tokens_before_shared_rgb_projector():
    class DummyPaliGemma:
        def __init__(self):
            self.rgb_vision_tower = object()
            self.thermal_vision_tower = object()
            self.rgb_projector = object()
            self.projector_input = None
            self.paligemma = SimpleNamespace(
                model=SimpleNamespace(
                    vision_tower=self.rgb_vision_tower,
                    multi_modal_projector=self.rgb_projector,
                )
            )

        def embed_image_tokens_with_modules(self, image, vision_tower):
            value = 2.0 if vision_tower is self.rgb_vision_tower else 3.0
            return torch.full((image.shape[0], 4, 6), value)

        def project_image_tokens_with_modules(self, image_tokens, projector, out_dtype):
            assert projector is self.rgb_projector
            self.projector_input = image_tokens
            return image_tokens.to(dtype=out_dtype)

    class DummyModel:
        _prepare_thermal_vit_image = PI05Pytorch._prepare_thermal_vit_image
        _project_siglip_tokens_with_rgb_projector = PI05Pytorch._project_siglip_tokens_with_rgb_projector
        _embed_resvit_head_image = PI05Pytorch._embed_resvit_head_image

        def __init__(self):
            self.paligemma_with_expert = DummyPaliGemma()
            self.thermal_vision_tower = self.paligemma_with_expert.thermal_vision_tower

    model = DummyModel()
    fused, token_mask = model._embed_resvit_head_image(
        rgb_img=torch.zeros(2, 3, 8, 8),
        thermal_img=torch.zeros(2, 1, 8, 8),
        rgb_img_mask=torch.tensor([True, True]),
        thermal_img_mask=torch.tensor([True, False]),
    )

    expected_projector_input = torch.stack(
        [torch.full((4, 6), 5.0), torch.full((4, 6), 2.0)],
        dim=0,
    )
    torch.testing.assert_close(model.paligemma_with_expert.projector_input, expected_projector_input)
    torch.testing.assert_close(fused, expected_projector_input)
    assert token_mask.tolist() == [[True] * 4, [True] * 4]


def test_twogrey_resvit_adds_scaled_cold_and_hot_raw_tokens_to_rgb():
    class DummyPaliGemma:
        def __init__(self):
            self.rgb_vision_tower = object()
            self.thermal_vision_tower = object()
            self.rgb_projector = object()
            self.projector_input = None
            self.paligemma = SimpleNamespace(
                model=SimpleNamespace(
                    vision_tower=self.rgb_vision_tower,
                    multi_modal_projector=self.rgb_projector,
                )
            )

        def embed_image_tokens_with_modules(self, image, vision_tower):
            if vision_tower is self.rgb_vision_tower:
                value = 2.0
            else:
                assert vision_tower is self.thermal_vision_tower
                value = float(image.mean().item())
            return torch.full((image.shape[0], 4, 6), value)

        def project_image_tokens_with_modules(self, image_tokens, projector, out_dtype):
            assert projector is self.rgb_projector
            self.projector_input = image_tokens
            return image_tokens.to(dtype=out_dtype)

    class DummyModel:
        _prepare_thermal_vit_image = PI05Pytorch._prepare_thermal_vit_image
        _project_siglip_tokens_with_rgb_projector = PI05Pytorch._project_siglip_tokens_with_rgb_projector
        _embed_twogrey_resvit_head_image = PI05Pytorch._embed_twogrey_resvit_head_image

        def __init__(self):
            self.config = SimpleNamespace(
                thermal_twogrey_resvit_cold_alpha=0.5,
                thermal_twogrey_resvit_hot_beta=2.0,
            )
            self.paligemma_with_expert = DummyPaliGemma()
            self.thermal_vision_tower = self.paligemma_with_expert.thermal_vision_tower

    model = DummyModel()
    fused, token_mask = model._embed_twogrey_resvit_head_image(
        rgb_img=torch.zeros(2, 3, 8, 8),
        cold_img=torch.stack(
            [torch.full((1, 8, 8), 4.0), torch.full((1, 8, 8), 4.0)]
        ),
        hot_img=torch.stack(
            [torch.full((1, 8, 8), 7.0), torch.full((1, 8, 8), 7.0)]
        ),
        rgb_img_mask=torch.tensor([True, True]),
        cold_img_mask=torch.tensor([True, False]),
        hot_img_mask=torch.tensor([False, True]),
    )

    expected_projector_input = torch.stack(
        [torch.full((4, 6), 4.0), torch.full((4, 6), 16.0)],
        dim=0,
    )
    torch.testing.assert_close(model.paligemma_with_expert.projector_input, expected_projector_input)
    torch.testing.assert_close(fused, expected_projector_input)
    assert token_mask.tolist() == [[True] * 4, [True] * 4]


def test_twogrey_gatevit_fusion_returns_one_normalized_thermal_stream():
    class FixedBatchGate(nn.Module):
        def forward(self, gate_input):
            return gate_input.new_tensor([[4.0, -4.0], [-4.0, 4.0]])

    fusion = TwoGreyGateViTFusion(
        embed_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.0,
    )
    fusion.gate_mlp = FixedBatchGate()

    fused, fused_mask, gate_info = fusion(
        cold_tokens=torch.ones(2, 4, 4),
        hot_tokens=3.0 * torch.ones(2, 4, 4),
        text_tokens=torch.ones(2, 2, 4),
        cold_token_mask=torch.ones(2, 4, dtype=torch.bool),
        hot_token_mask=torch.ones(2, 4, dtype=torch.bool),
        text_token_mask=torch.ones(2, 2, dtype=torch.bool),
    )

    torch.testing.assert_close(
        gate_info["cold_weight"] + gate_info["hot_weight"],
        torch.ones(2),
    )
    assert gate_info["cold_weight"][0] > 0.99
    assert gate_info["hot_weight"][1] > 0.99
    assert fused[0].mean() < 1.01
    assert fused[1].mean() > 2.99
    assert fused_mask.all()


def test_twogrey_gateactionvit_keeps_streams_and_balanced_gate_has_zero_bias():
    class BalancedGate(nn.Module):
        def forward(self, gate_input):
            return torch.zeros(gate_input.shape[0], 2, device=gate_input.device)

    fusion = TwoGreyGateActionViTFusion(
        embed_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.0,
        bias_init_strength=0.5,
    )
    fusion.gate_mlp = BalancedGate()
    cold_tokens = torch.randn(2, 4, 4)
    hot_tokens = torch.randn(2, 4, 4)
    token_mask = torch.ones(2, 4, dtype=torch.bool)

    cold_out, cold_mask, hot_out, hot_mask, attention_bias, gate_info = fusion(
        cold_tokens=cold_tokens,
        hot_tokens=hot_tokens,
        text_tokens=torch.randn(2, 3, 4),
        cold_token_mask=token_mask,
        hot_token_mask=token_mask,
        text_token_mask=torch.ones(2, 3, dtype=torch.bool),
    )

    torch.testing.assert_close(cold_out, cold_tokens)
    torch.testing.assert_close(hot_out, hot_tokens)
    assert cold_mask.all()
    assert hot_mask.all()
    torch.testing.assert_close(attention_bias, torch.zeros_like(attention_bias))
    torch.testing.assert_close(gate_info["cold_weight"], torch.full((2,), 0.5))
    torch.testing.assert_close(gate_info["hot_weight"], torch.full((2,), 0.5))


def test_twogrey_gateactionvit_bias_direction_and_gradient_reach_gate():
    class FixedBatchGate(nn.Module):
        def forward(self, gate_input):
            return gate_input.new_tensor([[4.0, -4.0], [-4.0, 4.0]])

    directional_fusion = TwoGreyGateActionViTFusion(
        embed_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.0,
        bias_init_strength=0.5,
    )
    directional_fusion.gate_mlp = FixedBatchGate()
    token_mask = torch.ones(2, 4, dtype=torch.bool)
    *_, attention_bias, gate_info = directional_fusion(
        cold_tokens=torch.randn(2, 4, 4),
        hot_tokens=torch.randn(2, 4, 4),
        text_tokens=torch.randn(2, 3, 4),
        cold_token_mask=token_mask,
        hot_token_mask=token_mask,
        text_token_mask=torch.ones(2, 3, dtype=torch.bool),
    )

    assert attention_bias[0, 0] > 0
    assert attention_bias[0, 1] < 0
    assert attention_bias[1, 0] < 0
    assert attention_bias[1, 1] > 0
    torch.testing.assert_close(attention_bias.sum(dim=-1), torch.zeros(2), atol=1e-6, rtol=0)
    assert gate_info["cold_weight"][0] > gate_info["hot_weight"][0]

    trainable_fusion = TwoGreyGateActionViTFusion(
        embed_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.02,
        bias_init_strength=0.5,
    )
    *_, trainable_bias, _ = trainable_fusion(
        cold_tokens=torch.randn(2, 4, 4),
        hot_tokens=torch.randn(2, 4, 4),
        text_tokens=torch.randn(2, 3, 4),
        cold_token_mask=token_mask,
        hot_token_mask=token_mask,
        text_token_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    (trainable_bias[:, 0] - trainable_bias[:, 1]).sum().backward()

    assert trainable_fusion.gate_mlp[-1].weight.grad is not None
    assert trainable_fusion.gate_mlp[-1].weight.grad.abs().sum() > 0
    assert trainable_fusion.raw_bias_strength.grad is not None


def test_twogrey_doublegatevit_combines_text_and_rgb_before_merging():
    class FixedTextGate(nn.Module):
        def forward(self, gate_input):
            return gate_input.new_tensor([[3.0, -3.0], [3.0, -3.0]])

    class FixedRGBGate(nn.Module):
        def forward(self, gate_input):
            return gate_input.new_tensor([[0.0, 0.0], [-8.0, 8.0]])

    fusion = TwoGreyDoubleGateViTFusion(
        embed_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.0,
        rgb_gate_max_strength=1.0,
        rgb_gate_init_strength=1.0,
    )
    fusion.gate_mlp = FixedTextGate()
    fusion.rgb_gate_mlp = FixedRGBGate()
    token_mask = torch.ones(2, 4, dtype=torch.bool)

    fused, fused_mask, gate_info = fusion(
        rgb_tokens=torch.randn(2, 4, 4),
        cold_tokens=torch.ones(2, 4, 4),
        hot_tokens=3.0 * torch.ones(2, 4, 4),
        text_tokens=torch.randn(2, 3, 4),
        rgb_token_mask=token_mask,
        cold_token_mask=token_mask,
        hot_token_mask=token_mask,
        text_token_mask=torch.ones(2, 3, dtype=torch.bool),
    )

    assert gate_info["text_cold_weight"][0] > 0.99
    assert gate_info["rgb_cold_alignment_weight"][0] == pytest.approx(0.5)
    assert gate_info["rgb_hot_alignment_weight"][1] > 0.99
    assert gate_info["cold_weight"][0] > 0.99
    assert gate_info["hot_weight"][1] > 0.99
    assert fused[0].mean() < 1.01
    assert fused[1].mean() > 2.99
    assert fused_mask.all()


def test_twogrey_doublegatevit_gradients_reach_both_gate_branches():
    fusion = TwoGreyDoubleGateViTFusion(
        embed_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.02,
        rgb_gate_init_strength=0.5,
    )
    token_mask = torch.ones(2, 4, dtype=torch.bool)
    fused, _, _ = fusion(
        rgb_tokens=torch.randn(2, 4, 4),
        cold_tokens=torch.randn(2, 4, 4),
        hot_tokens=2.0 + torch.randn(2, 4, 4),
        text_tokens=torch.randn(2, 3, 4),
        rgb_token_mask=token_mask,
        cold_token_mask=token_mask,
        hot_token_mask=token_mask,
        text_token_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    fused.mean().backward()

    assert fusion.gate_mlp[-1].weight.grad is not None
    assert fusion.gate_mlp[-1].weight.grad.abs().sum() > 0
    assert fusion.rgb_gate_mlp[-1].weight.grad is not None
    assert fusion.rgb_gate_mlp[-1].weight.grad.abs().sum() > 0
    assert fusion.raw_rgb_gate_strength.grad is not None


def test_twogrey_doublegatetworesvit_combines_rgb_gate_and_emits_two_streams():
    class FixedTextGate(nn.Module):
        def forward(self, gate_input):
            return gate_input.new_tensor([[3.0, -3.0], [3.0, -3.0]])

    class FixedRGBGate(nn.Module):
        def forward(self, gate_input):
            return gate_input.new_tensor([[0.0, 0.0], [-8.0, 8.0]])

    fusion = TwoGreyDoubleGateTwoResViTFusion(
        embed_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.0,
        rgb_gate_max_strength=1.0,
        rgb_gate_init_strength=1.0,
    )
    fusion.gate_mlp = FixedTextGate()
    fusion.rgb_gate_mlp = FixedRGBGate()
    token_mask = torch.ones(2, 4, dtype=torch.bool)
    rgb_tokens = torch.full((2, 4, 4), 10.0)
    cold_tokens = torch.ones(2, 4, 4)
    hot_tokens = 3.0 * torch.ones(2, 4, 4)
    (
        cold_res,
        cold_res_mask,
        hot_res,
        hot_res_mask,
        gate_info,
    ) = fusion(
        rgb_tokens=rgb_tokens,
        cold_tokens=cold_tokens,
        hot_tokens=hot_tokens,
        text_tokens=torch.randn(2, 3, 4),
        rgb_token_mask=token_mask,
        cold_token_mask=token_mask,
        hot_token_mask=token_mask,
        text_token_mask=torch.ones(2, 3, dtype=torch.bool),
        base_cold_alpha=0.5,
        base_hot_beta=0.25,
    )

    assert gate_info["text_cold_weight"][0] > 0.99
    assert gate_info["rgb_cold_alignment_weight"][0] == pytest.approx(0.5)
    assert gate_info["rgb_hot_alignment_weight"][1] > 0.99
    assert gate_info["cold_alpha"][0] > gate_info["hot_beta"][0]
    assert gate_info["cold_alpha"][1] < gate_info["hot_beta"][1]
    assert cold_res.shape == rgb_tokens.shape
    assert hot_res.shape == rgb_tokens.shape
    assert cold_res_mask.all()
    assert hot_res_mask.all()
    assert cold_res[0].mean() > 10.9
    assert hot_res[1].mean() > 11.4


def test_twogrey_patch_evidence_follows_cold_and_hot_image_regions():
    boundary = 2.0 * 80.0 / 255.0 - 1.0
    cold = torch.full((1, 3, 4, 4), boundary)
    hot = torch.full((1, 3, 4, 4), boundary)
    cold[:, :, :2, :2] = -1.0
    hot[:, :, 2:, 2:] = 1.0

    cold_evidence, hot_evidence = twogrey_patch_evidence(
        cold,
        hot,
        target_background=80.0,
        token_count=4,
    )

    torch.testing.assert_close(cold_evidence, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    torch.testing.assert_close(hot_evidence, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))


def test_twoblack_patch_evidence_treats_dark_pixels_as_importance():
    cold = torch.ones((1, 3, 4, 4))
    hot = torch.ones((1, 3, 4, 4))
    cold[:, :, :2, :2] = -1.0
    hot[:, :, 2:, 2:] = -1.0

    cold_evidence, hot_evidence = twogrey_patch_evidence(
        cold,
        hot,
        target_background=80.0,
        token_count=4,
        black_importance=True,
    )

    torch.testing.assert_close(cold_evidence, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    torch.testing.assert_close(hot_evidence, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))


def test_twogrey_patchgatevit_has_visible_patchwise_temperature_selection():
    fusion = TwoGreyPatchGateViTFusion(
        embed_dim=8,
        gate_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.0,
        evidence_strength=2.0,
        evidence_floor=0.01,
    )
    token_mask = torch.ones(1, 4, dtype=torch.bool)
    fused, fused_mask, gate_info = fusion(
        rgb_tokens=torch.randn(1, 6, 8),
        cold_tokens=torch.ones(1, 4, 8),
        hot_tokens=3.0 * torch.ones(1, 4, 8),
        text_tokens=torch.randn(1, 3, 8),
        rgb_token_mask=torch.ones(1, 6, dtype=torch.bool),
        cold_token_mask=token_mask,
        hot_token_mask=token_mask,
        text_token_mask=torch.ones(1, 3, dtype=torch.bool),
        cold_evidence=torch.tensor([[1.0, 0.0, 0.8, 0.0]]),
        hot_evidence=torch.tensor([[0.0, 1.0, 0.0, 0.8]]),
    )

    assert gate_info["cold_weight"][0, 0] > 0.99
    assert gate_info["hot_weight"][0, 1] > 0.99
    assert gate_info["cold_weight"][0, 2] > 0.99
    assert gate_info["hot_weight"][0, 3] > 0.99
    torch.testing.assert_close(
        gate_info["cold_weight"] + gate_info["hot_weight"],
        torch.ones(1, 4),
    )
    torch.testing.assert_close(gate_info["patch_relevance"], torch.ones(1, 4))
    torch.testing.assert_close(fused[0, 0], torch.ones(8), atol=1e-3, rtol=0)
    torch.testing.assert_close(fused[0, 1], torch.full((8,), 3.0), atol=1e-3, rtol=0)
    assert fused_mask.all()


def test_twogrey_smallpatchgatevit_uses_coarse_4x4_gate_grid():
    fusion = TwoGreySmallPatchGateViTFusion(
        embed_dim=8,
        gate_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.0,
        evidence_strength=2.0,
        evidence_floor=0.01,
    )
    token_mask = torch.ones(1, 64, dtype=torch.bool)
    fused, fused_mask, gate_info = fusion(
        rgb_tokens=torch.randn(1, 64, 8),
        cold_tokens=torch.ones(1, 64, 8),
        hot_tokens=3.0 * torch.ones(1, 64, 8),
        text_tokens=torch.randn(1, 3, 8),
        rgb_token_mask=token_mask,
        cold_token_mask=token_mask,
        hot_token_mask=token_mask,
        text_token_mask=torch.ones(1, 3, dtype=torch.bool),
        cold_evidence=torch.ones(1, 64),
        hot_evidence=torch.zeros(1, 64),
    )

    assert fusion.gate_grid == (4, 4)
    assert gate_info["small_patch_cold_weight"].shape == (1, 16)
    assert gate_info["small_patch_hot_weight"].shape == (1, 16)
    assert gate_info["cold_weight"].shape == (1, 64)
    assert gate_info["hot_weight"].shape == (1, 64)
    torch.testing.assert_close(
        gate_info["small_patch_cold_weight"] + gate_info["small_patch_hot_weight"],
        torch.ones(1, 16),
    )
    torch.testing.assert_close(
        gate_info["cold_weight"] + gate_info["hot_weight"],
        torch.ones(1, 64),
    )
    assert gate_info["small_patch_cold_weight"].mean() > 0.99
    torch.testing.assert_close(gate_info["patch_relevance"], torch.ones(1, 64))
    torch.testing.assert_close(fused, torch.ones(1, 64, 8), atol=1e-3, rtol=0)
    assert fused_mask.all()


def test_twogrey_hardsmallpatchgatevit_hard_routes_and_keeps_soft_gradients():
    fusion = TwoGreyHardSmallPatchGateViTFusion(
        embed_dim=8,
        gate_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.01,
        evidence_strength=2.0,
        evidence_floor=0.01,
    )
    with torch.no_grad():
        fusion.patch_gate_mlp[-1].bias.copy_(torch.tensor([2.0, -2.0, 0.0]))

    token_mask = torch.ones(1, 64, dtype=torch.bool)
    cold_tokens = torch.ones(1, 64, 8, requires_grad=True)
    hot_tokens = (3.0 * torch.ones(1, 64, 8)).requires_grad_()
    fused, fused_mask, gate_info = fusion(
        rgb_tokens=torch.randn(1, 64, 8),
        cold_tokens=cold_tokens,
        hot_tokens=hot_tokens,
        text_tokens=torch.randn(1, 3, 8),
        rgb_token_mask=token_mask,
        cold_token_mask=token_mask,
        hot_token_mask=token_mask,
        text_token_mask=torch.ones(1, 3, dtype=torch.bool),
        # Opposite evidence deliberately proves that this mode does not use
        # the hand-crafted image-evidence prior.
        cold_evidence=torch.zeros(1, 64),
        hot_evidence=torch.ones(1, 64),
    )

    assert fusion.gate_grid == (4, 4)
    assert torch.equal(fused, torch.ones_like(fused))
    assert fused_mask.all()
    torch.testing.assert_close(gate_info["hard_cold_weight"], torch.ones(1, 64))
    torch.testing.assert_close(gate_info["hard_hot_weight"], torch.zeros(1, 64))
    torch.testing.assert_close(
        gate_info["small_patch_hard_cold_weight"], torch.ones(1, 16)
    )
    assert gate_info["soft_hot_weight"].min() > 0
    assert gate_info["small_patch_soft_hot_weight"].min() > 0

    fused.square().mean().backward()
    assert cold_tokens.grad is not None
    assert cold_tokens.grad.abs().sum() > 0
    assert hot_tokens.grad is not None
    assert hot_tokens.grad.abs().sum() > 0
    assert fusion.patch_gate_mlp[-1].bias.grad is not None
    assert fusion.patch_gate_mlp[-1].bias.grad.abs().sum() > 0
    assert fusion.text_cross_attn.in_proj_weight.grad is not None
    assert fusion.text_cross_attn.in_proj_weight.grad.abs().sum() > 0


def test_twogrey_smallpatchgatetworesvit_emits_two_rgb_residual_streams():
    fusion = TwoGreySmallPatchGateTwoResViTFusion(
        embed_dim=8,
        gate_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.0,
        evidence_strength=2.0,
        evidence_floor=0.01,
    )
    token_mask = torch.ones(1, 64, dtype=torch.bool)
    rgb_tokens = torch.full((1, 64, 8), 10.0)
    cold_tokens = torch.ones(1, 64, 8)
    hot_tokens = 3.0 * torch.ones(1, 64, 8)
    (
        cold_res,
        cold_res_mask,
        hot_res,
        hot_res_mask,
        gate_info,
    ) = fusion(
        rgb_tokens=rgb_tokens,
        cold_tokens=cold_tokens,
        hot_tokens=hot_tokens,
        text_tokens=torch.randn(1, 3, 8),
        rgb_token_mask=token_mask,
        cold_token_mask=token_mask,
        hot_token_mask=token_mask,
        text_token_mask=torch.ones(1, 3, dtype=torch.bool),
        cold_evidence=torch.ones(1, 64),
        hot_evidence=torch.zeros(1, 64),
        base_cold_alpha=0.5,
        base_hot_beta=0.25,
    )

    assert cold_res.shape == rgb_tokens.shape
    assert hot_res.shape == rgb_tokens.shape
    assert cold_res_mask.all()
    assert hot_res_mask.all()
    assert fusion.gate_grid == (4, 4)
    assert gate_info["small_patch_cold_weight"].shape == (1, 16)
    assert gate_info["cold_residual_scale"].shape == (1, 64)
    assert gate_info["hot_residual_scale"].shape == (1, 64)
    assert torch.all(gate_info["cold_residual_scale"] > gate_info["hot_residual_scale"])
    torch.testing.assert_close(cold_res, rgb_tokens + gate_info["cold_residual_scale"][..., None])
    torch.testing.assert_close(
        hot_res,
        rgb_tokens + 3.0 * gate_info["hot_residual_scale"][..., None],
    )


def test_twogrey_patchsinglegateresvit_uses_full_patch_text_only_residual_gate():
    fusion = TwoGreyPatchSingleGateResViTFusion(
        embed_dim=8,
        gate_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.0,
        evidence_strength=2.0,
        evidence_floor=0.01,
    )
    token_mask = torch.ones(1, 64, dtype=torch.bool)
    rgb_tokens = torch.full((1, 64, 8), 10.0)
    cold_tokens = torch.ones(1, 64, 8)
    hot_tokens = 3.0 * torch.ones(1, 64, 8)
    fused, fused_mask, gate_info = fusion(
        rgb_tokens=rgb_tokens,
        cold_tokens=cold_tokens,
        hot_tokens=hot_tokens,
        text_tokens=torch.randn(1, 3, 8),
        rgb_token_mask=token_mask,
        cold_token_mask=token_mask,
        hot_token_mask=token_mask,
        text_token_mask=torch.ones(1, 3, dtype=torch.bool),
        cold_evidence=torch.ones(1, 64),
        hot_evidence=torch.zeros(1, 64),
        base_cold_alpha=0.5,
        base_hot_beta=0.25,
    )

    assert fusion.gate_input_norm.normalized_shape == (20,)
    assert not any("rgb_cross_attn" in name for name, _ in fusion.named_parameters())
    assert fused.shape == rgb_tokens.shape
    assert fused_mask.all()
    assert gate_info["cold_weight"].shape == (1, 64)
    assert "small_patch_cold_weight" not in gate_info
    assert gate_info["cold_residual_scale"].shape == (1, 64)
    assert torch.all(gate_info["cold_residual_scale"] > gate_info["hot_residual_scale"])
    expected = (
        rgb_tokens
        + gate_info["cold_residual_scale"][..., None]
        + 3.0 * gate_info["hot_residual_scale"][..., None]
    )
    torch.testing.assert_close(fused, expected)


def test_twogrey_smallpatchgateresvit_fuses_patch_scaled_residuals_into_one_rgb_stream():
    fusion = TwoGreySmallPatchGateResViTFusion(
        embed_dim=8,
        gate_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.0,
        evidence_strength=2.0,
        evidence_floor=0.01,
    )
    token_mask = torch.ones(1, 64, dtype=torch.bool)
    rgb_tokens = torch.full((1, 64, 8), 10.0)
    cold_tokens = torch.ones(1, 64, 8)
    hot_tokens = 3.0 * torch.ones(1, 64, 8)
    fused, fused_mask, gate_info = fusion(
        rgb_tokens=rgb_tokens,
        cold_tokens=cold_tokens,
        hot_tokens=hot_tokens,
        text_tokens=torch.randn(1, 3, 8),
        rgb_token_mask=token_mask,
        cold_token_mask=token_mask,
        hot_token_mask=token_mask,
        text_token_mask=torch.ones(1, 3, dtype=torch.bool),
        cold_evidence=torch.ones(1, 64),
        hot_evidence=torch.zeros(1, 64),
        base_cold_alpha=0.5,
        base_hot_beta=0.25,
    )

    assert fused.shape == rgb_tokens.shape
    assert fused_mask.all()
    assert fusion.gate_grid == (4, 4)
    assert gate_info["small_patch_cold_weight"].shape == (1, 16)
    assert gate_info["cold_residual_scale"].shape == (1, 64)
    assert gate_info["hot_residual_scale"].shape == (1, 64)
    assert torch.all(gate_info["cold_residual_scale"] > gate_info["hot_residual_scale"])
    expected = (
        rgb_tokens
        + gate_info["cold_residual_scale"][..., None]
        + 3.0 * gate_info["hot_residual_scale"][..., None]
    )
    torch.testing.assert_close(fused, expected)


def test_twogrey_smallpatchsinglegateresvit_uses_text_only_gate_context():
    fusion = TwoGreySmallPatchSingleGateResViTFusion(
        embed_dim=8,
        gate_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.0,
        evidence_strength=2.0,
        evidence_floor=0.01,
    )
    token_mask = torch.ones(1, 64, dtype=torch.bool)
    rgb_tokens = torch.full((1, 64, 8), 10.0)
    cold_tokens = torch.ones(1, 64, 8)
    hot_tokens = 3.0 * torch.ones(1, 64, 8)
    fused, fused_mask, gate_info = fusion(
        rgb_tokens=rgb_tokens,
        cold_tokens=cold_tokens,
        hot_tokens=hot_tokens,
        text_tokens=torch.randn(1, 3, 8),
        rgb_token_mask=token_mask,
        cold_token_mask=token_mask,
        hot_token_mask=token_mask,
        text_token_mask=torch.ones(1, 3, dtype=torch.bool),
        cold_evidence=torch.ones(1, 64),
        hot_evidence=torch.zeros(1, 64),
        base_cold_alpha=0.5,
        base_hot_beta=0.25,
    )

    assert fusion.gate_input_norm.normalized_shape == (20,)
    assert not any("rgb_cross_attn" in name for name, _ in fusion.named_parameters())
    assert fused.shape == rgb_tokens.shape
    assert fused_mask.all()
    assert gate_info["small_patch_cold_weight"].shape == (1, 16)
    assert torch.all(gate_info["cold_residual_scale"] > gate_info["hot_residual_scale"])


def test_twogrey_smallpatchsinglegatetworesvit_uses_text_only_gate_and_two_streams():
    fusion = TwoGreySmallPatchSingleGateTwoResViTFusion(
        embed_dim=8,
        gate_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.0,
        evidence_strength=2.0,
        evidence_floor=0.01,
    )
    token_mask = torch.ones(1, 64, dtype=torch.bool)
    rgb_tokens = torch.full((1, 64, 8), 10.0)
    cold_tokens = torch.ones(1, 64, 8)
    hot_tokens = 3.0 * torch.ones(1, 64, 8)
    (
        cold_res,
        cold_res_mask,
        hot_res,
        hot_res_mask,
        gate_info,
    ) = fusion(
        rgb_tokens=rgb_tokens,
        cold_tokens=cold_tokens,
        hot_tokens=hot_tokens,
        text_tokens=torch.randn(1, 3, 8),
        rgb_token_mask=token_mask,
        cold_token_mask=token_mask,
        hot_token_mask=token_mask,
        text_token_mask=torch.ones(1, 3, dtype=torch.bool),
        cold_evidence=torch.ones(1, 64),
        hot_evidence=torch.zeros(1, 64),
        base_cold_alpha=0.5,
        base_hot_beta=0.25,
    )

    assert fusion.gate_input_norm.normalized_shape == (20,)
    assert not any("rgb_cross_attn" in name for name, _ in fusion.named_parameters())
    assert cold_res.shape == rgb_tokens.shape
    assert hot_res.shape == rgb_tokens.shape
    assert cold_res_mask.all()
    assert hot_res_mask.all()
    assert fusion.gate_grid == (4, 4)
    assert gate_info["small_patch_cold_weight"].shape == (1, 16)
    assert gate_info["cold_residual_scale"].shape == (1, 64)
    assert gate_info["hot_residual_scale"].shape == (1, 64)
    assert torch.all(gate_info["cold_residual_scale"] > gate_info["hot_residual_scale"])
    torch.testing.assert_close(cold_res, rgb_tokens + gate_info["cold_residual_scale"][..., None])
    torch.testing.assert_close(
        hot_res,
        rgb_tokens + 3.0 * gate_info["hot_residual_scale"][..., None],
    )


def test_twogrey_patchgatevit_gradients_reach_lightweight_gate_branches():
    fusion = TwoGreyPatchGateViTFusion(
        embed_dim=8,
        gate_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.02,
        detach_head_rgb=True,
    )
    token_mask = torch.ones(2, 4, dtype=torch.bool)
    rgb_tokens = torch.randn(2, 4, 8, requires_grad=True)
    fused, _, _ = fusion(
        rgb_tokens=rgb_tokens,
        cold_tokens=torch.randn(2, 4, 8),
        hot_tokens=2.0 + torch.randn(2, 4, 8),
        text_tokens=torch.randn(2, 3, 8),
        rgb_token_mask=token_mask,
        cold_token_mask=token_mask,
        hot_token_mask=token_mask,
        text_token_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    fused.square().mean().backward()

    assert fusion.patch_gate_mlp[-1].weight.grad is not None
    assert fusion.patch_gate_mlp[-1].weight.grad.abs().sum() > 0
    assert fusion.global_text_gate[-1].weight.grad is not None
    assert fusion.global_text_gate[-1].weight.grad.abs().sum() > 0
    assert fusion.image_adapter.weight.grad is not None
    assert fusion.image_adapter.weight.grad.abs().sum() > 0
    assert fusion.text_adapter.weight.grad is not None
    assert fusion.text_adapter.weight.grad.abs().sum() > 0
    assert rgb_tokens.grad is None


def test_twogrey_patchgatevit_masks_missing_modalities_without_nan():
    fusion = TwoGreyPatchGateViTFusion(
        embed_dim=8,
        gate_dim=4,
        num_heads=2,
        hidden_dim=8,
    )
    fused, fused_mask, gate_info = fusion(
        rgb_tokens=torch.randn(2, 4, 8),
        cold_tokens=torch.randn(2, 4, 8),
        hot_tokens=torch.randn(2, 4, 8),
        text_tokens=torch.randn(2, 3, 8),
        rgb_token_mask=torch.tensor([[True] * 4, [False] * 4]),
        cold_token_mask=torch.tensor([[True] * 4, [False] * 4]),
        hot_token_mask=torch.tensor([[False] * 4, [False] * 4]),
        text_token_mask=torch.tensor([[True] * 3, [False] * 3]),
    )

    assert torch.isfinite(fused).all()
    assert torch.isfinite(gate_info["cold_weight"]).all()
    assert torch.isfinite(gate_info["hot_weight"]).all()
    assert torch.all(gate_info["cold_weight"][0] == 1)
    assert torch.all(gate_info["hot_weight"][0] == 0)
    assert fused_mask[0].all()
    assert not fused_mask[1].any()
    assert torch.all(fused[1] == 0)


def test_gatevitmix_zero_context_scale_preserves_independent_thermal_tokens():
    fusion = GateViTMixFusion(
        embed_dim=8,
        mix_dim=4,
        num_heads=2,
        context_scale=0.0,
    )
    rgb_tokens = torch.randn(2, 4, 8)
    thermal_tokens = torch.randn(2, 4, 8)
    original_rgb = rgb_tokens.clone()

    mixed_thermal, mix_info = fusion(
        rgb_tokens=rgb_tokens,
        thermal_tokens=thermal_tokens,
        rgb_token_mask=torch.ones(2, 4, dtype=torch.bool),
        thermal_token_mask=torch.ones(2, 4, dtype=torch.bool),
    )

    torch.testing.assert_close(mixed_thermal, thermal_tokens)
    torch.testing.assert_close(rgb_tokens, original_rgb)
    torch.testing.assert_close(
        mix_info["head_mix_delta_rms"],
        torch.zeros(2),
    )
    assert mix_info["head_mix_valid"].all()


def test_gatevitmix_detaches_thermal_branch_gradient_from_head_rgb():
    fusion = GateViTMixFusion(
        embed_dim=8,
        mix_dim=4,
        num_heads=2,
        context_scale=0.5,
        detach_head_rgb=True,
    )
    rgb_tokens = torch.randn(2, 4, 8, requires_grad=True)
    thermal_tokens = torch.randn(2, 4, 8, requires_grad=True)

    mixed_thermal, _ = fusion(
        rgb_tokens=rgb_tokens,
        thermal_tokens=thermal_tokens,
        rgb_token_mask=torch.ones(2, 4, dtype=torch.bool),
        thermal_token_mask=torch.ones(2, 4, dtype=torch.bool),
    )
    mixed_thermal.square().mean().backward()

    assert rgb_tokens.grad is None
    assert thermal_tokens.grad is not None
    assert thermal_tokens.grad.abs().sum() > 0
    assert fusion.rgb_to_mix.weight.grad is not None
    assert fusion.thermal_to_mix.weight.grad is not None


def test_gatevitmix_prefix_attention_isolates_thermal_but_keeps_action_access():
    model = SimpleNamespace(
        config=SimpleNamespace(uses_gate_thermal_vit_mix=True),
        _last_prefix_token_layout=[
            {"kind": "image", "start": 0, "end": 2},
            {"kind": "image_gatevitmix_twogrey", "start": 2, "end": 4},
            {"kind": "language", "start": 4, "end": 6},
        ],
    )
    full_mask = torch.ones(1, 8, 8, dtype=torch.bool)

    scoped = PI05Pytorch._apply_gatevitmix_prefix_attention_scope(
        model,
        full_mask,
        prefix_len=6,
    )

    assert scoped[:, :2, 2:4].sum() == 0
    assert scoped[:, 4:6, 2:4].sum() == 0
    assert scoped[:, 2:4, :2].sum() == 0
    assert scoped[:, 2:4, 4:6].sum() == 0
    assert scoped[:, 2:4, 2:4].all()
    assert scoped[:, 6:8, 2:4].all()
    assert scoped[:, :2, 4:6].all()


def test_gateactionvit_attention_bias_changes_only_action_to_cold_hot_logits():
    live_bias = torch.tensor([[0.4, -0.4]], requires_grad=True)
    model = SimpleNamespace(
        config=SimpleNamespace(uses_gate_action_thermal_vit=True),
        _current_gateactionvit_attention_bias=live_bias,
        _last_prefix_token_layout=[
            {"kind": "image", "start": 0, "end": 2},
            {"kind": "image_gateactionvit_twogrey_cold", "start": 2, "end": 4},
            {"kind": "image_gateactionvit_twogrey_hot", "start": 4, "end": 6},
            {"kind": "language", "start": 6, "end": 8},
        ],
    )
    full_mask = torch.zeros(1, 1, 10, 10)

    biased = PI05Pytorch._apply_gateactionvit_action_attention_bias(
        model,
        full_mask,
        prefix_len=8,
    )

    torch.testing.assert_close(biased[:, :, :8], torch.zeros(1, 1, 8, 10))
    torch.testing.assert_close(biased[:, :, 8:, 2:4], torch.full((1, 1, 2, 2), 0.4))
    torch.testing.assert_close(biased[:, :, 8:, 4:6], torch.full((1, 1, 2, 2), -0.4))
    torch.testing.assert_close(biased[:, :, 8:, :2], torch.zeros(1, 1, 2, 2))
    torch.testing.assert_close(biased[:, :, 8:, 6:], torch.zeros(1, 1, 2, 4))
    biased[:, :, 8:, 2:4].sum().backward()
    assert live_bias.grad is not None
    assert live_bias.grad[0, 0] > 0
    assert live_bias.grad[0, 1] == 0

    suffix_mask = torch.zeros(1, 1, 2, 10)
    suffix_biased = PI05Pytorch._apply_gateactionvit_action_attention_bias(
        model,
        suffix_mask,
        prefix_len=8,
        suffix_queries_only=True,
    )
    torch.testing.assert_close(
        suffix_biased[:, :, :, 2:4],
        torch.full((1, 1, 2, 2), 0.4),
    )
    torch.testing.assert_close(
        suffix_biased[:, :, :, 4:6],
        torch.full((1, 1, 2, 2), -0.4),
    )


def test_twogrey_gateresvit_fusion_text_tokens_bias_cold_and_hot_coefficients():
    class ReadFirstGate(nn.Module):
        def forward(self, gate_input):
            return torch.stack([gate_input[:, 0], -gate_input[:, 0]], dim=-1)

    fusion = TwoGreyGateResViTFusion(
        embed_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.0,
    )
    fusion.gate_mlp = ReadFirstGate()

    text_tokens = torch.tensor(
        [
            [[3.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]],
            [[-3.0, 0.0, 0.0, 0.0], [-3.0, 0.0, 0.0, 0.0]],
        ]
    )
    _, gate_info = fusion(
        rgb_tokens=torch.zeros(2, 4, 4),
        cold_tokens=torch.ones(2, 4, 4),
        hot_tokens=torch.ones(2, 4, 4),
        text_tokens=text_tokens,
        rgb_token_mask=torch.ones(2, 4, dtype=torch.bool),
        cold_token_mask=torch.ones(2, 4, dtype=torch.bool),
        hot_token_mask=torch.ones(2, 4, dtype=torch.bool),
        text_token_mask=torch.ones(2, 2, dtype=torch.bool),
        base_cold_alpha=1.0,
        base_hot_beta=1.0,
    )

    assert gate_info["cold_alpha"][0] > gate_info["hot_beta"][0]
    assert gate_info["cold_alpha"][1] < gate_info["hot_beta"][1]


def test_twogrey_gatetworesvit_fusion_emits_two_scalar_rgb_residual_streams():
    class ZeroGate(nn.Module):
        def forward(self, gate_input):
            return torch.zeros(gate_input.shape[0], 2, device=gate_input.device, dtype=gate_input.dtype)

    fusion = TwoGreyGateTwoResViTFusion(
        embed_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.0,
    )
    fusion.gate_mlp = ZeroGate()

    token_mask = torch.ones(1, 4, dtype=torch.bool)
    rgb_tokens = torch.full((1, 4, 4), 10.0)
    cold_tokens = torch.ones(1, 4, 4)
    hot_tokens = 3.0 * torch.ones(1, 4, 4)
    (
        cold_res,
        cold_res_mask,
        hot_res,
        hot_res_mask,
        gate_info,
    ) = fusion(
        rgb_tokens=rgb_tokens,
        cold_tokens=cold_tokens,
        hot_tokens=hot_tokens,
        text_tokens=torch.randn(1, 3, 4),
        rgb_token_mask=token_mask,
        cold_token_mask=token_mask,
        hot_token_mask=token_mask,
        text_token_mask=torch.ones(1, 3, dtype=torch.bool),
        base_cold_alpha=0.5,
        base_hot_beta=0.25,
    )

    assert cold_res.shape == rgb_tokens.shape
    assert hot_res.shape == rgb_tokens.shape
    assert cold_res_mask.all()
    assert hot_res_mask.all()
    torch.testing.assert_close(gate_info["cold_residual_scale"], torch.tensor([0.5]))
    torch.testing.assert_close(gate_info["hot_residual_scale"], torch.tensor([0.25]))
    torch.testing.assert_close(cold_res, torch.full_like(rgb_tokens, 10.5))
    torch.testing.assert_close(hot_res, torch.full_like(rgb_tokens, 10.75))


def test_grey_gateoneresvit_fusion_emits_one_scalar_rgb_residual_stream():
    class ZeroGate(nn.Module):
        def forward(self, gate_input):
            return torch.zeros(gate_input.shape[0], 1, device=gate_input.device, dtype=gate_input.dtype)

    fusion = OneGreyGateOneResViTFusion(
        embed_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.0,
    )
    fusion.gate_mlp = ZeroGate()

    token_mask = torch.ones(1, 4, dtype=torch.bool)
    rgb_tokens = torch.full((1, 4, 4), 10.0)
    thermal_tokens = 2.0 * torch.ones(1, 4, 4)
    fused, fused_mask, gate_info = fusion(
        rgb_tokens=rgb_tokens,
        thermal_tokens=thermal_tokens,
        text_tokens=torch.randn(1, 3, 4),
        rgb_token_mask=token_mask,
        thermal_token_mask=token_mask,
        text_token_mask=torch.ones(1, 3, dtype=torch.bool),
    )

    assert fused.shape == rgb_tokens.shape
    assert fused_mask.all()
    torch.testing.assert_close(gate_info["thermal_alpha"], torch.tensor([0.5]))
    torch.testing.assert_close(gate_info["thermal_residual_scale"], torch.tensor([0.5]))
    torch.testing.assert_close(fused, torch.full_like(rgb_tokens, 11.0))


def test_grey_gateoneresvit_preserves_rgb_when_thermal_is_invalid():
    class ZeroGate(nn.Module):
        def forward(self, gate_input):
            return torch.zeros(gate_input.shape[0], 1, device=gate_input.device, dtype=gate_input.dtype)

    fusion = OneGreyGateOneResViTFusion(
        embed_dim=4,
        num_heads=2,
        hidden_dim=8,
        gate_init_std=0.0,
    )
    fusion.gate_mlp = ZeroGate()

    rgb_mask = torch.ones(1, 4, dtype=torch.bool)
    thermal_mask = torch.zeros(1, 4, dtype=torch.bool)
    rgb_tokens = torch.full((1, 4, 4), 10.0)
    thermal_tokens = 2.0 * torch.ones(1, 4, 4)
    fused, fused_mask, gate_info = fusion(
        rgb_tokens=rgb_tokens,
        thermal_tokens=thermal_tokens,
        text_tokens=torch.randn(1, 3, 4),
        rgb_token_mask=rgb_mask,
        thermal_token_mask=thermal_mask,
        text_token_mask=torch.ones(1, 3, dtype=torch.bool),
    )

    assert fused_mask.all()
    torch.testing.assert_close(gate_info["thermal_alpha"], torch.tensor([0.0]))
    torch.testing.assert_close(fused, rgb_tokens)


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


def test_twogrey_gateresvit3_confidence_uses_thermal_statistics():
    class BalancedGate(nn.Module):
        def forward(self, gate_input):
            return torch.zeros(gate_input.shape[0], 2, device=gate_input.device, dtype=gate_input.dtype)

    fusion = TwoGreyGateResViT3Fusion(
        embed_dim=4,
        num_heads=2,
        hidden_dim=8,
        stat_hidden_dim=8,
        residual_hidden_dim=8,
        gate_init_std=0.0,
        confidence_bias_init=-3.0,
        evidence_gain=48.0,
        evidence_threshold=0.08,
    )
    fusion.gate_mlp = BalancedGate()

    thermal_stats = torch.zeros(2, TwoGreyGateResViT3Fusion.stat_dim)
    thermal_stats[1, 6] = 0.2
    thermal_stats[1, 12] = 0.05
    thermal_stats[1, 13] = 0.08
    thermal_stats[1, 14] = 0.4

    fused, gate_info = fusion(
        rgb_tokens=torch.zeros(2, 4, 4),
        cold_tokens=torch.ones(2, 4, 4),
        hot_tokens=2.0 * torch.ones(2, 4, 4),
        text_tokens=torch.ones(2, 2, 4),
        rgb_token_mask=torch.ones(2, 4, dtype=torch.bool),
        cold_token_mask=torch.ones(2, 4, dtype=torch.bool),
        hot_token_mask=torch.ones(2, 4, dtype=torch.bool),
        text_token_mask=torch.ones(2, 2, dtype=torch.bool),
        thermal_stats=thermal_stats,
        base_cold_alpha=1.0,
        base_hot_beta=1.0,
    )

    assert gate_info["thermal_confidence"][0] < 0.01
    assert gate_info["thermal_confidence"][1] > 0.95
    assert gate_info["thermal_evidence"][1] > gate_info["thermal_evidence"][0]
    assert gate_info["cold_alpha"][1] > 100 * gate_info["cold_alpha"][0]
    assert fused[1].abs().mean() > 100 * fused[0].abs().mean()


def test_twogrey_gateresvit3_thermal_stats_detect_contrast():
    boundary = -1.0 + 2.0 * (80.0 / 255.0)
    cold = torch.full((2, 3, 8, 8), boundary)
    hot = torch.full((2, 3, 8, 8), boundary)
    cold[1, :, 2:6, 2:6] = -1.0 + 2.0 * (40.0 / 255.0)
    hot[1, :, 1:3, 1:3] = -1.0 + 2.0 * (130.0 / 255.0)

    stats = TwoGreyGateResViT3Fusion.compute_thermal_stats(
        cold_images=cold,
        hot_images=hot,
        background_value=80.0,
    )

    assert stats.shape == (2, TwoGreyGateResViT3Fusion.stat_dim)
    assert stats[0, 14] == pytest.approx(0.0)
    assert stats[1, 14] > 0.15
    assert stats[1, 12] > stats[0, 12]


def test_twoblack_gateresvit3_thermal_stats_treat_dark_pixels_as_evidence():
    cold = torch.ones((2, 3, 8, 8))
    hot = torch.ones((2, 3, 8, 8))
    cold[1, :, 2:6, 2:6] = 0.25
    hot[1, :, 1:3, 1:3] = 0.5

    stats = TwoGreyGateResViT3Fusion.compute_thermal_stats(
        cold_images=cold,
        hot_images=hot,
        background_value=80.0,
        input_range="zero_to_one",
        black_importance=True,
    )

    assert stats.shape == (2, TwoGreyGateResViT3Fusion.stat_dim)
    assert stats[0, 14] == pytest.approx(0.0)
    assert stats[1, 14] == pytest.approx(0.75)
    assert stats[1, 12] > stats[0, 12]


def test_embed_prefix_routes_twogrey_gatevit_as_separate_thermal_slot():
    class DummyLanguageModel:
        def embed_language_tokens(self, tokens):
            return torch.zeros(tokens.shape[0], tokens.shape[1], 8)

    class DummyModel:
        _apply_checkpoint = PI05Pytorch._apply_checkpoint

        def __init__(self):
            self.config = SimpleNamespace(
                image_features={
                    RGB_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_COLD_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_HOT_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                },
                uses_thermal_resnet18_encoder=False,
                uses_anythermal_encoder=False,
                uses_thermal_cvae_encoder=False,
                uses_dedicated_thermal_vit=False,
                uses_shared_projector_thermal_vit=False,
                uses_residual_thermal_vit=False,
                uses_gate_thermal_vit=True,
                uses_gate_residual_thermal_vit=False,
                uses_gate_residual_thermal_vit3=False,
                uses_resvit_attention_encoder=False,
                uses_twogrey_thermal_input=True,
                thermal_cvae_source_feature=RGB_FEATURE,
            )
            self.gradient_checkpointing_enabled = False
            self.training = False
            self.rgb_calls = 0
            self.gatevit_calls = 0
            self.rgb_encoder = None
            self.thermal_image_feature_set = {THERMAL_COLD_FEATURE, THERMAL_HOT_FEATURE}
            self.thermal_encoder = None
            self.thermal_head_fusion = None
            self.rgb_thermal_align_aligner = None
            self.rgb_thermal_align_thermal_feature = None
            self.rgb_thermal_align_head_feature = RGB_FEATURE
            self.thermal_residual_fusion = None
            self.rgb_thermal_residual_head_feature = RGB_FEATURE
            self.rgb_thermal_residual_thermal_feature = None
            self.rgb_thermal_resvit_head_feature = RGB_FEATURE
            self.rgb_thermal_resvit_thermal_feature = None
            self.rgb_thermal_resvit_thermal_feature_set = set()
            self.rgb_thermal_gatevit_source_feature = THERMAL_FEATURE
            self.rgb_thermal_gatevit_cold_feature = THERMAL_COLD_FEATURE
            self.rgb_thermal_gatevit_hot_feature = THERMAL_HOT_FEATURE
            self.rgb_thermal_gatevit_thermal_feature_set = {
                THERMAL_COLD_FEATURE,
                THERMAL_HOT_FEATURE,
            }
            self.twogrey_gatevit_fusion = object()
            self.rgb_thermal_gateresvit_head_feature = RGB_FEATURE
            self.rgb_thermal_gateresvit_thermal_feature_set = set()
            self.twogrey_gateresvit_fusion = None
            self.rgb_thermal_gateresvit3_head_feature = RGB_FEATURE
            self.rgb_thermal_gateresvit3_thermal_feature_set = set()
            self.twogrey_gateresvit3_fusion = None
            self.resvit_attention_fusion = None
            self.rgb_thermal_resvit_attention_head_feature = RGB_FEATURE
            self.rgb_thermal_resvit_attention_thermal_feature = None
            self.paligemma_with_expert = DummyLanguageModel()
            self._last_prefix_token_layout = []

        def _embed_rgb_image(self, img):
            self.rgb_calls += 1
            return torch.ones(img.shape[0], 4, 8)

        def _embed_twogrey_gatevit_image(self, **kwargs):
            self.gatevit_calls += 1
            assert kwargs["text_tokens"].shape == (2, 3, 8)
            cold_img = kwargs["cold_img"]
            cold_img_mask = kwargs["cold_img_mask"]
            self._last_twogrey_gatevit_gate_summary = {
                "cold_alpha": {"mean": 0.8},
                "hot_beta": {"mean": 0.2},
                "cold_weight": {"mean": 0.8},
                "hot_weight": {"mean": 0.2},
            }
            return torch.zeros(cold_img.shape[0], 4, 8), cold_img_mask[:, None].expand(
                cold_img.shape[0], 4
            )

    model = DummyModel()
    PI05Pytorch.embed_prefix(
        model,
        images=[
            torch.zeros(2, 3, 8, 8),
            torch.zeros(2, 3, 8, 8),
            torch.zeros(2, 3, 8, 8),
        ],
        img_masks=[torch.ones(2, dtype=torch.bool) for _ in range(3)],
        tokens=torch.zeros(2, 3, dtype=torch.long),
        masks=torch.ones(2, 3, dtype=torch.bool),
    )

    assert model.rgb_calls == 1
    assert model.gatevit_calls == 1
    image_layout = [
        item for item in model._last_prefix_token_layout if item["kind"].startswith("image")
    ]
    assert [item["kind"] for item in image_layout] == ["image", "image_gatevit_twogrey"]
    assert image_layout[0]["name"] == RGB_FEATURE
    assert image_layout[1]["name"] == THERMAL_FEATURE
    assert image_layout[1]["thermal_features"] == [
        THERMAL_COLD_FEATURE,
        THERMAL_HOT_FEATURE,
    ]
    assert image_layout[1]["gate"]["cold_weight"]["mean"] == 0.8


def test_embed_prefix_routes_gateactionvit_as_two_independent_thermal_slots():
    class DummyLanguageModel:
        def embed_language_tokens(self, tokens):
            return torch.zeros(tokens.shape[0], tokens.shape[1], 8)

    class DummyModel:
        _apply_checkpoint = PI05Pytorch._apply_checkpoint

        def __init__(self):
            self.config = SimpleNamespace(
                image_features={
                    RGB_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_COLD_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_HOT_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                },
                uses_thermal_resnet18_encoder=False,
                uses_anythermal_encoder=False,
                uses_thermal_cvae_encoder=False,
                uses_dedicated_thermal_vit=False,
                uses_shared_projector_thermal_vit=False,
                uses_residual_thermal_vit=False,
                uses_gate_thermal_vit=False,
                uses_gate_action_thermal_vit=True,
                uses_gate_thermal_vit_mix=False,
                uses_gate_residual_thermal_vit=False,
                uses_gate_residual_thermal_vit3=False,
                uses_resvit_attention_encoder=False,
                uses_twogrey_thermal_input=True,
                thermal_cvae_source_feature=RGB_FEATURE,
            )
            self.gradient_checkpointing_enabled = False
            self.training = False
            self.rgb_calls = 0
            self.gateactionvit_calls = 0
            self.rgb_encoder = None
            self.thermal_image_feature_set = {THERMAL_COLD_FEATURE, THERMAL_HOT_FEATURE}
            self.thermal_encoder = None
            self.thermal_head_fusion = None
            self.rgb_thermal_align_aligner = None
            self.rgb_thermal_align_thermal_feature = None
            self.rgb_thermal_align_head_feature = RGB_FEATURE
            self.thermal_residual_fusion = None
            self.rgb_thermal_residual_head_feature = RGB_FEATURE
            self.rgb_thermal_residual_thermal_feature = None
            self.rgb_thermal_resvit_head_feature = RGB_FEATURE
            self.rgb_thermal_resvit_thermal_feature = None
            self.rgb_thermal_resvit_thermal_feature_set = set()
            self.twogrey_gatevit_fusion = None
            self.twogrey_gateactionvit_fusion = object()
            self.rgb_thermal_gateactionvit_source_feature = THERMAL_FEATURE
            self.rgb_thermal_gateactionvit_cold_feature = THERMAL_COLD_FEATURE
            self.rgb_thermal_gateactionvit_hot_feature = THERMAL_HOT_FEATURE
            self.twogrey_gatevitmix_fusion = None
            self.twogrey_gateresvit_fusion = None
            self.twogrey_gateresvit3_fusion = None
            self.resvit_attention_fusion = None
            self.rgb_thermal_resvit_attention_thermal_feature = None
            self.paligemma_with_expert = DummyLanguageModel()
            self._last_prefix_token_layout = []

        def _embed_rgb_image(self, img):
            self.rgb_calls += 1
            return torch.ones(img.shape[0], 4, 8)

        def _embed_twogrey_gateactionvit_images(self, **kwargs):
            self.gateactionvit_calls += 1
            batch_size = kwargs["cold_img"].shape[0]
            token_mask = torch.ones(batch_size, 4, dtype=torch.bool)
            self._current_gateactionvit_attention_bias = torch.tensor(
                [[0.3, -0.3]] * batch_size
            )
            self._last_twogrey_gateactionvit_gate_summary = {
                "cold_alpha": {"mean": 0.75},
                "hot_beta": {"mean": 0.25},
                "cold_attention_bias": {"mean": 0.3},
                "hot_attention_bias": {"mean": -0.3},
            }
            return (
                torch.full((batch_size, 4, 8), 2.0),
                token_mask,
                torch.full((batch_size, 4, 8), 3.0),
                token_mask,
            )

    model = DummyModel()
    prefix_embs, _, _ = PI05Pytorch.embed_prefix(
        model,
        images=[
            torch.zeros(2, 3, 8, 8),
            torch.zeros(2, 3, 8, 8),
            torch.zeros(2, 3, 8, 8),
        ],
        img_masks=[torch.ones(2, dtype=torch.bool) for _ in range(3)],
        tokens=torch.zeros(2, 3, dtype=torch.long),
        masks=torch.ones(2, 3, dtype=torch.bool),
    )

    assert model.rgb_calls == 1
    assert model.gateactionvit_calls == 1
    image_layout = [
        item for item in model._last_prefix_token_layout if item["kind"].startswith("image")
    ]
    assert [item["kind"] for item in image_layout] == [
        "image",
        "image_gateactionvit_twogrey_cold",
        "image_gateactionvit_twogrey_hot",
    ]
    torch.testing.assert_close(prefix_embs[:, :4], torch.ones(2, 4, 8))
    torch.testing.assert_close(prefix_embs[:, 4:8], torch.full((2, 4, 8), 2.0))
    torch.testing.assert_close(prefix_embs[:, 8:12], torch.full((2, 4, 8), 3.0))
    assert image_layout[1]["gate"]["cold_alpha"]["mean"] == 0.75
    assert "gate" not in image_layout[2]


def test_embed_prefix_routes_doublegatevit_as_one_merged_thermal_slot():
    class DummyLanguageModel:
        def embed_language_tokens(self, tokens):
            return torch.zeros(tokens.shape[0], tokens.shape[1], 8)

    class DummyModel:
        _apply_checkpoint = PI05Pytorch._apply_checkpoint

        def __init__(self):
            self.config = SimpleNamespace(
                image_features={
                    RGB_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_COLD_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_HOT_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    WRIST_LEFT_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                },
                uses_thermal_resnet18_encoder=False,
                uses_anythermal_encoder=False,
                uses_thermal_cvae_encoder=False,
                uses_dedicated_thermal_vit=False,
                uses_shared_projector_thermal_vit=False,
                uses_residual_thermal_vit=False,
                uses_gate_thermal_vit=False,
                uses_gate_action_thermal_vit=False,
                uses_double_gate_thermal_vit=True,
                uses_gate_thermal_vit_mix=False,
                uses_gate_residual_thermal_vit=False,
                uses_gate_residual_thermal_vit3=False,
                uses_resvit_attention_encoder=False,
                uses_twogrey_thermal_input=True,
                thermal_cvae_source_feature=RGB_FEATURE,
            )
            self.gradient_checkpointing_enabled = False
            self.training = False
            self.rgb_calls = 0
            self.doublegatevit_calls = 0
            self.rgb_encoder = None
            self.thermal_image_feature_set = {THERMAL_COLD_FEATURE, THERMAL_HOT_FEATURE}
            self.thermal_encoder = None
            self.thermal_head_fusion = None
            self.rgb_thermal_align_aligner = None
            self.rgb_thermal_align_thermal_feature = None
            self.thermal_residual_fusion = None
            self.rgb_thermal_residual_thermal_feature = None
            self.rgb_thermal_resvit_thermal_feature_set = set()
            self.twogrey_gatevit_fusion = None
            self.twogrey_gateactionvit_fusion = None
            self.twogrey_doublegatevit_fusion = object()
            self.rgb_thermal_doublegatevit_head_feature = RGB_FEATURE
            self.rgb_thermal_doublegatevit_source_feature = THERMAL_FEATURE
            self.rgb_thermal_doublegatevit_cold_feature = THERMAL_COLD_FEATURE
            self.rgb_thermal_doublegatevit_hot_feature = THERMAL_HOT_FEATURE
            self.twogrey_gatevitmix_fusion = None
            self.twogrey_gateresvit_fusion = None
            self.twogrey_gateresvit3_fusion = None
            self.resvit_attention_fusion = None
            self.rgb_thermal_resvit_attention_thermal_feature = None
            self.paligemma_with_expert = DummyLanguageModel()
            self._last_prefix_token_layout = []

        def _embed_rgb_image(self, img):
            self.rgb_calls += 1
            return torch.ones(img.shape[0], 4, 8)

        def _embed_twogrey_doublegatevit_head_and_thermal(self, **kwargs):
            self.doublegatevit_calls += 1
            batch_size = kwargs["rgb_img"].shape[0]
            token_mask = torch.ones(batch_size, 4, dtype=torch.bool)
            self._last_twogrey_doublegatevit_gate_summary = {
                "cold_alpha": {"mean": 0.7},
                "hot_beta": {"mean": 0.3},
                "text_cold_weight": {"mean": 0.8},
                "rgb_cold_alignment_weight": {"mean": 0.6},
            }
            return (
                torch.full((batch_size, 4, 8), 7.0),
                token_mask,
                torch.full((batch_size, 4, 8), 9.0),
                token_mask,
            )

    model = DummyModel()
    prefix_embs, _, _ = PI05Pytorch.embed_prefix(
        model,
        images=[torch.zeros(2, 3, 8, 8) for _ in range(4)],
        img_masks=[torch.ones(2, dtype=torch.bool) for _ in range(4)],
        tokens=torch.zeros(2, 3, dtype=torch.long),
        masks=torch.ones(2, 3, dtype=torch.bool),
    )

    assert model.doublegatevit_calls == 1
    assert model.rgb_calls == 1
    image_layout = [
        item for item in model._last_prefix_token_layout if item["kind"].startswith("image")
    ]
    assert [item["kind"] for item in image_layout] == [
        "image",
        "image_doublegatevit_twogrey",
        "image",
    ]
    assert [item["name"] for item in image_layout] == [
        RGB_FEATURE,
        THERMAL_FEATURE,
        WRIST_LEFT_FEATURE,
    ]
    torch.testing.assert_close(prefix_embs[:, :4], torch.full((2, 4, 8), 7.0))
    torch.testing.assert_close(prefix_embs[:, 4:8], torch.full((2, 4, 8), 9.0))
    torch.testing.assert_close(prefix_embs[:, 8:12], torch.ones(2, 4, 8))
    assert image_layout[1]["gate"]["cold_alpha"]["mean"] == 0.7


def test_embed_prefix_routes_patchgatevit_as_one_independent_thermal_slot():
    class DummyLanguageModel:
        def embed_language_tokens(self, tokens):
            return torch.zeros(tokens.shape[0], tokens.shape[1], 8)

    class DummyModel:
        _apply_checkpoint = PI05Pytorch._apply_checkpoint

        def __init__(self):
            self.config = SimpleNamespace(
                image_features={
                    RGB_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_COLD_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_HOT_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    WRIST_LEFT_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                },
                uses_thermal_resnet18_encoder=False,
                uses_anythermal_encoder=False,
                uses_thermal_cvae_encoder=False,
                uses_dedicated_thermal_vit=False,
                uses_shared_projector_thermal_vit=False,
                uses_residual_thermal_vit=False,
                uses_gate_thermal_vit=False,
                uses_gate_action_thermal_vit=False,
                uses_double_gate_thermal_vit=False,
                uses_patch_gate_thermal_vit=True,
                uses_gate_thermal_vit_mix=False,
                uses_gate_residual_thermal_vit=False,
                uses_gate_residual_thermal_vit3=False,
                uses_resvit_attention_encoder=False,
                uses_twogrey_thermal_input=True,
                thermal_cvae_source_feature=RGB_FEATURE,
            )
            self.gradient_checkpointing_enabled = False
            self.training = False
            self.rgb_calls = 0
            self.patchgatevit_calls = 0
            self.rgb_encoder = None
            self.thermal_image_feature_set = {THERMAL_COLD_FEATURE, THERMAL_HOT_FEATURE}
            self.thermal_encoder = None
            self.thermal_head_fusion = None
            self.rgb_thermal_align_aligner = None
            self.rgb_thermal_align_thermal_feature = None
            self.thermal_residual_fusion = None
            self.rgb_thermal_residual_thermal_feature = None
            self.rgb_thermal_resvit_thermal_feature_set = set()
            self.twogrey_gatevit_fusion = None
            self.twogrey_gateactionvit_fusion = None
            self.twogrey_doublegatevit_fusion = None
            self.twogrey_patchgatevit_fusion = object()
            self.rgb_thermal_patchgatevit_head_feature = RGB_FEATURE
            self.rgb_thermal_patchgatevit_source_feature = THERMAL_FEATURE
            self.rgb_thermal_patchgatevit_cold_feature = THERMAL_COLD_FEATURE
            self.rgb_thermal_patchgatevit_hot_feature = THERMAL_HOT_FEATURE
            self.twogrey_gatevitmix_fusion = None
            self.twogrey_gateresvit_fusion = None
            self.twogrey_gateresvit3_fusion = None
            self.resvit_attention_fusion = None
            self.rgb_thermal_resvit_attention_thermal_feature = None
            self.paligemma_with_expert = DummyLanguageModel()
            self._last_prefix_token_layout = []

        def _embed_rgb_image(self, img):
            self.rgb_calls += 1
            return torch.ones(img.shape[0], 4, 8)

        def _embed_twogrey_patchgatevit_head_and_thermal(self, **kwargs):
            self.patchgatevit_calls += 1
            batch_size = kwargs["rgb_img"].shape[0]
            token_mask = torch.ones(batch_size, 4, dtype=torch.bool)
            self._last_twogrey_patchgatevit_gate_summary = (
                {
                    "cold_weight": {"mean": 0.7},
                    "hot_weight": {"mean": 0.3},
                    "patch_relevance": {"mean": 1.0},
                }
                if kwargs.get("capture_gate_summary", False)
                else None
            )
            return (
                torch.full((batch_size, 4, 8), 7.0),
                token_mask,
                torch.full((batch_size, 4, 8), 9.0),
                token_mask,
            )

    model = DummyModel()
    prefix_embs, _, _ = PI05Pytorch.embed_prefix(
        model,
        images=[torch.zeros(2, 3, 8, 8) for _ in range(4)],
        img_masks=[torch.ones(2, dtype=torch.bool) for _ in range(4)],
        tokens=torch.zeros(2, 3, dtype=torch.long),
        masks=torch.ones(2, 3, dtype=torch.bool),
        capture_patchgatevit_summary=True,
    )

    assert model.patchgatevit_calls == 1
    assert model.rgb_calls == 1
    image_layout = [
        item for item in model._last_prefix_token_layout if item["kind"].startswith("image")
    ]
    assert [item["kind"] for item in image_layout] == [
        "image",
        "image_patchgatevit_twogrey",
        "image",
    ]
    assert [item["name"] for item in image_layout] == [
        RGB_FEATURE,
        THERMAL_FEATURE,
        WRIST_LEFT_FEATURE,
    ]
    torch.testing.assert_close(prefix_embs[:, :4], torch.full((2, 4, 8), 7.0))
    torch.testing.assert_close(prefix_embs[:, 4:8], torch.full((2, 4, 8), 9.0))
    torch.testing.assert_close(prefix_embs[:, 8:12], torch.ones(2, 4, 8))
    assert image_layout[1]["gate"]["patch_relevance"]["mean"] == 1.0

    PI05Pytorch.embed_prefix(
        model,
        images=[torch.zeros(2, 3, 8, 8) for _ in range(4)],
        img_masks=[torch.ones(2, dtype=torch.bool) for _ in range(4)],
        tokens=torch.zeros(2, 3, dtype=torch.long),
        masks=torch.ones(2, 3, dtype=torch.bool),
    )
    uncaptured_layout = [
        item for item in model._last_prefix_token_layout if item["kind"].startswith("image")
    ]
    assert uncaptured_layout[1]["gate"] is None


def test_embed_prefix_routes_gatevitmix_without_replacing_head_rgb():
    class DummyLanguageModel:
        def embed_language_tokens(self, tokens):
            return torch.zeros(tokens.shape[0], tokens.shape[1], 8)

    class DummyModel:
        _apply_checkpoint = PI05Pytorch._apply_checkpoint

        def __init__(self):
            self.config = SimpleNamespace(
                image_features={
                    RGB_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_COLD_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_HOT_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    WRIST_LEFT_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                },
                uses_thermal_resnet18_encoder=False,
                uses_anythermal_encoder=False,
                uses_thermal_cvae_encoder=False,
                uses_dedicated_thermal_vit=False,
                uses_shared_projector_thermal_vit=False,
                uses_residual_thermal_vit=False,
                uses_gate_thermal_vit=False,
                uses_gate_thermal_vit_mix=True,
                uses_gate_residual_thermal_vit=False,
                uses_gate_residual_thermal_vit3=False,
                uses_resvit_attention_encoder=False,
                uses_twogrey_thermal_input=True,
                thermal_cvae_source_feature=RGB_FEATURE,
            )
            self.gradient_checkpointing_enabled = False
            self.training = False
            self.rgb_calls = 0
            self.gatevitmix_calls = 0
            self.rgb_encoder = None
            self.thermal_image_feature_set = {THERMAL_COLD_FEATURE, THERMAL_HOT_FEATURE}
            self.thermal_encoder = None
            self.thermal_head_fusion = None
            self.rgb_thermal_align_aligner = None
            self.rgb_thermal_align_thermal_feature = None
            self.rgb_thermal_align_head_feature = RGB_FEATURE
            self.thermal_residual_fusion = None
            self.rgb_thermal_residual_head_feature = RGB_FEATURE
            self.rgb_thermal_residual_thermal_feature = None
            self.rgb_thermal_resvit_head_feature = RGB_FEATURE
            self.rgb_thermal_resvit_thermal_feature = None
            self.rgb_thermal_resvit_thermal_feature_set = set()
            self.twogrey_gatevit_fusion = None
            self.rgb_thermal_gatevit_hot_feature = None
            self.twogrey_gatevitmix_fusion = object()
            self.gatevitmix_fusion = object()
            self.rgb_thermal_gatevitmix_head_feature = RGB_FEATURE
            self.rgb_thermal_gatevitmix_source_feature = THERMAL_FEATURE
            self.rgb_thermal_gatevitmix_cold_feature = THERMAL_COLD_FEATURE
            self.rgb_thermal_gatevitmix_hot_feature = THERMAL_HOT_FEATURE
            self.rgb_thermal_gatevitmix_thermal_feature_set = {
                THERMAL_COLD_FEATURE,
                THERMAL_HOT_FEATURE,
            }
            self.rgb_thermal_gateresvit_head_feature = RGB_FEATURE
            self.rgb_thermal_gateresvit_thermal_feature_set = set()
            self.twogrey_gateresvit_fusion = None
            self.rgb_thermal_gateresvit3_head_feature = RGB_FEATURE
            self.rgb_thermal_gateresvit3_thermal_feature_set = set()
            self.twogrey_gateresvit3_fusion = None
            self.resvit_attention_fusion = None
            self.rgb_thermal_resvit_attention_head_feature = RGB_FEATURE
            self.rgb_thermal_resvit_attention_thermal_feature = None
            self.paligemma_with_expert = DummyLanguageModel()
            self._last_prefix_token_layout = []

        def _embed_rgb_image(self, img):
            self.rgb_calls += 1
            return torch.ones(img.shape[0], 4, 8)

        def _embed_twogrey_gatevitmix_head_and_thermal(self, **kwargs):
            self.gatevitmix_calls += 1
            batch_size = kwargs["rgb_img"].shape[0]
            token_mask = torch.ones(batch_size, 4, dtype=torch.bool)
            self._last_twogrey_gatevitmix_gate_summary = {
                "cold_alpha": {"mean": 0.75},
                "hot_beta": {"mean": 0.25},
                "cold_weight": {"mean": 0.75},
                "hot_weight": {"mean": 0.25},
                "head_mix_delta_rms": {"mean": 0.4},
            }
            return (
                torch.full((batch_size, 4, 8), 7.0),
                token_mask,
                torch.full((batch_size, 4, 8), 9.0),
                token_mask,
            )

    model = DummyModel()
    prefix_embs, _, _ = PI05Pytorch.embed_prefix(
        model,
        images=[
            torch.zeros(2, 3, 8, 8),
            torch.zeros(2, 3, 8, 8),
            torch.zeros(2, 3, 8, 8),
            torch.zeros(2, 3, 8, 8),
        ],
        img_masks=[torch.ones(2, dtype=torch.bool) for _ in range(4)],
        tokens=torch.zeros(2, 3, dtype=torch.long),
        masks=torch.ones(2, 3, dtype=torch.bool),
    )

    assert model.gatevitmix_calls == 1
    assert model.rgb_calls == 1
    image_layout = [
        item for item in model._last_prefix_token_layout if item["kind"].startswith("image")
    ]
    assert [item["kind"] for item in image_layout] == [
        "image",
        "image_gatevitmix_twogrey",
        "image",
    ]
    assert [item["name"] for item in image_layout] == [
        RGB_FEATURE,
        THERMAL_FEATURE,
        WRIST_LEFT_FEATURE,
    ]
    torch.testing.assert_close(prefix_embs[:, :4], torch.full((2, 4, 8), 7.0))
    torch.testing.assert_close(prefix_embs[:, 4:8], torch.full((2, 4, 8), 9.0))
    torch.testing.assert_close(prefix_embs[:, 8:12], torch.ones(2, 4, 8))
    assert image_layout[1]["gate"]["head_mix_delta_rms"]["mean"] == 0.4


def test_embed_prefix_routes_twogrey_gateresvit_into_single_rgb_slot():
    class DummyLanguageModel:
        def embed_language_tokens(self, tokens):
            return torch.zeros(tokens.shape[0], tokens.shape[1], 8)

    class DummyModel:
        _apply_checkpoint = PI05Pytorch._apply_checkpoint

        def __init__(self):
            self.config = SimpleNamespace(
                image_features={
                    RGB_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_COLD_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_HOT_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                },
                uses_thermal_resnet18_encoder=False,
                uses_anythermal_encoder=False,
                uses_thermal_cvae_encoder=False,
                uses_dedicated_thermal_vit=False,
                uses_shared_projector_thermal_vit=False,
                uses_residual_thermal_vit=False,
                uses_gate_residual_thermal_vit=True,
                uses_resvit_attention_encoder=False,
                uses_twogrey_thermal_input=True,
                thermal_cvae_source_feature=RGB_FEATURE,
            )
            self.gradient_checkpointing_enabled = False
            self.training = False
            self.gateresvit_calls = 0
            self.rgb_encoder = None
            self.thermal_image_feature_set = {THERMAL_COLD_FEATURE, THERMAL_HOT_FEATURE}
            self.thermal_encoder = None
            self.thermal_head_fusion = None
            self.rgb_thermal_align_aligner = None
            self.rgb_thermal_align_thermal_feature = None
            self.rgb_thermal_align_head_feature = RGB_FEATURE
            self.thermal_residual_fusion = None
            self.rgb_thermal_residual_head_feature = RGB_FEATURE
            self.rgb_thermal_residual_thermal_feature = None
            self.rgb_thermal_resvit_head_feature = RGB_FEATURE
            self.rgb_thermal_resvit_thermal_feature = None
            self.rgb_thermal_resvit_thermal_feature_set = set()
            self.rgb_thermal_gateresvit_head_feature = RGB_FEATURE
            self.rgb_thermal_gateresvit_cold_feature = THERMAL_COLD_FEATURE
            self.rgb_thermal_gateresvit_hot_feature = THERMAL_HOT_FEATURE
            self.rgb_thermal_gateresvit_thermal_feature_set = {
                THERMAL_COLD_FEATURE,
                THERMAL_HOT_FEATURE,
            }
            self.twogrey_gateresvit_fusion = object()
            self.resvit_attention_fusion = None
            self.rgb_thermal_resvit_attention_head_feature = RGB_FEATURE
            self.rgb_thermal_resvit_attention_thermal_feature = None
            self.thermal_vision_tower = object()
            self.thermal_multi_modal_projector = None
            self.paligemma_with_expert = DummyLanguageModel()
            self._last_prefix_token_layout = []
            self._last_twogrey_gateresvit_gate_summary = None

        def _embed_twogrey_gateresvit_head_image(self, **kwargs):
            self.gateresvit_calls += 1
            assert kwargs["text_tokens"].shape == (2, 3, 8)
            rgb_img = kwargs["rgb_img"]
            rgb_img_mask = kwargs["rgb_img_mask"]
            self._last_twogrey_gateresvit_gate_summary = {
                "cold_alpha": {"mean": 1.25},
                "hot_beta": {"mean": 0.75},
            }
            return torch.zeros(rgb_img.shape[0], 4, 8), rgb_img_mask[:, None].expand(
                rgb_img.shape[0], 4
            )

    model = DummyModel()
    PI05Pytorch.embed_prefix(
        model,
        images=[
            torch.zeros(2, 3, 8, 8),
            torch.zeros(2, 3, 8, 8),
            torch.zeros(2, 3, 8, 8),
        ],
        img_masks=[torch.ones(2, dtype=torch.bool) for _ in range(3)],
        tokens=torch.zeros(2, 3, dtype=torch.long),
        masks=torch.ones(2, 3, dtype=torch.bool),
    )

    assert model.gateresvit_calls == 1
    image_layout = [
        item for item in model._last_prefix_token_layout if item["kind"].startswith("image")
    ]
    assert [item["kind"] for item in image_layout] == ["image_gateresvit_twogrey_residual"]
    assert image_layout[0]["thermal_features"] == [THERMAL_COLD_FEATURE, THERMAL_HOT_FEATURE]
    assert image_layout[0]["gate"]["cold_alpha"]["mean"] == 1.25


def test_embed_prefix_routes_twogrey_gateresandvit_into_residual_and_original_rgb_slots():
    class DummyLanguageModel:
        def embed_language_tokens(self, tokens):
            return torch.zeros(tokens.shape[0], tokens.shape[1], 8)

    class DummyModel:
        _apply_checkpoint = PI05Pytorch._apply_checkpoint

        def __init__(self):
            self.config = SimpleNamespace(
                image_features={
                    RGB_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_COLD_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_HOT_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                },
                uses_thermal_resnet18_encoder=False,
                uses_anythermal_encoder=False,
                uses_thermal_cvae_encoder=False,
                uses_dedicated_thermal_vit=False,
                uses_shared_projector_thermal_vit=False,
                uses_residual_thermal_vit=False,
                uses_gate_residual_thermal_vit=False,
                uses_gate_res_and_thermal_vit=True,
                uses_resvit_attention_encoder=False,
                uses_twogrey_thermal_input=True,
                thermal_cvae_source_feature=RGB_FEATURE,
            )
            self.gradient_checkpointing_enabled = False
            self.training = False
            self.gateresandvit_calls = 0
            self.rgb_encoder = None
            self.thermal_image_feature_set = {THERMAL_COLD_FEATURE, THERMAL_HOT_FEATURE}
            self.thermal_encoder = None
            self.thermal_head_fusion = None
            self.rgb_thermal_align_aligner = None
            self.rgb_thermal_align_thermal_feature = None
            self.rgb_thermal_align_head_feature = RGB_FEATURE
            self.thermal_residual_fusion = None
            self.rgb_thermal_residual_head_feature = RGB_FEATURE
            self.rgb_thermal_residual_thermal_feature = None
            self.rgb_thermal_resvit_head_feature = RGB_FEATURE
            self.rgb_thermal_resvit_thermal_feature = None
            self.rgb_thermal_resvit_thermal_feature_set = set()
            self.rgb_thermal_gateresvit_head_feature = RGB_FEATURE
            self.rgb_thermal_gateresvit_cold_feature = THERMAL_COLD_FEATURE
            self.rgb_thermal_gateresvit_hot_feature = THERMAL_HOT_FEATURE
            self.rgb_thermal_gateresvit_thermal_feature_set = {
                THERMAL_COLD_FEATURE,
                THERMAL_HOT_FEATURE,
            }
            self.twogrey_gateresvit_fusion = object()
            self.resvit_attention_fusion = None
            self.rgb_thermal_resvit_attention_head_feature = RGB_FEATURE
            self.rgb_thermal_resvit_attention_thermal_feature = None
            self.thermal_vision_tower = object()
            self.thermal_multi_modal_projector = None
            self.paligemma_with_expert = DummyLanguageModel()
            self._last_prefix_token_layout = []
            self._last_twogrey_gateresandvit_gate_summary = None

        def _embed_twogrey_gateresvit_head_and_original_images(self, **kwargs):
            self.gateresandvit_calls += 1
            assert kwargs["text_tokens"].shape == (2, 3, 8)
            rgb_img = kwargs["rgb_img"]
            rgb_img_mask = kwargs["rgb_img_mask"]
            self._last_twogrey_gateresandvit_gate_summary = {
                "cold_alpha": {"mean": 1.25},
                "hot_beta": {"mean": 0.75},
            }
            token_mask = rgb_img_mask[:, None].expand(rgb_img.shape[0], 4)
            return torch.ones(rgb_img.shape[0], 4, 8), 2 * torch.ones(rgb_img.shape[0], 4, 8), token_mask

    model = DummyModel()
    embs, pad_masks, _ = PI05Pytorch.embed_prefix(
        model,
        images=[
            torch.zeros(2, 3, 8, 8),
            torch.zeros(2, 3, 8, 8),
            torch.zeros(2, 3, 8, 8),
        ],
        img_masks=[torch.ones(2, dtype=torch.bool) for _ in range(3)],
        tokens=torch.zeros(2, 3, dtype=torch.long),
        masks=torch.ones(2, 3, dtype=torch.bool),
    )

    assert model.gateresandvit_calls == 1
    image_layout = [
        item for item in model._last_prefix_token_layout if item["kind"].startswith("image")
    ]
    assert [item["kind"] for item in image_layout] == [
        "image_gateresandvit_twogrey_residual",
        "image_gateresandvit_original_rgb",
    ]
    assert [item["stream"] for item in image_layout] == ["thermal_residual", "original_rgb"]
    assert image_layout[0]["gate"]["cold_alpha"]["mean"] == 1.25
    assert image_layout[1]["gate"] is None
    assert pad_masks[:, :8].all()
    torch.testing.assert_close(embs[:, :4], torch.ones(2, 4, 8))
    torch.testing.assert_close(embs[:, 4:8], 2 * torch.ones(2, 4, 8))


def test_embed_prefix_routes_twogrey_gateresvit3_into_single_rgb_slot():
    class DummyLanguageModel:
        def embed_language_tokens(self, tokens):
            return torch.zeros(tokens.shape[0], tokens.shape[1], 8)

    class DummyModel:
        _apply_checkpoint = PI05Pytorch._apply_checkpoint

        def __init__(self):
            self.config = SimpleNamespace(
                image_features={
                    RGB_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_COLD_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_HOT_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                },
                uses_thermal_resnet18_encoder=False,
                uses_anythermal_encoder=False,
                uses_thermal_cvae_encoder=False,
                uses_dedicated_thermal_vit=False,
                uses_shared_projector_thermal_vit=False,
                uses_residual_thermal_vit=False,
                uses_gate_residual_thermal_vit=False,
                uses_gate_residual_thermal_vit3=True,
                uses_resvit_attention_encoder=False,
                uses_twogrey_thermal_input=True,
                thermal_cvae_source_feature=RGB_FEATURE,
            )
            self.gradient_checkpointing_enabled = False
            self.training = False
            self.gateresvit3_calls = 0
            self.rgb_encoder = None
            self.thermal_image_feature_set = {THERMAL_COLD_FEATURE, THERMAL_HOT_FEATURE}
            self.thermal_encoder = None
            self.thermal_head_fusion = None
            self.rgb_thermal_align_aligner = None
            self.rgb_thermal_align_thermal_feature = None
            self.rgb_thermal_align_head_feature = RGB_FEATURE
            self.thermal_residual_fusion = None
            self.rgb_thermal_residual_head_feature = RGB_FEATURE
            self.rgb_thermal_residual_thermal_feature = None
            self.rgb_thermal_resvit_head_feature = RGB_FEATURE
            self.rgb_thermal_resvit_thermal_feature = None
            self.rgb_thermal_resvit_thermal_feature_set = set()
            self.rgb_thermal_gateresvit_head_feature = RGB_FEATURE
            self.rgb_thermal_gateresvit_thermal_feature_set = set()
            self.twogrey_gateresvit_fusion = None
            self.rgb_thermal_gateresvit3_head_feature = RGB_FEATURE
            self.rgb_thermal_gateresvit3_cold_feature = THERMAL_COLD_FEATURE
            self.rgb_thermal_gateresvit3_hot_feature = THERMAL_HOT_FEATURE
            self.rgb_thermal_gateresvit3_thermal_feature_set = {
                THERMAL_COLD_FEATURE,
                THERMAL_HOT_FEATURE,
            }
            self.twogrey_gateresvit3_fusion = object()
            self.resvit_attention_fusion = None
            self.rgb_thermal_resvit_attention_head_feature = RGB_FEATURE
            self.rgb_thermal_resvit_attention_thermal_feature = None
            self.thermal_vision_tower = object()
            self.thermal_multi_modal_projector = None
            self.paligemma_with_expert = DummyLanguageModel()
            self._last_prefix_token_layout = []
            self._last_twogrey_gateresvit_gate_summary = None
            self._last_twogrey_gateresvit3_gate_summary = None

        def _embed_twogrey_gateresvit3_head_image(self, **kwargs):
            self.gateresvit3_calls += 1
            assert kwargs["text_tokens"].shape == (2, 3, 8)
            rgb_img = kwargs["rgb_img"]
            rgb_img_mask = kwargs["rgb_img_mask"]
            self._last_twogrey_gateresvit3_gate_summary = {
                "cold_alpha": {"mean": 0.35},
                "hot_beta": {"mean": 0.05},
                "thermal_confidence": {"mean": 0.4},
                "thermal_evidence": {"mean": 0.2},
            }
            return torch.zeros(rgb_img.shape[0], 4, 8), rgb_img_mask[:, None].expand(
                rgb_img.shape[0], 4
            )

    model = DummyModel()
    PI05Pytorch.embed_prefix(
        model,
        images=[
            torch.zeros(2, 3, 8, 8),
            torch.zeros(2, 3, 8, 8),
            torch.zeros(2, 3, 8, 8),
        ],
        img_masks=[torch.ones(2, dtype=torch.bool) for _ in range(3)],
        tokens=torch.zeros(2, 3, dtype=torch.long),
        masks=torch.ones(2, 3, dtype=torch.bool),
    )

    assert model.gateresvit3_calls == 1
    image_layout = [
        item for item in model._last_prefix_token_layout if item["kind"].startswith("image")
    ]
    assert [item["kind"] for item in image_layout] == ["image_gateresvit3_twogrey_residual"]
    assert image_layout[0]["thermal_features"] == [THERMAL_COLD_FEATURE, THERMAL_HOT_FEATURE]
    assert image_layout[0]["gate"]["thermal_confidence"]["mean"] == 0.4


def test_resvit_attention_fusion_zero_initialized_residual_preserves_rgb_tokens():
    fusion = ResViTAttentionFusion(
        embed_dim=8,
        num_heads=2,
        merge_hidden_dim=12,
        residual_hidden_dim=10,
    )
    rgb_tokens = torch.randn(2, 4, 8)
    fused = fusion(
        rgb_tokens=rgb_tokens,
        thermal_tokens=torch.randn(2, 4, 8),
        text_tokens=torch.randn(2, 3, 8),
        rgb_token_mask=torch.ones(2, 4, dtype=torch.bool),
        thermal_token_mask=torch.tensor([[True] * 4, [False] * 4]),
        text_token_mask=torch.tensor([[True, True, False], [True, False, False]]),
    )

    assert fused.shape == rgb_tokens.shape
    torch.testing.assert_close(fused, rgb_tokens)


def test_resvit_attention_projects_then_fuses_with_text_tokens():
    class DummyFusion:
        def __init__(self):
            self.calls = 0
            self.kwargs = None

        def __call__(self, **kwargs):
            self.calls += 1
            self.kwargs = kwargs
            return kwargs["rgb_tokens"] + kwargs["thermal_tokens"]

    class DummyPaliGemma:
        def __init__(self):
            self.rgb_vision_tower = object()
            self.thermal_vision_tower = object()
            self.rgb_projector = object()
            self.paligemma = SimpleNamespace(
                model=SimpleNamespace(
                    vision_tower=self.rgb_vision_tower,
                    multi_modal_projector=self.rgb_projector,
                )
            )

        def embed_image(self, image):
            return torch.full((image.shape[0], 4, 8), 2.0)

        def embed_image_tokens_with_modules(self, image, vision_tower):
            assert vision_tower is self.thermal_vision_tower
            return torch.full((image.shape[0], 4, 6), 3.0)

        def project_image_tokens_with_modules(self, image_tokens, projector, out_dtype):
            assert projector is self.rgb_projector
            return torch.full((image_tokens.shape[0], image_tokens.shape[1], 8), 5.0).to(
                dtype=out_dtype
            )

    class DummyModel:
        _prepare_thermal_vit_image = PI05Pytorch._prepare_thermal_vit_image
        _project_siglip_tokens_with_rgb_projector = PI05Pytorch._project_siglip_tokens_with_rgb_projector
        _embed_resvit_attention_head_image = PI05Pytorch._embed_resvit_attention_head_image

        def __init__(self):
            self.paligemma_with_expert = DummyPaliGemma()
            self.thermal_vision_tower = self.paligemma_with_expert.thermal_vision_tower
            self.resvit_attention_fusion = DummyFusion()

        def _embed_rgb_image(self, image):
            return self.paligemma_with_expert.embed_image(image)

    model = DummyModel()
    text_tokens = torch.randn(2, 3, 8)
    fused, token_mask = model._embed_resvit_attention_head_image(
        rgb_img=torch.zeros(2, 3, 8, 8),
        thermal_img=torch.zeros(2, 1, 8, 8),
        rgb_img_mask=torch.tensor([True, True]),
        thermal_img_mask=torch.tensor([True, False]),
        text_tokens=text_tokens,
        text_token_mask=torch.tensor([[True, True, False], [True, False, False]]),
    )

    assert model.resvit_attention_fusion.calls == 1
    kwargs = model.resvit_attention_fusion.kwargs
    torch.testing.assert_close(kwargs["rgb_tokens"], torch.full((2, 4, 8), 2.0))
    torch.testing.assert_close(kwargs["thermal_tokens"], torch.full((2, 4, 8), 5.0))
    torch.testing.assert_close(kwargs["text_tokens"], text_tokens)
    assert kwargs["thermal_token_mask"].tolist() == [[True] * 4, [False] * 4]
    torch.testing.assert_close(fused, torch.full((2, 4, 8), 7.0))
    assert token_mask.tolist() == [[True] * 4, [True] * 4]


def test_thermal_vit_image_preparation_uses_three_channels():
    class DummyModel:
        _prepare_thermal_vit_image = PI05Pytorch._prepare_thermal_vit_image

    model = DummyModel()
    one_channel = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 1, 3, 4)
    three_channel = model._prepare_thermal_vit_image(one_channel)

    assert three_channel.shape == (2, 3, 3, 4)
    torch.testing.assert_close(three_channel[:, 0], one_channel[:, 0])
    torch.testing.assert_close(three_channel[:, 1], one_channel[:, 0])
    torch.testing.assert_close(three_channel[:, 2], one_channel[:, 0])


@pytest.mark.parametrize("thermal_encoder_channel", ["'anythermal'", "'resthermal'"])
def test_thermal_encoder_channel_accepts_quoted_anythermal_modes(thermal_encoder_channel):
    config = PI05Config(thermal_encoder_channel=thermal_encoder_channel)
    normalized_channel = thermal_encoder_channel.strip("'")

    assert config.thermal_encoder_channel == normalized_channel
    assert config.uses_anythermal_encoder == (normalized_channel == "anythermal")
    assert config.uses_resthermal_encoder == (normalized_channel == "resthermal")


def test_anythermal_preprocess_matches_dinov2_thermal_normalization():
    encoder = AnyThermalEncoder.__new__(AnyThermalEncoder)
    nn.Module.__init__(encoder)
    encoder.input_range = "minus_one_to_one"
    encoder.patch_size = 14

    image = torch.linspace(-1.0, 1.0, 14 * 14).view(1, 1, 14, 14)
    processed = encoder._preprocess_for_dinov2(image)

    zero_to_one = (image + 1.0) * 0.5
    mean = torch.tensor(ANYTHERMAL_DINOV2_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(ANYTHERMAL_DINOV2_STD).view(1, 3, 1, 1)
    expected = (zero_to_one.expand(-1, 3, -1, -1) - mean) / std
    torch.testing.assert_close(processed, expected)


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


def test_anythermal_dinov2_repo_paths_prefer_explicit_path(tmp_path):
    explicit_path = tmp_path / "dinov2-explicit"

    paths = AnyThermalEncoder._candidate_dinov2_repo_paths(str(explicit_path))

    assert paths[0] == explicit_path
    assert any(str(path).endswith("unitree_lerobot/dinov2") for path in paths)
    assert any("pretrained_checkpoints" in str(path) for path in paths)


def test_resthermal_zero_initialized_residual_preserves_rgb_tokens():
    class DummyResthermalFusion(AnyThermalResidualFusion):
        def __init__(self):
            nn.Module.__init__(self)
            self.backbone = nn.Identity()
            self.freeze_backbone = True
            self.thermal_norm = nn.LayerNorm(4)
            self.thermal_adapter = nn.Linear(4, 2)
            self.rgb_norm = nn.LayerNorm(6)
            self.rgb_adapter = nn.Linear(6, 2)
            self.cross_attn = nn.MultiheadAttention(2, 1, batch_first=True)
            self.context_norm = nn.LayerNorm(2)
            self.residual_decoder = nn.Sequential(
                nn.Linear(2, 3),
                nn.GELU(),
                nn.Linear(3, 6),
            )
            nn.init.zeros_(self.residual_decoder[-1].weight)
            nn.init.zeros_(self.residual_decoder[-1].bias)

        def extract_thermal_patch_tokens(self, images):
            batch_size = images.shape[0]
            return torch.ones(batch_size, 4, 4)

    fusion = DummyResthermalFusion()
    rgb_tokens = torch.randn(2, 4, 6)
    fused = fusion(
        rgb_tokens=rgb_tokens,
        thermal_images=torch.zeros(2, 1, 4, 4),
        rgb_token_mask=torch.ones(2, 4, dtype=torch.bool),
        thermal_image_mask=torch.tensor([True, False]),
    )

    torch.testing.assert_close(fused, rgb_tokens)


def test_rgb_thermal_token_aligner_and_decoder_shapes():
    aligner = RGBThermalTokenAligner(token_dim=32, num_heads=4)
    fused_tokens, fused_mask = aligner(
        rgb_tokens=torch.rand(2, 16, 32),
        thermal_tokens=torch.rand(2, 16, 32),
        rgb_token_mask=torch.ones(2, 16, dtype=torch.bool),
        thermal_token_mask=torch.tensor([[True] * 16, [False] * 16]),
    )

    assert fused_tokens.shape == (2, 16, 32)
    assert fused_mask.shape == (2, 16)
    assert fused_mask.all()

    decoder = RGBThermalFusionDecoder(token_dim=32, output_size=(8, 8), hidden_dim=16)
    image = decoder(fused_tokens)
    assert image.shape == (2, 3, 8, 8)


def test_rgb_thermal_align_fusion_requires_vit_channel():
    config = PI05Config(thermal_encoder_channel="'vit'", rgb_thermal_align_fusion=True)
    assert config.rgb_thermal_align_fusion
    assert config.uses_dedicated_thermal_vit
    assert config.rgb_thermal_align_gt_feature == RGB_THERMAL_GT_FEATURE

    with pytest.raises(ValueError, match="thermal_encoder_channel='vit'"):
        PI05Config(thermal_encoder_channel="'resnet18'", rgb_thermal_align_fusion=True)

    combined_config = PI05Config(
        thermal_encoder_channel="'vit'",
        rgb_thermal_align_fusion=True,
        mix_rgb_thermal=True,
    )
    assert combined_config.rgb_thermal_align_fusion
    assert combined_config.mix_rgb_thermal


def test_rgb_thermal_align_fusion_validates_gt_feature_name():
    policy = SimpleNamespace(
        mix_rgb_thermal=False,
        rgb_thermal_align_fusion=True,
        thermal_fusion_head_rgb_feature=RGB_FEATURE,
        rgb_thermal_mix_rgb_feature=RGB_FEATURE,
        rgb_thermal_mix_thermal_feature=THERMAL_FEATURE,
        rgb_thermal_align_gt_feature=RGB_THERMAL_GT_FEATURE,
    )
    cfg = SimpleNamespace(
        trainable_config=policy,
        dataset=SimpleNamespace(
            feature_names=["action", RGB_FEATURE, THERMAL_FEATURE, RGB_THERMAL_GT_FEATURE]
        ),
    )
    validate_rgb_thermal_mix_feature_names(cfg)

    cfg.dataset.feature_names = ["action", RGB_FEATURE, THERMAL_FEATURE]
    with pytest.raises(ValueError, match=RGB_THERMAL_GT_FEATURE):
        validate_rgb_thermal_mix_feature_names(cfg)


def test_rgb_thermal_align_fusion_with_precomputed_mix_requires_mix_and_gt():
    policy = SimpleNamespace(
        mix_rgb_thermal=True,
        rgb_thermal_align_fusion=True,
        rgb_thermal_mix_source="precomputed",
        rgb_thermal_mix_use_precomputed=False,
        thermal_fusion_head_rgb_feature=RGB_FEATURE,
        rgb_thermal_mix_rgb_feature=RGB_FEATURE,
        rgb_thermal_mix_thermal_feature=THERMAL_FEATURE,
        rgb_thermal_mix_precomputed_feature=RGB_THERMAL_MIX_FEATURE,
        rgb_thermal_align_gt_feature=RGB_THERMAL_GT_FEATURE,
    )
    cfg = SimpleNamespace(
        trainable_config=policy,
        dataset=SimpleNamespace(
            feature_names=[
                "action",
                RGB_FEATURE,
                THERMAL_FEATURE,
                RGB_THERMAL_MIX_FEATURE,
                RGB_THERMAL_GT_FEATURE,
            ]
        ),
    )
    validate_rgb_thermal_mix_feature_names(cfg)

    cfg.dataset.feature_names = ["action", RGB_FEATURE, THERMAL_FEATURE, RGB_THERMAL_GT_FEATURE]
    with pytest.raises(ValueError, match=RGB_THERMAL_MIX_FEATURE):
        validate_rgb_thermal_mix_feature_names(cfg)


def test_pi05_cvae_optimizer_group_uses_smaller_lr_and_respects_freeze():
    class DummyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                uses_thermal_cvae_encoder=True,
                optimizer_lr=2e-5,
            )
            self.model = nn.Module()
            self.model.thermal_encoder = nn.Linear(2, 2)
            self.model.action_head = nn.Linear(2, 2)

    policy = DummyPolicy()
    groups = PI05Policy.get_optim_params(policy)

    assert len(groups) == 2
    assert groups[1]["lr"] == pytest.approx(2e-6)
    assert set(groups[1]["params"]) == set(policy.model.thermal_encoder.parameters())

    for param in policy.model.thermal_encoder.parameters():
        param.requires_grad = False
    groups = PI05Policy.get_optim_params(policy)

    assert len(groups) == 1
    assert "lr" not in groups[0]
    assert set(groups[0]["params"]) == set(policy.model.action_head.parameters())


def test_preprocess_images_preserves_missing_camera_slot_order(monkeypatch):
    class DummyPolicy(nn.Module):
        _preprocess_image_tensor = PI05Policy._preprocess_image_tensor

        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(
                mix_rgb_thermal=False,
                image_resolution=(8, 8),
                image_features={
                    RGB_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                },
            )
            self._logged_missing_image_features = False
            self._logged_image_input_summary = False
            self._logged_rgb_thermal_mix = False

    monkeypatch.setattr(
        "lerobot.policies.pi05.modeling_pi05.save_pi05_image_input_snapshots",
        lambda *args, **kwargs: None,
    )
    policy = DummyPolicy()
    images, masks = PI05Policy._preprocess_images(
        policy,
        {THERMAL_FEATURE: torch.full((1, 3, 8, 8), 0.5)},
    )

    assert len(images) == len(masks) == 2
    assert not masks[0].any()
    assert masks[1].all()
    torch.testing.assert_close(images[0], torch.full_like(images[0], -1.0))
    torch.testing.assert_close(images[1], torch.zeros_like(images[1]))


def test_preprocess_images_computes_gateresvit3_stats_before_resize_padding(monkeypatch):
    class DummyPolicy(nn.Module):
        _preprocess_image_tensor = PI05Policy._preprocess_image_tensor

        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(
                thermal_input_type="twogrey",
                thermal_grey_background_roi=(0, 0, 1, 2),
                thermal_grey_background_value=80.0,
                uses_gate_residual_thermal_vit3=True,
                mix_rgb_thermal=False,
                image_resolution=(8, 8),
                legacy_resize_padding=False,
                image_features={
                    RGB_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_COLD_FEATURE: SimpleNamespace(shape=(3, 4, 8)),
                    THERMAL_HOT_FEATURE: SimpleNamespace(shape=(3, 4, 8)),
                },
                thermal_twogrey_feature_map=lambda: {
                    THERMAL_FEATURE: (THERMAL_COLD_FEATURE, THERMAL_HOT_FEATURE)
                },
            )
            self._logged_missing_image_features = False
            self._logged_image_input_summary = False
            self._logged_rgb_thermal_mix = False
            self._logged_thermal_twogrey_input = False

    monkeypatch.setattr(
        "lerobot.policies.pi05.modeling_pi05.save_pi05_image_input_snapshots",
        lambda *args, **kwargs: None,
    )
    policy = DummyPolicy()
    thermal = torch.full((1, 1, 4, 8), 80.0 / 255.0)
    thermal[:, :, 1:3, 3:5] = 130.0 / 255.0

    PI05Policy._preprocess_images(policy, {THERMAL_FEATURE: thermal})

    stats = policy._last_twogrey_gateresvit3_stats
    assert stats.shape == (1, TwoGreyGateResViT3Fusion.stat_dim)
    assert stats[0, 7].item() == pytest.approx(4 / 32)
    assert stats[0, 14].item() == pytest.approx(50 / 255)


def test_preprocess_images_uses_mix_input_and_align_gt_target(monkeypatch):
    class DummyPolicy(nn.Module):
        _preprocess_image_tensor = PI05Policy._preprocess_image_tensor

        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(
                mix_rgb_thermal=True,
                rgb_thermal_align_fusion=True,
                rgb_thermal_mix_source="precomputed",
                rgb_thermal_mix_use_precomputed=True,
                rgb_thermal_mix_rgb_feature=RGB_FEATURE,
                rgb_thermal_mix_thermal_feature=THERMAL_FEATURE,
                rgb_thermal_mix_precomputed_feature=RGB_THERMAL_MIX_FEATURE,
                rgb_thermal_align_gt_feature=RGB_THERMAL_GT_FEATURE,
                image_resolution=(8, 8),
                image_features={
                    RGB_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                    THERMAL_FEATURE: SimpleNamespace(shape=(3, 8, 8)),
                },
            )
            self._logged_missing_image_features = False
            self._logged_image_input_summary = False
            self._logged_rgb_thermal_mix = False

    monkeypatch.setattr(
        "lerobot.policies.pi05.modeling_pi05.save_pi05_image_input_snapshots",
        lambda *args, **kwargs: None,
    )
    policy = DummyPolicy()
    batch = {
        RGB_FEATURE: torch.zeros((1, 3, 8, 8)),
        THERMAL_FEATURE: torch.zeros((1, 3, 8, 8)),
        RGB_THERMAL_MIX_FEATURE: torch.ones((1, 3, 8, 8)),
        RGB_THERMAL_GT_FEATURE: torch.full((1, 3, 8, 8), 0.25),
    }

    images, masks = PI05Policy._preprocess_images(policy, batch)
    target = PI05Policy._preprocess_rgb_thermal_align_target(policy, batch)

    assert masks[0].all()
    assert masks[1].all()
    torch.testing.assert_close(images[1], torch.ones_like(images[1]))
    torch.testing.assert_close(target, torch.full_like(target, -0.5))


def test_get_modal_rgb_color():
    image = torch.tensor(
        [
            [[10, 10], [90, 10]],
            [[20, 20], [80, 20]],
            [[30, 30], [70, 30]],
        ],
        dtype=torch.uint8,
    )
    assert get_modal_rgb_color(image) == (10, 20, 30)


def test_words_ignore_is_case_insensitive_and_cleans_spacing():
    step = Pi05PrepareStateTokenizerProcessorStep(words_ignore=["please", "carefully"])
    assert step._remove_ignored_words("Please pick carefully, then place.") == "pick, then place."


def test_feature_selection_keeps_bookkeeping_and_requested_features():
    features = {
        "timestamp": {"dtype": "float32"},
        "frame_index": {"dtype": "int64"},
        "episode_index": {"dtype": "int64"},
        "index": {"dtype": "int64"},
        "task_index": {"dtype": "int64"},
        "action": {"dtype": "float32"},
        RGB_FEATURE: {"dtype": "video"},
        THERMAL_FEATURE: {"dtype": "video"},
        "observation.images.unused": {"dtype": "video"},
    }
    meta = SimpleNamespace(
        info=SimpleNamespace(features=features),
        stats={key: {} for key in features},
    )
    meta.features = meta.info.features

    filter_dataset_metadata_features(meta, ["action", RGB_FEATURE, THERMAL_FEATURE])

    assert set(meta.info.features) == {
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "action",
        RGB_FEATURE,
        THERMAL_FEATURE,
    }
    assert "observation.images.unused" not in meta.stats


def test_feature_selection_rejects_unknown_feature():
    meta = SimpleNamespace(
        info=SimpleNamespace(features={"action": {"dtype": "float32"}}),
        stats={},
    )
    meta.features = meta.info.features

    with pytest.raises(ValueError, match="not present"):
        filter_dataset_metadata_features(meta, ["missing"])


@pytest.mark.parametrize(
    "feature_names",
    [
        None,
        ["action", RGB_FEATURE],
        ["action", THERMAL_FEATURE],
    ],
)
def test_rgb_thermal_mix_rejects_missing_feature_names(feature_names):
    policy = SimpleNamespace(
        mix_rgb_thermal=True,
        rgb_thermal_mix_rgb_feature=RGB_FEATURE,
        rgb_thermal_mix_thermal_feature=THERMAL_FEATURE,
    )
    cfg = SimpleNamespace(
        trainable_config=policy,
        dataset=SimpleNamespace(feature_names=feature_names),
    )
    with pytest.raises(ValueError, match="dataset.feature_names"):
        validate_rgb_thermal_mix_feature_names(cfg)


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
