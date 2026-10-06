#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import math
from copy import deepcopy
from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224
DEFAULT_ANYTHERMAL_CHECKPOINT_PATH = (
    "pretrained_checkpoints/backbone/AnyThermal_full/model20.pth"
)
DEFAULT_RGB_DINOV2_CHECKPOINT_PATH = (
    "pretrained_checkpoints/backbone/Dinov2/dinov2_vitb14_pretrain.pth"
)
DEFAULT_THERMAL_GREY_BACKGROUND_ROI = (0, 0, 64, 64)
DEFAULT_THERMAL_GREY_BACKGROUND_VALUE = 80.0
DEFAULT_THERMAL_BLACK_BACKGROUND_OUTLIER_THRESHOLD = 12.0
DEFAULT_THERMAL_BLACK_BACKGROUND_OUTLIER_RATIO = 2.0
DEFAULT_THERMAL_TWOGREY_COLD_SUFFIX = "_cold"
DEFAULT_THERMAL_TWOGREY_HOT_SUFFIX = "_hot"
FIXED_MATCHED_THERMAL_INPUT_TYPES = frozenset({"twofixmatchingblack"})
ONLINE_MATCHED_THERMAL_INPUT_TYPES = frozenset({"matching", "twomatchingblack"})
TWO_STREAM_THERMAL_INPUT_TYPES = frozenset(
    {"twogrey", "twoblack", "twomatchingblack", "twofixmatchingblack"}
)
BLACK_IMPORTANCE_THERMAL_INPUT_TYPES = frozenset(
    {"twoblack", "twomatchingblack", "twofixmatchingblack"}
)
MATCHED_THERMAL_INPUT_TYPES = ONLINE_MATCHED_THERMAL_INPUT_TYPES | FIXED_MATCHED_THERMAL_INPUT_TYPES


def thermal_twogrey_feature_names(
    feature_name: str,
    *,
    cold_suffix: str = DEFAULT_THERMAL_TWOGREY_COLD_SUFFIX,
    hot_suffix: str = DEFAULT_THERMAL_TWOGREY_HOT_SUFFIX,
) -> tuple[str, str]:
    return f"{feature_name}{cold_suffix}", f"{feature_name}{hot_suffix}"


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Relative actions: converts absolute actions to relative (relative to state).
    use_relative_actions: bool = False
    # Joint names to exclude from relative (kept absolute). Empty list = all dims relative.
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated at runtime from dataset metadata by make_policy.
    action_feature_names: list[str] | None = None

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`
    # Compatibility switch for old Unitree PI0.5 checkpoints trained/evaluated with
    # lerobot0's float letterbox padding. When enabled, float padding is inserted as
    # -1.0 before the final `img * 2 - 1` normalization, producing -3.0 padding in
    # the logged/model image tensor. Keep False for new training/eval.
    legacy_resize_padding: bool = False

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    # Select the encoder/fusion path for regular RGB cameras. "vit" keeps the
    # default PI05 SigLIP/PaliGemma vision path. "dinov2" replaces regular RGB
    # image embeddings with project-local DINOv2 ViT-B/14 tokens projected to
    # PI05 dim. "GateTwoResViT" keeps the regular RGB ViT for original camera
    # images, but replaces the configured head-camera slot with three text-gated
    # residual streams built from red/green/blue black-importance views. Those
    # three views share one dedicated color ViT whose parameters are independent
    # from the original RGB ViT.
    rgb_input_type: str | None = "vit"

    # RGB preprocessing selected by the top-level ``--rgb_input_type`` alias.
    # "withthreeblack" derives red/green/blue evidence maps from the head RGB
    # image. Strong channel-specific color evidence is black and neutral
    # background is white. Selecting GateTwoResViT enables this automatically.
    rgb_threeblack_input_type: str | bool | None = None
    rgb_threeblack_head_feature: str = "observation.images.cam_left_high"
    rgb_threeblack_num_heads: int = 8
    rgb_threeblack_hidden_dim: int | None = None
    rgb_threeblack_temperature: float = 1.0
    rgb_threeblack_gate_init_std: float = 0.02
    rgb_threeblack_red_alpha: float = 1.0
    rgb_threeblack_green_alpha: float = 1.0
    rgb_threeblack_blue_alpha: float = 1.0

    # Friendly training/eval selector for thermal inputs. "anythermal" maps the
    # configured thermal feature(s) to the AnyThermal-DINOv2 encoder below.
    # "resthermal" cross-attends AnyThermal patch tokens into the head RGB tokens
    # and adds a zero-initialized residual to that RGB token stream. "ViT"
    # creates a thermal-only ViT that shares the RGB projector. "ResViT" adds
    # thermal raw ViT tokens to head RGB raw ViT tokens before the shared RGB projector.
    # "GateViT" projects cold/hot tokens to PI05 space, text-gates them into one
    # independent thermal token stream, and keeps all RGB streams unchanged.
    # "GateActionViT" keeps both projected thermal streams and uses the
    # text-derived cold/hot gates only as action-to-thermal attention biases.
    # "DoubleGateViT" combines text and head-RGB-to-thermal alignment gates,
    # then merges cold/hot into one independent thermal token stream.
    # "DoubleGateTwoResViT" uses the same scalar double gate but emits two
    # head-RGB residual streams.
    # "PatchGateViT" uses lightweight text/RGB context to choose cold/hot and
    # relevance independently for every projected thermal patch token.
    # "SmallPatchGateViT" uses the same gate at a coarser square spatial grid.
    # "HardSmallPatchGateViT" uses a fixed 4x4 coarse gate and straight-through
    # hard routing so every output patch is exactly one cold or hot token.
    # "PatchSingleGateResViT" uses a full per-ViT-patch text-only gate and
    # fuses cold/hot residuals into one head-RGB stream.
    # "SmallPatchGateTwoResViT" reuses that coarse gate but emits separate
    # head-RGB+cold and head-RGB+hot residual streams.
    # "SmallPatchSingleGateTwoResViT" keeps that two-stream output but drops
    # the RGB gate branch, so the coarse gates are text-conditioned only.
    # "SmallPatchGateResViT" uses the same coarse gate but fuses cold/hot
    # residuals into one head-RGB stream; "SmallPatchSingleGateResViT" drops
    # the RGB gate branch and keeps only text-conditioned coarse gates.
    # "GateTwoResViT" uses the GateResViT scalar text gate but emits those two
    # residual streams. "GateOneResViT" is the single thermal-image ablation:
    # it uses thermal_input_type='grey', text-gates one thermal residual, and
    # emits one head-RGB residual stream.
    # "GateViTMix" additionally applies local bidirectional cross-attention only
    # between that thermal stream and head RGB, then emits both unchanged head
    # RGB and a head-conditioned independent thermal stream.
    # "GateResViT" projects RGB/cold/hot tokens to PI05 space, then uses text
    # cross-attention to choose cold/hot residual coefficients. "GateResandViT"
    # emits those thermal-residual head RGB tokens and keeps original head RGB
    # tokens too. "GateResViT3" adds a thermal-statistics confidence gate for
    # the whole thermal residual.
    # "ResViTAttention" projects both streams to PI05 space, refines thermal
    # tokens with RGB/text cross-attention, and injects a gated residual into head RGB.
    thermal_input_type: str | bool | None = None

    # Select the thermal-specific visual path. False means no separate thermal
    # channel: thermal image features are embedded like any RGB camera by the
    # shared PaliGemma/SigLIP ViT. "vit" creates a dedicated thermal ViT +
    # multimodal projector. "shared_projector_vit" creates a dedicated thermal
    # ViT and reuses the RGB multimodal projector. "resvit" adds thermal raw ViT
    # tokens to head RGB raw ViT tokens before that shared projector.
    # "gatevit" uses thermal_input_type='twogrey' and text-gates projected
    # cold/hot tokens into one independent thermal token stream. "gateactionvit"
    # keeps both streams and biases only action queries toward cold or hot.
    # "doublegatevit" combines an independent head-RGB alignment gate with the
    # text gate before merging cold/hot into one thermal stream.
    # "doublegatetworesvit" keeps the scalar double gate but emits separate
    # cold/hot head-RGB residual streams.
    # "gatevitmix"
    # lets that stream and head RGB cross-attend locally, while preserving the
    # original head RGB prefix tokens. "gateresvit"
    # projects RGB/cold/hot tokens to PI05 space, then text-gates cold/hot
    # residual coefficients into one RGB stream. "gateresandvit" also keeps the
    # unmodified head RGB stream in the prefix. "gatetworesvit" uses the same
    # scalar gate but emits separate cold-RGB and hot-RGB residual streams.
    # "gateoneresvit" uses thermal_input_type='grey' and emits one scalar
    # text-gated thermal residual stream.
    # "gateresvit3" additionally gates thermal residual strength from cold/hot
    # image statistics such as contrast, area, and variance.
    # "resvitattention" uses projected 2048-dim RGB/thermal/text tokens and
    # injects a Thermo-VL-style gated residual into head RGB. "resnet18" uses a
    # ResNet18 thermal encoder. "cvae" uses the copied cold-water-bottle CVAE
    # encoder. "anythermal" uses the AnyThermal DINOv2 backbone and projects its
    # tokens to PI05 space. "resthermal" injects AnyThermal information as a
    # residual on head RGB tokens.
    thermal_encoder_channel: str | bool | None = None

    # Deprecated compatibility aliases. Prefer thermal_encoder_channel.
    share_rgb_thermal_vit: bool | None = None
    use_thermal_encoder: bool | None = None
    thermal_image_features: list[str] = field(
        default_factory=lambda: ["observation.images.cam_thermal"]
    )
    # thermal_input_type='grey' converts raw thermal frames to one-channel grey,
    # shifts this fixed ROI median to thermal_grey_background_value in uint8
    # space, then repeats the channel to RGB.
    # thermal_input_type='twogrey' applies the same ROI normalization, then
    # splits values below/above thermal_grey_background_value into cold/hot views.
    # thermal_input_type='twoblack' estimates the background from three top ROIs,
    # optionally rejects one clear outlier, and maps important cold/hot pixels
    # to black while mapping the common background to white.
    # thermal_input_type='twomatchingblack' aligns thermal to head RGB with
    # frame-varying MINIMA homographies. 'twofixmatchingblack' estimates one
    # dataset-level homography before training, stores it in every checkpoint,
    # and reuses it for every training and evaluation frame. Both then apply the
    # black-importance split with white missing areas.
    thermal_grey_background_roi: tuple[int, int, int, int] = DEFAULT_THERMAL_GREY_BACKGROUND_ROI
    thermal_grey_background_value: float = DEFAULT_THERMAL_GREY_BACKGROUND_VALUE
    thermal_black_background_outlier_threshold: float = (
        DEFAULT_THERMAL_BLACK_BACKGROUND_OUTLIER_THRESHOLD
    )
    thermal_black_background_outlier_ratio: float = DEFAULT_THERMAL_BLACK_BACKGROUND_OUTLIER_RATIO
    thermal_twogrey_cold_suffix: str = DEFAULT_THERMAL_TWOGREY_COLD_SUFFIX
    thermal_twogrey_hot_suffix: str = DEFAULT_THERMAL_TWOGREY_HOT_SUFFIX
    thermal_twogrey_resvit_cold_alpha: float = 1.0
    thermal_twogrey_resvit_hot_beta: float = 1.0
    thermal_twogrey_gatevit_num_heads: int = 8
    thermal_twogrey_gatevit_hidden_dim: int | None = None
    thermal_twogrey_gatevit_temperature: float = 1.0
    thermal_twogrey_gatevit_gate_init_std: float = 0.02
    thermal_twogrey_gateactionvit_num_heads: int = 8
    thermal_twogrey_gateactionvit_hidden_dim: int | None = None
    thermal_twogrey_gateactionvit_temperature: float = 1.0
    thermal_twogrey_gateactionvit_gate_init_std: float = 0.02
    thermal_twogrey_gateactionvit_bias_epsilon: float = 0.1
    thermal_twogrey_gateactionvit_bias_max_strength: float = 1.0
    thermal_twogrey_gateactionvit_bias_init_strength: float = 0.25
    thermal_twogrey_doublegatevit_num_heads: int = 8
    thermal_twogrey_doublegatevit_hidden_dim: int | None = None
    thermal_twogrey_doublegatevit_temperature: float = 1.0
    thermal_twogrey_doublegatevit_gate_init_std: float = 0.02
    thermal_twogrey_doublegatevit_rgb_gate_max_strength: float = 1.0
    thermal_twogrey_doublegatevit_rgb_gate_init_strength: float = 0.5
    thermal_twogrey_patchgatevit_gate_dim: int = 256
    thermal_twogrey_patchgatevit_num_heads: int = 4
    thermal_twogrey_patchgatevit_hidden_dim: int = 256
    thermal_twogrey_patchgatevit_temperature: float = 1.0
    thermal_twogrey_patchgatevit_gate_init_std: float = 0.01
    thermal_twogrey_patchgatevit_evidence_strength: float = 1.0
    thermal_twogrey_patchgatevit_evidence_floor: float = 0.05
    thermal_twogrey_patchgatevit_detach_head_rgb: bool = True
    thermal_twogrey_smallpatchgatevit_gate_grid: tuple[int, int] = (4, 4)
    thermal_twogrey_gatevitmix_num_heads: int = 8
    thermal_twogrey_gatevitmix_hidden_dim: int | None = None
    thermal_twogrey_gatevitmix_mix_dim: int = 512
    thermal_twogrey_gatevitmix_temperature: float = 1.0
    thermal_twogrey_gatevitmix_gate_init_std: float = 0.02
    thermal_twogrey_gatevitmix_context_scale: float = 0.5
    thermal_twogrey_gatevitmix_detach_head_rgb: bool = True
    thermal_twogrey_gateresvit_num_heads: int = 8
    thermal_twogrey_gateresvit_hidden_dim: int | None = None
    thermal_twogrey_gateresvit_temperature: float = 1.0
    thermal_twogrey_gateresvit_gate_init_std: float = 0.02
    thermal_grey_gateoneresvit_num_heads: int = 8
    thermal_grey_gateoneresvit_hidden_dim: int | None = None
    thermal_grey_gateoneresvit_temperature: float = 1.0
    thermal_grey_gateoneresvit_gate_init_std: float = 0.02
    thermal_twogrey_gateresvit3_num_heads: int = 8
    thermal_twogrey_gateresvit3_hidden_dim: int | None = None
    thermal_twogrey_gateresvit3_stat_hidden_dim: int = 64
    thermal_twogrey_gateresvit3_residual_hidden_dim: int | None = None
    thermal_twogrey_gateresvit3_temperature: float = 1.0
    thermal_twogrey_gateresvit3_gate_init_std: float = 0.02
    thermal_twogrey_gateresvit3_confidence_bias_init: float = -3.0
    thermal_twogrey_gateresvit3_evidence_gain: float = 48.0
    thermal_twogrey_gateresvit3_evidence_threshold: float = 0.08
    # Kept for old configs and future custom encoders; ResNet18 uses a fixed 512-dim feature map.
    thermal_encoder_hidden_dim: int = 128
    thermal_encoder_token_grid: tuple[int, int] = (7, 7)
    thermal_encoder_dropout: float = 0.0

    # AnyThermal backbone defaults use project-local checkpoints and DINOv2 code.
    thermal_anythermal_model_type: str = "auto"
    thermal_anythermal_checkpoint_path: str | None = DEFAULT_ANYTHERMAL_CHECKPOINT_PATH
    thermal_anythermal_dinov2_repo_path: str | None = None
    thermal_anythermal_freeze_backbone: bool = True
    thermal_anythermal_include_cls_token: bool = True
    thermal_anythermal_include_register_tokens: bool = True
    thermal_anythermal_input_range: str = "minus_one_to_one"
    thermal_resthermal_adapter_dim: int = 512
    thermal_resthermal_decoder_hidden_dim: int = 1024
    thermal_resthermal_num_heads: int = 8
    thermal_resvit_attention_num_heads: int = 8
    thermal_resvit_attention_merge_hidden_dim: int | None = None
    thermal_resvit_attention_residual_hidden_dim: int | None = None

    rgb_dinov2_model_type: str = "dinov2_vitb14"
    rgb_dinov2_checkpoint_path: str | None = DEFAULT_RGB_DINOV2_CHECKPOINT_PATH
    rgb_dinov2_repo_path: str | None = None
    rgb_dinov2_freeze_backbone: bool = True
    rgb_dinov2_input_range: str = "minus_one_to_one"

    # ColdWaterBottleImageCVAE defaults copied from the cold_water_bottle_cvae config.
    thermal_cvae_source_feature: str = "observation.images.cam_left_high"
    thermal_cvae_input_channels: int = 3
    thermal_cvae_condition_channels: int = 3
    thermal_cvae_target_channels: int = 3
    thermal_cvae_latent_dim: int = 256
    thermal_cvae_hidden_dims: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    thermal_cvae_image_size: int = 224
    thermal_cvae_token_grid_size: int = 16
    thermal_cvae_token_upsample_mode: str = "bilinear"
    thermal_cvae_token_norm: str = "layernorm"
    thermal_cvae_token_scale_init: float = 1.0
    thermal_cvae_pretrained_path: str | None = None
    thermal_cvae_freeze_encoder: bool = False
    # Deprecated compatibility alias. Prefer thermal_cvae_pretrained_path.
    thermal_cvae_checkpoint_path: str | None = None
    thermal_cvae_checkpoint_strict: bool = True

    # Append thermal-query tokens enriched by cross-attention to head RGB.
    thermal_fuse_with_head_rgb: bool = False
    thermal_fusion_head_rgb_feature: str = "observation.images.cam_left_high"
    thermal_fusion_num_heads: int = 8

    # Optional pixel-level RGB/thermal alignment and blending. The RGB camera
    # remains unchanged while the thermal camera slot receives the mixed image.
    mix_rgb_thermal: bool = False
    rgb_thermal_mix_rgb_feature: str = "observation.images.cam_left_high"
    rgb_thermal_mix_thermal_feature: str = "observation.images.cam_thermal"
    rgb_thermal_mix_precomputed_feature: str = "observation.images.cam_rgb_thermal_mixing"
    rgb_thermal_mix_source: str = "auto"
    rgb_thermal_mix_use_precomputed: bool = False
    rgb_thermal_mix_horizontal_offset: int = -85
    rgb_thermal_mix_vertical_offset: int = 0
    rgb_thermal_mix_thermal_width: int = 600
    rgb_thermal_mix_thermal_weight: float = 0.8
    rgb_thermal_mix_fill_color: tuple[int, int, int] | None = None
    rgb_thermal_mix_episode_fill_colors: list[tuple[int, int, int]] = field(default_factory=list)

    # MINIMA alignment used by thermal_input_type='matching',
    # 'twomatchingblack', and 'twofixmatchingblack'. Online modes share nearby
    # frame homographies. The fixed mode samples frames once before training and
    # stores a robust dataset-level homography in the policy checkpoint.
    thermal_matching_minima_root: str = "./MINIMA"
    thermal_matching_checkpoint: str = "./weights/minima_lightglue.pth"
    thermal_matching_ransac_reproj_threshold: float = 5.0
    thermal_matching_min_matches: int = 4
    thermal_matching_min_inliers: int = 4
    thermal_matching_match_every_n_frames: int = 10
    thermal_matching_cache_size: int = 32768
    thermal_matching_outer_padding: int = 30
    thermal_matching_preload: bool = True
    thermal_fixed_matching_num_samples: int = 16
    thermal_fixed_matching_min_valid_samples: int = 4
    thermal_fixed_matching_seed: int = 0
    thermal_fixed_matching_homographies: dict[str, list[list[float]]] = field(default_factory=dict)
    thermal_fixed_matching_source_sizes: dict[str, list[int]] = field(default_factory=dict)
    thermal_fixed_matching_output_sizes: dict[str, list[int]] = field(default_factory=dict)
    thermal_fixed_matching_dataset_roots: list[str] = field(default_factory=list)
    thermal_fixed_matching_sample_indices: list[int] = field(default_factory=list)
    thermal_fixed_matching_sample_timestamps: list[float] = field(default_factory=list)
    thermal_fixed_matching_diagnostics: dict[str, dict[str, float]] = field(default_factory=dict)

    # Token-level RGB/thermal alignment. This first implementation supports
    # thermal_encoder_channel='vit': head RGB and thermal are encoded by
    # separate ViTs, thermal tokens cross-attend into the head-RGB token grid,
    # and a training-only decoder learns against rgb_thermal_align_gt_feature.
    rgb_thermal_align_fusion: bool = False
    rgb_thermal_align_gt_feature: str = "observation.images.cam_rgb_thermal_mixing_GT"
    rgb_thermal_align_loss_weight: float = 0.1

    tokenizer_max_length: int = 200  # see openpi `__post_init__`
    words_ignore: list[str] = field(default_factory=list)

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections
    # Freeze the pretrained RGB/text/state PaliGemma path while keeping thermal
    # modules and the action expert trainable. For shared-projector thermal ViT
    # modes, the shared RGB/thermal projector remains frozen.
    freeze_non_thermal_input: bool = False

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    @staticmethod
    def _normalize_rgb_input_type(value: str | None) -> str:
        text = str(value or "vit").strip().strip("\"'“”‘’").lower()
        if text in {"", "vit", "siglip", "paligemma", "default"}:
            return "vit"
        if text in {"dinov2", "dino", "dino2"}:
            return "dinov2"
        if text in {
            "gatetworesvit",
            "gate_two_res_vit",
            "gate_twores_vit",
            "gate-two-res-vit",
            "gate2resvit",
            "gate_2res_vit",
            "withthreeblack",
            "with_three_black",
            "with-three-black",
        }:
            return "gatetworesvit"
        raise ValueError(
            "rgb_input_type must be 'vit', 'dinov2', 'GateTwoResViT', or "
            f"'withthreeblack'. Got {value!r}."
        )

    @staticmethod
    def _normalize_rgb_threeblack_input_type(value: str | bool | None) -> str:
        if isinstance(value, bool):
            if value is False:
                return "false"
            raise ValueError("rgb_threeblack_input_type=true is ambiguous; use 'withthreeblack'.")
        text = str(value or "false").strip().strip("\"'“”‘’").lower()
        if text in {"", "0", "false", "none", "off", "no", "disable", "disabled"}:
            return "false"
        if text in {"withthreeblack", "with_three_black", "with-three-black", "threeblack"}:
            return "withthreeblack"
        raise ValueError(
            "rgb_threeblack_input_type must be false or 'withthreeblack'. "
            f"Got {value!r}."
        )

    def _resolve_rgb_input_type(self) -> str:
        encoder_mode = self._normalize_rgb_input_type(self.rgb_input_type)
        preprocessing_mode = self._normalize_rgb_threeblack_input_type(
            self.rgb_threeblack_input_type
        )
        if encoder_mode == "gatetworesvit":
            preprocessing_mode = "withthreeblack"
        elif preprocessing_mode == "withthreeblack":
            if encoder_mode == "dinov2":
                raise ValueError(
                    "rgb_threeblack_input_type='withthreeblack' cannot be combined with "
                    "rgb_input_type='dinov2'. Use rgb_input_type='GateTwoResViT'."
                )
            encoder_mode = "gatetworesvit"
        self.rgb_threeblack_input_type = preprocessing_mode
        return encoder_mode

    @staticmethod
    def _normalize_thermal_input_type(value: str | bool | None) -> str:
        if isinstance(value, bool):
            if value is False:
                return "false"
            raise ValueError(
                "thermal_input_type=true is ambiguous; use false, 'grey', 'twogrey', "
                "'twoblack', 'twomatchingblack', 'twofixmatchingblack', 'mixing', "
                "'matching', 'anythermal', "
                "'resthermal', 'ViT', 'ResViT', or "
                "'ResViTAttention'."
            )
        raw_text = str(value or "false").strip().strip("\"'")
        if raw_text == "ViT" or raw_text in {
            "shared_projector_vit",
            "shared-projector-vit",
            "vit_shared_projector",
        }:
            return "shared_projector_vit"
        text = raw_text.lower()
        if text in {"", "0", "false", "none", "off", "no", "disable", "disabled"}:
            return "false"
        if text == "vit":
            return "vit"
        if text in {"grey", "gray"}:
            return "grey"
        if text in {"twogrey", "two_grey", "two-grey", "twogray", "two_gray", "two-gray"}:
            return "twogrey"
        if text in {"twoblack", "two_black", "two-black"}:
            return "twoblack"
        if text in {
            "twomatchingblack",
            "two_matching_black",
            "two-matching-black",
            "twomatchedblack",
            "two_matched_black",
            "two-matched-black",
        }:
            return "twomatchingblack"
        if text in {
            "twofixmatchingblack",
            "two_fix_matching_black",
            "two-fix-matching-black",
            "twofixedmatchingblack",
            "two_fixed_matching_black",
            "two-fixed-matching-black",
        }:
            return "twofixmatchingblack"
        if text in {"mix", "mixing"}:
            return "mixing"
        if text in {"match", "matched", "matching"}:
            return "matching"
        if text in {"anythermal", "any_thermal"}:
            return "anythermal"
        if text in {"resthermal", "res_thermal", "residual_thermal"}:
            return "resthermal"
        if text in {"resvit", "res_vit", "residual_vit", "residualvit"}:
            return "resvit"
        if text in {
            "resvitattention",
            "resvit_attention",
            "res_vit_attention",
            "residual_vit_attention",
            "residualvitattention",
        }:
            return "resvitattention"
        raise ValueError(
            "thermal_input_type must be one of false, 'grey', 'twogrey', 'twoblack', "
            "'twomatchingblack', 'twofixmatchingblack', 'mixing', 'matching', "
            f"'vit', 'ViT', 'ResViT', 'ResViTAttention', 'anythermal', or "
            f"'resthermal'. Got {value!r}."
        )

    @staticmethod
    def _normalize_thermal_encoder_channel(channel: str | bool | None) -> str | bool:
        if channel is None:
            return False
        if channel is False:
            return False
        if channel is True:
            raise ValueError(
                "thermal_encoder_channel=true is ambiguous; use false, 'vit', 'resnet18', "
                "'cvae', 'anythermal', 'resthermal', 'ViT', 'ResViT', 'GateViT', "
                "'GateActionViT', 'DoubleGateViT', 'DoubleGateTwoResViT', 'PatchGateViT', "
                "'PatchSingleGateResViT', 'SmallPatchGateViT', "
                "'HardSmallPatchGateViT', 'SmallPatchGateTwoResViT', "
                "'SmallPatchSingleGateTwoResViT', 'SmallPatchGateResViT', "
                "'SmallPatchSingleGateResViT', 'GateViTMix', 'GateResViT', "
                "'GateResandViT', 'GateTwoResViT', 'GateOneResViT', 'GateResViT3', "
                "or 'ResViTAttention'."
            )

        raw_channel_name = str(channel).strip().strip("\"'“”‘’")
        if raw_channel_name == "ViT" or raw_channel_name in {
            "shared_projector_vit",
            "shared-projector-vit",
            "vit_shared_projector",
        }:
            return "shared_projector_vit"
        channel_name = raw_channel_name.lower()
        if channel_name in {"", "false", "none", "off", "no", "rgb", "shared", "shared_vit"}:
            return False
        if channel_name in {
            "vit",
            "resnet18",
            "cvae",
            "anythermal",
            "resthermal",
            "resvit",
            "resvitattention",
        }:
            return channel_name
        if channel_name in {"gateresvit", "gate_resvit", "gate_res_vit", "gate-resvit"}:
            return "gateresvit"
        if channel_name in {
            "gateresandvit",
            "gate_res_and_vit",
            "gate_resand_vit",
            "gate-res-and-vit",
            "gateresvitandvit",
            "gate_res_vit_and_vit",
            "gateresplusvit",
            "gate_res_plus_vit",
        }:
            return "gateresandvit"
        if channel_name in {
            "gatetworesvit",
            "gate_two_res_vit",
            "gate_twores_vit",
            "gate-two-res-vit",
            "gate2resvit",
            "gate_2res_vit",
        }:
            return "gatetworesvit"
        if channel_name in {
            "gateoneresvit",
            "gate_one_res_vit",
            "gate_oneres_vit",
            "gate-one-res-vit",
            "gate1resvit",
            "gate_1res_vit",
        }:
            return "gateoneresvit"
        if channel_name in {"gatevit", "gate_vit", "gate-vit"}:
            return "gatevit"
        if channel_name in {
            "gateactionvit",
            "gate_action_vit",
            "gateaction_vit",
            "gate-action-vit",
        }:
            return "gateactionvit"
        if channel_name in {
            "doublegatevit",
            "double_gate_vit",
            "doublegate_vit",
            "double-gate-vit",
        }:
            return "doublegatevit"
        if channel_name in {
            "doublegatetworesvit",
            "double_gate_two_res_vit",
            "doublegate_two_res_vit",
            "double-gate-two-res-vit",
            "doublegate2resvit",
            "double_gate_2res_vit",
        }:
            return "doublegatetworesvit"
        if channel_name in {
            "patchgatevit",
            "patch_gate_vit",
            "patchgate_vit",
            "patch-gate-vit",
        }:
            return "patchgatevit"
        if channel_name in {
            "patchsinglegateresvit",
            "patch_single_gate_res_vit",
            "patchsingle_gate_res_vit",
            "patch-single-gate-res-vit",
            "patchsingleresvit",
            "patch_single_res_vit",
            "patch-single-res-vit",
        }:
            return "patchsinglegateresvit"
        if channel_name in {
            "smallpatchgatevit",
            "small_patch_gate_vit",
            "smallpatch_gate_vit",
            "small-patch-gate-vit",
            "smallpatchvit",
            "small_patch_vit",
            "small-patch-vit",
        }:
            return "smallpatchgatevit"
        if channel_name in {
            "hardsmallpatchgatevit",
            "hard_small_patch_gate_vit",
            "hardsmall_patch_gate_vit",
            "hard-small-patch-gate-vit",
            "hardsmallpatchvit",
            "hard_small_patch_vit",
            "hard-small-patch-vit",
        }:
            return "hardsmallpatchgatevit"
        if channel_name in {
            "smallpatchgatetworesvit",
            "small_patch_gate_two_res_vit",
            "smallpatch_gate_two_res_vit",
            "small-patch-gate-two-res-vit",
            "smallpatchgate2resvit",
            "small_patch_gate_2res_vit",
        }:
            return "smallpatchgatetworesvit"
        if channel_name in {
            "smallpatchsinglegatetworesvit",
            "small_patch_single_gate_two_res_vit",
            "smallpatch_single_gate_two_res_vit",
            "small-patch-single-gate-two-res-vit",
            "smallpatchsinglegate2resvit",
            "small_patch_single_gate_2res_vit",
            "smallpatchsingle2resvit",
            "small_patch_single_2res_vit",
        }:
            return "smallpatchsinglegatetworesvit"
        if channel_name in {
            "smallpatchgateresvit",
            "small_patch_gate_res_vit",
            "smallpatch_gate_res_vit",
            "small-patch-gate-res-vit",
            "smallpatchresvit",
            "small_patch_res_vit",
            "small-patch-res-vit",
        }:
            return "smallpatchgateresvit"
        if channel_name in {
            "smallpatchsinglegateresvit",
            "small_patch_single_gate_res_vit",
            "smallpatch_single_gate_res_vit",
            "small-patch-single-gate-res-vit",
            "smallpatchsingleresvit",
            "small_patch_single_res_vit",
            "small-patch-single-res-vit",
        }:
            return "smallpatchsinglegateresvit"
        if channel_name in {
            "gatevitmix",
            "gate_vit_mix",
            "gatevit_mix",
            "gate-vit-mix",
        }:
            return "gatevitmix"
        if channel_name in {
            "gateresvit3",
            "gate_resvit3",
            "gate_res_vit3",
            "gate-resvit3",
            "gateresvit_3",
        }:
            return "gateresvit3"
        raise ValueError(
            "thermal_encoder_channel must be false, 'vit', 'resnet18', 'cvae', "
            "'anythermal', 'resthermal', 'ViT', 'ResViT', 'GateViT', 'GateActionViT', "
            "'DoubleGateViT', 'DoubleGateTwoResViT', 'PatchGateViT', "
            "'PatchSingleGateResViT', 'SmallPatchGateViT', "
            "'HardSmallPatchGateViT', 'SmallPatchGateTwoResViT', "
            "'SmallPatchSingleGateTwoResViT', 'SmallPatchGateResViT', "
            "'SmallPatchSingleGateResViT', 'GateViTMix', 'GateResViT', "
            "'GateResandViT', 'GateTwoResViT', 'GateOneResViT', 'GateResViT3', "
            "or 'ResViTAttention'. "
            f"Got {channel!r}."
        )

    def _resolve_thermal_encoder_channel(self) -> str | bool:
        channel = (
            self._normalize_thermal_encoder_channel(self.thermal_encoder_channel)
            if self.thermal_encoder_channel is not None
            else None
        )
        if self.thermal_input_type in {
            "vit",
            "shared_projector_vit",
            "resvit",
            "resvitattention",
            "anythermal",
            "resthermal",
        }:
            if channel not in {None, False, self.thermal_input_type}:
                raise ValueError(
                    f"thermal_input_type={self.thermal_input_type!r} cannot be combined with "
                    f"thermal_encoder_channel={self.thermal_encoder_channel!r}."
                )
            return self.thermal_input_type
        if self.thermal_input_type in TWO_STREAM_THERMAL_INPUT_TYPES:
            if channel not in {
                None,
                False,
                "shared_projector_vit",
                "resvit",
                "gatevit",
                "gateactionvit",
                "doublegatevit",
                "doublegatetworesvit",
                "patchgatevit",
                "patchsinglegateresvit",
                "smallpatchgatevit",
                "hardsmallpatchgatevit",
                "smallpatchgatetworesvit",
                "smallpatchsinglegatetworesvit",
                "smallpatchgateresvit",
                "smallpatchsinglegateresvit",
                "gatevitmix",
                "gateresvit",
                "gateresandvit",
                "gatetworesvit",
                "gateresvit3",
            }:
                raise ValueError(
                    f"thermal_input_type={self.thermal_input_type!r} currently supports only "
                    "thermal_encoder_channel='ViT' / 'shared_projector_vit' / "
                    "'ResViT' / 'GateViT' / 'GateActionViT' / "
                    "'DoubleGateViT' / 'DoubleGateTwoResViT' / "
                    "'PatchGateViT' / 'PatchSingleGateResViT' / "
                    "'SmallPatchGateViT' / 'SmallPatchGateTwoResViT' / "
                    "'SmallPatchSingleGateTwoResViT' / 'SmallPatchGateResViT' / "
                    "'SmallPatchSingleGateResViT' / 'HardSmallPatchGateViT' / "
                    "'GateViTMix' / 'GateResViT' / 'GateResandViT' / "
                    "'GateTwoResViT' / 'GateResViT3'. "
                    f"Got thermal_encoder_channel={self.thermal_encoder_channel!r}."
                )
            return (
                channel
                if channel
                in {
                    "resvit",
                    "gatevit",
                    "gateactionvit",
                    "doublegatevit",
                    "doublegatetworesvit",
                    "patchgatevit",
                    "patchsinglegateresvit",
                    "smallpatchgatevit",
                    "hardsmallpatchgatevit",
                    "smallpatchgatetworesvit",
                    "smallpatchsinglegatetworesvit",
                    "smallpatchgateresvit",
                    "smallpatchsinglegateresvit",
                    "gatevitmix",
                    "gateresvit",
                    "gateresandvit",
                    "gatetworesvit",
                    "gateresvit3",
                }
                else "shared_projector_vit"
            )
        if channel is not None:
            return channel
        if self.use_thermal_encoder:
            return "resnet18"
        if self.share_rgb_thermal_vit is False:
            return "vit"
        return False

    def apply_thermal_input_type(self, value: str | bool | None) -> None:
        """Apply the top-level train/eval ``--thermal_input_type`` alias."""
        mode = self._normalize_thermal_input_type(value)
        current_channel = self._normalize_thermal_encoder_channel(self.thermal_encoder_channel)
        if mode == "false" and current_channel == "gateoneresvit":
            mode = "grey"
        if mode == "false" and current_channel in {
            "gatevit",
            "gateactionvit",
            "doublegatevit",
            "doublegatetworesvit",
            "patchgatevit",
            "patchsinglegateresvit",
            "smallpatchgatevit",
            "hardsmallpatchgatevit",
            "smallpatchgatetworesvit",
            "smallpatchsinglegatetworesvit",
            "smallpatchgateresvit",
            "smallpatchsinglegateresvit",
            "gatevitmix",
            "gateresvit",
            "gateresandvit",
            "gatetworesvit",
            "gateresvit3",
        }:
            mode = "twogrey"
        if mode in {
            "vit",
            "shared_projector_vit",
            "resvit",
            "resvitattention",
            "anythermal",
            "resthermal",
        }:
            if current_channel not in {False, mode}:
                raise ValueError(
                    f"thermal_input_type={mode!r} cannot override "
                    f"thermal_encoder_channel={self.thermal_encoder_channel!r}."
                )
            self.thermal_encoder_channel = mode
            self.mix_rgb_thermal = False
            if mode in {"anythermal", "resthermal"}:
                self._validate_anythermal_config()
            if mode == "resthermal":
                self._validate_resthermal_config()
            if mode == "shared_projector_vit":
                self._validate_shared_projector_thermal_vit_config("ViT")
            if mode == "resvit":
                self._validate_residual_thermal_vit_config()
            if mode == "resvitattention":
                self._validate_resvit_attention_config()
        elif mode in TWO_STREAM_THERMAL_INPUT_TYPES:
            if current_channel not in {
                False,
                "shared_projector_vit",
                "resvit",
                "gatevit",
                "gateactionvit",
                "doublegatevit",
                "doublegatetworesvit",
                "patchgatevit",
                "patchsinglegateresvit",
                "smallpatchgatevit",
                "hardsmallpatchgatevit",
                "smallpatchgatetworesvit",
                "smallpatchsinglegatetworesvit",
                "smallpatchgateresvit",
                "smallpatchsinglegateresvit",
                "gatevitmix",
                "gateresvit",
                "gateresandvit",
                "gatetworesvit",
                "gateresvit3",
            }:
                raise ValueError(
                    f"thermal_input_type={mode!r} currently supports only "
                    "thermal_encoder_channel='ViT' / 'shared_projector_vit' / "
                    "'ResViT' / 'GateViT' / 'GateActionViT' / "
                    "'DoubleGateViT' / 'DoubleGateTwoResViT' / "
                    "'PatchGateViT' / 'PatchSingleGateResViT' / "
                    "'SmallPatchGateViT' / 'HardSmallPatchGateViT' / "
                    "'SmallPatchGateTwoResViT' / 'SmallPatchSingleGateTwoResViT' / "
                    "'SmallPatchGateResViT' / 'SmallPatchSingleGateResViT' / "
                    "'GateViTMix' / 'GateResViT' / 'GateResandViT' / "
                    "'GateTwoResViT' / 'GateResViT3'. "
                    f"Got thermal_encoder_channel={self.thermal_encoder_channel!r}."
                )
            self.thermal_encoder_channel = (
                current_channel
                if current_channel
                in {
                    "resvit",
                    "gatevit",
                    "gateactionvit",
                    "doublegatevit",
                    "doublegatetworesvit",
                    "patchgatevit",
                    "patchsinglegateresvit",
                    "smallpatchgatevit",
                    "hardsmallpatchgatevit",
                    "smallpatchgatetworesvit",
                    "smallpatchsinglegatetworesvit",
                    "smallpatchgateresvit",
                    "smallpatchsinglegateresvit",
                    "gatevitmix",
                    "gateresvit",
                    "gateresandvit",
                    "gatetworesvit",
                    "gateresvit3",
                }
                else "shared_projector_vit"
            )
            self.mix_rgb_thermal = False
            self._validate_thermal_grey_config()
            self._validate_thermal_twogrey_config()
            if mode in BLACK_IMPORTANCE_THERMAL_INPUT_TYPES:
                self._validate_thermal_black_config()
            if mode in MATCHED_THERMAL_INPUT_TYPES:
                self._validate_thermal_matching_config()
            if mode in FIXED_MATCHED_THERMAL_INPUT_TYPES:
                self._validate_thermal_fixed_matching_config()
            if self.thermal_encoder_channel == "resvit":
                self._validate_residual_thermal_vit_config()
            elif self.thermal_encoder_channel == "gatevit":
                self._validate_gate_thermal_vit_config()
            elif self.thermal_encoder_channel == "gateactionvit":
                self._validate_gate_action_thermal_vit_config()
            elif self.thermal_encoder_channel in {"doublegatevit", "doublegatetworesvit"}:
                self._validate_double_gate_thermal_vit_config()
            elif self.thermal_encoder_channel in {
                "patchgatevit",
                "patchsinglegateresvit",
                "smallpatchgatevit",
                "hardsmallpatchgatevit",
                "smallpatchgatetworesvit",
                "smallpatchsinglegatetworesvit",
                "smallpatchgateresvit",
                "smallpatchsinglegateresvit",
            }:
                self._validate_patch_gate_thermal_vit_config()
            elif self.thermal_encoder_channel == "gatevitmix":
                self._validate_gate_thermal_vit_mix_config()
            elif self.thermal_encoder_channel in {"gateresvit", "gateresandvit"}:
                self._validate_gate_residual_thermal_vit_config()
            elif self.thermal_encoder_channel == "gatetworesvit":
                self._validate_gate_residual_thermal_vit_config()
            elif self.thermal_encoder_channel == "gateresvit3":
                self._validate_gate_residual_thermal_vit3_config()
            else:
                self._validate_shared_projector_thermal_vit_config(
                    "TwoBlack" if mode in BLACK_IMPORTANCE_THERMAL_INPUT_TYPES else "TwoGrey"
                )
        elif mode in {"grey", "matching"}:
            self.mix_rgb_thermal = False
            if mode == "grey":
                self._validate_thermal_grey_config()
            else:
                self._validate_thermal_matching_config()
        self.thermal_input_type = mode
        if self.thermal_encoder_channel == "gateoneresvit":
            self._validate_gate_one_residual_thermal_vit_config()

    def _validate_thermal_grey_config(self) -> None:
        if len(self.thermal_grey_background_roi) != 4:
            raise ValueError(
                "thermal_grey_background_roi must be (top, left, height, width), "
                f"got {self.thermal_grey_background_roi!r}."
            )
        top, left, height, width = (int(value) for value in self.thermal_grey_background_roi)
        if top < 0 or left < 0 or height <= 0 or width <= 0:
            raise ValueError(
                "thermal_grey_background_roi must have non-negative top/left and positive height/width, "
                f"got {self.thermal_grey_background_roi!r}."
            )
        self.thermal_grey_background_roi = (top, left, height, width)
        if not 0.0 <= float(self.thermal_grey_background_value) <= 255.0:
            raise ValueError(
                "thermal_grey_background_value must be in [0, 255], "
                f"got {self.thermal_grey_background_value!r}."
            )
        self.thermal_grey_background_value = float(self.thermal_grey_background_value)

    def _validate_thermal_twogrey_config(self) -> None:
        cold_suffix = str(self.thermal_twogrey_cold_suffix or "")
        hot_suffix = str(self.thermal_twogrey_hot_suffix or "")
        if not cold_suffix or not hot_suffix:
            raise ValueError("thermal twogrey cold/hot suffixes must be non-empty strings.")
        if cold_suffix == hot_suffix:
            raise ValueError(
                "thermal_twogrey_cold_suffix and thermal_twogrey_hot_suffix must be different."
            )
        self.thermal_twogrey_cold_suffix = cold_suffix
        self.thermal_twogrey_hot_suffix = hot_suffix
        self.thermal_twogrey_resvit_cold_alpha = float(self.thermal_twogrey_resvit_cold_alpha)
        self.thermal_twogrey_resvit_hot_beta = float(self.thermal_twogrey_resvit_hot_beta)
        if not math.isfinite(self.thermal_twogrey_resvit_cold_alpha):
            raise ValueError("thermal_twogrey_resvit_cold_alpha must be finite")
        if not math.isfinite(self.thermal_twogrey_resvit_hot_beta):
            raise ValueError("thermal_twogrey_resvit_hot_beta must be finite")

    def _validate_thermal_black_config(self) -> None:
        self.thermal_black_background_outlier_threshold = float(
            self.thermal_black_background_outlier_threshold
        )
        self.thermal_black_background_outlier_ratio = float(
            self.thermal_black_background_outlier_ratio
        )
        if (
            not math.isfinite(self.thermal_black_background_outlier_threshold)
            or self.thermal_black_background_outlier_threshold < 0
        ):
            raise ValueError(
                "thermal_black_background_outlier_threshold must be finite and non-negative"
            )
        if (
            not math.isfinite(self.thermal_black_background_outlier_ratio)
            or self.thermal_black_background_outlier_ratio < 1
        ):
            raise ValueError(
                "thermal_black_background_outlier_ratio must be finite and at least 1"
            )

    def _validate_thermal_matching_config(self) -> None:
        if not str(self.thermal_matching_minima_root or "").strip():
            raise ValueError("thermal_matching_minima_root must be a non-empty path")
        if not str(self.thermal_matching_checkpoint or "").strip():
            raise ValueError("thermal_matching_checkpoint must be a non-empty path")
        self.thermal_matching_ransac_reproj_threshold = float(
            self.thermal_matching_ransac_reproj_threshold
        )
        if (
            not math.isfinite(self.thermal_matching_ransac_reproj_threshold)
            or self.thermal_matching_ransac_reproj_threshold <= 0
        ):
            raise ValueError(
                "thermal_matching_ransac_reproj_threshold must be finite and positive"
            )
        for field_name in (
            "thermal_matching_min_matches",
            "thermal_matching_min_inliers",
            "thermal_matching_match_every_n_frames",
            "thermal_matching_cache_size",
        ):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
            setattr(self, field_name, value)
        self.thermal_matching_outer_padding = int(self.thermal_matching_outer_padding)
        if self.thermal_matching_outer_padding < 0:
            raise ValueError("thermal_matching_outer_padding must be non-negative")
        if self.rgb_thermal_mix_rgb_feature in set(self.thermal_image_features):
            raise ValueError(
                "Thermal matching requires distinct head RGB and thermal features, "
                f"got {self.rgb_thermal_mix_rgb_feature!r}."
            )

    def _validate_thermal_fixed_matching_config(self) -> None:
        self.thermal_fixed_matching_num_samples = int(self.thermal_fixed_matching_num_samples)
        self.thermal_fixed_matching_min_valid_samples = int(
            self.thermal_fixed_matching_min_valid_samples
        )
        self.thermal_fixed_matching_seed = int(self.thermal_fixed_matching_seed)
        if self.thermal_fixed_matching_num_samples <= 0:
            raise ValueError("thermal_fixed_matching_num_samples must be positive")
        if self.thermal_fixed_matching_min_valid_samples <= 0:
            raise ValueError("thermal_fixed_matching_min_valid_samples must be positive")
        if (
            self.thermal_fixed_matching_min_valid_samples
            > self.thermal_fixed_matching_num_samples
        ):
            raise ValueError(
                "thermal_fixed_matching_min_valid_samples cannot exceed "
                "thermal_fixed_matching_num_samples"
            )

        for feature_name, homography in self.thermal_fixed_matching_homographies.items():
            if len(homography) != 3 or any(len(row) != 3 for row in homography):
                raise ValueError(
                    "Each thermal_fixed_matching_homographies entry must be a 3x3 matrix; "
                    f"got {feature_name!r}: {homography!r}."
                )
            values = [float(value) for row in homography for value in row]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"Fixed thermal homography for {feature_name!r} contains non-finite values."
                )
            determinant = (
                values[0] * (values[4] * values[8] - values[5] * values[7])
                - values[1] * (values[3] * values[8] - values[5] * values[6])
                + values[2] * (values[3] * values[7] - values[4] * values[6])
            )
            if abs(determinant) < 1e-10:
                raise ValueError(
                    f"Fixed thermal homography for {feature_name!r} is singular."
                )
            for size_field_name in (
                "thermal_fixed_matching_source_sizes",
                "thermal_fixed_matching_output_sizes",
            ):
                size_map = getattr(self, size_field_name)
                size = size_map.get(feature_name)
                if size is None or len(size) != 2 or any(int(value) <= 0 for value in size):
                    raise ValueError(
                        f"{size_field_name}[{feature_name!r}] must be [height, width]."
                    )

    def thermal_twogrey_feature_names(self, feature_name: str) -> tuple[str, str]:
        return thermal_twogrey_feature_names(
            feature_name,
            cold_suffix=self.thermal_twogrey_cold_suffix,
            hot_suffix=self.thermal_twogrey_hot_suffix,
        )

    def thermal_twogrey_feature_map(self) -> dict[str, tuple[str, str]]:
        return {
            feature_name: self.thermal_twogrey_feature_names(feature_name)
            for feature_name in self.thermal_image_features
        }

    def thermal_model_image_features(self) -> list[str]:
        if self.thermal_input_type not in TWO_STREAM_THERMAL_INPUT_TYPES:
            return list(self.thermal_image_features)
        return [
            derived_feature
            for feature_name in self.thermal_image_features
            for derived_feature in self.thermal_twogrey_feature_names(feature_name)
        ]

    def reconcile_visual_features_for_consistency(
        self,
        provided_visuals: set[str],
        expected_visuals: set[str],
    ) -> tuple[set[str], set[str]]:
        if self.thermal_input_type not in TWO_STREAM_THERMAL_INPUT_TYPES:
            return provided_visuals, expected_visuals

        expected_visuals = set(expected_visuals)
        for feature_name, (cold_feature, hot_feature) in self.thermal_twogrey_feature_map().items():
            if feature_name not in provided_visuals:
                continue
            if cold_feature in expected_visuals or hot_feature in expected_visuals:
                expected_visuals.discard(cold_feature)
                expected_visuals.discard(hot_feature)
                expected_visuals.add(feature_name)
        return provided_visuals, expected_visuals

    @staticmethod
    def _feature_shape(feature) -> tuple[int, ...]:
        shape = getattr(feature, "shape", None)
        if shape is None and isinstance(feature, dict):
            shape = feature.get("shape")
        return tuple(int(dim) for dim in shape) if shape else ()

    def _three_channel_visual_feature(self, feature) -> PolicyFeature:
        shape = self._feature_shape(feature)
        if shape and shape[0] in {1, 3, 4}:
            output_shape = (3, *shape[1:])
        elif shape and shape[-1] in {1, 3, 4}:
            output_shape = (*shape[:-1], 3)
        else:
            output_shape = (3, *self.image_resolution)
        return PolicyFeature(type=FeatureType.VISUAL, shape=output_shape)

    def rewrite_twogrey_input_features(self, dataset_features: dict | None = None) -> None:
        if (
            self.thermal_input_type not in TWO_STREAM_THERMAL_INPUT_TYPES
            or self.input_features is None
        ):
            return

        self._validate_thermal_grey_config()
        self._validate_thermal_twogrey_config()
        if self.thermal_input_type in BLACK_IMPORTANCE_THERMAL_INPUT_TYPES:
            self._validate_thermal_black_config()
        if self.thermal_input_type in MATCHED_THERMAL_INPUT_TYPES:
            self._validate_thermal_matching_config()
        if self.thermal_input_type in FIXED_MATCHED_THERMAL_INPUT_TYPES:
            self._validate_thermal_fixed_matching_config()
        twogrey_map = self.thermal_twogrey_feature_map()
        source_features = set(twogrey_map)
        derived_features = {
            derived_feature
            for derived_pair in twogrey_map.values()
            for derived_feature in derived_pair
        }
        new_input_features: dict[str, PolicyFeature] = {}
        inserted_sources: set[str] = set()

        for feature_name, feature in self.input_features.items():
            if feature_name in source_features:
                cold_feature, hot_feature = twogrey_map[feature_name]
                visual_feature = self._three_channel_visual_feature(feature)
                new_input_features[cold_feature] = deepcopy(visual_feature)
                new_input_features[hot_feature] = deepcopy(visual_feature)
                inserted_sources.add(feature_name)
                continue
            if feature_name in derived_features:
                new_input_features[feature_name] = self._three_channel_visual_feature(feature)
                continue
            new_input_features[feature_name] = feature

        dataset_features = dataset_features or {}
        for feature_name, (cold_feature, hot_feature) in twogrey_map.items():
            if feature_name in inserted_sources:
                continue
            source_feature = self.input_features.get(feature_name) or dataset_features.get(feature_name)
            if source_feature is None and cold_feature in self.input_features:
                source_feature = self.input_features[cold_feature]
            if source_feature is None and hot_feature in self.input_features:
                source_feature = self.input_features[hot_feature]
            if source_feature is None:
                continue
            visual_feature = self._three_channel_visual_feature(source_feature)
            new_input_features.setdefault(cold_feature, deepcopy(visual_feature))
            new_input_features.setdefault(hot_feature, deepcopy(visual_feature))

        self.input_features = new_input_features

    def _validate_anythermal_config(self) -> None:
        if not str(self.thermal_anythermal_model_type or "").strip():
            raise ValueError("thermal_anythermal_model_type must be non-empty")
        if self.thermal_anythermal_input_range not in {"minus_one_to_one", "zero_to_one", "auto"}:
            raise ValueError(
                "thermal_anythermal_input_range must be one of: minus_one_to_one, zero_to_one, auto"
            )

    def _validate_resthermal_config(self) -> None:
        if self.thermal_resthermal_adapter_dim <= 0:
            raise ValueError("thermal_resthermal_adapter_dim must be positive")
        if self.thermal_resthermal_decoder_hidden_dim <= 0:
            raise ValueError("thermal_resthermal_decoder_hidden_dim must be positive")
        if self.thermal_resthermal_num_heads <= 0:
            raise ValueError("thermal_resthermal_num_heads must be positive")
        if self.thermal_resthermal_adapter_dim % self.thermal_resthermal_num_heads != 0:
            raise ValueError(
                "thermal_resthermal_num_heads must divide thermal_resthermal_adapter_dim "
                f"({self.thermal_resthermal_adapter_dim})"
            )

    def _validate_residual_thermal_vit_config(self) -> None:
        self._validate_shared_projector_thermal_vit_config("ResViT")
        if self.thermal_fusion_head_rgb_feature in set(self.thermal_image_features):
            raise ValueError(
                "thermal_input_type='ResViT' requires distinct head RGB and thermal features, "
                f"got {self.thermal_fusion_head_rgb_feature!r}."
            )

    def _validate_gate_thermal_vit_config(self) -> None:
        self._validate_shared_projector_thermal_vit_config("GateViT")
        self._validate_thermal_grey_config()
        self._validate_thermal_twogrey_config()
        if self.thermal_input_type not in TWO_STREAM_THERMAL_INPUT_TYPES:
            raise ValueError(
                "thermal_encoder_channel='GateViT' requires thermal_input_type='twogrey', "
                "'twoblack', 'twomatchingblack', or 'twofixmatchingblack' "
                "so the model receives cold/hot thermal views."
            )
        if self.thermal_twogrey_gatevit_num_heads <= 0:
            raise ValueError("thermal_twogrey_gatevit_num_heads must be positive")
        image_embed_dim = 2048
        if image_embed_dim % self.thermal_twogrey_gatevit_num_heads != 0:
            raise ValueError(
                "thermal_twogrey_gatevit_num_heads must divide the PI05 projected image "
                f"dimension ({image_embed_dim})"
            )
        if (
            self.thermal_twogrey_gatevit_hidden_dim is not None
            and self.thermal_twogrey_gatevit_hidden_dim <= 0
        ):
            raise ValueError("thermal_twogrey_gatevit_hidden_dim must be positive")
        self.thermal_twogrey_gatevit_temperature = float(self.thermal_twogrey_gatevit_temperature)
        if self.thermal_twogrey_gatevit_temperature <= 0:
            raise ValueError("thermal_twogrey_gatevit_temperature must be positive")
        self.thermal_twogrey_gatevit_gate_init_std = float(
            self.thermal_twogrey_gatevit_gate_init_std
        )
        if self.thermal_twogrey_gatevit_gate_init_std < 0:
            raise ValueError("thermal_twogrey_gatevit_gate_init_std must be non-negative")

    def _validate_gate_action_thermal_vit_config(self) -> None:
        self._validate_shared_projector_thermal_vit_config("GateActionViT")
        self._validate_thermal_grey_config()
        self._validate_thermal_twogrey_config()
        if self.thermal_input_type not in TWO_STREAM_THERMAL_INPUT_TYPES:
            raise ValueError(
                "thermal_encoder_channel='GateActionViT' requires thermal_input_type='twogrey', "
                "'twoblack', 'twomatchingblack', or 'twofixmatchingblack' "
                "so the model receives separate cold/hot thermal views."
            )
        if self.thermal_twogrey_gateactionvit_num_heads <= 0:
            raise ValueError("thermal_twogrey_gateactionvit_num_heads must be positive")
        image_embed_dim = 2048
        if image_embed_dim % self.thermal_twogrey_gateactionvit_num_heads != 0:
            raise ValueError(
                "thermal_twogrey_gateactionvit_num_heads must divide the PI05 projected image "
                f"dimension ({image_embed_dim})"
            )
        if (
            self.thermal_twogrey_gateactionvit_hidden_dim is not None
            and self.thermal_twogrey_gateactionvit_hidden_dim <= 0
        ):
            raise ValueError("thermal_twogrey_gateactionvit_hidden_dim must be positive")
        self.thermal_twogrey_gateactionvit_temperature = float(
            self.thermal_twogrey_gateactionvit_temperature
        )
        if self.thermal_twogrey_gateactionvit_temperature <= 0:
            raise ValueError("thermal_twogrey_gateactionvit_temperature must be positive")
        self.thermal_twogrey_gateactionvit_gate_init_std = float(
            self.thermal_twogrey_gateactionvit_gate_init_std
        )
        if self.thermal_twogrey_gateactionvit_gate_init_std < 0:
            raise ValueError("thermal_twogrey_gateactionvit_gate_init_std must be non-negative")
        self.thermal_twogrey_gateactionvit_bias_epsilon = float(
            self.thermal_twogrey_gateactionvit_bias_epsilon
        )
        if not 0 <= self.thermal_twogrey_gateactionvit_bias_epsilon < 1:
            raise ValueError(
                "thermal_twogrey_gateactionvit_bias_epsilon must be in [0, 1)"
            )
        self.thermal_twogrey_gateactionvit_bias_max_strength = float(
            self.thermal_twogrey_gateactionvit_bias_max_strength
        )
        if (
            not math.isfinite(self.thermal_twogrey_gateactionvit_bias_max_strength)
            or self.thermal_twogrey_gateactionvit_bias_max_strength <= 0
        ):
            raise ValueError(
                "thermal_twogrey_gateactionvit_bias_max_strength must be finite and positive"
            )
        self.thermal_twogrey_gateactionvit_bias_init_strength = float(
            self.thermal_twogrey_gateactionvit_bias_init_strength
        )
        if not (
            0
            <= self.thermal_twogrey_gateactionvit_bias_init_strength
            <= self.thermal_twogrey_gateactionvit_bias_max_strength
        ):
            raise ValueError(
                "thermal_twogrey_gateactionvit_bias_init_strength must be between 0 and "
                "thermal_twogrey_gateactionvit_bias_max_strength"
            )

    def _validate_double_gate_thermal_vit_config(self) -> None:
        self._validate_shared_projector_thermal_vit_config("DoubleGateViT")
        self._validate_thermal_grey_config()
        self._validate_thermal_twogrey_config()
        if self.thermal_input_type not in TWO_STREAM_THERMAL_INPUT_TYPES:
            raise ValueError(
                "thermal_encoder_channel='DoubleGateViT' requires thermal_input_type='twogrey', "
                "'twoblack', 'twomatchingblack', or 'twofixmatchingblack' "
                "so the model receives separate cold/hot thermal views."
            )
        if self.thermal_fusion_head_rgb_feature in set(self.thermal_image_features):
            raise ValueError(
                "thermal_encoder_channel='DoubleGateViT' requires distinct head RGB and "
                f"thermal features, got {self.thermal_fusion_head_rgb_feature!r}."
            )
        if self.thermal_twogrey_doublegatevit_num_heads <= 0:
            raise ValueError("thermal_twogrey_doublegatevit_num_heads must be positive")
        image_embed_dim = 2048
        if image_embed_dim % self.thermal_twogrey_doublegatevit_num_heads != 0:
            raise ValueError(
                "thermal_twogrey_doublegatevit_num_heads must divide the PI05 projected "
                f"image dimension ({image_embed_dim})"
            )
        if (
            self.thermal_twogrey_doublegatevit_hidden_dim is not None
            and self.thermal_twogrey_doublegatevit_hidden_dim <= 0
        ):
            raise ValueError("thermal_twogrey_doublegatevit_hidden_dim must be positive")
        self.thermal_twogrey_doublegatevit_temperature = float(
            self.thermal_twogrey_doublegatevit_temperature
        )
        if self.thermal_twogrey_doublegatevit_temperature <= 0:
            raise ValueError("thermal_twogrey_doublegatevit_temperature must be positive")
        self.thermal_twogrey_doublegatevit_gate_init_std = float(
            self.thermal_twogrey_doublegatevit_gate_init_std
        )
        if self.thermal_twogrey_doublegatevit_gate_init_std < 0:
            raise ValueError("thermal_twogrey_doublegatevit_gate_init_std must be non-negative")

        self.thermal_twogrey_doublegatevit_rgb_gate_max_strength = float(
            self.thermal_twogrey_doublegatevit_rgb_gate_max_strength
        )
        if (
            not math.isfinite(self.thermal_twogrey_doublegatevit_rgb_gate_max_strength)
            or self.thermal_twogrey_doublegatevit_rgb_gate_max_strength <= 0
        ):
            raise ValueError(
                "thermal_twogrey_doublegatevit_rgb_gate_max_strength must be finite and positive"
            )
        self.thermal_twogrey_doublegatevit_rgb_gate_init_strength = float(
            self.thermal_twogrey_doublegatevit_rgb_gate_init_strength
        )
        if not (
            0
            <= self.thermal_twogrey_doublegatevit_rgb_gate_init_strength
            <= self.thermal_twogrey_doublegatevit_rgb_gate_max_strength
        ):
            raise ValueError(
                "thermal_twogrey_doublegatevit_rgb_gate_init_strength must be between 0 and "
                "thermal_twogrey_doublegatevit_rgb_gate_max_strength"
            )

    def _validate_patch_gate_thermal_vit_config(self) -> None:
        self._validate_shared_projector_thermal_vit_config("PatchGateViT")
        self._validate_thermal_grey_config()
        self._validate_thermal_twogrey_config()
        if self.thermal_input_type not in TWO_STREAM_THERMAL_INPUT_TYPES:
            raise ValueError(
                "thermal_encoder_channel='PatchGateViT' requires thermal_input_type='twogrey', "
                "'twoblack', 'twomatchingblack', or 'twofixmatchingblack' so patch evidence can follow the "
                "configured cold/hot polarity."
            )
        if len(self.thermal_image_features) != 1:
            raise ValueError(
                "thermal_encoder_channel='PatchGateViT' currently requires exactly one "
                "thermal_image_features source so every cold/hot stream is patch-gated; "
                f"got {self.thermal_image_features}."
            )
        if self.thermal_fusion_head_rgb_feature in set(self.thermal_image_features):
            raise ValueError(
                "thermal_encoder_channel='PatchGateViT' requires distinct head RGB and "
                f"thermal features, got {self.thermal_fusion_head_rgb_feature!r}."
            )
        gate_dim = int(self.thermal_twogrey_patchgatevit_gate_dim)
        num_heads = int(self.thermal_twogrey_patchgatevit_num_heads)
        hidden_dim = int(self.thermal_twogrey_patchgatevit_hidden_dim)
        if gate_dim <= 0:
            raise ValueError("thermal_twogrey_patchgatevit_gate_dim must be positive")
        if num_heads <= 0 or gate_dim % num_heads != 0:
            raise ValueError(
                "thermal_twogrey_patchgatevit_num_heads must be positive and divide "
                f"thermal_twogrey_patchgatevit_gate_dim ({gate_dim})"
            )
        if hidden_dim <= 0:
            raise ValueError("thermal_twogrey_patchgatevit_hidden_dim must be positive")
        self.thermal_twogrey_patchgatevit_gate_dim = gate_dim
        self.thermal_twogrey_patchgatevit_num_heads = num_heads
        self.thermal_twogrey_patchgatevit_hidden_dim = hidden_dim
        self.thermal_twogrey_patchgatevit_temperature = float(
            self.thermal_twogrey_patchgatevit_temperature
        )
        if (
            not math.isfinite(self.thermal_twogrey_patchgatevit_temperature)
            or self.thermal_twogrey_patchgatevit_temperature <= 0
        ):
            raise ValueError(
                "thermal_twogrey_patchgatevit_temperature must be finite and positive"
            )
        self.thermal_twogrey_patchgatevit_gate_init_std = float(
            self.thermal_twogrey_patchgatevit_gate_init_std
        )
        if (
            not math.isfinite(self.thermal_twogrey_patchgatevit_gate_init_std)
            or self.thermal_twogrey_patchgatevit_gate_init_std < 0
        ):
            raise ValueError(
                "thermal_twogrey_patchgatevit_gate_init_std must be finite and non-negative"
            )
        self.thermal_twogrey_patchgatevit_evidence_strength = float(
            self.thermal_twogrey_patchgatevit_evidence_strength
        )
        if (
            not math.isfinite(self.thermal_twogrey_patchgatevit_evidence_strength)
            or self.thermal_twogrey_patchgatevit_evidence_strength < 0
        ):
            raise ValueError(
                "thermal_twogrey_patchgatevit_evidence_strength must be finite and non-negative"
            )
        self.thermal_twogrey_patchgatevit_evidence_floor = float(
            self.thermal_twogrey_patchgatevit_evidence_floor
        )
        if (
            not math.isfinite(self.thermal_twogrey_patchgatevit_evidence_floor)
            or not 0 < self.thermal_twogrey_patchgatevit_evidence_floor <= 1
        ):
            raise ValueError(
                "thermal_twogrey_patchgatevit_evidence_floor must be finite and in (0, 1]"
            )
        if len(self.thermal_twogrey_smallpatchgatevit_gate_grid) != 2:
            raise ValueError(
                "thermal_twogrey_smallpatchgatevit_gate_grid must be (rows, cols), "
                f"got {self.thermal_twogrey_smallpatchgatevit_gate_grid!r}."
            )
        gate_rows, gate_cols = (
            int(value) for value in self.thermal_twogrey_smallpatchgatevit_gate_grid
        )
        if gate_rows <= 0 or gate_cols <= 0:
            raise ValueError(
                "thermal_twogrey_smallpatchgatevit_gate_grid entries must be positive, "
                f"got {self.thermal_twogrey_smallpatchgatevit_gate_grid!r}."
            )
        self.thermal_twogrey_smallpatchgatevit_gate_grid = (gate_rows, gate_cols)
        if (
            self.thermal_encoder_channel == "hardsmallpatchgatevit"
            and self.thermal_twogrey_smallpatchgatevit_gate_grid != (4, 4)
        ):
            raise ValueError(
                "thermal_encoder_channel='HardSmallPatchGateViT' requires "
                "thermal_twogrey_smallpatchgatevit_gate_grid=(4, 4)"
            )

    def _validate_gate_thermal_vit_mix_config(self) -> None:
        self._validate_shared_projector_thermal_vit_config("GateViTMix")
        self._validate_thermal_grey_config()
        self._validate_thermal_twogrey_config()
        if self.thermal_input_type not in TWO_STREAM_THERMAL_INPUT_TYPES:
            raise ValueError(
                "thermal_encoder_channel='GateViTMix' requires thermal_input_type='twogrey', "
                "'twoblack', 'twomatchingblack', or 'twofixmatchingblack' "
                "so the model receives cold/hot thermal views."
            )
        if self.thermal_fusion_head_rgb_feature in set(self.thermal_image_features):
            raise ValueError(
                "thermal_encoder_channel='GateViTMix' requires distinct head RGB and thermal "
                f"features, got {self.thermal_fusion_head_rgb_feature!r}."
            )
        if self.thermal_twogrey_gatevitmix_num_heads <= 0:
            raise ValueError("thermal_twogrey_gatevitmix_num_heads must be positive")
        image_embed_dim = 2048
        if image_embed_dim % self.thermal_twogrey_gatevitmix_num_heads != 0:
            raise ValueError(
                "thermal_twogrey_gatevitmix_num_heads must divide the PI05 projected image "
                f"dimension ({image_embed_dim})"
            )
        if (
            self.thermal_twogrey_gatevitmix_hidden_dim is not None
            and self.thermal_twogrey_gatevitmix_hidden_dim <= 0
        ):
            raise ValueError("thermal_twogrey_gatevitmix_hidden_dim must be positive")
        if self.thermal_twogrey_gatevitmix_mix_dim <= 0:
            raise ValueError("thermal_twogrey_gatevitmix_mix_dim must be positive")
        if (
            self.thermal_twogrey_gatevitmix_mix_dim
            % self.thermal_twogrey_gatevitmix_num_heads
            != 0
        ):
            raise ValueError(
                "thermal_twogrey_gatevitmix_num_heads must divide "
                f"thermal_twogrey_gatevitmix_mix_dim ({self.thermal_twogrey_gatevitmix_mix_dim})"
            )
        self.thermal_twogrey_gatevitmix_temperature = float(
            self.thermal_twogrey_gatevitmix_temperature
        )
        if self.thermal_twogrey_gatevitmix_temperature <= 0:
            raise ValueError("thermal_twogrey_gatevitmix_temperature must be positive")
        self.thermal_twogrey_gatevitmix_gate_init_std = float(
            self.thermal_twogrey_gatevitmix_gate_init_std
        )
        if self.thermal_twogrey_gatevitmix_gate_init_std < 0:
            raise ValueError("thermal_twogrey_gatevitmix_gate_init_std must be non-negative")
        self.thermal_twogrey_gatevitmix_context_scale = float(
            self.thermal_twogrey_gatevitmix_context_scale
        )
        if (
            not math.isfinite(self.thermal_twogrey_gatevitmix_context_scale)
            or self.thermal_twogrey_gatevitmix_context_scale < 0
        ):
            raise ValueError(
                "thermal_twogrey_gatevitmix_context_scale must be finite and non-negative"
            )

    def _validate_gate_residual_thermal_vit_config(self) -> None:
        self._validate_shared_projector_thermal_vit_config("GateResViT/GateResandViT")
        self._validate_thermal_grey_config()
        self._validate_thermal_twogrey_config()
        if self.thermal_input_type not in TWO_STREAM_THERMAL_INPUT_TYPES:
            raise ValueError(
                "thermal_encoder_channel='GateResViT'/'GateResandViT' requires thermal_input_type='twogrey', "
                "'twoblack', 'twomatchingblack', or 'twofixmatchingblack' "
                "so the model receives cold/hot thermal views."
            )
        if self.thermal_fusion_head_rgb_feature in set(self.thermal_image_features):
            raise ValueError(
                "thermal_encoder_channel='GateResViT'/'GateResandViT' requires distinct head RGB and thermal "
                f"features, got {self.thermal_fusion_head_rgb_feature!r}."
            )
        if self.thermal_twogrey_gateresvit_num_heads <= 0:
            raise ValueError("thermal_twogrey_gateresvit_num_heads must be positive")
        image_embed_dim = 2048
        if image_embed_dim % self.thermal_twogrey_gateresvit_num_heads != 0:
            raise ValueError(
                "thermal_twogrey_gateresvit_num_heads must divide the PI05 projected image "
                f"dimension ({image_embed_dim})"
            )
        if (
            self.thermal_twogrey_gateresvit_hidden_dim is not None
            and self.thermal_twogrey_gateresvit_hidden_dim <= 0
        ):
            raise ValueError("thermal_twogrey_gateresvit_hidden_dim must be positive")
        self.thermal_twogrey_gateresvit_temperature = float(
            self.thermal_twogrey_gateresvit_temperature
        )
        if self.thermal_twogrey_gateresvit_temperature <= 0:
            raise ValueError("thermal_twogrey_gateresvit_temperature must be positive")
        self.thermal_twogrey_gateresvit_gate_init_std = float(
            self.thermal_twogrey_gateresvit_gate_init_std
        )
        if self.thermal_twogrey_gateresvit_gate_init_std < 0:
            raise ValueError("thermal_twogrey_gateresvit_gate_init_std must be non-negative")

    def _validate_gate_one_residual_thermal_vit_config(self) -> None:
        self._validate_shared_projector_thermal_vit_config("GateOneResViT")
        self._validate_thermal_grey_config()
        if self.thermal_input_type != "grey":
            raise ValueError(
                "thermal_encoder_channel='GateOneResViT' requires thermal_input_type='grey' "
                "so the model receives one background-normalized thermal view."
            )
        if self.thermal_fusion_head_rgb_feature in set(self.thermal_image_features):
            raise ValueError(
                "thermal_encoder_channel='GateOneResViT' requires distinct head RGB and thermal "
                f"features, got {self.thermal_fusion_head_rgb_feature!r}."
            )
        if self.thermal_grey_gateoneresvit_num_heads <= 0:
            raise ValueError("thermal_grey_gateoneresvit_num_heads must be positive")
        image_embed_dim = 2048
        if image_embed_dim % self.thermal_grey_gateoneresvit_num_heads != 0:
            raise ValueError(
                "thermal_grey_gateoneresvit_num_heads must divide the PI05 projected image "
                f"dimension ({image_embed_dim})"
            )
        if (
            self.thermal_grey_gateoneresvit_hidden_dim is not None
            and self.thermal_grey_gateoneresvit_hidden_dim <= 0
        ):
            raise ValueError("thermal_grey_gateoneresvit_hidden_dim must be positive")
        self.thermal_grey_gateoneresvit_temperature = float(
            self.thermal_grey_gateoneresvit_temperature
        )
        if self.thermal_grey_gateoneresvit_temperature <= 0:
            raise ValueError("thermal_grey_gateoneresvit_temperature must be positive")
        self.thermal_grey_gateoneresvit_gate_init_std = float(
            self.thermal_grey_gateoneresvit_gate_init_std
        )
        if self.thermal_grey_gateoneresvit_gate_init_std < 0:
            raise ValueError("thermal_grey_gateoneresvit_gate_init_std must be non-negative")

    def _validate_gate_residual_thermal_vit3_config(self) -> None:
        self._validate_shared_projector_thermal_vit_config("GateResViT3")
        self._validate_thermal_grey_config()
        self._validate_thermal_twogrey_config()
        if self.thermal_input_type not in TWO_STREAM_THERMAL_INPUT_TYPES:
            raise ValueError(
                "thermal_encoder_channel='GateResViT3' requires thermal_input_type='twogrey', "
                "'twoblack', 'twomatchingblack', or 'twofixmatchingblack' "
                "so the model receives cold/hot thermal views."
            )
        if self.thermal_fusion_head_rgb_feature in set(self.thermal_image_features):
            raise ValueError(
                "thermal_encoder_channel='GateResViT3' requires distinct head RGB and thermal "
                f"features, got {self.thermal_fusion_head_rgb_feature!r}."
            )
        if self.thermal_twogrey_gateresvit3_num_heads <= 0:
            raise ValueError("thermal_twogrey_gateresvit3_num_heads must be positive")
        image_embed_dim = 2048
        if image_embed_dim % self.thermal_twogrey_gateresvit3_num_heads != 0:
            raise ValueError(
                "thermal_twogrey_gateresvit3_num_heads must divide the PI05 projected image "
                f"dimension ({image_embed_dim})"
            )
        if (
            self.thermal_twogrey_gateresvit3_hidden_dim is not None
            and self.thermal_twogrey_gateresvit3_hidden_dim <= 0
        ):
            raise ValueError("thermal_twogrey_gateresvit3_hidden_dim must be positive")
        if self.thermal_twogrey_gateresvit3_stat_hidden_dim <= 0:
            raise ValueError("thermal_twogrey_gateresvit3_stat_hidden_dim must be positive")
        if (
            self.thermal_twogrey_gateresvit3_residual_hidden_dim is not None
            and self.thermal_twogrey_gateresvit3_residual_hidden_dim <= 0
        ):
            raise ValueError("thermal_twogrey_gateresvit3_residual_hidden_dim must be positive")
        self.thermal_twogrey_gateresvit3_temperature = float(
            self.thermal_twogrey_gateresvit3_temperature
        )
        if self.thermal_twogrey_gateresvit3_temperature <= 0:
            raise ValueError("thermal_twogrey_gateresvit3_temperature must be positive")
        self.thermal_twogrey_gateresvit3_gate_init_std = float(
            self.thermal_twogrey_gateresvit3_gate_init_std
        )
        if self.thermal_twogrey_gateresvit3_gate_init_std < 0:
            raise ValueError("thermal_twogrey_gateresvit3_gate_init_std must be non-negative")
        self.thermal_twogrey_gateresvit3_confidence_bias_init = float(
            self.thermal_twogrey_gateresvit3_confidence_bias_init
        )
        if not math.isfinite(self.thermal_twogrey_gateresvit3_confidence_bias_init):
            raise ValueError("thermal_twogrey_gateresvit3_confidence_bias_init must be finite")
        self.thermal_twogrey_gateresvit3_evidence_gain = float(
            self.thermal_twogrey_gateresvit3_evidence_gain
        )
        if self.thermal_twogrey_gateresvit3_evidence_gain <= 0:
            raise ValueError("thermal_twogrey_gateresvit3_evidence_gain must be positive")
        self.thermal_twogrey_gateresvit3_evidence_threshold = float(
            self.thermal_twogrey_gateresvit3_evidence_threshold
        )
        if self.thermal_twogrey_gateresvit3_evidence_threshold < 0:
            raise ValueError("thermal_twogrey_gateresvit3_evidence_threshold must be non-negative")

    def _validate_resvit_attention_config(self) -> None:
        if self.thermal_fusion_head_rgb_feature in set(self.thermal_image_features):
            raise ValueError(
                "thermal_input_type='ResViTAttention' requires distinct head RGB and thermal features, "
                f"got {self.thermal_fusion_head_rgb_feature!r}."
            )
        if self.thermal_resvit_attention_num_heads <= 0:
            raise ValueError("thermal_resvit_attention_num_heads must be positive")
        if (
            self.thermal_resvit_attention_merge_hidden_dim is not None
            and self.thermal_resvit_attention_merge_hidden_dim <= 0
        ):
            raise ValueError("thermal_resvit_attention_merge_hidden_dim must be positive")
        if (
            self.thermal_resvit_attention_residual_hidden_dim is not None
            and self.thermal_resvit_attention_residual_hidden_dim <= 0
        ):
            raise ValueError("thermal_resvit_attention_residual_hidden_dim must be positive")

    def _validate_shared_projector_thermal_vit_config(self, mode_name: str = "ViT") -> None:
        if self.rgb_input_type != "vit":
            raise ValueError(
                f"thermal_input_type='{mode_name}' uses the raw PaliGemma/SigLIP ViT path and "
                "shared RGB projector, so it requires rgb_input_type='vit'."
            )

    def apply_rgb_input_type(self, value: str | None) -> None:
        """Apply the top-level train/eval ``--rgb_input_type`` alias.

        Existing values (``vit``/``dinov2``) continue to select the RGB encoder.
        ``withthreeblack`` selects the RGB color preprocessing and its matching
        GateTwoResViT fusion without overwriting it with a second encoder name.
        """
        raw_value = str(value or "vit").strip().strip("\"'“”‘’").lower()
        if raw_value in {
            "withthreeblack",
            "with_three_black",
            "with-three-black",
            "threeblack",
        }:
            if self.uses_dinov2_rgb_encoder:
                raise ValueError(
                    "--rgb_input_type=withthreeblack cannot be combined with "
                    "--policy.rgb_input_type=dinov2."
                )
            self.rgb_threeblack_input_type = "withthreeblack"
            self.rgb_input_type = "gatetworesvit"
        else:
            self.rgb_input_type = self._normalize_rgb_input_type(value)
            self.rgb_threeblack_input_type = (
                "withthreeblack" if self.rgb_input_type == "gatetworesvit" else "false"
            )
        if self.uses_dinov2_rgb_encoder:
            self._validate_rgb_dinov2_config()
        if self.uses_gate_two_residual_rgb_vit:
            self._validate_gate_two_residual_rgb_vit_config()
        if self.uses_shared_projector_thermal_vit:
            self._validate_shared_projector_thermal_vit_config("ViT")
        if self.uses_residual_thermal_vit:
            self._validate_residual_thermal_vit_config()
        if self.uses_gate_residual_thermal_vit:
            self._validate_gate_residual_thermal_vit_config()
        if self.uses_gate_res_and_thermal_vit:
            self._validate_gate_residual_thermal_vit_config()
        if self.uses_gate_two_residual_thermal_vit:
            self._validate_gate_residual_thermal_vit_config()
        if self.uses_gate_one_residual_thermal_vit:
            self._validate_gate_one_residual_thermal_vit_config()
        if self.uses_gate_residual_thermal_vit3:
            self._validate_gate_residual_thermal_vit3_config()
        if self.uses_small_patch_gate_two_res_thermal_vit:
            self._validate_patch_gate_thermal_vit_config()
        if self.uses_patch_single_gate_residual_thermal_vit:
            self._validate_patch_gate_thermal_vit_config()
        if self.uses_small_patch_gate_residual_thermal_vit:
            self._validate_patch_gate_thermal_vit_config()
        if self.uses_resvit_attention_encoder:
            self._validate_resvit_attention_config()

    def _validate_rgb_dinov2_config(self) -> None:
        if not str(self.rgb_dinov2_model_type or "").strip():
            raise ValueError("rgb_dinov2_model_type must be non-empty")
        if self.rgb_dinov2_input_range not in {"minus_one_to_one", "zero_to_one", "auto"}:
            raise ValueError("rgb_dinov2_input_range must be one of: minus_one_to_one, zero_to_one, auto")

    def _validate_gate_two_residual_rgb_vit_config(self) -> None:
        if self.rgb_threeblack_input_type != "withthreeblack":
            raise ValueError(
                "rgb_input_type='GateTwoResViT' requires "
                "rgb_threeblack_input_type='withthreeblack'."
            )
        if self.thermal_encoder_channel is not False:
            raise ValueError(
                "rgb_input_type='GateTwoResViT' currently requires the separate thermal "
                "encoder/fusion path to be disabled. Set thermal_encoder_channel=false."
            )
        if self.paligemma_variant != "gemma_2b":
            raise ValueError(
                "rgb_input_type='GateTwoResViT' requires paligemma_variant='gemma_2b' so "
                "projected RGB and language tokens both use embedding dimension 2048."
            )
        if not str(self.rgb_threeblack_head_feature or "").strip():
            raise ValueError("rgb_threeblack_head_feature must be a non-empty feature name")
        if self.rgb_threeblack_num_heads <= 0:
            raise ValueError("rgb_threeblack_num_heads must be positive")
        if self.rgb_threeblack_hidden_dim is not None and self.rgb_threeblack_hidden_dim <= 0:
            raise ValueError("rgb_threeblack_hidden_dim must be positive")
        self.rgb_threeblack_temperature = float(self.rgb_threeblack_temperature)
        if not math.isfinite(self.rgb_threeblack_temperature) or self.rgb_threeblack_temperature <= 0:
            raise ValueError("rgb_threeblack_temperature must be finite and positive")
        self.rgb_threeblack_gate_init_std = float(self.rgb_threeblack_gate_init_std)
        if not math.isfinite(self.rgb_threeblack_gate_init_std) or self.rgb_threeblack_gate_init_std < 0:
            raise ValueError("rgb_threeblack_gate_init_std must be finite and non-negative")
        for field_name in (
            "rgb_threeblack_red_alpha",
            "rgb_threeblack_green_alpha",
            "rgb_threeblack_blue_alpha",
        ):
            field_value = float(getattr(self, field_name))
            if not math.isfinite(field_value):
                raise ValueError(f"{field_name} must be finite")
            setattr(self, field_name, field_value)

    @property
    def uses_dedicated_thermal_vit(self) -> bool:
        return self.thermal_encoder_channel == "vit"

    @property
    def uses_shared_projector_thermal_vit(self) -> bool:
        return self.thermal_encoder_channel == "shared_projector_vit"

    @property
    def uses_residual_thermal_vit(self) -> bool:
        return self.thermal_encoder_channel == "resvit"

    @property
    def uses_gate_thermal_vit(self) -> bool:
        return self.thermal_encoder_channel == "gatevit"

    @property
    def uses_gate_action_thermal_vit(self) -> bool:
        return self.thermal_encoder_channel == "gateactionvit"

    @property
    def uses_double_gate_thermal_vit(self) -> bool:
        return self.thermal_encoder_channel == "doublegatevit"

    @property
    def uses_double_gate_two_residual_thermal_vit(self) -> bool:
        return self.thermal_encoder_channel == "doublegatetworesvit"

    @property
    def uses_patch_gate_thermal_vit(self) -> bool:
        return self.thermal_encoder_channel in {
            "patchgatevit",
            "smallpatchgatevit",
            "hardsmallpatchgatevit",
        }

    @property
    def uses_patch_single_gate_residual_thermal_vit(self) -> bool:
        return self.thermal_encoder_channel == "patchsinglegateresvit"

    @property
    def uses_hard_small_patch_gate_thermal_vit(self) -> bool:
        return self.thermal_encoder_channel == "hardsmallpatchgatevit"

    @property
    def uses_small_patch_gate_two_res_thermal_vit(self) -> bool:
        return self.thermal_encoder_channel in {
            "smallpatchgatetworesvit",
            "smallpatchsinglegatetworesvit",
        }

    @property
    def uses_small_patch_single_gate_two_res_thermal_vit(self) -> bool:
        return self.thermal_encoder_channel == "smallpatchsinglegatetworesvit"

    @property
    def uses_small_patch_gate_residual_thermal_vit(self) -> bool:
        return self.thermal_encoder_channel in {
            "smallpatchgateresvit",
            "smallpatchsinglegateresvit",
        }

    @property
    def uses_small_patch_single_gate_residual_thermal_vit(self) -> bool:
        return self.thermal_encoder_channel == "smallpatchsinglegateresvit"

    @property
    def uses_gate_thermal_vit_mix(self) -> bool:
        return self.thermal_encoder_channel == "gatevitmix"

    @property
    def uses_gate_residual_thermal_vit(self) -> bool:
        return self.thermal_encoder_channel == "gateresvit"

    @property
    def uses_gate_res_and_thermal_vit(self) -> bool:
        return self.thermal_encoder_channel == "gateresandvit"

    @property
    def uses_gate_two_residual_thermal_vit(self) -> bool:
        return self.thermal_encoder_channel == "gatetworesvit"

    @property
    def uses_gate_one_residual_thermal_vit(self) -> bool:
        return self.thermal_encoder_channel == "gateoneresvit"

    @property
    def uses_gate_residual_thermal_vit3(self) -> bool:
        return self.thermal_encoder_channel == "gateresvit3"

    @property
    def uses_resvit_attention_encoder(self) -> bool:
        return self.thermal_encoder_channel == "resvitattention"

    @property
    def uses_thermal_vit_encoder(self) -> bool:
        return self.thermal_encoder_channel in {
            "vit",
            "shared_projector_vit",
            "resvit",
            "gatevit",
            "gateactionvit",
            "doublegatevit",
            "doublegatetworesvit",
            "patchgatevit",
            "patchsinglegateresvit",
            "smallpatchgatevit",
            "hardsmallpatchgatevit",
            "smallpatchgatetworesvit",
            "smallpatchsinglegatetworesvit",
            "smallpatchgateresvit",
            "smallpatchsinglegateresvit",
            "gatevitmix",
            "gateresvit",
            "gateresandvit",
            "gatetworesvit",
            "gateoneresvit",
            "gateresvit3",
            "resvitattention",
        }

    @property
    def uses_thermal_resnet18_encoder(self) -> bool:
        return self.thermal_encoder_channel == "resnet18"

    @property
    def uses_thermal_cvae_encoder(self) -> bool:
        return self.thermal_encoder_channel == "cvae"

    @property
    def uses_anythermal_encoder(self) -> bool:
        return self.thermal_encoder_channel == "anythermal"

    @property
    def uses_resthermal_encoder(self) -> bool:
        return self.thermal_encoder_channel == "resthermal"

    @property
    def uses_twogrey_thermal_input(self) -> bool:
        return self.thermal_input_type in TWO_STREAM_THERMAL_INPUT_TYPES

    @property
    def uses_dinov2_rgb_encoder(self) -> bool:
        return self.rgb_input_type == "dinov2"

    @property
    def uses_gate_two_residual_rgb_vit(self) -> bool:
        return self.rgb_input_type == "gatetworesvit"

    def __post_init__(self):
        super().__post_init__()
        if self.freeze_non_thermal_input and self.train_expert_only:
            raise ValueError(
                "freeze_non_thermal_input=true cannot be combined with "
                "train_expert_only=true because train_expert_only also freezes the thermal ViT"
            )
        if self.freeze_non_thermal_input and self.freeze_vision_encoder:
            raise ValueError(
                "freeze_non_thermal_input=true cannot be combined with "
                "freeze_vision_encoder=true because freeze_vision_encoder also freezes the thermal ViT"
            )
        self.rgb_input_type = self._resolve_rgb_input_type()
        self.thermal_input_type = self._normalize_thermal_input_type(self.thermal_input_type)
        self.thermal_encoder_channel = self._resolve_thermal_encoder_channel()
        if (
            self.thermal_encoder_channel
            in {
                "gatevit",
                "gateactionvit",
                "doublegatevit",
                "doublegatetworesvit",
                "patchgatevit",
                "patchsinglegateresvit",
                "smallpatchgatevit",
                "hardsmallpatchgatevit",
                "smallpatchgatetworesvit",
                "smallpatchsinglegatetworesvit",
                "smallpatchgateresvit",
                "smallpatchsinglegateresvit",
                "gatevitmix",
                "gateresvit",
                "gateresandvit",
                "gatetworesvit",
                "gateoneresvit",
                "gateresvit3",
            }
            and self.thermal_input_type == "false"
        ):
            self.thermal_input_type = (
                "grey" if self.thermal_encoder_channel == "gateoneresvit" else "twogrey"
            )
        if self.thermal_input_type in {
            "vit",
            "shared_projector_vit",
            "resvit",
            "resvitattention",
            "anythermal",
            "resthermal",
            "grey",
            "twogrey",
            "twoblack",
            "twomatchingblack",
            "twofixmatchingblack",
            "matching",
        }:
            self.mix_rgb_thermal = False
        if (
            self.thermal_input_type == "grey"
            or self.thermal_input_type in TWO_STREAM_THERMAL_INPUT_TYPES
        ):
            self._validate_thermal_grey_config()
        if self.thermal_input_type in TWO_STREAM_THERMAL_INPUT_TYPES:
            self._validate_thermal_twogrey_config()
        if self.thermal_input_type in BLACK_IMPORTANCE_THERMAL_INPUT_TYPES:
            self._validate_thermal_black_config()
        if self.thermal_input_type in MATCHED_THERMAL_INPUT_TYPES:
            self._validate_thermal_matching_config()
        if self.thermal_input_type in FIXED_MATCHED_THERMAL_INPUT_TYPES:
            self._validate_thermal_fixed_matching_config()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.thermal_encoder_hidden_dim <= 0:
            raise ValueError("thermal_encoder_hidden_dim must be positive")
        if len(self.thermal_encoder_token_grid) != 2 or any(
            dim <= 0 for dim in self.thermal_encoder_token_grid
        ):
            raise ValueError("thermal_encoder_token_grid must contain two positive integers")
        if self.thermal_encoder_dropout < 0:
            raise ValueError("thermal_encoder_dropout must be non-negative")
        if self.thermal_fuse_with_head_rgb and not self.uses_thermal_resnet18_encoder:
            raise ValueError("thermal_fuse_with_head_rgb requires thermal_encoder_channel='resnet18'")
        if self.uses_anythermal_encoder:
            self._validate_anythermal_config()
        if self.uses_resthermal_encoder:
            self._validate_anythermal_config()
            self._validate_resthermal_config()
        if self.uses_dinov2_rgb_encoder:
            self._validate_rgb_dinov2_config()
        if self.uses_gate_two_residual_rgb_vit:
            self._validate_gate_two_residual_rgb_vit_config()
        if self.uses_shared_projector_thermal_vit:
            self._validate_shared_projector_thermal_vit_config("ViT")
        if self.uses_residual_thermal_vit:
            self._validate_residual_thermal_vit_config()
        if self.uses_gate_thermal_vit:
            self._validate_gate_thermal_vit_config()
        if self.uses_gate_action_thermal_vit:
            self._validate_gate_action_thermal_vit_config()
        if self.uses_double_gate_thermal_vit:
            self._validate_double_gate_thermal_vit_config()
        if self.uses_double_gate_two_residual_thermal_vit:
            self._validate_double_gate_thermal_vit_config()
        if self.uses_patch_gate_thermal_vit:
            self._validate_patch_gate_thermal_vit_config()
        if self.uses_small_patch_gate_two_res_thermal_vit:
            self._validate_patch_gate_thermal_vit_config()
        if self.uses_patch_single_gate_residual_thermal_vit:
            self._validate_patch_gate_thermal_vit_config()
        if self.uses_small_patch_gate_residual_thermal_vit:
            self._validate_patch_gate_thermal_vit_config()
        if self.uses_gate_thermal_vit_mix:
            self._validate_gate_thermal_vit_mix_config()
        if self.uses_gate_residual_thermal_vit:
            self._validate_gate_residual_thermal_vit_config()
        if self.uses_gate_res_and_thermal_vit:
            self._validate_gate_residual_thermal_vit_config()
        if self.uses_gate_two_residual_thermal_vit:
            self._validate_gate_residual_thermal_vit_config()
        if self.uses_gate_one_residual_thermal_vit:
            self._validate_gate_one_residual_thermal_vit_config()
        if self.uses_gate_residual_thermal_vit3:
            self._validate_gate_residual_thermal_vit3_config()
        if self.uses_resvit_attention_encoder:
            self._validate_resvit_attention_config()
        if self.uses_thermal_cvae_encoder:
            if self.thermal_cvae_input_channels <= 0:
                raise ValueError("thermal_cvae_input_channels must be positive")
            if self.thermal_cvae_condition_channels <= 0:
                raise ValueError("thermal_cvae_condition_channels must be positive")
            if self.thermal_cvae_target_channels <= 0:
                raise ValueError("thermal_cvae_target_channels must be positive")
            if self.thermal_cvae_latent_dim <= 0:
                raise ValueError("thermal_cvae_latent_dim must be positive")
            if not self.thermal_cvae_hidden_dims or any(dim <= 0 for dim in self.thermal_cvae_hidden_dims):
                raise ValueError("thermal_cvae_hidden_dims must contain positive integers")
            if self.thermal_cvae_image_size <= 0:
                raise ValueError("thermal_cvae_image_size must be positive")
            if self.thermal_cvae_token_grid_size <= 0:
                raise ValueError("thermal_cvae_token_grid_size must be positive")
            if self.thermal_cvae_token_scale_init <= 0:
                raise ValueError("thermal_cvae_token_scale_init must be positive")
            if (
                self.thermal_cvae_pretrained_path is not None
                and self.thermal_cvae_checkpoint_path is not None
                and self.thermal_cvae_pretrained_path != self.thermal_cvae_checkpoint_path
            ):
                raise ValueError(
                    "thermal_cvae_pretrained_path and deprecated thermal_cvae_checkpoint_path "
                    "refer to different files. Please set only one of them."
                )
        if self.thermal_fusion_num_heads <= 0:
            raise ValueError("thermal_fusion_num_heads must be positive")
        if self.rgb_thermal_mix_rgb_feature == self.rgb_thermal_mix_thermal_feature:
            raise ValueError("RGB and thermal mix features must be different")
        if self.rgb_thermal_mix_source not in ["auto", "precomputed", "online"]:
            raise ValueError("rgb_thermal_mix_source must be one of: auto, precomputed, online")
        if self.rgb_thermal_mix_precomputed_feature in {
            self.rgb_thermal_mix_rgb_feature,
            self.rgb_thermal_mix_thermal_feature,
        }:
            raise ValueError(
                "rgb_thermal_mix_precomputed_feature must be different from the RGB and thermal features"
            )
        if self.rgb_thermal_mix_thermal_width <= 0:
            raise ValueError("rgb_thermal_mix_thermal_width must be positive")
        if not 0.0 <= self.rgb_thermal_mix_thermal_weight <= 1.0:
            raise ValueError("rgb_thermal_mix_thermal_weight must be in [0, 1]")
        if self.rgb_thermal_mix_fill_color is not None and (
            len(self.rgb_thermal_mix_fill_color) != 3
            or any(channel < 0 or channel > 255 for channel in self.rgb_thermal_mix_fill_color)
        ):
            raise ValueError(
                "rgb_thermal_mix_fill_color must contain three integer RGB values in [0, 255]"
            )
        if any(
            len(color) != 3 or any(channel < 0 or channel > 255 for channel in color)
            for color in self.rgb_thermal_mix_episode_fill_colors
        ):
            raise ValueError(
                "Each rgb_thermal_mix_episode_fill_colors entry must contain "
                "three integer RGB values in [0, 255]"
            )
        if self.rgb_thermal_align_fusion and not self.uses_dedicated_thermal_vit:
            raise ValueError(
                "rgb_thermal_align_fusion currently requires thermal_encoder_channel='vit'"
            )
        if self.rgb_thermal_align_fusion and self.rgb_thermal_align_gt_feature in {
            self.rgb_thermal_mix_rgb_feature,
            self.rgb_thermal_mix_thermal_feature,
        }:
            raise ValueError(
                "rgb_thermal_align_gt_feature must be different from the head RGB and thermal features"
            )
        if self.rgb_thermal_align_loss_weight < 0:
            raise ValueError("rgb_thermal_align_loss_weight must be non-negative")

    def set_dataset_feature_metadata(self, dataset_features: dict) -> None:
        """Keep precomputed mixed videos out of the model-visible camera list.

        When ``mix_rgb_thermal`` uses a cached mixed video feature, the dataloader
        still has to decode that helper feature.  The model, however, should keep
        seeing the original thermal slot so checkpoints stay compatible with the
        online-mixing path.
        """
        if self.input_features is None:
            return

        uses_precomputed_mix = self.mix_rgb_thermal and self.rgb_thermal_mix_source != "online"
        uses_alignment_target = self.rgb_thermal_align_fusion
        uses_twogrey = self.thermal_input_type in TWO_STREAM_THERMAL_INPUT_TYPES
        if not uses_precomputed_mix and not uses_alignment_target and not uses_twogrey:
            return

        thermal_feature = self.rgb_thermal_mix_thermal_feature
        if uses_precomputed_mix:
            precomputed_feature = self.rgb_thermal_mix_precomputed_feature
            if precomputed_feature in self.input_features:
                if thermal_feature not in self.input_features:
                    self.input_features[thermal_feature] = deepcopy(self.input_features[precomputed_feature])
                del self.input_features[precomputed_feature]

        if uses_alignment_target:
            align_gt_feature = self.rgb_thermal_align_gt_feature
            if align_gt_feature in self.input_features:
                del self.input_features[align_gt_feature]

        if uses_twogrey:
            self.rewrite_twogrey_input_features(dataset_features)

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
