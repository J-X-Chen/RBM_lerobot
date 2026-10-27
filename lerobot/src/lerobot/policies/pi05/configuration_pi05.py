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
from dataclasses import dataclass, field

import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: torch.dtype | None = torch.float32

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # MEM short-horizon observation memory (https://arxiv.org/abs/2603.03596).
    # Historical image tokens are fused inside SigLIP and dropped before the
    # language backbone. Historical proprioceptive states become one continuous
    # backbone token per frame. Both paths are opt-in and independent.
    #
    # MEM pre-trains on six observations spaced one second apart. `memory_stride` is
    # counted in dataset frames, so the default matches that spacing only at 30 fps,
    # the usual LeRobot recording rate. Scale it with the dataset: a 10 fps dataset
    # such as `lerobot/robomme` needs `memory_stride=10` for the same one second.
    use_visual_memory: bool = False
    use_proprioceptive_memory: bool = False
    memory_frames: int = 6
    memory_stride: int = 30
    memory_temporal_attention_every: int = 4

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
    # Maximum clean action-prefix length sampled during training. Zero disables trained RTC.
    rtc_training_max_delay: int = 0

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0
    # Keep preprocessing compatible with existing RGB-only S1/G1 checkpoints.
    legacy_resize_padding: bool = False
    words_ignore: list[str] = field(default_factory=list)
    rgb_input_type: str | None = "vit"
    rgb_threeblack_input_type: str | bool | None = None
    rgb_threeblack_head_feature: str = "observation.images.cam_left_high"
    rgb_threeblack_num_heads: int = 8
    rgb_threeblack_hidden_dim: int | None = None
    rgb_threeblack_temperature: float = 1.0
    rgb_threeblack_gate_init_std: float = 0.02
    rgb_threeblack_red_alpha: float = 1.0
    rgb_threeblack_green_alpha: float = 1.0
    rgb_threeblack_blue_alpha: float = 1.0
    rgb_threeblack_dropout: float = 0.0

    rgb_dinov2_model_type: str = "dinov2_vitb14"
    rgb_dinov2_checkpoint_path: str | None = (
        "pretrained_checkpoints/backbone/Dinov2/dinov2_vitb14_pretrain.pth"
    )
    rgb_dinov2_repo_path: str | None = None
    rgb_dinov2_freeze_backbone: bool = True
    rgb_dinov2_input_range: str = "minus_one_to_one"

    tokenizer_max_length: int = 200  # see openpi `__post_init__`
    text_tokenizer_name: str = "google/paligemma-3b-pt-224"

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
            f"rgb_input_type must be 'vit', 'dinov2', 'GateTwoResViT', or 'withthreeblack'. Got {value!r}."
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
        raise ValueError(f"rgb_threeblack_input_type must be false or 'withthreeblack'. Got {value!r}.")

    def _resolve_rgb_input_type(self) -> str:
        encoder_mode = self._normalize_rgb_input_type(self.rgb_input_type)
        preprocessing_mode = self._normalize_rgb_threeblack_input_type(self.rgb_threeblack_input_type)
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

    def _validate_rgb_dinov2_config(self) -> None:
        if not str(self.rgb_dinov2_model_type or "").strip():
            raise ValueError("rgb_dinov2_model_type must be non-empty")
        if self.rgb_dinov2_input_range not in {"minus_one_to_one", "zero_to_one", "auto"}:
            raise ValueError("rgb_dinov2_input_range must be one of: minus_one_to_one, zero_to_one, auto")

    def _validate_gate_two_residual_rgb_vit_config(self) -> None:
        if self.rgb_threeblack_input_type != "withthreeblack":
            raise ValueError(
                "rgb_input_type='GateTwoResViT' requires rgb_threeblack_input_type='withthreeblack'."
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
                    "--rgb_input_type=withthreeblack cannot be combined with --policy.rgb_input_type=dinov2."
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

    @property
    def uses_dinov2_rgb_encoder(self) -> bool:
        return self.rgb_input_type == "dinov2"

    @property
    def uses_gate_two_residual_rgb_vit(self) -> bool:
        return self.rgb_input_type == "gatetworesvit"

    def __post_init__(self):
        super().__post_init__()
        self.rgb_input_type = self._resolve_rgb_input_type()
        self.apply_rgb_input_type(self.rgb_input_type)

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if not 0 <= self.rtc_training_max_delay < self.chunk_size:
            raise ValueError(
                "rtc_training_max_delay must satisfy "
                f"0 <= delay < chunk_size ({self.chunk_size}), got {self.rtc_training_max_delay}"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in [torch.bfloat16, torch.float32]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.memory_frames < 1:
            raise ValueError("memory_frames must be at least 1")
        if self.memory_stride < 1:
            raise ValueError("memory_stride must be at least 1")
        if self.memory_temporal_attention_every < 1:
            raise ValueError("memory_temporal_attention_every must be at least 1")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        if self.input_features is None or self.output_features is None:
            raise ValueError(
                "input_features and output_features must be resolved (e.g. from the dataset) "
                "before calling validate_features()"
            )
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
    def image_observation_delta_indices(self) -> list[int] | None:
        if not self.use_visual_memory:
            return None
        horizon = (self.memory_frames - 1) * self.memory_stride
        return list(range(-horizon, 1, self.memory_stride))

    @property
    def state_observation_delta_indices(self) -> list[int] | None:
        if not self.use_proprioceptive_memory:
            return None
        horizon = (self.memory_frames - 1) * self.memory_stride
        return list(range(-horizon, 1, self.memory_stride))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
