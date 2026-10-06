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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig
from .thermoact import (
    DEFAULT_THERMOACT_TEMPERATURE_MAX_C,
    DEFAULT_THERMOACT_TEMPERATURE_MIN_C,
    DEFAULT_THERMOACT_THERMAL_FEATURE,
    THERMOACT_CHANNEL,
)

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi0")
@dataclass
class PI0Config(PreTrainedConfig):
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
    num_inference_steps: int = 10  # Number of denoising steps during inference
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

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    # ThermoAct-style PI0: keep PI0's shared visual encoder, but replace an
    # external RGB camera slot with thermal pseudo-RGB generated from Celsius
    # frames using 20-35 C normalization and the INFERNO colormap.
    thermal_encoder_channel: str | bool | None = None
    thermal_image_features: list[str] = field(
        default_factory=lambda: [DEFAULT_THERMOACT_THERMAL_FEATURE]
    )
    thermoact_temperature_min_c: float = DEFAULT_THERMOACT_TEMPERATURE_MIN_C
    thermoact_temperature_max_c: float = DEFAULT_THERMOACT_TEMPERATURE_MAX_C

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
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

    # Optimizer settings: see openpi `AdamW``
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

    tokenizer_max_length: int = 48  # see openpi `__post_init__`

    @staticmethod
    def _normalize_thermal_encoder_channel(channel: str | bool | None) -> str | bool:
        if channel is None or channel is False:
            return False
        if isinstance(channel, bool):
            raise ValueError(
                "thermal_encoder_channel=true is ambiguous; use 'ThermoAct' or false."
            )
        normalized = channel.strip().lower()
        if normalized in {"", "false", "none", "off", "no"}:
            return False
        if normalized == THERMOACT_CHANNEL:
            return THERMOACT_CHANNEL
        raise ValueError(
            "PI0 thermal_encoder_channel must be false or 'ThermoAct'. "
            f"Got {channel!r}."
        )

    def __post_init__(self):
        super().__post_init__()
        self.thermal_encoder_channel = self._normalize_thermal_encoder_channel(
            self.thermal_encoder_channel
        )

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

        if self.uses_thermoact_thermal_input and not self.thermal_image_features:
            raise ValueError(
                "PI0 thermal_encoder_channel='ThermoAct' requires at least one "
                "thermal_image_features entry."
            )

        if (
            self.uses_thermoact_thermal_input
            and self.thermoact_temperature_max_c <= self.thermoact_temperature_min_c
        ):
            raise ValueError(
                "thermoact_temperature_max_c must be greater than thermoact_temperature_min_c, "
                f"got {self.thermoact_temperature_min_c} and {self.thermoact_temperature_max_c}."
            )

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
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

        if self.uses_thermoact_thermal_input:
            image_features = {
                key
                for key, feature in self.input_features.items()
                if feature.type is FeatureType.VISUAL
            }
            missing_thermal_features = [
                feature_name
                for feature_name in self.thermal_image_features
                if feature_name not in image_features
            ]
            if missing_thermal_features:
                raise ValueError(
                    "PI0 thermal_encoder_channel='ThermoAct' requires thermal_image_features to be "
                    f"visual inputs. Missing: {missing_thermal_features}. "
                    f"Available image features: {sorted(image_features)}."
                )

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

    @property
    def uses_thermoact_thermal_input(self) -> bool:
        return self.thermal_encoder_channel == THERMOACT_CHANNEL
