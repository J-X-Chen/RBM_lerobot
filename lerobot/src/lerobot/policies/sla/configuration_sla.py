# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""SLA: Scaling Spatial-Language-Action Model via Continual Pre-training (arXiv:2611.08373).

SLA is pi0.5 trained with an extra *spatial forcing* objective: PaliGemma image-token hidden
states of an intermediate layer are projected and aligned (cosine) with frozen OmniVGGT 3D
features. Inference is identical to pi0.5; OmniVGGT is only needed for training.
"""

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..pi05.configuration_pi05 import PI05Config

SLA_TRAINING_COMPONENT_REPO = "haofuly/SLA_training_component"


@PreTrainedConfig.register_subclass("sla")
@dataclass
class SLAConfig(PI05Config):
    """pi0.5 + spatial representation alignment (SLA / Spatial Forcing)."""

    # Spatial alignment loss -------------------------------------------------------------
    align_loss_coeff: float = 0.5
    """Weight of the alignment loss: ``loss = action_loss + align_loss_coeff * align_loss``. 0 disables it."""

    vla_layers_align: int = 12
    """PaliGemma layer (0-based, negative counts from the end) whose image-token outputs are aligned."""

    vggt_layers_align: int = -1
    """OmniVGGT aggregator layer (0-based, negative counts from the end) used as alignment target."""

    vggt_dim: int = 1024
    """OmniVGGT embedding width; aggregator features have ``2 * vggt_dim`` channels."""

    use_vlm_norm: bool = True
    """Apply a LayerNorm to the VLM hidden states before the projector."""

    use_vggt_pe: bool = True
    """Add VGGT's UV positional embedding before resampling the target features."""

    pooling_func: str = "bilinear"
    """Resampling of the 37x37 VGGT patch grid onto the 16x16 PaliGemma token grid."""

    ignore_img_padding_area: bool = True
    """Exclude tokens of the resize-with-pad border from the alignment loss."""

    zscore_align_targets: bool = False
    """Per-image spatial z-score of the targets (``iSF_zscore`` in the reference code)."""

    vggt_weight_path: str = SLA_TRAINING_COMPONENT_REPO
    """``OmniVGGT.safetensors`` file, a directory containing it (directly or under ``omnivggt/``),
    or a Hub repo id holding ``omnivggt/OmniVGGT.safetensors``."""

    align_vggt_device: str | None = None
    """Device of the frozen OmniVGGT teacher (default: the policy device), e.g. ``cuda:1``."""

    align_feature_batch_size: int | None = None
    """Optional chunk size for OmniVGGT feature extraction to bound teacher activation memory."""

    # Training-time prompt augmentation ---------------------------------------------------
    state_dropout_prob: float = 0.25
    """Probability of removing the discretized state from the prompt of a training sample."""

    def __post_init__(self):
        super().__post_init__()
        if self.align_loss_coeff < 0:
            raise ValueError(f"align_loss_coeff must be >= 0, got {self.align_loss_coeff}")
        if not 0.0 <= self.state_dropout_prob <= 1.0:
            raise ValueError(f"state_dropout_prob must be in [0, 1], got {self.state_dropout_prob}")
        if self.pooling_func != "bilinear":
            raise ValueError(f"Only pooling_func='bilinear' is supported, got {self.pooling_func!r}")
        if self.align_feature_batch_size is not None and self.align_feature_batch_size <= 0:
            raise ValueError("align_feature_batch_size must be positive when set")
        if self.align_loss_coeff > 0:
            if self.use_visual_memory:
                raise ValueError("SLA alignment does not support use_visual_memory (MEM video tokens).")
            if self.uses_dinov2_rgb_encoder or self.uses_gate_two_residual_rgb_vit:
                raise ValueError(
                    "SLA alignment requires the default SigLIP RGB encoder (rgb_input_type='vit')."
                )
            if self.train_expert_only:
                raise ValueError(
                    "SLA alignment trains the VLM; it cannot be combined with train_expert_only."
                )
        if self.state_dropout_prob > 0 and self.use_proprioceptive_memory:
            raise ValueError(
                "state_dropout_prob applies to the discretized state prompt, not MEM state tokens."
            )
