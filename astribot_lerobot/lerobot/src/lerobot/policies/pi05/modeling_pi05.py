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

import builtins
import logging
import math
from copy import deepcopy
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available, require_package
from lerobot.utils.local_assets import resolve_local_model_path

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.cache_utils import DynamicCache
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma

    from ..pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaForCausalLM,
        _gated_residual,
        layernorm_forward,
    )
else:
    CONFIG_MAPPING = None
    DynamicCache = None
    modeling_gemma = None
    PiGemmaForCausalLM = None
    _gated_residual = None
    layernorm_forward = None
    PaliGemmaForConditionalGenerationWithPiGemma = None
from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)

from ..pretrained import PreTrainedPolicy, T
from ..rtc.modeling_rtc import RTCProcessor
from .configuration_pi05 import (
    BLACK_IMPORTANCE_THERMAL_INPUT_TYPES,
    DEFAULT_IMAGE_SIZE,
    FIXED_MATCHED_THERMAL_INPUT_TYPES,
    MATCHED_THERMAL_INPUT_TYPES,
    ONLINE_MATCHED_THERMAL_INPUT_TYPES,
    PI05Config,
    TWO_STREAM_THERMAL_INPUT_TYPES,
)
from .thermal_cvae import ThermalCVAEEncoder
from .thermal_matching import (
    MinimaThermalMatcher,
    rescale_homography,
    warp_thermal_batch,
)
from .thermal_utils import (
    AnyThermalEncoder,
    AnyThermalResidualFusion,
    DINOv2RGBEncoder,
    GateViTMixFusion,
    OneGreyGateOneResViTFusion,
    RGBThreeBlackGateThreeResViTFusion,
    RGBThermalFusionDecoder,
    RGBThermalTokenAligner,
    ResViTAttentionFusion,
    ThermalHeadFusion,
    ThermalResNet18Encoder,
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
    estimate_thermal_black_background_tensor,
    image_to_bchw,
    mix_rgb_and_thermal_images,
    rgb_to_three_black_dominance_tensor,
    save_pi05_image_input_snapshots,
    thermal_to_background_normalized_grey_tensor,
    thermal_to_two_background_normalized_grey_tensor,
    thermal_to_two_black_background_normalized_grey_tensor,
    twogrey_patch_evidence,
)


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None
    capture_attention_map: bool | None


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    # Beta sampling uses _sample_dirichlet which isn't implemented for MPS, so sample on CPU
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def clone_past_key_values(past_key_values):
    """Clone the DynamicCache returned by prefix prefill for compiled denoising."""
    return DynamicCache(
        tuple(
            (keys.clone(), values.clone(), sliding_window) for keys, values, sliding_window in past_key_values
        )
    )


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
    legacy_float_padding: bool = False,
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black.

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)
        legacy_float_padding: If True, use the old lerobot0 float
            padding value (-1.0 before the final `img * 2 - 1` normalization).

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0) if legacy_float_padding else resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else (-1.0 if legacy_float_padding else 0.0)
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


# Define the complete layer computation function for gradient checkpointing
def _append_attention_summary(attention_collector, att_weights, layer_idx):
    if attention_collector is None or att_weights is None:
        return

    prefix_len = int(attention_collector["prefix_len"])
    suffix_len = int(attention_collector["suffix_len"])
    total_len = prefix_len + suffix_len
    query_len = int(att_weights.shape[-2])
    key_len = int(att_weights.shape[-1])
    if key_len < prefix_len or query_len <= 0:
        return

    query_start = prefix_len if query_len >= total_len else max(0, query_len - suffix_len)
    query_end = min(query_len, query_start + suffix_len)
    if query_end <= query_start:
        return

    with torch.no_grad():
        action_attn = att_weights.detach()[:, :, query_start:query_end, :]
        action_to_all = action_attn.float().mean(dim=(0, 1, 2)).cpu()
        action_to_prefix = action_to_all[:prefix_len].contiguous()
        heads_to_prefix = action_attn[:, :, :, :prefix_len].float().mean(dim=(0, 2)).cpu().contiguous()

    attention_collector.setdefault("layers", []).append(
        {
            "layer": int(layer_idx),
            "action_to_all": action_to_all.contiguous(),
            "action_to_prefix": action_to_prefix,
            "heads_to_prefix": heads_to_prefix,
        }
    )


def compute_layer_complete(
    inputs_embeds,
    attention_mask,
    position_ids,
    adarms_cond,
    layers,
    rotary_emb,
    attention_collector=None,
    layer_idx=0,
):
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    paligemma_layer = layers[0]
    scaling = paligemma_layer.self_attn.scaling
    # Attention computation
    att_output, att_weights = modeling_gemma.eager_attention_forward(
        paligemma_layer.self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    _append_attention_summary(attention_collector, att_weights, layer_idx)
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma_layer.self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        freeze_non_thermal_input: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.freeze_non_thermal_input = freeze_non_thermal_input

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep full vision path in float32 so we never toggle (toggle causes optimizer
        # "same dtype" error). Saves memory vs full float32; more memory than only 3 params.
        params_to_keep_float32 = [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only or self.freeze_non_thermal_input:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only or self.freeze_non_thermal_input:
            self.paligemma.eval()

    def embed_image_with_modules(self, image: torch.Tensor, vision_tower: nn.Module, projector: nn.Module):
        # Vision tower and multi_modal_projector are kept in float32 (params_to_keep_float32).
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = vision_tower(image)
        features = projector(image_outputs.last_hidden_state)
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_image_tokens_with_modules(self, image: torch.Tensor, vision_tower: nn.Module):
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        return vision_tower(image).last_hidden_state

    def project_image_tokens_with_modules(
        self,
        image_tokens: torch.Tensor,
        projector: nn.Module,
        out_dtype: torch.dtype,
    ):
        features = projector(image_tokens)
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_image(self, image: torch.Tensor):
        return self.embed_image_with_modules(
            image,
            self.paligemma.model.vision_tower,
            self.paligemma.model.multi_modal_projector,
        )

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.model.language_model.get_input_embeddings()(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        attention_collector: dict | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            paligemma_layers = self.paligemma.model.language_model.layers
            gemma_expert_layers = self.gemma_expert.model.layers
            rotary_emb = self.paligemma.model.language_model.rotary_emb

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx, layers in enumerate(zip(paligemma_layers, gemma_expert_layers, strict=True)):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        layers=layers,
                        rotary_emb=rotary_emb,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        layers=layers,
                        rotary_emb=rotary_emb,
                        attention_collector=attention_collector,
                        layer_idx=layer_idx,
                    )

            # final norm
            final_norms = (
                self.paligemma.model.language_model.norm,
                self.gemma_expert.model.norm,
            )

            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = layernorm_forward(final_norms[i], hidden_states, adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class PI05Pytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI05 PyTorch model."""

    def __init__(self, config: PI05Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor
        self._last_attention_map = None
        self._last_prefix_token_layout = []

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            freeze_non_thermal_input=config.freeze_non_thermal_input,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        image_embedding_dim = (
            self.paligemma_with_expert.paligemma.model.multi_modal_projector.linear.out_features
        )
        raw_image_token_dim = (
            self.paligemma_with_expert.paligemma.model.multi_modal_projector.linear.in_features
        )

        self.rgb_encoder = None
        if config.uses_dinov2_rgb_encoder:
            self.rgb_encoder = DINOv2RGBEncoder(
                output_dim=image_embedding_dim,
                model_type=config.rgb_dinov2_model_type,
                checkpoint_path=config.rgb_dinov2_checkpoint_path,
                dinov2_repo_path=config.rgb_dinov2_repo_path,
                freeze_backbone=config.rgb_dinov2_freeze_backbone,
                input_range=config.rgb_dinov2_input_range,
                dropout=config.thermal_encoder_dropout,
            )
            if config.freeze_non_thermal_input:
                self.rgb_encoder.eval()
                self.rgb_encoder.requires_grad_(False)

        self.rgb_threeblack_vision_tower = None
        self.rgb_threeblack_multi_modal_projector = None
        self.rgb_threeblack_gatetworesvit_fusion = None
        self.rgb_threeblack_gatetworesvit_head_feature = config.rgb_threeblack_head_feature
        self._last_rgb_threeblack_gatetworesvit_gate_summary = None
        if config.uses_gate_two_residual_rgb_vit:
            if self.rgb_threeblack_gatetworesvit_head_feature not in config.image_features:
                raise ValueError(
                    "PI05 policy.rgb_input_type='GateTwoResViT' requires head RGB feature "
                    f"{self.rgb_threeblack_gatetworesvit_head_feature!r} in image_features. "
                    f"Available image_features: {list(config.image_features)}."
                )
            language_embedding_dim = (
                self.paligemma_with_expert.paligemma.model.language_model
                .get_input_embeddings()
                .embedding_dim
            )
            if language_embedding_dim != image_embedding_dim:
                raise ValueError(
                    "PI05 policy.rgb_input_type='GateTwoResViT' requires projected RGB and "
                    "language tokens to have the same embedding dimension, got "
                    f"rgb={image_embedding_dim} and language={language_embedding_dim}. "
                    "Use the default paligemma_variant='gemma_2b'."
                )
            if image_embedding_dim % config.rgb_threeblack_num_heads != 0:
                raise ValueError(
                    "rgb_threeblack_num_heads must divide the PI05 prefix embedding "
                    f"dimension ({image_embedding_dim})"
                )
            # Red/green/blue views share one dedicated color tower with each
            # other, but never share encoder parameters with the original RGB
            # path. The copied parameters are synchronized again after loading
            # an old/base checkpoint; deepcopy here also gives correct behavior
            # for models initialized without a checkpoint.
            self.rgb_threeblack_vision_tower = deepcopy(
                self.paligemma_with_expert.paligemma.model.vision_tower
            )
            self.rgb_threeblack_multi_modal_projector = deepcopy(
                self.paligemma_with_expert.paligemma.model.multi_modal_projector
            )
            self.rgb_threeblack_gatetworesvit_fusion = RGBThreeBlackGateThreeResViTFusion(
                embed_dim=image_embedding_dim,
                num_heads=config.rgb_threeblack_num_heads,
                hidden_dim=config.rgb_threeblack_hidden_dim,
                dropout=config.thermal_encoder_dropout,
                temperature=config.rgb_threeblack_temperature,
                gate_init_std=config.rgb_threeblack_gate_init_std,
            )
            logging.info(
                "PI05 RGB GateTwoResViT keeps the normal RGB ViT for original camera images, "
                "uses one separate weight-shared color ViT for red/green/blue views, and emits "
                "three text-gated head-RGB residual streams on %s: "
                "base_alpha=(%.3f, %.3f, %.3f).",
                self.rgb_threeblack_gatetworesvit_head_feature,
                config.rgb_threeblack_red_alpha,
                config.rgb_threeblack_green_alpha,
                config.rgb_threeblack_blue_alpha,
            )

        self.thermal_twogrey_source_features = []
        self.thermal_twogrey_feature_map = {}
        thermal_model_image_features = list(config.thermal_image_features)
        if getattr(config, "uses_twogrey_thermal_input", False):
            self.thermal_twogrey_source_features = list(config.thermal_image_features)
            self.thermal_twogrey_feature_map = config.thermal_twogrey_feature_map()
            thermal_model_image_features = config.thermal_model_image_features()
        self.thermal_image_features = [
            key for key in thermal_model_image_features if key in config.image_features
        ]
        self.thermal_image_feature_set = set(self.thermal_image_features)
        self.thermal_encoder = None
        self.thermal_head_fusion = None
        self.rgb_thermal_align_aligner = None
        self.rgb_thermal_align_decoder = None
        self.rgb_thermal_align_head_feature = config.thermal_fusion_head_rgb_feature
        self.rgb_thermal_align_thermal_feature = None
        self.thermal_residual_fusion = None
        self.rgb_thermal_residual_head_feature = config.thermal_fusion_head_rgb_feature
        self.rgb_thermal_residual_thermal_feature = None
        self.rgb_thermal_resvit_head_feature = config.thermal_fusion_head_rgb_feature
        self.rgb_thermal_resvit_thermal_feature = None
        self.rgb_thermal_resvit_thermal_features = []
        self.rgb_thermal_resvit_thermal_feature_set = set()
        self.rgb_thermal_resvit_twogrey_cold_feature = None
        self.rgb_thermal_resvit_twogrey_hot_feature = None
        self.twogrey_gatevit_fusion = None
        self.rgb_thermal_gatevit_source_feature = None
        self.rgb_thermal_gatevit_thermal_features = []
        self.rgb_thermal_gatevit_thermal_feature_set = set()
        self.rgb_thermal_gatevit_cold_feature = None
        self.rgb_thermal_gatevit_hot_feature = None
        self._last_twogrey_gatevit_gate_summary = None
        self.twogrey_gateactionvit_fusion = None
        self.rgb_thermal_gateactionvit_source_feature = None
        self.rgb_thermal_gateactionvit_thermal_features = []
        self.rgb_thermal_gateactionvit_thermal_feature_set = set()
        self.rgb_thermal_gateactionvit_cold_feature = None
        self.rgb_thermal_gateactionvit_hot_feature = None
        self._last_twogrey_gateactionvit_gate_summary = None
        self._current_gateactionvit_attention_bias = None
        self.twogrey_doublegatevit_fusion = None
        self.rgb_thermal_doublegatevit_head_feature = config.thermal_fusion_head_rgb_feature
        self.rgb_thermal_doublegatevit_source_feature = None
        self.rgb_thermal_doublegatevit_thermal_features = []
        self.rgb_thermal_doublegatevit_thermal_feature_set = set()
        self.rgb_thermal_doublegatevit_cold_feature = None
        self.rgb_thermal_doublegatevit_hot_feature = None
        self._last_twogrey_doublegatevit_gate_summary = None
        self.twogrey_doublegatetworesvit_fusion = None
        self.rgb_thermal_doublegatetworesvit_head_feature = config.thermal_fusion_head_rgb_feature
        self.rgb_thermal_doublegatetworesvit_source_feature = None
        self.rgb_thermal_doublegatetworesvit_thermal_features = []
        self.rgb_thermal_doublegatetworesvit_thermal_feature_set = set()
        self.rgb_thermal_doublegatetworesvit_cold_feature = None
        self.rgb_thermal_doublegatetworesvit_hot_feature = None
        self._last_twogrey_doublegatetworesvit_gate_summary = None
        self.twogrey_patchgatevit_fusion = None
        self.rgb_thermal_patchgatevit_head_feature = config.thermal_fusion_head_rgb_feature
        self.rgb_thermal_patchgatevit_source_feature = None
        self.rgb_thermal_patchgatevit_thermal_features = []
        self.rgb_thermal_patchgatevit_thermal_feature_set = set()
        self.rgb_thermal_patchgatevit_cold_feature = None
        self.rgb_thermal_patchgatevit_hot_feature = None
        self._last_twogrey_patchgatevit_gate_summary = None
        self.twogrey_patchsinglegateresvit_fusion = None
        self.rgb_thermal_patchsinglegateresvit_head_feature = config.thermal_fusion_head_rgb_feature
        self.rgb_thermal_patchsinglegateresvit_source_feature = None
        self.rgb_thermal_patchsinglegateresvit_thermal_features = []
        self.rgb_thermal_patchsinglegateresvit_thermal_feature_set = set()
        self.rgb_thermal_patchsinglegateresvit_cold_feature = None
        self.rgb_thermal_patchsinglegateresvit_hot_feature = None
        self._last_twogrey_patchsinglegateresvit_gate_summary = None
        self.twogrey_smallpatchgatetworesvit_fusion = None
        self.rgb_thermal_smallpatchgatetworesvit_head_feature = config.thermal_fusion_head_rgb_feature
        self.rgb_thermal_smallpatchgatetworesvit_source_feature = None
        self.rgb_thermal_smallpatchgatetworesvit_thermal_features = []
        self.rgb_thermal_smallpatchgatetworesvit_thermal_feature_set = set()
        self.rgb_thermal_smallpatchgatetworesvit_cold_feature = None
        self.rgb_thermal_smallpatchgatetworesvit_hot_feature = None
        self._last_twogrey_smallpatchgatetworesvit_gate_summary = None
        self._last_twogrey_smallpatchsinglegatetworesvit_gate_summary = None
        self.twogrey_smallpatchgateresvit_fusion = None
        self.rgb_thermal_smallpatchgateresvit_head_feature = config.thermal_fusion_head_rgb_feature
        self.rgb_thermal_smallpatchgateresvit_source_feature = None
        self.rgb_thermal_smallpatchgateresvit_thermal_features = []
        self.rgb_thermal_smallpatchgateresvit_thermal_feature_set = set()
        self.rgb_thermal_smallpatchgateresvit_cold_feature = None
        self.rgb_thermal_smallpatchgateresvit_hot_feature = None
        self._last_twogrey_smallpatchgateresvit_gate_summary = None
        self.twogrey_gatevitmix_fusion = None
        self.gatevitmix_fusion = None
        self.rgb_thermal_gatevitmix_head_feature = config.thermal_fusion_head_rgb_feature
        self.rgb_thermal_gatevitmix_source_feature = None
        self.rgb_thermal_gatevitmix_thermal_features = []
        self.rgb_thermal_gatevitmix_thermal_feature_set = set()
        self.rgb_thermal_gatevitmix_cold_feature = None
        self.rgb_thermal_gatevitmix_hot_feature = None
        self._last_twogrey_gatevitmix_gate_summary = None
        self.twogrey_gateresvit_fusion = None
        self.rgb_thermal_gateresvit_head_feature = config.thermal_fusion_head_rgb_feature
        self.rgb_thermal_gateresvit_thermal_features = []
        self.rgb_thermal_gateresvit_thermal_feature_set = set()
        self.rgb_thermal_gateresvit_cold_feature = None
        self.rgb_thermal_gateresvit_hot_feature = None
        self._last_twogrey_gateresvit_gate_summary = None
        self._last_twogrey_gateresandvit_gate_summary = None
        self.twogrey_gatetworesvit_fusion = None
        self.rgb_thermal_gatetworesvit_head_feature = config.thermal_fusion_head_rgb_feature
        self.rgb_thermal_gatetworesvit_thermal_features = []
        self.rgb_thermal_gatetworesvit_thermal_feature_set = set()
        self.rgb_thermal_gatetworesvit_cold_feature = None
        self.rgb_thermal_gatetworesvit_hot_feature = None
        self._last_twogrey_gatetworesvit_gate_summary = None
        self.grey_gateoneresvit_fusion = None
        self.rgb_thermal_gateoneresvit_head_feature = config.thermal_fusion_head_rgb_feature
        self.rgb_thermal_gateoneresvit_thermal_feature = None
        self._last_grey_gateoneresvit_gate_summary = None
        self.twogrey_gateresvit3_fusion = None
        self.rgb_thermal_gateresvit3_head_feature = config.thermal_fusion_head_rgb_feature
        self.rgb_thermal_gateresvit3_thermal_features = []
        self.rgb_thermal_gateresvit3_thermal_feature_set = set()
        self.rgb_thermal_gateresvit3_cold_feature = None
        self.rgb_thermal_gateresvit3_hot_feature = None
        self._last_twogrey_gateresvit3_gate_summary = None
        self.resvit_attention_fusion = None
        self.rgb_thermal_resvit_attention_head_feature = config.thermal_fusion_head_rgb_feature
        self.rgb_thermal_resvit_attention_thermal_feature = None
        self._last_rgb_thermal_align_loss = None
        self._last_rgb_thermal_align_l1_loss = None
        self._last_rgb_thermal_align_mse_loss = None
        self.thermal_vision_tower = None
        self.thermal_multi_modal_projector = None
        if config.uses_thermal_vit_encoder:
            if not self.thermal_image_features:
                if (
                    config.rgb_thermal_align_fusion
                    or config.uses_residual_thermal_vit
                    or config.uses_gate_thermal_vit
                    or config.uses_gate_action_thermal_vit
                    or config.uses_double_gate_thermal_vit
                    or config.uses_double_gate_two_residual_thermal_vit
                    or config.uses_patch_gate_thermal_vit
                    or config.uses_patch_single_gate_residual_thermal_vit
                    or config.uses_small_patch_gate_two_res_thermal_vit
                    or config.uses_small_patch_gate_residual_thermal_vit
                    or config.uses_gate_thermal_vit_mix
                    or config.uses_gate_residual_thermal_vit
                    or config.uses_gate_res_and_thermal_vit
                    or config.uses_gate_two_residual_thermal_vit
                    or config.uses_gate_one_residual_thermal_vit
                    or config.uses_gate_residual_thermal_vit3
                    or config.uses_resvit_attention_encoder
                ):
                    raise ValueError(
                        "PI05 thermal ViT fusion requires at least one configured "
                        f"thermal image feature from {config.thermal_image_features}; "
                        f"available image_features: {list(config.image_features)}."
                    )
                logging.warning(
                    "PI05 thermal ViT path is enabled, but none of %s are present in image_features=%s. "
                    "The model will keep using the shared RGB ViT path.",
                    config.thermal_image_features,
                    list(config.image_features),
                )
            else:
                self.thermal_vision_tower = deepcopy(
                    self.paligemma_with_expert.paligemma.model.vision_tower
                )
                if config.uses_dedicated_thermal_vit:
                    self.thermal_multi_modal_projector = deepcopy(
                        self.paligemma_with_expert.paligemma.model.multi_modal_projector
                    )
                self._configure_thermal_vit_trainability()
                if config.uses_residual_thermal_vit:
                    if self.rgb_thermal_resvit_head_feature not in config.image_features:
                        raise ValueError(
                            "PI05 thermal_input_type='ResViT' requires "
                            f"{self.rgb_thermal_resvit_head_feature!r} in image_features. "
                            f"Available image_features: {list(config.image_features)}."
                        )
                    if getattr(config, "uses_twogrey_thermal_input", False):
                        twogrey_pairs = [
                            (cold_feature, hot_feature)
                            for cold_feature, hot_feature in self.thermal_twogrey_feature_map.values()
                            if cold_feature in self.thermal_image_feature_set
                            and hot_feature in self.thermal_image_feature_set
                        ]
                        if not twogrey_pairs:
                            raise ValueError(
                                "PI05 thermal_input_type='twogrey' with thermal_encoder_channel='ResViT' "
                                "requires generated cold/hot thermal image features in image_features. "
                                f"Configured thermal features: {self.thermal_twogrey_feature_map}."
                            )
                        (
                            self.rgb_thermal_resvit_twogrey_cold_feature,
                            self.rgb_thermal_resvit_twogrey_hot_feature,
                        ) = twogrey_pairs[0]
                        self.rgb_thermal_resvit_thermal_features = [
                            feature for pair in twogrey_pairs for feature in pair
                        ]
                    else:
                        self.rgb_thermal_resvit_thermal_features = [self.thermal_image_features[0]]

                    self.rgb_thermal_resvit_thermal_feature = self.rgb_thermal_resvit_thermal_features[0]
                    self.rgb_thermal_resvit_thermal_feature_set = set(
                        self.rgb_thermal_resvit_thermal_features
                    )
                    if self.rgb_thermal_resvit_head_feature in self.rgb_thermal_resvit_thermal_feature_set:
                        raise ValueError(
                            "PI05 thermal_input_type='ResViT' requires distinct RGB head and thermal "
                            f"features, got {self.rgb_thermal_resvit_head_feature!r}."
                        )
                if config.uses_gate_thermal_vit:
                    twogrey_pairs = [
                        (source_feature, cold_feature, hot_feature)
                        for source_feature, (
                            cold_feature,
                            hot_feature,
                        ) in self.thermal_twogrey_feature_map.items()
                        if cold_feature in self.thermal_image_feature_set
                        and hot_feature in self.thermal_image_feature_set
                    ]
                    if not twogrey_pairs:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='GateViT' requires generated cold/hot "
                            "thermal image features in image_features. "
                            f"Configured thermal features: {self.thermal_twogrey_feature_map}."
                        )
                    (
                        self.rgb_thermal_gatevit_source_feature,
                        self.rgb_thermal_gatevit_cold_feature,
                        self.rgb_thermal_gatevit_hot_feature,
                    ) = twogrey_pairs[0]
                    self.rgb_thermal_gatevit_thermal_features = [
                        feature for _, cold, hot in twogrey_pairs for feature in (cold, hot)
                    ]
                    self.rgb_thermal_gatevit_thermal_feature_set = set(
                        self.rgb_thermal_gatevit_thermal_features
                    )
                    if image_embedding_dim % config.thermal_twogrey_gatevit_num_heads != 0:
                        raise ValueError(
                            "thermal_twogrey_gatevit_num_heads must divide the PI05 prefix "
                            f"embedding dimension ({image_embedding_dim})"
                        )
                    self.twogrey_gatevit_fusion = TwoGreyGateViTFusion(
                        embed_dim=image_embedding_dim,
                        num_heads=config.thermal_twogrey_gatevit_num_heads,
                        hidden_dim=config.thermal_twogrey_gatevit_hidden_dim,
                        dropout=config.thermal_encoder_dropout,
                        temperature=config.thermal_twogrey_gatevit_temperature,
                        gate_init_std=config.thermal_twogrey_gatevit_gate_init_std,
                    )
                    logging.info(
                        "PI05 GateViT merges projected cold/hot streams into one independent "
                        "thermal token stream: cold=%s, hot=%s.",
                        self.rgb_thermal_gatevit_cold_feature,
                        self.rgb_thermal_gatevit_hot_feature,
                    )
                if config.uses_gate_action_thermal_vit:
                    twogrey_pairs = [
                        (source_feature, cold_feature, hot_feature)
                        for source_feature, (
                            cold_feature,
                            hot_feature,
                        ) in self.thermal_twogrey_feature_map.items()
                        if cold_feature in self.thermal_image_feature_set
                        and hot_feature in self.thermal_image_feature_set
                    ]
                    if not twogrey_pairs:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='GateActionViT' requires generated "
                            "cold/hot thermal image features in image_features. "
                            f"Configured thermal features: {self.thermal_twogrey_feature_map}."
                        )
                    (
                        self.rgb_thermal_gateactionvit_source_feature,
                        self.rgb_thermal_gateactionvit_cold_feature,
                        self.rgb_thermal_gateactionvit_hot_feature,
                    ) = twogrey_pairs[0]
                    self.rgb_thermal_gateactionvit_thermal_features = [
                        feature for _, cold, hot in twogrey_pairs for feature in (cold, hot)
                    ]
                    self.rgb_thermal_gateactionvit_thermal_feature_set = set(
                        self.rgb_thermal_gateactionvit_thermal_features
                    )
                    self.twogrey_gateactionvit_fusion = TwoGreyGateActionViTFusion(
                        embed_dim=image_embedding_dim,
                        num_heads=config.thermal_twogrey_gateactionvit_num_heads,
                        hidden_dim=config.thermal_twogrey_gateactionvit_hidden_dim,
                        dropout=config.thermal_encoder_dropout,
                        temperature=config.thermal_twogrey_gateactionvit_temperature,
                        gate_init_std=config.thermal_twogrey_gateactionvit_gate_init_std,
                        bias_epsilon=config.thermal_twogrey_gateactionvit_bias_epsilon,
                        bias_max_strength=config.thermal_twogrey_gateactionvit_bias_max_strength,
                        bias_init_strength=config.thermal_twogrey_gateactionvit_bias_init_strength,
                    )
                    logging.info(
                        "PI05 GateActionViT preserves projected cold/hot streams and applies "
                        "text-gated action attention bias: cold=%s, hot=%s, "
                        "bias_init_strength=%.3f, bias_max_strength=%.3f.",
                        self.rgb_thermal_gateactionvit_cold_feature,
                        self.rgb_thermal_gateactionvit_hot_feature,
                        config.thermal_twogrey_gateactionvit_bias_init_strength,
                        config.thermal_twogrey_gateactionvit_bias_max_strength,
                    )
                if config.uses_double_gate_thermal_vit:
                    if self.rgb_thermal_doublegatevit_head_feature not in config.image_features:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='DoubleGateViT' requires "
                            f"{self.rgb_thermal_doublegatevit_head_feature!r} in image_features. "
                            f"Available image_features: {list(config.image_features)}."
                        )
                    twogrey_pairs = [
                        (source_feature, cold_feature, hot_feature)
                        for source_feature, (
                            cold_feature,
                            hot_feature,
                        ) in self.thermal_twogrey_feature_map.items()
                        if cold_feature in self.thermal_image_feature_set
                        and hot_feature in self.thermal_image_feature_set
                    ]
                    if not twogrey_pairs:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='DoubleGateViT' requires generated "
                            "cold/hot thermal image features in image_features. "
                            f"Configured thermal features: {self.thermal_twogrey_feature_map}."
                        )
                    (
                        self.rgb_thermal_doublegatevit_source_feature,
                        self.rgb_thermal_doublegatevit_cold_feature,
                        self.rgb_thermal_doublegatevit_hot_feature,
                    ) = twogrey_pairs[0]
                    self.rgb_thermal_doublegatevit_thermal_features = [
                        feature for _, cold, hot in twogrey_pairs for feature in (cold, hot)
                    ]
                    self.rgb_thermal_doublegatevit_thermal_feature_set = set(
                        self.rgb_thermal_doublegatevit_thermal_features
                    )
                    if (
                        self.rgb_thermal_doublegatevit_head_feature
                        in self.rgb_thermal_doublegatevit_thermal_feature_set
                    ):
                        raise ValueError(
                            "PI05 thermal_encoder_channel='DoubleGateViT' requires distinct "
                            "head RGB and thermal features."
                        )
                    self.twogrey_doublegatevit_fusion = TwoGreyDoubleGateViTFusion(
                        embed_dim=image_embedding_dim,
                        num_heads=config.thermal_twogrey_doublegatevit_num_heads,
                        hidden_dim=config.thermal_twogrey_doublegatevit_hidden_dim,
                        dropout=config.thermal_encoder_dropout,
                        temperature=config.thermal_twogrey_doublegatevit_temperature,
                        gate_init_std=config.thermal_twogrey_doublegatevit_gate_init_std,
                        rgb_gate_max_strength=(
                            config.thermal_twogrey_doublegatevit_rgb_gate_max_strength
                        ),
                        rgb_gate_init_strength=(
                            config.thermal_twogrey_doublegatevit_rgb_gate_init_strength
                        ),
                    )
                    logging.info(
                        "PI05 DoubleGateViT combines text and head-RGB alignment gates, then "
                        "merges cold/hot into one thermal stream: head=%s, cold=%s, hot=%s, "
                        "rgb_gate_init_strength=%.3f.",
                        self.rgb_thermal_doublegatevit_head_feature,
                        self.rgb_thermal_doublegatevit_cold_feature,
                        self.rgb_thermal_doublegatevit_hot_feature,
                        config.thermal_twogrey_doublegatevit_rgb_gate_init_strength,
                    )
                if config.uses_double_gate_two_residual_thermal_vit:
                    if self.rgb_thermal_doublegatetworesvit_head_feature not in config.image_features:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='DoubleGateTwoResViT' requires "
                            f"{self.rgb_thermal_doublegatetworesvit_head_feature!r} in image_features. "
                            f"Available image_features: {list(config.image_features)}."
                        )
                    twogrey_pairs = [
                        (source_feature, cold_feature, hot_feature)
                        for source_feature, (
                            cold_feature,
                            hot_feature,
                        ) in self.thermal_twogrey_feature_map.items()
                        if cold_feature in self.thermal_image_feature_set
                        and hot_feature in self.thermal_image_feature_set
                    ]
                    if not twogrey_pairs:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='DoubleGateTwoResViT' requires generated "
                            "cold/hot thermal image features in image_features. "
                            f"Configured thermal features: {self.thermal_twogrey_feature_map}."
                        )
                    (
                        self.rgb_thermal_doublegatetworesvit_source_feature,
                        self.rgb_thermal_doublegatetworesvit_cold_feature,
                        self.rgb_thermal_doublegatetworesvit_hot_feature,
                    ) = twogrey_pairs[0]
                    self.rgb_thermal_doublegatetworesvit_thermal_features = [
                        feature for _, cold, hot in twogrey_pairs for feature in (cold, hot)
                    ]
                    self.rgb_thermal_doublegatetworesvit_thermal_feature_set = set(
                        self.rgb_thermal_doublegatetworesvit_thermal_features
                    )
                    if (
                        self.rgb_thermal_doublegatetworesvit_head_feature
                        in self.rgb_thermal_doublegatetworesvit_thermal_feature_set
                    ):
                        raise ValueError(
                            "PI05 thermal_encoder_channel='DoubleGateTwoResViT' requires distinct "
                            "head RGB and thermal features."
                        )
                    self.twogrey_doublegatetworesvit_fusion = TwoGreyDoubleGateTwoResViTFusion(
                        embed_dim=image_embedding_dim,
                        num_heads=config.thermal_twogrey_doublegatevit_num_heads,
                        hidden_dim=config.thermal_twogrey_doublegatevit_hidden_dim,
                        dropout=config.thermal_encoder_dropout,
                        temperature=config.thermal_twogrey_doublegatevit_temperature,
                        gate_init_std=config.thermal_twogrey_doublegatevit_gate_init_std,
                        rgb_gate_max_strength=(
                            config.thermal_twogrey_doublegatevit_rgb_gate_max_strength
                        ),
                        rgb_gate_init_strength=(
                            config.thermal_twogrey_doublegatevit_rgb_gate_init_strength
                        ),
                    )
                    logging.info(
                        "PI05 DoubleGateTwoResViT combines text and head-RGB alignment gates, "
                        "then emits two head-RGB residual streams: head=%s, cold=%s, hot=%s, "
                        "base_alpha=%.3f, base_beta=%.3f, rgb_gate_init_strength=%.3f.",
                        self.rgb_thermal_doublegatetworesvit_head_feature,
                        self.rgb_thermal_doublegatetworesvit_cold_feature,
                        self.rgb_thermal_doublegatetworesvit_hot_feature,
                        config.thermal_twogrey_resvit_cold_alpha,
                        config.thermal_twogrey_resvit_hot_beta,
                        config.thermal_twogrey_doublegatevit_rgb_gate_init_strength,
                    )
                if config.uses_patch_gate_thermal_vit:
                    if self.rgb_thermal_patchgatevit_head_feature not in config.image_features:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='PatchGateViT' requires "
                            f"{self.rgb_thermal_patchgatevit_head_feature!r} in image_features. "
                            f"Available image_features: {list(config.image_features)}."
                        )
                    twogrey_pairs = [
                        (source_feature, cold_feature, hot_feature)
                        for source_feature, (
                            cold_feature,
                            hot_feature,
                        ) in self.thermal_twogrey_feature_map.items()
                        if cold_feature in self.thermal_image_feature_set
                        and hot_feature in self.thermal_image_feature_set
                    ]
                    if not twogrey_pairs:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='PatchGateViT' requires generated "
                            "cold/hot thermal image features in image_features. "
                            f"Configured thermal features: {self.thermal_twogrey_feature_map}."
                        )
                    (
                        self.rgb_thermal_patchgatevit_source_feature,
                        self.rgb_thermal_patchgatevit_cold_feature,
                        self.rgb_thermal_patchgatevit_hot_feature,
                    ) = twogrey_pairs[0]
                    self.rgb_thermal_patchgatevit_thermal_features = [
                        feature for _, cold, hot in twogrey_pairs for feature in (cold, hot)
                    ]
                    self.rgb_thermal_patchgatevit_thermal_feature_set = set(
                        self.rgb_thermal_patchgatevit_thermal_features
                    )
                    if (
                        self.rgb_thermal_patchgatevit_head_feature
                        in self.rgb_thermal_patchgatevit_thermal_feature_set
                    ):
                        raise ValueError(
                            "PI05 thermal_encoder_channel='PatchGateViT' requires distinct "
                            "head RGB and thermal features."
                        )
                    patch_gate_fusion_classes = {
                        "patchgatevit": TwoGreyPatchGateViTFusion,
                        "smallpatchgatevit": TwoGreySmallPatchGateViTFusion,
                        "hardsmallpatchgatevit": TwoGreyHardSmallPatchGateViTFusion,
                    }
                    patch_gate_fusion_class = patch_gate_fusion_classes[
                        config.thermal_encoder_channel
                    ]
                    patch_gate_kwargs = {
                        "embed_dim": image_embedding_dim,
                        "gate_dim": config.thermal_twogrey_patchgatevit_gate_dim,
                        "num_heads": config.thermal_twogrey_patchgatevit_num_heads,
                        "hidden_dim": config.thermal_twogrey_patchgatevit_hidden_dim,
                        "dropout": config.thermal_encoder_dropout,
                        "temperature": config.thermal_twogrey_patchgatevit_temperature,
                        "gate_init_std": config.thermal_twogrey_patchgatevit_gate_init_std,
                        "evidence_strength": (
                            0.0
                            if config.uses_hard_small_patch_gate_thermal_vit
                            else config.thermal_twogrey_patchgatevit_evidence_strength
                        ),
                        "evidence_floor": config.thermal_twogrey_patchgatevit_evidence_floor,
                        "detach_head_rgb": config.thermal_twogrey_patchgatevit_detach_head_rgb,
                    }
                    if config.thermal_encoder_channel in {
                        "smallpatchgatevit",
                        "hardsmallpatchgatevit",
                    }:
                        patch_gate_kwargs["gate_grid"] = (
                            config.thermal_twogrey_smallpatchgatevit_gate_grid
                        )
                    self.twogrey_patchgatevit_fusion = patch_gate_fusion_class(
                        **patch_gate_kwargs
                    )
                    patch_gate_mode_name = {
                        "patchgatevit": "PatchGateViT",
                        "smallpatchgatevit": "SmallPatchGateViT",
                        "hardsmallpatchgatevit": "HardSmallPatchGateViT",
                    }[config.thermal_encoder_channel]
                    logging.info(
                        "PI05 %s uses patch-level cold/hot routing: "
                        "head=%s, cold=%s, hot=%s, gate_dim=%d, gate_grid=%s, "
                        "parameters=%d, evidence_strength=%.3f, detach_head_rgb=%s.",
                        patch_gate_mode_name,
                        self.rgb_thermal_patchgatevit_head_feature,
                        self.rgb_thermal_patchgatevit_cold_feature,
                        self.rgb_thermal_patchgatevit_hot_feature,
                        config.thermal_twogrey_patchgatevit_gate_dim,
                        (
                            config.thermal_twogrey_smallpatchgatevit_gate_grid
                            if config.thermal_encoder_channel
                            in {"smallpatchgatevit", "hardsmallpatchgatevit"}
                            else None
                        ),
                        self.twogrey_patchgatevit_fusion.parameter_count(),
                        self.twogrey_patchgatevit_fusion.evidence_strength,
                        config.thermal_twogrey_patchgatevit_detach_head_rgb,
                    )
                if config.uses_patch_single_gate_residual_thermal_vit:
                    if self.rgb_thermal_patchsinglegateresvit_head_feature not in config.image_features:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='PatchSingleGateResViT' requires "
                            f"{self.rgb_thermal_patchsinglegateresvit_head_feature!r} in "
                            f"image_features. Available image_features: {list(config.image_features)}."
                        )
                    twogrey_pairs = [
                        (source_feature, cold_feature, hot_feature)
                        for source_feature, (
                            cold_feature,
                            hot_feature,
                        ) in self.thermal_twogrey_feature_map.items()
                        if cold_feature in self.thermal_image_feature_set
                        and hot_feature in self.thermal_image_feature_set
                    ]
                    if not twogrey_pairs:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='PatchSingleGateResViT' requires generated "
                            "cold/hot thermal image features in image_features. "
                            f"Configured thermal features: {self.thermal_twogrey_feature_map}."
                        )
                    (
                        self.rgb_thermal_patchsinglegateresvit_source_feature,
                        self.rgb_thermal_patchsinglegateresvit_cold_feature,
                        self.rgb_thermal_patchsinglegateresvit_hot_feature,
                    ) = twogrey_pairs[0]
                    self.rgb_thermal_patchsinglegateresvit_thermal_features = [
                        feature for _, cold, hot in twogrey_pairs for feature in (cold, hot)
                    ]
                    self.rgb_thermal_patchsinglegateresvit_thermal_feature_set = set(
                        self.rgb_thermal_patchsinglegateresvit_thermal_features
                    )
                    if (
                        self.rgb_thermal_patchsinglegateresvit_head_feature
                        in self.rgb_thermal_patchsinglegateresvit_thermal_feature_set
                    ):
                        raise ValueError(
                            "PI05 thermal_encoder_channel='PatchSingleGateResViT' requires "
                            "distinct head RGB and thermal features."
                        )
                    self.twogrey_patchsinglegateresvit_fusion = TwoGreyPatchSingleGateResViTFusion(
                        embed_dim=image_embedding_dim,
                        gate_dim=config.thermal_twogrey_patchgatevit_gate_dim,
                        num_heads=config.thermal_twogrey_patchgatevit_num_heads,
                        hidden_dim=config.thermal_twogrey_patchgatevit_hidden_dim,
                        dropout=config.thermal_encoder_dropout,
                        temperature=config.thermal_twogrey_patchgatevit_temperature,
                        gate_init_std=config.thermal_twogrey_patchgatevit_gate_init_std,
                        evidence_strength=config.thermal_twogrey_patchgatevit_evidence_strength,
                        evidence_floor=config.thermal_twogrey_patchgatevit_evidence_floor,
                        detach_head_rgb=config.thermal_twogrey_patchgatevit_detach_head_rgb,
                    )
                    logging.info(
                        "PI05 PatchSingleGateResViT fuses full-patch text-gated cold/hot "
                        "residuals into one head-RGB stream: head=%s, cold=%s, hot=%s, "
                        "gate_dim=%d, parameters=%d, base_alpha=%.3f, base_beta=%.3f.",
                        self.rgb_thermal_patchsinglegateresvit_head_feature,
                        self.rgb_thermal_patchsinglegateresvit_cold_feature,
                        self.rgb_thermal_patchsinglegateresvit_hot_feature,
                        config.thermal_twogrey_patchgatevit_gate_dim,
                        self.twogrey_patchsinglegateresvit_fusion.parameter_count(),
                        config.thermal_twogrey_resvit_cold_alpha,
                        config.thermal_twogrey_resvit_hot_beta,
                    )
                if config.uses_small_patch_gate_two_res_thermal_vit:
                    small_patch_two_res_mode_name = (
                        "SmallPatchSingleGateTwoResViT"
                        if config.uses_small_patch_single_gate_two_res_thermal_vit
                        else "SmallPatchGateTwoResViT"
                    )
                    if self.rgb_thermal_smallpatchgatetworesvit_head_feature not in config.image_features:
                        raise ValueError(
                            f"PI05 thermal_encoder_channel='{small_patch_two_res_mode_name}' requires "
                            f"{self.rgb_thermal_smallpatchgatetworesvit_head_feature!r} in "
                            f"image_features. Available image_features: {list(config.image_features)}."
                        )
                    twogrey_pairs = [
                        (source_feature, cold_feature, hot_feature)
                        for source_feature, (
                            cold_feature,
                            hot_feature,
                        ) in self.thermal_twogrey_feature_map.items()
                        if cold_feature in self.thermal_image_feature_set
                        and hot_feature in self.thermal_image_feature_set
                    ]
                    if not twogrey_pairs:
                        raise ValueError(
                            f"PI05 thermal_encoder_channel='{small_patch_two_res_mode_name}' requires generated "
                            "cold/hot thermal image features in image_features. "
                            f"Configured thermal features: {self.thermal_twogrey_feature_map}."
                        )
                    (
                        self.rgb_thermal_smallpatchgatetworesvit_source_feature,
                        self.rgb_thermal_smallpatchgatetworesvit_cold_feature,
                        self.rgb_thermal_smallpatchgatetworesvit_hot_feature,
                    ) = twogrey_pairs[0]
                    self.rgb_thermal_smallpatchgatetworesvit_thermal_features = [
                        feature for _, cold, hot in twogrey_pairs for feature in (cold, hot)
                    ]
                    self.rgb_thermal_smallpatchgatetworesvit_thermal_feature_set = set(
                        self.rgb_thermal_smallpatchgatetworesvit_thermal_features
                    )
                    if (
                        self.rgb_thermal_smallpatchgatetworesvit_head_feature
                        in self.rgb_thermal_smallpatchgatetworesvit_thermal_feature_set
                    ):
                        raise ValueError(
                            f"PI05 thermal_encoder_channel='{small_patch_two_res_mode_name}' requires "
                            "distinct head RGB and thermal features."
                        )
                    small_patch_two_res_fusion_class = (
                        TwoGreySmallPatchSingleGateTwoResViTFusion
                        if config.uses_small_patch_single_gate_two_res_thermal_vit
                        else TwoGreySmallPatchGateTwoResViTFusion
                    )
                    self.twogrey_smallpatchgatetworesvit_fusion = (
                        small_patch_two_res_fusion_class(
                            embed_dim=image_embedding_dim,
                            gate_dim=config.thermal_twogrey_patchgatevit_gate_dim,
                            gate_grid=config.thermal_twogrey_smallpatchgatevit_gate_grid,
                            num_heads=config.thermal_twogrey_patchgatevit_num_heads,
                            hidden_dim=config.thermal_twogrey_patchgatevit_hidden_dim,
                            dropout=config.thermal_encoder_dropout,
                            temperature=config.thermal_twogrey_patchgatevit_temperature,
                            gate_init_std=config.thermal_twogrey_patchgatevit_gate_init_std,
                            evidence_strength=(
                                config.thermal_twogrey_patchgatevit_evidence_strength
                            ),
                            evidence_floor=config.thermal_twogrey_patchgatevit_evidence_floor,
                            detach_head_rgb=config.thermal_twogrey_patchgatevit_detach_head_rgb,
                        )
                    )
                    logging.info(
                        "PI05 %s emits two head-RGB residual streams: "
                        "head=%s, cold=%s, hot=%s, gate_dim=%d, gate_grid=%s, "
                        "text_only_gate=%s, parameters=%d, base_alpha=%.3f, base_beta=%.3f.",
                        small_patch_two_res_mode_name,
                        self.rgb_thermal_smallpatchgatetworesvit_head_feature,
                        self.rgb_thermal_smallpatchgatetworesvit_cold_feature,
                        self.rgb_thermal_smallpatchgatetworesvit_hot_feature,
                        config.thermal_twogrey_patchgatevit_gate_dim,
                        config.thermal_twogrey_smallpatchgatevit_gate_grid,
                        config.uses_small_patch_single_gate_two_res_thermal_vit,
                        self.twogrey_smallpatchgatetworesvit_fusion.parameter_count(),
                        config.thermal_twogrey_resvit_cold_alpha,
                        config.thermal_twogrey_resvit_hot_beta,
                    )
                if config.uses_small_patch_gate_residual_thermal_vit:
                    if self.rgb_thermal_smallpatchgateresvit_head_feature not in config.image_features:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='SmallPatchGateResViT' requires "
                            f"{self.rgb_thermal_smallpatchgateresvit_head_feature!r} in "
                            f"image_features. Available image_features: {list(config.image_features)}."
                        )
                    twogrey_pairs = [
                        (source_feature, cold_feature, hot_feature)
                        for source_feature, (
                            cold_feature,
                            hot_feature,
                        ) in self.thermal_twogrey_feature_map.items()
                        if cold_feature in self.thermal_image_feature_set
                        and hot_feature in self.thermal_image_feature_set
                    ]
                    if not twogrey_pairs:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='SmallPatchGateResViT' requires generated "
                            "cold/hot thermal image features in image_features. "
                            f"Configured thermal features: {self.thermal_twogrey_feature_map}."
                        )
                    (
                        self.rgb_thermal_smallpatchgateresvit_source_feature,
                        self.rgb_thermal_smallpatchgateresvit_cold_feature,
                        self.rgb_thermal_smallpatchgateresvit_hot_feature,
                    ) = twogrey_pairs[0]
                    self.rgb_thermal_smallpatchgateresvit_thermal_features = [
                        feature for _, cold, hot in twogrey_pairs for feature in (cold, hot)
                    ]
                    self.rgb_thermal_smallpatchgateresvit_thermal_feature_set = set(
                        self.rgb_thermal_smallpatchgateresvit_thermal_features
                    )
                    if (
                        self.rgb_thermal_smallpatchgateresvit_head_feature
                        in self.rgb_thermal_smallpatchgateresvit_thermal_feature_set
                    ):
                        raise ValueError(
                            "PI05 thermal_encoder_channel='SmallPatchGateResViT' requires "
                            "distinct head RGB and thermal features."
                        )
                    small_patch_res_fusion_class = (
                        TwoGreySmallPatchSingleGateResViTFusion
                        if config.uses_small_patch_single_gate_residual_thermal_vit
                        else TwoGreySmallPatchGateResViTFusion
                    )
                    self.twogrey_smallpatchgateresvit_fusion = small_patch_res_fusion_class(
                        embed_dim=image_embedding_dim,
                        gate_dim=config.thermal_twogrey_patchgatevit_gate_dim,
                        gate_grid=config.thermal_twogrey_smallpatchgatevit_gate_grid,
                        num_heads=config.thermal_twogrey_patchgatevit_num_heads,
                        hidden_dim=config.thermal_twogrey_patchgatevit_hidden_dim,
                        dropout=config.thermal_encoder_dropout,
                        temperature=config.thermal_twogrey_patchgatevit_temperature,
                        gate_init_std=config.thermal_twogrey_patchgatevit_gate_init_std,
                        evidence_strength=config.thermal_twogrey_patchgatevit_evidence_strength,
                        evidence_floor=config.thermal_twogrey_patchgatevit_evidence_floor,
                        detach_head_rgb=config.thermal_twogrey_patchgatevit_detach_head_rgb,
                    )
                    small_patch_res_mode_name = (
                        "SmallPatchSingleGateResViT"
                        if config.uses_small_patch_single_gate_residual_thermal_vit
                        else "SmallPatchGateResViT"
                    )
                    logging.info(
                        "PI05 %s fuses coarse patch-gated cold/hot residuals into one "
                        "head-RGB stream: head=%s, cold=%s, hot=%s, gate_dim=%d, "
                        "gate_grid=%s, parameters=%d, base_alpha=%.3f, base_beta=%.3f.",
                        small_patch_res_mode_name,
                        self.rgb_thermal_smallpatchgateresvit_head_feature,
                        self.rgb_thermal_smallpatchgateresvit_cold_feature,
                        self.rgb_thermal_smallpatchgateresvit_hot_feature,
                        config.thermal_twogrey_patchgatevit_gate_dim,
                        config.thermal_twogrey_smallpatchgatevit_gate_grid,
                        self.twogrey_smallpatchgateresvit_fusion.parameter_count(),
                        config.thermal_twogrey_resvit_cold_alpha,
                        config.thermal_twogrey_resvit_hot_beta,
                    )
                if config.uses_gate_thermal_vit_mix:
                    if self.rgb_thermal_gatevitmix_head_feature not in config.image_features:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='GateViTMix' requires "
                            f"{self.rgb_thermal_gatevitmix_head_feature!r} in image_features. "
                            f"Available image_features: {list(config.image_features)}."
                        )
                    twogrey_pairs = [
                        (source_feature, cold_feature, hot_feature)
                        for source_feature, (
                            cold_feature,
                            hot_feature,
                        ) in self.thermal_twogrey_feature_map.items()
                        if cold_feature in self.thermal_image_feature_set
                        and hot_feature in self.thermal_image_feature_set
                    ]
                    if not twogrey_pairs:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='GateViTMix' requires generated cold/hot "
                            "thermal image features in image_features. "
                            f"Configured thermal features: {self.thermal_twogrey_feature_map}."
                        )
                    (
                        self.rgb_thermal_gatevitmix_source_feature,
                        self.rgb_thermal_gatevitmix_cold_feature,
                        self.rgb_thermal_gatevitmix_hot_feature,
                    ) = twogrey_pairs[0]
                    self.rgb_thermal_gatevitmix_thermal_features = [
                        feature for _, cold, hot in twogrey_pairs for feature in (cold, hot)
                    ]
                    self.rgb_thermal_gatevitmix_thermal_feature_set = set(
                        self.rgb_thermal_gatevitmix_thermal_features
                    )
                    if (
                        self.rgb_thermal_gatevitmix_head_feature
                        in self.rgb_thermal_gatevitmix_thermal_feature_set
                    ):
                        raise ValueError(
                            "PI05 thermal_encoder_channel='GateViTMix' requires distinct head RGB "
                            f"and thermal features, got {self.rgb_thermal_gatevitmix_head_feature!r}."
                        )
                    self.twogrey_gatevitmix_fusion = TwoGreyGateViTFusion(
                        embed_dim=image_embedding_dim,
                        num_heads=config.thermal_twogrey_gatevitmix_num_heads,
                        hidden_dim=config.thermal_twogrey_gatevitmix_hidden_dim,
                        dropout=config.thermal_encoder_dropout,
                        temperature=config.thermal_twogrey_gatevitmix_temperature,
                        gate_init_std=config.thermal_twogrey_gatevitmix_gate_init_std,
                    )
                    self.gatevitmix_fusion = GateViTMixFusion(
                        embed_dim=image_embedding_dim,
                        mix_dim=config.thermal_twogrey_gatevitmix_mix_dim,
                        num_heads=config.thermal_twogrey_gatevitmix_num_heads,
                        dropout=config.thermal_encoder_dropout,
                        context_scale=config.thermal_twogrey_gatevitmix_context_scale,
                        detach_head_rgb=config.thermal_twogrey_gatevitmix_detach_head_rgb,
                    )
                    logging.info(
                        "PI05 GateViTMix preserves %s and emits an independent head-conditioned "
                        "thermal stream: cold=%s, hot=%s, mix_dim=%d, context_scale=%.3f, "
                        "detach_head_rgb=%s.",
                        self.rgb_thermal_gatevitmix_head_feature,
                        self.rgb_thermal_gatevitmix_cold_feature,
                        self.rgb_thermal_gatevitmix_hot_feature,
                        config.thermal_twogrey_gatevitmix_mix_dim,
                        config.thermal_twogrey_gatevitmix_context_scale,
                        config.thermal_twogrey_gatevitmix_detach_head_rgb,
                    )
                if config.uses_gate_residual_thermal_vit or config.uses_gate_res_and_thermal_vit:
                    if self.rgb_thermal_gateresvit_head_feature not in config.image_features:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='GateResViT'/'GateResandViT' requires "
                            f"{self.rgb_thermal_gateresvit_head_feature!r} in image_features. "
                            f"Available image_features: {list(config.image_features)}."
                        )
                    twogrey_pairs = [
                        (cold_feature, hot_feature)
                        for cold_feature, hot_feature in self.thermal_twogrey_feature_map.values()
                        if cold_feature in self.thermal_image_feature_set
                        and hot_feature in self.thermal_image_feature_set
                    ]
                    if not twogrey_pairs:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='GateResViT'/'GateResandViT' requires "
                            "generated cold/hot thermal image features in image_features. "
                            f"Configured thermal features: {self.thermal_twogrey_feature_map}."
                        )
                    (
                        self.rgb_thermal_gateresvit_cold_feature,
                        self.rgb_thermal_gateresvit_hot_feature,
                    ) = twogrey_pairs[0]
                    self.rgb_thermal_gateresvit_thermal_features = [
                        feature for pair in twogrey_pairs for feature in pair
                    ]
                    self.rgb_thermal_gateresvit_thermal_feature_set = set(
                        self.rgb_thermal_gateresvit_thermal_features
                    )
                    if (
                        self.rgb_thermal_gateresvit_head_feature
                        in self.rgb_thermal_gateresvit_thermal_feature_set
                    ):
                        raise ValueError(
                            "PI05 thermal_encoder_channel='GateResViT'/'GateResandViT' requires "
                            "distinct RGB head and thermal features, got "
                            f"{self.rgb_thermal_gateresvit_head_feature!r}."
                        )
                    if image_embedding_dim % config.thermal_twogrey_gateresvit_num_heads != 0:
                        raise ValueError(
                            "thermal_twogrey_gateresvit_num_heads must divide the PI05 prefix "
                            f"embedding dimension ({image_embedding_dim})"
                        )
                    self.twogrey_gateresvit_fusion = TwoGreyGateResViTFusion(
                        embed_dim=image_embedding_dim,
                        num_heads=config.thermal_twogrey_gateresvit_num_heads,
                        hidden_dim=config.thermal_twogrey_gateresvit_hidden_dim,
                        dropout=config.thermal_encoder_dropout,
                        temperature=config.thermal_twogrey_gateresvit_temperature,
                        gate_init_std=config.thermal_twogrey_gateresvit_gate_init_std,
                    )
                    gate_res_mode_name = (
                        "GateResandViT"
                        if config.uses_gate_res_and_thermal_vit
                        else "GateResViT"
                    )
                    logging.info(
                        "PI05 %s uses text-gated cold/hot residuals on %s: "
                        "cold=%s, hot=%s, base_alpha=%.3f, base_beta=%.3f, "
                        "keeps_original_rgb=%s.",
                        gate_res_mode_name,
                        self.rgb_thermal_gateresvit_head_feature,
                        self.rgb_thermal_gateresvit_cold_feature,
                        self.rgb_thermal_gateresvit_hot_feature,
                        config.thermal_twogrey_resvit_cold_alpha,
                        config.thermal_twogrey_resvit_hot_beta,
                        config.uses_gate_res_and_thermal_vit,
                    )
                if config.uses_gate_two_residual_thermal_vit:
                    if self.rgb_thermal_gatetworesvit_head_feature not in config.image_features:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='GateTwoResViT' requires "
                            f"{self.rgb_thermal_gatetworesvit_head_feature!r} in image_features. "
                            f"Available image_features: {list(config.image_features)}."
                        )
                    twogrey_pairs = [
                        (cold_feature, hot_feature)
                        for cold_feature, hot_feature in self.thermal_twogrey_feature_map.values()
                        if cold_feature in self.thermal_image_feature_set
                        and hot_feature in self.thermal_image_feature_set
                    ]
                    if not twogrey_pairs:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='GateTwoResViT' requires generated "
                            "cold/hot thermal image features in image_features. "
                            f"Configured thermal features: {self.thermal_twogrey_feature_map}."
                        )
                    (
                        self.rgb_thermal_gatetworesvit_cold_feature,
                        self.rgb_thermal_gatetworesvit_hot_feature,
                    ) = twogrey_pairs[0]
                    self.rgb_thermal_gatetworesvit_thermal_features = [
                        feature for pair in twogrey_pairs for feature in pair
                    ]
                    self.rgb_thermal_gatetworesvit_thermal_feature_set = set(
                        self.rgb_thermal_gatetworesvit_thermal_features
                    )
                    if (
                        self.rgb_thermal_gatetworesvit_head_feature
                        in self.rgb_thermal_gatetworesvit_thermal_feature_set
                    ):
                        raise ValueError(
                            "PI05 thermal_encoder_channel='GateTwoResViT' requires distinct RGB head "
                            f"and thermal features, got {self.rgb_thermal_gatetworesvit_head_feature!r}."
                        )
                    if image_embedding_dim % config.thermal_twogrey_gateresvit_num_heads != 0:
                        raise ValueError(
                            "thermal_twogrey_gateresvit_num_heads must divide the PI05 prefix "
                            f"embedding dimension ({image_embedding_dim})"
                        )
                    self.twogrey_gatetworesvit_fusion = TwoGreyGateTwoResViTFusion(
                        embed_dim=image_embedding_dim,
                        num_heads=config.thermal_twogrey_gateresvit_num_heads,
                        hidden_dim=config.thermal_twogrey_gateresvit_hidden_dim,
                        dropout=config.thermal_encoder_dropout,
                        temperature=config.thermal_twogrey_gateresvit_temperature,
                        gate_init_std=config.thermal_twogrey_gateresvit_gate_init_std,
                    )
                    logging.info(
                        "PI05 GateTwoResViT emits two scalar text-gated head-RGB residual streams "
                        "on %s: cold=%s, hot=%s, base_alpha=%.3f, base_beta=%.3f.",
                        self.rgb_thermal_gatetworesvit_head_feature,
                        self.rgb_thermal_gatetworesvit_cold_feature,
                        self.rgb_thermal_gatetworesvit_hot_feature,
                        config.thermal_twogrey_resvit_cold_alpha,
                        config.thermal_twogrey_resvit_hot_beta,
                    )
                if config.uses_gate_one_residual_thermal_vit:
                    if self.rgb_thermal_gateoneresvit_head_feature not in config.image_features:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='GateOneResViT' requires "
                            f"{self.rgb_thermal_gateoneresvit_head_feature!r} in image_features. "
                            f"Available image_features: {list(config.image_features)}."
                        )
                    self.rgb_thermal_gateoneresvit_thermal_feature = self.thermal_image_features[0]
                    if (
                        self.rgb_thermal_gateoneresvit_thermal_feature
                        == self.rgb_thermal_gateoneresvit_head_feature
                    ):
                        raise ValueError(
                            "PI05 thermal_encoder_channel='GateOneResViT' requires distinct RGB head "
                            f"and thermal features, got {self.rgb_thermal_gateoneresvit_head_feature!r}."
                        )
                    if image_embedding_dim % config.thermal_grey_gateoneresvit_num_heads != 0:
                        raise ValueError(
                            "thermal_grey_gateoneresvit_num_heads must divide the PI05 prefix "
                            f"embedding dimension ({image_embedding_dim})"
                        )
                    self.grey_gateoneresvit_fusion = OneGreyGateOneResViTFusion(
                        embed_dim=image_embedding_dim,
                        num_heads=config.thermal_grey_gateoneresvit_num_heads,
                        hidden_dim=config.thermal_grey_gateoneresvit_hidden_dim,
                        dropout=config.thermal_encoder_dropout,
                        temperature=config.thermal_grey_gateoneresvit_temperature,
                        gate_init_std=config.thermal_grey_gateoneresvit_gate_init_std,
                    )
                    logging.info(
                        "PI05 GateOneResViT emits one scalar text-gated head-RGB residual stream "
                        "on %s: thermal=%s.",
                        self.rgb_thermal_gateoneresvit_head_feature,
                        self.rgb_thermal_gateoneresvit_thermal_feature,
                    )
                if config.uses_gate_residual_thermal_vit3:
                    if self.rgb_thermal_gateresvit3_head_feature not in config.image_features:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='GateResViT3' requires "
                            f"{self.rgb_thermal_gateresvit3_head_feature!r} in image_features. "
                            f"Available image_features: {list(config.image_features)}."
                        )
                    twogrey_pairs = [
                        (cold_feature, hot_feature)
                        for cold_feature, hot_feature in self.thermal_twogrey_feature_map.values()
                        if cold_feature in self.thermal_image_feature_set
                        and hot_feature in self.thermal_image_feature_set
                    ]
                    if not twogrey_pairs:
                        raise ValueError(
                            "PI05 thermal_encoder_channel='GateResViT3' requires generated cold/hot "
                            "thermal image features in image_features. "
                            f"Configured thermal features: {self.thermal_twogrey_feature_map}."
                        )
                    (
                        self.rgb_thermal_gateresvit3_cold_feature,
                        self.rgb_thermal_gateresvit3_hot_feature,
                    ) = twogrey_pairs[0]
                    self.rgb_thermal_gateresvit3_thermal_features = [
                        feature for pair in twogrey_pairs for feature in pair
                    ]
                    self.rgb_thermal_gateresvit3_thermal_feature_set = set(
                        self.rgb_thermal_gateresvit3_thermal_features
                    )
                    if (
                        self.rgb_thermal_gateresvit3_head_feature
                        in self.rgb_thermal_gateresvit3_thermal_feature_set
                    ):
                        raise ValueError(
                            "PI05 thermal_encoder_channel='GateResViT3' requires distinct RGB head and "
                            f"thermal features, got {self.rgb_thermal_gateresvit3_head_feature!r}."
                        )
                    if image_embedding_dim % config.thermal_twogrey_gateresvit3_num_heads != 0:
                        raise ValueError(
                            "thermal_twogrey_gateresvit3_num_heads must divide the PI05 prefix "
                            f"embedding dimension ({image_embedding_dim})"
                        )
                    self.twogrey_gateresvit3_fusion = TwoGreyGateResViT3Fusion(
                        embed_dim=image_embedding_dim,
                        num_heads=config.thermal_twogrey_gateresvit3_num_heads,
                        hidden_dim=config.thermal_twogrey_gateresvit3_hidden_dim,
                        stat_hidden_dim=config.thermal_twogrey_gateresvit3_stat_hidden_dim,
                        residual_hidden_dim=config.thermal_twogrey_gateresvit3_residual_hidden_dim,
                        dropout=config.thermal_encoder_dropout,
                        temperature=config.thermal_twogrey_gateresvit3_temperature,
                        gate_init_std=config.thermal_twogrey_gateresvit3_gate_init_std,
                        confidence_bias_init=config.thermal_twogrey_gateresvit3_confidence_bias_init,
                        evidence_gain=config.thermal_twogrey_gateresvit3_evidence_gain,
                        evidence_threshold=config.thermal_twogrey_gateresvit3_evidence_threshold,
                    )
                    logging.info(
                        "PI05 GateResViT3 uses text-gated cold/hot residuals and thermal-stat "
                        "confidence on %s: cold=%s, hot=%s, confidence_bias_init=%.3f, "
                        "evidence_gain=%.3f, evidence_threshold=%.3f.",
                        self.rgb_thermal_gateresvit3_head_feature,
                        self.rgb_thermal_gateresvit3_cold_feature,
                        self.rgb_thermal_gateresvit3_hot_feature,
                        config.thermal_twogrey_gateresvit3_confidence_bias_init,
                        config.thermal_twogrey_gateresvit3_evidence_gain,
                        config.thermal_twogrey_gateresvit3_evidence_threshold,
                    )
                if config.uses_resvit_attention_encoder:
                    if self.rgb_thermal_resvit_attention_head_feature not in config.image_features:
                        raise ValueError(
                            "PI05 thermal_input_type='ResViTAttention' requires "
                            f"{self.rgb_thermal_resvit_attention_head_feature!r} in image_features. "
                            f"Available image_features: {list(config.image_features)}."
                        )
                    self.rgb_thermal_resvit_attention_thermal_feature = self.thermal_image_features[0]
                    if (
                        self.rgb_thermal_resvit_attention_thermal_feature
                        == self.rgb_thermal_resvit_attention_head_feature
                    ):
                        raise ValueError(
                            "PI05 thermal_input_type='ResViTAttention' requires distinct RGB head "
                            f"and thermal features, got {self.rgb_thermal_resvit_attention_head_feature!r}."
                        )
                    if image_embedding_dim % config.thermal_resvit_attention_num_heads != 0:
                        raise ValueError(
                            "thermal_resvit_attention_num_heads must divide the PI05 prefix embedding "
                            f"dimension ({image_embedding_dim})"
                        )
                    self.resvit_attention_fusion = ResViTAttentionFusion(
                        embed_dim=image_embedding_dim,
                        num_heads=config.thermal_resvit_attention_num_heads,
                        merge_hidden_dim=config.thermal_resvit_attention_merge_hidden_dim,
                        residual_hidden_dim=config.thermal_resvit_attention_residual_hidden_dim,
                        dropout=config.thermal_encoder_dropout,
                    )
                if config.rgb_thermal_align_fusion:
                    self.rgb_thermal_align_thermal_feature = config.rgb_thermal_mix_thermal_feature
                    if config.thermal_fusion_head_rgb_feature not in config.image_features:
                        raise ValueError(
                            "PI05 rgb_thermal_align_fusion=true requires "
                            f"{config.thermal_fusion_head_rgb_feature!r} in image_features. "
                            f"Available image_features: {list(config.image_features)}."
                        )
                    if self.rgb_thermal_align_thermal_feature not in self.thermal_image_feature_set:
                        raise ValueError(
                            "PI05 rgb_thermal_align_fusion=true requires "
                            f"{self.rgb_thermal_align_thermal_feature!r} to be a configured "
                            f"thermal image feature. Configured thermal features: {self.thermal_image_features}."
                        )
                    if raw_image_token_dim % config.thermal_fusion_num_heads != 0:
                        raise ValueError(
                            "thermal_fusion_num_heads must divide the raw PI05 ViT token dimension "
                            f"({raw_image_token_dim})"
                        )
                    self.rgb_thermal_align_aligner = RGBThermalTokenAligner(
                        token_dim=raw_image_token_dim,
                        num_heads=config.thermal_fusion_num_heads,
                        dropout=config.thermal_encoder_dropout,
                    )
                    self.rgb_thermal_align_decoder = RGBThermalFusionDecoder(
                        token_dim=raw_image_token_dim,
                        output_size=config.image_resolution,
                    )
                    logging.info(
                        "PI05 RGB/thermal alignment fusion uses raw ViT token dim=%d "
                        "and projector output dim=%d.",
                        raw_image_token_dim,
                        image_embedding_dim,
                    )
        if config.uses_thermal_resnet18_encoder:
            if not self.thermal_image_features:
                logging.warning(
                    "PI05 thermal_encoder_channel='resnet18' but none of %s are present in image_features=%s. "
                    "The model will keep using the regular PI05 image path.",
                    config.thermal_image_features,
                    list(config.image_features),
                )
            else:
                thermal_key = self.thermal_image_features[0]
                thermal_feature = config.image_features[thermal_key]
                shape = tuple(thermal_feature.shape)
                thermal_channels = (
                    int(shape[0])
                    if shape and shape[0] in {1, 3, 4}
                    else int(shape[-1])
                    if shape and shape[-1] in {1, 3, 4}
                    else 3
                )
                self.thermal_encoder = ThermalResNet18Encoder(
                    in_channels=thermal_channels,
                    hidden_dim=config.thermal_encoder_hidden_dim,
                    output_dim=image_embedding_dim,
                    token_grid=config.thermal_encoder_token_grid,
                    dropout=config.thermal_encoder_dropout,
                )
                if config.thermal_fuse_with_head_rgb:
                    if config.thermal_fusion_head_rgb_feature not in config.image_features:
                        logging.warning(
                            "PI05 thermal_fuse_with_head_rgb=true but %s is not present in "
                            "image_features=%s. Thermal-head fusion will be skipped.",
                            config.thermal_fusion_head_rgb_feature,
                            list(config.image_features),
                        )
                    elif image_embedding_dim % config.thermal_fusion_num_heads != 0:
                        raise ValueError(
                            "thermal_fusion_num_heads must divide the PI05 prefix embedding dimension "
                            f"({image_embedding_dim})"
                        )
                    else:
                        self.thermal_head_fusion = ThermalHeadFusion(
                            embed_dim=image_embedding_dim,
                            num_heads=config.thermal_fusion_num_heads,
                            dropout=config.thermal_encoder_dropout,
                        )
        if config.uses_anythermal_encoder:
            if not self.thermal_image_features:
                raise ValueError(
                    "PI05 thermal_input_type='anythermal' requires at least one configured thermal "
                    f"image feature from {config.thermal_image_features}; available image_features: "
                    f"{list(config.image_features)}."
                )
            else:
                self.thermal_encoder = AnyThermalEncoder(
                    output_dim=image_embedding_dim,
                    model_type=config.thermal_anythermal_model_type,
                    checkpoint_path=config.thermal_anythermal_checkpoint_path,
                    dinov2_repo_path=config.thermal_anythermal_dinov2_repo_path,
                    freeze_backbone=config.thermal_anythermal_freeze_backbone,
                    include_cls_token=config.thermal_anythermal_include_cls_token,
                    include_register_tokens=config.thermal_anythermal_include_register_tokens,
                    input_range=config.thermal_anythermal_input_range,
                    dropout=config.thermal_encoder_dropout,
                )
        if config.uses_resthermal_encoder:
            if not self.thermal_image_features:
                raise ValueError(
                    "PI05 thermal_input_type='resthermal' requires at least one configured thermal "
                    f"image feature from {config.thermal_image_features}; available image_features: "
                    f"{list(config.image_features)}."
                )
            if self.rgb_thermal_residual_head_feature not in config.image_features:
                raise ValueError(
                    "PI05 thermal_input_type='resthermal' requires "
                    f"{self.rgb_thermal_residual_head_feature!r} in image_features. "
                    f"Available image_features: {list(config.image_features)}."
                )
            self.rgb_thermal_residual_thermal_feature = self.thermal_image_features[0]
            if self.rgb_thermal_residual_thermal_feature == self.rgb_thermal_residual_head_feature:
                raise ValueError(
                    "PI05 thermal_input_type='resthermal' requires distinct RGB head and thermal "
                    f"features, got {self.rgb_thermal_residual_head_feature!r}."
                )
            self.thermal_residual_fusion = AnyThermalResidualFusion(
                rgb_dim=image_embedding_dim,
                adapter_dim=config.thermal_resthermal_adapter_dim,
                decoder_hidden_dim=config.thermal_resthermal_decoder_hidden_dim,
                num_heads=config.thermal_resthermal_num_heads,
                model_type=config.thermal_anythermal_model_type,
                checkpoint_path=config.thermal_anythermal_checkpoint_path,
                dinov2_repo_path=config.thermal_anythermal_dinov2_repo_path,
                freeze_backbone=config.thermal_anythermal_freeze_backbone,
                input_range=config.thermal_anythermal_input_range,
                dropout=config.thermal_encoder_dropout,
            )
        if config.uses_thermal_cvae_encoder:
            if not self.thermal_image_features:
                logging.warning(
                    "PI05 thermal_encoder_channel='cvae' but none of %s are present in image_features=%s. "
                    "The model will keep using the regular PI05 image path.",
                    config.thermal_image_features,
                    list(config.image_features),
                )
            elif config.thermal_cvae_source_feature not in config.image_features:
                raise ValueError(
                    "PI05 thermal_encoder_channel='cvae' requires thermal_cvae_source_feature="
                    f"{config.thermal_cvae_source_feature!r} to be present in image_features. "
                    f"Available image_features: {list(config.image_features)}."
                )
            else:
                self.thermal_encoder = ThermalCVAEEncoder(
                    input_channels=config.thermal_cvae_input_channels,
                    condition_channels=config.thermal_cvae_condition_channels,
                    target_channels=config.thermal_cvae_target_channels,
                    latent_dim=config.thermal_cvae_latent_dim,
                    hidden_dims=config.thermal_cvae_hidden_dims,
                    image_size=config.thermal_cvae_image_size,
                    output_dim=image_embedding_dim,
                    token_grid_size=config.thermal_cvae_token_grid_size,
                    token_upsample_mode=config.thermal_cvae_token_upsample_mode,
                    token_norm=config.thermal_cvae_token_norm,
                    token_scale_init=config.thermal_cvae_token_scale_init,
                    checkpoint_path=(
                        config.thermal_cvae_pretrained_path or config.thermal_cvae_checkpoint_path
                    ),
                    checkpoint_strict=config.thermal_cvae_checkpoint_strict,
                )
                if config.thermal_cvae_freeze_encoder:
                    self.thermal_encoder.eval()
                    for param in self.thermal_encoder.parameters():
                        param.requires_grad = False

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        if self.rgb_threeblack_vision_tower is not None:
            self.rgb_threeblack_vision_tower.gradient_checkpointing = True
        if self.thermal_vision_tower is not None:
            self.thermal_vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        if self.rgb_threeblack_vision_tower is not None:
            self.rgb_threeblack_vision_tower.gradient_checkpointing = False
        if self.thermal_vision_tower is not None:
            self.thermal_vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def initialize_thermal_vit_from_rgb(self):
        """Initialize dedicated thermal ViT weights from the current RGB ViT path."""
        if self.thermal_vision_tower is None:
            return
        self.thermal_vision_tower.load_state_dict(
            self.paligemma_with_expert.paligemma.model.vision_tower.state_dict()
        )
        if self.thermal_multi_modal_projector is not None:
            self.thermal_multi_modal_projector.load_state_dict(
                self.paligemma_with_expert.paligemma.model.multi_modal_projector.state_dict()
            )

    def initialize_rgb_threeblack_vit_from_rgb(self) -> None:
        """Initialize the independent three-color tower from pretrained RGB weights."""
        if self.rgb_threeblack_vision_tower is None:
            return
        self.rgb_threeblack_vision_tower.load_state_dict(
            self.paligemma_with_expert.paligemma.model.vision_tower.state_dict()
        )
        if self.rgb_threeblack_multi_modal_projector is not None:
            self.rgb_threeblack_multi_modal_projector.load_state_dict(
                self.paligemma_with_expert.paligemma.model.multi_modal_projector.state_dict()
            )

    def _configure_thermal_vit_trainability(self) -> None:
        """Keep copied thermal modules trainable when only non-thermal inputs are frozen."""
        if self.thermal_vision_tower is None:
            return

        # The thermal ViT is deep-copied after PaliGemma is frozen, so it inherits
        # requires_grad=False and eval mode. Undo that only for the new thermal-
        # preserving finetuning mode. A dedicated thermal projector is also kept
        # trainable; a shared projector belongs to frozen PaliGemma and is untouched.
        if self.config.freeze_non_thermal_input:
            self.thermal_vision_tower.train()
            self.thermal_vision_tower.requires_grad_(True)
            if self.thermal_multi_modal_projector is not None:
                self.thermal_multi_modal_projector.train()
                self.thermal_multi_modal_projector.requires_grad_(True)

        # Existing stronger freeze modes retain their original behavior if flags
        # are combined: they freeze both RGB and thermal vision modules.
        if self.config.freeze_vision_encoder or self.config.train_expert_only:
            self.thermal_vision_tower.eval()
            self.thermal_vision_tower.requires_grad_(False)
            if self.thermal_multi_modal_projector is not None:
                self.thermal_multi_modal_projector.eval()
                self.thermal_multi_modal_projector.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if (
            self.thermal_vision_tower is not None
            and (self.config.freeze_vision_encoder or self.config.train_expert_only)
        ):
            self.thermal_vision_tower.eval()
            if self.thermal_multi_modal_projector is not None:
                self.thermal_multi_modal_projector.eval()
        if (
            getattr(self, "rgb_threeblack_vision_tower", None) is not None
            and (
                self.config.freeze_vision_encoder
                or self.config.train_expert_only
                or self.config.freeze_non_thermal_input
            )
        ):
            self.rgb_threeblack_vision_tower.eval()
            if self.rgb_threeblack_multi_modal_projector is not None:
                self.rgb_threeblack_multi_modal_projector.eval()
        if (
            self.config.uses_thermal_cvae_encoder
            and self.config.thermal_cvae_freeze_encoder
            and self.thermal_encoder is not None
        ):
            self.thermal_encoder.eval()
        if (
            self.config.uses_anythermal_encoder
            and self.config.thermal_anythermal_freeze_backbone
            and self.thermal_encoder is not None
        ):
            self.thermal_encoder.backbone.eval()
        if (
            self.config.uses_dinov2_rgb_encoder
            and (
                self.config.rgb_dinov2_freeze_backbone
                or self.config.freeze_non_thermal_input
            )
            and self.rgb_encoder is not None
        ):
            if self.config.freeze_non_thermal_input:
                self.rgb_encoder.eval()
            else:
                self.rgb_encoder.backbone.eval()
        if (
            self.config.uses_resthermal_encoder
            and self.config.thermal_anythermal_freeze_backbone
            and self.thermal_residual_fusion is not None
        ):
            self.thermal_residual_fusion.backbone.eval()
        return self

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _embed_rgb_image(self, image: Tensor) -> Tensor:
        if self.rgb_encoder is not None:
            return self.rgb_encoder(image)
        return self.paligemma_with_expert.embed_image(image)

    def _embed_rgb_threeblack_image(self, image: Tensor) -> Tensor:
        """Embed a color-derived view with the shared, RGB-independent color tower."""
        if (
            self.rgb_threeblack_vision_tower is None
            or self.rgb_threeblack_multi_modal_projector is None
        ):
            raise RuntimeError("RGB GateTwoResViT color ViT modules are not initialized.")
        return self.paligemma_with_expert.embed_image_with_modules(
            image,
            self.rgb_threeblack_vision_tower,
            self.rgb_threeblack_multi_modal_projector,
        )

    def _prepare_thermal_vit_image(self, image: Tensor) -> Tensor:
        if image.ndim != 4:
            return image
        if image.shape[1] == 1:
            return image.repeat(1, 3, 1, 1).contiguous()
        if image.shape[1] == 4:
            return image[:, :3].contiguous()
        return image

    def _project_siglip_tokens_with_rgb_projector(
        self,
        image_tokens: Tensor,
        out_dtype: torch.dtype,
    ) -> Tensor:
        return self.paligemma_with_expert.project_image_tokens_with_modules(
            image_tokens,
            self.paligemma_with_expert.paligemma.model.multi_modal_projector,
            out_dtype=out_dtype,
        )

    def _embed_shared_projector_thermal_vit_image(self, thermal_img: Tensor) -> Tensor:
        if self.thermal_vision_tower is None:
            raise RuntimeError("Shared-projector thermal ViT module is not initialized.")

        thermal_img = self._prepare_thermal_vit_image(thermal_img)
        out_dtype = thermal_img.dtype
        thermal_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            thermal_img,
            self.thermal_vision_tower,
        )
        return self._project_siglip_tokens_with_rgb_projector(thermal_tokens, out_dtype=out_dtype)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def _apply_gatevitmix_prefix_attention_scope(
        self,
        att_2d_masks: Tensor,
        *,
        prefix_len: int,
    ) -> Tensor:
        """Isolate GateViTMix thermal tokens inside the prefix while keeping suffix access."""
        if not getattr(self.config, "uses_gate_thermal_vit_mix", False):
            return att_2d_masks
        if att_2d_masks.ndim != 3 or prefix_len <= 0:
            return att_2d_masks
        if prefix_len > att_2d_masks.shape[-2] or prefix_len > att_2d_masks.shape[-1]:
            raise ValueError(
                "GateViTMix prefix attention scope received an invalid prefix length: "
                f"prefix_len={prefix_len}, mask_shape={tuple(att_2d_masks.shape)}."
            )

        thermal_positions = torch.zeros(
            prefix_len,
            dtype=torch.bool,
            device=att_2d_masks.device,
        )
        has_thermal_positions = False
        for segment in self._last_prefix_token_layout:
            if segment.get("kind") != "image_gatevitmix_twogrey":
                continue
            start = max(0, int(segment.get("start", 0)))
            end = min(prefix_len, int(segment.get("end", 0)))
            if end > start:
                thermal_positions[start:end] = True
                has_thermal_positions = True
        if not has_thermal_positions:
            return att_2d_masks

        same_scope = thermal_positions[:, None] == thermal_positions[None, :]
        scoped_masks = att_2d_masks.clone()
        scoped_masks[:, :prefix_len, :prefix_len] &= same_scope[None, :, :]
        return scoped_masks

    def _apply_gateactionvit_action_attention_bias(
        self,
        attention_mask_4d: Tensor,
        *,
        prefix_len: int,
        suffix_queries_only: bool = False,
    ) -> Tensor:
        """Bias only action queries toward the selected cold/hot token stream."""
        if not getattr(self.config, "uses_gate_action_thermal_vit", False):
            return attention_mask_4d
        attention_bias = getattr(self, "_current_gateactionvit_attention_bias", None)
        if attention_bias is None:
            return attention_mask_4d
        if attention_mask_4d.ndim != 4 or attention_mask_4d.shape[1] != 1:
            raise ValueError(
                "GateActionViT expects an additive attention mask with shape [B, 1, Q, K], "
                f"got {tuple(attention_mask_4d.shape)}."
            )
        if attention_bias.ndim != 2 or attention_bias.shape[1] != 2:
            raise ValueError(
                "GateActionViT cold/hot attention bias must have shape [B, 2], "
                f"got {tuple(attention_bias.shape)}."
            )

        batch_size, _, query_len, key_len = attention_mask_4d.shape
        if attention_bias.shape[0] != batch_size:
            raise ValueError(
                "GateActionViT attention bias batch size does not match the attention mask: "
                f"bias={attention_bias.shape[0]}, mask={batch_size}."
            )
        if prefix_len <= 0 or prefix_len > key_len:
            raise ValueError(
                "GateActionViT received an invalid prefix length: "
                f"prefix_len={prefix_len}, mask_shape={tuple(attention_mask_4d.shape)}."
            )

        spans = {}
        for segment in self._last_prefix_token_layout:
            kind = segment.get("kind")
            if kind == "image_gateactionvit_twogrey_cold":
                spans["cold"] = (int(segment["start"]), int(segment["end"]))
            elif kind == "image_gateactionvit_twogrey_hot":
                spans["hot"] = (int(segment["start"]), int(segment["end"]))
        if set(spans) != {"cold", "hot"}:
            raise RuntimeError(
                "GateActionViT could not find both cold and hot token spans in the prefix layout."
            )

        key_bias = attention_mask_4d.new_zeros((batch_size, key_len))
        live_bias = attention_bias.to(
            device=attention_mask_4d.device,
            dtype=attention_mask_4d.dtype,
        )
        for bias_index, name in enumerate(("cold", "hot")):
            start, end = spans[name]
            if start < 0 or end <= start or end > prefix_len:
                raise ValueError(
                    f"GateActionViT {name} token span {(start, end)} is outside "
                    f"prefix_len={prefix_len}."
                )
            key_bias[:, start:end] = live_bias[:, bias_index, None]

        query_start = 0 if suffix_queries_only else prefix_len
        if query_start >= query_len:
            return attention_mask_4d
        action_queries = attention_mask_4d.new_zeros(query_len)
        action_queries[query_start:] = 1.0
        return attention_mask_4d + (
            action_queries[None, None, :, None] * key_bias[:, None, None, :]
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def pop_last_attention_map(self):
        attention_map = self._last_attention_map
        self._last_attention_map = None
        return attention_map

    def _capture_attention_map(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        tokens,
        masks,
        x_t,
        timestep,
        denoise_step_idx,
        num_steps,
    ) -> None:
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        prefix_len = int(prefix_pad_masks.shape[1])
        att_2d_masks = self._apply_gatevitmix_prefix_attention_scope(
            att_2d_masks,
            prefix_len=prefix_len,
        )
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        att_2d_masks_4d = self._apply_gateactionvit_action_attention_bias(
            att_2d_masks_4d,
            prefix_len=prefix_len,
        )

        suffix_len = int(suffix_pad_masks.shape[1])
        attention_collector = {
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "total_len": int(prefix_len + suffix_len),
            "denoise_step": int(denoise_step_idx),
            "num_inference_steps": int(num_steps),
            "prefix_token_layout": deepcopy(self._last_prefix_token_layout),
        }

        self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            attention_collector=attention_collector,
        )

        layers = attention_collector.get("layers", [])
        if layers:
            attention_collector["token_ids"] = tokens.detach().cpu()
            attention_collector["token_mask"] = masks.detach().cpu()
            self._last_attention_map = attention_collector

    def _embed_rgb_thermal_aligned_image(
        self,
        rgb_img: Tensor,
        thermal_img: Tensor,
        rgb_img_mask: Tensor,
        thermal_img_mask: Tensor,
        rgb_thermal_align_target: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if (
            self.rgb_thermal_align_aligner is None
            or self.rgb_thermal_align_decoder is None
            or self.thermal_vision_tower is None
        ):
            raise RuntimeError("RGB/thermal alignment fusion modules are not initialized.")

        out_dtype = rgb_img.dtype
        rgb_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            rgb_img,
            self.paligemma_with_expert.paligemma.model.vision_tower,
        )
        thermal_img = self._prepare_thermal_vit_image(thermal_img)
        thermal_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            thermal_img,
            self.thermal_vision_tower,
        )

        batch_size, num_rgb_tokens = rgb_tokens.shape[:2]
        num_thermal_tokens = thermal_tokens.shape[1]
        rgb_token_mask = rgb_img_mask[:, None].expand(batch_size, num_rgb_tokens)
        thermal_token_mask = thermal_img_mask[:, None].expand(batch_size, num_thermal_tokens)
        fused_tokens, fused_token_mask = self.rgb_thermal_align_aligner(
            rgb_tokens=rgb_tokens,
            thermal_tokens=thermal_tokens,
            rgb_token_mask=rgb_token_mask,
            thermal_token_mask=thermal_token_mask,
        )

        if self.training and rgb_thermal_align_target is not None:
            pred = self.rgb_thermal_align_decoder(fused_tokens)
            target = rgb_thermal_align_target.to(device=pred.device, dtype=pred.dtype)
            l1_loss = F.l1_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
            mse_loss = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
            valid_loss_mask = (rgb_img_mask & thermal_img_mask).to(device=pred.device)
            l1_loss = l1_loss.masked_fill(~valid_loss_mask, 0.0)
            mse_loss = mse_loss.masked_fill(~valid_loss_mask, 0.0)
            self._last_rgb_thermal_align_l1_loss = l1_loss
            self._last_rgb_thermal_align_mse_loss = mse_loss
            self._last_rgb_thermal_align_loss = l1_loss + 0.25 * mse_loss

        fused_emb = self.paligemma_with_expert.project_image_tokens_with_modules(
            fused_tokens,
            self.paligemma_with_expert.paligemma.model.multi_modal_projector,
            out_dtype=out_dtype,
        )
        return fused_emb, fused_token_mask

    def _embed_resvit_head_image(
        self,
        rgb_img: Tensor,
        thermal_img: Tensor,
        rgb_img_mask: Tensor,
        thermal_img_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.thermal_vision_tower is None:
            raise RuntimeError("Residual thermal ViT module is not initialized.")

        out_dtype = rgb_img.dtype
        rgb_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            rgb_img,
            self.paligemma_with_expert.paligemma.model.vision_tower,
        )
        thermal_img = self._prepare_thermal_vit_image(thermal_img)
        thermal_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            thermal_img,
            self.thermal_vision_tower,
        )
        if rgb_tokens.shape != thermal_tokens.shape:
            raise ValueError(
                "ResViT requires head RGB and thermal ViT raw tokens to have the same shape, "
                f"got rgb={tuple(rgb_tokens.shape)} and thermal={tuple(thermal_tokens.shape)}."
            )

        batch_size, num_rgb_tokens = rgb_tokens.shape[:2]
        thermal_valid = thermal_img_mask.to(device=thermal_tokens.device, dtype=torch.bool)
        thermal_tokens = thermal_tokens * thermal_valid[:, None, None].to(dtype=thermal_tokens.dtype)
        fused_tokens = rgb_tokens + thermal_tokens
        fused_emb = self._project_siglip_tokens_with_rgb_projector(fused_tokens, out_dtype=out_dtype)
        rgb_token_mask = rgb_img_mask[:, None].expand(batch_size, num_rgb_tokens)
        return fused_emb, rgb_token_mask

    def _embed_twogrey_resvit_head_image(
        self,
        rgb_img: Tensor,
        cold_img: Tensor,
        hot_img: Tensor,
        rgb_img_mask: Tensor,
        cold_img_mask: Tensor,
        hot_img_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.thermal_vision_tower is None:
            raise RuntimeError("TwoGrey residual thermal ViT module is not initialized.")

        out_dtype = rgb_img.dtype
        rgb_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            rgb_img,
            self.paligemma_with_expert.paligemma.model.vision_tower,
        )
        cold_img = self._prepare_thermal_vit_image(cold_img)
        hot_img = self._prepare_thermal_vit_image(hot_img)
        cold_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            cold_img,
            self.thermal_vision_tower,
        )
        hot_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            hot_img,
            self.thermal_vision_tower,
        )
        if rgb_tokens.shape != cold_tokens.shape or rgb_tokens.shape != hot_tokens.shape:
            raise ValueError(
                "TwoGrey ResViT requires RGB, cold thermal, and hot thermal raw ViT tokens "
                "to have the same shape, got "
                f"rgb={tuple(rgb_tokens.shape)}, cold={tuple(cold_tokens.shape)}, "
                f"hot={tuple(hot_tokens.shape)}."
            )

        batch_size, num_rgb_tokens = rgb_tokens.shape[:2]
        cold_valid = cold_img_mask.to(device=cold_tokens.device, dtype=torch.bool)
        hot_valid = hot_img_mask.to(device=hot_tokens.device, dtype=torch.bool)
        cold_tokens = cold_tokens * cold_valid[:, None, None].to(dtype=cold_tokens.dtype)
        hot_tokens = hot_tokens * hot_valid[:, None, None].to(dtype=hot_tokens.dtype)

        cold_alpha = float(getattr(self.config, "thermal_twogrey_resvit_cold_alpha", 1.0))
        hot_beta = float(getattr(self.config, "thermal_twogrey_resvit_hot_beta", 1.0))
        fused_tokens = rgb_tokens + cold_alpha * cold_tokens + hot_beta * hot_tokens
        fused_emb = self._project_siglip_tokens_with_rgb_projector(fused_tokens, out_dtype=out_dtype)
        rgb_token_mask = rgb_img_mask[:, None].expand(batch_size, num_rgb_tokens)
        return fused_emb, rgb_token_mask

    @staticmethod
    def _summarize_gate_tensor(values: Tensor) -> dict[str, Any]:
        flat = values.detach().to(dtype=torch.float32, device="cpu").flatten()
        if flat.numel() == 0:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "shape": list(values.shape), "values": []}
        return {
            "mean": round(float(flat.mean().item()), 6),
            "min": round(float(flat.min().item()), 6),
            "max": round(float(flat.max().item()), 6),
            "shape": list(values.shape),
            "values": [round(float(value), 6) for value in flat.tolist()],
        }

    def _summarize_twogrey_gateresvit_gate(
        self,
        gate_info: dict[str, Tensor],
    ) -> dict[str, Any] | None:
        # JSON summaries contain CPU transfers and scalar extraction. Keeping
        # them out of a compiled training graph avoids cudagraph graph breaks
        # and large private CUDA pools; eval disables compile and still records
        # the complete maps.
        if torch.compiler.is_compiling():
            return None

        summary = {}
        for key in (
            "cold_alpha",
            "hot_beta",
            "thermal_alpha",
            "cold_multiplier",
            "hot_multiplier",
            "thermal_weight",
            "logits",
            "cold_weight",
            "hot_weight",
            "hard_cold_weight",
            "hard_hot_weight",
            "soft_cold_weight",
            "soft_hot_weight",
            "routing_margin",
            "patch_relevance",
            "temperature_logits",
            "relevance_logits",
            "global_text_logits",
            "cold_evidence",
            "hot_evidence",
            "small_patch_cold_weight",
            "small_patch_hot_weight",
            "small_patch_hard_cold_weight",
            "small_patch_hard_hot_weight",
            "small_patch_soft_cold_weight",
            "small_patch_soft_hot_weight",
            "small_patch_relevance",
            "small_patch_temperature_logits",
            "small_patch_relevance_logits",
            "small_patch_cold_evidence",
            "small_patch_hot_evidence",
            "cold_residual_scale",
            "hot_residual_scale",
            "thermal_residual_scale",
            "cold_attention_bias",
            "hot_attention_bias",
            "attention_bias_strength",
            "text_logits",
            "rgb_logits",
            "text_cold_weight",
            "text_hot_weight",
            "rgb_cold_alignment_weight",
            "rgb_hot_alignment_weight",
            "rgb_gate_strength",
            "thermal_confidence",
            "effective_thermal_confidence",
            "confidence_logit",
            "learned_confidence_logit",
            "thermal_evidence",
            "evidence_logit",
            "thermal_stats",
            "head_mix_context_scale",
            "head_mix_context_rms",
            "head_mix_delta_rms",
        ):
            if key in gate_info:
                summary[key] = self._summarize_gate_tensor(gate_info[key])
        for key in (
            "cold_valid",
            "hot_valid",
            "thermal_valid",
            "text_valid",
            "rgb_valid",
            "rgb_cold_valid",
            "rgb_hot_valid",
            "head_mix_valid",
            "cold_residual_valid",
            "hot_residual_valid",
            "residual_valid",
        ):
            if key not in gate_info:
                continue
            valid = gate_info[key].detach().to(dtype=torch.bool, device="cpu").flatten()
            summary[f"{key}_count"] = int(valid.sum().item())
        return summary

    def _summarize_rgb_threeblack_gate(
        self,
        gate_info: dict[str, Tensor],
    ) -> dict[str, Any] | None:
        if torch.compiler.is_compiling():
            return None

        summary = {}
        for color_name in ("red", "green", "blue"):
            for suffix in ("alpha", "multiplier", "residual_scale"):
                key = f"{color_name}_{suffix}"
                if key in gate_info:
                    summary[key] = self._summarize_gate_tensor(gate_info[key])
        if "logits" in gate_info:
            summary["logits"] = self._summarize_gate_tensor(gate_info["logits"])
        for key in (
            "red_valid",
            "green_valid",
            "blue_valid",
            "text_valid",
            "red_residual_valid",
            "green_residual_valid",
            "blue_residual_valid",
        ):
            if key not in gate_info:
                continue
            valid = gate_info[key].detach().to(dtype=torch.bool, device="cpu").flatten()
            summary[f"{key}_count"] = int(valid.sum().item())
        return summary

    def _embed_rgb_threeblack_gatetworesvit_head_images(
        self,
        *,
        rgb_img: Tensor,
        rgb_img_mask: Tensor,
        text_tokens: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        fusion = self.rgb_threeblack_gatetworesvit_fusion
        if fusion is None:
            raise RuntimeError("RGB GateTwoResViT fusion module is not initialized.")

        rgb_emb = self._apply_checkpoint(self._embed_rgb_image, rgb_img)
        red_img, green_img, blue_img = rgb_to_three_black_dominance_tensor(
            rgb_img,
            feature_name=self.rgb_threeblack_gatetworesvit_head_feature,
            input_range="minus_one_to_one",
        )
        red_emb = self._apply_checkpoint(self._embed_rgb_threeblack_image, red_img)
        green_emb = self._apply_checkpoint(self._embed_rgb_threeblack_image, green_img)
        blue_emb = self._apply_checkpoint(self._embed_rgb_threeblack_image, blue_img)
        for color_name, color_emb in (
            ("red", red_emb),
            ("green", green_emb),
            ("blue", blue_emb),
        ):
            if color_emb.shape != rgb_emb.shape:
                raise ValueError(
                    "RGB GateTwoResViT requires original and three-black projected token grids "
                    f"to match, got rgb={tuple(rgb_emb.shape)} and "
                    f"{color_name}={tuple(color_emb.shape)}."
                )

        batch_size, num_tokens = rgb_emb.shape[:2]
        rgb_token_mask = rgb_img_mask[:, None].expand(batch_size, num_tokens)
        (
            red_res_emb,
            red_res_token_mask,
            green_res_emb,
            green_res_token_mask,
            blue_res_emb,
            blue_res_token_mask,
            gate_info,
        ) = fusion(
            rgb_tokens=rgb_emb,
            red_tokens=red_emb,
            green_tokens=green_emb,
            blue_tokens=blue_emb,
            text_tokens=text_tokens,
            rgb_token_mask=rgb_token_mask,
            red_token_mask=rgb_token_mask,
            green_token_mask=rgb_token_mask,
            blue_token_mask=rgb_token_mask,
            text_token_mask=text_token_mask,
            base_red_alpha=self.config.rgb_threeblack_red_alpha,
            base_green_alpha=self.config.rgb_threeblack_green_alpha,
            base_blue_alpha=self.config.rgb_threeblack_blue_alpha,
        )
        self._last_rgb_threeblack_gatetworesvit_gate_summary = (
            self._summarize_rgb_threeblack_gate(gate_info)
        )
        return (
            red_res_emb,
            red_res_token_mask,
            green_res_emb,
            green_res_token_mask,
            blue_res_emb,
            blue_res_token_mask,
        )

    def _embed_twogrey_gatevit_image(
        self,
        *,
        cold_img: Tensor,
        hot_img: Tensor,
        cold_img_mask: Tensor,
        hot_img_mask: Tensor,
        text_tokens: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.thermal_vision_tower is None or self.twogrey_gatevit_fusion is None:
            raise RuntimeError("GateViT modules are not initialized.")

        cold_img = self._prepare_thermal_vit_image(cold_img)
        hot_img = self._prepare_thermal_vit_image(hot_img)
        out_dtype = cold_img.dtype
        cold_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            cold_img,
            self.thermal_vision_tower,
        )
        hot_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            hot_img,
            self.thermal_vision_tower,
        )
        cold_emb = self._project_siglip_tokens_with_rgb_projector(
            cold_raw_tokens,
            out_dtype=out_dtype,
        )
        hot_emb = self._project_siglip_tokens_with_rgb_projector(
            hot_raw_tokens,
            out_dtype=out_dtype,
        )
        if cold_emb.shape != hot_emb.shape:
            raise ValueError(
                "GateViT requires cold and hot projected tokens to have the same shape, "
                f"got cold={tuple(cold_emb.shape)} and hot={tuple(hot_emb.shape)}."
            )

        batch_size, num_thermal_tokens = cold_emb.shape[:2]
        cold_token_mask = cold_img_mask[:, None].expand(batch_size, num_thermal_tokens)
        hot_token_mask = hot_img_mask[:, None].expand(batch_size, num_thermal_tokens)
        fused_emb, fused_token_mask, gate_info = self.twogrey_gatevit_fusion(
            cold_tokens=cold_emb,
            hot_tokens=hot_emb,
            text_tokens=text_tokens,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
        )
        self._last_twogrey_gatevit_gate_summary = self._summarize_twogrey_gateresvit_gate(
            gate_info
        )
        return fused_emb, fused_token_mask

    def _embed_twogrey_gateactionvit_images(
        self,
        *,
        cold_img: Tensor,
        hot_img: Tensor,
        cold_img_mask: Tensor,
        hot_img_mask: Tensor,
        text_tokens: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.thermal_vision_tower is None or self.twogrey_gateactionvit_fusion is None:
            raise RuntimeError("GateActionViT modules are not initialized.")

        cold_img = self._prepare_thermal_vit_image(cold_img)
        hot_img = self._prepare_thermal_vit_image(hot_img)
        out_dtype = cold_img.dtype
        cold_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            cold_img,
            self.thermal_vision_tower,
        )
        hot_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            hot_img,
            self.thermal_vision_tower,
        )
        cold_emb = self._project_siglip_tokens_with_rgb_projector(
            cold_raw_tokens,
            out_dtype=out_dtype,
        )
        hot_emb = self._project_siglip_tokens_with_rgb_projector(
            hot_raw_tokens,
            out_dtype=out_dtype,
        )
        if cold_emb.shape != hot_emb.shape:
            raise ValueError(
                "GateActionViT requires cold and hot projected tokens to have the same shape, "
                f"got cold={tuple(cold_emb.shape)} and hot={tuple(hot_emb.shape)}."
            )

        batch_size, num_thermal_tokens = cold_emb.shape[:2]
        cold_token_mask = cold_img_mask[:, None].expand(batch_size, num_thermal_tokens)
        hot_token_mask = hot_img_mask[:, None].expand(batch_size, num_thermal_tokens)
        (
            cold_emb,
            cold_token_mask,
            hot_emb,
            hot_token_mask,
            attention_bias,
            gate_info,
        ) = self.twogrey_gateactionvit_fusion(
            cold_tokens=cold_emb,
            hot_tokens=hot_emb,
            text_tokens=text_tokens,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
        )
        self._current_gateactionvit_attention_bias = attention_bias
        self._last_twogrey_gateactionvit_gate_summary = (
            self._summarize_twogrey_gateresvit_gate(gate_info)
        )
        return cold_emb, cold_token_mask, hot_emb, hot_token_mask

    def _embed_twogrey_doublegatevit_head_and_thermal(
        self,
        *,
        rgb_img: Tensor,
        cold_img: Tensor,
        hot_img: Tensor,
        rgb_img_mask: Tensor,
        cold_img_mask: Tensor,
        hot_img_mask: Tensor,
        text_tokens: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.thermal_vision_tower is None or self.twogrey_doublegatevit_fusion is None:
            raise RuntimeError("DoubleGateViT modules are not initialized.")

        rgb_emb = self._embed_rgb_image(rgb_img)
        cold_img = self._prepare_thermal_vit_image(cold_img)
        hot_img = self._prepare_thermal_vit_image(hot_img)
        out_dtype = rgb_emb.dtype
        cold_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            cold_img,
            self.thermal_vision_tower,
        )
        hot_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            hot_img,
            self.thermal_vision_tower,
        )
        cold_emb = self._project_siglip_tokens_with_rgb_projector(
            cold_raw_tokens,
            out_dtype=out_dtype,
        )
        hot_emb = self._project_siglip_tokens_with_rgb_projector(
            hot_raw_tokens,
            out_dtype=out_dtype,
        )
        if cold_emb.shape != hot_emb.shape:
            raise ValueError(
                "DoubleGateViT requires cold and hot projected tokens to have the same shape, "
                f"got cold={tuple(cold_emb.shape)} and hot={tuple(hot_emb.shape)}."
            )

        batch_size, num_rgb_tokens = rgb_emb.shape[:2]
        num_thermal_tokens = cold_emb.shape[1]
        rgb_token_mask = rgb_img_mask[:, None].expand(batch_size, num_rgb_tokens)
        cold_token_mask = cold_img_mask[:, None].expand(batch_size, num_thermal_tokens)
        hot_token_mask = hot_img_mask[:, None].expand(batch_size, num_thermal_tokens)
        fused_emb, fused_token_mask, gate_info = self.twogrey_doublegatevit_fusion(
            rgb_tokens=rgb_emb,
            cold_tokens=cold_emb,
            hot_tokens=hot_emb,
            text_tokens=text_tokens,
            rgb_token_mask=rgb_token_mask,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
        )
        self._last_twogrey_doublegatevit_gate_summary = (
            self._summarize_twogrey_gateresvit_gate(gate_info)
        )
        return rgb_emb, rgb_token_mask, fused_emb, fused_token_mask

    def _embed_twogrey_doublegatetworesvit_head_images(
        self,
        *,
        rgb_img: Tensor,
        cold_img: Tensor,
        hot_img: Tensor,
        rgb_img_mask: Tensor,
        cold_img_mask: Tensor,
        hot_img_mask: Tensor,
        text_tokens: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if (
            self.thermal_vision_tower is None
            or self.twogrey_doublegatetworesvit_fusion is None
        ):
            raise RuntimeError("DoubleGateTwoResViT modules are not initialized.")

        rgb_emb = self._embed_rgb_image(rgb_img)
        cold_img = self._prepare_thermal_vit_image(cold_img)
        hot_img = self._prepare_thermal_vit_image(hot_img)
        out_dtype = rgb_emb.dtype
        cold_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            cold_img,
            self.thermal_vision_tower,
        )
        hot_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            hot_img,
            self.thermal_vision_tower,
        )
        cold_emb = self._project_siglip_tokens_with_rgb_projector(
            cold_raw_tokens,
            out_dtype=out_dtype,
        )
        hot_emb = self._project_siglip_tokens_with_rgb_projector(
            hot_raw_tokens,
            out_dtype=out_dtype,
        )
        if rgb_emb.shape != cold_emb.shape or rgb_emb.shape != hot_emb.shape:
            raise ValueError(
                "DoubleGateTwoResViT requires RGB, cold thermal, and hot thermal projected "
                "tokens to have the same shape, got "
                f"rgb={tuple(rgb_emb.shape)}, cold={tuple(cold_emb.shape)}, "
                f"hot={tuple(hot_emb.shape)}."
            )

        batch_size, num_tokens = rgb_emb.shape[:2]
        rgb_token_mask = rgb_img_mask[:, None].expand(batch_size, num_tokens)
        cold_token_mask = cold_img_mask[:, None].expand(batch_size, num_tokens)
        hot_token_mask = hot_img_mask[:, None].expand(batch_size, num_tokens)
        (
            cold_res_emb,
            cold_res_token_mask,
            hot_res_emb,
            hot_res_token_mask,
            gate_info,
        ) = self.twogrey_doublegatetworesvit_fusion(
            rgb_tokens=rgb_emb,
            cold_tokens=cold_emb,
            hot_tokens=hot_emb,
            text_tokens=text_tokens,
            rgb_token_mask=rgb_token_mask,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
            base_cold_alpha=self.config.thermal_twogrey_resvit_cold_alpha,
            base_hot_beta=self.config.thermal_twogrey_resvit_hot_beta,
        )
        self._last_twogrey_doublegatetworesvit_gate_summary = (
            self._summarize_twogrey_gateresvit_gate(gate_info)
        )
        return cold_res_emb, cold_res_token_mask, hot_res_emb, hot_res_token_mask

    def _embed_twogrey_patchgatevit_head_and_thermal(
        self,
        *,
        rgb_img: Tensor,
        cold_img: Tensor,
        hot_img: Tensor,
        rgb_img_mask: Tensor,
        cold_img_mask: Tensor,
        hot_img_mask: Tensor,
        text_tokens: Tensor,
        text_token_mask: Tensor,
        capture_gate_summary: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.thermal_vision_tower is None or self.twogrey_patchgatevit_fusion is None:
            raise RuntimeError("PatchGateViT modules are not initialized.")

        rgb_emb = self._embed_rgb_image(rgb_img)
        cold_img = self._prepare_thermal_vit_image(cold_img)
        hot_img = self._prepare_thermal_vit_image(hot_img)
        out_dtype = rgb_emb.dtype
        cold_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            cold_img,
            self.thermal_vision_tower,
        )
        hot_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            hot_img,
            self.thermal_vision_tower,
        )
        cold_emb = self._project_siglip_tokens_with_rgb_projector(
            cold_raw_tokens,
            out_dtype=out_dtype,
        )
        hot_emb = self._project_siglip_tokens_with_rgb_projector(
            hot_raw_tokens,
            out_dtype=out_dtype,
        )
        if cold_emb.shape != hot_emb.shape:
            raise ValueError(
                "PatchGateViT requires cold and hot projected tokens to have the same shape, "
                f"got cold={tuple(cold_emb.shape)} and hot={tuple(hot_emb.shape)}."
            )

        batch_size, num_rgb_tokens = rgb_emb.shape[:2]
        num_thermal_tokens = cold_emb.shape[1]
        rgb_token_mask = rgb_img_mask[:, None].expand(batch_size, num_rgb_tokens)
        cold_token_mask = cold_img_mask[:, None].expand(batch_size, num_thermal_tokens)
        hot_token_mask = hot_img_mask[:, None].expand(batch_size, num_thermal_tokens)
        cold_evidence = None
        hot_evidence = None
        if not self.config.uses_hard_small_patch_gate_thermal_vit:
            cold_evidence, hot_evidence = twogrey_patch_evidence(
                cold_img,
                hot_img,
                target_background=self.config.thermal_grey_background_value,
                token_count=num_thermal_tokens,
                black_importance=(
                    self.config.thermal_input_type in BLACK_IMPORTANCE_THERMAL_INPUT_TYPES
                ),
            )
        fused_emb, fused_token_mask, gate_info = self.twogrey_patchgatevit_fusion(
            rgb_tokens=rgb_emb,
            cold_tokens=cold_emb,
            hot_tokens=hot_emb,
            text_tokens=text_tokens,
            rgb_token_mask=rgb_token_mask,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
            cold_evidence=cold_evidence,
            hot_evidence=hot_evidence,
        )
        self._last_twogrey_patchgatevit_gate_summary = (
            self._summarize_twogrey_gateresvit_gate(gate_info)
            if capture_gate_summary
            else None
        )
        return rgb_emb, rgb_token_mask, fused_emb, fused_token_mask

    def _embed_twogrey_patchsinglegateresvit_head_image(
        self,
        *,
        rgb_img: Tensor,
        cold_img: Tensor,
        hot_img: Tensor,
        rgb_img_mask: Tensor,
        cold_img_mask: Tensor,
        hot_img_mask: Tensor,
        text_tokens: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if (
            self.thermal_vision_tower is None
            or self.twogrey_patchsinglegateresvit_fusion is None
        ):
            raise RuntimeError("PatchSingleGateResViT modules are not initialized.")

        rgb_emb = self._embed_rgb_image(rgb_img)
        cold_img = self._prepare_thermal_vit_image(cold_img)
        hot_img = self._prepare_thermal_vit_image(hot_img)
        out_dtype = rgb_emb.dtype
        cold_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            cold_img,
            self.thermal_vision_tower,
        )
        hot_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            hot_img,
            self.thermal_vision_tower,
        )
        cold_emb = self._project_siglip_tokens_with_rgb_projector(
            cold_raw_tokens,
            out_dtype=out_dtype,
        )
        hot_emb = self._project_siglip_tokens_with_rgb_projector(
            hot_raw_tokens,
            out_dtype=out_dtype,
        )
        if rgb_emb.shape != cold_emb.shape or rgb_emb.shape != hot_emb.shape:
            raise ValueError(
                "PatchSingleGateResViT requires aligned projected head RGB, cold, and hot "
                "token grids, got "
                f"rgb={tuple(rgb_emb.shape)}, cold={tuple(cold_emb.shape)}, "
                f"hot={tuple(hot_emb.shape)}."
            )

        batch_size, num_tokens = rgb_emb.shape[:2]
        rgb_token_mask = rgb_img_mask[:, None].expand(batch_size, num_tokens)
        cold_token_mask = cold_img_mask[:, None].expand(batch_size, num_tokens)
        hot_token_mask = hot_img_mask[:, None].expand(batch_size, num_tokens)
        cold_evidence, hot_evidence = twogrey_patch_evidence(
            cold_img,
            hot_img,
            target_background=self.config.thermal_grey_background_value,
            token_count=num_tokens,
            black_importance=(
                self.config.thermal_input_type in BLACK_IMPORTANCE_THERMAL_INPUT_TYPES
            ),
        )
        fused_emb, fused_token_mask, gate_info = self.twogrey_patchsinglegateresvit_fusion(
            rgb_tokens=rgb_emb,
            cold_tokens=cold_emb,
            hot_tokens=hot_emb,
            text_tokens=text_tokens,
            rgb_token_mask=rgb_token_mask,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
            cold_evidence=cold_evidence,
            hot_evidence=hot_evidence,
            base_cold_alpha=self.config.thermal_twogrey_resvit_cold_alpha,
            base_hot_beta=self.config.thermal_twogrey_resvit_hot_beta,
        )
        self._last_twogrey_patchsinglegateresvit_gate_summary = (
            self._summarize_twogrey_gateresvit_gate(gate_info)
        )
        return fused_emb, fused_token_mask

    def _embed_twogrey_smallpatchgatetworesvit_head_images(
        self,
        *,
        rgb_img: Tensor,
        cold_img: Tensor,
        hot_img: Tensor,
        rgb_img_mask: Tensor,
        cold_img_mask: Tensor,
        hot_img_mask: Tensor,
        text_tokens: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if (
            self.thermal_vision_tower is None
            or self.twogrey_smallpatchgatetworesvit_fusion is None
        ):
            raise RuntimeError("SmallPatchGateTwoResViT modules are not initialized.")

        rgb_emb = self._embed_rgb_image(rgb_img)
        cold_img = self._prepare_thermal_vit_image(cold_img)
        hot_img = self._prepare_thermal_vit_image(hot_img)
        out_dtype = rgb_emb.dtype
        cold_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            cold_img,
            self.thermal_vision_tower,
        )
        hot_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            hot_img,
            self.thermal_vision_tower,
        )
        cold_emb = self._project_siglip_tokens_with_rgb_projector(
            cold_raw_tokens,
            out_dtype=out_dtype,
        )
        hot_emb = self._project_siglip_tokens_with_rgb_projector(
            hot_raw_tokens,
            out_dtype=out_dtype,
        )
        if rgb_emb.shape != cold_emb.shape or rgb_emb.shape != hot_emb.shape:
            raise ValueError(
                "SmallPatchGateTwoResViT requires aligned projected head RGB, cold, and hot "
                "token grids, got "
                f"rgb={tuple(rgb_emb.shape)}, cold={tuple(cold_emb.shape)}, "
                f"hot={tuple(hot_emb.shape)}."
            )

        batch_size, num_tokens = rgb_emb.shape[:2]
        rgb_token_mask = rgb_img_mask[:, None].expand(batch_size, num_tokens)
        cold_token_mask = cold_img_mask[:, None].expand(batch_size, num_tokens)
        hot_token_mask = hot_img_mask[:, None].expand(batch_size, num_tokens)
        cold_evidence, hot_evidence = twogrey_patch_evidence(
            cold_img,
            hot_img,
            target_background=self.config.thermal_grey_background_value,
            token_count=num_tokens,
            black_importance=(
                self.config.thermal_input_type in BLACK_IMPORTANCE_THERMAL_INPUT_TYPES
            ),
        )
        (
            cold_res_emb,
            cold_res_token_mask,
            hot_res_emb,
            hot_res_token_mask,
            gate_info,
        ) = self.twogrey_smallpatchgatetworesvit_fusion(
            rgb_tokens=rgb_emb,
            cold_tokens=cold_emb,
            hot_tokens=hot_emb,
            text_tokens=text_tokens,
            rgb_token_mask=rgb_token_mask,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
            cold_evidence=cold_evidence,
            hot_evidence=hot_evidence,
            base_cold_alpha=self.config.thermal_twogrey_resvit_cold_alpha,
            base_hot_beta=self.config.thermal_twogrey_resvit_hot_beta,
        )
        gate_summary = self._summarize_twogrey_gateresvit_gate(gate_info)
        if self.config.uses_small_patch_single_gate_two_res_thermal_vit:
            self._last_twogrey_smallpatchsinglegatetworesvit_gate_summary = gate_summary
        else:
            self._last_twogrey_smallpatchgatetworesvit_gate_summary = gate_summary
        return cold_res_emb, cold_res_token_mask, hot_res_emb, hot_res_token_mask

    def _embed_twogrey_smallpatchgateresvit_head_image(
        self,
        *,
        rgb_img: Tensor,
        cold_img: Tensor,
        hot_img: Tensor,
        rgb_img_mask: Tensor,
        cold_img_mask: Tensor,
        hot_img_mask: Tensor,
        text_tokens: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if (
            self.thermal_vision_tower is None
            or self.twogrey_smallpatchgateresvit_fusion is None
        ):
            raise RuntimeError("SmallPatchGateResViT modules are not initialized.")

        rgb_emb = self._embed_rgb_image(rgb_img)
        cold_img = self._prepare_thermal_vit_image(cold_img)
        hot_img = self._prepare_thermal_vit_image(hot_img)
        out_dtype = rgb_emb.dtype
        cold_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            cold_img,
            self.thermal_vision_tower,
        )
        hot_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            hot_img,
            self.thermal_vision_tower,
        )
        cold_emb = self._project_siglip_tokens_with_rgb_projector(
            cold_raw_tokens,
            out_dtype=out_dtype,
        )
        hot_emb = self._project_siglip_tokens_with_rgb_projector(
            hot_raw_tokens,
            out_dtype=out_dtype,
        )
        if rgb_emb.shape != cold_emb.shape or rgb_emb.shape != hot_emb.shape:
            raise ValueError(
                "SmallPatchGateResViT requires aligned projected head RGB, cold, and hot "
                "token grids, got "
                f"rgb={tuple(rgb_emb.shape)}, cold={tuple(cold_emb.shape)}, "
                f"hot={tuple(hot_emb.shape)}."
            )

        batch_size, num_tokens = rgb_emb.shape[:2]
        rgb_token_mask = rgb_img_mask[:, None].expand(batch_size, num_tokens)
        cold_token_mask = cold_img_mask[:, None].expand(batch_size, num_tokens)
        hot_token_mask = hot_img_mask[:, None].expand(batch_size, num_tokens)
        cold_evidence, hot_evidence = twogrey_patch_evidence(
            cold_img,
            hot_img,
            target_background=self.config.thermal_grey_background_value,
            token_count=num_tokens,
            black_importance=(
                self.config.thermal_input_type in BLACK_IMPORTANCE_THERMAL_INPUT_TYPES
            ),
        )
        fused_emb, fused_token_mask, gate_info = self.twogrey_smallpatchgateresvit_fusion(
            rgb_tokens=rgb_emb,
            cold_tokens=cold_emb,
            hot_tokens=hot_emb,
            text_tokens=text_tokens,
            rgb_token_mask=rgb_token_mask,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
            cold_evidence=cold_evidence,
            hot_evidence=hot_evidence,
            base_cold_alpha=self.config.thermal_twogrey_resvit_cold_alpha,
            base_hot_beta=self.config.thermal_twogrey_resvit_hot_beta,
        )
        self._last_twogrey_smallpatchgateresvit_gate_summary = (
            self._summarize_twogrey_gateresvit_gate(gate_info)
        )
        return fused_emb, fused_token_mask

    def _embed_twogrey_gatevitmix_head_and_thermal(
        self,
        *,
        rgb_img: Tensor,
        cold_img: Tensor,
        hot_img: Tensor,
        rgb_img_mask: Tensor,
        cold_img_mask: Tensor,
        hot_img_mask: Tensor,
        text_tokens: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if (
            self.thermal_vision_tower is None
            or self.twogrey_gatevitmix_fusion is None
            or self.gatevitmix_fusion is None
        ):
            raise RuntimeError("GateViTMix modules are not initialized.")

        rgb_emb = self._embed_rgb_image(rgb_img)
        out_dtype = rgb_emb.dtype
        cold_img = self._prepare_thermal_vit_image(cold_img)
        hot_img = self._prepare_thermal_vit_image(hot_img)
        cold_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            cold_img,
            self.thermal_vision_tower,
        )
        hot_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            hot_img,
            self.thermal_vision_tower,
        )
        cold_emb = self._project_siglip_tokens_with_rgb_projector(
            cold_raw_tokens,
            out_dtype=out_dtype,
        )
        hot_emb = self._project_siglip_tokens_with_rgb_projector(
            hot_raw_tokens,
            out_dtype=out_dtype,
        )
        if rgb_emb.shape != cold_emb.shape or rgb_emb.shape != hot_emb.shape:
            raise ValueError(
                "GateViTMix requires aligned projected head RGB, cold, and hot token grids, "
                f"got rgb={tuple(rgb_emb.shape)}, cold={tuple(cold_emb.shape)}, "
                f"hot={tuple(hot_emb.shape)}."
            )

        batch_size, num_tokens = rgb_emb.shape[:2]
        rgb_token_mask = rgb_img_mask[:, None].expand(batch_size, num_tokens)
        cold_token_mask = cold_img_mask[:, None].expand(batch_size, num_tokens)
        hot_token_mask = hot_img_mask[:, None].expand(batch_size, num_tokens)
        thermal_emb, thermal_token_mask, gate_info = self.twogrey_gatevitmix_fusion(
            cold_tokens=cold_emb,
            hot_tokens=hot_emb,
            text_tokens=text_tokens,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
        )
        mixed_thermal_emb, mix_info = self.gatevitmix_fusion(
            rgb_tokens=rgb_emb,
            thermal_tokens=thermal_emb,
            rgb_token_mask=rgb_token_mask,
            thermal_token_mask=thermal_token_mask,
        )
        gate_info.update(mix_info)
        self._last_twogrey_gatevitmix_gate_summary = (
            self._summarize_twogrey_gateresvit_gate(gate_info)
        )
        return rgb_emb, rgb_token_mask, mixed_thermal_emb, thermal_token_mask

    def _embed_grey_gateoneresvit_head_image(
        self,
        *,
        rgb_img: Tensor,
        thermal_img: Tensor,
        rgb_img_mask: Tensor,
        thermal_img_mask: Tensor,
        text_tokens: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.thermal_vision_tower is None or self.grey_gateoneresvit_fusion is None:
            raise RuntimeError("GateOneResViT modules are not initialized.")

        rgb_emb = self._embed_rgb_image(rgb_img)
        out_dtype = rgb_emb.dtype
        thermal_img = self._prepare_thermal_vit_image(thermal_img)
        thermal_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            thermal_img,
            self.thermal_vision_tower,
        )
        thermal_emb = self._project_siglip_tokens_with_rgb_projector(
            thermal_raw_tokens,
            out_dtype=out_dtype,
        )
        if rgb_emb.shape != thermal_emb.shape:
            raise ValueError(
                "GateOneResViT requires RGB and thermal projected tokens to have the same shape, "
                f"got rgb={tuple(rgb_emb.shape)} and thermal={tuple(thermal_emb.shape)}."
            )

        batch_size, num_tokens = rgb_emb.shape[:2]
        rgb_token_mask = rgb_img_mask[:, None].expand(batch_size, num_tokens)
        thermal_token_mask = thermal_img_mask[:, None].expand(batch_size, num_tokens)
        fused_emb, fused_token_mask, gate_info = self.grey_gateoneresvit_fusion(
            rgb_tokens=rgb_emb,
            thermal_tokens=thermal_emb,
            text_tokens=text_tokens,
            rgb_token_mask=rgb_token_mask,
            thermal_token_mask=thermal_token_mask,
            text_token_mask=text_token_mask,
        )
        self._last_grey_gateoneresvit_gate_summary = self._summarize_twogrey_gateresvit_gate(
            gate_info
        )
        return fused_emb, fused_token_mask

    def _embed_twogrey_gateresvit_head_and_original_images(
        self,
        *,
        rgb_img: Tensor,
        cold_img: Tensor,
        hot_img: Tensor,
        rgb_img_mask: Tensor,
        cold_img_mask: Tensor,
        hot_img_mask: Tensor,
        text_tokens: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.thermal_vision_tower is None or self.twogrey_gateresvit_fusion is None:
            raise RuntimeError("GateResViT modules are not initialized.")

        rgb_emb = self._embed_rgb_image(rgb_img)
        out_dtype = rgb_emb.dtype
        cold_img = self._prepare_thermal_vit_image(cold_img)
        hot_img = self._prepare_thermal_vit_image(hot_img)
        cold_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            cold_img,
            self.thermal_vision_tower,
        )
        hot_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            hot_img,
            self.thermal_vision_tower,
        )
        cold_emb = self._project_siglip_tokens_with_rgb_projector(
            cold_raw_tokens,
            out_dtype=out_dtype,
        )
        hot_emb = self._project_siglip_tokens_with_rgb_projector(
            hot_raw_tokens,
            out_dtype=out_dtype,
        )
        if rgb_emb.shape != cold_emb.shape or rgb_emb.shape != hot_emb.shape:
            raise ValueError(
                "GateResViT requires RGB, cold thermal, and hot thermal projected tokens "
                "to have the same shape, got "
                f"rgb={tuple(rgb_emb.shape)}, cold={tuple(cold_emb.shape)}, "
                f"hot={tuple(hot_emb.shape)}."
            )

        batch_size, num_rgb_tokens = rgb_emb.shape[:2]
        rgb_token_mask = rgb_img_mask[:, None].expand(batch_size, num_rgb_tokens)
        cold_token_mask = cold_img_mask[:, None].expand(batch_size, num_rgb_tokens)
        hot_token_mask = hot_img_mask[:, None].expand(batch_size, num_rgb_tokens)
        fused_emb, gate_info = self.twogrey_gateresvit_fusion(
            rgb_tokens=rgb_emb,
            cold_tokens=cold_emb,
            hot_tokens=hot_emb,
            text_tokens=text_tokens,
            rgb_token_mask=rgb_token_mask,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
            base_cold_alpha=self.config.thermal_twogrey_resvit_cold_alpha,
            base_hot_beta=self.config.thermal_twogrey_resvit_hot_beta,
        )
        gate_summary = self._summarize_twogrey_gateresvit_gate(gate_info)
        if getattr(self.config, "uses_gate_res_and_thermal_vit", False):
            self._last_twogrey_gateresandvit_gate_summary = gate_summary
        else:
            self._last_twogrey_gateresvit_gate_summary = gate_summary
        return fused_emb, rgb_emb, rgb_token_mask

    def _embed_twogrey_gateresvit_head_image(
        self,
        *,
        rgb_img: Tensor,
        cold_img: Tensor,
        hot_img: Tensor,
        rgb_img_mask: Tensor,
        cold_img_mask: Tensor,
        hot_img_mask: Tensor,
        text_tokens: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        fused_emb, _, rgb_token_mask = self._embed_twogrey_gateresvit_head_and_original_images(
            rgb_img=rgb_img,
            cold_img=cold_img,
            hot_img=hot_img,
            rgb_img_mask=rgb_img_mask,
            cold_img_mask=cold_img_mask,
            hot_img_mask=hot_img_mask,
            text_tokens=text_tokens,
            text_token_mask=text_token_mask,
        )
        return fused_emb, rgb_token_mask

    def _embed_twogrey_gatetworesvit_head_images(
        self,
        *,
        rgb_img: Tensor,
        cold_img: Tensor,
        hot_img: Tensor,
        rgb_img_mask: Tensor,
        cold_img_mask: Tensor,
        hot_img_mask: Tensor,
        text_tokens: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.thermal_vision_tower is None or self.twogrey_gatetworesvit_fusion is None:
            raise RuntimeError("GateTwoResViT modules are not initialized.")

        rgb_emb = self._embed_rgb_image(rgb_img)
        out_dtype = rgb_emb.dtype
        cold_img = self._prepare_thermal_vit_image(cold_img)
        hot_img = self._prepare_thermal_vit_image(hot_img)
        cold_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            cold_img,
            self.thermal_vision_tower,
        )
        hot_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            hot_img,
            self.thermal_vision_tower,
        )
        cold_emb = self._project_siglip_tokens_with_rgb_projector(
            cold_raw_tokens,
            out_dtype=out_dtype,
        )
        hot_emb = self._project_siglip_tokens_with_rgb_projector(
            hot_raw_tokens,
            out_dtype=out_dtype,
        )
        if rgb_emb.shape != cold_emb.shape or rgb_emb.shape != hot_emb.shape:
            raise ValueError(
                "GateTwoResViT requires RGB, cold thermal, and hot thermal projected tokens "
                "to have the same shape, got "
                f"rgb={tuple(rgb_emb.shape)}, cold={tuple(cold_emb.shape)}, "
                f"hot={tuple(hot_emb.shape)}."
            )

        batch_size, num_tokens = rgb_emb.shape[:2]
        rgb_token_mask = rgb_img_mask[:, None].expand(batch_size, num_tokens)
        cold_token_mask = cold_img_mask[:, None].expand(batch_size, num_tokens)
        hot_token_mask = hot_img_mask[:, None].expand(batch_size, num_tokens)
        (
            cold_res_emb,
            cold_res_token_mask,
            hot_res_emb,
            hot_res_token_mask,
            gate_info,
        ) = self.twogrey_gatetworesvit_fusion(
            rgb_tokens=rgb_emb,
            cold_tokens=cold_emb,
            hot_tokens=hot_emb,
            text_tokens=text_tokens,
            rgb_token_mask=rgb_token_mask,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
            base_cold_alpha=self.config.thermal_twogrey_resvit_cold_alpha,
            base_hot_beta=self.config.thermal_twogrey_resvit_hot_beta,
        )
        self._last_twogrey_gatetworesvit_gate_summary = (
            self._summarize_twogrey_gateresvit_gate(gate_info)
        )
        return cold_res_emb, cold_res_token_mask, hot_res_emb, hot_res_token_mask

    def _embed_twogrey_gateresvit3_head_image(
        self,
        *,
        rgb_img: Tensor,
        cold_img: Tensor,
        hot_img: Tensor,
        rgb_img_mask: Tensor,
        cold_img_mask: Tensor,
        hot_img_mask: Tensor,
        text_tokens: Tensor,
        text_token_mask: Tensor,
        thermal_stats: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.thermal_vision_tower is None or self.twogrey_gateresvit3_fusion is None:
            raise RuntimeError("GateResViT3 modules are not initialized.")

        rgb_emb = self._embed_rgb_image(rgb_img)
        out_dtype = rgb_emb.dtype
        cold_img = self._prepare_thermal_vit_image(cold_img)
        hot_img = self._prepare_thermal_vit_image(hot_img)
        cold_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            cold_img,
            self.thermal_vision_tower,
        )
        hot_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            hot_img,
            self.thermal_vision_tower,
        )
        cold_emb = self._project_siglip_tokens_with_rgb_projector(
            cold_raw_tokens,
            out_dtype=out_dtype,
        )
        hot_emb = self._project_siglip_tokens_with_rgb_projector(
            hot_raw_tokens,
            out_dtype=out_dtype,
        )
        if rgb_emb.shape != cold_emb.shape or rgb_emb.shape != hot_emb.shape:
            raise ValueError(
                "GateResViT3 requires RGB, cold thermal, and hot thermal projected tokens "
                "to have the same shape, got "
                f"rgb={tuple(rgb_emb.shape)}, cold={tuple(cold_emb.shape)}, "
                f"hot={tuple(hot_emb.shape)}."
            )

        batch_size, num_rgb_tokens = rgb_emb.shape[:2]
        rgb_token_mask = rgb_img_mask[:, None].expand(batch_size, num_rgb_tokens)
        cold_token_mask = cold_img_mask[:, None].expand(batch_size, num_rgb_tokens)
        hot_token_mask = hot_img_mask[:, None].expand(batch_size, num_rgb_tokens)
        if thermal_stats is None:
            thermal_stats = self.twogrey_gateresvit3_fusion.compute_thermal_stats(
                cold_images=cold_img,
                hot_images=hot_img,
                background_value=self.config.thermal_grey_background_value,
                black_importance=(
                    self.config.thermal_input_type in BLACK_IMPORTANCE_THERMAL_INPUT_TYPES
                ),
            )
        fused_emb, gate_info = self.twogrey_gateresvit3_fusion(
            rgb_tokens=rgb_emb,
            cold_tokens=cold_emb,
            hot_tokens=hot_emb,
            text_tokens=text_tokens,
            rgb_token_mask=rgb_token_mask,
            cold_token_mask=cold_token_mask,
            hot_token_mask=hot_token_mask,
            text_token_mask=text_token_mask,
            thermal_stats=thermal_stats,
            base_cold_alpha=self.config.thermal_twogrey_resvit_cold_alpha,
            base_hot_beta=self.config.thermal_twogrey_resvit_hot_beta,
        )
        self._last_twogrey_gateresvit3_gate_summary = self._summarize_twogrey_gateresvit_gate(
            gate_info
        )
        return fused_emb, rgb_token_mask

    def _embed_resvit_attention_head_image(
        self,
        *,
        rgb_img: Tensor,
        thermal_img: Tensor,
        rgb_img_mask: Tensor,
        thermal_img_mask: Tensor,
        text_tokens: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.thermal_vision_tower is None or self.resvit_attention_fusion is None:
            raise RuntimeError("ResViTAttention fusion modules are not initialized.")

        rgb_emb = self._embed_rgb_image(rgb_img)
        out_dtype = rgb_emb.dtype
        thermal_img = self._prepare_thermal_vit_image(thermal_img)
        thermal_raw_tokens = self.paligemma_with_expert.embed_image_tokens_with_modules(
            thermal_img,
            self.thermal_vision_tower,
        )
        thermal_emb = self._project_siglip_tokens_with_rgb_projector(
            thermal_raw_tokens,
            out_dtype=out_dtype,
        )
        if rgb_emb.shape != thermal_emb.shape:
            raise ValueError(
                "ResViTAttention requires head RGB and thermal projected tokens to have the same shape, "
                f"got rgb={tuple(rgb_emb.shape)} and thermal={tuple(thermal_emb.shape)}."
            )

        batch_size, num_rgb_tokens = rgb_emb.shape[:2]
        rgb_token_mask = rgb_img_mask[:, None].expand(batch_size, num_rgb_tokens)
        thermal_token_mask = thermal_img_mask[:, None].expand(batch_size, num_rgb_tokens)
        fused_emb = self.resvit_attention_fusion(
            rgb_tokens=rgb_emb,
            thermal_tokens=thermal_emb,
            text_tokens=text_tokens,
            rgb_token_mask=rgb_token_mask,
            thermal_token_mask=thermal_token_mask,
            text_token_mask=text_token_mask,
        )
        return fused_emb, rgb_token_mask

    def _embed_resthermal_head_image(
        self,
        rgb_img: Tensor,
        thermal_img: Tensor,
        rgb_img_mask: Tensor,
        thermal_img_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.thermal_residual_fusion is None:
            raise RuntimeError("AnyThermal residual fusion module is not initialized.")

        rgb_emb = self._embed_rgb_image(rgb_img)
        batch_size, num_rgb_tokens = rgb_emb.shape[:2]
        rgb_token_mask = rgb_img_mask[:, None].expand(batch_size, num_rgb_tokens)
        fused_emb = self.thermal_residual_fusion(
            rgb_tokens=rgb_emb,
            thermal_images=thermal_img,
            rgb_token_mask=rgb_token_mask,
            thermal_image_mask=thermal_img_mask,
        )
        return fused_emb, rgb_token_mask

    def embed_prefix(
        self,
        images,
        img_masks,
        tokens,
        masks,
        rgb_thermal_align_target=None,
        twogrey_gateresvit3_stats: Tensor | None = None,
        capture_patchgatevit_summary: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []
        prefix_token_layout = []
        token_offset = 0
        track_image_embs = self.thermal_head_fusion is not None
        image_embs_by_key = {}
        image_token_masks_by_key = {}
        image_keys = list(self.config.image_features)
        image_inputs_by_key = dict(zip(image_keys, images, strict=True))
        image_masks_by_key = dict(zip(image_keys, img_masks, strict=True))
        self._last_rgb_threeblack_gatetworesvit_gate_summary = None
        self._last_twogrey_gatevit_gate_summary = None
        self._last_twogrey_gateactionvit_gate_summary = None
        self._current_gateactionvit_attention_bias = None
        self._last_twogrey_doublegatevit_gate_summary = None
        self._last_twogrey_doublegatetworesvit_gate_summary = None
        self._last_twogrey_patchgatevit_gate_summary = None
        self._last_twogrey_patchsinglegateresvit_gate_summary = None
        self._last_twogrey_smallpatchgatetworesvit_gate_summary = None
        self._last_twogrey_smallpatchsinglegatetworesvit_gate_summary = None
        self._last_twogrey_smallpatchgateresvit_gate_summary = None
        self._last_twogrey_gatevitmix_gate_summary = None
        self._last_twogrey_gateresvit_gate_summary = None
        self._last_twogrey_gateresandvit_gate_summary = None
        self._last_twogrey_gatetworesvit_gate_summary = None
        self._last_grey_gateoneresvit_gate_summary = None
        self._last_twogrey_gateresvit3_gate_summary = None

        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            return lang_emb

        lang_emb = None
        if (
            getattr(self, "rgb_threeblack_gatetworesvit_fusion", None) is not None
            or self.resvit_attention_fusion is not None
            or getattr(self, "twogrey_gatevit_fusion", None) is not None
            or getattr(self, "twogrey_gateactionvit_fusion", None) is not None
            or getattr(self, "twogrey_doublegatevit_fusion", None) is not None
            or getattr(self, "twogrey_doublegatetworesvit_fusion", None) is not None
            or getattr(self, "twogrey_patchgatevit_fusion", None) is not None
            or getattr(self, "twogrey_patchsinglegateresvit_fusion", None) is not None
            or getattr(self, "twogrey_smallpatchgatetworesvit_fusion", None) is not None
            or getattr(self, "twogrey_smallpatchgateresvit_fusion", None) is not None
            or getattr(self, "twogrey_gatevitmix_fusion", None) is not None
            or getattr(self, "twogrey_gateresvit_fusion", None) is not None
            or getattr(self, "twogrey_gatetworesvit_fusion", None) is not None
            or getattr(self, "grey_gateoneresvit_fusion", None) is not None
            or getattr(self, "twogrey_gateresvit3_fusion", None) is not None
        ):
            lang_emb = self._apply_checkpoint(lang_embed_func, tokens)

        rgb_threeblack_precomputed = {}
        if getattr(self, "rgb_threeblack_gatetworesvit_fusion", None) is not None:
            head_key = self.rgb_threeblack_gatetworesvit_head_feature
            if head_key not in image_inputs_by_key:
                raise ValueError(
                    "PI05 policy.rgb_input_type='GateTwoResViT' is missing head RGB feature "
                    f"{head_key!r}. Available image features: {image_keys}."
                )
            if lang_emb is None:
                raise RuntimeError("RGB GateTwoResViT expected precomputed language embeddings.")
            (
                red_res_emb,
                red_res_token_mask,
                green_res_emb,
                green_res_token_mask,
                blue_res_emb,
                blue_res_token_mask,
            ) = self._embed_rgb_threeblack_gatetworesvit_head_images(
                rgb_img=image_inputs_by_key[head_key],
                rgb_img_mask=image_masks_by_key[head_key],
                text_tokens=lang_emb,
                text_token_mask=masks,
            )
            rgb_threeblack_precomputed = {
                head_key: (
                    ("red", red_res_emb, red_res_token_mask),
                    ("green", green_res_emb, green_res_token_mask),
                    ("blue", blue_res_emb, blue_res_token_mask),
                )
            }

        doublegate_precomputed = {}
        if getattr(self, "twogrey_doublegatevit_fusion", None) is not None:
            head_key = self.rgb_thermal_doublegatevit_head_feature
            cold_key = self.rgb_thermal_doublegatevit_cold_feature
            hot_key = self.rgb_thermal_doublegatevit_hot_feature
            missing_keys = [
                key for key in (head_key, cold_key, hot_key) if key not in image_inputs_by_key
            ]
            if missing_keys:
                raise ValueError(
                    "PI05 thermal_encoder_channel='DoubleGateViT' is missing image features "
                    f"{missing_keys}. Available image features: {image_keys}."
                )
            if lang_emb is None:
                raise RuntimeError("DoubleGateViT expected precomputed language embeddings.")
            (
                rgb_emb,
                rgb_token_mask,
                thermal_emb,
                thermal_token_mask,
            ) = self._embed_twogrey_doublegatevit_head_and_thermal(
                rgb_img=image_inputs_by_key[head_key],
                cold_img=image_inputs_by_key[cold_key],
                hot_img=image_inputs_by_key[hot_key],
                rgb_img_mask=image_masks_by_key[head_key],
                cold_img_mask=image_masks_by_key[cold_key],
                hot_img_mask=image_masks_by_key[hot_key],
                text_tokens=lang_emb,
                text_token_mask=masks,
            )
            doublegate_precomputed = {
                head_key: ("rgb", rgb_emb, rgb_token_mask),
                cold_key: ("thermal", thermal_emb, thermal_token_mask),
            }

        doublegatetwores_precomputed = {}
        if getattr(self, "twogrey_doublegatetworesvit_fusion", None) is not None:
            head_key = self.rgb_thermal_doublegatetworesvit_head_feature
            cold_key = self.rgb_thermal_doublegatetworesvit_cold_feature
            hot_key = self.rgb_thermal_doublegatetworesvit_hot_feature
            missing_keys = [
                key for key in (head_key, cold_key, hot_key) if key not in image_inputs_by_key
            ]
            if missing_keys:
                raise ValueError(
                    "PI05 thermal_encoder_channel='DoubleGateTwoResViT' is missing image features "
                    f"{missing_keys}. Available image features: {image_keys}."
                )
            if lang_emb is None:
                raise RuntimeError("DoubleGateTwoResViT expected precomputed language embeddings.")
            (
                cold_res_emb,
                cold_res_token_mask,
                hot_res_emb,
                hot_res_token_mask,
            ) = self._embed_twogrey_doublegatetworesvit_head_images(
                rgb_img=image_inputs_by_key[head_key],
                cold_img=image_inputs_by_key[cold_key],
                hot_img=image_inputs_by_key[hot_key],
                rgb_img_mask=image_masks_by_key[head_key],
                cold_img_mask=image_masks_by_key[cold_key],
                hot_img_mask=image_masks_by_key[hot_key],
                text_tokens=lang_emb,
                text_token_mask=masks,
            )
            doublegatetwores_precomputed = {
                head_key: ("cold_residual", cold_res_emb, cold_res_token_mask),
                cold_key: ("hot_residual", hot_res_emb, hot_res_token_mask),
            }

        patchgate_precomputed = {}
        if getattr(self, "twogrey_patchgatevit_fusion", None) is not None:
            head_key = self.rgb_thermal_patchgatevit_head_feature
            cold_key = self.rgb_thermal_patchgatevit_cold_feature
            hot_key = self.rgb_thermal_patchgatevit_hot_feature
            missing_keys = [
                key for key in (head_key, cold_key, hot_key) if key not in image_inputs_by_key
            ]
            if missing_keys:
                raise ValueError(
                    "PI05 thermal_encoder_channel='PatchGateViT' is missing image features "
                    f"{missing_keys}. Available image features: {image_keys}."
                )
            if lang_emb is None:
                raise RuntimeError("PatchGateViT expected precomputed language embeddings.")
            (
                rgb_emb,
                rgb_token_mask,
                thermal_emb,
                thermal_token_mask,
            ) = self._embed_twogrey_patchgatevit_head_and_thermal(
                rgb_img=image_inputs_by_key[head_key],
                cold_img=image_inputs_by_key[cold_key],
                hot_img=image_inputs_by_key[hot_key],
                rgb_img_mask=image_masks_by_key[head_key],
                cold_img_mask=image_masks_by_key[cold_key],
                hot_img_mask=image_masks_by_key[hot_key],
                text_tokens=lang_emb,
                text_token_mask=masks,
                capture_gate_summary=capture_patchgatevit_summary,
            )
            patchgate_precomputed = {
                head_key: ("rgb", rgb_emb, rgb_token_mask),
                cold_key: ("thermal", thermal_emb, thermal_token_mask),
            }

        smallpatch_twores_precomputed = {}
        if getattr(self, "twogrey_smallpatchgatetworesvit_fusion", None) is not None:
            smallpatch_twores_mode_name = (
                "SmallPatchSingleGateTwoResViT"
                if self.config.uses_small_patch_single_gate_two_res_thermal_vit
                else "SmallPatchGateTwoResViT"
            )
            head_key = self.rgb_thermal_smallpatchgatetworesvit_head_feature
            cold_key = self.rgb_thermal_smallpatchgatetworesvit_cold_feature
            hot_key = self.rgb_thermal_smallpatchgatetworesvit_hot_feature
            missing_keys = [
                key for key in (head_key, cold_key, hot_key) if key not in image_inputs_by_key
            ]
            if missing_keys:
                raise ValueError(
                    f"PI05 thermal_encoder_channel='{smallpatch_twores_mode_name}' is missing image "
                    f"features {missing_keys}. Available image features: {image_keys}."
                )
            if lang_emb is None:
                raise RuntimeError(
                    f"{smallpatch_twores_mode_name} expected precomputed language embeddings."
                )
            (
                cold_res_emb,
                cold_res_token_mask,
                hot_res_emb,
                hot_res_token_mask,
            ) = self._embed_twogrey_smallpatchgatetworesvit_head_images(
                rgb_img=image_inputs_by_key[head_key],
                cold_img=image_inputs_by_key[cold_key],
                hot_img=image_inputs_by_key[hot_key],
                rgb_img_mask=image_masks_by_key[head_key],
                cold_img_mask=image_masks_by_key[cold_key],
                hot_img_mask=image_masks_by_key[hot_key],
                text_tokens=lang_emb,
                text_token_mask=masks,
            )
            smallpatch_twores_precomputed = {
                head_key: ("cold_residual", cold_res_emb, cold_res_token_mask),
                cold_key: ("hot_residual", hot_res_emb, hot_res_token_mask),
            }

        gatetwores_precomputed = {}
        if getattr(self, "twogrey_gatetworesvit_fusion", None) is not None:
            head_key = self.rgb_thermal_gatetworesvit_head_feature
            cold_key = self.rgb_thermal_gatetworesvit_cold_feature
            hot_key = self.rgb_thermal_gatetworesvit_hot_feature
            missing_keys = [
                key for key in (head_key, cold_key, hot_key) if key not in image_inputs_by_key
            ]
            if missing_keys:
                raise ValueError(
                    "PI05 thermal_encoder_channel='GateTwoResViT' is missing image features "
                    f"{missing_keys}. Available image features: {image_keys}."
                )
            if lang_emb is None:
                raise RuntimeError("GateTwoResViT expected precomputed language embeddings.")
            (
                cold_res_emb,
                cold_res_token_mask,
                hot_res_emb,
                hot_res_token_mask,
            ) = self._embed_twogrey_gatetworesvit_head_images(
                rgb_img=image_inputs_by_key[head_key],
                cold_img=image_inputs_by_key[cold_key],
                hot_img=image_inputs_by_key[hot_key],
                rgb_img_mask=image_masks_by_key[head_key],
                cold_img_mask=image_masks_by_key[cold_key],
                hot_img_mask=image_masks_by_key[hot_key],
                text_tokens=lang_emb,
                text_token_mask=masks,
            )
            gatetwores_precomputed = {
                head_key: ("cold_residual", cold_res_emb, cold_res_token_mask),
                cold_key: ("hot_residual", hot_res_emb, hot_res_token_mask),
            }

        # Process images
        for key, img, img_mask in zip(image_keys, images, img_masks, strict=True):
            if key in rgb_threeblack_precomputed:
                for color_name, img_emb, token_mask in rgb_threeblack_precomputed[key]:
                    num_img_embs = img_emb.shape[1]
                    embs.append(img_emb)
                    pad_masks.append(token_mask)
                    att_masks += [0] * num_img_embs
                    prefix_token_layout.append(
                        {
                            "kind": f"image_rgb_gatetworesvit_{color_name}_residual",
                            "name": key,
                            "start": int(token_offset),
                            "end": int(token_offset + num_img_embs),
                            "token_count": int(num_img_embs),
                            "valid": bool(token_mask.detach().any().item()),
                            "head_rgb_feature": key,
                            "color_view": f"{color_name}_black_importance",
                            "residual_role": f"{color_name}_residual",
                            "gate": (
                                deepcopy(
                                    self._last_rgb_threeblack_gatetworesvit_gate_summary
                                )
                                if color_name == "red"
                                else None
                            ),
                        }
                    )
                    token_offset += int(num_img_embs)
                continue

            if (
                getattr(self.config, "uses_double_gate_thermal_vit", False)
                and key == getattr(self, "rgb_thermal_doublegatevit_hot_feature", None)
            ):
                continue
            if key in doublegate_precomputed:
                role, img_emb, token_mask = doublegate_precomputed[key]
                num_img_embs = img_emb.shape[1]
                embs.append(img_emb)
                pad_masks.append(token_mask)
                att_masks += [0] * num_img_embs
                if role == "thermal":
                    prefix_token_layout.append(
                        {
                            "kind": "image_doublegatevit_twogrey",
                            "name": self.rgb_thermal_doublegatevit_source_feature,
                            "start": int(token_offset),
                            "end": int(token_offset + num_img_embs),
                            "token_count": int(num_img_embs),
                            "valid": bool(token_mask.detach().any().item()),
                            "head_rgb_feature": self.rgb_thermal_doublegatevit_head_feature,
                            "thermal_features": [
                                self.rgb_thermal_doublegatevit_cold_feature,
                                self.rgb_thermal_doublegatevit_hot_feature,
                            ],
                            "gate": deepcopy(self._last_twogrey_doublegatevit_gate_summary),
                        }
                    )
                else:
                    prefix_token_layout.append(
                        {
                            "kind": "image",
                            "name": key,
                            "start": int(token_offset),
                            "end": int(token_offset + num_img_embs),
                            "token_count": int(num_img_embs),
                            "valid": bool(token_mask.detach().any().item()),
                        }
                    )
                token_offset += int(num_img_embs)
                continue

            if (
                getattr(self.config, "uses_double_gate_two_residual_thermal_vit", False)
                and key == getattr(self, "rgb_thermal_doublegatetworesvit_hot_feature", None)
            ):
                continue
            if key in doublegatetwores_precomputed:
                role, img_emb, token_mask = doublegatetwores_precomputed[key]
                num_img_embs = img_emb.shape[1]
                embs.append(img_emb)
                pad_masks.append(token_mask)
                att_masks += [0] * num_img_embs
                is_cold_role = role == "cold_residual"
                prefix_token_layout.append(
                    {
                        "kind": (
                            "image_doublegatetworesvit_cold_residual"
                            if is_cold_role
                            else "image_doublegatetworesvit_hot_residual"
                        ),
                        "name": (
                            self.rgb_thermal_doublegatetworesvit_head_feature
                            if is_cold_role
                            else self.rgb_thermal_doublegatetworesvit_source_feature
                        ),
                        "start": int(token_offset),
                        "end": int(token_offset + num_img_embs),
                        "token_count": int(num_img_embs),
                        "valid": bool(token_mask.detach().any().item()),
                        "head_rgb_feature": self.rgb_thermal_doublegatetworesvit_head_feature,
                        "thermal_features": [
                            self.rgb_thermal_doublegatetworesvit_cold_feature,
                            self.rgb_thermal_doublegatetworesvit_hot_feature,
                        ],
                        "residual_role": role,
                        "gate": (
                            deepcopy(self._last_twogrey_doublegatetworesvit_gate_summary)
                            if is_cold_role
                            else None
                        ),
                    }
                )
                token_offset += int(num_img_embs)
                continue

            if (
                getattr(self.config, "uses_patch_gate_thermal_vit", False)
                and key == getattr(self, "rgb_thermal_patchgatevit_hot_feature", None)
            ):
                continue
            if key in patchgate_precomputed:
                role, img_emb, token_mask = patchgate_precomputed[key]
                num_img_embs = img_emb.shape[1]
                embs.append(img_emb)
                pad_masks.append(token_mask)
                att_masks += [0] * num_img_embs
                if role == "thermal":
                    prefix_token_layout.append(
                        {
                            "kind": "image_patchgatevit_twogrey",
                            "name": self.rgb_thermal_patchgatevit_source_feature,
                            "start": int(token_offset),
                            "end": int(token_offset + num_img_embs),
                            "token_count": int(num_img_embs),
                            "valid": bool(token_mask.detach().any().item()),
                            "head_rgb_feature": self.rgb_thermal_patchgatevit_head_feature,
                            "thermal_features": [
                                self.rgb_thermal_patchgatevit_cold_feature,
                                self.rgb_thermal_patchgatevit_hot_feature,
                            ],
                            "gate": deepcopy(self._last_twogrey_patchgatevit_gate_summary),
                        }
                    )
                else:
                    prefix_token_layout.append(
                        {
                            "kind": "image",
                            "name": key,
                            "start": int(token_offset),
                            "end": int(token_offset + num_img_embs),
                            "token_count": int(num_img_embs),
                            "valid": bool(token_mask.detach().any().item()),
                        }
                    )
                token_offset += int(num_img_embs)
                continue

            if (
                getattr(self.config, "uses_small_patch_gate_two_res_thermal_vit", False)
                and key == getattr(self, "rgb_thermal_smallpatchgatetworesvit_hot_feature", None)
            ):
                continue
            if key in smallpatch_twores_precomputed:
                role, img_emb, token_mask = smallpatch_twores_precomputed[key]
                num_img_embs = img_emb.shape[1]
                embs.append(img_emb)
                pad_masks.append(token_mask)
                att_masks += [0] * num_img_embs
                is_cold_role = role == "cold_residual"
                smallpatch_twores_kind_prefix = (
                    "image_smallpatchsinglegatetworesvit"
                    if self.config.uses_small_patch_single_gate_two_res_thermal_vit
                    else "image_smallpatchgatetworesvit"
                )
                smallpatch_twores_gate_summary = (
                    self._last_twogrey_smallpatchsinglegatetworesvit_gate_summary
                    if self.config.uses_small_patch_single_gate_two_res_thermal_vit
                    else self._last_twogrey_smallpatchgatetworesvit_gate_summary
                )
                prefix_token_layout.append(
                    {
                        "kind": (
                            f"{smallpatch_twores_kind_prefix}_cold_residual"
                            if is_cold_role
                            else f"{smallpatch_twores_kind_prefix}_hot_residual"
                        ),
                        "name": (
                            self.rgb_thermal_smallpatchgatetworesvit_head_feature
                            if is_cold_role
                            else self.rgb_thermal_smallpatchgatetworesvit_source_feature
                        ),
                        "start": int(token_offset),
                        "end": int(token_offset + num_img_embs),
                        "token_count": int(num_img_embs),
                        "valid": bool(token_mask.detach().any().item()),
                        "head_rgb_feature": self.rgb_thermal_smallpatchgatetworesvit_head_feature,
                        "thermal_features": [
                            self.rgb_thermal_smallpatchgatetworesvit_cold_feature,
                            self.rgb_thermal_smallpatchgatetworesvit_hot_feature,
                        ],
                        "residual_role": role,
                        "gate": (
                            deepcopy(smallpatch_twores_gate_summary)
                            if is_cold_role
                            else None
                        ),
                    }
                )
                token_offset += int(num_img_embs)
                continue

            if (
                getattr(self.config, "uses_gate_two_residual_thermal_vit", False)
                and key == getattr(self, "rgb_thermal_gatetworesvit_hot_feature", None)
            ):
                continue
            if key in gatetwores_precomputed:
                role, img_emb, token_mask = gatetwores_precomputed[key]
                num_img_embs = img_emb.shape[1]
                embs.append(img_emb)
                pad_masks.append(token_mask)
                att_masks += [0] * num_img_embs
                is_cold_role = role == "cold_residual"
                prefix_token_layout.append(
                    {
                        "kind": (
                            "image_gatetworesvit_cold_residual"
                            if is_cold_role
                            else "image_gatetworesvit_hot_residual"
                        ),
                        "name": (
                            self.rgb_thermal_gatetworesvit_head_feature
                            if is_cold_role
                            else self.rgb_thermal_gatetworesvit_cold_feature
                        ),
                        "start": int(token_offset),
                        "end": int(token_offset + num_img_embs),
                        "token_count": int(num_img_embs),
                        "valid": bool(token_mask.detach().any().item()),
                        "head_rgb_feature": self.rgb_thermal_gatetworesvit_head_feature,
                        "thermal_features": [
                            self.rgb_thermal_gatetworesvit_cold_feature,
                            self.rgb_thermal_gatetworesvit_hot_feature,
                        ],
                        "residual_role": role,
                        "gate": (
                            deepcopy(self._last_twogrey_gatetworesvit_gate_summary)
                            if is_cold_role
                            else None
                        ),
                    }
                )
                token_offset += int(num_img_embs)
                continue

            if (
                self.rgb_thermal_align_aligner is not None
                and key == self.rgb_thermal_align_thermal_feature
            ):
                continue
            if (
                self.thermal_residual_fusion is not None
                and key == self.rgb_thermal_residual_thermal_feature
            ):
                continue
            if (
                self.config.uses_residual_thermal_vit
                and key in self.rgb_thermal_resvit_thermal_feature_set
            ):
                continue
            if (
                getattr(self.config, "uses_gate_thermal_vit", False)
                and key == self.rgb_thermal_gatevit_hot_feature
            ):
                continue
            if (
                getattr(self.config, "uses_gate_action_thermal_vit", False)
                and key == getattr(self, "rgb_thermal_gateactionvit_hot_feature", None)
            ):
                continue
            if (
                getattr(self.config, "uses_gate_thermal_vit_mix", False)
                and key in self.rgb_thermal_gatevitmix_thermal_feature_set
            ):
                continue
            if (
                getattr(self.config, "uses_double_gate_two_residual_thermal_vit", False)
                and key in self.rgb_thermal_doublegatetworesvit_thermal_feature_set
            ):
                continue
            if (
                getattr(self.config, "uses_small_patch_gate_two_res_thermal_vit", False)
                and key in self.rgb_thermal_smallpatchgatetworesvit_thermal_feature_set
            ):
                continue
            if (
                getattr(self.config, "uses_patch_single_gate_residual_thermal_vit", False)
                and key in self.rgb_thermal_patchsinglegateresvit_thermal_feature_set
            ):
                continue
            if (
                getattr(self.config, "uses_small_patch_gate_residual_thermal_vit", False)
                and key in self.rgb_thermal_smallpatchgateresvit_thermal_feature_set
            ):
                continue
            if (
                (
                    getattr(self.config, "uses_gate_residual_thermal_vit", False)
                    or getattr(self.config, "uses_gate_res_and_thermal_vit", False)
                )
                and key in self.rgb_thermal_gateresvit_thermal_feature_set
            ):
                continue
            if (
                getattr(self.config, "uses_gate_two_residual_thermal_vit", False)
                and key in self.rgb_thermal_gatetworesvit_thermal_feature_set
            ):
                continue
            if (
                getattr(self.config, "uses_gate_one_residual_thermal_vit", False)
                and key == self.rgb_thermal_gateoneresvit_thermal_feature
            ):
                continue
            if (
                getattr(self.config, "uses_gate_residual_thermal_vit3", False)
                and key in self.rgb_thermal_gateresvit3_thermal_feature_set
            ):
                continue
            if (
                self.resvit_attention_fusion is not None
                and key == self.rgb_thermal_resvit_attention_thermal_feature
            ):
                continue

            if (
                getattr(self.config, "uses_gate_thermal_vit_mix", False)
                and key == self.rgb_thermal_gatevitmix_head_feature
            ):
                cold_key = self.rgb_thermal_gatevitmix_cold_feature
                hot_key = self.rgb_thermal_gatevitmix_hot_feature
                if cold_key not in image_inputs_by_key or hot_key not in image_inputs_by_key:
                    raise ValueError(
                        "PI05 thermal_encoder_channel='GateViTMix' requires cold/hot "
                        f"thermal image features {cold_key!r} and {hot_key!r}. "
                        f"Available image features: {image_keys}."
                    )
                if lang_emb is None:
                    raise RuntimeError("GateViTMix expected precomputed language embeddings.")
                (
                    rgb_emb,
                    rgb_token_mask,
                    thermal_emb,
                    thermal_token_mask,
                ) = self._embed_twogrey_gatevitmix_head_and_thermal(
                    rgb_img=img,
                    cold_img=image_inputs_by_key[cold_key],
                    hot_img=image_inputs_by_key[hot_key],
                    rgb_img_mask=img_mask,
                    cold_img_mask=image_masks_by_key[cold_key],
                    hot_img_mask=image_masks_by_key[hot_key],
                    text_tokens=lang_emb,
                    text_token_mask=masks,
                )

                num_rgb_embs = rgb_emb.shape[1]
                embs.append(rgb_emb)
                pad_masks.append(rgb_token_mask)
                att_masks += [0] * num_rgb_embs
                prefix_token_layout.append(
                    {
                        "kind": "image",
                        "name": key,
                        "start": int(token_offset),
                        "end": int(token_offset + num_rgb_embs),
                        "token_count": int(num_rgb_embs),
                        "valid": bool(rgb_token_mask.detach().any().item()),
                    }
                )
                token_offset += int(num_rgb_embs)

                num_thermal_embs = thermal_emb.shape[1]
                embs.append(thermal_emb)
                pad_masks.append(thermal_token_mask)
                att_masks += [0] * num_thermal_embs
                prefix_token_layout.append(
                    {
                        "kind": "image_gatevitmix_twogrey",
                        "name": self.rgb_thermal_gatevitmix_source_feature,
                        "start": int(token_offset),
                        "end": int(token_offset + num_thermal_embs),
                        "token_count": int(num_thermal_embs),
                        "valid": bool(thermal_token_mask.detach().any().item()),
                        "head_rgb_feature": key,
                        "thermal_features": [cold_key, hot_key],
                        "prefix_attention_scope": "thermal_only; action_suffix_readable",
                        "gate": deepcopy(self._last_twogrey_gatevitmix_gate_summary),
                    }
                )
                token_offset += int(num_thermal_embs)
                continue

            if (
                getattr(self.config, "uses_gate_action_thermal_vit", False)
                and key == getattr(self, "rgb_thermal_gateactionvit_cold_feature", None)
            ):
                cold_key = self.rgb_thermal_gateactionvit_cold_feature
                hot_key = self.rgb_thermal_gateactionvit_hot_feature
                if hot_key not in image_inputs_by_key:
                    raise ValueError(
                        "PI05 thermal_encoder_channel='GateActionViT' requires cold/hot "
                        f"thermal image features {cold_key!r} and {hot_key!r}. "
                        f"Available image features: {image_keys}."
                    )
                if lang_emb is None:
                    raise RuntimeError("GateActionViT expected precomputed language embeddings.")
                (
                    cold_emb,
                    cold_token_mask,
                    hot_emb,
                    hot_token_mask,
                ) = self._embed_twogrey_gateactionvit_images(
                    cold_img=img,
                    hot_img=image_inputs_by_key[hot_key],
                    cold_img_mask=img_mask,
                    hot_img_mask=image_masks_by_key[hot_key],
                    text_tokens=lang_emb,
                    text_token_mask=masks,
                )

                num_cold_embs = cold_emb.shape[1]
                embs.append(cold_emb)
                pad_masks.append(cold_token_mask)
                att_masks += [0] * num_cold_embs
                prefix_token_layout.append(
                    {
                        "kind": "image_gateactionvit_twogrey_cold",
                        "name": cold_key,
                        "thermal_source_feature": self.rgb_thermal_gateactionvit_source_feature,
                        "start": int(token_offset),
                        "end": int(token_offset + num_cold_embs),
                        "token_count": int(num_cold_embs),
                        "valid": bool(cold_token_mask.detach().any().item()),
                        "thermal_features": [cold_key, hot_key],
                        "action_attention_bias_scope": "action_queries_to_cold_hot_keys",
                        "gate": deepcopy(self._last_twogrey_gateactionvit_gate_summary),
                    }
                )
                token_offset += int(num_cold_embs)

                num_hot_embs = hot_emb.shape[1]
                embs.append(hot_emb)
                pad_masks.append(hot_token_mask)
                att_masks += [0] * num_hot_embs
                prefix_token_layout.append(
                    {
                        "kind": "image_gateactionvit_twogrey_hot",
                        "name": hot_key,
                        "thermal_source_feature": self.rgb_thermal_gateactionvit_source_feature,
                        "start": int(token_offset),
                        "end": int(token_offset + num_hot_embs),
                        "token_count": int(num_hot_embs),
                        "valid": bool(hot_token_mask.detach().any().item()),
                        "thermal_features": [cold_key, hot_key],
                        "action_attention_bias_scope": "action_queries_to_cold_hot_keys",
                        "gate_role": "hot",
                    }
                )
                token_offset += int(num_hot_embs)
                continue

            if (
                getattr(self.config, "uses_gate_thermal_vit", False)
                and key == self.rgb_thermal_gatevit_cold_feature
            ):
                cold_key = self.rgb_thermal_gatevit_cold_feature
                hot_key = self.rgb_thermal_gatevit_hot_feature
                if hot_key not in image_inputs_by_key:
                    raise ValueError(
                        "PI05 thermal_encoder_channel='GateViT' requires cold/hot thermal "
                        f"image features {cold_key!r} and {hot_key!r}. "
                        f"Available image features: {image_keys}."
                    )
                if lang_emb is None:
                    raise RuntimeError("GateViT expected precomputed language embeddings.")
                img_emb, token_mask = self._embed_twogrey_gatevit_image(
                    cold_img=img,
                    hot_img=image_inputs_by_key[hot_key],
                    cold_img_mask=img_mask,
                    hot_img_mask=image_masks_by_key[hot_key],
                    text_tokens=lang_emb,
                    text_token_mask=masks,
                )
                num_img_embs = img_emb.shape[1]
                embs.append(img_emb)
                pad_masks.append(token_mask)
                att_masks += [0] * num_img_embs
                prefix_token_layout.append(
                    {
                        "kind": "image_gatevit_twogrey",
                        "name": self.rgb_thermal_gatevit_source_feature,
                        "start": int(token_offset),
                        "end": int(token_offset + num_img_embs),
                        "token_count": int(num_img_embs),
                        "valid": bool(token_mask.detach().any().item()),
                        "thermal_features": [cold_key, hot_key],
                        "gate": deepcopy(self._last_twogrey_gatevit_gate_summary),
                    }
                )
                token_offset += int(num_img_embs)
                continue

            if (
                (self.config.uses_thermal_resnet18_encoder or self.config.uses_anythermal_encoder)
                and self.thermal_encoder is not None
                and key in self.thermal_image_feature_set
            ):

                def image_embed_func(img):
                    return self.thermal_encoder(img)

            elif (
                self.thermal_residual_fusion is not None
                and key == self.rgb_thermal_residual_head_feature
            ):
                thermal_key = self.rgb_thermal_residual_thermal_feature
                thermal_img = image_inputs_by_key[thermal_key]
                thermal_img_mask = image_masks_by_key[thermal_key]
                img_emb, token_mask = self._embed_resthermal_head_image(
                    rgb_img=img,
                    thermal_img=thermal_img,
                    rgb_img_mask=img_mask,
                    thermal_img_mask=thermal_img_mask,
                )
                num_img_embs = img_emb.shape[1]
                embs.append(img_emb)
                pad_masks.append(token_mask)
                att_masks += [0] * num_img_embs
                prefix_token_layout.append(
                    {
                        "kind": "image_resthermal_residual",
                        "name": key,
                        "start": int(token_offset),
                        "end": int(token_offset + num_img_embs),
                        "token_count": int(num_img_embs),
                        "valid": bool(token_mask.detach().any().item()),
                        "thermal_feature": thermal_key,
                    }
                )
                token_offset += int(num_img_embs)
                continue

            elif (
                getattr(self.config, "uses_patch_single_gate_residual_thermal_vit", False)
                and key == self.rgb_thermal_patchsinglegateresvit_head_feature
            ):
                cold_key = self.rgb_thermal_patchsinglegateresvit_cold_feature
                hot_key = self.rgb_thermal_patchsinglegateresvit_hot_feature
                if cold_key not in image_inputs_by_key or hot_key not in image_inputs_by_key:
                    raise ValueError(
                        "PI05 thermal_encoder_channel='PatchSingleGateResViT' requires cold/hot "
                        f"thermal image features {cold_key!r} and {hot_key!r}. "
                        f"Available image features: {image_keys}."
                    )
                if lang_emb is None:
                    raise RuntimeError("PatchSingleGateResViT expected precomputed language embeddings.")
                img_emb, token_mask = self._embed_twogrey_patchsinglegateresvit_head_image(
                    rgb_img=img,
                    cold_img=image_inputs_by_key[cold_key],
                    hot_img=image_inputs_by_key[hot_key],
                    rgb_img_mask=img_mask,
                    cold_img_mask=image_masks_by_key[cold_key],
                    hot_img_mask=image_masks_by_key[hot_key],
                    text_tokens=lang_emb,
                    text_token_mask=masks,
                )
                num_img_embs = img_emb.shape[1]
                embs.append(img_emb)
                pad_masks.append(token_mask)
                att_masks += [0] * num_img_embs
                prefix_token_layout.append(
                    {
                        "kind": "image_patchsinglegateresvit_twogrey_residual",
                        "name": key,
                        "start": int(token_offset),
                        "end": int(token_offset + num_img_embs),
                        "token_count": int(num_img_embs),
                        "valid": bool(token_mask.detach().any().item()),
                        "thermal_features": [cold_key, hot_key],
                        "gate": deepcopy(self._last_twogrey_patchsinglegateresvit_gate_summary),
                    }
                )
                token_offset += int(num_img_embs)
                continue

            elif (
                getattr(self.config, "uses_small_patch_gate_residual_thermal_vit", False)
                and key == self.rgb_thermal_smallpatchgateresvit_head_feature
            ):
                cold_key = self.rgb_thermal_smallpatchgateresvit_cold_feature
                hot_key = self.rgb_thermal_smallpatchgateresvit_hot_feature
                if cold_key not in image_inputs_by_key or hot_key not in image_inputs_by_key:
                    raise ValueError(
                        "PI05 thermal_encoder_channel='SmallPatchGateResViT' requires cold/hot "
                        f"thermal image features {cold_key!r} and {hot_key!r}. "
                        f"Available image features: {image_keys}."
                    )
                if lang_emb is None:
                    raise RuntimeError("SmallPatchGateResViT expected precomputed language embeddings.")
                img_emb, token_mask = self._embed_twogrey_smallpatchgateresvit_head_image(
                    rgb_img=img,
                    cold_img=image_inputs_by_key[cold_key],
                    hot_img=image_inputs_by_key[hot_key],
                    rgb_img_mask=img_mask,
                    cold_img_mask=image_masks_by_key[cold_key],
                    hot_img_mask=image_masks_by_key[hot_key],
                    text_tokens=lang_emb,
                    text_token_mask=masks,
                )
                num_img_embs = img_emb.shape[1]
                embs.append(img_emb)
                pad_masks.append(token_mask)
                att_masks += [0] * num_img_embs
                prefix_token_layout.append(
                    {
                        "kind": (
                            "image_smallpatchsinglegateresvit_twogrey_residual"
                            if self.config.uses_small_patch_single_gate_residual_thermal_vit
                            else "image_smallpatchgateresvit_twogrey_residual"
                        ),
                        "name": key,
                        "start": int(token_offset),
                        "end": int(token_offset + num_img_embs),
                        "token_count": int(num_img_embs),
                        "valid": bool(token_mask.detach().any().item()),
                        "thermal_features": [cold_key, hot_key],
                        "gate": deepcopy(self._last_twogrey_smallpatchgateresvit_gate_summary),
                    }
                )
                token_offset += int(num_img_embs)
                continue

            elif (
                getattr(self.config, "uses_gate_residual_thermal_vit3", False)
                and key == self.rgb_thermal_gateresvit3_head_feature
            ):
                cold_key = self.rgb_thermal_gateresvit3_cold_feature
                hot_key = self.rgb_thermal_gateresvit3_hot_feature
                if cold_key not in image_inputs_by_key or hot_key not in image_inputs_by_key:
                    raise ValueError(
                        "PI05 thermal_encoder_channel='GateResViT3' requires cold/hot "
                        f"thermal image features {cold_key!r} and {hot_key!r}. "
                        f"Available image features: {image_keys}."
                    )
                if lang_emb is None:
                    raise RuntimeError("GateResViT3 expected precomputed language embeddings.")
                img_emb, token_mask = self._embed_twogrey_gateresvit3_head_image(
                    rgb_img=img,
                    cold_img=image_inputs_by_key[cold_key],
                    hot_img=image_inputs_by_key[hot_key],
                    rgb_img_mask=img_mask,
                    cold_img_mask=image_masks_by_key[cold_key],
                    hot_img_mask=image_masks_by_key[hot_key],
                    text_tokens=lang_emb,
                    text_token_mask=masks,
                    thermal_stats=twogrey_gateresvit3_stats,
                )
                num_img_embs = img_emb.shape[1]
                embs.append(img_emb)
                pad_masks.append(token_mask)
                att_masks += [0] * num_img_embs
                prefix_token_layout.append(
                    {
                        "kind": "image_gateresvit3_twogrey_residual",
                        "name": key,
                        "start": int(token_offset),
                        "end": int(token_offset + num_img_embs),
                        "token_count": int(num_img_embs),
                        "valid": bool(token_mask.detach().any().item()),
                        "thermal_features": [cold_key, hot_key],
                        "gate": deepcopy(self._last_twogrey_gateresvit3_gate_summary),
                    }
                )
                token_offset += int(num_img_embs)
                continue

            elif (
                getattr(self.config, "uses_gate_one_residual_thermal_vit", False)
                and key == self.rgb_thermal_gateoneresvit_head_feature
            ):
                thermal_key = self.rgb_thermal_gateoneresvit_thermal_feature
                if thermal_key not in image_inputs_by_key:
                    raise ValueError(
                        "PI05 thermal_encoder_channel='GateOneResViT' requires thermal "
                        f"image feature {thermal_key!r}. Available image features: {image_keys}."
                    )
                if lang_emb is None:
                    raise RuntimeError("GateOneResViT expected precomputed language embeddings.")
                img_emb, token_mask = self._embed_grey_gateoneresvit_head_image(
                    rgb_img=img,
                    thermal_img=image_inputs_by_key[thermal_key],
                    rgb_img_mask=img_mask,
                    thermal_img_mask=image_masks_by_key[thermal_key],
                    text_tokens=lang_emb,
                    text_token_mask=masks,
                )
                num_img_embs = img_emb.shape[1]
                embs.append(img_emb)
                pad_masks.append(token_mask)
                att_masks += [0] * num_img_embs
                prefix_token_layout.append(
                    {
                        "kind": "image_gateoneresvit_grey_residual",
                        "name": key,
                        "start": int(token_offset),
                        "end": int(token_offset + num_img_embs),
                        "token_count": int(num_img_embs),
                        "valid": bool(token_mask.detach().any().item()),
                        "head_rgb_feature": key,
                        "thermal_features": [thermal_key],
                        "gate": deepcopy(self._last_grey_gateoneresvit_gate_summary),
                    }
                )
                token_offset += int(num_img_embs)
                continue

            elif (
                getattr(self.config, "uses_gate_res_and_thermal_vit", False)
                and key == self.rgb_thermal_gateresvit_head_feature
            ):
                cold_key = self.rgb_thermal_gateresvit_cold_feature
                hot_key = self.rgb_thermal_gateresvit_hot_feature
                if cold_key not in image_inputs_by_key or hot_key not in image_inputs_by_key:
                    raise ValueError(
                        "PI05 thermal_encoder_channel='GateResandViT' requires cold/hot "
                        f"thermal image features {cold_key!r} and {hot_key!r}. "
                        f"Available image features: {image_keys}."
                    )
                if lang_emb is None:
                    raise RuntimeError("GateResandViT expected precomputed language embeddings.")
                fused_emb, original_rgb_emb, token_mask = (
                    self._embed_twogrey_gateresvit_head_and_original_images(
                        rgb_img=img,
                        cold_img=image_inputs_by_key[cold_key],
                        hot_img=image_inputs_by_key[hot_key],
                        rgb_img_mask=img_mask,
                        cold_img_mask=image_masks_by_key[cold_key],
                        hot_img_mask=image_masks_by_key[hot_key],
                        text_tokens=lang_emb,
                        text_token_mask=masks,
                    )
                )
                for stream_name, stream_emb, stream_kind, stream_gate in (
                    (
                        "thermal_residual",
                        fused_emb,
                        "image_gateresandvit_twogrey_residual",
                        deepcopy(self._last_twogrey_gateresandvit_gate_summary),
                    ),
                    (
                        "original_rgb",
                        original_rgb_emb,
                        "image_gateresandvit_original_rgb",
                        None,
                    ),
                ):
                    num_img_embs = stream_emb.shape[1]
                    embs.append(stream_emb)
                    pad_masks.append(token_mask)
                    att_masks += [0] * num_img_embs
                    prefix_token_layout.append(
                        {
                            "kind": stream_kind,
                            "name": key,
                            "start": int(token_offset),
                            "end": int(token_offset + num_img_embs),
                            "token_count": int(num_img_embs),
                            "valid": bool(token_mask.detach().any().item()),
                            "head_rgb_feature": key,
                            "thermal_features": [cold_key, hot_key],
                            "stream": stream_name,
                            "gate": stream_gate,
                        }
                    )
                    token_offset += int(num_img_embs)
                continue

            elif (
                getattr(self.config, "uses_gate_residual_thermal_vit", False)
                and key == self.rgb_thermal_gateresvit_head_feature
            ):
                cold_key = self.rgb_thermal_gateresvit_cold_feature
                hot_key = self.rgb_thermal_gateresvit_hot_feature
                if cold_key not in image_inputs_by_key or hot_key not in image_inputs_by_key:
                    raise ValueError(
                        "PI05 thermal_encoder_channel='GateResViT' requires cold/hot "
                        f"thermal image features {cold_key!r} and {hot_key!r}. "
                        f"Available image features: {image_keys}."
                    )
                if lang_emb is None:
                    raise RuntimeError("GateResViT expected precomputed language embeddings.")
                img_emb, token_mask = self._embed_twogrey_gateresvit_head_image(
                    rgb_img=img,
                    cold_img=image_inputs_by_key[cold_key],
                    hot_img=image_inputs_by_key[hot_key],
                    rgb_img_mask=img_mask,
                    cold_img_mask=image_masks_by_key[cold_key],
                    hot_img_mask=image_masks_by_key[hot_key],
                    text_tokens=lang_emb,
                    text_token_mask=masks,
                )
                num_img_embs = img_emb.shape[1]
                embs.append(img_emb)
                pad_masks.append(token_mask)
                att_masks += [0] * num_img_embs
                prefix_token_layout.append(
                    {
                        "kind": "image_gateresvit_twogrey_residual",
                        "name": key,
                        "start": int(token_offset),
                        "end": int(token_offset + num_img_embs),
                        "token_count": int(num_img_embs),
                        "valid": bool(token_mask.detach().any().item()),
                        "thermal_features": [cold_key, hot_key],
                        "gate": deepcopy(self._last_twogrey_gateresvit_gate_summary),
                    }
                )
                token_offset += int(num_img_embs)
                continue

            elif (
                self.config.uses_residual_thermal_vit
                and key == self.rgb_thermal_resvit_head_feature
            ):
                if getattr(self.config, "uses_twogrey_thermal_input", False):
                    cold_key = self.rgb_thermal_resvit_twogrey_cold_feature
                    hot_key = self.rgb_thermal_resvit_twogrey_hot_feature
                    if cold_key not in image_inputs_by_key or hot_key not in image_inputs_by_key:
                        raise ValueError(
                            "PI05 thermal_input_type='twogrey' with thermal_encoder_channel='ResViT' "
                            f"requires {cold_key!r} and {hot_key!r} image features. "
                            f"Available image features: {image_keys}."
                        )
                    img_emb, token_mask = self._embed_twogrey_resvit_head_image(
                        rgb_img=img,
                        cold_img=image_inputs_by_key[cold_key],
                        hot_img=image_inputs_by_key[hot_key],
                        rgb_img_mask=img_mask,
                        cold_img_mask=image_masks_by_key[cold_key],
                        hot_img_mask=image_masks_by_key[hot_key],
                    )
                    thermal_layout = {
                        "kind": "image_resvit_twogrey_residual",
                        "thermal_features": [cold_key, hot_key],
                        "thermal_cold_alpha": float(
                            getattr(self.config, "thermal_twogrey_resvit_cold_alpha", 1.0)
                        ),
                        "thermal_hot_beta": float(
                            getattr(self.config, "thermal_twogrey_resvit_hot_beta", 1.0)
                        ),
                    }
                else:
                    thermal_key = self.rgb_thermal_resvit_thermal_feature
                    thermal_img = image_inputs_by_key[thermal_key]
                    thermal_img_mask = image_masks_by_key[thermal_key]
                    img_emb, token_mask = self._embed_resvit_head_image(
                        rgb_img=img,
                        thermal_img=thermal_img,
                        rgb_img_mask=img_mask,
                        thermal_img_mask=thermal_img_mask,
                    )
                    thermal_layout = {
                        "kind": "image_resvit_residual",
                        "thermal_feature": thermal_key,
                    }
                num_img_embs = img_emb.shape[1]
                embs.append(img_emb)
                pad_masks.append(token_mask)
                att_masks += [0] * num_img_embs
                prefix_token_layout.append(
                    {
                        "kind": thermal_layout["kind"],
                        "name": key,
                        "start": int(token_offset),
                        "end": int(token_offset + num_img_embs),
                        "token_count": int(num_img_embs),
                        "valid": bool(token_mask.detach().any().item()),
                        **{field: value for field, value in thermal_layout.items() if field != "kind"},
                    }
                )
                token_offset += int(num_img_embs)
                continue

            elif (
                self.resvit_attention_fusion is not None
                and key == self.rgb_thermal_resvit_attention_head_feature
            ):
                thermal_key = self.rgb_thermal_resvit_attention_thermal_feature
                thermal_img = image_inputs_by_key[thermal_key]
                thermal_img_mask = image_masks_by_key[thermal_key]
                if lang_emb is None:
                    raise RuntimeError("ResViTAttention expected precomputed language embeddings.")
                img_emb, token_mask = self._embed_resvit_attention_head_image(
                    rgb_img=img,
                    thermal_img=thermal_img,
                    rgb_img_mask=img_mask,
                    thermal_img_mask=thermal_img_mask,
                    text_tokens=lang_emb,
                    text_token_mask=masks,
                )
                num_img_embs = img_emb.shape[1]
                embs.append(img_emb)
                pad_masks.append(token_mask)
                att_masks += [0] * num_img_embs
                prefix_token_layout.append(
                    {
                        "kind": "image_resvit_attention",
                        "name": key,
                        "start": int(token_offset),
                        "end": int(token_offset + num_img_embs),
                        "token_count": int(num_img_embs),
                        "valid": bool(token_mask.detach().any().item()),
                        "thermal_feature": thermal_key,
                    }
                )
                token_offset += int(num_img_embs)
                continue

            elif (
                self.rgb_thermal_align_aligner is not None
                and key == self.rgb_thermal_align_head_feature
            ):
                thermal_key = self.rgb_thermal_align_thermal_feature
                thermal_img = image_inputs_by_key[thermal_key]
                thermal_img_mask = image_masks_by_key[thermal_key]
                img_emb, token_mask = self._embed_rgb_thermal_aligned_image(
                    rgb_img=img,
                    thermal_img=thermal_img,
                    rgb_img_mask=img_mask,
                    thermal_img_mask=thermal_img_mask,
                    rgb_thermal_align_target=rgb_thermal_align_target,
                )
                bsize, num_img_embs = img_emb.shape[:2]
                embs.append(img_emb)
                pad_masks.append(token_mask)
                att_masks += [0] * num_img_embs
                prefix_token_layout.append(
                    {
                        "kind": "image_rgb_thermal_aligned",
                        "name": key,
                        "start": int(token_offset),
                        "end": int(token_offset + num_img_embs),
                        "token_count": int(num_img_embs),
                        "valid": bool(token_mask.detach().any().item()),
                        "thermal_feature": thermal_key,
                    }
                )
                token_offset += int(num_img_embs)
                if track_image_embs:
                    image_embs_by_key[key] = img_emb
                    image_token_masks_by_key[key] = token_mask
                continue

            elif (
                self.config.uses_thermal_cvae_encoder
                and self.thermal_encoder is not None
                and key in self.thermal_image_feature_set
            ):
                source_key = self.config.thermal_cvae_source_feature
                source_img = image_inputs_by_key[source_key]
                source_img_mask = image_masks_by_key[source_key]
                img_mask = img_mask & source_img_mask

                def image_embed_func(img, source_img=source_img, source_key=source_key, key=key):
                    return self.thermal_encoder(
                        source_img,
                        img,
                        source_feature=source_key,
                        condition_feature=key,
                    )

            elif (
                self.config.uses_shared_projector_thermal_vit
                and self.thermal_vision_tower is not None
                and key in self.thermal_image_feature_set
            ):

                def image_embed_func(img):
                    return self._embed_shared_projector_thermal_vit_image(img)

            elif (
                self.config.uses_dedicated_thermal_vit
                and self.thermal_vision_tower is not None
                and self.thermal_multi_modal_projector is not None
                and key in self.thermal_image_feature_set
            ):

                def image_embed_func(img):
                    img = self._prepare_thermal_vit_image(img)
                    return self.paligemma_with_expert.embed_image_with_modules(
                        img,
                        self.thermal_vision_tower,
                        self.thermal_multi_modal_projector,
                    )

            else:

                def image_embed_func(img):
                    return self._embed_rgb_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]
            token_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(token_mask)
            att_masks += [0] * num_img_embs
            prefix_token_layout.append(
                {
                    "kind": "image",
                    "name": key,
                    "start": int(token_offset),
                    "end": int(token_offset + num_img_embs),
                    "token_count": int(num_img_embs),
                    "valid": bool(img_mask.detach().any().item()),
                }
            )
            token_offset += int(num_img_embs)
            if track_image_embs:
                image_embs_by_key[key] = img_emb
                image_token_masks_by_key[key] = token_mask

        if track_image_embs:
            head_key = self.config.thermal_fusion_head_rgb_feature
            thermal_keys = [
                key
                for key in image_keys
                if key in self.thermal_image_feature_set and key in image_embs_by_key
            ]
            if head_key in image_embs_by_key and thermal_keys:
                thermal_tokens = torch.cat([image_embs_by_key[key] for key in thermal_keys], dim=1)
                thermal_token_mask = torch.cat(
                    [image_token_masks_by_key[key] for key in thermal_keys], dim=1
                )
                fused_emb, fused_mask = self.thermal_head_fusion(
                    thermal_tokens=thermal_tokens,
                    head_tokens=image_embs_by_key[head_key],
                    thermal_token_mask=thermal_token_mask,
                    head_token_mask=image_token_masks_by_key[head_key],
                )
                embs.append(fused_emb)
                pad_masks.append(fused_mask)
                att_masks += [0] * fused_emb.shape[1]
                prefix_token_layout.append(
                    {
                        "kind": "image_fusion",
                        "name": f"{head_key}_thermal_fusion",
                        "start": int(token_offset),
                        "end": int(token_offset + fused_emb.shape[1]),
                        "token_count": int(fused_emb.shape[1]),
                        "valid": bool(fused_mask.detach().any().item()),
                    }
                )
                token_offset += int(fused_emb.shape[1])

        # Process language tokens
        if lang_emb is None:
            lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        if (
            self.thermal_encoder is not None
            or self.thermal_residual_fusion is not None
            or self.resvit_attention_fusion is not None
            or getattr(self, "twogrey_gatevit_fusion", None) is not None
            or getattr(self, "twogrey_gateactionvit_fusion", None) is not None
            or getattr(self, "twogrey_doublegatevit_fusion", None) is not None
            or getattr(self, "twogrey_doublegatetworesvit_fusion", None) is not None
            or getattr(self, "twogrey_patchgatevit_fusion", None) is not None
            or getattr(self, "twogrey_patchsinglegateresvit_fusion", None) is not None
            or getattr(self, "twogrey_smallpatchgatetworesvit_fusion", None) is not None
            or getattr(self, "twogrey_smallpatchgateresvit_fusion", None) is not None
            or getattr(self, "twogrey_gatevitmix_fusion", None) is not None
            or getattr(self, "twogrey_gateresvit_fusion", None) is not None
            or getattr(self, "twogrey_gatetworesvit_fusion", None) is not None
            or getattr(self, "twogrey_gateresvit3_fusion", None) is not None
            or self.rgb_encoder is not None
            or getattr(self, "rgb_threeblack_gatetworesvit_fusion", None) is not None
        ):
            embs = [emb.to(dtype=lang_emb.dtype) for emb in embs]
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs
        prefix_token_layout.append(
            {
                "kind": "language",
                "name": "language",
                "start": int(token_offset),
                "end": int(token_offset + num_lang_embs),
                "token_count": int(num_lang_embs),
                "valid": bool(masks.detach().any().item()),
            }
        )
        self._last_prefix_token_layout = prefix_token_layout

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise,
        time,
        rgb_thermal_align_target=None,
        twogrey_gateresvit3_stats: Tensor | None = None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        self._last_rgb_thermal_align_loss = None
        self._last_rgb_thermal_align_l1_loss = None
        self._last_rgb_thermal_align_mse_loss = None
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            tokens,
            masks,
            rgb_thermal_align_target=rgb_thermal_align_target,
            twogrey_gateresvit3_stats=twogrey_gateresvit3_stats,
            capture_patchgatevit_summary=True,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        att_2d_masks = self._apply_gatevitmix_prefix_attention_scope(
            att_2d_masks,
            prefix_len=int(prefix_pad_masks.shape[1]),
        )
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        att_2d_masks_4d = self._apply_gateactionvit_action_attention_bias(
            att_2d_masks_4d,
            prefix_len=int(prefix_pad_masks.shape[1]),
        )

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        twogrey_gateresvit3_stats: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device
        capture_attention_map = bool(kwargs.get("capture_attention_map", False))
        if capture_attention_map:
            self._last_attention_map = None

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images,
            img_masks,
            tokens,
            masks,
            twogrey_gateresvit3_stats=twogrey_gateresvit3_stats,
            capture_patchgatevit_summary=capture_attention_map,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_masks = self._apply_gatevitmix_prefix_attention_scope(
            prefix_att_2d_masks,
            prefix_len=int(prefix_pad_masks.shape[1]),
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            if capture_attention_map and step == num_steps - 1:
                try:
                    self._capture_attention_map(
                        prefix_embs=prefix_embs,
                        prefix_pad_masks=prefix_pad_masks,
                        prefix_att_masks=prefix_att_masks,
                        tokens=tokens,
                        masks=masks,
                        x_t=x_t,
                        timestep=time_tensor,
                        denoise_step_idx=step,
                        num_steps=num_steps,
                    )
                except Exception as exc:
                    logging.warning("PI05 attention-map capture failed and will be skipped: %s", exc)
                    self._last_attention_map = None

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        full_att_2d_masks_4d = self._apply_gateactionvit_action_attention_bias(
            full_att_2d_masks_4d,
            prefix_len=int(prefix_len),
            suffix_queries_only=True,
        )
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = clone_past_key_values(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class PI05Policy(PreTrainedPolicy):
    """PI05 Policy for LeRobot."""

    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        require_package("transformers", extra="pi")
        super().__init__(config)
        if getattr(config, "uses_twogrey_thermal_input", False):
            config.rewrite_twogrey_input_features()
        config.validate_features()
        self.config = config
        self._thermal_matcher = None
        if (
            config.thermal_input_type in FIXED_MATCHED_THERMAL_INPUT_TYPES
            and any(
                feature_name not in config.thermal_fixed_matching_homographies
                for feature_name in config.thermal_image_features
            )
        ):
            raise ValueError(
                "thermal_input_type='twofixmatchingblack' requires dataset-level fixed "
                "homographies before PI05Policy construction. Start training through "
                "lerobot_train so it can sample the dataset, or load a checkpoint that "
                "already contains thermal_fixed_matching_homographies."
            )

        if config.mix_rgb_thermal:
            required_mix_features = {
                config.rgb_thermal_mix_rgb_feature,
                config.rgb_thermal_mix_thermal_feature,
            }
            missing_mix_features = required_mix_features.difference(config.image_features)
            if missing_mix_features:
                raise ValueError(
                    "PI05 mix_rgb_thermal=true requires both image features in the policy config. "
                    f"Missing: {sorted(missing_mix_features)}; "
                    f"available: {list(config.image_features)}."
                )
            uses_precomputed_mix = (
                config.rgb_thermal_mix_use_precomputed
                or config.rgb_thermal_mix_source == "precomputed"
            )
            if not uses_precomputed_mix and config.rgb_thermal_mix_fill_color is None:
                raise ValueError(
                    "PI05 mix_rgb_thermal=true requires rgb_thermal_mix_fill_color. "
                    "Training resolves it automatically from the first thermal frame; "
                    "for direct policy construction or inference, provide an RGB tuple."
                )

        episode_fill_colors = torch.tensor(
            config.rgb_thermal_mix_episode_fill_colors,
            dtype=torch.float32,
        ).reshape(-1, 3)
        self.register_buffer(
            "_rgb_thermal_episode_fill_colors",
            episode_fill_colors,
            persistent=False,
        )

        # Initialize the core PI05 model
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        if (
            config.thermal_input_type in ONLINE_MATCHED_THERMAL_INPUT_TYPES
            and config.thermal_matching_preload
        ):
            # Load once during policy construction so the first training batch
            # does not pay model-loading cost and dependency errors fail early.
            self._get_thermal_matcher()

        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        local_pretrained_path = resolve_local_model_path(pretrained_name_or_path)
        if local_pretrained_path is not None:
            pretrained_name_or_path = local_pretrained_path
            local_files_only = True

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Load state dict (expects keys with "model." prefix)
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                resolved_file = None
                checkpoint_file = "model.safetensors"
                cached_file_kwargs = {
                    "cache_dir": cache_dir,
                    "force_download": force_download,
                    "resume_download": resume_download,
                    "proxies": proxies,
                    "token": token,
                    "revision": revision,
                }

                # Prefer an already downloaded checkpoint. Hugging Face's normal path may
                # still attempt a network lookup when local_files_only=False, which is
                # painful for PI0.5 because model.safetensors is ~14.5GB.
                if not force_download:
                    local_checkpoint = Path(pretrained_name_or_path).expanduser() / checkpoint_file
                    if local_checkpoint.is_file():
                        resolved_file = str(local_checkpoint)
                        print(f"✓ Found local {checkpoint_file}: {resolved_file}")
                    else:
                        try:
                            resolved_file = cached_file(
                                pretrained_name_or_path,
                                checkpoint_file,
                                local_files_only=True,
                                **cached_file_kwargs,
                            )
                            print(f"✓ Found cached {checkpoint_file}: {resolved_file}")
                        except Exception as local_error:
                            if local_files_only:
                                raise local_error
                            print(f"No local cached {checkpoint_file} found; downloading from Hub.")

                if resolved_file is None:
                    resolved_file = cached_file(
                        pretrained_name_or_path,
                        checkpoint_file,
                        local_files_only=local_files_only,
                        **cached_file_kwargs,
                    )

                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from local/cache/remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences (see openpi model.py, _fix_pytorch_state_dict_keys)
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # Base PI05 checkpoints legitimately omit optional vision modules.
            uses_optional_vision_modules = getattr(
                model.config, "uses_thermal_resnet18_encoder", False
            ) or getattr(model.config, "uses_thermal_vit_encoder", False) or getattr(
                model.config, "uses_thermal_cvae_encoder", False
            ) or getattr(
                model.config, "uses_anythermal_encoder", False
            ) or getattr(
                model.config, "uses_resthermal_encoder", False
            ) or getattr(
                model.config, "uses_dinov2_rgb_encoder", False
            ) or getattr(
                model.config, "uses_gate_two_residual_rgb_vit", False
            ) or getattr(
                model.config, "uses_patch_single_gate_residual_thermal_vit", False
            ) or getattr(
                model.config, "uses_small_patch_gate_residual_thermal_vit", False
            )
            if uses_optional_vision_modules:
                missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
                allowed_missing_prefixes = (
                    "model.rgb_encoder.",
                    "model.rgb_threeblack_vision_tower.",
                    "model.rgb_threeblack_multi_modal_projector.",
                    "model.rgb_threeblack_gatetworesvit_fusion.",
                    "model.thermal_encoder.",
                    "model.thermal_residual_fusion.",
                    "model.resvit_attention_fusion.",
                    "model.twogrey_gatevit_fusion.",
                    "model.twogrey_gateactionvit_fusion.",
                    "model.twogrey_doublegatevit_fusion.",
                    "model.twogrey_doublegatetworesvit_fusion.",
                    "model.twogrey_patchgatevit_fusion.",
                    "model.twogrey_patchsinglegateresvit_fusion.",
                    "model.twogrey_smallpatchgatetworesvit_fusion.",
                    "model.twogrey_smallpatchgateresvit_fusion.",
                    "model.twogrey_gatevitmix_fusion.",
                    "model.gatevitmix_fusion.",
                    "model.twogrey_gateresvit_fusion.",
                    "model.twogrey_gatetworesvit_fusion.",
                    "model.grey_gateoneresvit_fusion.",
                    "model.twogrey_gateresvit3_fusion.",
                    "model.thermal_head_fusion.",
                    "model.rgb_thermal_align_aligner.",
                    "model.rgb_thermal_align_decoder.",
                    "model.thermal_vision_tower.",
                    "model.thermal_multi_modal_projector.",
                )
                problematic_missing_keys = [
                    key for key in missing_keys if not key.startswith(allowed_missing_prefixes)
                ]
                if strict and (problematic_missing_keys or unexpected_keys):
                    raise RuntimeError(
                        "Error(s) in loading state_dict for PI05Policy with optional vision modules:\n"
                        f"Unexpected keys: {unexpected_keys}\n"
                        f"Problematic missing keys: {problematic_missing_keys}"
                    )
                thermal_vit_missing = any(
                    key.startswith(("model.thermal_vision_tower.", "model.thermal_multi_modal_projector."))
                    for key in missing_keys
                )
                if thermal_vit_missing:
                    model.model.initialize_thermal_vit_from_rgb()
                    print("Initialized thermal ViT weights from the RGB ViT checkpoint weights")
                rgb_threeblack_vit_missing = any(
                    key.startswith(
                        (
                            "model.rgb_threeblack_vision_tower.",
                            "model.rgb_threeblack_multi_modal_projector.",
                        )
                    )
                    for key in missing_keys
                )
                if rgb_threeblack_vit_missing:
                    model.model.initialize_rgb_threeblack_vit_from_rgb()
                    print(
                        "Initialized the independent RGB three-black ViT from the RGB "
                        "checkpoint weights"
                    )
            else:
                missing_keys, unexpected_keys = model.load_state_dict(
                    remapped_state_dict, strict=strict
                )

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load state dict: {e}")

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self):
        if not getattr(self.config, "uses_thermal_cvae_encoder", False):
            return self.parameters()

        cvae_params = []
        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("model.thermal_encoder."):
                cvae_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if other_params:
            param_groups.append({"params": other_params})
        if cvae_params:
            param_groups.append(
                {
                    "params": cvae_params,
                    "lr": self.config.optimizer_lr * 0.1,
                }
            )
        return param_groups

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        self._logged_missing_image_features = False
        self._logged_image_input_summary = False
        self._logged_rgb_thermal_mix = False
        self._logged_thermal_grey_input = False
        self._logged_thermal_twogrey_input = False
        self._logged_thermal_matching_input = False
        self._last_twogrey_gateresvit3_stats = None
        if hasattr(self, "model"):
            self.model.pop_last_attention_map()
            self.model._last_rgb_threeblack_gatetworesvit_gate_summary = None
            self.model._last_twogrey_gatevit_gate_summary = None
            self.model._last_twogrey_gateactionvit_gate_summary = None
            self.model._current_gateactionvit_attention_bias = None
            self.model._last_twogrey_doublegatevit_gate_summary = None
            self.model._last_twogrey_doublegatetworesvit_gate_summary = None
            self.model._last_twogrey_patchgatevit_gate_summary = None
            self.model._last_twogrey_patchsinglegateresvit_gate_summary = None
            self.model._last_twogrey_smallpatchgatetworesvit_gate_summary = None
            self.model._last_twogrey_smallpatchsinglegatetworesvit_gate_summary = None
            self.model._last_twogrey_smallpatchgateresvit_gate_summary = None
            self.model._last_twogrey_gatevitmix_gate_summary = None
            self.model._last_twogrey_gateresvit_gate_summary = None
            self.model._last_twogrey_gateresandvit_gate_summary = None
            self.model._last_twogrey_gatetworesvit_gate_summary = None
            self.model._last_grey_gateoneresvit_gate_summary = None
            self.model._last_twogrey_gateresvit3_gate_summary = None

    def pop_last_attention_map(self):
        return self.model.pop_last_attention_map()

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _get_thermal_matcher(self) -> MinimaThermalMatcher:
        matcher = getattr(self, "_thermal_matcher", None)
        if matcher is None:
            matcher = MinimaThermalMatcher(
                minima_root=self.config.thermal_matching_minima_root,
                checkpoint=self.config.thermal_matching_checkpoint,
                ransac_reproj_threshold=self.config.thermal_matching_ransac_reproj_threshold,
                min_matches=self.config.thermal_matching_min_matches,
                min_inliers=self.config.thermal_matching_min_inliers,
                match_every_n_frames=self.config.thermal_matching_match_every_n_frames,
                cache_size=self.config.thermal_matching_cache_size,
                outer_padding=self.config.thermal_matching_outer_padding,
            )
            self._thermal_matcher = matcher
        return matcher

    def _preprocess_image_tensor(self, img: Tensor, key: str, device: torch.device) -> Tensor:
        if img.device != device:
            img = img.to(device)

        if img.dtype != torch.float32:
            img = img.to(torch.float32)

        if img.ndim != 4:
            raise ValueError(
                f"Image feature {key} must have shape [B,C,H,W] or [B,H,W,C], "
                f"got {img.shape}."
            )

        is_channels_first = img.shape[1] in {1, 3, 4}
        if is_channels_first:
            img = img.permute(0, 2, 3, 1)

        if img.shape[1:3] != self.config.image_resolution:
            img = resize_with_pad_torch(
                img,
                *self.config.image_resolution,
                legacy_float_padding=self.config.legacy_resize_padding,
            )

        img = img * 2.0 - 1.0

        if is_channels_first:
            img = img.permute(0, 3, 1, 2)

        return img

    def _preprocess_rgb_thermal_align_target(self, batch: dict[str, Tensor]) -> Tensor | None:
        if not self.config.rgb_thermal_align_fusion:
            return None

        target_key = self.config.rgb_thermal_align_gt_feature
        if target_key not in batch:
            raise ValueError(
                "PI05 rgb_thermal_align_fusion=true requires decoder GT image feature "
                f"{target_key!r} in the training batch. Available batch keys: {list(batch)}."
            )

        device = next(self.parameters()).device
        return self._preprocess_image_tensor(batch[target_key], target_key, device)

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []
        self._last_twogrey_gateresvit3_stats = None
        thermal_input_type = getattr(self.config, "thermal_input_type", "false")
        matched_backgrounds: dict[str, Tensor] = {}
        matched_alphas: dict[str, Tensor] = {}

        if thermal_input_type in MATCHED_THERMAL_INPUT_TYPES:
            batch = dict(batch)
            rgb_key = self.config.rgb_thermal_mix_rgb_feature
            if rgb_key not in batch:
                raise ValueError(
                    f"PI05 thermal_input_type={thermal_input_type!r} requires head RGB feature "
                    f"{rgb_key!r}. Available batch keys: {list(batch)}."
                )
            matcher = (
                self._get_thermal_matcher()
                if thermal_input_type in ONLINE_MATCHED_THERMAL_INPUT_TYPES
                else None
            )
            matched_features = []
            total_matches = 0
            total_cache_hits = 0
            for thermal_key in self.config.thermal_image_features:
                if thermal_key not in batch:
                    continue
                source_thermal = batch[thermal_key]
                if thermal_input_type in BLACK_IMPORTANCE_THERMAL_INPUT_TYPES:
                    matched_backgrounds[thermal_key] = estimate_thermal_black_background_tensor(
                        source_thermal,
                        feature_name=thermal_key,
                        background_roi=self.config.thermal_grey_background_roi,
                        outlier_threshold=self.config.thermal_black_background_outlier_threshold,
                        outlier_ratio=self.config.thermal_black_background_outlier_ratio,
                    )
                if thermal_input_type in FIXED_MATCHED_THERMAL_INPUT_TYPES:
                    source_bchw = image_to_bchw(source_thermal, thermal_key)
                    rgb_bchw = image_to_bchw(batch[rgb_key], rgb_key)
                    stored_homography = self.config.thermal_fixed_matching_homographies.get(
                        thermal_key
                    )
                    stored_source_size = self.config.thermal_fixed_matching_source_sizes.get(
                        thermal_key
                    )
                    stored_output_size = self.config.thermal_fixed_matching_output_sizes.get(
                        thermal_key
                    )
                    if (
                        stored_homography is None
                        or stored_source_size is None
                        or stored_output_size is None
                    ):
                        raise ValueError(
                            "Missing fixed thermal matching data for "
                            f"{thermal_key!r} in the policy checkpoint."
                        )
                    homography = rescale_homography(
                        stored_homography,
                        stored_source_size=stored_source_size,
                        current_source_size=tuple(source_bchw.shape[-2:]),
                        stored_output_size=stored_output_size,
                        current_output_size=tuple(rgb_bchw.shape[-2:]),
                    )
                    aligned, valid_alpha = warp_thermal_batch(
                        source_bchw,
                        homography[None].repeat(int(source_bchw.shape[0]), axis=0),
                        output_size=tuple(rgb_bchw.shape[-2:]),
                        outer_padding=self.config.thermal_matching_outer_padding,
                        feature_name=thermal_key,
                    )
                    match_stats = {"matches": 0, "cache_hits": int(source_bchw.shape[0])}
                else:
                    if matcher is None:
                        raise RuntimeError("Online thermal matcher is not initialized.")
                    aligned, valid_alpha, match_stats = matcher.align_batch(
                        batch[rgb_key],
                        source_thermal,
                        rgb_feature_name=rgb_key,
                        thermal_feature_name=thermal_key,
                        sample_indices=batch.get("index"),
                        episode_indices=batch.get("episode_index"),
                        frame_indices=batch.get("frame_index"),
                        dataset_indices=batch.get("dataset_index"),
                    )
                if thermal_input_type == "matching":
                    # Preserve the matched color thermal representation. Only
                    # the pixels added outside its border fade to black.
                    batch[thermal_key] = aligned * valid_alpha.to(dtype=aligned.dtype)
                else:
                    batch[thermal_key] = aligned
                    matched_alphas[thermal_key] = valid_alpha
                matched_features.append(thermal_key)
                total_matches += match_stats["matches"]
                total_cache_hits += match_stats["cache_hits"]
            if matched_features and not self._logged_thermal_matching_input:
                if thermal_input_type in FIXED_MATCHED_THERMAL_INPUT_TYPES:
                    logging.info(
                        "PI05 thermal_input_type=%r applied checkpoint-fixed homographies "
                        "for %s to %s: batch_frames=%d, outer_padding=%d.",
                        thermal_input_type,
                        matched_features,
                        rgb_key,
                        total_cache_hits,
                        self.config.thermal_matching_outer_padding,
                    )
                else:
                    logging.info(
                        "PI05 thermal_input_type=%r MINIMA-aligned %s to %s: matches=%d, "
                        "cache_hits=%d, match_every_n_frames=%d, outer_padding=%d.",
                        thermal_input_type,
                        matched_features,
                        rgb_key,
                        total_matches,
                        total_cache_hits,
                        self.config.thermal_matching_match_every_n_frames,
                        self.config.thermal_matching_outer_padding,
                    )
                self._logged_thermal_matching_input = True

        if thermal_input_type in TWO_STREAM_THERMAL_INPUT_TYPES:
            batch = dict(batch)
            converted_features = []
            for thermal_key, (cold_key, hot_key) in self.config.thermal_twogrey_feature_map().items():
                if thermal_key not in batch:
                    continue
                if thermal_input_type in BLACK_IMPORTANCE_THERMAL_INPUT_TYPES:
                    cold_image, hot_image = (
                        thermal_to_two_black_background_normalized_grey_tensor(
                            batch[thermal_key],
                            feature_name=thermal_key,
                            background_roi=self.config.thermal_grey_background_roi,
                            target_background=self.config.thermal_grey_background_value,
                            outlier_threshold=(
                                self.config.thermal_black_background_outlier_threshold
                            ),
                            outlier_ratio=self.config.thermal_black_background_outlier_ratio,
                            background=matched_backgrounds.get(thermal_key),
                            valid_alpha=matched_alphas.get(thermal_key),
                        )
                    )
                else:
                    cold_image, hot_image = thermal_to_two_background_normalized_grey_tensor(
                        batch[thermal_key],
                        feature_name=thermal_key,
                        background_roi=self.config.thermal_grey_background_roi,
                        target_background=self.config.thermal_grey_background_value,
                    )
                batch[cold_key] = cold_image
                batch[hot_key] = hot_image
                if (
                    getattr(self.config, "uses_gate_residual_thermal_vit3", False)
                    and self._last_twogrey_gateresvit3_stats is None
                ):
                    stats_input_range = "zero_to_one" if cold_image.is_floating_point() else "uint8"
                    self._last_twogrey_gateresvit3_stats = (
                        TwoGreyGateResViT3Fusion.compute_thermal_stats(
                            cold_images=cold_image,
                            hot_images=hot_image,
                            background_value=self.config.thermal_grey_background_value,
                            input_range=stats_input_range,
                            black_importance=(
                                thermal_input_type in BLACK_IMPORTANCE_THERMAL_INPUT_TYPES
                            ),
                        )
                    )
                converted_features.append((thermal_key, cold_key, hot_key))
            if converted_features and not getattr(self, "_logged_thermal_twogrey_input", False):
                logging.info(
                    "PI05 thermal_input_type=%r split %s using ROI=%s, boundary=%.1f.",
                    thermal_input_type,
                    converted_features,
                    self.config.thermal_grey_background_roi,
                    self.config.thermal_grey_background_value,
                )
                self._logged_thermal_twogrey_input = True

        if thermal_input_type == "grey":
            batch = dict(batch)
            converted_features = []
            for thermal_key in self.config.thermal_image_features:
                if thermal_key not in batch:
                    continue
                batch[thermal_key] = thermal_to_background_normalized_grey_tensor(
                    batch[thermal_key],
                    feature_name=thermal_key,
                    background_roi=self.config.thermal_grey_background_roi,
                    target_background=self.config.thermal_grey_background_value,
                )
                converted_features.append(thermal_key)
            if converted_features and not self._logged_thermal_grey_input:
                logging.info(
                    "PI05 thermal_input_type='grey' converted %s using ROI=%s, target_background=%.1f.",
                    converted_features,
                    self.config.thermal_grey_background_roi,
                    self.config.thermal_grey_background_value,
                )
                self._logged_thermal_grey_input = True

        if self.config.mix_rgb_thermal:
            rgb_key = self.config.rgb_thermal_mix_rgb_feature
            thermal_key = self.config.rgb_thermal_mix_thermal_feature
            precomputed_key = self.config.rgb_thermal_mix_precomputed_feature
            batch = dict(batch)
            can_use_precomputed = (
                self.config.rgb_thermal_mix_source in {"auto", "precomputed"}
                and precomputed_key in batch
            )
            must_use_precomputed = (
                self.config.rgb_thermal_mix_use_precomputed
                or self.config.rgb_thermal_mix_source == "precomputed"
            )

            if can_use_precomputed:
                missing_batch_features = [key for key in (rgb_key, precomputed_key) if key not in batch]
                if missing_batch_features:
                    raise ValueError(
                        "PI05 precomputed RGB/thermal mix cannot run because the batch is missing "
                        f"{missing_batch_features}. Available batch keys: {list(batch)}."
                    )
                batch[thermal_key] = batch[precomputed_key]
            else:
                if must_use_precomputed:
                    raise ValueError(
                        "PI05 RGB/thermal mixing was configured to use a precomputed mixed video, "
                        f"but {precomputed_key!r} is missing from the batch. "
                        f"Available batch keys: {list(batch)}."
                    )

                missing_batch_features = [key for key in (rgb_key, thermal_key) if key not in batch]
                if missing_batch_features:
                    raise ValueError(
                        "PI05 RGB/thermal mixing cannot run because the batch is missing "
                        f"{missing_batch_features}. Available batch keys: {list(batch)}."
                    )

                fill_color: tuple[int, int, int] | Tensor | None = self.config.rgb_thermal_mix_fill_color
                if fill_color is None:
                    raise ValueError(
                        "PI05 online RGB/thermal mixing requires rgb_thermal_mix_fill_color. "
                        "Training resolves it automatically unless a precomputed mixed video is used."
                    )
                episode_indices = batch.get("episode_index")
                if self._rgb_thermal_episode_fill_colors.numel() > 0 and episode_indices is not None:
                    episode_indices = torch.as_tensor(
                        episode_indices,
                        device=self._rgb_thermal_episode_fill_colors.device,
                        dtype=torch.long,
                    ).flatten()
                    batch_size = int(batch[rgb_key].shape[0])
                    if episode_indices.numel() != batch_size:
                        raise ValueError(
                            "episode_index must contain one value per RGB/thermal sample, "
                            f"got {episode_indices.numel()} indices for batch size {batch_size}."
                        )
                    if (
                        episode_indices.min() < 0
                        or episode_indices.max() >= self._rgb_thermal_episode_fill_colors.shape[0]
                    ):
                        raise ValueError(
                            "episode_index is outside the configured per-episode fill color range: "
                            f"{episode_indices.tolist()}."
                        )
                    fill_color = self._rgb_thermal_episode_fill_colors[episode_indices]

                batch[thermal_key] = mix_rgb_and_thermal_images(
                    batch[rgb_key],
                    batch[thermal_key],
                    rgb_feature=rgb_key,
                    thermal_feature=thermal_key,
                    horizontal_offset=self.config.rgb_thermal_mix_horizontal_offset,
                    vertical_offset=self.config.rgb_thermal_mix_vertical_offset,
                    thermal_width=self.config.rgb_thermal_mix_thermal_width,
                    thermal_weight=self.config.rgb_thermal_mix_thermal_weight,
                    fill_color=fill_color,
                )
            if not self._logged_rgb_thermal_mix:
                if can_use_precomputed:
                    logging.info(
                        "PI05 replaced %s with precomputed RGB/thermal mix feature %s.",
                        thermal_key,
                        precomputed_key,
                    )
                else:
                    logging.info(
                        "PI05 replaced %s with an online RGB/thermal mix: rgb=%s, "
                        "thermal_weight=%.3f, horizontal_offset=%d, vertical_offset=%d, "
                        "thermal_width=%d, episode_fill_colors=%d, fallback_fill_color=%s.",
                        thermal_key,
                        rgb_key,
                        self.config.rgb_thermal_mix_thermal_weight,
                        self.config.rgb_thermal_mix_horizontal_offset,
                        self.config.rgb_thermal_mix_vertical_offset,
                        self.config.rgb_thermal_mix_thermal_width,
                        len(self.config.rgb_thermal_mix_episode_fill_colors),
                        self.config.rgb_thermal_mix_fill_color,
                    )
                self._logged_rgb_thermal_mix = True

        # Get device from model parameters
        device = next(self.parameters()).device

        expected_img_keys = list(self.config.image_features)
        present_img_keys = [key for key in expected_img_keys if key in batch]
        missing_img_keys = [key for key in expected_img_keys if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        if missing_img_keys and not self._logged_missing_image_features:
            logging.warning(
                "PI05 image input is missing feature(s) %s; padding those camera slots with masked "
                "empty images. Present image feature(s): %s.",
                missing_img_keys,
                present_img_keys,
            )
            self._logged_missing_image_features = True

        first_present_img = batch[present_img_keys[0]]
        if first_present_img.ndim < 1:
            raise ValueError(
                f"Image feature {present_img_keys[0]} has invalid shape {first_present_img.shape}."
            )
        batch_size = int(first_present_img.shape[0])

        # Preserve checkpoint camera order. Missing slots stay in place and are masked.
        for key in expected_img_keys:
            if key not in batch:
                feature = self.config.image_features[key]
                shape = tuple(feature.shape)
                channels = (
                    int(shape[0])
                    if shape and shape[0] in {1, 3, 4}
                    else int(shape[-1])
                    if shape and shape[-1] in {1, 3, 4}
                    else 3
                )
                images.append(
                    torch.full(
                        (batch_size, channels, *self.config.image_resolution),
                        -1.0,
                        dtype=torch.float32,
                        device=device,
                    )
                )
                img_masks.append(torch.zeros(batch_size, dtype=torch.bool, device=device))
                continue

            img = self._preprocess_image_tensor(batch[key], key, device)

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        if not self._logged_image_input_summary:
            summaries = []
            for key, img, mask in zip(expected_img_keys, images, img_masks, strict=True):
                img_stats = img.detach()
                summaries.append(
                    {
                        "key": key,
                        "shape": tuple(img.shape),
                        "dtype": str(img.dtype),
                        "mask_true": int(mask.sum().item()),
                        "min": round(float(img_stats.min().item()), 4),
                        "mean": round(float(img_stats.mean().item()), 4),
                        "max": round(float(img_stats.max().item()), 4),
                    }
                )
            context = getattr(self, "_image_input_log_context", "unknown")
            logging.info("PI05 image input slots (%s): %s", context, summaries)
            save_pi05_image_input_snapshots(context, expected_img_keys, images, img_masks)
            self._logged_image_input_summary = True

        return images, img_masks

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            capture_attention_map = bool(getattr(self, "_eval_capture_attention_map", False))
            actions = self.predict_action_chunk(
                batch,
                capture_attention_map=capture_attention_map,
            )[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        twogrey_gateresvit3_stats = getattr(self, "_last_twogrey_gateresvit3_stats", None)

        # Sample actions using the model (pass through RTC kwargs, no separate state needed for PI05)
        actions = self.model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            twogrey_gateresvit3_stats=twogrey_gateresvit3_stats,
            **kwargs,
        )

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        rgb_thermal_align_target = self._preprocess_rgb_thermal_align_target(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        twogrey_gateresvit3_stats = getattr(self, "_last_twogrey_gateresvit3_stats", None)

        actions = self.prepare_action(batch)

        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)

        # Compute loss (no separate state needed for PI05)
        losses = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise,
            time,
            rgb_thermal_align_target=rgb_thermal_align_target,
            twogrey_gateresvit3_stats=twogrey_gateresvit3_stats,
        )

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        action_per_sample_loss = losses.mean(dim=(1, 2))
        action_loss = action_per_sample_loss.mean()

        rgb_thermal_align_loss = self.model._last_rgb_thermal_align_loss
        if rgb_thermal_align_loss is not None:
            rgb_thermal_align_loss = rgb_thermal_align_loss.to(device=action_per_sample_loss.device)
            total_per_sample_loss = (
                action_per_sample_loss
                + self.config.rgb_thermal_align_loss_weight * rgb_thermal_align_loss
            )
        else:
            total_per_sample_loss = action_per_sample_loss

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
            "action_loss": action_loss.item(),
        }
        if rgb_thermal_align_loss is not None:
            align_loss_mean = rgb_thermal_align_loss.mean()
            loss_dict["rgb_thermal_align_loss"] = align_loss_mean.item()
            loss_dict["rgb_thermal_align_weighted_loss"] = (
                self.config.rgb_thermal_align_loss_weight * align_loss_mean
            ).item()
            if self.model._last_rgb_thermal_align_l1_loss is not None:
                loss_dict["rgb_thermal_align_l1_loss"] = (
                    self.model._last_rgb_thermal_align_l1_loss.mean().item()
                )
            if self.model._last_rgb_thermal_align_mse_loss is not None:
                loss_dict["rgb_thermal_align_mse_loss"] = (
                    self.model._last_rgb_thermal_align_mse_loss.mean().item()
                )

        gate_summary = getattr(self.model, "_last_twogrey_gateresvit_gate_summary", None)
        if gate_summary is not None:
            cold_alpha = gate_summary.get("cold_alpha", {})
            hot_beta = gate_summary.get("hot_beta", {})
            cold_multiplier = gate_summary.get("cold_multiplier", {})
            hot_multiplier = gate_summary.get("hot_multiplier", {})
            loss_dict["twogrey_gateresvit_cold_alpha"] = float(cold_alpha.get("mean", 0.0))
            loss_dict["twogrey_gateresvit_hot_beta"] = float(hot_beta.get("mean", 0.0))
            loss_dict["twogrey_gateresvit_cold_multiplier"] = float(
                cold_multiplier.get("mean", 0.0)
            )
            loss_dict["twogrey_gateresvit_hot_multiplier"] = float(hot_multiplier.get("mean", 0.0))
            loss_dict["twogrey_gateresvit_alpha_minus_beta"] = (
                loss_dict["twogrey_gateresvit_cold_alpha"]
                - loss_dict["twogrey_gateresvit_hot_beta"]
            )

        gate_res_and_summary = getattr(
            self.model,
            "_last_twogrey_gateresandvit_gate_summary",
            None,
        )
        if gate_res_and_summary is not None:
            cold_alpha = gate_res_and_summary.get("cold_alpha", {})
            hot_beta = gate_res_and_summary.get("hot_beta", {})
            cold_multiplier = gate_res_and_summary.get("cold_multiplier", {})
            hot_multiplier = gate_res_and_summary.get("hot_multiplier", {})
            loss_dict["twogrey_gateresandvit_cold_alpha"] = float(cold_alpha.get("mean", 0.0))
            loss_dict["twogrey_gateresandvit_hot_beta"] = float(hot_beta.get("mean", 0.0))
            loss_dict["twogrey_gateresandvit_cold_multiplier"] = float(
                cold_multiplier.get("mean", 0.0)
            )
            loss_dict["twogrey_gateresandvit_hot_multiplier"] = float(
                hot_multiplier.get("mean", 0.0)
            )
            loss_dict["twogrey_gateresandvit_alpha_minus_beta"] = (
                loss_dict["twogrey_gateresandvit_cold_alpha"]
                - loss_dict["twogrey_gateresandvit_hot_beta"]
            )

        gatetwores_summary = getattr(self.model, "_last_twogrey_gatetworesvit_gate_summary", None)
        if gatetwores_summary is not None:
            for source_name in (
                "cold_alpha",
                "hot_beta",
                "cold_multiplier",
                "hot_multiplier",
                "cold_residual_scale",
                "hot_residual_scale",
            ):
                summary = gatetwores_summary.get(source_name, {})
                loss_dict[f"twogrey_gatetworesvit_{source_name}"] = float(
                    summary.get("mean", 0.0)
                )
            loss_dict["twogrey_gatetworesvit_alpha_minus_beta"] = (
                loss_dict["twogrey_gatetworesvit_cold_alpha"]
                - loss_dict["twogrey_gatetworesvit_hot_beta"]
            )
            loss_dict["twogrey_gatetworesvit_cold_res_minus_hot_res"] = (
                loss_dict["twogrey_gatetworesvit_cold_residual_scale"]
                - loss_dict["twogrey_gatetworesvit_hot_residual_scale"]
            )

        gateoneres_summary = getattr(self.model, "_last_grey_gateoneresvit_gate_summary", None)
        if gateoneres_summary is not None:
            for source_name in (
                "thermal_alpha",
                "thermal_weight",
                "thermal_residual_scale",
                "logits",
            ):
                summary = gateoneres_summary.get(source_name, {})
                loss_dict[f"grey_gateoneresvit_{source_name}"] = float(summary.get("mean", 0.0))

        rgb_threeblack_summary = getattr(
            self.model,
            "_last_rgb_threeblack_gatetworesvit_gate_summary",
            None,
        )
        if rgb_threeblack_summary is not None:
            for color_name in ("red", "green", "blue"):
                for source_name in ("alpha", "multiplier", "residual_scale"):
                    summary_name = f"{color_name}_{source_name}"
                    summary = rgb_threeblack_summary.get(summary_name, {})
                    loss_dict[f"rgb_threeblack_gatetworesvit_{summary_name}"] = float(
                        summary.get("mean", 0.0)
                    )

        gatevit_summary = getattr(self.model, "_last_twogrey_gatevit_gate_summary", None)
        if gatevit_summary is not None:
            cold_weight = gatevit_summary.get("cold_weight", {})
            hot_weight = gatevit_summary.get("hot_weight", {})
            cold_multiplier = gatevit_summary.get("cold_multiplier", {})
            hot_multiplier = gatevit_summary.get("hot_multiplier", {})
            loss_dict["twogrey_gatevit_cold_weight"] = float(cold_weight.get("mean", 0.0))
            loss_dict["twogrey_gatevit_hot_weight"] = float(hot_weight.get("mean", 0.0))
            loss_dict["twogrey_gatevit_cold_multiplier"] = float(
                cold_multiplier.get("mean", 0.0)
            )
            loss_dict["twogrey_gatevit_hot_multiplier"] = float(
                hot_multiplier.get("mean", 0.0)
            )
            loss_dict["twogrey_gatevit_cold_minus_hot"] = (
                loss_dict["twogrey_gatevit_cold_weight"]
                - loss_dict["twogrey_gatevit_hot_weight"]
            )

        gateactionvit_summary = getattr(
            self.model,
            "_last_twogrey_gateactionvit_gate_summary",
            None,
        )
        if gateactionvit_summary is not None:
            for source_name in (
                "cold_weight",
                "hot_weight",
                "cold_multiplier",
                "hot_multiplier",
                "cold_attention_bias",
                "hot_attention_bias",
                "attention_bias_strength",
            ):
                summary = gateactionvit_summary.get(source_name, {})
                loss_dict[f"twogrey_gateactionvit_{source_name}"] = float(
                    summary.get("mean", 0.0)
                )
            loss_dict["twogrey_gateactionvit_cold_minus_hot"] = (
                loss_dict["twogrey_gateactionvit_cold_weight"]
                - loss_dict["twogrey_gateactionvit_hot_weight"]
            )

        doublegatevit_summary = getattr(
            self.model,
            "_last_twogrey_doublegatevit_gate_summary",
            None,
        )
        if doublegatevit_summary is not None:
            for source_name in (
                "cold_weight",
                "hot_weight",
                "text_cold_weight",
                "text_hot_weight",
                "rgb_cold_alignment_weight",
                "rgb_hot_alignment_weight",
                "rgb_gate_strength",
            ):
                summary = doublegatevit_summary.get(source_name, {})
                loss_dict[f"twogrey_doublegatevit_{source_name}"] = float(
                    summary.get("mean", 0.0)
                )
            loss_dict["twogrey_doublegatevit_cold_minus_hot"] = (
                loss_dict["twogrey_doublegatevit_cold_weight"]
                - loss_dict["twogrey_doublegatevit_hot_weight"]
            )

        doublegate_twores_summary = getattr(
            self.model,
            "_last_twogrey_doublegatetworesvit_gate_summary",
            None,
        )
        if doublegate_twores_summary is not None:
            for source_name in (
                "cold_alpha",
                "hot_beta",
                "cold_weight",
                "hot_weight",
                "cold_multiplier",
                "hot_multiplier",
                "cold_residual_scale",
                "hot_residual_scale",
                "text_cold_weight",
                "text_hot_weight",
                "rgb_cold_alignment_weight",
                "rgb_hot_alignment_weight",
                "rgb_gate_strength",
            ):
                summary = doublegate_twores_summary.get(source_name, {})
                loss_dict[f"twogrey_doublegatetworesvit_{source_name}"] = float(
                    summary.get("mean", 0.0)
                )
            loss_dict["twogrey_doublegatetworesvit_alpha_minus_beta"] = (
                loss_dict["twogrey_doublegatetworesvit_cold_alpha"]
                - loss_dict["twogrey_doublegatetworesvit_hot_beta"]
            )
            loss_dict["twogrey_doublegatetworesvit_cold_res_minus_hot_res"] = (
                loss_dict["twogrey_doublegatetworesvit_cold_residual_scale"]
                - loss_dict["twogrey_doublegatetworesvit_hot_residual_scale"]
            )

        patchgatevit_summary = getattr(
            self.model,
            "_last_twogrey_patchgatevit_gate_summary",
            None,
        )
        if patchgatevit_summary is not None:
            for source_name in (
                "cold_weight",
                "hot_weight",
                "hard_cold_weight",
                "hard_hot_weight",
                "soft_cold_weight",
                "soft_hot_weight",
                "routing_margin",
                "patch_relevance",
                "cold_evidence",
                "hot_evidence",
            ):
                summary = patchgatevit_summary.get(source_name, {})
                loss_dict[f"twogrey_patchgatevit_{source_name}"] = float(
                    summary.get("mean", 0.0)
                )
            loss_dict["twogrey_patchgatevit_cold_minus_hot"] = (
                loss_dict["twogrey_patchgatevit_cold_weight"]
                - loss_dict["twogrey_patchgatevit_hot_weight"]
            )

        patch_single_res_summary = getattr(
            self.model,
            "_last_twogrey_patchsinglegateresvit_gate_summary",
            None,
        )
        if patch_single_res_summary is not None:
            for source_name in (
                "cold_weight",
                "hot_weight",
                "patch_relevance",
                "cold_residual_scale",
                "hot_residual_scale",
                "cold_evidence",
                "hot_evidence",
            ):
                summary = patch_single_res_summary.get(source_name, {})
                loss_dict[f"twogrey_patchsinglegateresvit_{source_name}"] = float(
                    summary.get("mean", 0.0)
                )
            loss_dict["twogrey_patchsinglegateresvit_cold_minus_hot"] = (
                loss_dict["twogrey_patchsinglegateresvit_cold_weight"]
                - loss_dict["twogrey_patchsinglegateresvit_hot_weight"]
            )
            loss_dict["twogrey_patchsinglegateresvit_cold_res_minus_hot_res"] = (
                loss_dict["twogrey_patchsinglegateresvit_cold_residual_scale"]
                - loss_dict["twogrey_patchsinglegateresvit_hot_residual_scale"]
            )

        smallpatch_twores_summary = getattr(
            self.model,
            (
                "_last_twogrey_smallpatchsinglegatetworesvit_gate_summary"
                if self.config.uses_small_patch_single_gate_two_res_thermal_vit
                else "_last_twogrey_smallpatchgatetworesvit_gate_summary"
            ),
            None,
        )
        if smallpatch_twores_summary is not None:
            smallpatch_twores_prefix = (
                "twogrey_smallpatchsinglegatetworesvit"
                if self.config.uses_small_patch_single_gate_two_res_thermal_vit
                else "twogrey_smallpatchgatetworesvit"
            )
            for source_name in (
                "cold_weight",
                "hot_weight",
                "patch_relevance",
                "cold_residual_scale",
                "hot_residual_scale",
                "small_patch_cold_weight",
                "small_patch_hot_weight",
                "small_patch_relevance",
            ):
                summary = smallpatch_twores_summary.get(source_name, {})
                loss_dict[f"{smallpatch_twores_prefix}_{source_name}"] = float(
                    summary.get("mean", 0.0)
                )
            loss_dict[f"{smallpatch_twores_prefix}_cold_minus_hot"] = (
                loss_dict[f"{smallpatch_twores_prefix}_cold_weight"]
                - loss_dict[f"{smallpatch_twores_prefix}_hot_weight"]
            )
            loss_dict[f"{smallpatch_twores_prefix}_cold_res_minus_hot_res"] = (
                loss_dict[f"{smallpatch_twores_prefix}_cold_residual_scale"]
                - loss_dict[f"{smallpatch_twores_prefix}_hot_residual_scale"]
            )

        smallpatch_res_summary = getattr(
            self.model,
            "_last_twogrey_smallpatchgateresvit_gate_summary",
            None,
        )
        if smallpatch_res_summary is not None:
            smallpatch_res_prefix = (
                "twogrey_smallpatchsinglegateresvit"
                if self.config.uses_small_patch_single_gate_residual_thermal_vit
                else "twogrey_smallpatchgateresvit"
            )
            for source_name in (
                "cold_weight",
                "hot_weight",
                "patch_relevance",
                "cold_residual_scale",
                "hot_residual_scale",
                "small_patch_cold_weight",
                "small_patch_hot_weight",
                "small_patch_relevance",
            ):
                summary = smallpatch_res_summary.get(source_name, {})
                loss_dict[f"{smallpatch_res_prefix}_{source_name}"] = float(
                    summary.get("mean", 0.0)
                )
            loss_dict[f"{smallpatch_res_prefix}_cold_minus_hot"] = (
                loss_dict[f"{smallpatch_res_prefix}_cold_weight"]
                - loss_dict[f"{smallpatch_res_prefix}_hot_weight"]
            )
            loss_dict[f"{smallpatch_res_prefix}_cold_res_minus_hot_res"] = (
                loss_dict[f"{smallpatch_res_prefix}_cold_residual_scale"]
                - loss_dict[f"{smallpatch_res_prefix}_hot_residual_scale"]
            )

        gatevitmix_summary = getattr(
            self.model,
            "_last_twogrey_gatevitmix_gate_summary",
            None,
        )
        if gatevitmix_summary is not None:
            for source_name, metric_name in (
                ("cold_weight", "cold_weight"),
                ("hot_weight", "hot_weight"),
                ("cold_multiplier", "cold_multiplier"),
                ("hot_multiplier", "hot_multiplier"),
                ("head_mix_context_rms", "head_mix_context_rms"),
                ("head_mix_delta_rms", "head_mix_delta_rms"),
            ):
                summary = gatevitmix_summary.get(source_name, {})
                loss_dict[f"twogrey_gatevitmix_{metric_name}"] = float(
                    summary.get("mean", 0.0)
                )
            loss_dict["twogrey_gatevitmix_cold_minus_hot"] = (
                loss_dict["twogrey_gatevitmix_cold_weight"]
                - loss_dict["twogrey_gatevitmix_hot_weight"]
            )

        gate3_summary = getattr(self.model, "_last_twogrey_gateresvit3_gate_summary", None)
        if gate3_summary is not None:
            cold_alpha = gate3_summary.get("cold_alpha", {})
            hot_beta = gate3_summary.get("hot_beta", {})
            cold_multiplier = gate3_summary.get("cold_multiplier", {})
            hot_multiplier = gate3_summary.get("hot_multiplier", {})
            cold_weight = gate3_summary.get("cold_weight", {})
            hot_weight = gate3_summary.get("hot_weight", {})
            thermal_confidence = gate3_summary.get("thermal_confidence", {})
            effective_thermal_confidence = gate3_summary.get("effective_thermal_confidence", {})
            thermal_evidence = gate3_summary.get("thermal_evidence", {})
            confidence_logit = gate3_summary.get("confidence_logit", {})
            loss_dict["twogrey_gateresvit3_cold_alpha"] = float(cold_alpha.get("mean", 0.0))
            loss_dict["twogrey_gateresvit3_hot_beta"] = float(hot_beta.get("mean", 0.0))
            loss_dict["twogrey_gateresvit3_cold_multiplier"] = float(
                cold_multiplier.get("mean", 0.0)
            )
            loss_dict["twogrey_gateresvit3_hot_multiplier"] = float(
                hot_multiplier.get("mean", 0.0)
            )
            loss_dict["twogrey_gateresvit3_cold_weight"] = float(cold_weight.get("mean", 0.0))
            loss_dict["twogrey_gateresvit3_hot_weight"] = float(hot_weight.get("mean", 0.0))
            loss_dict["twogrey_gateresvit3_thermal_confidence"] = float(
                thermal_confidence.get("mean", 0.0)
            )
            loss_dict["twogrey_gateresvit3_effective_thermal_confidence"] = float(
                effective_thermal_confidence.get("mean", 0.0)
            )
            loss_dict["twogrey_gateresvit3_thermal_evidence"] = float(
                thermal_evidence.get("mean", 0.0)
            )
            loss_dict["twogrey_gateresvit3_confidence_logit"] = float(
                confidence_logit.get("mean", 0.0)
            )
            loss_dict["twogrey_gateresvit3_alpha_minus_beta"] = (
                loss_dict["twogrey_gateresvit3_cold_alpha"]
                - loss_dict["twogrey_gateresvit3_hot_beta"]
            )

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            loss_dict["loss"] = total_per_sample_loss.mean().item()
            return total_per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = total_per_sample_loss.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for PI0.5 fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }
