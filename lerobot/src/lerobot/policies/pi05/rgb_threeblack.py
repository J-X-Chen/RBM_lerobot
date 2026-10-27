"""RGB-only color evidence and text-gated residual fusion for S1/G1."""

import torch
from torch import Tensor, nn

from .rgb_encoder import image_to_bchw


def rgb_to_three_black_dominance_tensor(
    images: Tensor,
    *,
    feature_name: str,
    input_range: str = "auto",
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert RGB into red/green/blue black-importance views.

    Each view measures how much its selected channel exceeds the other two
    channels. Strong channel-specific evidence maps to black, while neutral
    pixels (including white/grey background) map to white. Outputs are
    three-channel tensors that preserve the input's layout, numeric range,
    device, and dtype.
    """
    if images.ndim != 4:
        raise ValueError(
            f"RGB three-black input {feature_name} must have shape [B,C,H,W] or "
            f"[B,H,W,C], got {tuple(images.shape)}."
        )
    if input_range not in {"auto", "minus_one_to_one", "zero_to_one", "uint8"}:
        raise ValueError("input_range must be one of: auto, minus_one_to_one, zero_to_one, uint8")

    channels_first = images.shape[1] in {3, 4}
    images_bchw = image_to_bchw(images, feature_name)
    if images_bchw.shape[1] not in {3, 4}:
        raise ValueError(
            f"RGB three-black input {feature_name} requires three or four channels, "
            f"got {images_bchw.shape[1]}."
        )

    rgb = images_bchw[:, :3].to(dtype=torch.float32)
    resolved_range = input_range
    if resolved_range == "auto":
        if not images_bchw.is_floating_point():
            resolved_range = "uint8"
        else:
            min_value = float(rgb.detach().amin().item())
            max_value = float(rgb.detach().amax().item())
            if min_value < -0.05:
                resolved_range = "minus_one_to_one"
            elif max_value <= 1.0 + 1e-5:
                resolved_range = "zero_to_one"
            else:
                resolved_range = "uint8"

    if resolved_range == "minus_one_to_one":
        rgb_unit = (rgb + 1.0) * 0.5
    elif resolved_range == "zero_to_one":
        rgb_unit = rgb
    else:
        rgb_unit = rgb / 255.0
    rgb_unit = rgb_unit.clamp(0.0, 1.0)

    views = []
    for channel_index in range(3):
        other_indices = [index for index in range(3) if index != channel_index]
        other_max = rgb_unit[:, other_indices].amax(dim=1, keepdim=True)
        color_evidence = (rgb_unit[:, channel_index : channel_index + 1] - other_max).clamp(0.0, 1.0)
        black_importance = 1.0 - color_evidence
        view = black_importance.repeat(1, 3, 1, 1)
        if resolved_range == "minus_one_to_one":
            view = view * 2.0 - 1.0
        elif resolved_range == "uint8":
            view = torch.round(view * 255.0)
        view = view.to(dtype=images.dtype)
        if not channels_first:
            view = view.permute(0, 2, 3, 1).contiguous()
        views.append(view)

    return views[0], views[1], views[2]


class _RGBFusionHelpers:
    @staticmethod
    def _validate_token_shape(name: str, tokens: Tensor, embed_dim: int) -> None:
        if tokens.ndim != 3 or tokens.shape[-1] != embed_dim:
            raise ValueError(f"{name} must have shape [B, N, {embed_dim}], got {tuple(tokens.shape)}.")

    @staticmethod
    def _validate_mask_shape(name: str, mask: Tensor, expected_shape: tuple[int, int]) -> None:
        if mask.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(mask.shape)}.")

    @staticmethod
    def _masked_mean(tokens: Tensor, mask: Tensor) -> Tensor:
        mask = mask.to(device=tokens.device, dtype=torch.bool)
        masked = tokens.masked_fill(~mask[:, :, None], 0.0)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=tokens.dtype)
        return masked.sum(dim=1) / denom

    @staticmethod
    def _safe_cross_attention(
        attn: nn.MultiheadAttention,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if key.shape[1] == 0:
            valid = torch.zeros(query.shape[0], dtype=torch.bool, device=query.device)
            return torch.zeros_like(query), valid

        valid = key_mask.any(dim=1)
        # MultiheadAttention cannot consume a row with every key masked. Make
        # key 0 temporarily visible for those rows, then zero their outputs.
        # This tensor-only form stays inside torch.compile graphs.
        safe_key_mask = key_mask.clone()
        safe_key_mask[:, 0] = safe_key_mask[:, 0] | ~valid

        context, _ = attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=~safe_key_mask,
            need_weights=False,
        )
        context = context.masked_fill(~valid[:, None, None], 0.0)
        return context, valid


class RGBThreeBlackGateThreeResViTFusion(nn.Module):
    """Emit red/green/blue text-gated residual streams on one base RGB token grid."""

    _COLOR_NAMES = ("red", "green", "blue")

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int = 8,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        temperature: float = 1.0,
        gate_init_std: float = 0.02,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError(f"num_heads must divide embed_dim ({embed_dim}), got {num_heads}.")
        if hidden_dim is None:
            hidden_dim = max(128, embed_dim // 4)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if gate_init_std < 0:
            raise ValueError("gate_init_std must be non-negative")

        self.temperature = float(temperature)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.color_norm = nn.LayerNorm(embed_dim)
        self.text_to_color_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        gate_input_dim = embed_dim * 7
        second_hidden_dim = max(64, hidden_dim // 2)
        self.gate_input_norm = nn.LayerNorm(gate_input_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, second_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden_dim, 3),
        )
        nn.init.normal_(self.gate_mlp[-1].weight, mean=0.0, std=gate_init_std)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(
        self,
        *,
        rgb_tokens: Tensor,
        red_tokens: Tensor,
        green_tokens: Tensor,
        blue_tokens: Tensor,
        text_tokens: Tensor,
        rgb_token_mask: Tensor,
        red_token_mask: Tensor,
        green_token_mask: Tensor,
        blue_token_mask: Tensor,
        text_token_mask: Tensor,
        base_red_alpha: float = 1.0,
        base_green_alpha: float = 1.0,
        base_blue_alpha: float = 1.0,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        dict[str, Tensor],
    ]:
        embed_dim = self.text_norm.normalized_shape[0]
        _RGBFusionHelpers._validate_token_shape("rgb_tokens", rgb_tokens, embed_dim)
        _RGBFusionHelpers._validate_token_shape("text_tokens", text_tokens, embed_dim)
        color_tokens = (red_tokens, green_tokens, blue_tokens)
        color_masks = (red_token_mask, green_token_mask, blue_token_mask)
        for color_name, tokens in zip(self._COLOR_NAMES, color_tokens, strict=True):
            _RGBFusionHelpers._validate_token_shape(f"{color_name}_tokens", tokens, embed_dim)
            if tokens.shape != rgb_tokens.shape:
                raise ValueError(
                    "RGB GateTwoResViT requires aligned original/red/green/blue token grids, "
                    f"got rgb={tuple(rgb_tokens.shape)} and "
                    f"{color_name}={tuple(tokens.shape)}."
                )
        if text_tokens.shape[0] != rgb_tokens.shape[0]:
            raise ValueError(
                "RGB GateTwoResViT image and text batch sizes must match, "
                f"got rgb={rgb_tokens.shape[0]} and text={text_tokens.shape[0]}."
            )

        batch_size, num_rgb_tokens = rgb_tokens.shape[:2]
        num_text_tokens = text_tokens.shape[1]
        rgb_token_mask = rgb_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        text_token_mask = text_token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
        _RGBFusionHelpers._validate_mask_shape("rgb_token_mask", rgb_token_mask, (batch_size, num_rgb_tokens))
        _RGBFusionHelpers._validate_mask_shape(
            "text_token_mask", text_token_mask, (batch_size, num_text_tokens)
        )

        module_dtype = self.text_norm.weight.dtype
        rgb_in = rgb_tokens.to(dtype=module_dtype)
        text_in = text_tokens.to(device=rgb_tokens.device, dtype=module_dtype)
        text_norm = self.text_norm(text_in)
        text_valid = text_token_mask.any(dim=1)
        text_pool = _RGBFusionHelpers._masked_mean(text_norm, text_token_mask)

        color_inputs = []
        normalized_color_masks = []
        color_validity = []
        color_pools = []
        for color_name, tokens, token_mask in zip(
            self._COLOR_NAMES,
            color_tokens,
            color_masks,
            strict=True,
        ):
            token_mask = token_mask.to(device=rgb_tokens.device, dtype=torch.bool)
            _RGBFusionHelpers._validate_mask_shape(
                f"{color_name}_token_mask", token_mask, (batch_size, num_rgb_tokens)
            )
            color_in = tokens.to(device=rgb_tokens.device, dtype=module_dtype)
            color_norm = self.color_norm(color_in)
            color_context, color_valid = _RGBFusionHelpers._safe_cross_attention(
                self.text_to_color_attn,
                query=text_norm,
                key=color_norm,
                value=color_norm,
                key_mask=token_mask,
            )
            color_inputs.append(color_in)
            normalized_color_masks.append(token_mask)
            color_validity.append(color_valid)
            color_pools.append(_RGBFusionHelpers._masked_mean(color_context, text_token_mask))

        gate_input = torch.cat(
            [
                text_pool,
                *color_pools,
                *(color_pool - text_pool for color_pool in color_pools),
            ],
            dim=-1,
        )
        logits = self.gate_mlp(self.gate_input_norm(gate_input))
        multipliers = 3.0 * torch.softmax(logits / self.temperature, dim=-1)
        base_alphas = (base_red_alpha, base_green_alpha, base_blue_alpha)

        output_tokens = []
        output_masks = []
        gate_info: dict[str, Tensor] = {
            "logits": logits.detach(),
            "text_valid": text_valid.detach(),
        }
        for color_index, (
            color_name,
            color_in,
            color_mask,
            color_valid,
            base_alpha,
        ) in enumerate(
            zip(
                self._COLOR_NAMES,
                color_inputs,
                normalized_color_masks,
                color_validity,
                base_alphas,
                strict=True,
            )
        ):
            alpha = (
                float(base_alpha)
                * multipliers[:, color_index]
                * color_valid.to(dtype=module_dtype)
                * text_valid.to(dtype=module_dtype)
            )
            output_mask = rgb_token_mask & color_mask
            residual = color_in.masked_fill(~color_mask[:, :, None], 0.0)
            output = rgb_in + alpha[:, None, None] * residual
            output = output.masked_fill(~output_mask[:, :, None], 0.0)
            output_tokens.append(output.to(dtype=rgb_tokens.dtype))
            output_masks.append(output_mask)
            gate_info[f"{color_name}_alpha"] = alpha.detach()
            gate_info[f"{color_name}_multiplier"] = multipliers[:, color_index].detach()
            gate_info[f"{color_name}_residual_scale"] = alpha.detach()
            gate_info[f"{color_name}_valid"] = color_valid.detach()
            gate_info[f"{color_name}_residual_valid"] = output_mask.any(dim=1).detach()

        return (
            output_tokens[0],
            output_masks[0],
            output_tokens[1],
            output_masks[1],
            output_tokens[2],
            output_masks[2],
            gate_info,
        )
