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

"""Spatial alignment components of SLA (Spatial-Language-Action model).

Ports of the SLA reference implementation (OpenHelix-Team/Spatial-Forcing, ``SLA/``):

- ``AlignProjector``          <- ``openpi/models/projectors.py`` (JAX)
- ``preprocess_images_for_vggt`` <- ``vggt/utils/load_fn.py: preprocess_images_from_openpi``
- ``pool_vggt_features``       <- ``vggt/heads/utils.py: interpolate_pooling``
- ``make_align_mask``          <- ``openpi/training/align_utils.py: _make_align_mask``
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

VGGT_IMAGE_SIZE = 518
VGGT_PATCH_SIZE = 14


class AlignProjector(nn.Module):
    """Project VLM image-token hidden states and compute the cosine alignment loss.

    Mirrors the JAX ``AlignProjector``: optional LayerNorm, ``Linear -> GELU(tanh) -> Linear``
    into ``2 * vggt_dim`` channels, then ``1 - cos`` per token averaged over valid tokens.
    """

    def __init__(
        self, llm_dim: int, vggt_dim: int, *, use_vlm_norm: bool = True, zscore_targets: bool = False
    ):
        super().__init__()
        target_dim = 2 * vggt_dim  # VGGT aggregator concatenates frame and global tokens.
        # flax.nnx.LayerNorm uses epsilon=1e-6.
        self.vlm_norm = nn.LayerNorm(llm_dim, eps=1e-6) if use_vlm_norm else None
        self.fc1 = nn.Linear(llm_dim, target_dim)
        self.fc2 = nn.Linear(target_dim, target_dim)
        self.zscore_targets = zscore_targets

    def project(self, llm_embedding: Tensor) -> Tensor:
        if self.vlm_norm is not None:
            llm_embedding = self.vlm_norm(llm_embedding)
        # jax.nn.gelu defaults to the tanh approximation.
        return self.fc2(F.gelu(self.fc1(llm_embedding), approximate="tanh"))

    @staticmethod
    def spatial_zscore(feat: Tensor, eps: float = 1e-6) -> Tensor:
        mean = feat.mean(dim=1, keepdim=True)
        std = feat.std(dim=1, keepdim=True, unbiased=False)
        return (feat - mean) / (std + eps)

    def forward(self, llm_embs: Tensor, target_embs: Tensor, align_mask: Tensor) -> Tensor:
        """Return the per-sample alignment loss ``[B]`` (the reference implementation takes its mean).

        Args:
            llm_embs: ``[B, S, llm_dim]`` VLM hidden states of the image tokens.
            target_embs: ``[B, S, 2 * vggt_dim]`` pooled VGGT features (no gradient).
            align_mask: ``[B, S]`` boolean mask of tokens that take part in the loss.
        """
        param_dtype = self.fc1.weight.dtype
        # The projector and loss run in the parameter dtype (float32 by default), as in JAX
        # where bf16 activations are promoted by the float32 projector parameters.
        with torch.autocast(device_type=llm_embs.device.type, enabled=False):
            llm_emb = self.project(llm_embs.to(param_dtype)).float()
            target_emb = target_embs.to(llm_emb.dtype)
            if self.zscore_targets:
                target_emb = self.spatial_zscore(target_emb)
            llm_emb = llm_emb / llm_emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            target_emb = target_emb / target_emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            per_token_loss = 1.0 - (llm_emb * target_emb).sum(dim=-1)
            mask = align_mask.to(per_token_loss.dtype)
            denom = mask.sum(dim=-1).clamp_min(1.0)
            return (per_token_loss * mask).sum(dim=-1) / denom


def resize_with_pad_valid_mask(
    height: int, width: int, target_height: int, target_width: int, device: torch.device
) -> Tensor:
    """Valid-pixel mask of ``resize_with_pad`` (same geometry as openpi/LeRobot ``resize_with_pad``)."""
    mask = torch.zeros(target_height, target_width, dtype=torch.bool, device=device)
    if (height, width) == (target_height, target_width):
        mask.fill_(True)
        return mask
    ratio = max(width / target_width, height / target_height)
    resized_height = int(height / ratio)
    resized_width = int(width / ratio)
    pad_h0 = (target_height - resized_height) // 2
    pad_w0 = (target_width - resized_width) // 2
    mask[pad_h0 : pad_h0 + resized_height, pad_w0 : pad_w0 + resized_width] = True
    return mask


def preprocess_images_for_vggt(images: list[Tensor], target_size: int = VGGT_IMAGE_SIZE) -> Tensor:
    """Resize ``[B, 3, H, W]`` images in ``[0, 1]`` to the VGGT input size (``crop`` mode).

    Returns ``[B, N, 3, target_size, target_size]``.
    """
    if not images:
        raise ValueError("At least 1 image is required")
    height, width = images[0].shape[-2:]
    new_width = target_size
    new_height = round(height * (new_width / width) / VGGT_PATCH_SIZE) * VGGT_PATCH_SIZE
    resized = [
        F.interpolate(img, size=(new_height, new_width), mode="bicubic", align_corners=False)
        for img in images
    ]
    start_y = max((new_height - target_size) // 2, 0)
    cropped = [img[:, :, start_y : start_y + target_size, :] for img in resized]
    return torch.stack(cropped, dim=1)


def _make_sincos_pos_embed(embed_dim: int, pos: Tensor, omega_0: float = 100) -> Tensor:
    omega = torch.arange(embed_dim // 2, dtype=torch.double, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega
    out = torch.einsum("m,d->md", pos.reshape(-1).double(), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()


def _position_grid_to_embed(pos_grid: Tensor, embed_dim: int, omega_0: float = 100) -> Tensor:
    height, width, _ = pos_grid.shape
    pos_flat = pos_grid.reshape(-1, 2)
    emb_x = _make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y = _make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)
    return torch.cat([emb_x, emb_y], dim=-1).view(height, width, embed_dim)


def _create_uv_grid(width: int, height: int, aspect_ratio: float, dtype: torch.dtype, device) -> Tensor:
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor
    x_coords = torch.linspace(
        -span_x * (width - 1) / width, span_x * (width - 1) / width, steps=width, dtype=dtype, device=device
    )
    y_coords = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        steps=height,
        dtype=dtype,
        device=device,
    )
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    return torch.stack((uu, vv), dim=-1)


def _apply_vggt_pos_embed(x: Tensor, img_width: int, img_height: int, ratio: float = 0.1) -> Tensor:
    patch_h, patch_w = x.shape[-2:]
    pos_embed = _create_uv_grid(patch_w, patch_h, img_width / img_height, x.dtype, x.device)
    pos_embed = _position_grid_to_embed(pos_embed, x.shape[1]) * ratio
    pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
    return x + pos_embed


def pool_vggt_features(
    hidden: Tensor,
    patch_hw: tuple[int, int],
    img_hw: tuple[int, int],
    tokens_per_image: int,
    *,
    pooling_func: str = "bilinear",
    use_vggt_pe: bool = True,
) -> Tensor:
    """Resample VGGT patch features ``[B, N, P, D]`` to the VLM token grid ``[B, N * tokens_per_image, D]``."""
    if pooling_func != "bilinear":
        raise NotImplementedError(f"Pooling function {pooling_func} is not implemented.")
    patch_h, patch_w = patch_hw
    img_h, img_w = img_hw
    bsize, num_images, num_patches, dim = hidden.shape
    resample_ratio = 1 / math.sqrt(num_patches / tokens_per_image)
    x = hidden.permute(0, 1, 3, 2).reshape(bsize * num_images, dim, patch_h, patch_w)
    if use_vggt_pe:
        x = _apply_vggt_pos_embed(x, img_w, img_h)
    # The reference passes `scale_factor`; with align_corners=True only the output size matters, and
    # an explicit size avoids floating-point flooring (e.g. 37 * 16/37 -> 15.999...).
    out_size = (round(patch_h * resample_ratio), round(patch_w * resample_ratio))
    x = F.interpolate(x, size=out_size, mode=pooling_func, align_corners=True)
    return x.reshape(bsize, num_images, dim, -1).permute(0, 1, 3, 2).reshape(bsize, -1, dim)


def make_align_mask(
    img_masks: list[Tensor],
    padding_masks: list[Tensor] | None,
    tokens_per_image: int,
    grid_size: int,
) -> Tensor:
    """Token mask ``[B, N * tokens_per_image]``: present cameras, optionally minus padded patches.

    ``grid_size`` is the per-side number of vision patches (224 // 14 = 16 for PaliGemma).
    """
    mask = torch.repeat_interleave(torch.stack(img_masks, dim=1), repeats=tokens_per_image, dim=1)
    if padding_masks is not None:
        padding = torch.stack(padding_masks, dim=1)  # [B, N, H, W]
        downsampled = (
            F.interpolate(padding.float(), size=(grid_size, grid_size), mode="nearest").bool().flatten(1)
        )
        if downsampled.shape != mask.shape:
            raise ValueError(
                f"Alignment token mask {tuple(mask.shape)} does not match padding mask "
                f"{tuple(downsampled.shape)}; check the image resolution / patch size."
            )
        mask = mask & downsampled
    return mask
