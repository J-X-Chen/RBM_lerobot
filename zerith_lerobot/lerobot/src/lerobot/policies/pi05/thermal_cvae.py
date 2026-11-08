#!/usr/bin/env python

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

"""Cold-water-bottle CVAE thermal encoder copied into the PI0.5 package."""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .thermal_utils import image_to_bchw


class PI05TokenProjector(nn.Module):
    """Project a spatial CVAE latent map into PaliGemma/Gemma visual tokens."""

    def __init__(
        self,
        in_channels: int,
        output_dim: int = 2048,
        output_grid_size: int = 16,
        upsample_mode: str = "bilinear",
        norm_type: str = "layernorm",
        scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.output_grid_size = output_grid_size
        self.upsample_mode = upsample_mode

        self.proj = nn.Conv2d(in_channels, output_dim, kernel_size=1)
        if norm_type == "layernorm":
            self.norm = nn.LayerNorm(output_dim)
        elif norm_type in ("none", None):
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported token projector norm_type: {norm_type}")

        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, latent: Tensor) -> Tensor:
        if latent.ndim != 4:
            raise ValueError(f"Expected latent shape [B, C, H, W], got {tuple(latent.shape)}")

        if latent.shape[-2:] != (self.output_grid_size, self.output_grid_size):
            interpolate_kwargs = {}
            if self.upsample_mode in ("linear", "bilinear", "bicubic", "trilinear"):
                interpolate_kwargs["align_corners"] = False
            latent = F.interpolate(
                latent,
                size=(self.output_grid_size, self.output_grid_size),
                mode=self.upsample_mode,
                **interpolate_kwargs,
            )

        tokens = self.proj(latent)
        tokens = tokens.flatten(2).transpose(1, 2).contiguous()
        tokens = self.norm(tokens)
        return tokens * self.scale


class ColdWaterBottleImageCVAE(nn.Module):
    """Spatial-latent CVAE from the cold-water-bottle experiment."""

    def __init__(
        self,
        input_channels: int = 3,
        condition_channels: int = 3,
        target_channels: int = 3,
        latent_dim: int = 256,
        hidden_dims: list[int] | None = None,
        image_size: int = 224,
        enable_pi05_token_projector: bool = False,
        pi05_token_dim: int = 2048,
        pi05_token_grid_size: int = 16,
        pi05_token_upsample_mode: str = "bilinear",
        pi05_token_norm: str = "layernorm",
        pi05_token_scale_init: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs

        self.input_channels = input_channels
        self.condition_channels = condition_channels
        self.target_channels = target_channels
        self.latent_dim = latent_dim
        self.image_size = image_size

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        hidden_dims = list(hidden_dims)
        self.hidden_dims = hidden_dims

        downsample_factor = 2 ** len(hidden_dims)
        if image_size % downsample_factor != 0:
            raise ValueError(f"image_size={image_size} must be divisible by {downsample_factor}.")
        self.feature_size = image_size // downsample_factor
        self.bottleneck_channels = hidden_dims[-1]

        posterior_channels = input_channels + condition_channels + target_channels
        context_channels = input_channels + condition_channels

        self.posterior_encoder = self._build_encoder(posterior_channels, hidden_dims)
        self.context_encoder = self._build_encoder(context_channels, hidden_dims)

        self.latent_shape = (latent_dim, self.feature_size, self.feature_size)
        self.to_mu = nn.Conv2d(self.bottleneck_channels, latent_dim, kernel_size=1)
        self.to_log_var = nn.Conv2d(self.bottleneck_channels, latent_dim, kernel_size=1)
        self.decoder_input = nn.Conv2d(latent_dim, self.bottleneck_channels, kernel_size=1)
        self.decoder = self._build_decoder(hidden_dims, target_channels)
        self.pi05_token_projector = (
            PI05TokenProjector(
                in_channels=latent_dim,
                output_dim=pi05_token_dim,
                output_grid_size=pi05_token_grid_size,
                upsample_mode=pi05_token_upsample_mode,
                norm_type=pi05_token_norm,
                scale_init=pi05_token_scale_init,
            )
            if enable_pi05_token_projector
            else None
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def _build_encoder(self, in_channels: int, hidden_dims: list[int]) -> nn.Sequential:
        modules = []
        for h_dim in hidden_dims:
            modules.append(self._conv_block(in_channels, h_dim))
            in_channels = h_dim
        return nn.Sequential(*modules)

    def _build_decoder(self, hidden_dims: list[int], out_channels: int) -> nn.Sequential:
        modules = [
            nn.Sequential(
                nn.Conv2d(hidden_dims[-1] * 2, hidden_dims[-1], kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
            )
        ]

        decoder_dims = list(reversed(hidden_dims))
        for i in range(len(decoder_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        decoder_dims[i],
                        decoder_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(decoder_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    decoder_dims[-1],
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.Tanh(),
            )
        )
        return nn.Sequential(*modules)

    def encode(self, source: Tensor, condition: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        posterior_input = torch.cat([source, condition, target], dim=1)
        result = self.posterior_encoder(posterior_input)
        mu = self.to_mu(result)
        log_var = self.to_log_var(result)
        return mu, log_var

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode_prior_mu(self, source: Tensor, condition: Tensor) -> Tensor:
        context_input = torch.cat([source, condition], dim=1)
        context = self.context_encoder(context_input)
        return self.to_mu(context)

    def project_pi05_tokens(self, latent: Tensor) -> Tensor:
        if self.pi05_token_projector is None:
            raise RuntimeError(
                "PI05 token projector is disabled. Set enable_pi05_token_projector=True."
            )
        return self.pi05_token_projector(latent)

    def encode_pi05_tokens(
        self,
        source: Tensor,
        condition: Tensor,
        target: Tensor | None = None,
        use_mu: bool = True,
    ) -> Tensor:
        if target is None:
            latent = self.encode_prior_mu(source, condition)
        else:
            mu, log_var = self.encode(source, condition, target)
            latent = mu if use_mu else self.reparameterize(mu, log_var)
        return self.project_pi05_tokens(latent)

    def decode(self, z: Tensor, source: Tensor, condition: Tensor) -> Tensor:
        context_input = torch.cat([source, condition], dim=1)
        context = self.context_encoder(context_input)

        z_features = self.decoder_input(z)
        decoder_input = torch.cat([z_features, context], dim=1)
        return self.decoder(decoder_input)

    def forward(
        self, source: Tensor, condition: Tensor, target: Tensor | None = None, **kwargs
    ) -> list[Tensor | None]:
        if kwargs.get("return_pi05_tokens", False):
            return [self.encode_pi05_tokens(source, condition, target), target, None, None]

        if target is None:
            z = self.encode_prior_mu(source, condition)
            return [self.decode(z, source, condition), None, None, None]

        mu, log_var = self.encode(source, condition, target)
        z = self.reparameterize(mu, log_var) if self.training else mu
        return [self.decode(z, source, condition), target, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict[str, Tensor]:
        recons = args[0]
        target = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs["M_N"]
        recons_loss = F.mse_loss(recons, target)
        kld_per_location = -0.5 * torch.sum(
            1 + log_var - mu**2 - log_var.exp(),
            dim=1,
        )
        kld_loss = torch.mean(kld_per_location)

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss,
            "KLD": -kld_loss,
        }

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        del batch_size, current_device
        source = kwargs["source"]
        condition = kwargs["condition"]
        target = kwargs.get("target")
        if target is None:
            z = self.encode_prior_mu(source, condition)
        else:
            z = self.encode(source, condition, target)[0]
        return self.decode(z, source, condition)

    def generate(
        self, source: Tensor, condition: Tensor, target: Tensor | None = None, **kwargs
    ) -> Tensor:
        return self.forward(source, condition, target, **kwargs)[0]


class ThermalCVAEEncoder(nn.Module):
    """Adapter that exposes the copied CVAE as a PI0.5 thermal image encoder."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        condition_channels: int = 3,
        target_channels: int = 3,
        latent_dim: int = 256,
        hidden_dims: list[int] | None = None,
        image_size: int = 224,
        output_dim: int = 2048,
        token_grid_size: int = 16,
        token_upsample_mode: str = "bilinear",
        token_norm: str = "layernorm",
        token_scale_init: float = 1.0,
        checkpoint_path: str | None = None,
        checkpoint_strict: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.cvae = ColdWaterBottleImageCVAE(
            input_channels=input_channels,
            condition_channels=condition_channels,
            target_channels=target_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            image_size=image_size,
            enable_pi05_token_projector=True,
            pi05_token_dim=output_dim,
            pi05_token_grid_size=token_grid_size,
            pi05_token_upsample_mode=token_upsample_mode,
            pi05_token_norm=token_norm,
            pi05_token_scale_init=token_scale_init,
        )
        if checkpoint_path:
            self.load_cvae_checkpoint(checkpoint_path, strict=checkpoint_strict)

    def _prepare_image(self, images: Tensor, feature_name: str) -> Tensor:
        images = image_to_bchw(images, feature_name)
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return images

    @staticmethod
    def _extract_cvae_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        cvae_state_dict = {}
        for key, value in state_dict.items():
            local_key = key
            for prefix in (
                "model.thermal_encoder.cvae.",
                "model.thermal_encoder.",
                "thermal_encoder.cvae.",
                "thermal_encoder.",
                "model.",
                "cvae.",
            ):
                if local_key.startswith(prefix):
                    local_key = local_key[len(prefix) :]
                    break
            cvae_state_dict[local_key] = value
        return cvae_state_dict

    def load_cvae_checkpoint(self, checkpoint_path: str | Path, strict: bool = True) -> None:
        checkpoint_path = Path(checkpoint_path).expanduser()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = self._extract_cvae_state_dict(state_dict)
        incompatible = self.cvae.load_state_dict(state_dict, strict=strict)
        logging.info(
            "Loaded PI05 thermal CVAE checkpoint from %s (missing=%d, unexpected=%d).",
            checkpoint_path,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )

    def forward(
        self,
        source: Tensor,
        condition: Tensor,
        target: Tensor | None = None,
        *,
        source_feature: str = "observation.images.cam_left_high",
        condition_feature: str = "observation.images.cam_thermal",
        target_feature: str = "observation.images.cam_rgb_thermal_mixing",
    ) -> Tensor:
        source = self._prepare_image(source, source_feature)
        condition = self._prepare_image(condition, condition_feature)
        if target is not None:
            target = self._prepare_image(target, target_feature)
        return self.cvae.encode_pi05_tokens(source, condition, target=target, use_mu=True)
