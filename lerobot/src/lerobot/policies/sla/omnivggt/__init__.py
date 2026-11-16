# Copyright (c) Meta Platforms, Inc. and affiliates.
# Vendored subset of OmniVGGT (https://github.com/Livioni/OmniVGGT-official) as used by SLA
# (https://github.com/OpenHelix-Team/Spatial-Forcing) to produce frozen 3D alignment targets.
# Only the aggregator is kept: SLA aligns against aggregator features and never runs the heads.

from pathlib import Path

import torch
from torch import nn

from .omnivggt_aggregator import ZeroAggregator

__all__ = ["OmniVGGTFeatureExtractor"]


class OmniVGGTFeatureExtractor(nn.Module):
    """OmniVGGT aggregator without prediction heads."""

    def __init__(self, img_size: int = 518, patch_size: int = 14, embed_dim: int = 1024):
        super().__init__()
        self.aggregator = ZeroAggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, pose_hidden_dim=9
        )
        self.patch_size = patch_size
        self.embed_dim = embed_dim

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> "OmniVGGTFeatureExtractor":
        """Build the extractor and load ``OmniVGGT.safetensors`` (heads are ignored)."""
        from safetensors.torch import load_file

        model = cls()
        state_dict = load_file(str(checkpoint_path))
        aggregator_state = {
            key.removeprefix("aggregator."): value
            for key, value in state_dict.items()
            if key.startswith("aggregator.")
        }
        missing, unexpected = model.aggregator.load_state_dict(aggregator_state, strict=False)
        # `mask_token` belongs to DINOv2's masked-image pretraining and is never used here.
        missing = [key for key in missing if not key.endswith("mask_token")]
        if missing or unexpected:
            raise RuntimeError(
                f"Incompatible OmniVGGT checkpoint {checkpoint_path}: "
                f"missing={missing[:10]} unexpected={unexpected[:10]}"
            )
        return model

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        *,
        feature_layer: int | None = -1,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        camera_gt_index: list[int] | None = None,
    ) -> tuple[list[torch.Tensor], int]:
        """Return aggregator features ``[B, S, P, 2*embed_dim]`` and the first patch-token index."""
        return self.aggregator(
            images=images,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            camera_gt_index=camera_gt_index or [],
            feature_layer=feature_layer,
        )
