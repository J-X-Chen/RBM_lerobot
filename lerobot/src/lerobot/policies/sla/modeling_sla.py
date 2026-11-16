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

"""SLA policy: LeRobot's PyTorch pi0.5 plus SLA's spatial representation alignment.

Reference: ``SLA/src/openpi/models/pi0_align.py`` and ``SLA/scripts/train_align.py`` in
https://github.com/OpenHelix-Team/Spatial-Forcing. The VLA itself is pi0.5, so everything except
the alignment branch (projector + frozen OmniVGGT teacher) is inherited from ``PI05Policy``.
"""

import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.local_assets import resolve_local_model_path

from ..common.flow_matching import FlowConvention, make_flow_matching_inputs
from ..common.vla_utils import make_att_2d_masks
from ..pi05.modeling_pi05 import (
    PI05Policy,
    PI05Pytorch,
    _reduce_training_rtc_loss,
    _sample_training_rtc_prefix_mask,
)
from .alignment import (
    AlignProjector,
    make_align_mask,
    pool_vggt_features,
    preprocess_images_for_vggt,
    resize_with_pad_valid_mask,
)
from .configuration_sla import SLAConfig
from .processor_sla import OBS_LANGUAGE_ATTENTION_MASK_WITHOUT_STATE, OBS_LANGUAGE_TOKENS_WITHOUT_STATE

OMNIVGGT_FILENAME = "OmniVGGT.safetensors"
OMNIVGGT_REPO_SUBPATH = f"omnivggt/{OMNIVGGT_FILENAME}"


def resolve_omnivggt_checkpoint(weight_path: str) -> Path:
    """Locate ``OmniVGGT.safetensors`` from a file, a directory, project-local models or the Hub."""
    path = Path(weight_path).expanduser()
    candidates = [path, path / OMNIVGGT_FILENAME, path / OMNIVGGT_REPO_SUBPATH]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    local_dir = resolve_local_model_path(weight_path, required_files=(OMNIVGGT_REPO_SUBPATH,))
    if local_dir is not None:
        return local_dir / OMNIVGGT_REPO_SUBPATH
    from huggingface_hub import hf_hub_download

    logging.info("Downloading %s from the Hub repo %s", OMNIVGGT_REPO_SUBPATH, weight_path)
    return Path(hf_hub_download(repo_id=weight_path, filename=OMNIVGGT_REPO_SUBPATH))


class SLAPytorch(PI05Pytorch):
    """pi0.5 core model with the SLA alignment projector and a lazily loaded OmniVGGT teacher."""

    def __init__(self, config: SLAConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        vlm_config = self.paligemma_with_expert.paligemma.config
        depth = vlm_config.text_config.num_hidden_layers
        self.vision_patch_size = vlm_config.vision_config.patch_size
        layer = config.vla_layers_align
        self.vla_layer_index = layer if layer >= 0 else depth + layer
        if not 0 <= self.vla_layer_index < depth:
            raise ValueError(f"vla_layers_align={layer} is outside the {depth}-layer PaliGemma backbone")
        self.align_projector = AlignProjector(
            vlm_config.text_config.hidden_size,
            config.vggt_dim,
            use_vlm_norm=config.use_vlm_norm,
            zscore_targets=config.zscore_align_targets,
        )
        # The frozen teacher is deliberately *not* a registered submodule: it must stay out of
        # state_dict/checkpoints, the optimizer, DDP/FSDP wrapping and `.to(dtype)` casts.
        self.__dict__["_vggt_teacher"] = None

    # ----------------------------------------------------------------------------------------
    # Alignment targets (frozen OmniVGGT)
    # ----------------------------------------------------------------------------------------
    def _get_vggt_teacher(self, default_device: torch.device):
        teacher = self.__dict__["_vggt_teacher"]
        if teacher is None:
            from .omnivggt import OmniVGGTFeatureExtractor

            checkpoint = resolve_omnivggt_checkpoint(self.config.vggt_weight_path)
            logging.info("Loading SLA OmniVGGT alignment teacher from %s", checkpoint)
            device = torch.device(self.config.align_vggt_device or default_device)
            if device.type == "cuda" and device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            teacher = OmniVGGTFeatureExtractor.from_checkpoint(checkpoint).to(device).eval()
            teacher.requires_grad_(False)
            self.__dict__["_vggt_teacher"] = teacher
        return teacher

    @torch.no_grad()
    def compute_align_targets(
        self, images: list[Tensor], img_masks: list[Tensor], padding_masks: list[Tensor] | None
    ) -> tuple[Tensor, Tensor]:
        """OmniVGGT targets ``[B, N*T, 2*vggt_dim]`` and token mask ``[B, N*T]`` for PaliGemma inputs.

        Args:
            images: per-camera ``[B, 3, H, W]`` PaliGemma inputs in ``[-1, 1]`` (after resize-with-pad).
            img_masks: per-camera ``[B]`` presence masks.
            padding_masks: per-camera ``[B, H, W]`` valid-pixel masks of the resize-with-pad, or None.
        """
        out_device = images[0].device
        teacher = self._get_vggt_teacher(out_device)
        device = next(teacher.parameters()).device
        image_size = images[0].shape[-1]
        tokens_per_image = (image_size // self.vision_patch_size) ** 2

        # Same input convention as SLA's TorchAlignFeatureExtractor: [0, 1] range, padded border
        # and missing cameras painted white.
        vggt_inputs = []
        for idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=True)):
            img = torch.clamp(img.to(device=device, dtype=torch.float32) / 2.0 + 0.5, 0, 1)
            if padding_masks is not None:
                img = torch.where(padding_masks[idx].to(device)[:, None], img, torch.ones_like(img))
            img = torch.where(img_mask.to(device)[:, None, None, None], img, torch.ones_like(img))
            vggt_inputs.append(img)

        bsize = vggt_inputs[0].shape[0]
        chunk = self.config.align_feature_batch_size or bsize
        targets = []
        for start in range(0, bsize, chunk):
            chunk_images = preprocess_images_for_vggt([img[start : start + chunk] for img in vggt_inputs])
            autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
            with autocast:
                features, patch_start_idx = teacher(chunk_images, feature_layer=self.config.vggt_layers_align)
            hidden = features[0][:, :, patch_start_idx:, :]
            height, width = chunk_images.shape[-2:]
            pooled = pool_vggt_features(
                hidden,
                (height // teacher.patch_size, width // teacher.patch_size),
                (height, width),
                tokens_per_image,
                pooling_func=self.config.pooling_func,
                use_vggt_pe=self.config.use_vggt_pe,
            )
            targets.append(pooled.to(torch.bfloat16))
        target = torch.cat(targets, dim=0).to(out_device, non_blocking=True)
        align_mask = make_align_mask(
            [mask.to(out_device) for mask in img_masks],
            [mask.to(out_device) for mask in padding_masks] if padding_masks is not None else None,
            tokens_per_image,
            image_size // self.vision_patch_size,
        )
        return target, align_mask

    # ----------------------------------------------------------------------------------------
    # Training forward
    # ----------------------------------------------------------------------------------------
    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise,
        time,
        prefix_mask: Tensor | None = None,
        states=None,
        state_masks=None,
        align_targets: Tensor | None = None,
        align_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """pi0.5 training forward that also returns the per-sample alignment loss ``[B]`` (or None)."""
        x_t, u_t, model_time = make_flow_matching_inputs(
            actions, noise, time, convention=FlowConvention.NOISE_AT_ONE, prefix_mask=prefix_mask
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks, states, state_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, model_time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        capture_layer = None
        image_token_count = 0
        if align_targets is not None:
            # Image tokens lead the prefix (see `embed_prefix`); SLA aligns exactly those.
            layout = self._last_prefix_token_layout
            image_token_count = sum(item["token_count"] for item in layout if item["kind"] == "image")
            if layout[0]["kind"] != "image" or image_token_count != align_targets.shape[1]:
                raise ValueError(
                    f"SLA alignment expects {align_targets.shape[1]} leading image tokens, "
                    f"got prefix layout {[(item['kind'], item['token_count']) for item in layout]}"
                )
            capture_layer = self.vla_layer_index

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            capture = (
                {"layer": capture_layer, "tokens": image_token_count} if capture_layer is not None else None
            )
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                prefix_hidden_capture=capture,
            )
            return suffix_out, (capture["hidden"] if capture is not None else None)

        suffix_out, vision_hidden = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
        v_t = self._apply_checkpoint(self.action_out_proj, suffix_out)
        action_losses = torch.nn.functional.mse_loss(u_t, v_t, reduction="none")

        align_loss = None
        if align_targets is not None:
            align_loss = self.align_projector(vision_hidden, align_targets, align_mask)
        return action_losses, align_loss


class SLAPolicy(PI05Policy):
    """SLA policy (pi0.5 + spatial alignment). Inference is identical to ``PI05Policy``."""

    config_class = SLAConfig
    name = "sla"
    model_class = SLAPytorch

    def _prepare_pretrained_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        state_dict = super()._prepare_pretrained_state_dict(state_dict)
        # Keep the fresh initialization of tensors a source checkpoint may legitimately lack:
        # - the align projector, absent from pi0.5 checkpoints;
        # - the action expert's lm_head, never used by pi0.5 and absent from JAX conversions.
        optional_prefixes = (
            "model.align_projector.",
            "model.paligemma_with_expert.gemma_expert.lm_head.",
        )
        current = self.state_dict()
        missing = [key for key in current if key.startswith(optional_prefixes) and key not in state_dict]
        if missing:
            logging.info("Keeping freshly initialized tensors absent from the checkpoint: %s", missing)
            for key in missing:
                state_dict[key] = current[key]
        return state_dict

    def _get_default_peft_targets(self) -> dict[str, Any]:
        targets = super()._get_default_peft_targets()
        targets["modules_to_save"] = [*targets.get("modules_to_save", []), "model.align_projector"]
        return targets

    def _image_padding_masks(self, batch: dict[str, Tensor], images: list[Tensor]) -> list[Tensor] | None:
        """Valid-pixel masks of the resize-with-pad applied in ``_preprocess_images``."""
        if not self.config.ignore_img_padding_area:
            return None
        target_h, target_w = self.config.image_resolution
        masks = []
        for key, img in zip(self.config.image_features, images, strict=True):
            raw = batch.get(key)
            if raw is None:
                height, width = target_h, target_w
            else:
                channels_first = raw.shape[1] == 3
                height, width = raw.shape[-2:] if channels_first else raw.shape[-3:-1]
            valid = resize_with_pad_valid_mask(int(height), int(width), target_h, target_w, img.device)
            masks.append(valid[None].expand(img.shape[0], -1, -1))
        return masks

    def _select_training_prompt(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        prob = self.config.state_dropout_prob
        if not (self.training and prob > 0 and OBS_LANGUAGE_TOKENS_WITHOUT_STATE in batch):
            return tokens, masks
        drop = torch.rand(tokens.shape[0], device=tokens.device) < prob
        tokens = torch.where(
            drop[:, None], batch[OBS_LANGUAGE_TOKENS_WITHOUT_STATE].to(tokens.device), tokens
        )
        masks = torch.where(
            drop[:, None], batch[OBS_LANGUAGE_ATTENTION_MASK_WITHOUT_STATE].to(masks.device), masks
        )
        return tokens, masks

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Training loss: ``action_loss + align_loss_coeff * align_loss`` (alignment only in train mode)."""
        images, img_masks = self._preprocess_images(batch)
        states, state_masks = self._prepare_memory_states(batch)
        tokens, masks = self._select_training_prompt(batch)
        actions = self.prepare_action(batch)

        use_align = self.training and self.config.align_loss_coeff > 0
        align_targets = align_mask = None
        if use_align:
            align_targets, align_mask = self.model.compute_align_targets(
                images, img_masks, self._image_padding_masks(batch, images)
            )

        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)
        prefix_mask = _sample_training_rtc_prefix_mask(
            actions.shape[0], actions.shape[1], self.config.rtc_training_max_delay, actions.device
        )
        losses, align_loss = self.model(
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise,
            time,
            prefix_mask=prefix_mask,
            states=states,
            state_masks=state_masks,
            align_targets=align_targets,
            align_mask=align_mask,
        )

        if self.config.output_features is None:
            raise ValueError("output_features must be set (validate_features) before computing the loss")
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        if prefix_mask is None:
            loss_per_dim = losses.mean(dim=(0, 1))
        else:
            postfix_mask = (~prefix_mask).unsqueeze(-1).expand_as(losses)
            loss_per_dim = (losses * postfix_mask).sum(dim=(0, 1)) / postfix_mask.sum(dim=(0, 1)).clamp(min=1)
        loss_dict: dict[str, Any] = {"loss_per_dim": loss_per_dim.detach().cpu().numpy().tolist()}

        coeff = self.config.align_loss_coeff
        if reduction == "none":
            per_sample = _reduce_training_rtc_loss(losses, prefix_mask, reduction="none")
            loss_dict["action_loss"] = per_sample.mean().item()
            if align_loss is not None:
                loss_dict["align_loss"] = align_loss.mean().item()
                per_sample = per_sample + coeff * align_loss
            loss_dict["loss"] = per_sample.mean().item()
            return per_sample, loss_dict

        action_loss = _reduce_training_rtc_loss(losses, prefix_mask, reduction="mean")
        loss = action_loss
        loss_dict["action_loss"] = action_loss.item()
        if align_loss is not None:
            align_loss = align_loss.mean()
            loss = action_loss + coeff * align_loss
            loss_dict["align_loss"] = align_loss.item()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict


__all__ = ["SLAPolicy", "SLAPytorch", "resolve_omnivggt_checkpoint"]
