"""Optional RGB DINOv2 adapter retained from the S1/G1 training stack."""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

DEFAULT_RGB_DINOV2_CHECKPOINT_PATH = "pretrained_checkpoints/backbone/Dinov2/dinov2_vitb14_pretrain.pth"
DEFAULT_DINOV2_REPO_PATH = "dinov2"
DINOV2_IMAGENET_MEAN = (0.485, 0.456, 0.406)
DINOV2_IMAGENET_STD = (0.229, 0.224, 0.225)


def _project_root() -> Path:
    """Return the repository root that holds project-local checkpoints."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pretrained_checkpoints").is_dir() or (parent / "dinov2" / "hubconf.py").is_file():
            return parent
    return Path.cwd().resolve()


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(str(path)).expanduser()
    if path.is_absolute():
        return path
    return _project_root() / path


def image_to_bchw(images: Tensor, feature_name: str) -> Tensor:
    """Convert a four-dimensional image batch to ``[B, C, H, W]``."""
    if images.ndim != 4:
        raise ValueError(
            f"Image feature {feature_name} must have shape [B,C,H,W] or [B,H,W,C], got {tuple(images.shape)}."
        )
    if images.shape[1] in {1, 3, 4}:
        return images
    if images.shape[-1] in {1, 3, 4}:
        return images.permute(0, 3, 1, 2)
    raise ValueError(
        f"Cannot determine the channel dimension for image feature {feature_name}: {tuple(images.shape)}."
    )


class _DINOv2TokenEncoder(nn.Module):
    @staticmethod
    def _candidate_dinov2_repo_paths(repo_path: str | None = None) -> list[Path]:
        paths: list[Path] = []
        if repo_path is not None:
            paths.append(_resolve_project_path(repo_path))

        paths.extend(
            [
                _resolve_project_path(DEFAULT_DINOV2_REPO_PATH),
                _resolve_project_path("pretrained_checkpoints/Dinov2/dinov2"),
                _resolve_project_path("pretrained_checkpoints/dinov2"),
            ]
        )

        unique_paths: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            normalized = str(path)
            if normalized not in seen:
                unique_paths.append(path)
                seen.add(normalized)
        return unique_paths

    @classmethod
    def _load_dinov2_backbone(cls, model_type: str, repo_path: str | None = None) -> nn.Module:
        searched_paths = cls._candidate_dinov2_repo_paths(repo_path)
        for path in searched_paths:
            if (path / "hubconf.py").is_file():
                try:
                    return torch.hub.load(
                        str(path),
                        model_type,
                        source="local",
                        pretrained=False,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Found a local DINOv2 repo but failed to load the backbone from "
                        f"{path}. Check that this is facebookresearch/dinov2 and that "
                        f"rgb_dinov2_model_type={model_type!r} is valid."
                    ) from exc

        searched = "\n".join(f"  - {path}" for path in searched_paths)
        expected_path = _resolve_project_path(DEFAULT_DINOV2_REPO_PATH)
        raise FileNotFoundError(
            "DINOv2 repo was not found in the project. Put facebookresearch/dinov2 at:\n"
            f"  {expected_path}\n"
            "or pass --policy.rgb_dinov2_repo_path=/path/to/dinov2.\n"
            "Searched paths:\n"
            f"{searched}"
        )

    def _resolve_backbone_embed_dim(self) -> int:
        embed_dim = getattr(self.backbone, "embed_dim", None)
        if embed_dim is not None:
            return int(embed_dim)
        norm = getattr(self.backbone, "norm", None)
        normalized_shape = getattr(norm, "normalized_shape", None)
        if normalized_shape:
            return int(normalized_shape[0])
        raise ValueError(f"Could not infer DINOv2 embed_dim for RGB model {self.model_type!r}.")

    def _tokens_from_forward_features(self, images: Tensor) -> Tensor | None:
        forward_features = getattr(self.backbone, "forward_features", None)
        if not callable(forward_features):
            return None
        features = forward_features(images)
        if not isinstance(features, dict):
            return None

        tokens = []
        cls_token = features.get("x_norm_clstoken")
        if self.include_cls_token and cls_token is not None:
            tokens.append(cls_token[:, None])
        reg_tokens = features.get("x_norm_regtokens")
        if self.include_register_tokens and reg_tokens is not None and reg_tokens.numel() > 0:
            tokens.append(reg_tokens)
        patch_tokens = features.get("x_norm_patchtokens")
        if patch_tokens is not None:
            tokens.append(patch_tokens)
        if not tokens:
            return None
        return torch.cat(tokens, dim=1)

    def _tokens_from_final_block_hook(self, images: Tensor) -> Tensor:
        blocks = getattr(self.backbone, "blocks", None)
        if not blocks:
            raise RuntimeError("DINOv2 backbone does not expose forward_features or transformer blocks.")

        captured: dict[str, Tensor] = {}

        def hook_fn(_module, _inputs, output):
            captured["tokens"] = output[0] if isinstance(output, tuple) else output

        handle = blocks[-1].register_forward_hook(hook_fn)
        try:
            _ = self.backbone(images)
        finally:
            handle.remove()

        raw_tokens = captured.get("tokens")
        if raw_tokens is None:
            raise RuntimeError("Failed to capture RGB DINOv2 final block tokens.")
        norm = getattr(self.backbone, "norm", None)
        if callable(norm):
            raw_tokens = norm(raw_tokens)

        tokens = []
        if self.include_cls_token:
            tokens.append(raw_tokens[:, :1])
        patch_start = 1
        register_end = patch_start + self.num_register_tokens
        if self.include_register_tokens and self.num_register_tokens > 0:
            tokens.append(raw_tokens[:, patch_start:register_end])
        tokens.append(raw_tokens[:, register_end:])
        return torch.cat(tokens, dim=1)

    def _extract_tokens(self, images: Tensor) -> Tensor:
        tokens = self._tokens_from_forward_features(images)
        if tokens is not None:
            return tokens
        return self._tokens_from_final_block_hook(images)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        images = self._preprocess_for_dinov2(images)
        if self.freeze_backbone:
            with torch.no_grad():
                tokens = self._extract_tokens(images)
        else:
            tokens = self._extract_tokens(images)
        tokens = self.feature_norm(tokens)
        tokens = self.output_proj(tokens)
        tokens = self.output_norm(tokens)
        return self.dropout(tokens)


class DINOv2RGBEncoder(_DINOv2TokenEncoder):
    """RGB DINOv2 backbone adapter that emits PI05-compatible image tokens."""

    def __init__(
        self,
        *,
        output_dim: int,
        model_type: str = "dinov2_vitb14",
        checkpoint_path: str | None = DEFAULT_RGB_DINOV2_CHECKPOINT_PATH,
        dinov2_repo_path: str | None = None,
        freeze_backbone: bool = True,
        include_cls_token: bool = False,
        include_register_tokens: bool = False,
        input_range: str = "minus_one_to_one",
        dropout: float = 0.0,
    ):
        nn.Module.__init__(self)
        if not str(model_type or "").strip():
            raise ValueError("model_type must be non-empty")

        self.model_type = model_type
        self.include_cls_token = include_cls_token
        self.include_register_tokens = include_register_tokens
        self.freeze_backbone = freeze_backbone
        if input_range not in {"minus_one_to_one", "zero_to_one", "auto"}:
            raise ValueError("input_range must be one of: minus_one_to_one, zero_to_one, auto")
        self.input_range = input_range

        self.backbone = self._load_dinov2_backbone(model_type, dinov2_repo_path)
        self.patch_size = int(getattr(self.backbone, "patch_size", 14))
        if self.patch_size != 14:
            logging.warning(
                "RGB DINOv2 default checkpoint expects patch_size=14, but %s reports patch_size=%s.",
                model_type,
                self.patch_size,
            )
        self.num_register_tokens = int(getattr(self.backbone, "num_register_tokens", 0))
        self.embed_dim = self._resolve_backbone_embed_dim()
        self._load_rgb_dinov2_backbone_state(checkpoint_path)

        self.feature_norm = nn.LayerNorm(self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

    @classmethod
    def _resolve_rgb_checkpoint_path(cls, checkpoint_path: str | None) -> Path:
        if checkpoint_path is None:
            checkpoint_path = DEFAULT_RGB_DINOV2_CHECKPOINT_PATH
        return _resolve_project_path(checkpoint_path)

    @classmethod
    def _load_rgb_checkpoint_state_dict(cls, checkpoint_path: str | None) -> dict:
        path = cls._resolve_rgb_checkpoint_path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                "RGB DINOv2 checkpoint not found: "
                f"{path}. Put dinov2_vitb14_pretrain.pth under "
                "pretrained_checkpoints/backbone/Dinov2 or pass "
                "--policy.rgb_dinov2_checkpoint_path=/path/to/dinov2_vitb14_pretrain.pth."
            )
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError(f"RGB DINOv2 checkpoint must be a state dict, got {type(state).__name__}.")
        for key in ("model", "state_dict", "teacher", "student"):
            nested = state.get(key)
            if isinstance(nested, dict):
                state = nested
                break
        return {key.removeprefix("module."): value for key, value in state.items()}

    def _load_rgb_dinov2_backbone_state(self, checkpoint_path: str | None) -> None:
        state = self._load_rgb_checkpoint_state_dict(checkpoint_path)
        try:
            self.backbone.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load RGB DINOv2 backbone weights. Check that "
                f"rgb_dinov2_model_type={self.model_type!r} matches the checkpoint."
            ) from exc
        logging.info("Loaded RGB DINOv2 backbone checkpoint for %s.", self.model_type)

    def _preprocess_for_dinov2(self, images: Tensor) -> Tensor:
        images = image_to_bchw(images, "rgb_dinov2_encoder_input").to(dtype=torch.float32)
        if images.shape[1] == 4:
            images = images[:, :3]
        elif images.shape[1] != 3:
            raise ValueError(f"RGB DINOv2 requires three- or four-channel images, got {images.shape[1]}.")

        if (
            self.input_range == "minus_one_to_one"
            or self.input_range == "auto"
            and images.detach().amin() < -0.05
        ):
            images = (images + 1.0) * 0.5
        images = images.clamp(0.0, 1.0)

        height, width = images.shape[-2:]
        new_height = max(self.patch_size, (height // self.patch_size) * self.patch_size)
        new_width = max(self.patch_size, (width // self.patch_size) * self.patch_size)
        if (new_height, new_width) != (height, width):
            images = F.interpolate(
                images,
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        mean = torch.tensor(DINOV2_IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(DINOV2_IMAGENET_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        return (images - mean) / std
