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

"""Convert an SLA (openpi / JAX orbax) checkpoint into a LeRobot ``sla`` policy checkpoint.

SLA's model is pi0.5, so LeRobot reuses its PyTorch pi0.5 implementation; only the *weights* of
the official releases (``haofuly/SLA_base`` and the ``SLA_benchmark_ckpts``) exist solely as JAX
orbax checkpoints. This script renames them to LeRobot's pi0.5 layout (same mapping as openpi's
``examples/convert_jax_model_to_pytorch.py``) and adds the SLA ``align_projector``.

Requires ``pip install "jax[cpu]" orbax-checkpoint`` (CPU JAX is enough).

Example::

    python -m lerobot.policies.sla.convert_sla_jax_checkpoint \\
        --checkpoint-dir base_models/SLA_base_jax \\
        --output-dir base_models/sla_base
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch

from .configuration_sla import SLAConfig

VISION_LAYERS = 27
VISION_WIDTH = 1152
LLM_PREFIX = "paligemma_with_expert.paligemma.model.language_model"
VISION_PREFIX = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
EXPERT_PREFIX = "paligemma_with_expert.gemma_expert.model"


def restore_jax_params(params_dir: Path) -> dict[str, np.ndarray]:
    """Restore an openpi ``params`` orbax checkpoint as a flat ``{"a/b/c": ndarray}`` dict (float32)."""
    import jax
    import orbax.checkpoint as ocp

    params_dir = params_dir.resolve()
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(params_dir)
        tree = metadata.item_metadata if hasattr(metadata, "item_metadata") else metadata
        item = {"params": tree["params"]}
        restore_args = jax.tree.map(
            lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray, dtype=np.float32), item
        )
        params = checkpointer.restore(
            params_dir, ocp.args.PyTreeRestore(item=item, restore_args=restore_args)
        )["params"]

    flat: dict[str, np.ndarray] = {}

    def _flatten(node, prefix: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _flatten(value, (*prefix, str(key)))
        else:
            flat["/".join(prefix)] = np.asarray(node)

    _flatten(params, ())
    # Checkpoints saved during openpi training wrap every leaf in an nnx "value" node.
    if flat and all(key.endswith("/value") for key in flat):
        flat = {key.removesuffix("/value"): value for key, value in flat.items()}
    return flat


def _t(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def convert_params(params: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Map flat openpi pi0.5 (+ SLA projector) params onto LeRobot ``PI05Pytorch`` / ``SLAPytorch`` keys."""
    p = dict(params)
    out: dict[str, torch.Tensor] = {}

    def pop(key: str) -> np.ndarray:
        if key not in p:
            raise KeyError(f"Missing parameter {key!r} in the JAX checkpoint")
        return p.pop(key)

    # --- SigLIP vision tower -----------------------------------------------------------------
    out[f"{VISION_PREFIX}.embeddings.patch_embedding.weight"] = _t(
        pop("PaliGemma/img/embedding/kernel").transpose(3, 2, 0, 1)
    )
    out[f"{VISION_PREFIX}.embeddings.patch_embedding.bias"] = _t(pop("PaliGemma/img/embedding/bias"))
    out[f"{VISION_PREFIX}.embeddings.position_embedding.weight"] = _t(
        pop("PaliGemma/img/pos_embedding").reshape(-1, VISION_WIDTH)
    )
    enc = "PaliGemma/img/Transformer/encoderblock"
    ln0_s, ln0_b = pop(f"{enc}/LayerNorm_0/scale"), pop(f"{enc}/LayerNorm_0/bias")
    ln1_s, ln1_b = pop(f"{enc}/LayerNorm_1/scale"), pop(f"{enc}/LayerNorm_1/bias")
    fc1_k, fc1_b = pop(f"{enc}/MlpBlock_0/Dense_0/kernel"), pop(f"{enc}/MlpBlock_0/Dense_0/bias")
    fc2_k, fc2_b = pop(f"{enc}/MlpBlock_0/Dense_1/kernel"), pop(f"{enc}/MlpBlock_0/Dense_1/bias")
    attn = {
        name: (
            pop(f"{enc}/MultiHeadDotProductAttention_0/{name}/kernel"),
            pop(f"{enc}/MultiHeadDotProductAttention_0/{name}/bias"),
        )
        for name in ("query", "key", "value", "out")
    }
    torch_attn_names = {"query": "q_proj", "key": "k_proj", "value": "v_proj", "out": "out_proj"}
    for i in range(VISION_LAYERS):
        layer = f"{VISION_PREFIX}.encoder.layers.{i}"
        out[f"{layer}.layer_norm1.weight"] = _t(ln0_s[i])
        out[f"{layer}.layer_norm1.bias"] = _t(ln0_b[i])
        out[f"{layer}.layer_norm2.weight"] = _t(ln1_s[i])
        out[f"{layer}.layer_norm2.bias"] = _t(ln1_b[i])
        out[f"{layer}.mlp.fc1.weight"] = _t(fc1_k[i].transpose())
        out[f"{layer}.mlp.fc1.bias"] = _t(fc1_b[i])
        out[f"{layer}.mlp.fc2.weight"] = _t(fc2_k[i].transpose())
        out[f"{layer}.mlp.fc2.bias"] = _t(fc2_b[i])
        for name, (kernel, bias) in attn.items():
            torch_name = torch_attn_names[name]
            out[f"{layer}.self_attn.{torch_name}.weight"] = _t(
                kernel[i].reshape(-1, VISION_WIDTH).transpose()
            )
            out[f"{layer}.self_attn.{torch_name}.bias"] = _t(bias[i].reshape(-1))
    out[f"{VISION_PREFIX}.post_layernorm.weight"] = _t(pop("PaliGemma/img/Transformer/encoder_norm/scale"))
    out[f"{VISION_PREFIX}.post_layernorm.bias"] = _t(pop("PaliGemma/img/Transformer/encoder_norm/bias"))
    out["paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"] = _t(
        pop("PaliGemma/img/head/kernel").transpose()
    )
    out["paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"] = _t(
        pop("PaliGemma/img/head/bias")
    )

    # --- Gemma 2B (PaliGemma LLM, expert 0) ------------------------------------------------------
    embedding = _t(pop("PaliGemma/llm/embedder/input_embedding"))
    out[f"{LLM_PREFIX}.embed_tokens.weight"] = embedding
    out["paligemma_with_expert.paligemma.lm_head.weight"] = embedding.clone()
    _convert_gemma_layers(p, pop, out, suffix="", prefix=LLM_PREFIX, adarms=False)
    out[f"{LLM_PREFIX}.norm.weight"] = _t(pop("PaliGemma/llm/final_norm/scale"))

    # --- Gemma 300M action expert (expert 1, adaRMS conditioned in pi0.5) ------------------------
    _convert_gemma_layers(p, pop, out, suffix="_1", prefix=EXPERT_PREFIX, adarms=True)
    out[f"{EXPERT_PREFIX}.norm.dense.weight"] = _t(
        pop("PaliGemma/llm/final_norm_1/Dense_0/kernel").transpose()
    )
    out[f"{EXPERT_PREFIX}.norm.dense.bias"] = _t(pop("PaliGemma/llm/final_norm_1/Dense_0/bias"))

    # --- pi0.5 projections and the SLA align projector -------------------------------------------
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        out[f"{name}.weight"] = _t(pop(f"{name}/kernel").transpose())
        out[f"{name}.bias"] = _t(pop(f"{name}/bias"))
    if "align_projector/fc1/kernel" in p:
        for name in ("fc1", "fc2"):
            out[f"align_projector.{name}.weight"] = _t(pop(f"align_projector/{name}/kernel").transpose())
            out[f"align_projector.{name}.bias"] = _t(pop(f"align_projector/{name}/bias"))
        if "align_projector/vlm_norm/scale" in p:
            out["align_projector.vlm_norm.weight"] = _t(pop("align_projector/vlm_norm/scale"))
            out["align_projector.vlm_norm.bias"] = _t(pop("align_projector/vlm_norm/bias"))
    else:
        logging.warning("No align_projector in the checkpoint; it will be initialized from scratch.")

    if p:
        logging.warning("Unconverted JAX parameters (ignored): %s", sorted(p))
    return out


def _convert_gemma_layers(p, pop, out, *, suffix: str, prefix: str, adarms: bool) -> None:
    llm = "PaliGemma/llm/layers"
    q_einsum = pop(f"{llm}/attn/q_einsum{suffix}/w")  # [L, heads, width, head_dim]
    kv_einsum = pop(f"{llm}/attn/kv_einsum{suffix}/w")  # [L, 2, kv_heads, width, head_dim]
    o_einsum = pop(f"{llm}/attn/attn_vec_einsum{suffix}/w")  # [L, heads, head_dim, width]
    gating = pop(f"{llm}/mlp{suffix}/gating_einsum")  # [L, 2, width, mlp_dim]
    linear = pop(f"{llm}/mlp{suffix}/linear")  # [L, mlp_dim, width]
    if adarms:
        norms = {
            name: (
                pop(f"{llm}/{jax_name}{suffix}/Dense_0/kernel"),
                pop(f"{llm}/{jax_name}{suffix}/Dense_0/bias"),
            )
            for name, jax_name in (
                ("input_layernorm", "pre_attention_norm"),
                ("post_attention_layernorm", "pre_ffw_norm"),
            )
        }
    else:
        norms = {
            "input_layernorm": pop(f"{llm}/pre_attention_norm{suffix}/scale"),
            "post_attention_layernorm": pop(f"{llm}/pre_ffw_norm{suffix}/scale"),
        }
    num_layers, num_heads, width, head_dim = q_einsum.shape
    for i in range(num_layers):
        layer = f"{prefix}.layers.{i}"
        out[f"{layer}.self_attn.q_proj.weight"] = _t(
            q_einsum[i].transpose(0, 2, 1).reshape(num_heads * head_dim, width)
        )
        out[f"{layer}.self_attn.k_proj.weight"] = _t(kv_einsum[i, 0, 0].transpose())
        out[f"{layer}.self_attn.v_proj.weight"] = _t(kv_einsum[i, 1, 0].transpose())
        out[f"{layer}.self_attn.o_proj.weight"] = _t(
            o_einsum[i].reshape(num_heads * head_dim, width).transpose()
        )
        out[f"{layer}.mlp.gate_proj.weight"] = _t(gating[i, 0].transpose())
        out[f"{layer}.mlp.up_proj.weight"] = _t(gating[i, 1].transpose())
        out[f"{layer}.mlp.down_proj.weight"] = _t(linear[i].transpose())
        for name, value in norms.items():
            if adarms:
                kernel, bias = value
                out[f"{layer}.{name}.dense.weight"] = _t(kernel[i].transpose())
                out[f"{layer}.{name}.dense.bias"] = _t(bias[i])
            else:
                out[f"{layer}.{name}.weight"] = _t(value[i])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, required=True, help="openpi checkpoint (contains params/)"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=50, help="action horizon of the checkpoint")
    args = parser.parse_args()
    # force=True: imported libraries may already have configured logging, which would hide progress.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)

    params_dir = args.checkpoint_dir / "params"
    if not params_dir.is_dir():
        raise FileNotFoundError(f"{params_dir} does not exist")
    logging.info("Restoring JAX params from %s (this takes a few minutes)...", params_dir)
    state_dict = convert_params(restore_jax_params(params_dir))

    from safetensors.torch import save_file

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(args.output_dir / "model.safetensors"), metadata={"format": "pt"})
    config = SLAConfig(chunk_size=args.chunk_size, n_action_steps=args.chunk_size, device="cpu")
    config._save_pretrained(args.output_dir)
    for assets in (args.checkpoint_dir / "assets",):
        if assets.is_dir():
            shutil.copytree(assets, args.output_dir / "openpi_assets", dirs_exist_ok=True)
    (args.output_dir / "conversion_info.json").write_text(
        json.dumps({"source": str(args.checkpoint_dir.resolve()), "num_tensors": len(state_dict)}, indent=2)
    )
    logging.info("Saved %d tensors to %s", len(state_dict), args.output_dir)


if __name__ == "__main__":
    main()
