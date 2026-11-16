# SLA（Spatial-Language-Action）

LeRobot 版 SLA：[SLA: Scaling Spatial-Language-Action Model via Continual Pre-training](https://arxiv.org/abs/2611.08373)
（参考实现：`Spatial-Forcing/SLA`，openpi/JAX）。

SLA 的 VLA 就是 pi0.5，因此本策略直接继承 LeRobot 的 PyTorch `PI05Policy` / `PI05Pytorch`，
只增加训练期的空间对齐分支：

| 组件 | 参考实现 | LeRobot |
| --- | --- | --- |
| VLA 主体（SigLIP + Gemma 2B + 300M 动作专家，flow matching） | `openpi/models/pi0.py` (pi05) | 复用 `policies/pi05` |
| 对齐层：PaliGemma 第 12 层（0 起）图像 token 输出 | `gemma.py: HiddenCaptureSpec` | `PaliGemmaWithExpertModel.forward(prefix_hidden_capture=...)` |
| 投影器：LayerNorm → Linear → GELU(tanh) → Linear (2048→2048) | `models/projectors.py` | `alignment.py: AlignProjector` |
| 目标：冻结 OmniVGGT 末层 aggregator 特征（37×37 → 双线性 + VGGT PE → 16×16） | `training/align_utils.py` | `modeling_sla.py: compute_align_targets` |
| 损失：`action_loss + 0.5 * (1 - cos)`，忽略缺失相机和 resize-pad 填充区域 | `pi0_align.py` | `SLAPolicy.forward` |
| 训练期 state dropout（p=0.25，去掉 prompt 中的离散 state） | `transforms.StateDropout` | 预处理同时生成无 state prompt，策略在 train 模式下按样本选择 |

推理与 pi0.5 完全一致，不需要 OmniVGGT。OmniVGGT（约 9.4 亿参数，`vendor` 于 `sla/omnivggt/`）只在
训练时懒加载，不注册为子模块：不进 checkpoint、优化器、DDP，也不随 `.to(dtype)` 转换。

## 权重

- **OmniVGGT 教师**：`--policy.vggt_weight_path` 可为 `OmniVGGT.safetensors` 文件、包含它（或 `omnivggt/` 子目录）
  的目录、或 Hub 仓库（默认 `haofuly/SLA_training_component`）。
- **起点权重**（二选一）：
  1. `lerobot/pi05_base`（PyTorch，无需转换；对齐投影器随机初始化）。
  2. 官方 SLA 预训练 `haofuly/SLA_base`（仅发布了 JAX orbax 格式）。需要先一次性转换权重格式
     （需 `pip install "jax[cpu]" orbax-checkpoint`）：

     ```bash
     python -m lerobot.policies.sla.convert_sla_jax_checkpoint \
       --checkpoint-dir base_models/SLA_base_jax --output-dir base_models/sla_base
     ```

     `SLA_benchmark_ckpts` 中的后训练 checkpoint 同样适用（`--chunk-size` 设为其 action horizon）。

## 训练

```bash
lerobot-train \
  --dataset.repo_id=local/aloha_cup_selection_right \
  --dataset.root=lerobotv3_datasets/aloha_cup_selection_right \
  --policy.type=sla \
  --policy.pretrained_path=base_models/sla_base \
  --policy.vggt_weight_path=base_models/SLA_training_component \
  --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \
  --policy.device=cuda --policy.push_to_hub=false \
  --ema.enable=true --ema.decay=0.999 \
  --batch_size=8 --steps=20000 --output_dir=outputs/train/sla_cup
```

常用参数（默认值即参考实现的真实机器人配置）：

- `align_loss_coeff=0.5`（设 0 即退化为 pi0.5 训练）、`vla_layers_align=12`、`vggt_layers_align=-1`
- `use_vlm_norm=true`、`use_vggt_pe=true`、`ignore_img_padding_area=true`、`state_dropout_prob=0.25`
- `align_vggt_device=cuda:1`：单进程时把教师放到另一张卡；`align_feature_batch_size`：分块提取以省显存

注意：

- 对齐要求 PaliGemma 图像 token 与 VGGT 特征在空间上一一对应。参考实现在对齐训练时关闭了
  随机裁剪/旋转，只保留颜色抖动。不要在 SLA 训练中开启几何类图像增强。
- 不支持与 MEM 视觉记忆、DINOv2/GateTwoResViT RGB 编码器或 `train_expert_only` 同时使用（配置校验会报错）。
- OmniVGGT 的相机参数输入（`use_camera_params`）在参考实现的所有配置中都关闭，这里未实现。
