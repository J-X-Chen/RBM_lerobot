# ALOHA / Cobot Magic：PI0.5 与 DM0.5 训练

使用本项目 `lerobot/` 的算法和训练器，数据为项目根目录下
`lerobotv3_datasets/aloha_cup_selection_red`。无需机器人 SDK、ROS 或连接本体即可离线训练。

## 数据接口

- v3.0，30 FPS，105 个 episode，43,935 帧。
- `observation.state` 和 `action` 都是 14 维，顺序为左臂 joint1–6、左夹爪、右臂 joint1–6、右夹爪。
- 三路 RGB：`observation.images.head`、`.left`、`.right`，原始尺寸 480×640。
- 任务文本：`Select the red cup and place it on the black coaster`。
- 学习数据中存储的绝对关节目标，保留原始单位，不做 S1 裁剪、Dex3 压缩、夹爪缩放或额外时间偏移。
- 模型内部沿用已有 32 维容量和图像预处理，输出恢复到 14 维。

原 `meta/stats.json` 的 q01/q99 与原始数值不符，例如 action 的左 joint2
原 q99≈0.989，真实全局 q99≈2.130；该维约 78.8% 的动作落在原区间外。
两个模型均使用 QUANTILES，DM05 默认还会裁剪，因此不能直接沿用这些统计。
训练器在识别 `robot_type=cobot_magic` 后，从实际选中的训练帧计算精确数值统计，
验证集复用训练统计，原始数据文件不改写。保存的模型处理器包含新统计，推理应加载完整 checkpoint。
恢复训练沿用 checkpoint 保存的处理器统计；旧错误统计训练出的模型应重新微调，不能靠 resume 修复。

从基础 checkpoint 微调时，模型工厂用本数据集替换旧相机名和 state/action 特征。
S1/G1 分支和两种模型的核心算法保持原有实现。
本适配针对单个本地 v3 数据集；暂不支持 streaming、多数据集混合或相对动作。

## 启动

在项目根目录，进入已有训练环境，确保导入共享库：

```bash
export PYTHONPATH="$PWD/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
python -m lerobot.scripts.train_cobot_magic --policy pi05 --check-data
```

该检查无需模型权重，会计算精确统计，并读取每个 episode 的首尾帧及三路视频，检查动作块和任务文本。

训练 PI0.5：

```bash
python -m lerobot.scripts.train_cobot_magic --policy pi05 \
  --pretrained-path=/absolute/path/to/pi05_base \
  --output-dir=outputs/train/aloha_red_cup_pi05 \
  --batch_size=8 --steps=30000
```

训练 DM0.5：

```bash
python -m lerobot.scripts.train_cobot_magic --policy dm05 \
  --pretrained-path=/absolute/path/to/dm05_base \
  --output-dir=outputs/train/aloha_red_cup_dm05 \
  --batch_size=4 --steps=30000
```

`--pretrained-path` 省略时分别使用 `lerobot/pi05_base` 和 `lerobot/dm05_base`，需要能访问 Hugging Face
及对应模型授权。DM05 必须使用完整的 LeRobot 格式 checkpoint，不能直接把 `Dexmal/DM05` 当作
`--policy.path`。PI05 还需可用的 PaliGemma tokenizer；必要时追加
`--policy.text_tokenizer_name=/absolute/path/to/tokenizer`。

默认 batch_size=1、bfloat16、梯度检查点开启、compile 关闭、30,000 步、每 5,000 步保存、不开 WandB/Hub 上传。
上面的 batch_size 仅为可调示例，按实际显存调整。DM05 使用其独立的 VLM/AE 梯度检查点配置。
追加 `--dry-run` 可查看实际标准训练命令；其它训练参数用 `--key=value` 追加。

如需留出验证集，追加 `--dataset.eval_split=0.1`（按 episode 划分）。
权重与前后处理器位于输出目录 `checkpoints/last/pretrained_model`；续训使用标准训练器和保存的训练配置。

## 验证范围

`tests/datasets/test_cobot_magic.py` 检查精确分位数、训练/验证统计隔离、关节顺序和错误参数拦截。
`tests/policies/test_cobot_magic_training.py` 对两个缩小模型运行真实前后处理、三相机编码、
loss/backward、优化器更新、checkpoint 保存重载，以及 14 维动作输出比较。
设置 `COBOT_DATASET_ROOT` 可直接用这份实际视频数据执行该测试。

本次验证：49 项针对性测试通过（含 S1/G1 与 DM05 回归）；两个模型另用实际数据各通过一次
训练/重载测试。正式 `lerobot_train` CLI 也分别完成了两步 CPU 小模型微调并保存 checkpoint，
日志为 `/tmp/cobot-cli-pi05.log`、`/tmp/cobot-cli-dm05.log`。全部 105 个 episode 的首尾帧检查通过。
环境退出时有已有的 `multiprocess.resource_tracker` 清理告警，测试与训练退出码均为 0。

当前执行环境 NVIDIA 驱动不可用，因此 CPU 小模型验证不能替代完整预训练权重的 GPU 微调或实机成功率验证。

## 参考

参考目录 `agilex-lerobot/lerobot` 没有独立 cobot_magic 实现。
[AgileX ALOHA](https://github.com/agilexrobotics/aloha-agilex) 的
[训练入口](https://github.com/agilexrobotics/aloha-agilex/blob/master/act/aloha_train.py) 与
[数据读取](https://github.com/agilexrobotics/aloha-agilex/blob/master/act/utils.py)
提供旧 HDF5/ACT 训练参考。本次以现有 v3 数据的特征定义为准，不移植旧算法或 ROS 依赖。
