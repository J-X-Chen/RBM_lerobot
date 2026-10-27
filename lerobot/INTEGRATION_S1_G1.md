# Astribot S1 / Unitree G1 共享 LeRobot

本目录以新下载的上游 `146d8474` 为基础，迁移两份旧库相对上游
`8a74e0ac6d01706d67fddfed682a09d694d9c8c0` 的项目定制。
训练使用本目录，评估继续使用各本体原有的 `eval_robot`。
初次整合未修改两套 `eval_robot`；后续按要求补充 DM0.5 适配，已更新同步/异步评估入口和相对动作缓存。两份旧库仍未删除。
DM0.5 细节见 [专项说明](INTEGRATION_DM05_S1_G1.md)。

## 切换环境

在项目根目录、对应的 Python 3.12+ 环境执行：

```bash
uv pip install -e './lerobot[training,pi]'
python -c 'import lerobot; print(lerobot.__file__)'
```

需要 GR00T 时追加 `groot` extra；需要上游 G1 驱动时追加 `unitree_g1` extra。
训练环境和各本体的评估环境都需要分别安装共享库。请安装依赖，而不仅仅修改
`PYTHONPATH`：新上游要求 `draccus>=0.11.6,<0.12`，旧环境中的 0.10 不兼容。
完整依赖和 GPU 版本要求以本目录 `pyproject.toml` 为准。

**后续 DM0.5 适配已将两套同步/异步评估入口改为优先加载根目录 `lerobot/src`。**
因此这四个入口不再等待旧库删除才切换；其他未修改的回放等脚本需单独确认导入路径。
各评估环境仍需安装共享库的依赖，并保留本体 SDK。
训练时从项目根目录执行下面的模块入口，确认 `lerobot.__file__` 指向本目录。

## 训练参数

示例中的数据路径需要替换为实际路径。模型参数、训练步数、batch size 等可继续沿用
原有非热成像配置；同一次训练的数据必须具有兼容的特征、动作语义和本体类型。

S1 保留全身 25 维动作和状态：

```bash
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/s1 \
  --dataset.root=/path/to/s1_dataset \
  --dataset.astribot_action_view=full \
  --policy.type=act \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/s1_full
```

S1 仅训练双臂和夹爪时，改为 `--dataset.astribot_action_view=arms_grippers`。
它将 25 维 action 和 observation.state 同时筛成 16 维，并同步变换统计量和特征名；
兼容原生 16 维数据。`auto` 模式和旧参数 `compress_astribot_actions` 也保留。
默认 S1 使用 `full`，不会误触发 Dex3 压缩。

G1 Dex3 使用：

```bash
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/g1 \
  --dataset.root=/path/to/g1_dataset \
  --dataset.compress_dex3_actions=auto \
  --policy.type=act \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/g1_dex3
```

| G1 参数 | 行为 |
| --- | --- |
| `auto`（默认） | 仅对 robot_type 含 `dex3` 且 action 为 28 维的数据压缩成 16 维 |
| `true` | 显式启用 Dex3 压缩 |
| `false` | 保留原始动作 |
| `ignore` | 使用 16 维接口，两侧握持目标置零 |

G1 压缩 action 后 observation.state 仍为原始 28 维，与 S1 的状态筛选不同。
动作转换只发生在运行时，磁盘数据保持原样；训练集、离线验证集和 DataLoader 批量读取都应用相同转换。

多个同构本地数据集可用：

```bash
--dataset.root='["/path/to/session_a", "/path/to/session_b"]'
```

共享任务编号、episode/frame 索引和合并统计量会被重新组织。可以使用
`--dataset.feature_names='["observation.state","action","observation.images.cam_left_high"]'`
限制加载特征；必要的 bookkeeping 字段自动保留。
这不代表能够直接混合 S1 和 G1 的异构动作空间。
多根目录合并及动作视图用于 map-style 数据，不能直接与 streaming 模式组合。

## 保留的项目定制

- S1 全身/双臂夹爪视图及 G1 Dex3 压缩、忽略手部、动作还原工具。
- 本地多数据集合并、特征白名单、视频解码失败时的备用解码路径。
- 项目本地模型和 tokenizer 查找、离线缓存优先、W&B 离线回退和 SOCKS 地址规范化。
- π0 / π0.5 评估 attention map 接口，同时兼容两个本体原有的捕获开关。
- π0.5 旧版 resize padding、缺失相机的固定槽位顺序、`words_ignore`、旧处理器回退。
- 纯 RGB DINOv2，以及 `rgb_input_type=withthreeblack` / `GateTwoResViT` 的三色证据门控融合。
  这些组件已经从旧热成像模块中独立出来。RGB 三色分支的 dropout 参数改为
  `policy.rgb_threeblack_dropout`；读取旧 checkpoint 时自动迁移对应值。
- GR00T 的本地完整 checkpoint 初始化（`load_base_model=false`）、语言过滤和 S1/G1 embodiment。
- 旧 `lerobot.types`、设备工具、G1 locomotion 导入路径的兼容接口。

新上游已有的 G1 驱动使用新版实现，没有用旧驱动整体覆盖。
新上游的 MEM、训练时 RTC、批量视觉推理及训练框架继续保留。
旧 RGB DINOv2/三色门控路径尚不支持新 MEM 的视觉历史输入，启用时会明确拒绝该组合；
默认 SigLIP 路径保留视觉历史能力。

本地模型搜索包括 `base_models` 和 `pretrained_models`。
可用 `LEROBOT_LOCAL_MODEL_ROOT=/path/to/models` 显式指定搜索根目录。
DINOv2 需要本地代码和权重，可分别通过
`policy.rgb_dinov2_repo_path`、`policy.rgb_dinov2_checkpoint_path` 指定。

## 已舍弃的内容

按要求不再包含 ThermoAct、热成像编码器、热/RGB 融合、热图匹配、热 CVAE 及相关训练参数。
旧非热成像 checkpoint 自动忽略保存下来的未启用热成像配置字段。
真正启用了热成像功能的 checkpoint 会明确报错，不会当作普通 RGB 模型静默加载。
原 `eval_robot` 内的热成像相关分支未改动，但共享库不再提供其训练和模型实现。

## 验证范围

- 数据集、动作视图、配置、旧接口、RGB 分支和上游 π 模型针对性回归测试：188 项通过，8 项因 GPU 条件跳过。
- 改动文件的 Python 语法检查、`git diff --check` 和 Ruff 检查通过；Ruff 排除上游原有的 `UP042` 枚举风格提示。
- 检查两套 `eval_robot` 引用的 23 个非热成像 LeRobot 导入符号，全部可解析。
- 用临时合成数据分别完成 S1 25→16 和 G1 Dex3 28→16 的 CPU ACT 两步训练，保存 checkpoint；
  然后重新加载 checkpoint、恢复前后处理器并成功推理 16 维动作。
- 验证数据转换没有改写原始数据特征，以及多根目录数据集的批量动作序列读取。

这些验证不等同于完整 π0/π0.5/GR00T GPU 微调、真实数据训练或 S1/G1 实机评估。
测试使用隔离的临时工具目录，未替换现有环境中安装的 LeRobot。
现有测试环境在退出时有 `multiprocess.resource_tracker` 的清理告警；测试和 CPU 训练退出码为 0。
