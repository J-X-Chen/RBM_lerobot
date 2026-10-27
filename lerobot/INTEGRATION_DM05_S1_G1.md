# DM0.5 的 S1 / G1 适配

DM0.5 在库中的注册名是 `dm05`。它与 π0.5 (`pi05`) 是不同模型。
使用共享根目录 LeRobot 训练，S1/G1 继续使用各自的同步或异步 `eval_robot`。

## 动作与数据约定

| 本体/模式 | 原始 state/action | 训练 state/action | 训练参数 |
| --- | --- | --- | --- |
| S1 全身 | 25 / 25 | 25 / 25 | `dataset.astribot_action_view=full` |
| S1 双臂夹爪 | 25 / 25 | 16 / 16 | `dataset.astribot_action_view=arms_grippers` |
| G1 Dex3 压缩 | 28 / 28 | 28 / 16 | `dataset.compress_dex3_actions=true` |
| G1 完整动作 | 28 / 28 | 28 / 28 | `dataset.compress_dex3_actions=false` |
| G1 忽略手部 | 28 / 28 | 28 / 16 | `dataset.compress_dex3_actions=ignore` |

DM05 核心使用 32 维 action 容量，适配层自动填充并 mask 未使用的维度，输出时裁回训练维度。
S1/G1 的动作视图、统计量和特征名都在创建模型和处理器之前转换；验证集同步转换。
原始数据不会重写。S1 16 维动作在本体评估中展开到全身参考姿态，G1 16 维动作展开为 Dex3 28 维。

保持 `policy.use_relative_actions=false` 可直接学习现有数据中的目标动作。
相对动作需要等长 state/action，且统计量必须来自转换后的相对动作；G1 28/16 压缩模式不支持该选项。
两套同步评估现在将相对动作整块还原后缓存，避免每帧状态更新改变已生成动作块的基准；
异步评估原有的整块预测、解码流程继续适用。

## 这次修复

1. `lerobot_train.apply_robot_dataset_views` 统一训练与离线验证的 S1/G1 动作转换，适用于 DM05 和已有策略。
2. 模型工厂增加显式 `preserve_checkpoint_features` 推理选项。
   DM05 的各本体评估入口使用它，保留 checkpoint 的 state/action、相机特征和动作名，
   避免用原始 25/28 维参考数据覆盖压缩模型的 16 维接口。
   微调仍通过原有工厂流程采用目标数据集特征与统计量。
3. S1/G1 同步、异步入口优先加载项目根目录 `lerobot/src`，不再被旧嵌套库抢占。
4. 两套评估客户端的相对动作块缓存增加 DM05 支持。

未复制或修改 DM05 核心网络；其原有的状态离散化、Gemma3 多相机处理、32 维 mask、
保存/加载和优化器接口已具备所需能力。

## 运行

在项目根目录的相应环境安装：

```bash
python -m pip install -e './lerobot[training,dm05]'
```

各评估环境还要保留本体原有 SDK 等依赖。完整训练命令在项目根目录 `order.txt` 的第 9–13 节。

使用 `--policy.path=lerobot/dm05_base`，或完整的本地 LeRobot DM05 checkpoint。
不要将未经转换的 `Dexmal/DM05` 原始权重当作 `--policy.path`。
完整 checkpoint 包含模型配置、权重、前后处理器文件、处理器统计量及 `dm05_processor/`。
本地模型可放入 `base_models/dm05_base` 或 `pretrained_models/dm05_base`，也可传绝对路径。

DM05 的梯度检查点参数为 `policy.vlm_gradient_checkpointing` 和 `policy.ae_gradient_checkpointing`；
不能照搬 π0.5 的 `policy.gradient_checkpointing`。需要全参数训练时显式设置
`policy.freeze_vlm_embedding=false`。相机顺序可用 `policy.image_keys` 显式固定。
评估时只替换 checkpoint，并保持本体、相机配置和 S1 动作视图与训练一致。

## 验证

`tests/policies/dm05/test_s1_g1_dm05.py` 使用真实 LeRobot 临时图像数据集、真实 Gemma3 处理器
和缩小版 DM05 核心网络，对上表五种模式执行：

- DataLoader 读取、动作视图与验证集一致性检查；
- 前向、反向传播和优化器更新；
- 保存完整 checkpoint、重新加载并比较固定噪声推理结果；
- 用原始全维数据作为评估参考，确认 checkpoint 维度不会被覆盖；
- 调用两个本体实际的同步/异步预测函数，检查输出和动作展开；
- 检查四个入口的共享库优先级，以及相对动作块的状态基准。

测试隔离导入预测函数，未加载机器人 SDK 或连接实机。它验证代码及数据链路，
不代表已验证完整预训练模型的 GPU 微调、显存需求或实机效果。

本次针对性回归共 73 项通过，包含 DM05 上游测试及已有 S1/G1、π0.5 功能回归。
另外通过共享 `lerobot_train` CLI，对缩小版 DM05 分别运行 S1 16D 和 G1 Dex3 16D
两步 CPU 微调，从本地 checkpoint 加载处理器与权重，并成功保存新 checkpoint。
临时工具依赖与输出保存在 `/tmp/lerobot-integration`，未修改现有 conda 环境。
现有环境退出时仍有 `multiprocess.resource_tracker` 清理告警，测试及两次 CLI 训练退出码均为 0。
