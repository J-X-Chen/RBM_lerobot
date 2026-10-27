# S1 / G1 集成最终检查

本次检查覆盖共享训练库、DM0.5 在 Astribot S1 / Unitree G1 的数据维度转换、checkpoint 保存加载，以及各自同步/异步评估的推理代码。旧本体库仍保留；未执行实机控制。

## 检查中修复

- GR00T N1.7 加载 processor assets 时传入了不支持的 `words_ignore` 参数：移除错误传参，保留实际输入处理步骤中的词过滤功能。
- π0 / π0.5 读取缺失、损坏或不兼容权重时，原先可能警告后继续使用随机或部分加载权重：现在直接抛出异常。新增正常加载和失败路径回归测试。
- π0 权重下载转发调用者指定的 revision、cache_dir、local_files_only 等参数。
- `order.txt` 修正目录大小写，并补充独立的 S1 DM0.5 16D 命令：`astribot_action_view=arms_grippers`、`dm05_s1_16d` 输出目录及任务名。相关章节引用同步更新。

## 已验证

- 离线针对性回归：**351 passed, 11 skipped**。日志：`/tmp/lerobot-integration/audit-final.txt`；运行脚本：`/tmp/lerobot-integration/final_audit_tests.py`。这不是整个上游测试库全部通过的声明。
- 跳过项：8 项需要 CUDA，3 项需要缺失的 diffusers / GR00T extra。
- DM0.5 使用微型真实模型完成 forward、backward、优化器更新、checkpoint 和 processor 保存重载及固定噪声输出对比。覆盖 S1 state/action 25/25、16/16；G1 28/16、28/28 及 ignore 模式。
- S1 / G1 各完成共享训练 CLI 的 CPU 两步 DM0.5 微调冒烟测试，保存 checkpoint 成功；输出位于 `/tmp/lerobot-integration/dm05-cli-s1` 和 `dm05-cli-g1`。
- 同步/异步评估测试执行提取出的实际推理函数，检查维度恢复和相对动作队列锚点；未连接机器人 SDK 或真实硬件。
- S1 DM0.5 16D 新命令经 shell 分词和 draccus 配置解析验证。为避免下载，仅在验证时将 `policy.path` 换成等价的 `policy.type` / `policy.pretrained_path`；验证维度视图、三相机配置、输出目录、任务名及绝对动作选项，未启动训练。

## 尚未验证与环境条件

- 扩大测试范围时另有 2 项 OpenPI processor 对比依赖外部 tokenizer、13 项 relative-action 测试依赖远程数据集，离线环境缺少这些资产；未计入最终通过结果。
- 当前会话 CUDA 不可用，未验证完整 DM0.5 基础权重的 GPU 显存需求、长时间训练和实机动作效果。
- `s1_lerobot` 原环境缺少 accelerate、wandb、sentencepiece 等训练依赖。测试通过 `/tmp/lerobot-integration/test-tools` 临时补充依赖；未安装到原环境。正式训练前执行 `order.txt` 第 9 节安装步骤。
- 当前 TorchCodec 无法加载，训练命令显式选择 pyav。Python 退出时仍有 multiprocess ResourceTracker 的 RLock 清理异常提示；测试退出码为 0，此提示不代表已修复。

详细适配设计参见 [DM0.5 集成说明](INTEGRATION_DM05_S1_G1.md) 和 [整体集成说明](INTEGRATION_S1_G1.md)。
