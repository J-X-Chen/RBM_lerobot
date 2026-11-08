# Zerith H1 Pro

训练命令见仓库根目录 `zerith_order.txt`。PI0.5 和 DM0.5 统一使用根目录 `lerobot/src`。
可选 ACT 基线的 `python zerith_lerobot/scripts/train.py ...` 仍固定使用此目录下的 LeRobot。

- `scripts/inspect_dataset.py`：遍历全部 Parquet 数值列，检查有限值、维度顺序、URDF 限位，输出报告。
- `lerobot/src/lerobot/utils/zerith_action.py`：训练帧精确统计、恒定 action 筛选、动作恢复和映射序列化。
- `utils/constants.py`：数据列与 SDK 电机枚举、URDF 关节对应关系。底盘列是线/角速度，不能当作左右轮关节位置。
- `scripts/predict_dataset.py`：使用根目录 LeRobot 加载 PI0.5/DM0.5 checkpoint，对数据帧离线推理并导出 19D/23D 动作。
- `tests/test_zerith_action.py`：运行 `python -m unittest discover -s zerith_lerobot/tests -v`。

当前数据集为 35 episodes、34592 帧、30 FPS。严格恒定的 action 索引为
`[7, 19, 20, 21, 22]`（均为零），固定保留左夹爪，仅删除头部和底盘四维，训练使用 19 维。state 保留完整 23 维，
包括并非恒定的头部、底盘反馈。保留腰部三维和左臂七维，不能套用 Astribot 的 16D 切片。

`--dataset.zerith_action_view=auto` 在每次训练中自动检测，不修改原始数据。
左夹爪即使恒定也会参与训练。旧18D checkpoint保持原映射；19D训练应使用新输出目录。
动作映射同时保存于输出目录和 checkpoint 的 `pretrained_model/zerith_action_layout.json`。
推理必须用保存的映射恢复，不能再次根据单帧数据推断维度。使用绝对动作，禁止在
23D state / 19D action 上直接套用按索引相减的相对动作处理器。

本次提供 PI0.5 和 ACT 训练入口；此副本未包含 DM0.5。根目录 `lerobot` 已接入相同 Zerith 适配，DM0.5 训练命令见 `zerith_order.txt` 第7节。
机器人 SDK 1.3.6 的 Python 扩展需 Python 3.10；训练不导入该扩展。
数据夹爪值是 SDK 电机位置，并非 URDF 的手指平移量，训练不做未确认的单位转换。

H1部署服务见 `zerith_image_service/README.md`，评估入口见 `eval_robot/README.md`。
旧eval_s1入口已转发到H1评估，不再调用Astribot。提供离线/只读预览/交互执行、异步预取及数据回放。
本地模拟验证不代替H1真机联调。

## 已执行的验证

- 全量 34592 帧扫描，action/state 全部有限；可映射的 URDF 关节目标未超限。
- 6 项动作映射/统计/验证集测试通过。
- PI0.5 与 ACT 文档命令的配置解析通过。
- 此前18D版本的 CPU 小型 ACT 单步训练、反向传播、checkpoint 保存与重新加载通过；离线推理产出
  `[1, 2, 18]`，恢复为 `[1, 2, 23]`，五个恒定列恢复为零。
- 修复继承版本的 checkpoint 配置保存遗漏顶层 `type` 的问题。

当前环境没有可用 CUDA，未执行 PI0.5 完整训练。已有 `multiprocess` 在 Python 3.12
退出时会输出 `_recursion_count` 清理异常；上述训练和推理均正常完成、退出码为 0。

根目录版本验证：7项测试通过，包含小型DM0.5反向传播、保存、重载和推理。真实数据批次为
`action=[2,50,19]`、`state=[2,23]`，左夹爪列保留。尚未运行完整DM0.5微调。
