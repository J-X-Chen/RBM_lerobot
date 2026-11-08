# Zerith H1 评估

`eval_h1.py` 使用根目录 LeRobot 加载 PI0.5、DM0.5 的19D checkpoint，兼容ACT。
`eval_h1_async.py` 使用同一个状态机并启用异步预取。旧 `eval_s1.py` / `eval_s1_async.py`
已变成H1入口别名，不再启动Astribot控制器；参数以 `eval_h1.py --help` 为准。
`make_robot.py` 创建H1远程控制与相机客户端，原有S1资产/控制模块保留但新入口不导入。

## 测试流程

1. H1服务只读：相机预览 + `safe_probe` 读状态。
2. H1服务执行模式：保持当前姿态、单臂关节微动（独立于数据集和模型）。
3. `replay_h1.py` 不加 `--execute` 检查数据，必要时带 `--execute` 人工回放。回放不加载模型，也不代替前两步。
4. `eval_h1.py --mode=offline`：用数据集起始帧推理，不连接H1。
5. `--mode=preview`（默认）：用真实三相机和状态推理，仅记录预测，不arm也不发送动作。
6. `--mode=live --execute`：需交互终端；r慢速对齐数据集第一帧的19D状态，测量到位后s启动，p暂停并保持，b返回首帧，j保存三路相机快照，q锁止退出。

默认episode=0、30Hz；`--episode`切换测试episode，任务文本默认从数据读取，可用`--task`覆盖。
左夹爪保留；头部维持arm时姿态，底盘静止。模型必须声明23D状态、19D动作、三路Zerith图像、绝对动作；必须带 `zerith_action_layout.json`。
其他动作视图被拒绝，不能将旧18D checkpoint直接上线。

checkpoint的 `n_action_steps` 决定每个预测块执行多少步，CLI可覆盖；PI05训练命令为50，DM05为4。
同步模式执行完动作后再预测，推理期间服务端保持最后目标；心跳仍由主线程发送。
异步模式在剩余 `--prefetch-steps` 步时预取；根据预测期间已经实际执行的动作数丢弃重叠前缀，再替换剩余队列。
这是异步预取，不是spark参考代码中的RTC软引导；没有实现/宣称RTC。
暂停/回位改变generation，旧预测结果不会再次提交；推理默认超过15秒退出，SDK控制端仍独立检查心跳。
实时执行时通过控制响应提供的本体单调时基检查相机发布时效（0.5秒）和三路发布时间差（0.25秒），拒绝网络积压的旧帧；没有声称三路图像/机器人状态硬同步。

所有日志用新文件名，包含状态、预测动作、实际提交目标、服务端限速后命令及跟踪诊断。
出现故障、q退出或网络断开均无自动回位；先检查H1，再重启服务。
部署及参数见 `zerith_image_service/README.md`，完整命令见根目录 `zerith_order.txt` 第8节。
