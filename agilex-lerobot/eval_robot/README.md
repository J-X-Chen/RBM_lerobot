# ALOHA / cobot_magic PI0.5、DM0.5 实机评估

使用项目 `./lerobot` 算法及训练保存的处理器，输入三路 RGB 和14维状态，输出14维绝对动作。
顺序为左6关节+左夹爪、右6关节+右夹爪。关节rad，夹爪数据单位通过机器人端配置转换为物理米。
不涉及 S1/G1 控制代码。

## 推荐：网络执行

发送端使用 `agilex_image_service`，两端完整命令见 `order.txt` 第21/22节。

```bash
# 机器人端，已填写实际硬件标定：
bash start_all.sh --hardware-config=./hardware.json --execute-control
# 推理机，项目根目录、训练环境：
CUDA_VISIBLE_DEVICES=0 python agilex-lerobot/eval_robot/eval_aloha.py \
    --policy.path=./lerobot/outputs/pi05_aloha_red_cup_14d/checkpoints/last/pretrained_model \
    --mode=live --robot-host=机器人IP --execute --steps=3000
```

DM05替换为相应训练输出路径。去掉客户端 `--execute` 可只读预览，客户端不会发送动作请求。
机器人端也去掉 `--execute-control` 可完全禁止执行。
TCP6200读状态/下发动作，JPEG端口55555/55556/55557依次head/left/right；默认图像地址跟随robot-host。
可用 robot-port、image-server-address、head-camera-port/left-camera-port/right-camera-port 覆盖。

读取真实状态和图像，完成5次预热并清空动作队列后，按r对齐或按s开始时才使能；按键无需回车。q/第一次Ctrl+C正常退出时缓慢回到首次使能前的关节姿态。
默认关节0.3rad/s、夹爪0.02m/s，客户端/服务端同名参数取较小值；服务器限制不可被客户端绕过。
默认两臂预测；`--active-arm=left/right` 让另一臂保持启动时姿态，模型仍输入完整14维。
每步从最新反馈和图像推理/执行动作队列，n-action-steps默认使用checkpoint，可手动覆盖。
任务默认来自该数据集meta/tasks.parquet，也可通过 `--task` 指定。
只读日志command是预览目标，执行日志command是服务器返回的限速后下发目标。

无新动作5秒、反馈1秒/相机2秒过期或控制故障时停止；预热/用户输入不计入动作超时。
急停锁存后检查硬件并重启服务再运行，不自动回零、松爪、掉电或恢复急停。
实际控制频率受同步推理影响，默认目标30Hz。软件限速不是物理避碰保证。

## 本机 CAN（备用）

不传robot-host，改传 `--hardware-config=./agilex-lerobot/eval_robot/hardware.json`；
从hardware.example.json复制并填写实际值，本地安装 `pip install -e ./piper_sdk`。
该模式使用与服务端相同的 `agilex_image_service/piper_control.py`，图像地址需独立指定。
不要同时运行网络控制服务和本地CAN控制。

## 离线与验证

未传mode时仍支持原有数据集离线检查；代码保留但不作为日常实机流程的前置步骤。
测试位于tests/，模拟SDK并验证TCP/JPEG通信，不连接实际CAN或机器人。
实际相机、CAN和运动尚需现场验证；发送端部署与协议见 `agilex_image_service/README.md`。

## r / s / q 与退出回位

只读模式r/s无运动效果，q/Ctrl+C直接退出。执行模式r对齐episode首帧状态并等待s；s开始策略。
正常q、第一次Ctrl+C或steps结束时，按return-joint-speed（默认0.15rad/s）回到第一次使能前姿态，
夹爪保持退出时开口，不自动松爪。边界外小幅起始读数会投影到名义限位。
回位中再次q/Ctrl+C中止并请求急停；异常退出不尝试回位。return-timeout默认60秒。
return-gripper-speed默认0.01，影响r对齐夹爪；服务端速度限制仍有效。
首次Ctrl+C通过标志请求正常退出，不打断TCP帧；损坏/失败的TCP连接直接关闭，服务端停止。
推理同步执行，退出请求需等当前推理调用返回；必须保留实体急停。
相机/反馈过期、服务端已锁存故障时不能回位。该机制未经实机验证，不提供避碰。
