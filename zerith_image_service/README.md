# Zerith H1 本体服务

此目录可以与 `zerith_H1_SDK1.3.6` 一起单独放到 H1。H1 端不需要 Torch、LeRobot、模型权重、Astribot SDK 或 ROS。
工作站运行根目录 `lerobot` 的 PI0.5/DM0.5，入口为 `zerith_lerobot/eval_robot/eval_h1.py`。

`spark/` 保留原作者文件供参考，不作为此流程的运行入口。其 HTTP 服务原本在工作站推理并输出旧23D动作；本服务采用 Astribot 的本体端图像/控制分离方式。

## 部署

在工作站生成压缩包：

```bash
python -m zerith_image_service.build_package
```

将 `zerith_image_service/dist/zerith_h1_deploy.tar.gz` 复制到 H1 并解压。压缩包包含服务代码、测试和原始 SDK，排除 spark、训练库、权重和缓存。解压后在这两个目录的父目录运行命令。

优先使用厂商已有的 **Linux x86_64、Python 3.10** SDK 环境。先检查：

```bash
python --version
ldd zerith_H1_SDK1.3.6/h1_sdk_v1.3.6_python3.10/lib/lib_h1_sdk_python.so
python -m pip install -r zerith_image_service/requirements.txt
python -m zerith_image_service.server --list-cameras
```

`ldd` 不应有 `not found`。交付的机器人扩展要求 GLIBC_2.38、GLIBCXX_3.4.32，以及系统 protobuf/gRPC/ZCM；Python pip 包不能替代这些系统库。本工作站缺少这些库，尚不能加载原生SDK。
SDK 相机 requirements 中的 numpy==2.4.0 与 Python3.10 不兼容，本目录使用支持3.10的范围；最终仍需在厂商环境验证相机扩展。不会替换厂商系统库。

`--list-cameras` 输出真实相机ID。默认候选为 `rs/cam_high rs/cam_left_wrist rs/cam_right_wrist`，如不同，通过 `--camera-names` 按头/左腕/右腕顺序指定；服务不会把不存在的相机静默替换成别的相机。

## 分阶段测试（与 Astribot 相同顺序）

### 1. 默认只读

H1：

```bash
bash zerith_image_service/start.sh --bind-host=0.0.0.0
```

工作站（先将 H1_HOST 设为实际 H1 地址）：

```bash
python -m zerith_image_service.client --host="$H1_HOST"
python -m zerith_image_service.safe_probe --host="$H1_HOST" --log=eval_action_logs/h1_readonly.jsonl
```

仅连接、读反馈、读相机，不切模式、不初始化、不写关节。相机窗口按 q 退出。无图形界面时使用 `--no-display --frames=30`。

### 2. 当前姿态保持 / 单关节小动作

先在厂商控制端完成 LOW_LEVEL 初始化。若确实需要通过 SDK 初始化，可单独执行以下命令（初始化可能移动机器人，不能用来代替只读测试）：

```bash
python -m zerith_image_service.initialize --sdk-root=./zerith_H1_SDK1.3.6 --execute
```

H1 停掉只读服务后重新启动：

```bash
bash zerith_image_service/start.sh --execute-control
```

工作站先测试只保持，再重启服务做单关节测试：

```bash
python -m zerith_image_service.safe_probe --host="$H1_HOST" --execute --log=eval_action_logs/h1_hold.jsonl
python -m zerith_image_service.safe_probe --host="$H1_HOST" --execute --joint=0 --delta=0.01 --log=eval_action_logs/h1_joint0.jsonl
```

每次需终端输入 r；测试8秒，先保持2秒，再以0.005rad/s移动指定臂关节，最多0.02rad；q退出，不自动回位。
日志使用独占创建，重跑须换新文件名。每次 disarm、故障或控制客户端断开后需检查并重启服务。

### 3. 数据回放 / 模型评估

见根目录 `zerith_order.txt` 第8节，及 `zerith_lerobot/eval_robot/README.md`。先检查数据，再离线推理，再只读实时推理，最后才运行 `--mode=live --execute`。

## 接口与控制语义

- 图像：ZMQ PUB 默认端口55555/55556/55557；multipart `[JSON元数据, JPEG]`。SDK返回BGR，工作站模型输入时转RGB。只发布新的硬件时间戳，冻结帧不会被重新标记为新帧。图像携带本体发布单调时间，工作站用控制响应校准该时基以拒绝积压旧帧，不依赖两台机器墙钟同步。
- 控制：TCP 6200，10字节ASCII长度 + JSON。每个请求必须带 `protocol: zerith_h1_v1`，不能混用S1客户端。
- 请求操作：`get_state`、`arm`、`heartbeat`、`action`（严格19D）、`hold_current`、`disarm`，每次仅一个操作。
- state严格23D：左臂7、左夹爪、右臂7、右夹爪、腰部3、头部2、底盘线/角速度。SDK电机枚举顺序与数据顺序不同，`sdk.py`逐项映射。
- action严格19D：前19列。头部保持arm时测量姿态，底盘两轮持续发零速度。不会把删除维度训练时的零值当成头部目标角度。
- 夹爪保持SDK电机位置单位0~1.5，不转换成URDF手指位移米数。对齐时容忍开夹爪的微小负反馈，记录仍保留原始反馈。
- 本体循环默认100Hz；关节速度0.15rad/s、升降0.02m/s、夹爪0.5单位/s；加速度限制为各速度限值的4倍。URDF检查位置上下限，不把URDF升降的零velocity字段当作实际速度限值。
- arm时位移范围：关节±0.5rad、升降±0.15m；可通过CLI调整。完整任务可能需要扩大范围，应在小动作验证后再调整。
- 单客户端持有控制权；其他客户端可读取或disarm，不能替换目标。服务端检查断线/0.5秒心跳超时、周期超时、SDK状态、URDF限位、测量运动和跟踪误差；故障锁止后不再发新目标。
- `hold_current`继续有界保持；`disarm`停止下发，不是硬件急停。SDK可能保持最后目标或有自己的超时行为，需在H1上验证；程序不自动deinit，避免机械臂失去支撑，也不自动回位。
- SDK逐电机接口没有原子提交与硬件采样时间戳。`state_age_s`是成功完成一次SDK轮询后的时间，不证明底层反馈包的新鲜度；也不是全身同步采样或碰撞检测。局域网接口没有鉴权，仅用于受信任隔离机器人网络。

## 本机模拟

```bash
python -m zerith_image_service.server --mock --execute-control --bind-host=127.0.0.1
python -m zerith_image_service.client --host=127.0.0.1 --no-display --frames=5
python -m unittest discover -s zerith_image_service/tests -v
```

`--mock`只能显式选择，原生SDK加载失败不会回退到模拟。真实连接、相机方向/单位、夹爪力度和时序仍需H1联调。

## 本次验证记录

- 服务端15项测试通过：模拟SDK映射/限幅/超时/锁止，以及本机真实TCP+ZMQ收发、积压/过期图像拒绝、只读连接保活。
- 评估状态机2项、伪终端同步/异步全流程2项、小型PI0.5/DM0.5 checkpoint推理2项通过，共21项。
- 真实数据episode 0只读回放检查通过：1035帧、30Hz、19D动作；DM0.5离线入口输出有限19D动作。
- 服务模块已检查Python3.10语法。测试运行在工作站模拟环境；原生SDK加载、真实相机/电机、控制周期和力控未在H1验证。
