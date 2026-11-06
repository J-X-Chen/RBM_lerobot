# 可迁移部署（Ubuntu 20.04 x86_64 / ROS Noetic）

整个 agilex_image_service 目录为代码交付单元。不依赖同级 scripts、cobot_magic ；运行使用自行配置的 aloha Conda 环境。
系统 ROS、Linux CAN、USB 驱动和 apt 库仍须安装；不是包含操作系统的离线镜像。
Ubuntu 22.04 / ROS2 不可直接按此流程运行，需要适配或 Ubuntu20.04 容器及硬件透传。

## 新机器
先安装 Ubuntu20.04 对应的 ROS Noetic（/opt/ros/noetic），然后安装依赖：

```bash
sudo apt update
sudo apt install build-essential cmake pkg-config python3-dev python3-pip python3-opencv python3-numpy python3-yaml python3-rospkg python3-catkin-pkg python3-empy libuvc-dev libusb-1.0-0-dev libudev-dev libgoogle-glog-dev libeigen3-dev libopencv-dev libboost-all-dev ethtool iproute2 can-utils ros-noetic-cv-bridge ros-noetic-image-geometry ros-noetic-image-transport ros-noetic-camera-info-manager ros-noetic-rgbd-launch ros-noetic-tf ros-noetic-tf2-ros ros-noetic-dynamic-reconfigure ros-noetic-actionlib ros-noetic-control-msgs ros-noetic-trajectory-msgs
cd /你的路径/agilex_image_service
sudo cp vendor/camera_ws/src/ros_astra_camera/56-orbbec-usb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo usermod -aG dialout,plugdev,video "$USER"
# 重新登录使用户组生效，再执行：
bash deploy.sh
```

deploy.sh 使用系统 Python3.8 编译 ROS 工作空间，运行时激活 Conda aloha，并使用随包 SDK 0.6.1。脚本不会创建或安装 Python 环境。请先自行准备 Python3.8 的 aloha 环境，包含 numpy、opencv、pyzmq、python-can、rospkg、catkin_pkg，并确保能导入 ROS cv_bridge。默认 Conda 位于 ~/miniconda3；其他位置可通过 CONDA_BASE 指定。
换机器或改变目录位置后必须重新构建；传输包不包含 .venv 和构建产物。
Python 环境由用户自行配置。vendor/bootstrap 仅为历史引导工具，当前流程不使用。

## 硬件配置
原机相机序列号、USB-CAN 物理端口映射保留。新机器须检查：
- vendor/camera_ws/src/ros_astra_camera/launch/multi_camera.launch：三台相机序列号及左右对应。
- vendor/Piper_ros_private-ros-noetic/can_config.sh：USB 端口、接口名及波特率。
- supervisor.py 的 CAN 字典：应与 can_config.sh 的映射一致。
修改相机源码配置后重新运行 deploy.sh。相机固件和SDK版本兼容性需在新机验收。

## 与 PI0.5 / DM0.5 评估器联调

机器人端负责相机、CAN 和限速，推理机只需要训练环境以及网络。
完整两端命令见项目 `order.txt` 第 21/22 节。

```bash
cd /你的路径/agilex_image_service
# 包内已提供 hardware.json：can_left/can_right，启动时自动查询控制器限位
bash start_all.sh --hardware-config=./hardware.json
# 关闭上一个服务后，允许实机执行（仍不会自动使能）：
bash start_all.sh --hardware-config=./hardware.json --execute-control
```

配置格式：left/right 各包含 can、joint_min/joint_max（6个弧度值）、gripper_max_m（米）、
gripper_scale、gripper_offset_m。物理夹爪开口=数据值×scale+offset，反馈反向转换。
hardware.json 中的 auto 会在启动时查询六个关节限位和夹爪行程；收到有效回复后保存hardware.resolved.json供核查，下次启动重新查询。查询不使能、不改变模式、不下发动作；无有效回复会明确报错，不采用SDK零值或默认范围。需要手动固定限制时仍可填写数值。不要从训练数据的极值推测硬件范围；原 baseline 的
0.08m 限幅与训练数据约0.10的值需要核实标定。校准后不再强制全局0.08m限幅。

该模式直接使用随包 Piper V2 SDK，不启动 ROS 手臂节点，不需要主臂 can0。
左右 CAN 必须已 UP、1Mbit/s。不会自动按原机USB物理端口重新映射CAN。
ROS 仅提供三台相机的图像。不要运行其他CAN控制程序或遥操作程序。
不带 hardware-config 的旧只读采集模式及 `--camera-only` 仍保留，但不能用于新的模型执行。
如果 ROS/相机已由自己启动，可用 `bash start.sh` 加同样的硬件/执行参数，仅启动服务。

默认限制可用以下参数调整：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--joint-speed` | 0.3 | 下发关节目标变化率上限，rad/s |
| `--gripper-speed` | 0.02 | 物理夹爪开口目标变化率上限，m/s |
| `--timeout` | 5 | 使能后无动作超时，秒 |
| `--feedback-timeout` | 1 | SDK反馈时效，秒 |
| `--camera-timeout` | 2 | 原始相机回调时效，秒 |
| `--tracking-limit` | 0.35 | 反馈与上一条目标最大关节偏差，rad |

限速基于当前姿态和上一条命令，不拒绝硬件范围内的远处目标，也不会在推理停顿后积累大步长。
关节不超过硬件范围±0.05rad、夹爪不超过范围±0.002m的微小预测误差允许限幅；明显越界仍拒绝。
预热和等待用户输入期间不使能、不启动动作 watchdog，TCP 连接不会因为闲置两秒而断开。
开启执行后，断连、反馈故障、相机中断、无新动作超时触发 SDK 停止；普通状态查询不会续动作超时。
正常退出由新版评估器先限速回位再停止；检查硬件并重启服务后再开始下一轮。服务端自身不自动回零、闭爪或解除急停。
软件停止依赖本机进程和CAN仍正常，不能代替实体急停，也不提供避碰。

## 网络协议

- 默认 `55555/55556/55557`：head/left/right，即 ROS camera_f/camera_l/camera_r；每端口独立 ZMQ PUB，单消息完整 JPEG。
- 模型端解码 BGR→RGB CHW；不交换左右、不翻转画面。相机安装方向应与采集一致。
- `6200`：TCP，10字节ASCII十进制长度+UTF-8 JSON，响应同格式；最多65536字节。单客户端，无认证，仅可信网络使用。
- `get_state:true`：14维 left6+gripper+right6+gripper，返回 protocol=2、calibrated、执行/故障状态、camera_age、command_timeout。
- `enable:true`：仅服务已传 execute-control 时允许；可附 active_arm、joint_speed、gripper_speed。客户端只能减小服务端速度上限。
- `action:[14数], dt:秒`：仅已使能时实际执行，返回限速后 target。服务端周期不超过请求周期、实际收包间隔和0.1秒三者最小值。
- `stop:true` 或 `hold_current:true`：已使能时发送 SDK 急停，不是清空队列后继续运行。

相机新鲜度同时检查接收端和发送端原始回调时间，不要求两机时钟同步；协议不保证三相机曝光同步。
日志在 logs/；客户端保存动作 JSONL 和运行参数。SDK 控制代码 `piper_control.py` 与项目评估器共用。

## 验证边界

已使用模拟 SDK 测试真实本机 TCP/ZMQ、RGB/左右顺序、只读、显式使能、限速、断连、超时和异常停止。
并未执行实机运动；真实相机、CAN、夹爪单位和现场速度需要联调。
部署脚本的 test_protocol.py/test_control.py 不访问硬件。

## CAN 没有反馈时

`python passive_can.py` 仅监听，不发送CAN指令。它分别统计有效帧和错误帧；
旧脚本默认忽略错误帧，因此只看到空字典不能说明总线上完全没有流量。
`no_ack` 表示发送未得到应答，`ERROR-PASSIVE` 表示CAN控制器进入错误被动状态。
检查从臂供电、CAN端口/接线和波特率；不要通过增大模型超时来掩盖通信故障。
确认没有运行控制程序后，如需重置网卡，可在机器人终端整行执行：

```bash
sudo ip link set can_left down && sudo ip link set can_left up && sudo ip link set can_right down && sudo ip link set can_right up
```

这只重置网卡，不发送运动命令。若立即再次ERROR-PASSIVE，需继续排查硬件通信。
2026-10-29只读核查结果见REMOTE_INSPECTION.md；该次未验证实机运动。

## 推理执行与主从命名

推理不需要保留遥操作主从联动。服务直接通过hardware.json指定的左右CAN接口，
读取所连机械臂反馈并发送JointCtrl/GripperCtrl，不向另一只示教臂转发动作。
采集中的action来自/master话题，并不表示部署时必须驱动示教主臂：旧ROS节点也会把
/master话题作为执行臂的目标输入。部署时应连接实际完成任务、与训练state/相机对应的两只执行臂。
主从航空插头按设备手册断开后，仍需保证电脑到执行臂的独立CAN连接和反馈；
不能仅根据主/从叫法或CAN接口名判断物理接线，也不要为消除报错而恢复遥操作联动。

## 可调端口与共用服务

start_all.sh 和 start.sh 都支持 `--head-port`、`--left-wrist-port`、`--right-wrist-port`、
`--control-port`；相机发布及启动健康检查会使用同一组参数。四个端口须不同且在1～65535范围。
order.txt 第21.0节显式设为6100/6101/6102/6200，客户端四条命令同步指定这些端口。
第21.0节统一带execute-control，只表示允许执行；客户端只读时不会使能或下发动作。
第22节复用同一服务，实际执行仍需客户端execute及输入s。第22.0节不再单独启动服务。

配置文件随服务一并分发，不需要从示例手工复制。相对路径找不到时，也会查找服务目录下同名配置。
当前gripper_scale=1、offset=0沿用采集代码的SDK单位转换；不会把示教器行程比例再乘一遍。
自动查询得到的是控制器配置的限位/最大行程，不能证明实体机械标定正确，也不提供碰撞保护。
旧固件若不支持0x47E夹爪参数回复，需提供已确认的数值限制；程序不会猜测。

启动时对每路CAN被动监听1秒，要求状态0x2A1、关节0x2A5/6/7、夹爪0x2A8均有重复反馈且无采样错误帧。
ERROR-WARNING/ERROR-PASSIVE标签单独不再阻止启动；BUS-OFF/STOPPED、当前错误帧或缺失反馈仍拒绝。
这只是启动通信检查，后续真实限位查询、SDK状态故障及反馈时效检查仍保留。

只读连接保留真实反馈，不把反馈当目标拒绝或裁剪。使能时允许最多0.05rad的关节边界偏差，
仍保留硬件状态故障检查；该容差不代表已确认零位标定正确。执行目标裁到控制器名义范围，
从真实起始姿态限速推进，中间指令不得进一步越界；不在使能时把起始姿态直接跳到边界。
单臂模式另一臂保持启动反馈姿态。更大的起始偏差仍拒绝使能，需要检查姿态/标定。

新版评估器的q/第一次Ctrl+C正常退出会通过限速动作返回首次使能前关节姿态，夹爪保持当前开口；
服务端仍逐步验证反馈、时效、目标和速度。stop请求、异常和断连依旧立即停止，不自行启动回位。
get_state增加capabilities，enable返回启动anchor，prepare_target只验证/裁剪目标，不发送动作。
控制权限仅在显式enable后生效。部署时需同时更新评估端与发送端。
