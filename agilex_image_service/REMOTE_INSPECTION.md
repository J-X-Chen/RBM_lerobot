# 192.168.0.102 只读核查（2026-10-29）

- 用户：jchen572；目录：/home/jchen572/code_file/agilex_image_service。
- aloha 环境 Python 3.8.19；随服务 SDK 0.6.1。
- can_left：1Mbit/s，USB 1-13:1.0；can_right：1Mbit/s，USB 1-12:1.0。
- 初次检查两路 ERROR-PASSIVE，各三秒被动监听无有效帧；用户上电后首次复查仍无帧。用户完成接口down/up后仍立即进入ERROR-PASSIVE，仍无有效反馈。
- 未使能、未发送运动指令、未启动控制服务、未更改接口参数。
- 远端仍为原 baseline，尚未上传/部署本地新版服务。

## 采集与夹爪

远端 agilex_data_collection/collect_data.py 的 observation_and_action 直接拼接
puppet JointState.position 为状态、master JointState.position 为动作。
单臂采集时另一侧动作由 home_pose.json 固定，观测保留实测。
ROS piper_start_ms_node.py 把夹爪反馈/控制量除以1000000，没有额外比例转换。
home_pose.json 中固定夹爪为0.09975，原左右读数为0.1/0.0995。
旧 joint_callback 则把下发夹爪值截断到80000（0.08）；SDK默认参数表写0.07。
这些互不一致的常量不能作为实际夹爪硬件行程的证明，也不能据此随意给训练值乘0.8。
真实最大开口和关节限位仍待确认。

## 历史报告

logs/20261029_122702/report.json 和 logs/20261029_124229/report.json
都记录三路480×640图像，但 state.ok=false（反馈过期/缺失）。
这些报告不能作为机械臂链路已通的证据，也不代表当前相机已验证。

## 错误帧复查

开启原始套接字 CAN_RAW_ERR_FILTER，仅接收不发送：两秒内左路10044、右路9956条错误帧，
CAN ID=0x20000024，payload=0030000000008000。
按Linux can/error.h：CAN_ERR_FLAG + CAN_ERR_CRTL + CAN_ERR_ACK，控制器错误被动状态且发送未获ACK。
不能仅凭该错误区分供电、接线、波特率或收发器故障，需要现场核查。
用户要求所有代码变更仅在本地 agilex_image_service 完成，由用户自行替换；未上传源码、未覆盖远端。

## 用户补充

用户按手册安全要求主动断开主从航空插头。此前“从臂”表述指数据采集中的puppet执行臂，
不表示推理需要连接遥操作示教主臂。新版服务直接控制配置CAN上的执行臂。
当前未确认断开后电脑到执行臂的独立通信拓扑；不能据无ACK推定应重新插回主从联动插头。

## 硬件参数主动只读查询

用户要求读取硬件后，使用远端SDK ConnectPort(piper_init=False)，分别调用
SearchAllMotorMaxAngleSpd和ArmParamEnquiryAndConfig(param_enquiry=0x04)。
两路状态、关节限制及夹爪参数反馈的time_stamp均为0，未收到应答；零值是SDK初始化内容。
未调用使能、运动、参数设置指令。无法据此确认实际限位。
本地已提供hardware.json和自动参数查询；用户替换后在CAN恢复时自动读取，无需手工抄参数。
