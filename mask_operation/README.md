# LeRobot v3 mask 工具

在项目根目录运行。数据集路径指向本仓库；SDK、仿真模型及标定产物默认从 `~/code_file/cloak` 读取，可通过 `mask_generation.py --assets-root` 指定。

```bash
conda activate mask_lerobot
python mask_operation/scripts/mask_generation.py \
  --robot s1 \
  --dataset-root lerobotv3_dataset/lerobotv3_red_cup_only \
  --allow-candidate
```

ALOHA 使用 `--robot aloha` 和对应数据集路径。默认输出原数据集旁的 `<数据集名>_cloak`；通过 `--output lerobotv3_dataset/<名称>_mask` 指定其他副本名称。不会覆盖原数据集。

- `scripts/`：数据集导出、独立预览、标定采集/拟合、mask 标注入口。
- `src/cloak_mask/`：数据读取、SDK 状态适配、模型转换、MuJoCo 渲染、实时 mask、视频和 v3 导出实现。
- `tests/`：转换及渲染测试。

旧 `scripts/` 入口与 `src/cloak_mask/__init__.py` 仅保留兼容转发，不包含第二份实现；已有评估代码的 `import cloak_mask` 继续有效。训练策略里的图像填充和 attention 代码属于策略实现，仍在 LeRobot 中。

```bash
PYTHONPATH=mask_operation/src python -m pytest mask_operation/tests -q
```

详细使用说明仍见 `docs/mask_export/README.md`、`docs/aloha_mask/README.md` 和 `docs/astribot_mask/README.md`。

## 绿色填充视频模式

参考多模态 LeRobot v3 视频布局，将现有 `observation.cloak.*_mask` 应用到对应 RGB，输出三个独立 MP4 特征：

- `observation.images.cam_head_mask`
- `observation.images.cam_right_wrist_mask`
- `observation.images.cam_left_wrist_mask`

```bash
python mask_operation/scripts/mask_generation.py \
  --output-mode filled-videos \
  --dataset-root lerobotv3_dataset/aloha_cup_selection_red_cloak \
  --output lerobotv3_dataset/aloha_cup_selection_red_rgb_mask \
  --fill-rgb 0 255 0
```

输入必须是已经导出三路二值 mask 的数据副本；原始数据先运行默认 `masks` 模式。支持现有 S1/ALOHA 的 head/left/right 字段，不意味着已实现参考 Unitree 机器人的 mask 渲染。

输出保留原 RGB、状态、动作及已有 mask；删除副本中的 cam_thermal 字段和对应标准视频目录。三个新字段是**绿色不透明填充 RGB**，不是二值 mask，也不是透明叠加。默认采用 H264 CRF 23 / YUV420 有损压缩（可用 `--video-crf` 调节；`--video-lossless` 才启用体积较大的 RGB 无损编码），统计基于编码前填充图像，沿用对应原 RGB 的 chunk/file 分片，多个 episode 合并在同一视频中，保留原始时间偏移，写入 v3 视频索引、时间范围、全局及分 episode 图像统计。输出使用新目录，失败时 partial 目录不可用于训练。

训练选择这三路普通图像时，无需 Cloak 或在线 RGB 填充。原图与填充图同时保留，注意显式选择模型输入，否则默认特征推断可能将六路全部送入模型。

## 在输出副本内替换 RGB

`--output-mode filled-videos --mask_replace` 不添加 `cam_*_mask` 相机，而是把填充视频写入 **新输出副本** 的原 RGB 路径。沿用输入相机字段名、chunk/file 和 episode 时间偏移，更新视频编码信息和统计。源数据不修改，禁止输出指向源数据或已存在目录。二值 mask、状态、动作仍保留。

```bash
python mask_operation/scripts/mask_generation.py \
  --output-mode filled-videos --mask_replace \
  --dataset-root lerobotv3_dataset/aloha_cup_selection_red_cloak \
  --output lerobotv3_dataset/aloha_cup_selection_red_green_replaced \
  --fill-rgb 0 255 0 --video-crf 23
```

自动识别 `head/left/right`、`cam_head/cam_left_wrist/cam_right_wrist` 和 head 别名 `cam_left_high`。存在多个候选时拒绝猜测；可用 `--camera-keys '{"head":"observation.images.xxx","left":"observation.images.yyy","right":"observation.images.zzz"}'` 显式指定。此参数也适用于新增视频模式。mask 仍来自三个 `observation.cloak.{head,left,right}_mask` 字段。

## 实时评估 RGB 填充

`--robot-mask` 根据当前 measured feedback 渲染整个机器人轮廓，然后在图像归一化/缩放之前不透明填色；默认 RGB 为绿色 `[0,255,0]`。未开启时保持原行为。ALOHA 的同步/异步推理共用入口，S1 的 `eval_s1.py` / `eval_s1_async.py` 均支持。S1 在选择 16D policy state 前使用完整 25D feedback。

当前资源默认来自 `~/code_file/cloak`；可通过 `--mask-assets-root` 指定含 `artifacts`、`mobile_aloha_sim`（ALOHA）或 `astribot_simulation`、`astribot_sdk_ros2`（S1）的目录。代码从本仓库 `mask_operation` 导入。ALOHA 可用 `--mask-config` 覆盖 calibration JSON。S1 使用原导出程序的 head/wrist candidate、像素翻转和手腕下半区域限制。候选标定需要显式 `--allow-candidate-mask`，不代表已校准；相机分辨率或安装位置改变后需要重新标定。模型和 EGL context 只初始化一次，在专属线程中复用并关闭。

在同时具备策略依赖和 mask 依赖（MuJoCo >=3.17、OpenCV、SciPy、PyArrow、PyAV）的环境运行。可设置 `MUJOCO_GL=egl`（默认）或其他已安装的离屏后端。mask 单独环境不一定具有 DM05 策略依赖。

ALOHA 离线评估（项目根目录）：

```bash
python agilex-lerobot/eval_robot/eval_aloha.py \
  --policy.path lerobot/outputs/dm05_aloha_red_cup_14d_mask/checkpoints/030000/pretrained_model \
  --mode dataset --dataset-root lerobotv3_datasets/aloha_cup_selection_red \
  --steps 10 --robot-mask --mask-fill-rgb 0 255 0 --allow-candidate-mask
```

实时预览：将 `--mode dataset` 改成 `--mode live --robot-host <机器人IP>`；默认不发送动作。准备执行时沿用已有 `--execute` 流程。可加 `--async-prefetch`。原始数据集应作为输入，避免给已经填色的数据重复叠加 mask。

S1 在原评估命令后添加（其配置解析器使用 JSON 列表）：

```bash
--robot-mask --mask-fill-rgb='[0,255,0]' --allow-candidate-mask
```

也支持原生配置字段形式 `--robot_mask=true --mask_fill_rgb='[0,255,0]' --allow_candidate_mask=true`。这些参数作用于实时采集图像；S1 的离线数据诊断路径不经过实时 mask。填充色必须与训练保持一致；该功能不修改 checkpoint 或给模型添加 Cloak attention mask。
