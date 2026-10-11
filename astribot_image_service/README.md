# Astribot S1 Image Service

This service wraps Astribot S1 SDK camera callbacks into teleimager-style ZMQ PUB streams.
It lets `astribot_lerobot/eval_robot/eval_s1.py` keep reading camera frames from
`55555`, `55556`, `55557`, matching the Unitree-side image transport shape.

It can also run the S1-side control sender: a TCP action endpoint updates the latest
25D target, and a local 250 Hz SDK loop repeatedly sends that target with
`set_joints_position(..., control_way="filter")`. This keeps the timing-sensitive SDK
command stream on the robot host.

Run from the repository root with the currently activated Python environment.
On S1/Orin, `--sdk-root` must point to the native aarch64 Astribot SDK, not a
workstation/x86 copy. The current S1 environment must use `numpy<2` because the
SDK native extensions were compiled against NumPy 1.x, and it also needs
`filterpy`, `transforms3d`, and ROS's `tf_transformations` package:

```bash
sudo apt install -y ros-humble-tf-transformations
python -m pip uninstall -y meta
python -m pip install "numpy==1.24.0" filterpy transforms3d tabulate h5py opencv-python==4.10.0.82
```

```bash
export ASTRIBOT_SDK_ROOT=${ASTRIBOT_SDK_ROOT:-./astribot_sdk_aarch64}
export ASTRIBOT_LOG=1
source /opt/ros/humble/setup.bash
source "$ASTRIBOT_SDK_ROOT/env.sh"

python -m astribot_image_service.server \
  --sdk-root "$ASTRIBOT_SDK_ROOT" \
  --bind-host 0.0.0.0
```

The service needs `pyzmq`; installing this repository installs it as a project dependency.

To run image publishing plus the S1-side control endpoint on the robot:

```bash
export ASTRIBOT_SDK_ROOT=${ASTRIBOT_SDK_ROOT:-./astribot_sdk_aarch64}
export ASTRIBOT_LOG=1
source /opt/ros/humble/setup.bash
source "$ASTRIBOT_SDK_ROOT/env.sh"

python -m astribot_image_service.server \
  --sdk-root "$ASTRIBOT_SDK_ROOT" \
  --bind-host 0.0.0.0 \
  --head-port=55555 \
  --left-wrist-port=55556 \
  --right-wrist-port=55557 \
  --high-control-rights \
  --enable-control \
  --execute-control \
  --control-bind-host=0.0.0.0 \
  --control-port=6200 \
  --control-frequency=250 \
  --control-action-frequency=30 \
  --control-way=filter
```

The command above is live for robot motion because it includes `--execute-control`.
Run it only after the workspace is clear and S1 is in terminal control mode.

To preview the streams from another terminal:

```bash
python -m astribot_image_service.client --host 127.0.0.1
```

Use `--host <S1-or-service-host>` when the bridge runs on another machine. The viewer needs
OpenCV; install it with `pip install -e ".[viewer]"` or `pip install opencv-python`.

Default routes:

| S1 SDK camera | ZMQ port | Eval feature |
| --- | ---: | --- |
| `head_rgbd` | `55555` | `observation.images.cam_left_high` / `head` |
| `left_wrist_rgbd` | `55556` | `observation.images.cam_left_wrist` / `left` |
| `right_wrist_rgbd` | `55557` | `observation.images.cam_right_wrist` / `right` |

On the eval side, point the image client at the machine running this service:

```bash
python astribot_lerobot/eval_robot/eval_s1.py \
  ... \
  --image_transport=teleimager \
  --image_server_address=<S1-or-service-host> \
  --head_camera_zmq_port=55555 \
  --left_wrist_camera_zmq_port=55556 \
  --right_wrist_camera_zmq_port=55557
```

Extra S1 cameras can be published with repeated `--camera label:camera_name:port` entries, for example:

```bash
python -m astribot_image_service.server \
  --sdk-root "$ASTRIBOT_SDK_ROOT" \
  --camera torso:torso_rgbd:55558
```

## Control endpoint

The control endpoint uses length-prefixed JSON: a 10-byte ASCII length followed by a JSON object.
Supported requests:

```json
{"get_state": true}
{"hold_current": true}
{"move": [25 floats], "duration": 3.0}
{"action": [25 floats]}
{"chunk": [[25 floats], [25 floats]], "replace_buffer": true}
```

The server also accepts 16D arm+gripper actions in the order:
left arm 7, left gripper 1, right arm 7, right gripper 1. They are expanded to 25D using
the current chassis, torso, and head state.
