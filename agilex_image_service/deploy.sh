#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONPATH CMAKE_PREFIX_PATH ROS_PACKAGE_PATH LD_LIBRARY_PATH
source /opt/ros/noetic/setup.bash
mkdir -p ros_ws/src vendor/sdk
mkdir -p vendor/sdk/piper_sdk
cp -a vendor/piper_sdk_runtime/. vendor/sdk/piper_sdk/
cp -a vendor/camera_ws/src/ros_astra_camera ros_ws/src/
cp -a vendor/Piper_ros_private-ros-noetic/src/piper vendor/Piper_ros_private-ros-noetic/src/piper_msgs ros_ws/src/
cd ros_ws
catkin_make -j2 -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release
cd "$ROOT"
source env.sh
python -c 'import rospy,cv2,zmq,cv_bridge,piper_sdk; print(piper_sdk.__file__)'
python test_protocol.py
python test_control.py
