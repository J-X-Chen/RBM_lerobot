#!/usr/bin/env bash
SERVICE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/noetic/setup.bash
source "$SERVICE_ROOT/ros_ws/devel/setup.bash" --extend
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate aloha || return 1
export PYTHONPATH="$SERVICE_ROOT/vendor/sdk:$PYTHONPATH"
export LD_LIBRARY_PATH="$SERVICE_ROOT/ros_ws/src/ros_astra_camera/include/openni2_redist/x64:${LD_LIBRARY_PATH:-}"
