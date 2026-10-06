from multiprocessing import Array, Lock, Value, shared_memory
from typing import Any
import argparse
import threading

import numpy as np
import torch

from astribot_lerobot.eval_robot.image_server.image_client import ImageClient, TeleimagerImageClient
from astribot_lerobot.eval_robot.robot_control.robot_arm_astribot import (
    AstribotS1ArmController,
    AstribotS1ArmIK,
    AstribotS1GripperController,
)

import logging_mp
from astribot_lerobot.eval_robot.utils.logging_mp_compat import patch_logging_mp

patch_logging_mp(logging_mp)
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)


ARM_CONFIG = {
    "Astribot_S1": {
        "controller": AstribotS1ArmController,
        "ik_solver": AstribotS1ArmIK,
        "dof": 25,
    }
}

EE_CONFIG: dict[str, dict[str, Any]] = {
    "astribot": {
        "controller": AstribotS1GripperController,
        "dof": 1,
        "shared_mem_type": "Value",
        "out_len": 2,
    }
}


def _normalize_camera_feature_name(name: str) -> str:
    name = str(name).strip()
    if not name:
        return ""
    if name.startswith("observation.images."):
        return name
    if name.startswith("images."):
        return f"observation.{name}"
    return f"observation.images.{name}"


def _requested_camera_feature_names(args: argparse.Namespace) -> set[str]:
    feature_names: set[str] = set()
    policy_features = getattr(getattr(args, "policy", None), "input_features", None) or {}
    for feature_name in policy_features:
        if str(feature_name).startswith("observation.images."):
            feature_names.add(str(feature_name))

    for attr_name in ("feature_names", "camera_features"):
        for feature_name in getattr(args, attr_name, []) or []:
            normalized = _normalize_camera_feature_name(feature_name)
            if normalized:
                feature_names.add(normalized)
    return feature_names


def _requests_camera_feature(args: argparse.Namespace, feature_name: str) -> bool:
    return _normalize_camera_feature_name(feature_name) in _requested_camera_feature_names(args)


def setup_image_client(args: argparse.Namespace) -> dict[str, Any]:
    """Initializes and starts the image client and shared memory."""
    is_sim = getattr(args, "sim", False)
    if is_sim:
        raise NotImplementedError("Astribot_S1 evaluation uses astribot_sdk_ros2 and does not support --sim.")

    image_transport = str(getattr(args, "image_transport", "teleimager")).lower()
    image_server_address = str(getattr(args, "image_server_address", "192.168.123.164"))
    head_image_shape = [
        int(getattr(args, "head_camera_image_height", 480)),
        int(getattr(args, "head_camera_image_width", 640)),
    ]
    wrist_image_shape = [
        int(getattr(args, "wrist_camera_image_height", 480)),
        int(getattr(args, "wrist_camera_image_width", 640)),
    ]
    use_wrist_cameras = bool(getattr(args, "use_wrist_cameras", False))
    unsupported_camera_feature = "observation.images.cam_" + "ther" + "mal"
    if _requests_camera_feature(args, unsupported_camera_feature):
        raise ValueError("Unsupported camera feature in --camera_features. Use head/left/right RGB camera features.")

    img_config = {
        "fps": 30,
        "head_camera_type": "opencv",
        "head_camera_image_shape": head_image_shape,
        "head_camera_id_numbers": [0],
    }
    if use_wrist_cameras:
        img_config.update(
            {
                "wrist_camera_type": "opencv",
                "wrist_camera_image_shape": wrist_image_shape,
                "wrist_camera_id_numbers": [2, 4],
            }
        )

    aspect_ratio_threshold = 2.0
    is_binocular = len(img_config["head_camera_id_numbers"]) > 1 or (
        img_config["head_camera_image_shape"][1] / img_config["head_camera_image_shape"][0] > aspect_ratio_threshold
    )
    has_wrist_cam = "wrist_camera_type" in img_config

    if is_binocular and not (
        img_config["head_camera_image_shape"][1] / img_config["head_camera_image_shape"][0] > aspect_ratio_threshold
    ):
        tv_img_shape = (img_config["head_camera_image_shape"][0], img_config["head_camera_image_shape"][1] * 2, 3)
    else:
        tv_img_shape = (img_config["head_camera_image_shape"][0], img_config["head_camera_image_shape"][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)
    tv_img_array.fill(0)

    wrist_img_shape = None
    wrist_img_shm = None
    wrist_img_array = None
    if has_wrist_cam:
        wrist_img_shape = (img_config["wrist_camera_image_shape"][0], img_config["wrist_camera_image_shape"][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=wrist_img_shm.buf)
        wrist_img_array.fill(0)

    thermal_img_shape = None
    thermal_img_shm = None
    thermal_img_array = None

    if image_transport == "legacy_combined":
        img_client = ImageClient(
            tv_img_shape=tv_img_shape,
            tv_img_shm_name=tv_img_shm.name,
            wrist_img_shape=wrist_img_shape,
            wrist_img_shm_name=wrist_img_shm.name if wrist_img_shm is not None else None,
            server_address=image_server_address,
            port=int(getattr(args, "legacy_image_port", 5555)),
        )
    elif image_transport in {"teleimager", "official_teleimager"}:
        img_client = TeleimagerImageClient(
            tv_img_shape=tv_img_shape,
            tv_img_shm_name=tv_img_shm.name,
            wrist_img_shape=wrist_img_shape,
            wrist_img_shm_name=wrist_img_shm.name if wrist_img_shm is not None else None,
            thermal_img_shape=thermal_img_shape,
            thermal_img_shm_name=thermal_img_shm.name if thermal_img_shm is not None else None,
            server_address=image_server_address,
            head_port=int(getattr(args, "head_camera_zmq_port", 55555)),
            left_wrist_port=int(getattr(args, "left_wrist_camera_zmq_port", 55556)),
            right_wrist_port=int(getattr(args, "right_wrist_camera_zmq_port", 55557)),
        )
    else:
        raise ValueError(f"Unknown image_transport '{image_transport}'. Use 'teleimager' or 'legacy_combined'.")

    if image_transport in {"teleimager", "official_teleimager"}:
        if has_wrist_cam:
            logger_mp.info(
                "Image input transport: teleimager ZMQ at "
                f"{image_server_address}:{int(getattr(args, 'head_camera_zmq_port', 55555))} "
                f"(head), {int(getattr(args, 'left_wrist_camera_zmq_port', 55556))}/"
                f"{int(getattr(args, 'right_wrist_camera_zmq_port', 55557))} (wrists)."
            )
        else:
            logger_mp.info(
                "Image input transport: teleimager ZMQ at "
                f"{image_server_address}:{int(getattr(args, 'head_camera_zmq_port', 55555))} "
                "(head), wrists disabled."
            )
    else:
        logger_mp.info(
            "Image input transport: legacy combined ZMQ at "
            f"{image_server_address}:{int(getattr(args, 'legacy_image_port', 5555))}."
        )
    logger_mp.info(
        f"Image buffers: head={tv_img_shape}, wrist={wrist_img_shape if has_wrist_cam else None}, "
        f"is_binocular={is_binocular}."
    )

    image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
    image_receive_thread.start()

    return {
        "tv_img_array": tv_img_array,
        "wrist_img_array": wrist_img_array,
        "thermal_img_array": thermal_img_array,
        "tv_img_shape": tv_img_shape,
        "wrist_img_shape": wrist_img_shape,
        "thermal_img_shape": thermal_img_shape,
        "is_binocular": is_binocular,
        "has_wrist_cam": has_wrist_cam,
        "has_thermal_cam": False,
        "shm_resources": [tv_img_shm, wrist_img_shm, thermal_img_shm],
        "image_client": img_client,
    }


def _resolve_out_len(spec: dict[str, Any]) -> int:
    return int(spec.get("out_len", 2 * int(spec["dof"])))


def setup_robot_interface(args: argparse.Namespace) -> dict[str, Any]:
    """Initializes Astribot S1 controllers."""
    if args.arm not in ARM_CONFIG:
        raise ValueError(f"Unknown arm '{args.arm}'. Available: {list(ARM_CONFIG.keys())}")

    arm_spec = ARM_CONFIG[args.arm]
    is_sim = getattr(args, "sim", False)
    if is_sim:
        raise NotImplementedError("Astribot_S1 evaluation uses astribot_sdk_ros2 and does not support --sim.")

    arm_ik = arm_spec["ik_solver"]()
    arm_ctrl = arm_spec["controller"](
        motion_mode=args.motion,
        simulation_mode=is_sim,
        arm_velocity_limit=getattr(args, "arm_low_level_velocity_limit", None),
    )

    ee_ctrl, ee_shared_mem, ee_dof = None, {}, 0
    if ee_key := getattr(args, "ee", "").lower():
        if ee_key not in EE_CONFIG:
            raise ValueError(f"Unknown end-effector '{args.ee}'. Available: {list(EE_CONFIG.keys())}")

        spec = EE_CONFIG[ee_key]
        mem_type = spec["shared_mem_type"].lower()
        out_len = _resolve_out_len(spec)
        ee_dof = spec["dof"]
        data_lock = Lock()

        left_in, right_in = (
            (Array("d", spec["shared_mem_size"], lock=True), Array("d", spec["shared_mem_size"], lock=True))
            if mem_type == "array"
            else (Value("d", 0.0, lock=True), Value("d", 0.0, lock=True))
        )
        state_arr, action_arr = Array("d", out_len, lock=False), Array("d", out_len, lock=False)

        ee_ctrl = spec["controller"](left_in, right_in, data_lock, state_arr, action_arr, simulation_mode=is_sim)
        ee_shared_mem = {
            "left": left_in,
            "right": right_in,
            "state": state_arr,
            "action": action_arr,
            "lock": data_lock,
        }

    return {
        "arm_ctrl": arm_ctrl,
        "arm_ik": arm_ik,
        "ee_ctrl": ee_ctrl,
        "ee_shared_mem": ee_shared_mem,
        "arm_dof": int(arm_spec["dof"]),
        "ee_dof": ee_dof,
    }


def process_images_and_observations(
    tv_img_array,
    wrist_img_array,
    tv_img_shape,
    wrist_img_shape,
    is_binocular,
    has_wrist_cam,
    arm_ctrl,
    thermal_img_array=None,
    thermal_img_shape=None,
    has_thermal_cam=False,
):
    """Processes images and generates observations."""
    current_tv_image = tv_img_array.copy()
    current_wrist_image = wrist_img_array.copy() if has_wrist_cam else None

    left_top_cam = current_tv_image[:, : tv_img_shape[1] // 2] if is_binocular else current_tv_image
    right_top_cam = current_tv_image[:, tv_img_shape[1] // 2 :] if is_binocular else None

    left_wrist_cam = right_wrist_cam = None
    if has_wrist_cam and current_wrist_image is not None:
        left_wrist_cam = current_wrist_image[:, : wrist_img_shape[1] // 2]
        right_wrist_cam = current_wrist_image[:, wrist_img_shape[1] // 2 :]

    observation = {"observation.images.cam_left_high": torch.from_numpy(left_top_cam)}
    observation["observation.images.head"] = observation["observation.images.cam_left_high"]
    if is_binocular:
        observation["observation.images.cam_right_high"] = torch.from_numpy(right_top_cam)
    if has_wrist_cam:
        observation["observation.images.cam_left_wrist"] = torch.from_numpy(left_wrist_cam)
        observation["observation.images.cam_right_wrist"] = torch.from_numpy(right_wrist_cam)
        observation["observation.images.left"] = observation["observation.images.cam_left_wrist"]
        observation["observation.images.right"] = observation["observation.images.cam_right_wrist"]

    current_arm_q = arm_ctrl.get_current_dual_arm_q()
    return observation, current_arm_q


def publish_reset_category(category: int, publisher):
    logger_mp.info(f"reset category requested but simulation reset is not available: {category}")
