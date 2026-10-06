from multiprocessing import shared_memory, Value, Array, Lock
from typing import Any
import numpy as np
import argparse
import threading
import torch
import time
from unitree_lerobot.eval_robot.image_server.image_client import ImageClient, TeleimagerImageClient
from unitree_lerobot.eval_robot.robot_control.robot_arm_astribot import (
    AstribotS1ArmController,
    AstribotS1ArmIK,
    AstribotS1GripperController,
)

from unitree_lerobot.eval_robot.utils.episode_writer import EpisodeWriter

try:
    from unitree_lerobot.eval_robot.robot_control.robot_arm import (
        G1_23_ArmController,
        G1_29_ArmController,
    )
    from unitree_lerobot.eval_robot.robot_control.robot_arm_ik import G1_23_ArmIK, G1_29_ArmIK
    from unitree_lerobot.eval_robot.robot_control.robot_hand_brainco import Brainco_Controller
    from unitree_lerobot.eval_robot.robot_control.robot_hand_inspire import Inspire_Controller
    from unitree_lerobot.eval_robot.robot_control.robot_hand_unitree import (
        Dex1_1_Gripper_Controller,
        Dex3_1_Controller,
    )
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

    _UNITREE_AVAILABLE = True
    _UNITREE_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    G1_23_ArmController = None
    G1_29_ArmController = None
    G1_23_ArmIK = None
    G1_29_ArmIK = None
    Brainco_Controller = None
    Inspire_Controller = None
    Dex1_1_Gripper_Controller = None
    Dex3_1_Controller = None
    MotionSwitcherClient = None
    ChannelFactoryInitialize = None
    ChannelPublisher = None
    String_ = None
    _UNITREE_AVAILABLE = False
    _UNITREE_IMPORT_ERROR = exc

import logging_mp
from unitree_lerobot.eval_robot.utils.logging_mp_compat import patch_logging_mp

patch_logging_mp(logging_mp)
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

# Configuration for robot arms
ARM_CONFIG = {}
if _UNITREE_AVAILABLE:
    ARM_CONFIG.update(
        {
            "G1_29": {"controller": G1_29_ArmController, "ik_solver": G1_29_ArmIK, "dof": 14},
            "G1_23": {"controller": G1_23_ArmController, "ik_solver": G1_23_ArmIK, "dof": 14},
        }
    )
ARM_CONFIG["Astribot_S1"] = {
    "controller": AstribotS1ArmController,
    "ik_solver": AstribotS1ArmIK,
    "dof": 25,
    "skip_unitree_debug": True,
}

# Configuration for end-effectors
EE_CONFIG: dict[str, dict[str, Any]] = {}
if _UNITREE_AVAILABLE:
    EE_CONFIG.update(
        {
            "dex3": {
                "controller": Dex3_1_Controller,
                "dof": 7,
                "shared_mem_type": "Array",
                "shared_mem_size": 7,
                # "out_len": 14,
            },
            "dex1": {
                "controller": Dex1_1_Gripper_Controller,
                "dof": 1,
                "shared_mem_type": "Value",
                # "out_len": 2,
            },
            "inspire1": {
                "controller": Inspire_Controller,
                "dof": 6,
                "shared_mem_type": "Array",
                "shared_mem_size": 6,
                # "out_len": 12,
            },
            "brainco": {
                "controller": Brainco_Controller,
                "dof": 6,
                "shared_mem_type": "Array",
                "shared_mem_size": 6,
                # "out_len": 12,
            },
        }
    )
EE_CONFIG["astribot"] = {
    "controller": AstribotS1GripperController,
    "dof": 1,
    "shared_mem_type": "Value",
    # Astribot S1 SDK uses 0=open, 100=closed for each gripper.
    "out_len": 2,
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


def _dds_network_interface(args: argparse.Namespace) -> str | None:
    network_interface = str(getattr(args, "dds_network_interface", "") or "").strip()
    return network_interface or None


def enter_debug_mode_if_needed(args: argparse.Namespace) -> None:
    if getattr(args, "sim", False) or getattr(args, "motion", False):
        return
    if not _UNITREE_AVAILABLE:
        raise ImportError(
            "Unitree SDK is unavailable, so Unitree debug mode cannot be entered. "
            "Use --arm=Astribot_S1 --ee='' for the Astribot SDK path."
        ) from _UNITREE_IMPORT_ERROR

    logger_mp.info("Entering Unitree debug mode before low-level arm control...")
    network_interface = _dds_network_interface(args)
    if network_interface:
        logger_mp.info(f"Initializing Unitree DDS on network interface: {network_interface}")
        ChannelFactoryInitialize(0, network_interface)
    else:
        ChannelFactoryInitialize(0)
    motion_switcher = MotionSwitcherClient()
    motion_switcher.SetTimeout(1.0)
    motion_switcher.Init()

    status, result = motion_switcher.CheckMode()
    logger_mp.info(f"[MotionSwitcher] CheckMode response: status={status}, result={result}")
    if not isinstance(result, dict):
        raise RuntimeError(f"MotionSwitcher CheckMode returned unexpected result: {result!r}")

    while result.get("name"):
        release_result = motion_switcher.ReleaseMode()
        logger_mp.info(f"[MotionSwitcher] ReleaseMode response: {release_result}")
        time.sleep(1.0)
        status, result = motion_switcher.CheckMode()
        logger_mp.info(f"[MotionSwitcher] CheckMode response: status={status}, result={result}")
        if not isinstance(result, dict):
            raise RuntimeError(f"MotionSwitcher CheckMode returned unexpected result after ReleaseMode: {result!r}")

    if status != 0:
        raise RuntimeError(f"Failed to enter debug mode: status={status}, result={result}")

    logger_mp.info("Unitree debug mode is ready.")


def setup_image_client(args: argparse.Namespace) -> dict[str, Any]:
    """Initializes and starts the image client and shared memory."""
    is_sim = getattr(args, "sim", False)
    image_transport = str(getattr(args, "image_transport", "legacy_combined" if is_sim else "teleimager")).lower()
    image_server_address = str(getattr(args, "image_server_address", "127.0.0.1" if is_sim else "192.168.123.164"))
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
        raise ValueError(
            "Unsupported camera feature in --camera_features. Use head/left/right RGB camera features instead."
        )

    # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
    if is_sim:
        img_config = {
            "fps": 30,
            "head_camera_type": "opencv",
            "head_camera_image_shape": head_image_shape,  # Head camera resolution
            "head_camera_id_numbers": [0],
        }
    else:
        img_config = {
            "fps": 30,
            "head_camera_type": "opencv",
            "head_camera_image_shape": head_image_shape,  # Head camera resolution
            "head_camera_id_numbers": [0],
        }
    if use_wrist_cameras:
        img_config.update(
            {
                "wrist_camera_type": "opencv",
                "wrist_camera_image_shape": wrist_image_shape,  # Wrist camera resolution
                "wrist_camera_id_numbers": [2, 4],
            }
        )

    ASPECT_RATIO_THRESHOLD = 2.0  # If the aspect ratio exceeds this value, it is considered binocular
    if len(img_config["head_camera_id_numbers"]) > 1 or (
        img_config["head_camera_image_shape"][1] / img_config["head_camera_image_shape"][0] > ASPECT_RATIO_THRESHOLD
    ):
        BINOCULAR = True
    else:
        BINOCULAR = False
    if "wrist_camera_type" in img_config:
        WRIST = True
    else:
        WRIST = False

    if BINOCULAR and not (
        img_config["head_camera_image_shape"][1] / img_config["head_camera_image_shape"][0] > ASPECT_RATIO_THRESHOLD
    ):
        tv_img_shape = (img_config["head_camera_image_shape"][0], img_config["head_camera_image_shape"][1] * 2, 3)
    else:
        tv_img_shape = (img_config["head_camera_image_shape"][0], img_config["head_camera_image_shape"][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)

    tv_img_array.fill(0)

    thermal_img_shape = None
    thermal_img_shm = None
    thermal_img_array = None

    if WRIST and is_sim:
        wrist_img_shape = (img_config["wrist_camera_image_shape"][0], img_config["wrist_camera_image_shape"][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=wrist_img_shm.buf)
        wrist_img_array.fill(0)
        img_client = ImageClient(
            tv_img_shape=tv_img_shape,
            tv_img_shm_name=tv_img_shm.name,
            wrist_img_shape=wrist_img_shape,
            wrist_img_shm_name=wrist_img_shm.name,
            server_address=image_server_address,
            port=int(getattr(args, "legacy_image_port", 5555)),
        )
    elif WRIST and not is_sim:
        wrist_img_shape = (img_config["wrist_camera_image_shape"][0], img_config["wrist_camera_image_shape"][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=wrist_img_shm.buf)
        wrist_img_array.fill(0)
        if image_transport == "legacy_combined":
            img_client = ImageClient(
                tv_img_shape=tv_img_shape,
                tv_img_shm_name=tv_img_shm.name,
                wrist_img_shape=wrist_img_shape,
                wrist_img_shm_name=wrist_img_shm.name,
                server_address=image_server_address,
                port=int(getattr(args, "legacy_image_port", 5555)),
            )
        elif image_transport in {"teleimager", "official_teleimager"}:
            img_client = TeleimagerImageClient(
                tv_img_shape=tv_img_shape,
                tv_img_shm_name=tv_img_shm.name,
                wrist_img_shape=wrist_img_shape,
                wrist_img_shm_name=wrist_img_shm.name,
                thermal_img_shape=thermal_img_shape,
                thermal_img_shm_name=thermal_img_shm.name if thermal_img_shm is not None else None,
                server_address=image_server_address,
                head_port=int(getattr(args, "head_camera_zmq_port", 55555)),
                left_wrist_port=int(getattr(args, "left_wrist_camera_zmq_port", 55556)),
                right_wrist_port=int(getattr(args, "right_wrist_camera_zmq_port", 55557)),
            )
        else:
            raise ValueError(
                f"Unknown image_transport '{image_transport}'. Use 'teleimager' or 'legacy_combined'."
            )
    else:
        wrist_img_shape = None
        wrist_img_shm = None
        wrist_img_array = None
        if image_transport in {"teleimager", "official_teleimager"} and not is_sim:
            img_client = TeleimagerImageClient(
                tv_img_shape=tv_img_shape,
                tv_img_shm_name=tv_img_shm.name,
                thermal_img_shape=thermal_img_shape,
                thermal_img_shm_name=thermal_img_shm.name if thermal_img_shm is not None else None,
                server_address=image_server_address,
                head_port=int(getattr(args, "head_camera_zmq_port", 55555)),
            )
        else:
            img_client = ImageClient(
                tv_img_shape=tv_img_shape,
                tv_img_shm_name=tv_img_shm.name,
                server_address=image_server_address,
                port=int(getattr(args, "legacy_image_port", 5555)),
            )

    has_wrist_cam = "wrist_camera_type" in img_config
    has_thermal_cam = False

    if image_transport in {"teleimager", "official_teleimager"} and not is_sim:
        logger_mp.info(
            "Image input transport: teleimager ZMQ at "
            f"{image_server_address}:{int(getattr(args, 'head_camera_zmq_port', 55555))} "
            f"(head)"
            + (
                f", {int(getattr(args, 'left_wrist_camera_zmq_port', 55556))}/"
                f"{int(getattr(args, 'right_wrist_camera_zmq_port', 55557))} (wrists)"
                if has_wrist_cam
                else ", wrists disabled"
            )
            + ". "
            "WebRTC ports such as 60001 are not used by policy inference."
        )
    else:
        logger_mp.info(
            "Image input transport: legacy combined ZMQ at "
            f"{image_server_address}:{int(getattr(args, 'legacy_image_port', 5555))}."
        )
    logger_mp.info(
        f"Image buffers: head={tv_img_shape}, wrist={wrist_img_shape if has_wrist_cam else None}, "
        f"is_binocular={BINOCULAR}."
    )

    image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    return {
        "tv_img_array": tv_img_array,
        "wrist_img_array": wrist_img_array,
        "thermal_img_array": thermal_img_array,
        "tv_img_shape": tv_img_shape,
        "wrist_img_shape": wrist_img_shape,
        "thermal_img_shape": thermal_img_shape,
        "is_binocular": BINOCULAR,
        "has_wrist_cam": has_wrist_cam,
        "has_thermal_cam": has_thermal_cam,
        "shm_resources": [tv_img_shm, wrist_img_shm, thermal_img_shm],
        "image_client": img_client,
    }


def _resolve_out_len(spec: dict[str, Any]) -> int:
    return int(spec.get("out_len", 2 * int(spec["dof"])))


def setup_robot_interface(args: argparse.Namespace) -> dict[str, Any]:
    """
    Initializes robot controllers and IK solvers based on configuration.
    """
    # ---------- Arm ----------
    if args.arm not in ARM_CONFIG:
        raise ValueError(f"Unknown arm '{args.arm}'. Available: {list(ARM_CONFIG.keys())}")
    arm_spec = ARM_CONFIG[args.arm]
    if not arm_spec.get("skip_unitree_debug", False):
        enter_debug_mode_if_needed(args)

    arm_ik = arm_spec["ik_solver"]()
    is_sim = getattr(args, "sim", False)
    if is_sim and arm_spec.get("skip_unitree_debug", False):
        raise NotImplementedError("Astribot_S1 eval shim uses astribot_sdk_ros2 and does not support --sim.")
    arm_ctrl = arm_spec["controller"](
        motion_mode=args.motion,
        simulation_mode=is_sim,
        arm_velocity_limit=getattr(args, "arm_low_level_velocity_limit", None),
    )

    # ---------- End Effector (optional) ----------
    ee_ctrl, ee_shared_mem, ee_dof = None, {}, 0

    if ee_key := getattr(args, "ee", "").lower():
        if ee_key not in EE_CONFIG:
            raise ValueError(f"Unknown end-effector '{args.ee}'. Available: {list(EE_CONFIG.keys())}")

        spec = EE_CONFIG[ee_key]
        mem_type, out_len, ee_dof = spec["shared_mem_type"].lower(), _resolve_out_len(spec), spec["dof"]
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

    # ---------- Simulation helpers (optional) ----------
    episode_writer = None
    if is_sim:
        reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
        reset_pose_publisher.Init()
        from unitree_lerobot.eval_robot.utils.sim_state_topic import (
            start_sim_state_subscribe,
            start_sim_reward_subscribe,
        )

        sim_state_subscriber = start_sim_state_subscribe()
        sim_reward_subscriber = start_sim_reward_subscribe()
        if getattr(args, "save_data", False) and getattr(args, "task_dir", None):
            episode_writer = EpisodeWriter(args.task_dir, frequency=30, image_size=[640, 480])
        return {
            "arm_ctrl": arm_ctrl,
            "arm_ik": arm_ik,
            "ee_ctrl": ee_ctrl,
            "ee_shared_mem": ee_shared_mem,
            "arm_dof": int(arm_spec["dof"]),
            "ee_dof": ee_dof,
            "sim_state_subscriber": sim_state_subscriber,
            "sim_reward_subscriber": sim_reward_subscriber,
            "episode_writer": episode_writer,
            "reset_pose_publisher": reset_pose_publisher,
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
    current_thermal_image = thermal_img_array.copy() if has_thermal_cam and thermal_img_array is not None else None

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


def publish_reset_category(category: int, publisher):  # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")
