"""Resolve hardware limits via read-only SDK queries, without enabling/moving arms."""

import copy
import json
import math
import time
from pathlib import Path

from piper_control import ArmConfig


def load_hardware(path):
    path = Path(path).expanduser()
    if not path.exists() and not path.is_absolute() and path.parent == Path("."):
        path = Path(__file__).resolve().parent / path
    if not path.is_file():
        raise ValueError("Hardware configuration missing: %s; copy the supplied hardware.json with the service" % path)
    hardware = json.loads(path.read_text())
    names = [hardware[side]["can"] for side in ("left", "right")]
    if any(not isinstance(name, str) or not name for name in names) or names[0] == names[1]:
        raise ValueError("left/right require distinct nonempty CAN interface names")
    return path.resolve(), hardware


def automatic(value):
    return value is None or (isinstance(value, str) and value == "auto")


def wait_reply(read, requested, valid, description, timeout):
    deadline = time.monotonic() + timeout
    while True:
        reply = copy.deepcopy(read())
        if reply.time_stamp >= requested and valid(reply):
            return reply
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "No fresh hardware reply for %s; check CAN connection/firmware. "
                "SDK default zeros are not hardware limits." % description
            )
        time.sleep(0.01)


def resolve_hardware(hardware, sdk_factory=None, timeout=2):
    result = copy.deepcopy(hardware)
    for side in ("left", "right"):
        config = result[side]
        joints_auto = any(automatic(config.get(key)) for key in ("joint_min", "joint_max"))
        grip_auto = automatic(config.get("gripper_max_m"))
        if joints_auto or grip_auto:
            if sdk_factory is None:
                from piper_sdk import C_PiperInterface_V2

                sdk_factory = C_PiperInterface_V2
            arm = sdk_factory(can_name=config["can"])
            try:
                arm.ConnectPort(piper_init=False)
                if joints_auto:
                    lo, hi = [], []
                    for motor in range(1, 7):
                        requested = time.time()
                        arm.SearchMotorMaxAngleSpdAccLimit(motor_num=motor, search_content=0x01)
                        reply = wait_reply(
                            arm.GetCurrentMotorAngleLimitMaxVel,
                            requested,
                            lambda r: r.current_motor_angle_limit_max_vel.motor_num == motor,
                            "%s joint %d limits" % (config["can"], motor),
                            timeout,
                        )
                        limits = reply.current_motor_angle_limit_max_vel
                        if not limits.min_angle_limit < limits.max_angle_limit:
                            raise RuntimeError(
                                "Invalid/unconfigured hardware joint limits on %s motor %d" % (side, motor)
                            )
                        lo.append(math.radians(limits.min_angle_limit / 10))
                        hi.append(math.radians(limits.max_angle_limit / 10))
                    for key, values in (("joint_min", lo), ("joint_max", hi)):
                        if automatic(config.get(key)):
                            config[key] = values
                if grip_auto:
                    requested = time.time()
                    # Query only: all setting/effective fields retain SDK's disabled defaults.
                    arm.ArmParamEnquiryAndConfig(param_enquiry=0x04)
                    reply = wait_reply(
                        arm.GetGripperTeachingPendantParamFeedback,
                        requested,
                        lambda r: True,
                        "%s gripper range" % config["can"],
                        timeout,
                    )
                    mm = reply.arm_gripper_teaching_param_feedback.max_range_config
                    if mm not in (70, 100):
                        raise RuntimeError(
                            "Hardware reported unsupported/unconfigured gripper range %s on %s" % (mm, side)
                        )
                    config["gripper_max_m"] = mm / 1000
            finally:
                arm.DisconnectPort()
        ArmConfig(**config)  # validate explicit and queried values before either arm is enabled
    return result


def load_resolved_hardware(path):
    path, hardware = load_hardware(path)
    resolved = resolve_hardware(hardware)
    # Keep the input's auto values; query anew next launch. This file is an inspection record.
    destination = path.with_name(path.stem + ".resolved.json")
    destination.write_text(json.dumps(resolved, indent=2) + "\n")
    print("Hardware limits resolved; inspection record: %s" % destination, flush=True)
    return resolved
