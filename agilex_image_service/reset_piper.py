"""Explicit, operator-confirmed reset of one Piper arm. May remove motor power."""

import argparse
import sys
import time
from pathlib import Path


def fresh_status(arm, after=0, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = arm.GetArmStatus()
        if msg.time_stamp > after and 0 <= time.time() - msg.time_stamp < 1:
            return msg.arm_status
        time.sleep(0.05)
    raise RuntimeError("未收到新鲜状态反馈，停止操作")


def main():
    parser = argparse.ArgumentParser(description="手动解除单臂软件急停：可能失电下落，须先支撑机械臂")
    parser.add_argument("--can", default="can_left", help="要恢复的单臂 CAN 接口，默认 can_left")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor" / "sdk"))
    from piper_sdk import C_PiperInterface_V2

    arm = C_PiperInterface_V2(can_name=args.can)
    try:
        arm.ConnectPort(piper_init=False)
        status = fresh_status(arm)
        print(status)
        if status.arm_status == 0 and status.err_code == 0:
            print("%s 状态正常，无需恢复；未发送重置指令。" % args.can)
            return
        if status.arm_status != 1 or status.err_code != 0:
            raise RuntimeError("不是单纯急停状态，不自动恢复，请检查上述状态")
        print("警告：EmergencyStop(0x02) 与 ResetPiper 相同，可能立即失电下落。")
        print("先停止评估程序和发送端服务，取下负载，可靠支撑机械臂，并确认物理急停已释放。")
        # A terminal is not seekable: Python's buffered r+ stream cannot be used.
        with open("/dev/tty", "w") as prompt, open("/dev/tty", "r") as keyboard:
            prompt.write("确认 %s 已满足上述条件，输入 RESET：" % args.can)
            prompt.flush()
            answer = keyboard.readline().strip()
        if answer != "RESET":
            print("已取消，未发送恢复指令。")
            return
        # Recheck after the operator's potentially long pause.
        status = fresh_status(arm)
        if status.arm_status != 1 or status.err_code != 0:
            raise RuntimeError("确认期间控制器状态改变，取消恢复")
        sent_at = time.time()
        arm.EmergencyStop(0x02)
        time.sleep(0.5)
        status = fresh_status(arm, after=sent_at)
        print("恢复后的状态：", status)
        if status.arm_status != 0 or status.err_code != 0:
            raise RuntimeError("状态尚未恢复正常，请勿开始推理")
        print("%s 已恢复正常；未发送使能或位置指令。" % args.can)
    finally:
        arm.DisconnectPort()


if __name__ == "__main__":
    main()
