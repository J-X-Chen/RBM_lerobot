"""Explicit H1 initialization command. May move the robot; never called by the service/evaluator."""
import argparse
from pathlib import Path
import time
from .sdk import H1SDK


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk-root', type=Path, required=True)
    p.add_argument('--robot-host')
    p.add_argument('--execute', action='store_true', required=True)
    args = p.parse_args()
    adapter = H1SDK.connect(args.sdk_root, args.robot_host)
    robot = adapter.robot
    if not robot.switchControlMode(adapter.sdk.MotorControlMode.LOW_LEVEL):
        raise RuntimeError('Failed to request LOW_LEVEL')
    deadline = time.monotonic()+5
    while robot.getCurrentMode() != adapter.sdk.MotorControlMode.LOW_LEVEL:
        if time.monotonic() >= deadline:
            raise TimeoutError('LOW_LEVEL transition timeout')
        time.sleep(.05)
    if not robot.robot_init():
        raise RuntimeError('robot_init failed')
    deadline = time.monotonic()+20
    while not adapter.ready():
        if time.monotonic() >= deadline:
            raise TimeoutError('H1 initialization timeout')
        time.sleep(.1)
    print('H1 initialized in LOW_LEVEL. No test trajectory or deinitialization was sent.')


if __name__ == '__main__':
    main()
