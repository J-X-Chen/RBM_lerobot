"""Explicit H1 initialization command. May move the robot; never called by the service/evaluator."""
import argparse
from pathlib import Path
from .sdk import H1SDK, acquire_low_level


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk-root', type=Path, required=True)
    p.add_argument('--robot-host')
    p.add_argument('--execute', action='store_true', required=True)
    args = p.parse_args()
    adapter = H1SDK.connect(args.sdk_root, args.robot_host)
    acquire_low_level(adapter)
    print('H1 initialized in LOW_LEVEL. No test trajectory or deinitialization was sent.')


if __name__ == '__main__':
    main()
