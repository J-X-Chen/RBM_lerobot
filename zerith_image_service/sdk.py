"""Native SDK 1.3.6 adapter. Import/connection is lazy; constructors do not command motion."""
import importlib
from pathlib import Path
import platform
import sys
import time

import numpy as np
from .contract import MOTORS, vector


def acquire_low_level(adapter, log=print):
    """Return once adapter.ready() is true; may deinit/init, which briefly releases arm support.

    The native state machine refuses switchControlMode(LOW_LEVEL) while Init_Complete, and only
    accepts Uninit / Deinit_Complete / Error_State, so the deinit step is mandatory once a previous
    session left the robot initialized but in the default UNINITIALIZED control mode.
    """
    sdk, robot = adapter.sdk, adapter.robot
    if adapter.ready():
        log('SDK session already ready (LOW_LEVEL + Init_Complete).')
        return adapter
    log(f'Acquiring LOW_LEVEL: mode={robot.getCurrentMode()} init={robot.getInitState()}')
    switched = False
    if robot.getInitState() == sdk.InitState.Init_Complete:
        log('robot_deinit() to leave Init_Complete; arms may briefly lose support.')
        if robot.robot_deinit():
            deadline = time.monotonic() + 25
            while robot.getInitState() != sdk.InitState.Deinit_Complete:
                if time.monotonic() >= deadline:
                    raise TimeoutError('robot_deinit state timeout')
                time.sleep(.1)
        elif robot.switchControlMode(sdk.MotorControlMode.LOW_LEVEL):
            # The reported init state can be stale after the vendor control released the robot;
            # a switch needs Uninit/Deinit_Complete, so accept it if the robot allows it.
            switched = True
        else:
            raise RuntimeError(
                'Vendor state machine deadlocked: robot_deinit refused (control mode is '
                'UNINITIALIZED) and switchControlMode refused (init state is Init_Complete). '
                'Restart the vendor stack on the H1 (sudo systemctl restart robotd), wait for the '
                'camera service to come back, then start this service again.')
    if not switched and not robot.switchControlMode(sdk.MotorControlMode.LOW_LEVEL):
        raise RuntimeError('switchControlMode(LOW_LEVEL) refused')
    deadline = time.monotonic() + 5
    while robot.getCurrentMode() != sdk.MotorControlMode.LOW_LEVEL:
        if time.monotonic() >= deadline:
            raise TimeoutError('LOW_LEVEL transition timeout')
        time.sleep(.05)
    if adapter.ready():
        log('LOW_LEVEL acquired; robot was already initialized.')
        return adapter
    if not robot.robot_init():
        raise RuntimeError('robot_init failed')
    deadline = time.monotonic() + 30
    while not adapter.ready():
        if time.monotonic() >= deadline:
            raise TimeoutError('H1 initialization timeout')
        time.sleep(.1)
    log('LOW_LEVEL acquired and initialized.')
    return adapter


def load_sdk(root):
    if sys.version_info[:2] != (3, 10) or platform.machine() not in {'x86_64', 'AMD64'}:
        raise RuntimeError('Bundled H1 SDK requires Linux x86_64 Python 3.10')
    root = Path(root).resolve()
    sys.path.insert(0, str(root / 'h1_sdk_v1.3.6_python3.10'))
    return importlib.import_module('lib.lib_h1_sdk_python')


class H1SDK:
    def __init__(self, sdk, robot):
        self.sdk, self.robot = sdk, robot

    @classmethod
    def connect(cls, root, host=None):
        sdk = load_sdk(root)
        robot = sdk.H1Robot(host) if host else sdk.H1Robot()
        if not robot.robot_connect():
            raise RuntimeError('H1 robot_connect failed')
        return cls(sdk, robot)

    def ready(self):
        return (self.robot.isRobotConnected()
                and self.robot.getCurrentMode() == self.sdk.MotorControlMode.LOW_LEVEL
                and self.robot.getInitState() == self.sdk.InitState.Init_Complete)

    def read(self):
        if not self.robot.isRobotConnected():
            raise RuntimeError('H1 disconnected')
        state = []
        for name in MOTORS:
            ok, motor = self.robot.getMotorState(getattr(self.sdk.EtherCAT_Motor_Index, name))
            if not ok or int(motor.Error_flag):
                raise RuntimeError(f'Invalid motor feedback: {name}')
            state.append(float(motor.Position_Actual))
        ok, _, speed = self.robot.getChassisSpeedState()
        if not ok:
            raise RuntimeError('Invalid chassis speed feedback')
        return vector(state + list(speed), 23)

    def write(self, command):
        command = vector(command, 23)
        if not self.ready() or np.any(command[21:] != 0):
            raise RuntimeError('H1 not initialized in LOW_LEVEL, or nonzero chassis target')
        for name, position in zip(MOTORS, command[:21]):
            control = self.sdk.Motor_Control()
            control.Position = float(position)
            result = self.robot.setMotorControl_low(getattr(self.sdk.EtherCAT_Motor_Index, name), control)
            if not result:
                raise RuntimeError(f'SDK write failed: {name}')
        for name in ('MOTOR_WHEEL_LEFT', 'MOTOR_WHEEL_RIGHT'):
            control = self.sdk.Motor_Control()
            control.Speed = 0.0
            if not self.robot.setMotorControl_low(getattr(self.sdk.EtherCAT_Motor_Index, name), control):
                raise RuntimeError(f'SDK wheel stop failed: {name}')


class MockSDK:
    """Explicit simulation; never used as a fallback after a hardware error."""
    def __init__(self):
        self.state = np.zeros(23)
        self.state[16] = .65
        self.calls = []

    def ready(self):
        return True

    def read(self):
        return self.state.copy()

    def write(self, command):
        self.state = vector(command, 23)
        self.calls.append(self.state.copy())
