import socket
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from zerith_image_service.contract import (PROTOCOL, MOTORS, default_urdf, limits_from_urdf, send_json,
                                           recv_json, ControlClient)
from zerith_image_service.control import H1Control
from zerith_image_service.sdk import H1SDK, MockSDK
from zerith_image_service.safe_probe import probe_target, check_feedback
from zerith_image_service.server import ActionServer

ROOT = Path(__file__).resolve().parents[2]


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.sdk = MockSDK()
        self.sdk.state[19:21] = [.02, .05]
        limits = limits_from_urdf(default_urdf(ROOT / 'zerith_H1_SDK1.3.6'))
        self.control = H1Control(self.sdk, limits, execute=True, clock=lambda: self.now)

    def request(self, owner='one', **payload):
        return self.control.handle(dict(protocol=PROTOCOL, **payload), owner)

    def tick(self, dt=.01):
        self.now += dt
        self.control.tick()

    def test_read_only_startup(self):
        for _ in range(10):
            self.tick()
            self.request(get_state=True)
        self.assertEqual(self.sdk.calls, [])

    def test_head_and_base_locked_with_rate_limits(self):
        self.request(arm=True)
        target = self.sdk.state[:19].copy()
        target[0] += .1
        target[16] += .01
        previous = self.control.command.copy()
        previous_v = np.zeros(21)
        for _ in range(200):
            self.request(action=target.tolist())
            self.tick()
            self.assertIsNone(self.control.fault)
            q = self.control.command
            v = (q[:21]-previous[:21])/.01
            self.assertTrue(np.all(abs(v) <= self.control.velocity_limit+1e-8))
            self.assertTrue(np.all(abs(v-previous_v)/.01 <= self.control.acceleration_limit+1e-7))
            np.testing.assert_allclose(q[19:], [.02, .05, 0, 0])
            previous, previous_v = q.copy(), v
        self.assertGreater(self.sdk.state[0], .09)

    def test_gripper_reaches_boundary_without_overshoot(self):
        self.request(arm=True)
        target = self.sdk.state[:19].copy()
        target[15] = 1.5
        for _ in range(500):
            self.request(action=target.tolist())
            self.tick()
            self.assertIsNone(self.control.fault)
        self.assertAlmostEqual(self.sdk.state[15], 1.5, places=3)

    def test_slow_feedback_rejected_before_write(self):
        self.request(arm=True)
        original = self.sdk.read
        def slow_read():
            self.now += .6
            return original()
        self.sdk.read = slow_read
        self.tick()
        self.assertIn('polling timeout', self.control.fault)
        self.assertEqual(self.sdk.calls, [])

    def test_watchdog_latched(self):
        self.request(arm=True)
        self.tick(.6)
        self.assertIn('watchdog', self.control.fault)
        self.assertEqual(self.sdk.calls, [])
        with self.assertRaises(RuntimeError):
            self.request(hold_current=True)

    def test_disconnect_and_owner(self):
        self.request(arm=True)
        with self.assertRaises(RuntimeError):
            self.request(owner='two', action=self.sdk.state[:19].tolist())
        self.control.disconnect('two')
        self.assertIsNone(self.control.fault)
        self.control.disconnect('one')
        self.tick()
        self.assertEqual(self.sdk.calls, [])
        self.assertIn('disconnected', self.control.fault)

    def test_disarm_cannot_rearm(self):
        self.request(arm=True)
        self.request(disarm=True)
        with self.assertRaises(RuntimeError):
            self.request(arm=True)
        self.tick()
        self.assertEqual(self.sdk.calls, [])

    def test_gripper_roundoff_only_is_clamped(self):
        self.request(arm=True)
        target = self.sdk.state[:19].copy()
        target[7] = -1e-7
        self.request(action=target.tolist())
        self.assertEqual(self.control.target[7], 0.)
        target[7] = -.01
        with self.assertRaises(ValueError):
            self.request(action=target.tolist())

    def test_bad_targets(self):
        self.request(arm=True)
        for a in (np.zeros(23), np.full(19, np.nan), np.full(19, 10)):
            with self.assertRaises(ValueError):
                self.request(action=a.tolist())
        a = self.sdk.state[:19].copy()
        a[0] = .6
        with self.assertRaisesRegex(ValueError, 'envelope'):
            self.request(action=a.tolist())

    def test_runaway_before_sdk_write(self):
        self.request(arm=True)
        self.sdk.state[0] = .1
        self.tick()
        self.assertIn('velocity', self.control.fault)
        self.assertEqual(self.sdk.calls, [])

    def test_dry_run_never_writes(self):
        self.control.execute = False
        self.request(arm=True)
        self.tick()
        self.assertEqual(self.sdk.calls, [])
        self.assertFalse(self.control.diagnostic['sdk_sent'])

    def test_probe_bounds(self):
        origin = self.sdk.state.copy()
        self.assertAlmostEqual(probe_target(origin, 0, .01, 8)[0], .01)
        current = origin.copy()
        current[1] = .02
        with self.assertRaises(RuntimeError):
            check_feedback(origin, current, origin, .1, 0)


class WireTests(unittest.TestCase):
    def test_framed_json_and_invalid_size(self):
        a, b = socket.socketpair()
        try:
            send_json(a, dict(get_state=True))
            self.assertEqual(recv_json(b), dict(get_state=True))
            a.sendall(b'9999999999')
            with self.assertRaises(ValueError):
                recv_json(b)
        finally:
            a.close(); b.close()

    def test_sdk_order_and_wheel_speed(self):
        calls = []
        enum = SimpleNamespace(**{name: name for name in MOTORS+['MOTOR_WHEEL_LEFT', 'MOTOR_WHEEL_RIGHT']})
        sdk = SimpleNamespace(EtherCAT_Motor_Index=enum, MotorControlMode=SimpleNamespace(LOW_LEVEL=1), InitState=SimpleNamespace(Init_Complete=2),
                              Motor_Control=lambda: SimpleNamespace(Position=0., Speed=0.))
        robot = SimpleNamespace(isRobotConnected=lambda: True, getCurrentMode=lambda: 1, getInitState=lambda: 2,
            getMotorState=lambda name: (True, SimpleNamespace(Position_Actual=MOTORS.index(name)/100, Error_flag=0)),
            getChassisSpeedState=lambda: (True, [10, 20], [.1, .2]),
            setMotorControl_low=lambda name, c: (calls.append((name, c.Position, c.Speed)), True)[1])
        adapter = H1SDK(sdk, robot)
        q = adapter.read()
        np.testing.assert_allclose(q[-2:], [.1, .2])
        q[-2:] = 0
        adapter.write(q)
        self.assertEqual([c[0] for c in calls[:21]], MOTORS)
        self.assertEqual(calls[-2:], [('MOTOR_WHEEL_LEFT', 0., 0.), ('MOTOR_WHEEL_RIGHT', 0., 0.)])


if __name__ == '__main__':
    unittest.main()
