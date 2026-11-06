"""Hardware discovery tests: fake query-only SDK, no real CAN."""

import copy
import math
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from hardware_config import load_hardware, resolve_hardware


class QuerySDK:
    instances = []
    stale = False

    def __init__(self, can_name):
        self.name = can_name
        self.closed = False
        self.motor = 0
        self.instances.append(self)

    def ConnectPort(self, piper_init):
        assert piper_init is False

    def DisconnectPort(self):
        self.closed = True

    def SearchMotorMaxAngleSpdAccLimit(self, motor_num, search_content):
        assert search_content == 1
        self.motor = motor_num

    def GetCurrentMotorAngleLimitMaxVel(self):
        return NS(
            time_stamp=0 if self.stale else time.time(),
            current_motor_angle_limit_max_vel=NS(motor_num=self.motor, min_angle_limit=-900, max_angle_limit=900),
        )

    def ArmParamEnquiryAndConfig(self, **kwargs):
        assert kwargs == {"param_enquiry": 4}

    def GetGripperTeachingPendantParamFeedback(self):
        return NS(time_stamp=time.time(), arm_gripper_teaching_param_feedback=NS(max_range_config=100))


class HardwareTests(unittest.TestCase):
    def setUp(self):
        QuerySDK.instances = []
        QuerySDK.stale = False
        _, self.config = load_hardware(Path(__file__).with_name("hardware.json"))

    def test_queries_resolve_units_and_leave_input_unchanged(self):
        original = copy.deepcopy(self.config)
        result = resolve_hardware(self.config, QuerySDK)
        for side in ("left", "right"):
            self.assertAlmostEqual(result[side]["joint_min"][0], -math.pi / 2)
            self.assertAlmostEqual(result[side]["joint_max"][0], math.pi / 2)
            self.assertEqual(result[side]["gripper_max_m"], 0.1)
            self.assertEqual(result[side]["gripper_scale"], 1)
        self.assertEqual(self.config, original)
        self.assertTrue(all(arm.closed for arm in QuerySDK.instances))

    def test_zero_timestamp_is_not_a_hardware_reply(self):
        QuerySDK.stale = True
        with self.assertRaisesRegex(RuntimeError, "No fresh hardware reply"):
            resolve_hardware(self.config, QuerySDK, timeout=0.001)
        self.assertTrue(QuerySDK.instances[0].closed)

    def test_explicit_limits_need_no_query(self):
        config = resolve_hardware(self.config, QuerySDK)
        QuerySDK.instances = []
        self.assertEqual(resolve_hardware(config, QuerySDK), config)
        self.assertEqual(QuerySDK.instances, [])

    def test_bad_gripper_range_is_rejected(self):
        class InvalidGrip(QuerySDK):
            def GetGripperTeachingPendantParamFeedback(self):
                return NS(time_stamp=time.time(), arm_gripper_teaching_param_feedback=NS(max_range_config=0))

        with self.assertRaisesRegex(RuntimeError, "unconfigured gripper range"):
            resolve_hardware(self.config, InvalidGrip)
        self.assertTrue(QuerySDK.instances[-1].closed)


if __name__ == "__main__":
    unittest.main()
