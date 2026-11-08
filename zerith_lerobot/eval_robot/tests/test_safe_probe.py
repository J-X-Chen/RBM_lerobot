import unittest
import numpy as np

from astribot_image_service.safe_probe import check_feedback, probe_target


class SafeProbeTest(unittest.TestCase):
    def test_hold_and_bounded_single_joint_ramp(self):
        origin = np.arange(25, dtype=float)
        np.testing.assert_array_equal(probe_target(origin, None, .02, 8), origin)
        np.testing.assert_array_equal(probe_target(origin, 18, .02, 1), origin)
        target = probe_target(origin, 18, -.02, 4)
        self.assertAlmostEqual(target[18] - origin[18], -.01)
        np.testing.assert_array_equal(np.delete(target, 18), np.delete(origin, 18))
        self.assertAlmostEqual(probe_target(origin, 18, .02, 100)[18] - origin[18], .02)

    def test_wrong_joint_movement_is_rejected(self):
        origin = np.zeros(25)
        current = origin.copy()
        current[10] = .011
        with self.assertRaisesRegex(RuntimeError, "Uncommanded joint"):
            check_feedback(origin, current, current, .1, 18)

    def test_large_movement_and_speed_are_rejected(self):
        origin = np.zeros(25)
        for angle, message in [(1.76, "displacement"), (.02, "speed")]:
            current = origin.copy()
            current[18] = angle
            with self.assertRaisesRegex(RuntimeError, message):
                check_feedback(origin, current, origin, .05, 18)

    def test_stale_feedback_and_gripper_motion_rejected(self):
        origin = np.zeros(25)
        with self.assertRaisesRegex(RuntimeError, "interval"):
            check_feedback(origin, origin, origin, .6, None)
        current = origin.copy()
        current[14] = 3
        with self.assertRaisesRegex(RuntimeError, "gripper"):
            check_feedback(origin, current, origin, .1, None)
