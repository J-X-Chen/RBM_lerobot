import unittest

import numpy as np

from astribot_lerobot.eval_robot.utils.non_policy_joint_hold import NonPolicyJointHold


class NonPolicyJointHoldTest(unittest.TestCase):
    def test_schedule_and_retained_target(self):
        initial = np.zeros(25)
        hold = NonPolicyJointHold(initial, 0.0)
        measured = initial.copy()
        measured[3] = 0.01
        np.testing.assert_array_equal(hold.update(measured, 19.9), initial)
        np.testing.assert_array_equal(hold.update(measured, 20.0), initial)
        self.assertTrue(hold.active)
        corrected = hold.update(measured, 20.1)
        self.assertLess(corrected[3], 0)
        np.testing.assert_array_equal(hold.update(measured, 25.0), corrected)
        self.assertFalse(hold.active)
        np.testing.assert_array_equal(hold.update(measured, 39.9), corrected)
        hold.update(measured, 40.0)
        self.assertEqual(hold.cycle, 2)

    def test_simulated_tracking_bias_converges_without_touching_policy_joints(self):
        initial = np.linspace(-0.5, 0.5, 25)
        hold = NonPolicyJointHold(initial, 0.0)
        bias = np.zeros(25)
        bias[3:7] = [0.001, 0.006, -0.009, 0.008]
        bias[23:25] = [-0.003, -0.004]
        measured = initial + bias
        for now in np.arange(0.0, 26.0, 1 / 30):
            target = hold.update(measured, float(now))
            np.testing.assert_array_equal(target[7:23], initial[7:23])
            measured += 0.2 * (target + bias - measured)
        self.assertLess(np.abs(measured[hold.indices] - initial[hold.indices]).max(), 0.0022)

    def test_speed_offset_bounds_and_no_catchup_after_stall(self):
        hold = NonPolicyJointHold(np.zeros(25), 0.0)
        measured = np.ones(25)
        previous = hold.target.copy()
        for now in np.arange(0.0, 200.0, 0.1):
            target = hold.update(measured, float(now))
            delta = np.abs(target[hold.indices] - previous[hold.indices])
            self.assertTrue(np.all(delta <= hold.max_speed * 0.1 + 1e-12))
            self.assertTrue(np.all(np.abs(target[hold.indices]) <= hold.max_offset + 1e-12))
            previous = target
        self.assertTrue(hold.snapshot()["at_offset_limit"])
        np.testing.assert_array_equal(hold.update(measured, 1000.0), previous)

    def test_disturbance_removed_can_unwind_compensation(self):
        hold = NonPolicyJointHold(np.zeros(25), 0.0)
        measured = np.zeros(25)
        for now in np.arange(0.0, 46.0, 1 / 30):
            target = hold.update(measured, float(now))
            bias = np.zeros(25)
            if now < 30:
                bias[3] = 0.01
            measured += 0.2 * (target + bias - measured)
        self.assertLess(abs(measured[3]), 0.0022)

    def test_deadband_and_new_run_reference(self):
        hold = NonPolicyJointHold(np.zeros(25), 0.0)
        for now in [20.0, 20.1, 21.0]:
            np.testing.assert_array_equal(hold.update(np.full(25, 0.0001), now), np.zeros(25))
        restarted = NonPolicyJointHold(np.ones(25), 30.0)
        self.assertEqual(restarted.next_correction, 50.0)
        np.testing.assert_array_equal(restarted.target, np.ones(25))

    def test_invalid_feedback_is_rejected(self):
        hold = NonPolicyJointHold(np.zeros(25), 0.0)
        for invalid in [np.zeros(16), np.full(25, np.nan)]:
            with self.assertRaises(ValueError):
                hold.update(invalid, 20.0)
        with self.assertRaises(ValueError):
            hold.update(np.zeros(25), -1.0)


if __name__ == "__main__":
    unittest.main()
