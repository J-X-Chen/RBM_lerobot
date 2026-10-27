import unittest
from astribot_lerobot.eval_robot.utils.gripper_release_filter import GripperReleaseFilter


class ReleaseFilterTest(unittest.TestCase):
    def test_transient_and_independent_closing(self):
        f = GripperReleaseFilter()
        f.update([35, 10], 0)
        self.assertEqual(f.update([3, 20], .1), [35, 20])
        self.assertEqual(f.update([36, 20], .2), [36, 20])
        self.assertEqual(f.update([3, 20], .4), [36, 20])
        self.assertEqual(f.update([3, 20], .71), [3, 20])
        self.assertEqual(f.update([0, 20], .72), [0, 20])

    def test_reset_and_rearm(self):
        f = GripperReleaseFilter()
        f.update([35, 35], 0)
        f.update([10, 35], .1)
        f.update([10, 35], .41)
        f.update([30, 35], .5)
        self.assertEqual(f.update([0, 35], .6), [30, 35])
        f.reset()
        self.assertEqual(f.update([0, 0], .7), [0, 0])

    def test_no_slow_noise_drift_and_frequency_independent(self):
        for hz in (10, 30):
            f = GripperReleaseFilter()
            f.update([35, 35], 0)
            for k in range(1, 11):
                self.assertEqual(f.update([34.8, 35], k / hz), [35, 35])
            start = 2.
            f.update([0, 35], start)
            for k in range(1, int(.3 * hz)):
                self.assertEqual(f.update([0, 35], start + k / hz), [35, 35])
            self.assertEqual(f.update([0, 35], start + .301), [0, 35])
