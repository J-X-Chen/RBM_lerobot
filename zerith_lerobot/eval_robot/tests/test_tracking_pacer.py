import unittest
from astribot_lerobot.eval_robot.utils.tracking_pacer import TrackingPacer

class PacerTest(unittest.TestCase):
    def test_hysteresis_timeout_and_reset(self):
        p = TrackingPacer()
        self.assertFalse(p.wait(.06, 0))
        self.assertTrue(p.wait(.08, .1))
        self.assertTrue(p.wait(.06, .2))
        self.assertFalse(p.wait(.05, .3))
        self.assertTrue(p.wait(.08, .4))
        with self.assertRaises(RuntimeError):
            p.wait(.08, 2.5)
        p.reset()
        self.assertFalse(p.wait(.04, 3))
