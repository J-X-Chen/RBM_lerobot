import unittest
from collections import deque
from types import SimpleNamespace

import torch

from astribot_lerobot.eval_robot.utils.rtc_guidance import SynchronousRTCGuidance


class FakePolicy:
    def __init__(self):
        self.config = SimpleNamespace(n_action_steps=50)
        self._action_queue = deque(maxlen=50)
        self.prefixes = []

    def predict_action_chunk(self, observation, **kwargs):
        prefix = kwargs["prev_chunk_left_over"]
        self.prefixes.append(prefix)
        start = 0 if prefix is None else int(prefix[0, 0, 0])
        return torch.arange(start, start + 50, dtype=torch.float32).reshape(1, 50, 1)


class RTCGuidanceTest(unittest.TestCase):
    def test_overlap_is_pending_actions_without_skip_or_repeat(self):
        policy = FakePolicy()
        rtc = SynchronousRTCGuidance()
        values = [rtc.select_action(policy, {}).item() for _ in range(101)]
        self.assertEqual(values, list(range(101)))
        self.assertEqual(len(policy.prefixes), 3)
        self.assertIsNone(policy.prefixes[0])
        self.assertEqual(policy.prefixes[1].flatten().tolist(), list(range(45, 50)))
        self.assertEqual(policy.prefixes[2].flatten().tolist(), list(range(90, 95)))

    def test_postprocessing_cannot_modify_pending_prefix(self):
        policy = FakePolicy()
        rtc = SynchronousRTCGuidance()
        for _ in range(45):
            rtc.select_action(policy, {}).fill_(-999)
        rtc.select_action(policy, {})
        self.assertEqual(policy.prefixes[-1].flatten().tolist(), list(range(45, 50)))

    def test_waiting_does_not_advance_queue_and_reset_starts_fresh(self):
        policy = FakePolicy()
        rtc = SynchronousRTCGuidance()
        rtc.select_action(policy, {})
        for _ in range(100):
            self.assertFalse(rtc.needs_refill(policy))
        self.assertEqual(len(policy._action_queue), 49)
        policy._action_queue.clear()
        rtc.reset()
        self.assertEqual(rtc.select_action(policy, {}).item(), 0)
        self.assertIsNone(policy.prefixes[-1])
        self.assertEqual(rtc.status["chunk_id"], 0)

    def test_nonfinite_chunk_rejected_before_queue_replacement(self):
        policy = FakePolicy()
        policy.predict_action_chunk = lambda *a, **kw: torch.full((1, 50, 1), float("nan"))
        with self.assertRaisesRegex(RuntimeError, "invalid action chunk"):
            SynchronousRTCGuidance().select_action(policy, {})
        self.assertEqual(len(policy._action_queue), 0)


if __name__ == "__main__":
    unittest.main()
