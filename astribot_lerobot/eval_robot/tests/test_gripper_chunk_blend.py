import unittest
from collections import deque
from types import SimpleNamespace
import numpy as np
from astribot_lerobot.eval_robot.utils.gripper_chunk_blend import GripperChunkBlend, starts_new_action_chunk
from astribot_lerobot.eval_robot.utils.gripper_release_filter import GripperReleaseFilter


class ChunkBlendTest(unittest.TestCase):
    def test_dm05_queue_boundaries_and_restart(self):
        queue = deque()
        policy = SimpleNamespace(config=SimpleNamespace(type='dm05', use_relative_actions=False),
                                 _queues={'action': queue})
        self.assertTrue(starts_new_action_chunk(policy))
        queue.extend(range(50))
        for _ in range(50):
            self.assertFalse(starts_new_action_chunk(policy))
            queue.popleft()
        self.assertTrue(starts_new_action_chunk(policy))
        policy._queues = {'action': deque()}
        self.assertTrue(starts_new_action_chunk(policy))

    def test_relative_dm05_uses_decoded_queue(self):
        policy = SimpleNamespace(config=SimpleNamespace(type='dm05', use_relative_actions=True),
                                 _queues={'action': deque()})
        self.assertTrue(starts_new_action_chunk(policy))
        policy._eval_decoded_action_queue = [1, 2]
        self.assertFalse(starts_new_action_chunk(policy))
        policy._eval_decoded_action_queue.clear()
        self.assertTrue(starts_new_action_chunk(policy))

    def test_pi05_and_rtc_boundaries_preserved(self):
        policy = SimpleNamespace(config=SimpleNamespace(type='pi05'), _action_queue=deque())
        self.assertTrue(starts_new_action_chunk(policy))
        policy._action_queue.extend(range(5))
        self.assertFalse(starts_new_action_chunk(policy))
        policy._eval_rtc_guidance = SimpleNamespace(needs_refill=lambda p: len(p._action_queue) <= 5)
        self.assertTrue(starts_new_action_chunk(policy))

    def test_first_chunk_and_five_step_transition(self):
        f = GripperChunkBlend(5)
        self.assertEqual(f.update([36, 0], new_chunk=True, previous_sent=None), [36, 0])
        for i in range(6):
            result = f.update([0, 36], new_chunk=i == 0, previous_sent=[36, 0])
            np.testing.assert_allclose(result, [30 - 6*i, 6 + 6*i])

    def test_reference_is_sent_command_and_reset(self):
        f = GripperChunkBlend(5)
        self.assertEqual(f.update([0, 60], new_chunk=True, previous_sent=[30, 0]), [25, 10])
        self.assertEqual(f.update([0, 60], new_chunk=False, previous_sent=[25, 10]), [20, 20])
        # A shorter chunk restarts using its actual last command.
        np.testing.assert_allclose(f.update([0, 60], new_chunk=True, previous_sent=[20, 20]), [20*5/6, 20+40/6])
        f.reset()
        self.assertEqual(f.update([2, 3], new_chunk=True, previous_sent=None), [2, 3])

    def test_release_confirmation_composition(self):
        blend = GripperChunkBlend(5)
        release = GripperReleaseFilter(.3)
        sent = release.update([36, 0], 0)
        for i in range(6):
            target = blend.update([0, 36], new_chunk=i == 0, previous_sent=sent)
            sent = release.update(target, .1 + i*.1)
            if i < 3:
                self.assertEqual(sent[0], 36)
        self.assertEqual(sent, [0, 36])
