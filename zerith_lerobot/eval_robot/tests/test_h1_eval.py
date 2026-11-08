import unittest
import numpy as np
from zerith_lerobot.eval_robot.eval_h1 import EvaluationState


class H1EvalTests(unittest.TestCase):
    def test_start_requires_alignment(self):
        s = EvaluationState(np.zeros(23))
        self.assertEqual(s.key('s'), 'idle')
        self.assertEqual(s.key('r'), 'aligning')
        self.assertEqual(s.key('s'), 'aligning')
        self.assertTrue(s.aligned(np.zeros(23)))
        self.assertEqual(s.key('s'), 'running')
        self.assertEqual(s.key('p'), 'paused')
        self.assertEqual(s.key('s'), 'paused')
        self.assertEqual(s.key('b'), 'aligning')
        self.assertEqual(s.key('q'), 'stopped')

    def test_generation_invalidates_old_inference(self):
        s = EvaluationState(np.zeros(23))
        s.key('r'); s.aligned(np.zeros(23)); s.key('s')
        generation = s.generation
        s.key('b')
        self.assertNotEqual(generation, s.generation)
        q = np.zeros(23); q[16] = .01
        self.assertFalse(s.aligned(q))


if __name__ == '__main__':
    unittest.main()
