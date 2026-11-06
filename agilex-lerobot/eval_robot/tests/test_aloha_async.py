import sys
import time
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aloha_async import PrefetchPolicy


class Runner:
    n_action_steps = 50

    def __init__(self):
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.calls = 0

    def reset(self):
        pass

    def predict_chunk(self, obs, task):
        self.calls += 1
        self.entered.set()
        assert self.gate.wait(2)
        return np.repeat(np.arange(50)[:, None], 14, axis=1) + obs['observation.state'][0]


def test_background_predict_does_not_block_and_discards_elapsed_ticks():
    runner = Runner()
    policy = PrefetchPolicy(runner)
    obs = {'observation.state': np.zeros(14)}
    try:
        assert policy.action(obs, '') is None
        assert runner.entered.wait(1)
        assert policy.action(obs, '') is None
        obs['observation.state'][0] = 1000  # worker received an immutable snapshot
        runner.gate.set()
        policy.future.result(timeout=1)
        np.testing.assert_allclose(policy.action(obs, ''), 2)
        for _ in range(43):
            policy.action(obs, '')
        assert policy.future is not None  # refill started with five old steps left
    finally:
        runner.gate.set()
        policy.close()


def test_reset_discards_old_generation_and_timeout_stops():
    runner = Runner()
    policy = PrefetchPolicy(runner, timeout=0.03)
    obs = {'observation.state': np.zeros(14)}
    try:
        policy.action(obs, '')
        assert runner.entered.wait(1)
        policy.reset()
        runner.gate.set()
        policy.future.result(timeout=1)
        assert policy.action(obs, '') is None  # old generation not replayed
        policy.future.result(timeout=1)
        policy.submitted = time.monotonic() - 1
        with pytest.raises(RuntimeError, match='expired'):
            policy.action(obs, '')
    finally:
        runner.gate.set()
        policy.close()
