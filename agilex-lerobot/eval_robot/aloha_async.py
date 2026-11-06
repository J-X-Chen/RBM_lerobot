"""Small-overlap prefetch; worker never owns hardware or sends commands."""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class PrefetchPolicy:
    def __init__(self, runner, prefetch=5, timeout=2.0):
        self.runner = runner
        self.n_action_steps = runner.n_action_steps
        self.prefetch = min(prefetch, max(1, self.n_action_steps - 1))
        self.timeout = timeout
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.generation = 0
        self.worker_generation = -1
        self.index = 0
        self.queue = {}
        self.status = {}
        self.wait_started = None
        self.chunks = 0
        self.last_inference_s = None
        self.dropped = 0

    def reset(self):
        self.generation += 1
        self.index = 0
        self.queue.clear()
        self.wait_started = None
        if self.future is not None:
            self.future.cancel()
        # Serial worker resets the underlying policy, never concurrently with inference.

    def _predict(self, observation, task, generation):
        if generation != self.worker_generation:
            self.runner.reset()
            self.worker_generation = generation
        started = time.monotonic()
        chunk = self.runner.predict_chunk(observation, task)[:self.n_action_steps]
        return chunk, time.monotonic() - started

    def suspend_prediction(self):
        # Keep queued policy steps/index frozen, but discard pre-wait predictions.
        self.generation += 1
        self.wait_started = None
        if self.future is not None:
            self.future.cancel()

    def action(self, observation, task):
        if self.future is not None:
            if self.future.done():
                if not self.future.cancelled() and self.request_generation == self.generation:
                    result, self.last_inference_s = self.future.result()
                    if self.request_generation == self.generation:
                        if time.monotonic() - self.submitted > self.timeout:
                            raise RuntimeError("Async inference result expired")
                        self.chunks += 1
                        for offset, value in enumerate(result):
                            index = self.request_index + offset
                            if index < self.index:
                                self.dropped += 1
                                continue
                            value = np.asarray(value, dtype=float)
                            if value.shape != (14,) or not np.isfinite(value).all():
                                raise ValueError("Invalid async action")
                            # Same-timestep overlap only, as in S1; do not blend unrelated frames.
                            if index in self.queue:
                                value = 0.5 * (self.queue[index] + value)
                            self.queue[index] = value.copy()
                self.future = None
            elif time.monotonic() - self.submitted > self.timeout:
                raise RuntimeError("Async inference timed out")
        if self.future is None and len(self.queue) <= self.prefetch:
            snapshot = {key: np.array(value, copy=True) for key, value in observation.items()}
            self.request_index = self.index
            self.request_generation = self.generation
            self.submitted = time.monotonic()
            self.future = self.pool.submit(self._predict, snapshot, task, self.generation)
        value = self.queue.pop(self.index, None)
        if value is None:
            if self.wait_started is None:
                self.wait_started = time.monotonic()
            if time.monotonic() - self.wait_started > self.timeout:
                raise RuntimeError("Async action queue empty for too long")
        else:
            self.wait_started = None
        self.status = dict(index=self.index, queued=len(self.queue), waiting=value is None,
                           in_flight=self.future is not None, chunks=self.chunks,
                           inference_s=self.last_inference_s, dropped=self.dropped)
        # Wall-clock control ticks pass even when waiting; expired predictions are never replayed.
        self.index += 1
        return value

    def close(self):
        self.generation += 1
        self.queue.clear()
        if self.future is not None:
            self.future.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)
