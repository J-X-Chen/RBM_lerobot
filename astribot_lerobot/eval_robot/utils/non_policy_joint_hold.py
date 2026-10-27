"""Bounded feedback correction for S1 joints outside the 16D policy view."""

import numpy as np


class NonPolicyJointHold:
    """Keep the initial measured pose; periodically compensate tracking bias.

    Chassis x/y use metres; the remaining held joints use radians. Only indices
    0:7 and 23:25 are changed. Call once per fresh observation, using monotonic
    time. Correction does not block or generate arm/gripper commands.
    """

    indices = np.array([0, 1, 2, 3, 4, 5, 6, 23, 24])
    interval_s = 20.0
    duration_s = 5.0

    def __init__(self, initial_pose: np.ndarray, now: float):
        self.reference = self._pose(initial_pose).copy()
        self.target = self.reference.copy()
        self.next_correction = now + self.interval_s
        self.correction_end = now
        self.last_time = now
        self.cycle = 0
        self.active = False
        self.error = np.zeros(9)
        self.max_offset = np.array([0.01, 0.01] + [0.05] * 7)
        self.max_speed = np.array([0.002, 0.002] + [0.01] * 7)
        self.tolerance = np.array([0.001, 0.001] + [0.002] * 7)

    @staticmethod
    def _pose(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (25,) or not np.all(np.isfinite(value)):
            raise ValueError("Non-policy hold requires a finite 25D S1 pose.")
        return value

    def update(self, measured_pose: np.ndarray, now: float) -> np.ndarray:
        measured = self._pose(measured_pose)
        if not np.isfinite(now) or now < self.last_time:
            raise ValueError("Non-policy hold requires finite, monotonic time.")
        # Never turn a slow inference or a pause into a large correction jump.
        dt = min(now - self.last_time, 0.1)
        self.last_time = now
        if now >= self.next_correction:
            self.cycle += 1
            self.correction_end = now + self.duration_s
            self.next_correction = now + self.interval_s
            dt = 0.0
        self.active = now < self.correction_end
        self.error = self.reference[self.indices] - measured[self.indices]
        if self.active:
            error = np.where(np.abs(self.error) > self.tolerance, self.error, 0.0)
            increment = np.clip(0.5 * error * dt, -self.max_speed * dt, self.max_speed * dt)
            offset = self.target[self.indices] - self.reference[self.indices]
            offset = np.clip(offset + increment, -self.max_offset, self.max_offset)
            self.target[self.indices] = self.reference[self.indices] + offset
        return self.target.copy()

    def snapshot(self) -> dict:
        return {
            "cycle": self.cycle,
            "active": self.active,
            "indices": self.indices.tolist(),
            "reference": self.reference[self.indices].tolist(),
            "target": self.target[self.indices].tolist(),
            "error": self.error.tolist(),
            "at_offset_limit": bool(np.any(
                np.abs(self.target[self.indices] - self.reference[self.indices])
                >= self.max_offset - 1e-9
            )),
        }
