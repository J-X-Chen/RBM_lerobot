"""Debounce opening requests; larger S1 gripper values mean closing."""
import math


class GripperReleaseFilter:
    def __init__(self, confirm_s=0.3, deadband=0.5):
        if not math.isfinite(confirm_s) or confirm_s <= 0:
            raise ValueError("confirm_s must be finite and positive")
        self.confirm_s = confirm_s
        self.deadband = deadband
        self.reset()

    def reset(self):
        self.output = None
        self.pending = [None, None]
        self.releasing = [False, False]
        self.status = None

    def update(self, targets, now):
        values = [float(v) for v in targets]
        if len(values) != 2 or not all(math.isfinite(v) for v in values) or not math.isfinite(now):
            raise ValueError("finite left/right gripper targets and time required")
        values = [min(100., max(0., v)) for v in values]
        if self.output is None:
            self.output = values.copy()
        held = [False, False]
        for i, target in enumerate(values):
            previous = self.output[i]
            if target >= previous:
                if target > previous + self.deadband:
                    self.releasing[i] = False
                self.pending[i] = None
                self.output[i] = target
            elif self.releasing[i]:
                self.output[i] = target
            elif target < previous - self.deadband:
                if self.pending[i] is None:
                    self.pending[i] = now
                if now - self.pending[i] >= self.confirm_s:
                    self.releasing[i] = True
                    self.pending[i] = None
                    self.output[i] = target
                else:
                    held[i] = True
            else:
                self.pending[i] = None
                held[i] = True
        self.status = {"input": values, "output": self.output.copy(),
                       "held": held, "release_confirmed": self.releasing.copy()}
        return self.output.copy()
