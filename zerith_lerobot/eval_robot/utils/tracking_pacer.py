"""Bound lag without requiring every policy waypoint to settle."""
class TrackingPacer:
    def __init__(self, high=.075, low=.055, timeout=2.):
        self.high, self.low, self.timeout = high, low, timeout
        self.reset()

    def reset(self):
        self.started = None

    def wait(self, error, now):
        if self.started is None and error > self.high:
            self.started = now
        if self.started is not None:
            if error <= self.low:
                self.started = None
            elif now - self.started >= self.timeout:
                raise RuntimeError("Adaptive tracking stalled for 2s; stopping policy rather than skipping actions")
        return self.started is not None
