"""Bound policy-target lag without requiring every waypoint to fully settle."""
class TrackingPacer:
    def __init__(self, high=0.075, low=0.055, timeout=2.0):
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
                raise RuntimeError("Adaptive tracking stalled; stopping without skipping actions")
        return self.started is not None
