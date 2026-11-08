"""Add-style chunk-boundary blending, restricted to the two grippers."""


def starts_new_action_chunk(policy):
    """Inspect the queue actually consumed by eval, including relative DM05."""
    rtc = getattr(policy, "_eval_rtc_guidance", None)
    if rtc is not None:
        return rtc.needs_refill(policy)
    policy_type = getattr(policy.config, "type", None)
    if policy_type == "dm05":
        if getattr(policy.config, "use_relative_actions", False):
            return not getattr(policy, "_eval_decoded_action_queue", [])
        queue = getattr(policy, "_queues", {}).get("action")
    elif policy_type == "pi05":
        queue = getattr(policy, "_action_queue", None)
    else:
        raise ValueError("Gripper chunk blending requires PI05 or DM05")
    if queue is None:
        raise RuntimeError(f"Missing {policy_type} action queue for chunk blending")
    return len(queue) == 0


class GripperChunkBlend:
    def __init__(self, steps):
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        self.steps = steps
        self.reset()

    def reset(self):
        self.reference = None
        self.index = 0
        self.status = None

    def update(self, targets, *, new_chunk, previous_sent):
        targets = [float(v) for v in targets]
        if new_chunk:
            self.reference = None if previous_sent is None else list(previous_sent)
            self.index = 0
        active = self.reference is not None and self.index < self.steps
        alpha = (self.index + 1) / (self.steps + 1) if active else 1.0
        output = [p + alpha * (t - p) for p, t in zip(self.reference, targets)] if active else targets.copy()
        self.status = {"new_chunk": bool(new_chunk), "active": active,
                       "alpha": alpha, "input": targets, "output": output.copy(),
                       "reference": None if self.reference is None else self.reference.copy()}
        self.index += 1
        return output
