"""Conservative synchronous PI05 RTC prefix guidance for S1 evaluation.

This is not asynchronous latency compensation: the robot keeps its last target
while prediction runs. Queue progress follows consumed actions, never wall time.
"""

import torch


class SynchronousRTCGuidance:
    def __init__(self, overlap_steps: int = 5):
        self.overlap_steps = overlap_steps
        self.reset()

    def reset(self) -> None:
        self.chunk_id = -1
        self.status = None

    def needs_refill(self, policy) -> bool:
        return len(policy._action_queue) <= self.overlap_steps

    def select_action(self, policy, observation, capture_attention_map: bool = False):
        queue = policy._action_queue
        refilled = self.needs_refill(policy)
        prefix_steps = 0
        if refilled:
            # Keep the prefix in model coordinates, before unnormalization,
            # gripper filtering, expansion to 25D, or hardware safety clipping.
            prefix = torch.stack(list(queue), dim=1).detach().clone() if queue else None
            prefix_steps = len(queue)
            actions = policy.predict_action_chunk(
                observation,
                prev_chunk_left_over=prefix,
                inference_delay=0,
                execution_horizon=self.overlap_steps,
                capture_attention_map=capture_attention_map,
            )
            steps = int(policy.config.n_action_steps)
            if actions.ndim != 3 or actions.shape[1] < steps or not torch.isfinite(actions).all():
                raise RuntimeError("RTC returned an invalid action chunk; no action dispatched")
            # No action was consumed during synchronous prediction. New index 0
            # corresponds to the first pending old action, so do not skip steps.
            queue.clear()
            queue.extend(actions[:, :steps].detach().clone().transpose(0, 1))
            self.chunk_id += 1
        action = queue.popleft().clone()
        self.status = {
            "mode": "guided_sync",
            "chunk_id": self.chunk_id,
            "new_chunk": refilled,
            "prefix_steps": prefix_steps,
            "queue_remaining": len(queue),
            "overlap_steps": self.overlap_steps,
            "inference_delay_steps": 0,
        }
        return action


def configure_rtc_guidance(policy) -> None:
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    if policy.config.type != "pi05" or int(policy.config.n_action_steps) <= 5:
        raise ValueError("S1 RTC guidance requires PI05 with n_action_steps > 5")
    policy.config.rtc_config = RTCConfig(
        enabled=True, mode="guided", execution_horizon=5, max_guidance_weight=1.0,
    )
    policy.init_rtc_processor()
    policy.reset()
    policy._eval_rtc_guidance = SynchronousRTCGuidance()
