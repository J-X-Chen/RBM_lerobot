"""Checkpoint-owned PI05/DM05 inference, shared by offline and live evaluation."""

from collections import deque

import numpy as np
import torch

CAMERAS = [f"observation.images.{name}" for name in ("head", "left", "right")]
NAMES = [f"{side}.{j}" for side in ("left", "right") for j in (*[f"joint{i}" for i in range(1, 7)], "gripper")]


class AlohaPolicy:
    def __init__(self, checkpoint, device="cuda", n_action_steps=None):
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.pretrained_path = checkpoint
        config.device = device
        if config.type not in {"pi05", "dm05"}:
            raise ValueError("Expected a trained PI05 or DM05 checkpoint")
        if config.use_relative_actions:
            raise ValueError("This evaluator requires the absolute-action ALOHA training recipe")
        if config.input_features["observation.state"].shape != (14,) or config.output_features["action"].shape != (14,):
            raise ValueError("Checkpoint must have native 14D state and action")
        if set(config.image_features) != set(CAMERAS) or config.action_feature_names != NAMES:
            raise ValueError("Checkpoint cameras/joint order do not match the ALOHA training contract")
        config.compile_model = False
        self.n_action_steps = n_action_steps if n_action_steps is not None else config.n_action_steps
        if not 1 <= self.n_action_steps <= config.chunk_size:
            raise ValueError("n_action_steps must be in [1, chunk_size]")
        self.policy = get_policy_class(config.type).from_pretrained(checkpoint, config=config).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.queue = deque()

    def reset(self):
        self.queue.clear()
        self.policy.reset()
        for processor in (self.pre, self.post):
            if hasattr(processor, "reset"):
                processor.reset()

    def warmup(self, observation, task, steps=5):
        """Warm up without exposing any warm-up actions to the control queue."""
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("warm-up steps must be a nonnegative integer")
        self.reset()
        try:
            for _ in range(steps):
                self.predict_chunk(observation, task)
        finally:
            self.reset()

    @torch.inference_mode()
    def predict_chunk(self, observation, task):
        state = np.asarray(observation["observation.state"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Invalid state")
        if getattr(self, "robot_mask", None) is not None:
            observation = self.robot_mask.apply(observation, state, layout="CHW")
        batch = {"observation.state": torch.from_numpy(state), "task": task}
        for key in CAMERAS:
            image = torch.as_tensor(observation[key])
            # Interface is RGB CHW, as returned by LeRobotDataset and live capture.
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(f"Expected RGB CHW image for {key}")
            if image.dtype == torch.uint8:
                image = image.float() / 255
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError(f"Invalid RGB range for {key}")
            batch[key] = image
        normalized = self.policy.predict_action_chunk(self.pre(batch))
        decoded = self.post(normalized).detach().cpu().float().numpy()
        if decoded.ndim != 3 or decoded.shape[0] != 1 or decoded.shape[2] != 14 or not np.isfinite(decoded).all():
            raise ValueError("Policy returned invalid action chunk")
        return decoded[0]

    def action(self, observation, task):
        if not self.queue:
            self.queue.extend(self.predict_chunk(observation, task)[: self.n_action_steps])
        return self.queue.popleft().copy()
