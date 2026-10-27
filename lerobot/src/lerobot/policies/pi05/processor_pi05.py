#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.processor.tokenizer_processor import _resolve_cached_tokenizer_dir
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.local_assets import resolve_local_model_path

from .configuration_pi05 import PI05Config

PALIGEMMA_TOKENIZER_ID = "google/paligemma-3b-pt-224"
PALIGEMMA_TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)


def resolve_pi05_tokenizer_path(tokenizer_name: str = PALIGEMMA_TOKENIZER_ID) -> tuple[str, bool]:
    """Prefer a project-local or cached PaliGemma tokenizer without a Hub request."""

    local_path = resolve_local_model_path(
        tokenizer_name,
        required_files=PALIGEMMA_TOKENIZER_FILES,
    )
    if local_path is not None:
        return str(local_path), True

    local_path = _resolve_cached_tokenizer_dir(tokenizer_name)
    if local_path is not None:
        return local_path, True
    return tokenizer_name, False


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"
    # MEM section III-D represents proprioception with a linear projection into the
    # backbone instead of discretized prompt tokens, so the state is carried once.
    # Set from `PI05Config.use_proprioceptive_memory`; stock PI0.5 keeps it in the prompt.
    include_state_in_prompt: bool = True
    words_ignore: list[str] = field(default_factory=list)

    def _remove_ignored_words(self, text: str) -> str:
        words_ignore = [word.strip() for word in self.words_ignore if word.strip()]
        if not words_ignore:
            return text

        ignored_pattern = "|".join(re.escape(word) for word in sorted(words_ignore, key=len, reverse=True))
        pattern = rf"(?<!\w)({ignored_pattern})(?!\w)"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return re.sub(r"\s+", " ", text).strip()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        observation = transition.get(TransitionKey.OBSERVATION)
        state = observation.get(OBS_STATE) if observation is not None else None
        if state is None:
            raise ValueError("State is required for PI05")
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            raise ValueError("Complementary data is required for PI05")
        tasks = complementary_data.get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        discretized_states = None
        if self.include_state_in_prompt:
            # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
            # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
            prompt_state = state[:, -1] if state.ndim == 3 else state
            state_np = prompt_state.cpu().numpy()
            discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            cleaned_text = self._remove_ignored_words(cleaned_text)
            if discretized_states is None:
                full_prompt = f"Task: {cleaned_text};\nAction: "
            else:
                state_str = " ".join(map(str, discretized_states[i]))
                full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        complementary_data[self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "max_state_dim": self.max_state_dim,
            "task_key": self.task_key,
            "words_ignore": self.words_ignore,
            "include_state_in_prompt": self.include_state_in_prompt,
        }


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    tokenizer_path, tokenizer_is_local = resolve_pi05_tokenizer_path(config.text_tokenizer_name)

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    steps = make_default_policy_processor_steps(config, dataset_stats)

    # OpenPI order: raw → relative → normalize → model → unnormalize → absolute
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,  # To mimic the same processor as pretrained one
        steps.add_batch_dim,
        relative_step,
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        steps.normalize,
        Pi05PrepareStateTokenizerProcessorStep(
            max_state_dim=config.max_state_dim,
            include_state_in_prompt=not config.use_proprioceptive_memory,
            words_ignore=config.words_ignore,
        ),
        TokenizerProcessorStep(
            tokenizer_name=tokenizer_path,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
            prefer_local_files=True,
            local_files_only=tokenizer_is_local,
        ),
        steps.to_device,
    ]

    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
