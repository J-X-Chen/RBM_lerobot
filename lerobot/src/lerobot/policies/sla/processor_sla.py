# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""SLA processors: the pi0.5 pipeline plus a state-free prompt used for training-time state dropout.

The reference SLA training randomly removes the discretized robot state from the prompt
(``StateDropout``, p=0.25). Processor pipelines are shared between training and inference and
have no notion of the training mode, so the preprocessor emits both prompt variants and the
policy picks per sample, only while training.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.configs.types import FeatureType
from lerobot.lerobot_types import EnvTransition, RobotObservation, TransitionKey
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.processor.factory import load_pretrained_policy_processors
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, POLICY_PREPROCESSOR_DEFAULT_NAME

from ..pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep, resolve_pi05_tokenizer_path
from .configuration_sla import SLAConfig

STATELESS_TASK_KEY = "task_without_state"
OBS_LANGUAGE_TOKENS_WITHOUT_STATE = "observation.language.tokens_without_state"
OBS_LANGUAGE_ATTENTION_MASK_WITHOUT_STATE = "observation.language.attention_mask_without_state"
SLA_TOKENIZER_STEP_NAME = "sla_tokenizer_processor"


@ProcessorStepRegistry.register(name="sla_prepare_state_tokenizer_processor_step")
@dataclass
class SLAPrepareStateTokenizerProcessorStep(Pi05PrepareStateTokenizerProcessorStep):
    """pi0.5 prompt formatting that also stores the state-free prompt under ``task_without_state``."""

    emit_stateless_prompt: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        stateless_prompts = None
        if self.emit_stateless_prompt and self.include_state_in_prompt:
            complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
            tasks = complementary_data.get(self.task_key)
            if tasks is not None:
                tasks = [tasks] if isinstance(tasks, str) else list(tasks)
                stateless_prompts = [
                    f"Task: {self._remove_ignored_words(task.strip().replace('_', ' ').replace(chr(10), ' '))};"
                    "\nAction: "
                    for task in tasks
                ]
        transition = super().__call__(transition)
        if stateless_prompts is not None:
            complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
            complementary_data[STATELESS_TASK_KEY] = stateless_prompts
            transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data  # type: ignore[literal-required]
        return transition

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "emit_stateless_prompt": self.emit_stateless_prompt}


@ProcessorStepRegistry.register(name=SLA_TOKENIZER_STEP_NAME)
@dataclass
class SLATokenizerProcessorStep(TokenizerProcessorStep):
    """PaliGemma tokenizer that additionally tokenizes the state-free prompt when present."""

    def observation(self, observation: RobotObservation) -> RobotObservation:
        new_observation = super().observation(observation)
        complementary_data = self.transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        stateless = complementary_data.get(STATELESS_TASK_KEY)
        if stateless is None:
            return new_observation
        tokenized = self._tokenize_text(list(stateless))
        device = new_observation[OBS_LANGUAGE_TOKENS].device
        new_observation[OBS_LANGUAGE_TOKENS_WITHOUT_STATE] = tokenized["input_ids"].to(device)
        new_observation[OBS_LANGUAGE_ATTENTION_MASK_WITHOUT_STATE] = tokenized["attention_mask"].to(
            device=device, dtype=torch.bool
        )
        return new_observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features = super().transform_features(features)
        for key in (OBS_LANGUAGE_TOKENS_WITHOUT_STATE, OBS_LANGUAGE_ATTENTION_MASK_WITHOUT_STATE):
            features[PipelineFeatureType.OBSERVATION].setdefault(
                key, PolicyFeature(type=FeatureType.LANGUAGE, shape=(self.max_length,))
            )
        return features


def make_sla_pre_post_processors(
    config: SLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """pi0.5 pre/post-processing (see ``make_pi05_pre_post_processors``) with the SLA prompt steps."""
    tokenizer_path, tokenizer_is_local = resolve_pi05_tokenizer_path(config.text_tokenizer_name)

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )
    steps = make_default_policy_processor_steps(config, dataset_stats)

    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        relative_step,
        # The state must be normalized to [-1, 1] before it is discretized into the prompt.
        steps.normalize,
        SLAPrepareStateTokenizerProcessorStep(
            max_state_dim=config.max_state_dim,
            include_state_in_prompt=not config.use_proprioceptive_memory,
            words_ignore=config.words_ignore,
            emit_stateless_prompt=config.state_dropout_prob > 0,
        ),
        SLATokenizerProcessorStep(
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


def _saved_preprocessor_is_sla(pretrained_path: str | Path, filename: str) -> bool:
    config_file = Path(pretrained_path) / filename
    if not config_file.is_file():
        return False
    try:
        steps = json.loads(config_file.read_text()).get("steps", [])
    except (OSError, json.JSONDecodeError):
        return False
    return any(step.get("registry_name") == SLA_TOKENIZER_STEP_NAME for step in steps)


def make_sla_pre_post_processors_from_pretrained(
    config: SLAConfig,
    pretrained_path: str | Path,
    *,
    revision: str | None,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None,
    dataset_meta: Any | None,
    preprocessor_overrides: dict[str, Any] | None,
    postprocessor_overrides: dict[str, Any] | None,
    preprocessor_config_filename: str,
    postprocessor_config_filename: str,
) -> tuple[Any, Any]:
    """Load saved SLA processors, or build fresh ones when starting from a pi0.5/converted checkpoint.

    Base checkpoints such as ``lerobot/pi05_base`` or a converted ``SLA_base`` ship pi0.5 (or no)
    processor files, which lack the state-free prompt; those are rebuilt from the SLA config.
    """
    del dataset_meta
    if not _saved_preprocessor_is_sla(pretrained_path, preprocessor_config_filename):
        logging.info(
            "No SLA processors found at %s; building SLA processors from the policy config.", pretrained_path
        )
        overrides = preprocessor_overrides or {}
        stats = dataset_stats
        if stats is None:
            stats = (overrides.get("normalizer_processor") or {}).get("stats")
        preprocessor, postprocessor = make_sla_pre_post_processors(config, dataset_stats=stats)
        rename_map = (overrides.get("rename_observations_processor") or {}).get("rename_map")
        if rename_map:
            for step in preprocessor.steps:
                if isinstance(step, RenameObservationsProcessorStep):
                    step.rename_map = dict(rename_map)
        return preprocessor, postprocessor

    tokenizer_path, tokenizer_is_local = resolve_pi05_tokenizer_path(config.text_tokenizer_name)
    preprocessor_overrides = dict(preprocessor_overrides or {})
    tokenizer_overrides = dict(preprocessor_overrides.get(SLA_TOKENIZER_STEP_NAME, {}))
    tokenizer_overrides.update(
        {"tokenizer_name": tokenizer_path, "prefer_local_files": True, "local_files_only": tokenizer_is_local}
    )
    preprocessor_overrides[SLA_TOKENIZER_STEP_NAME] = tokenizer_overrides
    return load_pretrained_policy_processors(
        pretrained_path,
        revision=revision,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
        preprocessor_config_filename=preprocessor_config_filename
        or f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        postprocessor_config_filename=postprocessor_config_filename,
    )
