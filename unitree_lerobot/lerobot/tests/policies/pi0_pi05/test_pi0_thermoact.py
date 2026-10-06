#!/usr/bin/env python

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

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0.thermoact import (
    THERMOACT_CHANNEL,
    thermoact_thermal_to_pseudo_rgb,
)
from lerobot.utils.constants import ACTION, OBS_STATE


class PI0ThermoActInputAdapter(torch.nn.Module):
    """Expose PI0 image preprocessing without constructing the full PaliGemma model."""

    _preprocess_images = PI0Policy._preprocess_images

    def __init__(self, config: PI0Config) -> None:
        super().__init__()
        self.config = config
        self._device_anchor = torch.nn.Parameter(
            torch.empty((), device=config.device), requires_grad=False
        )


def test_pi0_config_accepts_thermoact_channel_case_insensitive():
    config = PI0Config(device="cpu", thermal_encoder_channel="ThermoAct")

    assert config.thermal_encoder_channel == THERMOACT_CHANNEL
    assert config.uses_thermoact_thermal_input


def test_pi0_config_rejects_non_thermoact_thermal_channels():
    with pytest.raises(ValueError, match="false or 'ThermoAct'"):
        PI0Config(device="cpu", thermal_encoder_channel="ViT")


def test_pi0_config_rejects_empty_thermoact_feature_list():
    with pytest.raises(ValueError, match="thermal_image_features"):
        PI0Config(device="cpu", thermal_encoder_channel="ThermoAct", thermal_image_features=[])


def test_thermoact_thermal_to_pseudo_rgb_normalizes_clamps_and_colorizes():
    raw_celsius = torch.tensor([[[[10.0, 20.0], [35.0, 50.0]]]])

    pseudo_rgb = thermoact_thermal_to_pseudo_rgb(raw_celsius)

    assert pseudo_rgb.shape == (1, 3, 2, 2)
    assert pseudo_rgb.dtype == torch.float32
    assert pseudo_rgb.min() >= 0
    assert pseudo_rgb.max() <= 1
    torch.testing.assert_close(pseudo_rgb[0, :, 0, 0], pseudo_rgb[0, :, 0, 1])
    torch.testing.assert_close(pseudo_rgb[0, :, 1, 0], pseudo_rgb[0, :, 1, 1])
    assert pseudo_rgb[0, :, 1, 0].mean() > pseudo_rgb[0, :, 0, 0].mean()


def test_thermoact_thermal_to_pseudo_rgb_preserves_channels_last_layout():
    raw_celsius = torch.tensor([[[20.0], [35.0]], [[27.5], [30.0]]])

    pseudo_rgb = thermoact_thermal_to_pseudo_rgb(raw_celsius)

    assert pseudo_rgb.shape == (2, 2, 3)
    assert pseudo_rgb.dtype == torch.float32


def test_thermoact_batched_scalar_images_keep_batch_dimension():
    raw_celsius = torch.linspace(20.0, 35.0, 16).reshape(1, 4, 4)

    pseudo_rgb = thermoact_thermal_to_pseudo_rgb(raw_celsius, batched=True)

    assert pseudo_rgb.shape == (1, 3, 4, 4)
    assert pseudo_rgb.dtype == torch.float32


def test_pi0_thermoact_preprocesses_only_thermal_image_slot():
    config = PI0Config(device="cpu", thermal_encoder_channel="ThermoAct")
    config.image_resolution = (224, 224)
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.images.cam_left_high": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 32, 32)
        ),
        "observation.images.cam_thermal": PolicyFeature(
            type=FeatureType.VISUAL, shape=(1, 16, 16)
        ),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
    }
    config.validate_features()

    adapter = PI0ThermoActInputAdapter(config)
    batch = {
        "observation.images.cam_left_high": torch.rand(2, 3, 32, 32),
        "observation.images.cam_thermal": torch.linspace(20.0, 35.0, 16 * 16)
        .reshape(1, 1, 16, 16)
        .repeat(2, 1, 1, 1),
    }

    images, img_masks = adapter._preprocess_images(batch)

    assert len(images) == 2
    assert len(img_masks) == 2
    wrist_rgb, thermal_rgb = images
    assert wrist_rgb.shape == (2, 3, 224, 224)
    assert thermal_rgb.shape == (2, 3, 224, 224)
    assert thermal_rgb.min() >= -1
    assert thermal_rgb.max() <= 1
    assert all(mask.tolist() == [True, True] for mask in img_masks)


def test_pi0_thermoact_preprocesses_batched_scalar_thermal_images():
    config = PI0Config(device="cpu", thermal_encoder_channel="ThermoAct")
    config.image_resolution = (224, 224)
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.images.cam_thermal": PolicyFeature(
            type=FeatureType.VISUAL, shape=(16, 16)
        ),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
    }
    config.validate_features()

    adapter = PI0ThermoActInputAdapter(config)
    batch = {
        "observation.images.cam_thermal": torch.linspace(20.0, 35.0, 16 * 16).reshape(
            1, 16, 16
        ),
    }

    images, img_masks = adapter._preprocess_images(batch)

    assert len(images) == 1
    assert images[0].shape == (1, 3, 224, 224)
    assert img_masks[0].tolist() == [True]
