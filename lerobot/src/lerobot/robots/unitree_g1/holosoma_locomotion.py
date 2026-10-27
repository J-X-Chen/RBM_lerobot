"""Compatibility import for the upstream controller module move."""

from .controllers.holosoma_locomotion import HolosomaLocomotionController, load_policy

__all__ = ["HolosomaLocomotionController", "load_policy"]
