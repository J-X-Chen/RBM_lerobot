"""Compatibility import for the upstream controller module move."""

from .controllers.gr00t_locomotion import GrootLocomotionController, load_groot_policies

__all__ = ["GrootLocomotionController", "load_groot_policies"]
