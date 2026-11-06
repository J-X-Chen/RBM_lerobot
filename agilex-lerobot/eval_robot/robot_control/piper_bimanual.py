"""Local CAN compatibility entry; robot-side controller is shared with the service."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "agilex_image_service"))
from piper_control import ARM_INDICES, RAD_TO_MDEG, ArmConfig, PiperBimanual, vector

__all__ = ["ARM_INDICES", "RAD_TO_MDEG", "ArmConfig", "PiperBimanual", "vector"]
