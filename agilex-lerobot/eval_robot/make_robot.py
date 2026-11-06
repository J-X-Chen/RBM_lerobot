"""ALOHA hardware factory. No Unitree/S1 dependencies."""

from robot_control.piper_bimanual import ArmConfig, PiperBimanual


def make_robot(hardware_config, **kwargs):
    return PiperBimanual(ArmConfig(**hardware_config["left"]), ArmConfig(**hardware_config["right"]), **kwargs)
