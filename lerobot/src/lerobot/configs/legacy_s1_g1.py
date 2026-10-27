"""Load nonthermal checkpoints saved by the older S1/G1 forks."""

from typing import Any


def migrate_s1_g1_policy_config(config: dict[str, Any], policy_type: str | None = None) -> dict[str, Any]:
    """Discard inactive legacy thermal options, never silently change an active model."""
    if (policy_type or config.get("type")) not in {"pi0", "pi05"}:
        return config
    config = dict(config)
    switches = (
        "thermal_input_type",
        "thermal_encoder_channel",
        "use_thermal_encoder",
        "share_rgb_thermal_vit",
        "thermal_fuse_with_head_rgb",
        "mix_rgb_thermal",
        "rgb_thermal_align_fusion",
        "freeze_non_thermal_input",
    )
    disabled = {"", "0", "false", "none", "off", "no", "disable", "disabled"}
    for key in switches:
        value = config.get(key)
        if value is not None and str(value).strip().strip("\"'“”‘’").lower() not in disabled:
            raise ValueError(
                f"This checkpoint enables removed S1/G1 thermal functionality ({key}={value!r}). "
                "Use a nonthermal checkpoint with the shared LeRobot library."
            )
    # The old RGB-only fusion reused the thermal module's dropout setting.
    if config.get("rgb_input_type") in {"gatetworesvit", "GateTwoResViT", "withthreeblack"}:
        config.setdefault("rgb_threeblack_dropout", config.get("thermal_encoder_dropout", 0.0))
    return {
        key: value
        for key, value in config.items()
        if not key.startswith(("thermal_", "thermoact_", "rgb_thermal_")) and key not in switches
    }
