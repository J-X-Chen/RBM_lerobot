"""A requested pretrained policy must never fall back to random parameters."""

import importlib

import pytest
import torch
from safetensors.torch import save_file
from torch import nn


@pytest.mark.parametrize("name", ["pi0", "pi05"])
@pytest.mark.parametrize("failure", ["missing", "corrupt", "incompatible", "valid"])
def test_pretrained_load_failures_propagate(tmp_path, monkeypatch, name, failure):
    module = importlib.import_module(f"lerobot.policies.{name}.modeling_{name}")
    config_module = importlib.import_module(f"lerobot.policies.{name}.configuration_{name}")
    policy_cls = getattr(module, f"{name.upper()}Policy")
    config_cls = getattr(config_module, f"{name.upper()}Config")

    def init(self, config, **kwargs):
        nn.Module.__init__(self)
        self.config = config
        self.model = nn.Linear(2, 2)

    monkeypatch.setattr(policy_cls, "__init__", init)
    # Keep this test about the loader, without allocating the full VLM.
    if name == "pi0":
        monkeypatch.setattr(policy_cls, "_fix_pytorch_state_dict_keys", lambda self, state, config: state)
    else:
        monkeypatch.setattr(policy_cls, "_convert_openpi_state_dict", lambda self, state: state)
    if failure == "corrupt":
        (tmp_path / "model.safetensors").write_bytes(b"not a safetensors checkpoint")
    elif failure == "incompatible":
        save_file({"model.weight": torch.zeros(3, 3)}, tmp_path / "model.safetensors")
    elif failure == "valid":
        save_file(
            {"model.weight": torch.full((2, 2), 3.0), "model.bias": torch.ones(2)},
            tmp_path / "model.safetensors",
        )
        policy = policy_cls.from_pretrained(tmp_path, config=config_cls(device="cpu"), local_files_only=True)
        torch.testing.assert_close(policy.model.weight, torch.full((2, 2), 3.0))
        torch.testing.assert_close(policy.model.bias, torch.ones(2))
        return
    with pytest.raises(RuntimeError, match="refusing to continue"):
        policy_cls.from_pretrained(tmp_path, config=config_cls(device="cpu"), local_files_only=True)


def test_pi0_forwards_download_options(monkeypatch, tmp_path):
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    def init(self, config, **kwargs):
        nn.Module.__init__(self)
        self.config = config
        self.model = nn.Linear(2, 2)

    calls = []

    def resolve(*args, **kwargs):
        calls.append(kwargs)
        raise OSError("offline test")

    monkeypatch.setattr(PI0Policy, "__init__", init)
    monkeypatch.setattr("transformers.utils.cached_file", resolve)
    with pytest.raises(RuntimeError):
        PI0Policy.from_pretrained(
            "test/checkpoint",
            config=PI0Config(device="cpu"),
            revision="specific-revision",
            cache_dir=tmp_path,
            local_files_only=True,
        )
    assert calls[0]["revision"] == "specific-revision"
    assert calls[0]["cache_dir"] == tmp_path
    assert calls[0]["local_files_only"] is True
