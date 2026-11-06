"""Shared trained-checkpoint inference through the real ALOHA evaluator."""

from pathlib import Path

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tests.policies.test_cobot_magic_training import test_cobot_training_roundtrip as build_checkpoint


@pytest.mark.parametrize("policy_type", ["pi05", "dm05"])
def test_aloha_evaluator_checkpoint(tmp_path, monkeypatch, policy_type):
    build_checkpoint(tmp_path, monkeypatch, policy_type)
    evaluation_root = Path(__file__).resolve().parents[3] / "agilex-lerobot/eval_robot"
    monkeypatch.syspath_prepend(str(evaluation_root))
    from aloha_policy import AlohaPolicy

    runner = AlohaPolicy(str(tmp_path / "checkpoint"), device="cpu", n_action_steps=2)
    dataset = LeRobotDataset("local/test", root=tmp_path / "data", video_backend="pyav")
    sample = dataset[0]
    actions = runner.predict_chunk(sample, sample["task"])
    assert actions.shape == (2, 14) and np.isfinite(actions).all()
    runner.reset()
    assert runner.action(sample, sample["task"]).shape == (14,)
    assert len(runner.queue) == 1
    assert runner.action(sample, sample["task"]).shape == (14,)
    assert len(runner.queue) == 0


def test_aloha_warmup_discards_actions_even_on_failure(monkeypatch):
    from collections import deque
    from types import SimpleNamespace

    evaluation_root = Path(__file__).resolve().parents[3] / "agilex-lerobot/eval_robot"
    monkeypatch.syspath_prepend(str(evaluation_root))
    from aloha_policy import AlohaPolicy

    runner = object.__new__(AlohaPolicy)
    runner.queue = deque(["old action"])
    resets = []
    runner.policy = SimpleNamespace(reset=lambda: resets.append("policy"))
    runner.pre = SimpleNamespace(reset=lambda: resets.append("pre"))
    runner.post = SimpleNamespace(reset=lambda: resets.append("post"))
    calls = []

    def predict(observation, task):
        calls.append(task)
        runner.queue.append("warmup action")

    monkeypatch.setattr(runner, "predict_chunk", predict)
    runner.warmup({}, "pick", steps=3)
    assert calls == ["pick"] * 3
    assert len(runner.queue) == 0
    assert resets == ["policy", "pre", "post"] * 2

    def fail(*args):
        runner.queue.append("invalid action")
        raise RuntimeError("warmup failed")

    monkeypatch.setattr(runner, "predict_chunk", fail)
    with pytest.raises(RuntimeError, match="warmup failed"):
        runner.warmup({}, "pick")
    assert len(runner.queue) == 0


def test_aloha_cli_records_effective_run_config(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    evaluation_root = Path(__file__).resolve().parents[3] / "agilex-lerobot/eval_robot"
    monkeypatch.syspath_prepend(str(evaluation_root))
    import eval_aloha

    runner = SimpleNamespace(
        policy=SimpleNamespace(
            config=SimpleNamespace(
                type="dm05",
                action_feature_names=["left.joint1"],
                image_features={"observation.images.head": None},
            )
        ),
        n_action_steps=4,
    )
    monkeypatch.setattr(eval_aloha, "AlohaPolicy", lambda *args: runner)
    calls = []
    monkeypatch.setattr(eval_aloha, "offline", lambda args, model: calls.append(args.mode))
    log = tmp_path / "actions.jsonl"
    eval_aloha.main(["--policy.path=local-checkpoint", "--device=cpu", f"--log={log}"])
    metadata = json.loads(log.with_suffix(".run.json").read_text())
    assert metadata["n_action_steps"] == 4
    assert metadata["arguments"]["policy_warmup_steps"] == 5
    assert "--device=cpu" in metadata["command"]
    assert calls == ["dataset"]
