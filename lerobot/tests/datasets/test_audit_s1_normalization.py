import json

import numpy as np
import unittest
import tempfile
from pathlib import Path
from safetensors.numpy import save_file

from lerobot.scripts.audit_s1_normalization import audit_checkpoint, exact_stats, repair_stats


def check_global_quantiles_do_not_average_episode_quantiles():
    values = np.concatenate([np.zeros((800, 2)), np.full((200, 2), 40.0)])
    stats = exact_stats(values)
    assert stats["q99"] == [40.0, 40.0]
    assert stats["mean"] == [8.0, 8.0]
    with unittest.TestCase().assertRaises(ValueError):
        exact_stats(np.full((2, 2), np.nan))


def check_repair_preserves_images_and_original_backup(tmp_path):
    (tmp_path / "meta").mkdir()
    original = {"action": {"q99": [16]}, "observation.images.head": {"mean": [0.5]}}
    path = tmp_path / "meta/stats.json"
    path.write_text(json.dumps(original))
    stats = {"action": exact_stats(np.array([[0], [40]]))}
    backup = repair_stats(tmp_path, stats)
    assert json.loads(backup.read_text()) == original
    assert json.loads(path.read_text())["observation.images.head"] == original["observation.images.head"]
    assert json.loads(path.read_text())["action"]["q99"] == [39.6]
    second = repair_stats(tmp_path, stats)
    assert backup != second
    assert json.loads(backup.read_text()) == original


def check_audit_detects_dm05_clipping_and_processor_mismatch(tmp_path):
    config = {"type": "dm05", "normalization_mapping": {"STATE": "QUANTILES", "ACTION": "QUANTILES"}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    stats = {}
    for key in ("action", "observation.state"):
        for name, value in exact_stats(np.tile([0.0, 16.0], (25, 1)).T).items():
            stats[f"{key}.{name}"] = np.asarray(value, dtype=np.float32)
    save_file(stats, str(tmp_path / "pre.safetensors"))
    save_file(stats, str(tmp_path / "post.safetensors"))
    for filename, registry, state_file in [
        ("policy_preprocessor.json", "normalizer_processor", "pre.safetensors"),
        ("policy_postprocessor.json", "unnormalizer_processor", "post.safetensors"),
    ]:
        steps = [{"registry_name": registry, "config": {}, "state_file": state_file}]
        steps.append({"registry_name": "dm05_clip_normalized_processor",
                      "config": {"clip_state": True, "clip_action": True}})
        (tmp_path / filename).write_text(json.dumps({"steps": steps}))
    data = {key: np.tile([0.0, 40.0], (25, 1)).T for key in ("action", "observation.state")}
    report = audit_checkpoint(tmp_path, data)
    action = report["features"]["action"]
    assert action["pre_post_stats_match"]
    assert min(action["clip_roundtrip_max_abs_error"]) > 20
    stats["action.q99"] += 1
    save_file(stats, str(tmp_path / "post.safetensors"))
    assert not audit_checkpoint(tmp_path, data)["features"]["action"]["pre_post_stats_match"]


class AuditNormalizationTest(unittest.TestCase):
    def test_quantiles(self):
        check_global_quantiles_do_not_average_episode_quantiles()

    def test_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            check_repair_preserves_images_and_original_backup(Path(directory))

    def test_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            check_audit_detects_dm05_clipping_and_processor_mismatch(Path(directory))


if __name__ == "__main__":
    unittest.main()
