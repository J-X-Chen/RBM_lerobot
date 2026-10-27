"""Audit S1 numeric normalization against raw parquet; optionally repair dataset stats.

This tool never loads model weights, contacts the Hub, or changes checkpoints.
Exact quantiles require keeping the selected numeric columns in memory.
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors.numpy import load_file

FEATURES = ("observation.state", "action")
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
ARM_INDICES = list(range(7, 23))


def exact_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite numeric matrix.")
    stats = {
        "count": [len(values)],
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
    }
    for q, value in zip(QUANTILES, np.quantile(values, QUANTILES, axis=0), strict=True):
        stats[f"q{round(q * 100):02d}"] = value.tolist()
    return stats


def read_numeric(root: Path) -> dict[str, np.ndarray]:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise ValueError(f"No data parquet found in {root / 'data'}")
    parts = {key: [] for key in FEATURES}
    for path in files:
        table = pq.read_table(path, columns=list(FEATURES)).to_pydict()
        for key in FEATURES:
            parts[key].append(np.asarray(table[key], dtype=np.float64))
    return {key: np.concatenate(values) for key, values in parts.items()}


def read_pipeline(checkpoint: Path, filename: str) -> tuple[dict, dict]:
    config = json.loads((checkpoint / filename).read_text())
    stats = {}
    for step in config["steps"]:
        if step["registry_name"] in {"normalizer_processor", "unnormalizer_processor"}:
            if "state_file" not in step:
                raise ValueError(f"Missing saved normalizer state in {filename}")
            stats.update(load_file(str(checkpoint / step["state_file"])))
    return config, stats


def audit_checkpoint(checkpoint: Path, values: dict[str, np.ndarray]) -> dict:
    config = json.loads((checkpoint / "config.json").read_text())
    pre_config, pre = read_pipeline(checkpoint, "policy_preprocessor.json")
    post_config, post = read_pipeline(checkpoint, "policy_postprocessor.json")
    clip = next(
        (s["config"] for s in pre_config["steps"]
         if s["registry_name"] == "dm05_clip_normalized_processor"), {}
    )
    if config.get("use_relative_actions", False):
        raise ValueError("This S1 audit currently requires absolute-action checkpoints.")
    if any(config["normalization_mapping"].get(k) != "QUANTILES" for k in ("STATE", "ACTION")):
        raise ValueError("This audit currently requires QUANTILES state/action normalization.")
    report = {
        "checkpoint": str(checkpoint),
        "policy_type": config["type"],
        "normalization_mapping": config["normalization_mapping"],
        "preprocessor": pre_config,
        "postprocessor": post_config,
        "features": {},
    }
    for key in FEATURES:
        dim = len(pre[f"{key}.q01"].reshape(-1))
        data = values[key]
        if dim == 16 and data.shape[1] == 25:
            indices = ARM_INDICES
        elif dim == data.shape[1]:
            indices = list(range(dim))
        else:
            raise ValueError(f"Unsupported {key} dimension mapping: {data.shape[1]} -> {dim}")
        data = data[:, indices]
        actual = exact_stats(data)
        low, high = (pre[f"{key}.{q}"].reshape(-1).astype(np.float64) for q in ("q01", "q99"))
        denom = high - low
        eps = next(s["config"].get("eps", 1e-8) for s in pre_config["steps"]
                   if s["registry_name"] == "normalizer_processor")
        denom = np.where(denom == 0, eps, denom)
        normalized = 2 * (data - low) / denom - 1
        clipped = clip.get("clip_action" if key == "action" else "clip_state", False)
        restored = (np.clip(normalized, -1, 1) + 1) * denom / 2 + low
        saved = {stat: pre[f"{key}.{stat}"].reshape(-1).tolist()
                 for stat in actual if f"{key}.{stat}" in pre}
        item = {
            "dataset_indices": indices,
            "exact_stats": actual,
            "saved_stats": saved,
            "saved_minus_exact": {
                stat: (np.asarray(saved[stat]) - actual[stat]).tolist() for stat in saved
            },
            "outside_q01_q99_fraction": ((normalized < -1) | (normalized > 1)).mean(0).tolist(),
            "normalized_min": normalized.min(0).tolist(),
            "normalized_max": normalized.max(0).tolist(),
            "training_clip_enabled": clipped,
            "clip_roundtrip_max_abs_error": np.abs(restored - data).max(0).tolist() if clipped else None,
        }
        if key == "action":
            item["pre_post_stats_match"] = all(
                f"{key}.{stat}" in post
                and np.array_equal(pre[f"{key}.{stat}"], post[f"{key}.{stat}"])
                for stat in saved
            )
        report["features"][key] = item
    return report


def repair_stats(root: Path, numeric_stats: dict) -> Path:
    """Replace numeric stats atomically, retaining other features and an exclusive backup."""
    path = root / "meta/stats.json"
    existing = json.loads(path.read_text())
    updated = {**existing, **numeric_stats}
    # Never overwrite an earlier backup, including on repeated repairs.
    fd, backup_name = tempfile.mkstemp(prefix="stats.before_exact.", suffix=".json", dir=path.parent)
    os.close(fd)
    backup = Path(backup_name)
    shutil.copy2(path, backup)
    fd, temporary = tempfile.mkstemp(prefix=".stats.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(updated, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repair-dataset-stats", action="store_true")
    args = parser.parse_args()
    values = read_numeric(args.root)
    exact = {key: exact_stats(array) for key, array in values.items()}
    stored = json.loads((args.root / "meta/stats.json").read_text())
    report = {
        "dataset_root": str(args.root),
        "dataset": {
            key: {"exact_stats": exact[key], "saved_stats": stored[key],
                  "saved_minus_exact": {
                      stat: (np.asarray(stored[key][stat]) - value).tolist()
                      for stat, value in exact[key].items() if stat in stored[key]
                  }}
            for key in FEATURES
        },
        "checkpoints": [audit_checkpoint(p, values) for p in args.checkpoint],
        "checkpoints_modified": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    # Finish every audit before writing dataset metadata.
    if args.repair_dataset_stats:
        report["dataset_stats_backup"] = str(repair_stats(args.root, exact))
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Audit written to {args.report}")
    if args.repair_dataset_stats:
        print(f"Numeric dataset stats repaired; backup: {report['dataset_stats_backup']}")


if __name__ == "__main__":
    main()
