#!/usr/bin/env python3
"""Offline validation for the LeRobot pi0.5 task6 checkpoint.

Loads the frozen checkpoint and runs one real inference on a dataset frame.
No robot, no ROS, no network.

    /home/xieshiyun/miniforge3/envs/evo-rl-gpu/bin/python \
        pi05_deploy/spark/offline_check.py --frame 0
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pi05_h1_pro.json"
DEFAULT_DATASET = Path(
    "/home/xieshiyun/lingbot_workspace/validation/mini_lerobot_20260814/task6"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-json", type=Path, default=None)
    return parser.parse_args()


def load_frame(dataset: Path, index: int) -> dict:
    import pyarrow.parquet as pq

    parquet = dataset / "data" / "chunk-000" / "file-000.parquet"
    columns = [
        "observation.state",
        "action",
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    table = pq.read_table(parquet, columns=columns).to_pydict()
    total = len(table["observation.state"])
    if not 0 <= index < total:
        raise SystemExit(f"frame {index} out of range 0..{total - 1}")

    def image(key: str) -> np.ndarray:
        payload = table[key][index]["bytes"]
        return np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))

    return {
        "index": index,
        "total_frames": total,
        "state": np.asarray(table["observation.state"][index], dtype=np.float64),
        "action": np.asarray(table["action"][index], dtype=np.float64),
        "cam_high": image("observation.images.cam_high"),
        "cam_left_wrist": image("observation.images.cam_left_wrist"),
        "cam_right_wrist": image("observation.images.cam_right_wrist"),
    }


def to_image_tensor(image: np.ndarray, device: str) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
    return tensor.unsqueeze(0).to(device)


def main() -> int:
    args = parse_args()
    config_doc = json.loads(args.config.read_text(encoding="utf-8"))
    device = args.device or config_doc.get("device", "cuda")
    checkpoint = Path(config_doc["checkpoint"])
    if not (checkpoint / "config.json").is_file():
        raise SystemExit(f"checkpoint config missing under {checkpoint}")

    frame = load_frame(args.dataset, args.frame)
    print(f"frame={frame['index']}/{frame['total_frames']} device={device}")
    for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        print(f"  image {key}: {frame[key].shape} {frame[key].dtype}")
    print(f"  state : {np.round(frame['state'], 4).tolist()}")
    print(f"  action: {np.round(frame['action'], 4).tolist()}")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    t0 = time.perf_counter()
    policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    policy_cfg.device = device
    policy_class = get_policy_class(policy_cfg.type)
    policy = policy_class.from_pretrained(str(checkpoint), config=policy_cfg)
    policy.to(device)
    policy.eval()
    print(f"loaded policy type={policy_cfg.type} in {time.perf_counter() - t0:.1f}s")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        dataset_stats=None,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    batch = {
        "observation.state": torch.from_numpy(frame["state"])
        .float()
        .unsqueeze(0)
        .to(device),
        "observation.images.cam_high": to_image_tensor(frame["cam_high"], device),
        "observation.images.cam_left_wrist": to_image_tensor(
            frame["cam_left_wrist"], device
        ),
        "observation.images.cam_right_wrist": to_image_tensor(
            frame["cam_right_wrist"], device
        ),
        "task": config_doc["task"],
    }

    with torch.no_grad():
        processed = preprocessor(batch)
        t1 = time.perf_counter()
        chunk = policy.predict_action_chunk(processed)
        latency = time.perf_counter() - t1
        actions = postprocessor(chunk)

    actions = actions.detach().to("cpu", dtype=torch.float32).numpy()
    print(f"\nraw chunk shape={tuple(chunk.shape)} latency={latency:.3f}s")
    print(f"postprocessed shape={actions.shape}")
    if actions.ndim == 3:
        actions = actions[0]
    print(f"action chunk = {actions.shape[0]} x {actions.shape[1]}")
    print(f"finite={bool(np.isfinite(actions).all())}")
    print(f"global min={actions.min():+.4f} max={actions.max():+.4f}")
    print("\nper-dim (policy 23D order):")
    for i, name in enumerate(config_doc["model_23d_order"]):
        column = actions[:, i]
        print(
            f"  [{i:2d}] {name:18s} first={column[0]:+.4f} "
            f"min={column.min():+.4f} max={column.max():+.4f} "
            f"delta={column.max() - column.min():+.4f}"
        )

    if args.save_json:
        payload = {
            "checkpoint": str(checkpoint),
            "frame": frame["index"],
            "device": device,
            "chunk_shape": list(actions.shape),
            "finite": bool(np.isfinite(actions).all()),
            "latency_s": latency,
            "state": frame["state"].tolist(),
            "dataset_action": frame["action"].tolist(),
            "predicted_first_action": actions[0].tolist(),
            "predicted_chunk": actions.tolist(),
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.save_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
