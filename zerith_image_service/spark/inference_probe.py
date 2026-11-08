#!/usr/bin/env python3
"""Foreground inference probe: isolates the policy call from the HTTP server.

Loads the checkpoint, runs one plain inference, then one with fixed noise, and
dumps a Python traceback if any call exceeds the watchdog.
"""

from __future__ import annotations

import faulthandler
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pi05_h1_pro.json"


def main() -> int:
    doc = json.loads(CONFIG.read_text(encoding="utf-8"))
    checkpoint = doc["checkpoint"]
    device = doc["device"]

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    started = time.perf_counter()
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_cfg.device = device
    policy_cfg.gradient_checkpointing = False
    policy = get_policy_class(policy_cfg.type).from_pretrained(
        checkpoint, config=policy_cfg
    )
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=checkpoint,
        dataset_stats=None,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    print(f"loaded in {time.perf_counter() - started:.1f}s", flush=True)

    rng = np.random.default_rng(0)
    state = torch.zeros(1, 23, device=device)
    batch = {"observation.state": state, "task": doc["task"]}
    for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        image = torch.from_numpy(
            (rng.random((480, 640, 3)) * 255).astype(np.uint8)
        ).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(device)
        batch[f"observation.images.{key}"] = image

    def run(tag: str) -> None:
        faulthandler.dump_traceback_later(90, exit=True)
        started = time.perf_counter()
        with torch.no_grad():
            processed = preprocessor(batch)
            chunk = policy.predict_action_chunk(processed)
            actions = postprocessor(chunk)
        faulthandler.cancel_dump_traceback_later()
        array = actions.detach().cpu().numpy()
        print(
            f"{tag}: {time.perf_counter() - started:.3f}s shape={array.shape} "
            f"first4={np.round(array[0, 0, :4], 4).tolist()}",
            flush=True,
        )

    run("plain")

    shape = (1, int(doc["chunk_size"]), 32)
    generator = torch.Generator(device="cpu").manual_seed(int(doc["fixed_noise_seed"]))
    cached = torch.randn(*shape, generator=generator).to(device)
    policy.model.sample_noise = lambda *_a, **_k: cached
    run("fixed-noise")
    run("fixed-noise-again")

    # Reproduce the server condition: the request is answered on a worker
    # thread, not the thread that loaded the model.
    print("\n--- worker thread (server condition) ---", flush=True)
    result: dict = {}

    fixed = cached

    def worker() -> None:
        try:
            faulthandler.dump_traceback_later(75, exit=True)
            started = time.perf_counter()
            with torch.no_grad():
                processed = preprocessor(batch)
                kwargs = {} if doc.get("fixed_noise_seed") is None else {"noise": fixed}
                chunk = policy.predict_action_chunk(processed, **kwargs)
                actions = postprocessor(chunk)
            faulthandler.cancel_dump_traceback_later()
            result["array"] = actions.detach().cpu().numpy()
            result["seconds"] = time.perf_counter() - started
        except Exception as error:  # noqa: BLE001
            result["error"] = repr(error)

    thread = threading.Thread(target=worker, name="pi05-worker")
    thread.start()
    thread.join(timeout=70)
    if thread.is_alive():
        print("WORKER STILL RUNNING after 70s", flush=True)
    if "error" in result:
        print("worker error:", result["error"], flush=True)
    if "array" in result:
        print(
            f"worker: {result['seconds']:.3f}s shape={result['array'].shape} "
            f"first4={np.round(result['array'][0, 0, :4], 4).tolist()}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
