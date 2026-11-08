#!/usr/bin/env python3
"""pi0.5 (LeRobot) policy server for the ZERITH H1 Pro deployment.

Loads the frozen pi05 checkpoint once and serves 50x23 action chunks over
HTTP/JSON. The robot-side bridge posts one observation and gets one chunk.

    .venv-pi05/bin/python pi05_deploy/spark/policy_server.py --host 172.31.200.211 --port 8790

Request body for /v1/pi05/observation:

    {
      "task": "Place different drinks in the basket.",
      "state": [23 floats],
      "images": {"cam_high": "<base64 jpeg>", "cam_left_wrist": "...", "cam_right_wrist": "..."}
    }

Response:

    {"actions": [[23 floats] x chunk_size], "latency_s": float, "chunk_shape": [50, 23]}
"""

from __future__ import annotations

import argparse
import atexit
import base64
import faulthandler
import io
import json
import signal
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pi05_h1_pro.json"


def _strict_optional_int(value) -> int | None:
    """Return an integral JSON value without accepting bool or lossy floats."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def resolve_rtc_leftover_window(
    *,
    has_previous_chunk: bool,
    previous_token: int | None,
    current_token: int,
    raw_start: int | None,
    inference_delay: int | None,
    requested_execution_horizon: int,
    chunk_size: int,
) -> tuple[int | None, int | None, str | None]:
    """Validate the bridge-provided absolute raw index and cap padding exposure."""
    if not has_previous_chunk:
        return None, None, "first_chunk"
    if previous_token != current_token:
        return None, None, "previous_chunk_token_mismatch"
    if raw_start is None:
        return None, None, "missing_rtc_leftover_raw_start"
    if inference_delay is None:
        return None, None, "missing_inference_delay"
    if raw_start < 0:
        return None, None, "rtc_leftover_raw_start_negative"
    if raw_start >= chunk_size:
        return None, None, "rtc_leftover_raw_start_out_of_range"
    if inference_delay < 0:
        return None, None, "inference_delay_negative"

    valid_leftover_steps = chunk_size - raw_start
    # The [0, inference_delay) prefix has weight 1. If it reaches padding,
    # there is no way to represent the required old trajectory without inventing
    # actions, so guidance must be skipped for this request.
    if inference_delay >= valid_leftover_steps:
        return None, None, "insufficient_valid_leftover_for_prefix"

    effective_execution_horizon = min(
        requested_execution_horizon,
        valid_leftover_steps,
    )
    if effective_execution_horizon <= inference_delay:
        return None, None, "no_valid_rtc_soft_guidance_window"
    return raw_start, effective_execution_horizon, None


class PolicyRuntime:
    def __init__(
        self,
        config_path: Path,
        device: str | None = None,
        denoise_steps: int | None = None,
        fixed_noise_seed: int | None = None,
        checkpoint_override: Path | None = None,
    ):
        self.doc = json.loads(config_path.read_text(encoding="utf-8"))
        self.device = device or self.doc.get("device", "cuda")
        self.denoise_steps = denoise_steps
        if fixed_noise_seed is None:
            fixed_noise_seed = self.doc.get("fixed_noise_seed")
        self.fixed_noise_seed = fixed_noise_seed
        self.checkpoint = Path(checkpoint_override or self.doc["checkpoint"])
        self.chunk_size = int(self.doc["chunk_size"])
        self.task = self.doc["task"]
        self.lock = threading.Lock()
        self.calls = 0
        self.failures = 0
        self.last_latency_s = 0.0
        self.last_raw_chunk = None
        self.rtc_state_token = 0
        self._load()

    def _load(self) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        started = time.perf_counter()
        policy_cfg = PreTrainedConfig.from_pretrained(str(self.checkpoint))
        policy_cfg.device = self.device
        policy_cfg.gradient_checkpointing = False
        # Defaults to False, so behaviour is unchanged unless a config sets it.
        # torch.compile keeps the model's own sample_actions path; enabling it
        # means the FIRST predict_action_chunk pays the inductor cost, so a
        # warmup call is required before the server is trusted for control.
        policy_cfg.compile_model = bool(self.doc.get("compile_model", False))
        if self.doc.get("compile_mode"):
            policy_cfg.compile_mode = str(self.doc["compile_mode"])
        if self.denoise_steps is not None:
            policy_cfg.num_inference_steps = int(self.denoise_steps)
        self.rtc_config = None
        rtc_doc = self.doc.get("rtc") or {}
        if rtc_doc.get("enabled"):
            from lerobot.configs.types import RTCAttentionSchedule
            from lerobot.policies.rtc.configuration_rtc import RTCConfig

            rtc = RTCConfig(
                enabled=True,
                execution_horizon=int(rtc_doc.get("execution_horizon", 10)),
                prefix_attention_schedule=RTCAttentionSchedule(
                    str(rtc_doc.get("prefix_attention_schedule", "LINEAR"))
                ),
                max_guidance_weight=float(rtc_doc.get("max_guidance_weight", 10.0)),
            )
            # Must be set before from_pretrained: PI05Policy.__init__ builds the
            # RTC processor from config.rtc_config.
            policy_cfg.rtc_config = rtc
            self.rtc_config = rtc
        # Do NOT enable torch.compile here. With compile_model=True the first
        # request pays the inductor cost *inside the HTTP worker thread*; the
        # inductor compile-worker pool does not complete there and the request
        # wedges (observed 2026-10-16: every request after load timed out and
        # the process eventually died). Warm inference without compilation is
        # ~0.32 s, well inside the 0.83 s replan budget, so compilation buys
        # nothing for this deployment.
        self.policy = get_policy_class(policy_cfg.type).from_pretrained(
            str(self.checkpoint), config=policy_cfg
        )
        self.policy.to(self.device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(self.checkpoint),
            dataset_stats=None,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        print(
            f"[pi05-server] policy={policy_cfg.type} device={self.device} "
            f"chunk={self.chunk_size} denoise={policy_cfg.num_inference_steps} "
            f"compile_model={policy_cfg.compile_model} "
            f"rtc={None if self.rtc_config is None else self.rtc_config.execution_horizon} "
            f"fixed_noise_seed={self.fixed_noise_seed} "
            f"load_s={time.perf_counter() - started:.1f}",
            flush=True,
        )
        self.fixed_noise = None
        if self.fixed_noise_seed is not None:
            # pi0.5 draws its flow-matching noise from the global torch RNG on
            # every call, so every replan produced a different plan and the arm
            # wobbled at the replan rate. Reseeding the global RNG is not an
            # option (torch.manual_seed re-initialises every CUDA device RNG and
            # this server answers on worker threads), and monkeypatching
            # sample_noise makes dynamo re-trace inside the worker thread and
            # stall. sample_actions accepts the noise as a plain argument, so
            # it is drawn once here and reused for every request.
            shape = (1, self.chunk_size, 32)
            generator = torch.Generator(device="cpu").manual_seed(
                int(self.fixed_noise_seed)
            )
            self.fixed_noise = torch.randn(
                *shape, generator=generator, dtype=torch.float32
            ).to(self.device)
            print(
                f"[pi05-server] fixed noise shape={shape} "
                f"seed={self.fixed_noise_seed}",
                flush=True,
            )
        if policy_cfg.compile_model:
            self._align_attention_mask_dtype(policy_cfg.dtype)
            if self.doc.get("compile_warmup", True):
                self._warmup()

    @staticmethod
    def _align_attention_mask_dtype(dtype_name: str) -> None:
        """Make the additive attention mask match the query dtype.

        ``PI05Pytorch._prepare_attention_masks_4d`` builds the mask with

            torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

        The Python scalars force float32 while the model runs bfloat16. Eager
        SDPA tolerates the mismatch; inductor lowers the attention to
        ``_scaled_dot_product_efficient_attention``, which rejects it:

            RuntimeError: invalid dtype for bias - should match query's dtype
            (observed 2026-10-16, buffer (1, 8, 968, 968))

        The patch is applied to this process only - the shared lerobot checkout
        under /home/xieshiyun/workspace/evo-rl is left untouched, so the
        Pro6000 training path and any other session are unaffected.
        """
        from lerobot.policies.pi05 import modeling_pi05
        from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE

        dtype = {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(str(dtype_name), torch.float32)

        def prepare(self, att_2d_masks):  # noqa: ANN001
            mask = att_2d_masks[:, None, :, :]
            return torch.where(mask, 0.0, OPENPI_ATTENTION_MASK_VALUE).to(dtype)

        modeling_pi05.PI05Pytorch._prepare_attention_masks_4d = prepare
        print(
            f"[pi05-server] attention mask dtype pinned to {dtype} for torch.compile",
            flush=True,
        )

    def _warmup(self) -> None:
        """Compile inductor on the MAIN thread before serving.

        With compile_model=True the first call builds the inductor graph. Doing
        that from an HTTP worker thread wedges the request forever: the
        inductor compile-worker pool does not complete there (observed
        2026-10-16 - every request timed out after load and the process died).
        Compiling during startup, before serve_forever(), keeps every request on
        the already-compiled path.

        The dummy observation uses exactly the shapes real requests use:
        batch 1, 23 state dims, three 480x640 RGB frames, chunk 50 x 32.
        """
        batch = {
            "observation.state": torch.zeros(
                1, 23, dtype=torch.float32, device=self.device
            ),
            "task": self.task,
        }
        for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            frame = np.full((480, 640, 3), 128, dtype=np.uint8)
            tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div_(255.0)
            batch[f"observation.images.{key}"] = tensor.unsqueeze(0).to(self.device)
        started = time.perf_counter()
        with self.lock, torch.no_grad():
            processed = self.preprocessor(batch)
            kwargs = {} if self.fixed_noise is None else {"noise": self.fixed_noise}
            # Warm the RTC variant first. predict_action_chunk forwards **kwargs
            # into sample_actions, and dynamo guards on the keyword set: the
            # plain call has only `noise`, while RTC adds the previous chunk and
            # fixed-shape weights. Delay/EH values live only in the weight tensor,
            # so changing them does not require another graph. The first real
            # request has no leftover and must reuse the plain graph, so build
            # that variant last.
            if self.rtc_config is not None:
                left = torch.zeros(
                    1, self.chunk_size, 23, dtype=torch.float32, device=self.device
                )
                rtc_kwargs = dict(kwargs)
                rtc_kwargs.update(
                    prev_chunk_left_over=left,
                    rtc_prefix_weights=self._build_rtc_prefix_weights(
                        inference_delay=14,
                        execution_horizon=39,
                    ),
                )
                self.postprocessor(
                    self.policy.predict_action_chunk(processed, **rtc_kwargs)
                )
                print("[pi05-server] RTC tensor-weight branch warmed before plain", flush=True)
            chunk = self.policy.predict_action_chunk(processed, **kwargs)
            self.postprocessor(chunk)
        print(
            f"[pi05-server] warmup done in {time.perf_counter() - started:.1f}s "
            f"(inductor compilation happens here, not on the first request)",
            flush=True,
        )

    def _build_rtc_prefix_weights(
        self,
        *,
        inference_delay: int,
        execution_horizon: int,
    ) -> torch.Tensor:
        """Build fixed-shape RTC weights before entering the compiled model."""
        if self.rtc_config is None or self.policy.rtc_processor is None:
            raise RuntimeError("RTC prefix weights requested while RTC is disabled")
        weights = self.policy.rtc_processor.get_prefix_weights(
            inference_delay,
            execution_horizon,
            self.chunk_size,
        )
        return weights.to(
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, self.chunk_size, 1)

    @staticmethod
    def _image(payload: str, key: str) -> torch.Tensor:
        raw = base64.b64decode(payload)
        array = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
        # PIL hands back a read-only array. torch.from_numpy keeps that view
        # read-only, and the in-place div_ below would then write through a
        # read-only buffer: undefined behaviour that killed the server process
        # mid-request. Copy first so the tensor owns writable memory.
        tensor = torch.from_numpy(np.ascontiguousarray(array).copy())
        tensor = tensor.permute(2, 0, 1).float().div_(255.0)
        return tensor.unsqueeze(0)

    def infer(self, body: dict) -> dict:
        state = np.asarray(body["state"], dtype=np.float32)
        if state.shape != (23,):
            raise ValueError(f"state must be 23 values, got {state.shape}")
        images = body["images"]
        task = str(body.get("task") or self.task)

        batch = {
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(self.device),
            "task": task,
        }
        for key in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            if key not in images:
                raise ValueError(f"missing image {key}")
            batch[f"observation.images.{key}"] = self._image(images[key], key).to(
                self.device
            )

        with self.lock, torch.no_grad():
            processed = self.preprocessor(batch)
            started = time.perf_counter()
            kwargs = {} if self.fixed_noise is None else {"noise": self.fixed_noise}
            rtc_info = {
                "rtc_applied": False,
                "rtc_skip_reason": None,
                "rtc_leftover_raw_start": None,
                "rtc_valid_leftover_steps": None,
                "rtc_requested_execution_horizon": None,
                "rtc_effective_execution_horizon": None,
            }
            if self.rtc_config is not None:
                # RTC compares against the *un-postprocessed* chunk, so the
                # left-over is kept here rather than round-tripped through the
                # bridge. The bridge sends an absolute row index in the prior
                # raw chunk, already including the prior execution crop start.
                requested_execution_horizon = _strict_optional_int(
                    body.get("execution_horizon")
                )
                if requested_execution_horizon is None:
                    requested_execution_horizon = self.rtc_config.execution_horizon
                raw_start, effective_execution_horizon, skip_reason = (
                    resolve_rtc_leftover_window(
                        has_previous_chunk=self.last_raw_chunk is not None,
                        previous_token=_strict_optional_int(
                            body.get("rtc_previous_chunk_token")
                        ),
                        current_token=self.rtc_state_token,
                        raw_start=_strict_optional_int(
                            body.get("rtc_leftover_raw_start")
                        ),
                        inference_delay=_strict_optional_int(
                            body.get("inference_delay")
                        ),
                        requested_execution_horizon=requested_execution_horizon,
                        chunk_size=self.chunk_size,
                    )
                )
                rtc_info["rtc_skip_reason"] = skip_reason
                rtc_info["rtc_leftover_raw_start"] = raw_start
                rtc_info["rtc_requested_execution_horizon"] = (
                    requested_execution_horizon
                )
                if raw_start is not None:
                    valid_leftover_steps = self.chunk_size - raw_start
                    rtc_info["rtc_valid_leftover_steps"] = valid_leftover_steps
                    rtc_info["rtc_effective_execution_horizon"] = (
                        effective_execution_horizon
                    )
                if skip_reason is not None and skip_reason != "first_chunk":
                    print(
                        f"[pi05-server] RTC guidance skipped: {skip_reason} "
                        f"raw_start={body.get('rtc_leftover_raw_start')} "
                        f"delay={body.get('inference_delay')} "
                        f"requested_eh={requested_execution_horizon}",
                        flush=True,
                    )
                if skip_reason is None:
                    assert raw_start is not None
                    assert effective_execution_horizon is not None
                    # Always hand RTC a full-width tensor. Dynamic slicing would
                    # make tensor shapes depend on the raw start and force extra
                    # Dynamo guards. Cap EH to valid_leftover_steps so the tail
                    # zeros have zero RTC prefix weight.
                    left = self.last_raw_chunk[:, raw_start:, :]
                    padding = self.chunk_size - left.shape[1]
                    if padding > 0:
                        left = torch.cat(
                            [
                                left,
                                torch.zeros(
                                    1, padding, left.shape[2],
                                    dtype=left.dtype, device=left.device,
                                ),
                            ],
                            dim=1,
                        )
                    kwargs["prev_chunk_left_over"] = left
                    kwargs["rtc_prefix_weights"] = self._build_rtc_prefix_weights(
                        inference_delay=_strict_optional_int(
                            body.get("inference_delay")
                        ),
                        execution_horizon=effective_execution_horizon,
                    )
                    rtc_info["rtc_applied"] = True
            chunk = self.policy.predict_action_chunk(processed, **kwargs)
            latency = time.perf_counter() - started
            if self.rtc_config is not None:
                self.last_raw_chunk = chunk.detach().clone()
                self.rtc_state_token += 1
            actions = self.postprocessor(chunk)

        actions = actions.detach().to("cpu", dtype=torch.float32).numpy()
        if actions.ndim == 3:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] != 23:
            raise ValueError(f"unexpected chunk shape {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("chunk contains non-finite values")

        self.calls += 1
        self.last_latency_s = latency
        result = {
            "actions": actions.tolist(),
            "chunk_shape": list(actions.shape),
            "latency_s": latency,
            "task": task,
        }
        if self.rtc_config is not None:
            result["rtc_state_token"] = self.rtc_state_token
            result.update(rtc_info)
        return result


def make_handler(runtime: PolicyRuntime):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter default logging
            return

        def _send(self, code: int, payload: dict) -> None:
            blob = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def do_GET(self):  # noqa: N802
            if self.path in ("/healthz", "/v1/pi05/healthz"):
                self._send(
                    200,
                    {
                        "status": "ok",
                        "checkpoint": str(runtime.checkpoint),
                        "device": runtime.device,
                        "calls": runtime.calls,
                        "failures": runtime.failures,
                        "last_latency_s": runtime.last_latency_s,
                    },
                )
                return
            self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            if self.path in ("/v1/pi05/reset",):
                self._send(200, {"status": "reset", "calls": runtime.calls})
                return
            if self.path not in ("/v1/pi05/observation", "/observation"):
                self._send(404, {"error": "not found"})
                return
            try:
                body = json.loads(raw.decode("utf-8"))
                result = runtime.infer(body)
            except Exception as error:  # noqa: BLE001
                runtime.failures += 1
                traceback.print_exc()
                self._send(400, {"error": f"{type(error).__name__}: {error}"})
                return
            self._send(200, result)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path override for portable deployments.",
    )
    parser.add_argument("--denoise-steps", type=int, default=None)
    parser.add_argument("--fixed-noise-seed", type=int, default=None)
    args = parser.parse_args()

    runtime = PolicyRuntime(
        args.config,
        args.device,
        args.denoise_steps,
        args.fixed_noise_seed,
        checkpoint_override=args.checkpoint,
    )

    def note(message: str) -> None:
        print(f"[pi05-server] {time.strftime('%H:%M:%S')} {message}", flush=True)

    def on_signal(number, _frame):
        note(f"received signal {number} ({signal.Signals(number).name}); shutting down")
        raise SystemExit(128 + number)

    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, on_signal)
    atexit.register(lambda: note("process exiting (atexit)"))

    def heartbeat() -> None:
        while True:
            time.sleep(30)
            note(
                f"heartbeat calls={runtime.calls} failures={runtime.failures} "
                f"last_latency={runtime.last_latency_s:.3f}"
            )

    threading.Thread(target=heartbeat, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    note(f"listening on {args.host}:{args.port}")
    # If a request wedges, dump every thread's Python stack so the stall is
    # visible instead of silent.
    faulthandler.enable()
    # The periodic watchdog is opt-in: it is useful when hunting a stall, but
    # the service was observed dying shortly after its first 75 s dump, so it
    # must not run in normal operation.
    watch_doc = json.loads(Path(args.config).read_text(encoding="utf-8"))
    dump_seconds = float(watch_doc.get("faulthandler_dump_seconds", 0) or 0)
    if dump_seconds > 0:
        faulthandler.dump_traceback_later(dump_seconds, repeat=True)
        note(f"faulthandler watchdog enabled every {dump_seconds:.0f}s")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
