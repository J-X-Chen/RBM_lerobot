#!/usr/bin/env python3
"""Preview Astribot S1 teleimager-style ZMQ camera streams."""

from __future__ import annotations

import argparse
import contextlib
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CameraRoute:
    label: str
    port: int


@dataclass
class CameraStats:
    frame_count: int = 0
    byte_count: int = 0
    decode_failures: int = 0
    last_frame_time: float = 0.0
    frame_times: deque[float] = field(default_factory=lambda: deque(maxlen=60))
    frame: np.ndarray | None = None
    last_payload_size: int = 0

    @property
    def fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        duration = self.frame_times[-1] - self.frame_times[0]
        if duration <= 0:
            return 0.0
        return float((len(self.frame_times) - 1) / duration)


def _extract_encoded_image_bytes(payload: bytes) -> bytes:
    jpeg_start = payload.find(b"\xff\xd8")
    if jpeg_start >= 0:
        jpeg_end = payload.rfind(b"\xff\xd9")
        if jpeg_end >= jpeg_start:
            return payload[jpeg_start : jpeg_end + 2]
        return payload[jpeg_start:]

    png_start = payload.find(b"\x89PNG\r\n\x1a\n")
    if png_start >= 0:
        return payload[png_start:]

    return payload


def decode_frame_parts(parts: list[bytes], cv2_module) -> np.ndarray | None:
    candidate_parts = sorted(enumerate(parts), key=lambda item: (len(item[1]), item[0]), reverse=True)
    for _, payload in candidate_parts:
        if not payload:
            continue
        for candidate in (payload, _extract_encoded_image_bytes(payload)):
            if not candidate:
                continue
            encoded = np.frombuffer(candidate, dtype=np.uint8)
            frame = cv2_module.imdecode(encoded, cv2_module.IMREAD_COLOR)
            if frame is not None:
                return frame
    return None


def parse_camera_routes(args: argparse.Namespace) -> list[CameraRoute]:
    routes = [CameraRoute("head", int(args.head_port))]
    if not args.disable_wrist_cameras:
        routes.extend(
            [
                CameraRoute("left_wrist", int(args.left_wrist_port)),
                CameraRoute("right_wrist", int(args.right_wrist_port)),
            ]
        )

    for item in args.camera or []:
        try:
            label, port_text = item.split(":", 1)
        except ValueError as exc:
            raise ValueError("--camera entries must be label:port, e.g. torso:55558") from exc
        routes.append(CameraRoute(label.strip(), int(port_text)))

    deduped: list[CameraRoute] = []
    seen_labels: set[str] = set()
    seen_ports: set[int] = set()
    for route in routes:
        if not route.label:
            continue
        if route.label in seen_labels:
            raise ValueError(f"Duplicate camera label {route.label!r}.")
        if route.port in seen_ports:
            raise ValueError(f"Duplicate ZMQ port {route.port}.")
        seen_labels.add(route.label)
        seen_ports.add(route.port)
        deduped.append(route)
    return deduped


def _display_frame(route: CameraRoute, stats: CameraStats, args: argparse.Namespace, cv2_module) -> None:
    frame = stats.frame
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2_module.putText(
            frame,
            f"Waiting for {route.label} on {args.host}:{route.port}",
            (24, 240),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
            cv2_module.LINE_AA,
        )
    else:
        frame = frame.copy()
        label = f"{route.label}  {stats.fps:.1f} fps  {frame.shape[1]}x{frame.shape[0]}"
        cv2_module.putText(
            frame,
            label,
            (12, 30),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2_module.LINE_AA,
        )

    if args.display_scale != 1.0:
        frame = cv2_module.resize(frame, None, fx=args.display_scale, fy=args.display_scale)
    if args.max_display_width > 0 and frame.shape[1] > args.max_display_width:
        scale = float(args.max_display_width) / float(frame.shape[1])
        frame = cv2_module.resize(frame, None, fx=scale, fy=scale)

    cv2_module.imshow(f"Astribot S1 {route.label}", frame)


def _print_status(routes: list[CameraRoute], stats_by_label: dict[str, CameraStats], host: str) -> None:
    now = time.time()
    for route in routes:
        stats = stats_by_label[route.label]
        age = now - stats.last_frame_time if stats.last_frame_time > 0 else float("inf")
        age_text = "never" if age == float("inf") else f"{age:.1f}s ago"
        mb = stats.byte_count / (1024 * 1024)
        print(
            f"[astribot-image-client] {route.label} {host}:{route.port}: "
            f"frames={stats.frame_count} fps={stats.fps:.1f} bytes={mb:.1f}MiB "
            f"last={age_text} decode_failures={stats.decode_failures}",
            flush=True,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host running astribot_image_service.server.")
    parser.add_argument("--head-port", type=int, default=55555, help="ZMQ port for the head camera.")
    parser.add_argument("--left-wrist-port", type=int, default=55556, help="ZMQ port for the left wrist camera.")
    parser.add_argument("--right-wrist-port", type=int, default=55557, help="ZMQ port for the right wrist camera.")
    parser.add_argument("--disable-wrist-cameras", action="store_true", help="Only preview the head camera.")
    parser.add_argument("--camera", action="append", default=[], help="Extra route as label:port, e.g. torso:55558.")
    parser.add_argument("--display-scale", type=float, default=1.0, help="Scale factor applied before display.")
    parser.add_argument("--max-display-width", type=int, default=960, help="Resize windows wider than this; 0 disables.")
    parser.add_argument("--poll-timeout-ms", type=int, default=20, help="ZMQ poll timeout in milliseconds.")
    parser.add_argument("--status-interval-s", type=float, default=2.0, help="Periodic terminal status interval.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    routes = parse_camera_routes(args)
    if not routes:
        raise ValueError("No camera routes configured.")

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise SystemExit("opencv-python is required for the viewer. Install with: pip install opencv-python") from exc
    try:
        import zmq
    except ModuleNotFoundError as exc:
        raise SystemExit("pyzmq is required for the viewer. Install the project deps first.") from exc

    context = zmq.Context()
    poller = zmq.Poller()
    route_by_socket = {}
    stats_by_label = {route.label: CameraStats() for route in routes}

    try:
        for route in routes:
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.RCVHWM, 1)
            socket.setsockopt(zmq.CONFLATE, 1)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(f"tcp://{args.host}:{route.port}")
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
            poller.register(socket, zmq.POLLIN)
            route_by_socket[socket] = route
            print(f"[astribot-image-client] subscribed {route.label} <- {args.host}:{route.port}", flush=True)

        next_status_time = 0.0
        while True:
            events = dict(poller.poll(timeout=max(1, int(args.poll_timeout_ms))))
            for socket, route in list(route_by_socket.items()):
                if socket not in events:
                    continue

                parts = socket.recv_multipart()
                while socket.poll(timeout=0):
                    with contextlib.suppress(zmq.Again):
                        parts = socket.recv_multipart(flags=zmq.NOBLOCK)

                stats = stats_by_label[route.label]
                frame = decode_frame_parts(parts, cv2)
                payload_size = sum(len(part) for part in parts)
                if frame is None:
                    stats.decode_failures += 1
                    continue

                now = time.time()
                stats.frame = frame
                stats.frame_count += 1
                stats.byte_count += payload_size
                stats.last_payload_size = payload_size
                stats.last_frame_time = now
                stats.frame_times.append(now)

            for route in routes:
                _display_frame(route, stats_by_label[route.label], args, cv2)

            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                break

            now = time.time()
            if args.status_interval_s > 0 and now >= next_status_time:
                _print_status(routes, stats_by_label, args.host)
                next_status_time = now + float(args.status_interval_s)
    except KeyboardInterrupt:
        pass
    finally:
        for socket in list(route_by_socket):
            poller.unregister(socket)
            socket.close()
        context.term()
        cv2.destroyAllWindows()
        print("[astribot-image-client] stopped.", flush=True)


if __name__ == "__main__":
    main()

