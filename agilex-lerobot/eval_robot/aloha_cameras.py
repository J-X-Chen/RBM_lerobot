"""agilex_image_service three-port camera transport: one JPEG message per ZMQ PUB/SUB frame.

Freshness here measures local reception time,
not camera exposure time; this protocol does not guarantee camera synchronization.
"""

import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)


def decode_rgb(payload):
    import cv2

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Expected a single JPEG image message, without headers or topic prefix")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()


class AlohaCameras:
    def __init__(self, host, ports, max_age=2.0):
        if set(ports) != {"head", "left", "right"} or len(set(ports.values())) != 3:
            raise ValueError("Three distinct head/left/right camera ports are required")
        if any(not isinstance(p, int) or not 1 <= p <= 65535 for p in ports.values()):
            raise ValueError("Invalid camera port")
        if not np.isfinite(max_age) or max_age <= 0:
            raise ValueError("Invalid camera max_age")
        self.host, self.ports, self.max_age = host, ports, max_age
        self.thread = None
        self.frames = {}
        self.error = None
        self.lock = threading.Lock()
        self.closed = threading.Event()

    def connect(self):
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        try:
            deadline = time.monotonic() + 15
            while True:
                try:
                    self.snapshot()
                    return
                except RuntimeError:
                    if self.error is not None or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.02)
        except BaseException:
            self.close()
            raise

    def _read(self):
        sockets = {}
        context = None
        try:
            import zmq

            context = zmq.Context()
            poller = zmq.Poller()
            for name, port in self.ports.items():
                socket = context.socket(zmq.SUB)
                sockets[socket] = name
                socket.setsockopt(zmq.SUBSCRIBE, b"")
                socket.setsockopt(zmq.CONFLATE, 1)
                socket.setsockopt(zmq.LINGER, 0)
                socket.connect(f"tcp://{self.host}:{port}")
                poller.register(socket, zmq.POLLIN)
            while not self.closed.is_set():
                for socket, event in poller.poll(100):
                    if event & zmq.POLLIN:
                        rgb = decode_rgb(socket.recv())
                        with self.lock:
                            self.frames[sockets[socket]] = (time.monotonic(), rgb)
        except Exception as exc:
            logger.exception("Camera receive thread failed")
            with self.lock:
                self.error = str(exc)
        finally:
            for socket in sockets:
                socket.close(linger=0)
            if context is not None:
                context.term()

    def snapshot(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError(f"Camera receiver failed: {self.error}")
            if set(self.frames) != set(self.ports):
                raise RuntimeError("Waiting for all three camera ports (sender not connected)")
            now = time.monotonic()
            if any(now - stamp > self.max_age for stamp, _ in self.frames.values()):
                raise RuntimeError("Camera stream is stale")
            return {f"observation.images.{name}": image.copy() for name, (_, image) in self.frames.items()}

    def close(self):
        self.closed.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
