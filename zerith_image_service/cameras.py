"""SDK BGR frames -> timestamped multipart JPEG ZMQ streams, one socket per camera."""
import importlib
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from .contract import CAMERAS


def connect_cameras(root, target):
    sys.path.insert(0, str(Path(root).resolve() / 'camera_sdk_python'))
    client = importlib.import_module('camera_client').CameraClient(grpc_target=target)
    client.start()
    return client


class CameraPublisher(threading.Thread):
    def __init__(self, client, names, host, ports, stop, mock=False, fps=30.):
        super().__init__(daemon=True)
        self.client, self.names, self.host, self.ports = client, names, host, ports
        self.stop_event, self.mock, self.fps = stop, mock, fps
        self.error = None
        self.ready = threading.Event()

    def run(self):
        import cv2
        import zmq
        context = zmq.Context()
        sockets = []
        previous = {}
        try:
            for port in self.ports:
                sock = context.socket(zmq.PUB)
                sock.setsockopt(zmq.SNDHWM, 1)
                sock.setsockopt(zmq.LINGER, 0)
                sockets.append(sock)
                sock.bind(f'tcp://{self.host}:{port}')
            if not self.mock:
                available = {c.camera_name for c in self.client.get_state().camera_configs}
                missing = set(self.names) - available
                if missing:
                    raise ValueError(f'Camera names missing: {missing}; available={sorted(available)}')
            self.ready.set()
            while not self.stop_event.is_set():
                for feature, name, sock in zip(CAMERAS, self.names, sockets):
                    if self.mock:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, feature, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 80), 2)
                        data = frame, time.monotonic()
                    else:
                        data = self.client.get_latest_frame(name)
                    if data is None:
                        continue
                    frame, stamp = data
                    stamp = float(stamp)
                    # Never refresh the arrival time by republishing a frozen SDK frame.
                    if not np.isfinite(stamp) or stamp == previous.get(name):
                        continue
                    previous[name] = stamp
                    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                        raise ValueError(f'Invalid BGR frame: {name}')
                    ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not ok:
                        raise RuntimeError('JPEG encode failed')
                    metadata = json.dumps(dict(camera=feature, source_timestamp=stamp, published_monotonic=time.monotonic(), color='bgr')).encode()
                    sock.send_multipart([metadata, jpeg.tobytes()])
                self.stop_event.wait(1/self.fps)
        except Exception as exc:
            self.error = str(exc)
            self.stop_event.set()
        finally:
            self.ready.set()
            for sock in sockets:
                sock.close()
            context.term()


class CameraSubscriber:
    """Own sockets on the evaluation thread; reject stalled or desynchronized streams."""
    def __init__(self, host, ports=(55555, 55556, 55557), max_age=.5, max_skew=.25):
        import zmq
        self.zmq = zmq
        self.context = zmq.Context()
        self.sockets = []
        self.frames = {}
        self.max_age, self.max_skew = max_age, max_skew
        self.clock_reference = None
        for port in ports:
            sock = self.context.socket(zmq.SUB)
            sock.setsockopt(zmq.SUBSCRIBE, b'')
            sock.setsockopt(zmq.RCVHWM, 2)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(f'tcp://{host}:{port}')
            self.sockets.append(sock)

    def read(self, timeout=2., server_time=None):
        import cv2
        import json
        if server_time is not None:
            if not np.isfinite(server_time):
                raise ValueError('Invalid server clock')
            self.clock_reference = float(server_time), time.monotonic()
        if self.clock_reference is None:
            raise ValueError('Camera freshness requires the control response server_time')
        def remote_now():
            return self.clock_reference[0] + time.monotonic() - self.clock_reference[1]
        deadline = time.monotonic() + timeout
        while True:
            for key, sock in zip(CAMERAS, self.sockets):
                latest = None
                while sock.poll(0):
                    latest = sock.recv_multipart()
                if latest:
                    if len(latest) != 2:
                        raise ValueError('Expected H1 camera metadata and JPEG')
                    meta = json.loads(latest[0])
                    if meta.get('camera') != key or meta.get('color') != 'bgr':
                        raise ValueError('Camera routing/color mismatch')
                    stamp = float(meta['source_timestamp'])
                    published = float(meta['published_monotonic'])
                    if not np.isfinite(published) or not -.1 <= remote_now()-published <= self.max_age:
                        continue
                    if not np.isfinite(stamp):
                        raise ValueError('Invalid camera timestamp')
                    if key in self.frames and self.frames[key][2] == stamp:
                        continue
                    frame = cv2.imdecode(np.frombuffer(latest[1], dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError('JPEG decode failed')
                    self.frames[key] = (frame, published, stamp)
            now = time.monotonic()
            if len(self.frames) == 3:
                arrivals = [v[1] for v in self.frames.values()]
                if remote_now()-min(arrivals) <= self.max_age and max(arrivals)-min(arrivals) <= self.max_skew:
                    return {key: frame[0].copy() for key, frame in self.frames.items()}
            if now >= deadline:
                raise TimeoutError('Three camera streams missing, stale, or arrival skew too large')
            time.sleep(.005)

    def close(self):
        for sock in self.sockets:
            sock.close()
        self.context.term()
