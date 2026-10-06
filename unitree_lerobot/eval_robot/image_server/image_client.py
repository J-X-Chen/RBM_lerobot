import cv2
import zmq
import numpy as np
import time
import struct
import threading
from collections import deque
from multiprocessing import shared_memory


def _decode_image_as_rgb(encoded_image: np.ndarray):
    image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class ImageClient:
    def __init__(
        self,
        tv_img_shape=None,
        tv_img_shm_name=None,
        wrist_img_shape=None,
        wrist_img_shm_name=None,
        image_show=False,
        server_address="192.168.123.164",
        port=5555,
        Unit_Test=False,
    ):
        """
        tv_img_shape: User's expected head camera resolution shape (H, W, C). It should match the output of the image service terminal.
        tv_img_shm_name: Shared memory is used to easily transfer images across processes to the Vuer.
        wrist_img_shape: User's expected wrist camera resolution shape (H, W, C). It should maintain the same shape as tv_img_shape.
        wrist_img_shm_name: Shared memory is used to easily transfer images.
        image_show: Whether to display received images in real time.
        server_address: The ip address to execute the image server script.
        port: The port number to bind to. It should be the same as the image server.
        Unit_Test: When both server and client are True, it can be used to test the image transfer latency, \
                   network jitter, frame loss rate and other information.
        """
        self.running = True
        self._image_show = image_show
        self._server_address = server_address
        self._port = port

        self.tv_img_shape = tv_img_shape
        self.wrist_img_shape = wrist_img_shape

        self.tv_enable_shm = False
        if self.tv_img_shape is not None and tv_img_shm_name is not None:
            self.tv_image_shm = shared_memory.SharedMemory(name=tv_img_shm_name)
            self.tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=self.tv_image_shm.buf)
            self.tv_enable_shm = True

        self.wrist_enable_shm = False
        if self.wrist_img_shape is not None and wrist_img_shm_name is not None:
            self.wrist_image_shm = shared_memory.SharedMemory(name=wrist_img_shm_name)
            self.wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=self.wrist_image_shm.buf)
            self.wrist_enable_shm = True

        # Performance evaluation parameters
        self._enable_performance_eval = Unit_Test
        if self._enable_performance_eval:
            self._init_performance_metrics()

    def _init_performance_metrics(self):
        self._frame_count = 0  # Total frames received
        self._last_frame_id = -1  # Last received frame ID

        # Real-time FPS calculation using a time window
        self._time_window = 1.0  # Time window size (in seconds)
        self._frame_times = deque()  # Timestamps of frames received within the time window

        # Data transmission quality metrics
        self._latencies = deque()  # Latencies of frames within the time window
        self._lost_frames = 0  # Total lost frames
        self._total_frames = 0  # Expected total frames based on frame IDs

    def _update_performance_metrics(self, timestamp, frame_id, receive_time):
        # Update latency
        latency = receive_time - timestamp
        self._latencies.append(latency)

        # Remove latencies outside the time window
        while self._latencies and self._frame_times and self._latencies[0] < receive_time - self._time_window:
            self._latencies.popleft()

        # Update frame times
        self._frame_times.append(receive_time)
        # Remove timestamps outside the time window
        while self._frame_times and self._frame_times[0] < receive_time - self._time_window:
            self._frame_times.popleft()

        # Update frame counts for lost frame calculation
        expected_frame_id = self._last_frame_id + 1 if self._last_frame_id != -1 else frame_id
        if frame_id != expected_frame_id:
            lost = frame_id - expected_frame_id
            if lost < 0:
                print(f"[Image Client] Received out-of-order frame ID: {frame_id}")
            else:
                self._lost_frames += lost
                print(
                    f"[Image Client] Detected lost frames: {lost}, Expected frame ID: {expected_frame_id}, Received frame ID: {frame_id}"
                )
        self._last_frame_id = frame_id
        self._total_frames = frame_id + 1

        self._frame_count += 1

    def _print_performance_metrics(self, receive_time):
        if self._frame_count % 30 == 0:
            # Calculate real-time FPS
            real_time_fps = len(self._frame_times) / self._time_window if self._time_window > 0 else 0

            # Calculate latency metrics
            if self._latencies:
                avg_latency = sum(self._latencies) / len(self._latencies)
                max_latency = max(self._latencies)
                min_latency = min(self._latencies)
                jitter = max_latency - min_latency
            else:
                avg_latency = max_latency = min_latency = jitter = 0

            # Calculate lost frame rate
            lost_frame_rate = (self._lost_frames / self._total_frames) * 100 if self._total_frames > 0 else 0

            print(
                f"[Image Client] Real-time FPS: {real_time_fps:.2f}, Avg Latency: {avg_latency * 1000:.2f} ms, Max Latency: {max_latency * 1000:.2f} ms, \
                  Min Latency: {min_latency * 1000:.2f} ms, Jitter: {jitter * 1000:.2f} ms, Lost Frame Rate: {lost_frame_rate:.2f}%"
            )

    def _close(self):
        self._socket.close()
        self._context.term()
        if self._image_show:
            cv2.destroyAllWindows()
        print("Image client has been closed.")

    def stop(self):
        self.running = False

    def receive_process(self):
        # Set up ZeroMQ context and socket
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{self._server_address}:{self._port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print("\nImage client has started, waiting to receive data...")
        try:
            while self.running:
                # Receive message
                message = self._socket.recv()
                receive_time = time.time()

                if self._enable_performance_eval:
                    header_size = struct.calcsize("dI")
                    try:
                        # Attempt to extract header and image data
                        header = message[:header_size]
                        jpg_bytes = message[header_size:]
                        timestamp, frame_id = struct.unpack("dI", header)
                    except struct.error as e:
                        print(f"[Image Client] Error unpacking header: {e}, discarding message.")
                        continue
                else:
                    # No header, entire message is image data
                    jpg_bytes = message
                # Decode image
                np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
                current_image = _decode_image_as_rgb(np_img)
                if current_image is None:
                    print("[Image Client] Failed to decode image.")
                    continue

                if self.tv_enable_shm:
                    np.copyto(self.tv_img_array, np.array(current_image[:, : self.tv_img_shape[1]]))

                if self.wrist_enable_shm:
                    np.copyto(self.wrist_img_array, np.array(current_image[:, -self.wrist_img_shape[1] :]))

                if self._image_show:
                    height, width = current_image.shape[:2]
                    resized_image = cv2.resize(current_image, (width // 2, height // 2))
                    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Image Client Stream", resized_image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.running = False

                if self._enable_performance_eval:
                    self._update_performance_metrics(timestamp, frame_id, receive_time)
                    self._print_performance_metrics(receive_time)

        except KeyboardInterrupt:
            print("Image client interrupted by user.")
        except Exception as e:
            print(f"[Image Client] An error occurred while receiving data: {e}")
        finally:
            self._close()


class TeleimagerImageClient:
    """Subscribe to official teleimager ZMQ camera streams and fill shared-memory image buffers."""

    def __init__(
        self,
        tv_img_shape=None,
        tv_img_shm_name=None,
        wrist_img_shape=None,
        wrist_img_shm_name=None,
        thermal_img_shape=None,
        thermal_img_shm_name=None,
        server_address="192.168.123.164",
        head_port=55555,
        left_wrist_port=55556,
        right_wrist_port=55557,
        thermal_port=55559,
    ):
        self.running = True
        self._server_address = server_address
        self._head_port = int(head_port)
        self._left_wrist_port = int(left_wrist_port)
        self._right_wrist_port = int(right_wrist_port)
        self._thermal_port = int(thermal_port)
        self._warned_shape_mismatch = set()
        self._warned_decode_failure = set()
        self._logged_frame_shapes = set()
        self._threads = []

        self.tv_img_shape = tv_img_shape
        self.wrist_img_shape = wrist_img_shape
        self.thermal_img_shape = thermal_img_shape

        self.tv_enable_shm = False
        if self.tv_img_shape is not None and tv_img_shm_name is not None:
            self.tv_image_shm = shared_memory.SharedMemory(name=tv_img_shm_name)
            self.tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=self.tv_image_shm.buf)
            self.tv_enable_shm = True

        self.wrist_enable_shm = False
        if self.wrist_img_shape is not None and wrist_img_shm_name is not None:
            self.wrist_image_shm = shared_memory.SharedMemory(name=wrist_img_shm_name)
            self.wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=self.wrist_image_shm.buf)
            self.wrist_enable_shm = True

        self.thermal_enable_shm = False
        if self.thermal_img_shape is not None and thermal_img_shm_name is not None:
            self.thermal_image_shm = shared_memory.SharedMemory(name=thermal_img_shm_name)
            self.thermal_img_array = np.ndarray(
                thermal_img_shape, dtype=np.uint8, buffer=self.thermal_image_shm.buf
            )
            self.thermal_enable_shm = True

    def stop(self):
        self.running = False

    def _copy_frame(self, name, image, target, x_slice=None):
        if target is None:
            return

        if x_slice is None:
            expected_h, expected_w = target.shape[:2]
            dst = target
        else:
            start, stop = x_slice
            expected_h, expected_w = target.shape[0], stop - start
            dst = target[:, start:stop]

        log_key = (name, x_slice)
        if log_key not in self._logged_frame_shapes:
            print(
                f"[Teleimager Image Client] {name} decoded frame shape {image.shape}; "
                f"target {(expected_h, expected_w, target.shape[2])}"
                + (f", x_slice={x_slice}" if x_slice is not None else "")
                + "."
            )
            self._logged_frame_shapes.add(log_key)

        if image.shape[:2] != (expected_h, expected_w):
            warning_key = (name, image.shape[:2], expected_h, expected_w)
            if warning_key not in self._warned_shape_mismatch:
                print(
                    f"[Teleimager Image Client] {name} frame shape {image.shape[:2]} does not match "
                    f"target {(expected_h, expected_w)}; resizing."
                )
                self._warned_shape_mismatch.add(warning_key)
            image = cv2.resize(image, (expected_w, expected_h))

        np.copyto(dst, image)

    @staticmethod
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

    def _decode_frame_parts(self, name: str, port: int, parts: list[bytes]):
        # Try larger/later parts first; multipart publishers often send topic/metadata before image bytes.
        candidate_parts = sorted(enumerate(parts), key=lambda item: (len(item[1]), item[0]), reverse=True)
        for _, payload in candidate_parts:
            if not payload:
                continue

            for candidate in (payload, self._extract_encoded_image_bytes(payload)):
                if not candidate:
                    continue
                np_img = np.frombuffer(candidate, dtype=np.uint8)
                image = _decode_image_as_rgb(np_img)
                if image is not None:
                    return image

        warning_key = (name, port)
        if warning_key not in self._warned_decode_failure:
            part_sizes = [len(part) for part in parts]
            part_prefixes = [part[:16].hex(" ") for part in parts[:4]]
            print(
                f"[Teleimager Image Client] Failed to decode {name} frame from {port}. "
                f"parts={part_sizes}, first_bytes={part_prefixes}. "
                "Expected JPEG/PNG encoded image bytes."
            )
            self._warned_decode_failure.add(warning_key)
        return None

    def _receive_camera(self, name, port, target, x_slice=None):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(f"tcp://{self._server_address}:{port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        try:
            while self.running:
                events = dict(poller.poll(timeout=100))
                if socket not in events:
                    continue

                parts = socket.recv_multipart()
                while socket.poll(timeout=0):
                    parts = socket.recv_multipart(flags=zmq.NOBLOCK)

                image = self._decode_frame_parts(name, port, parts)
                if image is None:
                    continue

                self._copy_frame(name, image, target, x_slice=x_slice)
        except Exception as exc:
            if self.running:
                print(f"[Teleimager Image Client] {name} receive loop failed on port {port}: {exc}")
        finally:
            socket.close()
            context.term()

    def receive_process(self):
        print(
            "\nTeleimager image client has started, waiting for ZMQ data "
            f"from {self._server_address}: head={self._head_port}, "
            f"left_wrist={self._left_wrist_port}, right_wrist={self._right_wrist_port}, "
            f"thermal={self._thermal_port if self.thermal_enable_shm else 'disabled'}..."
        )

        if self.tv_enable_shm:
            self._threads.append(
                threading.Thread(
                    target=self._receive_camera,
                    args=("head_camera", self._head_port, self.tv_img_array, None),
                    daemon=True,
                )
            )

        if self.wrist_enable_shm:
            mid = self.wrist_img_shape[1] // 2
            self._threads.append(
                threading.Thread(
                    target=self._receive_camera,
                    args=("left_wrist_camera", self._left_wrist_port, self.wrist_img_array, (0, mid)),
                    daemon=True,
                )
            )
            self._threads.append(
                threading.Thread(
                    target=self._receive_camera,
                    args=(
                        "right_wrist_camera",
                        self._right_wrist_port,
                        self.wrist_img_array,
                        (mid, self.wrist_img_shape[1]),
                    ),
                    daemon=True,
                )
            )

        if self.thermal_enable_shm:
            self._threads.append(
                threading.Thread(
                    target=self._receive_camera,
                    args=("thermal_camera", self._thermal_port, self.thermal_img_array, None),
                    daemon=True,
                )
            )

        for thread in self._threads:
            thread.start()

        try:
            while self.running:
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.stop()


if __name__ == "__main__":
    # example1
    # tv_img_shape = (480, 1280, 3)
    # img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    # img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=img_shm.buf)
    # img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = img_shm.name)
    # img_client.receive_process()

    # example2
    # Initialize the client with performance evaluation enabled
    # client = ImageClient(image_show = True, server_address='127.0.0.1', Unit_Test=True) # local test
    client = ImageClient(image_show=True, server_address="192.168.123.164", Unit_Test=False)  # deployment test
    client.receive_process()
