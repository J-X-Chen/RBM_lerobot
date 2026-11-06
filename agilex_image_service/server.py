"""Three JPEG streams and a single-owner TCP bridge to the Piper SDK (Python 3.8+)."""

import argparse
import json
import select
import socketserver
import threading
import time

import numpy as np
from piper_control import ArmConfig, PiperBimanual, vector


def receive(sock, n):
    data = b""
    while len(data) < n:
        part = sock.recv(n - len(data))
        if not part:
            raise EOFError()
        data += part
    return data


def validate(value):
    # Legacy protocol utility; calibrated control uses PiperBimanual limits instead.
    a = vector(value)
    if np.any(np.abs(a[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]) > 3.2):
        raise ValueError("joint target exceeds 3.2 rad envelope")
    a[[6, 13]] = np.clip(a[[6, 13]], 0, 0.08)
    return a


class Bridge:
    def __init__(self, args):
        import cv2
        import rospy
        import zmq
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image, JointState

        self.args, self.rospy = args, rospy
        self.lock = threading.RLock()
        self.states, self.frames = {}, {}
        self.error = None
        self.stop = threading.Event()
        self.robot = None
        self.context = zmq.Context()
        self.cv, self.cv2 = CvBridge(), cv2
        rospy.init_node("agilex_image_service", anonymous=False)
        try:
            if args.hardware_config:
                from hardware_config import load_resolved_hardware

                hardware = load_resolved_hardware(args.hardware_config)
                self.robot = PiperBimanual(
                    ArmConfig(**hardware["left"]),
                    ArmConfig(**hardware["right"]),
                    execute=args.execute_control,
                    joint_speed=args.joint_speed,
                    gripper_speed=args.gripper_speed,
                    command_timeout=args.timeout,
                    feedback_timeout=args.feedback_timeout,
                    tracking_limit=args.tracking_limit,
                    smooth_control=args.smooth_control,
                    control_frequency=args.control_frequency,
                    joint_acceleration=args.joint_acceleration,
                    gripper_acceleration=args.gripper_acceleration,
                    tracking_response_s=args.tracking_response_s,
                )
                self.robot.connect()  # feedback only, never auto-enable
            else:
                for side in ("left", "right"):
                    rospy.Subscriber("/puppet/joint_" + side, JointState, self.state_cb, side, queue_size=1)
            for label, suffix in [("head", "f"), ("left", "l"), ("right", "r")]:
                rospy.Subscriber(
                    "/camera_" + suffix + "/color/image_raw", Image, self.image_cb, label, queue_size=1, buff_size=2**22
                )
        except BaseException:
            if self.robot:
                self.robot.close()
            self.context.term()
            raise
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def state_cb(self, msg, side):
        if len(msg.position) == 7 and np.isfinite(msg.position).all():
            with self.lock:
                self.states[side] = (list(msg.position), time.monotonic())

    def image_cb(self, msg, label):
        with self.lock:
            self.frames[label] = (msg, time.monotonic())

    def state(self):
        if self.robot:
            return self.robot.read_state()
        now = time.monotonic()
        if any(s not in self.states or now - self.states[s][1] > self.args.feedback_timeout for s in ("left", "right")):
            raise ValueError("joint feedback missing or stale")
        return np.array(self.states["left"][0] + self.states["right"][0])

    def camera_ages(self):
        now = time.monotonic()
        return {k: now - v[1] for k, v in self.frames.items()}

    def check_cameras(self):
        ages = self.camera_ages()
        if set(ages) != {"head", "left", "right"} or max(ages.values()) > self.args.camera_timeout:
            raise RuntimeError("Robot-side camera input missing/stale")

    def request(self, req):
        with self.lock:
            if not isinstance(req, dict):
                raise ValueError("expected JSON object")
            if req.get("get_state"):
                return dict(
                    ok=True,
                    protocol=2,
                    capabilities=["prepare_target", "startup_anchor", "finish_hold", "accepted_target"],
                    state=self.state().tolist(),
                    execute_control=self.args.execute_control,
                    enabled=bool(self.robot and self.robot.enabled),
                    faulted=bool(self.robot and self.robot.faulted),
                    calibrated=bool(self.robot),
                    error=self.error,
                    command_timeout=self.args.timeout,
                    smooth_control=self.robot.smooth_control if self.robot else False,
                    camera_age=self.camera_ages(),
                )
            if req.get("hold_current") or req.get("stop"):
                self.stop_control()
                return dict(ok=True, message="SDK stop sent if enabled; restart service before re-enabling")
            if not self.robot:
                raise ValueError("Control/preview requires --hardware-config on server")
            if "prepare_target" in req:
                return dict(ok=True, target=self.robot.prepare_target(req["prepare_target"]).tolist())
            if req.get("finish_hold"):
                self.check_cameras()
                self.robot.finish_hold()
                return dict(ok=True, holding=True)
            if req.get("enable"):
                if not self.args.execute_control:
                    raise ValueError("Server is read-only; restart with --execute-control")
                if self.robot.enabled and not self.robot.holding:
                    raise ValueError("Already enabled")
                self.check_cameras()
                arm = req.get("active_arm", "both")
                if arm not in ("left", "right", "both"):
                    raise ValueError("Invalid active_arm")
                self.robot.active_arm = arm
                # Client may reduce, never raise server speed caps.
                for key, cap in [("joint_speed", self.args.joint_speed), ("gripper_speed", self.args.gripper_speed)]:
                    value = float(req.get(key, cap))
                    if not np.isfinite(value) or value <= 0:
                        raise ValueError("Invalid " + key)
                    setattr(self.robot, key, min(value, cap))
                self.robot.enable()
                self.last_action = time.monotonic()
                return dict(
                    ok=True,
                    enabled=True,
                    joint_speed=self.robot.joint_speed,
                    gripper_speed=self.robot.gripper_speed,
                    anchor=self.robot.anchor.tolist(),
                )
            if "action" not in req:
                raise ValueError("supported: get_state, enable, action, stop")
            self.check_cameras()
            dt = float(req.get("dt", 1 / 30))
            if not np.isfinite(dt) or not 0 < dt <= 1:
                raise ValueError("Invalid control period")
            if self.args.execute_control:
                # Never accumulate a large step after inference or a network stall.
                now = time.monotonic()
                dt = min(dt, max(1e-6, now - getattr(self, "last_action", now)), 0.1)
                self.last_action = now
            command = self.robot.send(req["action"], dt)
            return dict(ok=True, executed=self.robot.enabled, target=command.tolist(),
                        accepted_target=(self.robot.pending if self.robot.smooth_control else command).tolist())

    def stop_control(self):
        if self.robot and self.robot.enabled:
            self.robot.stop()

    def loop(self):
        import zmq

        sockets, seen = {}, {}
        try:
            for label, port in camera_routes(self.args):
                s = self.context.socket(zmq.PUB)
                sockets[label] = s
                s.setsockopt(zmq.SNDHWM, 1)
                s.setsockopt(zmq.LINGER, 0)
                s.bind("tcp://%s:%d" % (self.args.bind_host, port))
            while not self.stop.is_set() and not self.rospy.is_shutdown():
                with self.lock:
                    frames = dict(self.frames)
                    if self.robot and self.robot.enabled:
                        try:
                            self.check_cameras()
                        except Exception as exc:
                            self.error = str(exc)
                            self.stop_control()
                for label, (msg, stamp) in frames.items():
                    if seen.get(label) == stamp:
                        continue
                    seen[label] = stamp
                    try:
                        bgr = self.cv.imgmsg_to_cv2(msg, "bgr8")
                        ok, encoded = self.cv2.imencode(".jpg", bgr, [int(self.cv2.IMWRITE_JPEG_QUALITY), 90])
                        if ok:
                            sockets[label].send(encoded.tobytes(), zmq.NOBLOCK)
                    except Exception as exc:
                        self.rospy.logwarn_throttle(5, str(exc))
                self.stop.wait(1 / 30)
        except Exception as exc:
            self.error = str(exc)
            self.stop.set()
        finally:
            self.stop_control()
            for s in sockets.values():
                s.close()

    def close(self):
        self.stop.set()
        self.stop_control()
        self.thread.join(3)
        if self.robot:
            self.robot.close()
        self.context.term()


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        # Idle connections survive model warm-up and the operator's start prompt.
        # Once enabled, the independent SDK watchdog requires actual actions.
        self.request.settimeout(3)
        bridge = self.server.bridge
        finished = bool(bridge.robot and bridge.robot.holding)
        try:
            while not bridge.stop.is_set():
                if not select.select([self.request], [], [], 0.2)[0]:
                    continue
                size = int(receive(self.request, 10))
                if not 2 <= size <= 65536:
                    raise ValueError("invalid message size")
                try:
                    result = bridge.request(json.loads(receive(self.request, size)))
                    if result.get("holding"):
                        finished = True
                    elif result.get("enabled") and not bridge.robot.holding:
                        finished = False
                except Exception as exc:
                    bridge.stop_control()
                    result = dict(ok=False, error=str(exc))
                payload = json.dumps(result, allow_nan=False).encode()
                self.request.sendall(("%010d" % len(payload)).encode() + payload)
        except (OSError, EOFError, ValueError):
            pass
        finally:
            if not (finished and bridge.robot and bridge.robot.holding):
                bridge.stop_control()


class Server(socketserver.TCPServer):
    allow_reuse_address = True


def camera_routes(args):
    return [("head", args.head_port), ("left", args.left_wrist_port), ("right", args.right_wrist_port)]


def validate_ports(parser, args):
    ports = [args.control_port] + [port for _, port in camera_routes(args)]
    if any(not 1 <= port <= 65535 for port in ports) or len(set(ports)) != 4:
        parser.error("Camera/control ports must be distinct integers in 1..65535")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bind-host", default="127.0.0.1")
    p.add_argument("--control-port", type=int, default=6200)
    p.add_argument("--head-port", type=int, default=55555)
    p.add_argument("--left-wrist-port", type=int, default=55556)
    p.add_argument("--right-wrist-port", type=int, default=55557)
    p.add_argument("--hardware-config")
    p.add_argument("--execute-control", action="store_true")
    p.add_argument("--joint-speed", type=float, default=0.3)
    p.add_argument("--gripper-speed", type=float, default=0.02)
    p.add_argument("--timeout", type=float, default=5)
    p.add_argument("--feedback-timeout", type=float, default=1)
    p.add_argument("--camera-timeout", type=float, default=2)
    p.add_argument("--tracking-limit", type=float, default=0.35)
    p.add_argument("--smooth-control", action="store_true", help="Experimental independent rate/acceleration limited target following")
    p.add_argument("--control-frequency", type=float, default=30)
    p.add_argument("--joint-acceleration", type=float, default=0.6)
    p.add_argument("--gripper-acceleration", type=float, default=0.04)
    p.add_argument("--tracking-response-s", type=float, default=0.15,
                   help="Smooth tracking response time, 0.1..1.0 seconds; speed/acceleration limits still apply")
    return p


def main():
    p = build_parser()
    args = p.parse_args()
    validate_ports(p, args)
    if args.execute_control and not args.hardware_config:
        p.error("--execute-control requires --hardware-config")
    for key in ("joint_speed", "gripper_speed", "timeout", "feedback_timeout", "camera_timeout", "tracking_limit"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(key + " must be finite and positive")
    bridge = Bridge(args)
    try:
        with Server((args.bind_host, args.control_port), Handler) as server:
            server.bridge = bridge
            server.timeout = 0.2
            print(
                "Images: %s; control: %d; execute allowed=%s (not enabled)"
                % (camera_routes(args), args.control_port, args.execute_control),
                flush=True,
            )
            while not bridge.stop.is_set() and not bridge.rospy.is_shutdown():
                server.handle_request()
            if bridge.error:
                raise RuntimeError(bridge.error)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
