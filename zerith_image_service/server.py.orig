"""H1-side image and control service. Read-only by default; no implicit initialization."""
import argparse
from pathlib import Path
import signal
import select
import socketserver
import threading
import time
import uuid

from .cameras import CameraPublisher, connect_cameras
from .contract import PROTOCOL, default_urdf, limits_from_urdf, recv_json, send_json
from .control import H1Control
from .sdk import H1SDK, MockSDK


class ActionHandler(socketserver.BaseRequestHandler):
    def handle(self):
        owner = uuid.uuid4().hex
        try:
            while not self.server.stop_event.is_set():
                # Read-only inference clients may be idle while a large model runs.
                # Timeout only a partially sent message, not an idle connection.
                if not select.select([self.request], [], [], .2)[0]:
                    continue
                payload = recv_json(self.request, timeout=2.)
                try:
                    reply = self.server.control.handle(payload, owner)
                except Exception as exc:
                    # A rejected motion request cancels the owner's previous target as well.
                    self.server.control.disconnect(owner)
                    reply = dict(ok=False, protocol=PROTOCOL, error=str(exc))
                send_json(self.request, reply)
        except (ConnectionError, OSError, ValueError):
            pass
        finally:
            self.server.control.disconnect(owner)


class ActionServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, control, stop):
        super().__init__(address, ActionHandler)
        self.control, self.stop_event = control, stop


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk-root', type=Path, default=Path(__file__).resolve().parents[1] / 'zerith_H1_SDK1.3.6')
    p.add_argument('--robot-host', help='Omit when running on H1 host')
    p.add_argument('--urdf', type=Path)
    p.add_argument('--bind-host', default='0.0.0.0')
    p.add_argument('--control-port', type=int, default=6200)
    p.add_argument('--camera-target', default='localhost:50051')
    p.add_argument('--camera-names', nargs=3, default=['rs/cam_high', 'rs/cam_left_wrist', 'rs/cam_right_wrist'])
    p.add_argument('--camera-ports', nargs=3, type=int, default=[55555, 55556, 55557])
    p.add_argument('--list-cameras', action='store_true')
    p.add_argument('--no-cameras', action='store_true')
    p.add_argument('--mock', action='store_true')
    p.add_argument('--execute-control', action='store_true')
    p.add_argument('--control-frequency', type=float, default=100.)
    p.add_argument('--control-timeout', type=float, default=.5)
    p.add_argument('--joint-velocity', type=float, default=.15)
    p.add_argument('--lift-velocity', type=float, default=.02)
    p.add_argument('--gripper-velocity', type=float, default=.5)
    p.add_argument('--joint-envelope', type=float, default=.5)
    p.add_argument('--lift-envelope', type=float, default=.15)
    return p


def main():
    args = build_parser().parse_args()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    camera = publisher = server = control = None
    try:
        if args.list_cameras:
            camera = connect_cameras(args.sdk_root, args.camera_target)
            print('\n'.join(c.camera_name for c in camera.get_state().camera_configs))
            return
        sdk = MockSDK() if args.mock else H1SDK.connect(args.sdk_root, args.robot_host)
        control = H1Control(sdk, limits_from_urdf(args.urdf or default_urdf(args.sdk_root)),
                            execute=args.execute_control, frequency=args.control_frequency,
                            timeout=args.control_timeout, joint_velocity=args.joint_velocity,
                            lift_velocity=args.lift_velocity, gripper_velocity=args.gripper_velocity,
                            envelope=args.joint_envelope, lift_envelope=args.lift_envelope)
        if not args.no_cameras:
            camera = None if args.mock else connect_cameras(args.sdk_root, args.camera_target)
            publisher = CameraPublisher(camera, args.camera_names, args.bind_host, args.camera_ports, stop, args.mock)
            publisher.start()
            if not publisher.ready.wait(10) or publisher.error:
                raise RuntimeError(f'Camera publisher startup failed: {publisher.error}')
        server = ActionServer((args.bind_host, args.control_port), control, stop)
        worker = threading.Thread(target=control.run, args=(stop,), daemon=True)
        worker.start()
        network = threading.Thread(target=server.serve_forever, daemon=True)
        network.start()
        print(f'H1 service {PROTOCOL}: mock={args.mock}, execute={args.execute_control}, control={server.server_address}', flush=True)
        print('No initialization/motion until an explicit client arm; q/disarm is NOT a hardware emergency stop.', flush=True)
        while not stop.wait(.2):
            if control.fault:
                print(f'Latched control fault: {control.fault}', flush=True)
                break
    finally:
        stop.set()
        if control:
            with control.lock:
                control._latch('service shutdown')
        if server:
            server.shutdown()
            server.server_close()
        if publisher:
            publisher.join(timeout=2)
        if camera:
            camera.stop()
        # No deinit or automatic return: those can move/release supported joints.


if __name__ == '__main__':
    main()
