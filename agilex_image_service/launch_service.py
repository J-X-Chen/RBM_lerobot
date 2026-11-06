import datetime
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from server import build_parser, camera_routes, receive, validate_ports
from supervisor import Supervisor, can_problems, existing_ros


def main():
    p = build_parser()
    p.description = "Camera + Piper SDK service; foreground lifecycle"
    p.set_defaults(bind_host="0.0.0.0")
    p.add_argument("--camera-only", action="store_true")
    p.add_argument("--configure-can", action="store_true", help="Configure missing/down arm CAN interfaces before feedback checks")
    p.add_argument("--test-seconds", type=float, default=0)
    args = p.parse_args()
    validate_ports(p, args)
    if args.execute_control and (not args.hardware_config or args.camera_only):
        p.error("--execute-control requires --hardware-config and cannot use --camera-only")
    if args.camera_only and args.hardware_config:
        p.error("--camera-only cannot use --hardware-config")
    if args.hardware_config:
        from hardware_config import load_hardware

        try:
            path, hardware = load_hardware(args.hardware_config)
        except (ValueError, KeyError) as exc:
            p.error(str(exc))
        args.hardware_config = str(path)
        names = [hardware[side]["can"] for side in ("left", "right")]
        if args.configure_can:
            from can_setup import configure_can

            try:
                configure_can(hardware)
            except (ValueError, OSError, subprocess.CalledProcessError) as exc:
                p.error("CAN configuration failed: %s" % exc)
        # Check configured arm interfaces only; no collection master CAN or fixed USB topology.
        links = {
            x["ifname"]: x
            for x in json.loads(subprocess.check_output(["ip", "-j", "-details", "link", "show"], text=True))
        }
        from passive_can import check_feedback

        for name in names:
            link = links.get(name, {})
            try:
                check_feedback(name, link)
            except (ValueError, OSError) as exc:
                p.error(str(exc))
            state = link.get("linkinfo", {}).get("info_data", {}).get("state")
            print(
                "CAN %s: fresh status/joints/gripper feedback, no sampled errors (reported state=%s)" % (name, state),
                flush=True,
            )
    root = Path(__file__).resolve().parent
    logs = root / "logs" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs.mkdir(parents=True)
    manager = Supervisor(logs)
    context = None
    try:
        conflicts = existing_ros()
        if conflicts:
            raise RuntimeError("Existing ROS session: " + str(conflicts))
        if not args.camera_only and not args.hardware_config and can_problems():
            print("CAN needs configuration; enter sudo password in this terminal.", flush=True)
            subprocess.run(
                ["sudo", "bash", str(root / "vendor/Piper_ros_private-ros-noetic/can_config.sh")], check=True
            )
            if can_problems():
                raise RuntimeError("CAN verification failed after configuration")
        os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:11311"
        os.environ["ROS_IP"] = "127.0.0.1"
        os.environ.pop("ROS_HOSTNAME", None)
        manager.start("roscore", ["roscore"])
        manager.pause(3)
        manager.start("cameras", ["roslaunch", "astra_camera", "multi_camera.launch"])
        if not args.camera_only and not args.hardware_config:
            manager.start("arms", ["roslaunch", "piper", "start_ms_piper.launch", "mode:=0", "auto_enable:=false"])
        command = [
            sys.executable,
            str(root / "server.py"),
            "--bind-host",
            args.bind_host,
            "--control-port",
            str(args.control_port),
        ]
        for flag, value in [
            ("head-port", args.head_port),
            ("left-wrist-port", args.left_wrist_port),
            ("right-wrist-port", args.right_wrist_port),
        ]:
            command += ["--" + flag, str(value)]
        if args.hardware_config:
            command += ["--hardware-config", str(Path(args.hardware_config).resolve())]
        if args.execute_control:
            command += ["--execute-control"]
        if args.smooth_control:
            command += ["--smooth-control"]
        for key in ("control_frequency", "joint_acceleration", "gripper_acceleration", "tracking_response_s"):
            command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
        for key in ("joint_speed", "gripper_speed", "timeout", "feedback_timeout", "camera_timeout", "tracking_limit"):
            command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
        manager.start("bridge", command)
        import cv2
        import numpy as np
        import zmq

        context = zmq.Context()
        sockets = []
        counts = [0, 0, 0]
        last = [0.0, 0.0, 0.0]
        shapes = {}
        for _, port in camera_routes(args):
            s = context.socket(zmq.SUB)
            s.setsockopt(zmq.SUBSCRIBE, b"")
            s.setsockopt(zmq.LINGER, 0)
            s.connect("tcp://%s:%d" % ("127.0.0.1" if args.bind_host == "0.0.0.0" else args.bind_host, port))
            sockets.append(s)
        start = time.monotonic()
        ready = None
        while True:
            manager.check()
            for i, s in enumerate(sockets):
                if s.poll(10):
                    frame = cv2.imdecode(np.frombuffer(s.recv(), np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise RuntimeError("Invalid JPEG")
                    counts[i] += 1
                    last[i] = time.monotonic()
                    shapes[i] = list(frame.shape)
            now = time.monotonic()
            if ready is None and min(counts) >= 5:
                with socket.create_connection(
                    ("127.0.0.1" if args.bind_host == "0.0.0.0" else args.bind_host, args.control_port), timeout=3
                ) as c:
                    body = b'{"get_state":true}'
                    c.sendall(("%010d" % len(body)).encode() + body)
                    reply = json.loads(receive(c, int(receive(c, 10))))
                if not args.camera_only and not reply.get("ok"):
                    raise RuntimeError(str(reply))
                report = dict(frames=counts, shapes=shapes, state=reply)
                print("READY " + json.dumps(report), flush=True)
                (logs / "report.json").write_text(json.dumps(report, indent=2))
                ready = now
            if ready is None and now - start > 60:
                raise RuntimeError("Camera startup timeout: " + str(counts))
            if ready and any(now - t > 5 for t in last):
                raise RuntimeError("Camera stream stale")
            if ready and args.test_seconds and now - ready >= args.test_seconds:
                break
    except InterruptedError:
        pass
    finally:
        if context:
            for s in sockets:
                s.close()
            context.term()
        manager.cleanup()


if __name__ == "__main__":
    main()
