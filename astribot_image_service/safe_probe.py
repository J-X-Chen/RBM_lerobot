"""Bounded hardware diagnostic, independent of models and dataset poses.

Default: read feedback only. --execute requires safe service and terminal r.
Stopping publication is NOT a hardware emergency stop.
"""
import argparse
import json
from pathlib import Path
import select
import socket
import sys
import termios
import time
import tty

import numpy as np

from .server import recv_json, send_json, STATE_COMPONENTS

JOINTS = np.r_[3:14, 15:22, 23:25]
ARMS = tuple(range(7, 14)) + tuple(range(15, 22))
LABELS = [f"{name}[{i}]" for name, size in STATE_COMPONENTS for i in range(size)]


def request(sock, payload, *, allow_fault=False):
    send_json(sock, dict(payload, required_control_mode="safe"))
    response = recv_json(sock, timeout=.3)
    control = response.get("control", {})
    if not response.get("ok") or control.get("control_mode") != "safe":
        raise RuntimeError(f"Service refused request: {response}")
    if control.get("safe_protocol_version") != 2:
        raise RuntimeError("Update the S1 service before this test")
    if control.get("fault") and not allow_fault:
        raise RuntimeError(f"Service fault: {control['fault']}")
    return response


def measured(response):
    q = np.asarray(response["state"], dtype=float)
    if q.shape != (25,) or not np.isfinite(q).all():
        raise RuntimeError("Invalid 25D feedback")
    return q


def probe_target(origin, joint, delta, elapsed):
    target = origin.copy()
    if joint is not None:
        target[joint] += np.sign(delta) * min(abs(delta), max(0., elapsed - 2.) * .005)
    return target


def check_feedback(origin, current, previous, dt, joint):
    if dt <= 0 or dt > .5:
        raise RuntimeError("Feedback interval invalid or exceeded 0.5s")
    drift = np.abs(current - origin)
    if np.max(drift[JOINTS]) > .03:
        raise RuntimeError("Joint displacement exceeded 0.03 rad")
    other = [int(i) for i in JOINTS if i != joint]
    if np.max(drift[other]) > .01:
        raise RuntimeError("Uncommanded joint moved over 0.01 rad; inspect mapping/control")
    if np.max(np.abs(current[JOINTS] - previous[JOINTS])) / dt > .1:
        raise RuntimeError("Measured joint speed exceeded 0.1 rad/s")
    if np.max(drift[[14, 22]]) > 2:
        raise RuntimeError("Uncommanded gripper changed by more than 2 units")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.0.11")
    parser.add_argument("--port", type=int, default=6200)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--joint", type=int, choices=ARMS)
    parser.add_argument("--delta", type=float, default=.01)
    parser.add_argument("--log", default="eval_action_logs/safe_probe.jsonl")
    args = parser.parse_args()
    if not np.isfinite(args.delta) or abs(args.delta) > .02:
        parser.error("abs(delta) must be <= 0.02 rad")
    path = Path(args.log)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive file creation preserves earlier diagnostic evidence.
    with path.open("x") as log, socket.create_connection((args.host, args.port), timeout=.3) as sock:
        response = request(sock, {"get_state": True})
        origin = measured(response)
        print(json.dumps(dict(zip(LABELS, origin.tolist())), indent=2), flush=True)
        if not args.execute:
            log.write(json.dumps({"state": origin.tolist(), "control": response["control"]}) + "\n")
            return
        if not response["control"].get("execute"):
            raise RuntimeError("Service is dry-run; physical test requires --execute-control")
        if response["control"].get("has_target"):
            raise RuntimeError("Service already has a target; stop other clients and restart service")
        if not sys.stdin.isatty():
            raise RuntimeError("Interactive terminal required")
        print("Physical test: 8 seconds, no return motion. q / Ctrl-C ends publication; physical E-stop must be ready.")
        print(f"Joint: {LABELS[args.joint] if args.joint is not None else 'HOLD CURRENT'}, delta={args.delta if args.joint is not None else 0}")
        if input("Type r then Enter to begin: ").strip() != "r":
            return
        old = termios.tcgetattr(sys.stdin)
        armed = False
        try:
            tty.setcbreak(sys.stdin.fileno())
            response = request(sock, {"get_state": True})
            if response["control"].get("has_target"):
                raise RuntimeError("Another client supplied a target while awaiting confirmation")
            origin = measured(response)
            previous = origin.copy()
            started = last = time.monotonic()
            armed = True
            while time.monotonic() - started < 8:
                if select.select([sys.stdin], [], [], 0)[0] and sys.stdin.read(1).lower() == "q":
                    break
                response = request(sock, {"get_state": True})
                current = measured(response)
                now = time.monotonic()
                log.write(json.dumps({"t": now - started, "state": current.tolist(),
                                      "origin": origin.tolist(), "control": response["control"]}) + "\n")
                log.flush()
                check_feedback(origin, current, previous, now - last, args.joint)
                target = probe_target(origin, args.joint, args.delta, now - started)
                request(sock, {"action": target.tolist()})
                previous, last = current, now
                time.sleep(.05)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
            if armed:
                # Fresh connection also handles a failed/partially received response.
                try:
                    with socket.create_connection((args.host, args.port), timeout=.3) as stop_sock:
                        request(stop_sock, {"disarm": True}, allow_fault=True)
                    print("Disarmed; no auto-return. Inspect robot before restarting service.")
                except Exception as exc:
                    print(f"DISARM FAILED: {exc}; use physical E-stop", file=sys.stderr)


if __name__ == "__main__":
    main()
