"""Listen only: report valid CAN IDs and Linux CAN errors; never sends frames."""

import argparse
import collections
import json
import socket
import struct
import subprocess
import time


def describe_error(identifier, data):
    labels = []
    if identifier & 0x20:
        labels.append("no_ack")
    if identifier & 0x40:
        labels.append("bus_off")
    if identifier & 0x04:
        if data[1] & 0x10:
            labels.append("rx_error_passive")
        if data[1] & 0x20:
            labels.append("tx_error_passive")
    if identifier & 0x08:
        labels.append("protocol_error")
    return labels or ["other_can_error"]


def listen(name, seconds):
    valid, errors = collections.Counter(), collections.Counter()
    samples = {}
    with socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as sock:
        # CAN_RAW_ERR_FILTER=2: ordinary raw sockets hide error frames.
        sock.setsockopt(socket.SOL_CAN_RAW, 2, struct.pack("I", 0x1FFFFFFF))
        sock.settimeout(0.2)
        sock.bind((name,))
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                packet = sock.recv(16)
            except socket.timeout:
                continue
            identifier, length, data = struct.unpack("=IB3x8s", packet)
            if identifier & 0x20000000:  # CAN_ERR_FLAG
                errors.update(describe_error(identifier, data))
                samples[hex(identifier)] = data[:length].hex()
            else:
                valid[hex(identifier & 0x1FFFFFFF)] += 1
    return dict(interface=name, valid_frames=dict(valid), errors=dict(errors), error_samples=samples)


# Status, three joint-pair messages, and gripper feedback required by PiperBimanual.
REQUIRED_FEEDBACK = ("0x2a1", "0x2a5", "0x2a6", "0x2a7", "0x2a8")


def check_feedback(name, link, seconds=1.0, listener=None):
    info = link.get("linkinfo", {}).get("info_data", {})
    state = info.get("state")
    if "UP" not in link.get("flags", []) or info.get("bittiming", {}).get("bitrate") != 1000000:
        raise ValueError("Configure %s UP at 1000000 bit/s before starting" % name)
    if state in ("BUS-OFF", "STOPPED", "SLEEPING"):
        raise ValueError("%s CAN state=%s; interface cannot communicate" % (name, state))
    # WARNING/PASSIVE alone cannot establish whether fresh feedback is arriving.
    report = (listener or listen)(name, seconds)
    if report.get("error") or report.get("errors") or report.get("error_samples"):
        raise ValueError("%s has current CAN errors: %s; run python passive_can.py" % (name, report))
    missing = [key for key in REQUIRED_FEEDBACK if report.get("valid_frames", {}).get(key, 0) < 2]
    if missing:
        raise ValueError(
            "%s missing repeated Piper feedback %s (CAN state=%s); check connection" % (name, ", ".join(missing), state)
        )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interfaces", nargs="+", default=["can_left", "can_right"])
    p.add_argument("--seconds", type=float, default=3)
    args = p.parse_args()
    if not 0 < args.seconds <= 60:
        p.error("seconds must be in (0, 60]")
    report = []
    for name in args.interfaces:
        result = subprocess.run(["ip", "-j", "-details", "link", "show", "dev", name], capture_output=True, text=True)
        try:
            item = listen(name, args.seconds)
        except OSError as exc:
            item = dict(interface=name, valid_frames={}, error=str(exc))
        if result.returncode == 0:
            link = json.loads(result.stdout)[0]
            info = link.get("linkinfo", {}).get("info_data", {})
            item.update(
                flags=link.get("flags"), can_state=info.get("state"), bitrate=info.get("bittiming", {}).get("bitrate")
            )
        report.append(item)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if any(not item["valid_frames"] or item.get("errors") or item.get("error") for item in report):
        print("CAN feedback not healthy. Check arm power, CAN port/wiring and bitrate; no motion commands were sent.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
