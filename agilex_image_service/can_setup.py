"""Configure arm CAN interfaces by USB location; never send arm commands."""

import json
import subprocess


def configure_can(hardware):
    links = json.loads(subprocess.check_output(
        ["ip", "-j", "-details", "link", "show", "type", "can"], text=True))
    by_name = {link["ifname"]: link for link in links}
    default_usb = {"can_left": "1-13:1.0", "can_right": "1-12:1.0"}
    plans = []
    used = set()
    for side in ("left", "right"):
        config = hardware[side]
        target = config["can"]
        source = target
        if target not in by_name:
            usb = config.get("usb_port", default_usb.get(target))
            if not usb:
                raise ValueError("Missing CAN %s: configure usb_port in hardware.json" % target)
            matches = []
            for name in by_name:
                info = subprocess.check_output(["ethtool", "-i", name], text=True)
                if any(line.strip() == "bus-info: " + usb for line in info.splitlines()):
                    matches.append(name)
            if len(matches) != 1:
                raise ValueError("Cannot uniquely locate CAN %s at USB %s" % (target, usb))
            source = matches[0]
        if source in used or (source != target and source in [hardware[s]["can"] for s in ("left", "right")]):
            raise ValueError("Conflicting CAN mapping; check hardware.json")
        used.add(source)
        link = by_name[source]
        bitrate = link.get("linkinfo", {}).get("info_data", {}).get("bittiming", {}).get("bitrate")
        if source != target or "UP" not in link.get("flags", []) or bitrate != 1000000:
            plans.append((source, target))
    if not plans:
        return
    print("Configuring arm CAN interfaces; enter sudo password if requested (no arm enable).", flush=True)
    subprocess.run(["sudo", "-v"], check=True)
    for source, target in plans:
        subprocess.run(["sudo", "ip", "link", "set", "dev", source, "down"], check=True)
        if source != target:
            subprocess.run(["sudo", "ip", "link", "set", "dev", source, "name", target], check=True)
        subprocess.run(["sudo", "ip", "link", "set", "dev", target, "type", "can", "bitrate", "1000000"], check=True)
        subprocess.run(["sudo", "ip", "link", "set", "dev", target, "up"], check=True)
