#!/usr/bin/env python3
"""Copy one local folder to S1/Orin with same-name replacement.

Examples:
  python send_code_to_S1.py ./astribot_sdk_aarch64 192.168.0.11 /home/jchen572/code_file/astribot_service/
  python send_code_to_S1.py ./astribot_image_service 192.168.0.11 /home/jchen572/code_file/astribot_service/

If the remote path already ends with the local folder name, it is treated as the
final destination. Otherwise, it is treated as a parent directory, matching:
  scp -r ./astribot_sdk_aarch64/ jchen572@192.168.0.11:/home/jchen572/code_file/astribot_service
"""

from __future__ import annotations

import argparse
import getpass
import posixpath
import shlex
import subprocess
from pathlib import Path, PurePosixPath


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def normalize_remote_path(path: str) -> str:
    path = path.strip()
    if not path:
        raise ValueError("remote path cannot be empty")
    if path == "~":
        return "."
    if path.startswith("~/"):
        return path[2:]
    return path.rstrip("/") or "/"


def final_remote_dir(local_dir: Path, remote_path: str) -> str:
    remote_path = normalize_remote_path(remote_path)
    remote_name = PurePosixPath(remote_path).name
    if remote_name == local_dir.name:
        return remote_path
    if remote_path == "/":
        return f"/{local_dir.name}"
    return posixpath.join(remote_path, local_dir.name)


def remote_parent(path: str) -> str:
    parent = posixpath.dirname(path.rstrip("/"))
    return parent or "."


def ssh_target(user: str, ip: str) -> str:
    if "@" in ip:
        return ip
    return f"{user}@{ip}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("local_folder", help="Local folder to send, for example ./astribot_sdk_aarch64.")
    parser.add_argument("target_ip", help="S1/Orin IP, or a full SSH target like jchen572@192.168.0.11.")
    parser.add_argument(
        "remote_path",
        help=(
            "Remote parent directory or final same-name directory. If the basename matches "
            "local_folder, it is used as the final destination."
        ),
    )
    parser.add_argument("--user", default=getpass.getuser(), help="SSH username when target_ip has no user.")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without copying.")
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Do not delete remote files that are absent locally. Default is same-name mirror replacement.",
    )
    args = parser.parse_args()

    local_dir = Path(args.local_folder).expanduser().resolve()
    if not local_dir.exists():
        raise FileNotFoundError(local_dir)
    if not local_dir.is_dir():
        raise NotADirectoryError(local_dir)

    target = ssh_target(args.user, args.target_ip)
    remote_final = final_remote_dir(local_dir, args.remote_path)
    remote_final_parent = remote_parent(remote_final)

    print(f"[send_code_to_S1] local:  {local_dir}")
    print(f"[send_code_to_S1] target: {target}")
    print(f"[send_code_to_S1] remote: {remote_final}")
    print("[send_code_to_S1] mode:   same-name mirror replacement")

    rsync_path = f"mkdir -p -- {shlex.quote(remote_final_parent)} && rsync"
    rsync_cmd = [
        "rsync",
        "-az",
        "--info=progress2",
        "--rsync-path",
        rsync_path,
        "--exclude",
        "__pycache__/",
        "--exclude",
        "*.pyc",
    ]
    if not args.no_delete:
        rsync_cmd.append("--delete")
    if args.dry_run:
        rsync_cmd.append("--dry-run")

    run([*rsync_cmd, f"{local_dir}/", f"{target}:{remote_final}/"])


if __name__ == "__main__":
    main()
