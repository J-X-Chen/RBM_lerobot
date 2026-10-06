"""
Convert a thermal video to grayscale while keeping 3 identical RGB channels.

The script reads the input video, converts every frame to grayscale, and writes
back to the same path by replacing the original file after processing.

Example:
    python turn_thermal_to_grey.py \
        lerobotv3_datasets/cold_water_bottle_selection/videos/observation.images.cam_thermal/chunk-000/file-000.mp4
"""

from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import tempfile


DEFAULT_VIDEO = Path(
    "lerobotv3_datasets/cold_water_bottle_selection/videos/observation.images.cam_thermal/chunk-000/file-000.mp4"
)


def process_video(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")

    input_path = input_path.resolve()
    parent = input_path.parent
    suffix = input_path.suffix or ".mp4"

    with tempfile.NamedTemporaryFile(dir=parent, suffix=suffix, delete=False) as tmp:
        temp_path = Path(tmp.name)

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vf",
            "format=gray,format=rgb24",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(temp_path),
        ]
        subprocess.run(cmd, check=True)
        os.replace(temp_path, input_path)
        print(f"Done: {input_path}")
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a video to grayscale with 3 identical RGB channels")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_VIDEO), help="input video path")
    args = parser.parse_args()

    input_path = Path(args.path)
    if not input_path.exists():
        print(f"Path does not exist: {input_path}")
        sys.exit(1)

    if input_path.is_dir():
        print("Please provide a video file, not a directory.")
        sys.exit(1)

    process_video(input_path)


if __name__ == "__main__":
    main()

