"""Annotate fixed wrist-camera occlusion polygons from a local LeRobot video.

Requires an OpenCV GUI session. Writes normalized polygons, not image paths, so
the resulting policy config is portable. This is an approximation for a fixed
camera/mount; use per-frame rendered masks for articulated/variable silhouettes.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def annotate(video: Path, camera: str, start_frame: int) -> list[list[float]]:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    success, frame = capture.read()
    if not success:
        capture.release()
        raise RuntimeError(f"Cannot decode {video}; use an OpenCV build with AV1 support")
    height, width = frame.shape[:2]
    points: list[tuple[int, int]] = []
    window = f"Cloak: {camera}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and 0 <= x < width and 0 <= y < height:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.setMouseCallback(window, mouse)
    print(f"{camera}: left click=vertex, right click/Backspace=undo, n=+30 frames, Enter=accept, Esc=cancel")
    try:
        while True:
            preview = frame.copy()
            if len(points) >= 3:
                overlay = preview.copy()
                cv2.fillPoly(overlay, [np.asarray(points, dtype=np.int32)], (0, 0, 255))
                preview = cv2.addWeighted(preview, 0.65, overlay, 0.35, 0)
            if points:
                cv2.polylines(preview, [np.asarray(points, dtype=np.int32)], len(points) >= 3, (0, 255, 255), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError("Annotation cancelled; output was not written")
            if key in (8, 127) and points:
                points.pop()
            elif key in (10, 13) and len(points) >= 3:
                return [[x / width, y / height] for x, y in points]
            elif key == ord("n"):
                for _ in range(30):
                    success, next_frame = capture.read()
                    if not success:
                        break
                    frame = next_frame
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--cameras", nargs="+", default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output path; {args.output} already exists")
    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    cameras = args.cameras or [key for key in info["features"] if "wrist" in key and key.startswith("observation.images.")]
    if not cameras:
        parser.error("No wrist cameras found; pass --cameras explicitly")
    polygons = {}
    for camera in cameras:
        if camera not in info["features"]:
            parser.error(f"Unknown camera: {camera}")
        videos = sorted((args.dataset_root / "videos" / camera).glob("**/*.mp4"))
        if not videos:
            raise FileNotFoundError(f"No MP4 files for {camera}")
        polygons[camera] = annotate(videos[0], camera, args.frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(polygons, stream, indent=2)
        stream.write("\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
