# python unitree_lerobot/utils/match_and_merge_rgb_and_thermography.py   ./cold_water_bottle_selection   --input_format lerobot   --overwrite   --alpha 0.7   --boundary-feather 50   --thermal-border-crop 30   --match-every-n-frames 2
import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for this script
    tqdm = None


COLOR_RE = re.compile(r"^(?P<frame_id>.+)_color_0\.jpg$", re.IGNORECASE)
METRIC_FIELDS = [
    "status",
    "color_path",
    "thermal_path",
    "output_path",
    "raw_matches",
    "matches",
    "inliers",
    "inlier_ratio",
    "reproj_mean_px",
    "reproj_median_px",
    "edge_precision",
    "edge_recall",
    "edge_f1",
    "edge_iou",
    "color_edge_pixels",
    "thermal_edge_pixels",
    "warped_coverage",
    "homography_center_ratio",
    "metric_center_ratio",
    "center_fallback",
    "center_weight_strength",
    "center_weight_sigma",
    "weighted_refit",
    "modal_fill",
    "boundary_feather",
    "thermal_border_crop",
    "video_path",
    "frame_index",
    "fallback_reason",
]

DEFAULT_RGB_FEATURE = "observation.images.cam_left_high"
DEFAULT_THERMAL_FEATURE = "observation.images.cam_thermal"
DEFAULT_OUTPUT_FEATURE = "observation.images.cam_rgb_thermal_mixing"


def add_matching_arguments(parser, method):
    if method == "sp_lg":
        parser.add_argument("--ckpt", type=str, default="./weights/minima_lightglue.pth")
    elif method == "roma":
        parser.add_argument("--ckpt2", type=str, default="large")
        parser.add_argument("--ckpt", type=str, default="./weights/minima_roma.pth")
    else:
        raise ValueError(f"Unknown method: {method}")


def import_runtime_dependencies(minima_root=None):
    if minima_root is not None:
        minima_root = Path(minima_root).expanduser()
        if minima_root.is_dir():
            for python_path in (
                minima_root,
                minima_root / "third_party" / "RoMa_minima",
            ):
                if not python_path.is_dir():
                    continue
                resolved_path = str(python_path.resolve())
                if resolved_path not in sys.path:
                    sys.path.insert(0, resolved_path)

    try:
        import cv2
        import torch
        from load_model import load_model
    except ImportError as exc:
        raise ImportError(
            "Missing runtime dependency. Activate the project environment or install "
            "requirements before processing images."
        ) from exc

    return cv2, torch, load_model


def patch_runtime_compatibility(torch):
    """Keep older project utilities working on CPU-only or newer NumPy setups."""
    if not hasattr(np, "float"):
        np.float = float

    if not torch.cuda.is_available():
        torch.cuda.synchronize = lambda *args, **kwargs: None


def iter_episode_dirs(root_dir):
    root_dir = Path(root_dir)
    episode_dirs = []

    if root_dir.is_dir() and root_dir.name.startswith("episode"):
        episode_dirs.append(root_dir)

    episode_dirs.extend(p for p in root_dir.rglob("episode*") if p.is_dir())
    return sorted(set(episode_dirs))


def iter_image_pairs(root_dir, output_dir_name):
    for episode_dir in iter_episode_dirs(root_dir):
        colors_dir = episode_dir / "colors"
        if not colors_dir.is_dir():
            colors_dir = episode_dir / "images"
        thermal_dir = episode_dir / "thermography"
        if not colors_dir.is_dir() or not thermal_dir.is_dir():
            continue

        for color_path in sorted(colors_dir.iterdir()):
            if not color_path.is_file():
                continue

            match = COLOR_RE.match(color_path.name)
            if match is None:
                continue

            frame_id = match.group("frame_id")
            thermal_path = thermal_dir / f"{frame_id}_thermal_0.jpg"
            output_path = episode_dir / output_dir_name / f"{frame_id}_mixing_0.jpg"
            yield color_path, thermal_path, output_path


def get_feature_video_dir(dataset_path, feature):
    return Path(dataset_path) / "videos" / feature


def probe_video(path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,duration,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    import json

    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found in: {path}")

    stream = streams[0]
    frame_rate = Fraction(stream["avg_frame_rate"])
    if frame_rate <= 0:
        raise ValueError(f"Could not read a valid frame rate from: {path}")
    duration = float(stream.get("duration") or 0.0)
    frame_count = None
    if stream.get("nb_frames") not in (None, "N/A"):
        frame_count = int(stream["nb_frames"])
        if duration <= 0:
            duration = frame_count / float(frame_rate)
    if duration <= 0:
        raise ValueError(f"Could not read a valid duration from: {path}")

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": frame_rate,
        "duration": duration,
        "frame_count": frame_count,
    }


def iter_lerobot_video_jobs(root_dir, rgb_feature, thermal_feature, output_feature):
    root_dir = Path(root_dir)
    rgb_root = get_feature_video_dir(root_dir, rgb_feature)
    thermal_root = get_feature_video_dir(root_dir, thermal_feature)
    output_root = get_feature_video_dir(root_dir, output_feature)

    if not rgb_root.is_dir():
        raise FileNotFoundError(f"RGB video feature directory does not exist: {rgb_root}")
    if not thermal_root.is_dir():
        raise FileNotFoundError(f"Thermal video feature directory does not exist: {thermal_root}")

    rgb_files = sorted(rgb_root.rglob("*.mp4"))
    thermal_files = sorted(thermal_root.rglob("*.mp4"))
    if not rgb_files:
        raise FileNotFoundError(f"No RGB mp4 files found under: {rgb_root}")
    if not thermal_files:
        raise FileNotFoundError(f"No thermal mp4 files found under: {thermal_root}")

    jobs = []
    missing_thermal_files = []
    cumulative_rgb_duration_s = 0.0
    use_single_thermal_timeline = len(thermal_files) == 1
    for rgb_path in rgb_files:
        relative_path = rgb_path.relative_to(rgb_root)
        matching_thermal_path = thermal_root / relative_path
        output_path = output_root / relative_path
        if matching_thermal_path.is_file():
            thermal_path = matching_thermal_path
            thermal_start_s = 0.0
        elif use_single_thermal_timeline:
            thermal_path = thermal_files[0]
            thermal_start_s = cumulative_rgb_duration_s
        else:
            missing_thermal_files.append(matching_thermal_path)
            cumulative_rgb_duration_s += float(probe_video(rgb_path)["duration"])
            continue

        jobs.append((rgb_path, thermal_path, output_path, thermal_start_s))
        cumulative_rgb_duration_s += float(probe_video(rgb_path)["duration"])

    if missing_thermal_files:
        preview = "\n".join(f"  - {path}" for path in missing_thermal_files[:10])
        suffix = "" if len(missing_thermal_files) <= 10 else f"\n  ... and {len(missing_thermal_files) - 10} more"
        raise FileNotFoundError(
            "Thermal videos do not mirror RGB chunk/file layout, and thermal is not a single timeline:\n"
            f"{preview}{suffix}"
        )

    return jobs


def center_region_mask_for_points(points, image_shape, ratio):
    if ratio >= 1.0:
        return np.ones(len(points), dtype=bool)

    height, width = image_shape[:2]
    x_min = width * (1.0 - ratio) / 2.0
    x_max = width * (1.0 + ratio) / 2.0
    y_min = height * (1.0 - ratio) / 2.0
    y_max = height * (1.0 + ratio) / 2.0

    return (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
    )


def center_region_mask_for_image(image_shape, ratio):
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    if ratio >= 1.0:
        mask[:, :] = True
        return mask

    x_min = int(round(width * (1.0 - ratio) / 2.0))
    x_max = int(round(width * (1.0 + ratio) / 2.0))
    y_min = int(round(height * (1.0 - ratio) / 2.0))
    y_max = int(round(height * (1.0 + ratio) / 2.0))
    mask[y_min:y_max, x_min:x_max] = True
    return mask


def center_weights_for_points(points, image_shape, strength, sigma):
    if strength <= 0.0:
        return np.ones(len(points), dtype=np.float64)

    height, width = image_shape[:2]
    center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    half_size = np.array([max(width / 2.0, 1.0), max(height / 2.0, 1.0)], dtype=np.float64)
    normalized = (points.astype(np.float64) - center) / half_size
    radius = np.linalg.norm(normalized, axis=1) / np.sqrt(2.0)
    attention = np.exp(-0.5 * (radius / max(sigma, 1e-6)) ** 2)
    return 1.0 + strength * attention


def get_modal_bgr_color(image, border_crop=0):
    """Return the most frequent BGR color in an OpenCV image."""
    if image is None or image.size == 0:
        return (0, 0, 0)

    border_crop = int(border_crop)
    if border_crop > 0 and image.shape[0] > border_crop * 2 and image.shape[1] > border_crop * 2:
        image = image[border_crop:-border_crop, border_crop:-border_crop]

    pixels = np.ascontiguousarray(image.reshape(-1, image.shape[-1]))
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    return tuple(int(value) for value in colors[int(np.argmax(counts))])


def make_source_border_mask(image_shape, border_crop):
    height, width = image_shape[:2]
    mask = np.full((height, width), 255, dtype=np.uint8)
    border_crop = int(border_crop)
    if border_crop <= 0:
        return mask
    if border_crop * 2 >= min(height, width):
        raise ValueError(
            f"--thermal-border-crop={border_crop} is too large for thermal image size {width}x{height}."
        )

    mask[:border_crop, :] = 0
    mask[-border_crop:, :] = 0
    mask[:, :border_crop] = 0
    mask[:, -border_crop:] = 0
    return mask


def make_feathered_mask(mask, feather_radius):
    import cv2

    mask = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)
    if feather_radius <= 0:
        return mask

    kernel_size = int(feather_radius) * 2 + 1
    return np.clip(cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0), 0.0, 1.0)


def normalize_points(points):
    points = points.astype(np.float64)
    mean = points.mean(axis=0)
    centered = points - mean
    mean_distance = np.mean(np.linalg.norm(centered, axis=1))
    scale = np.sqrt(2.0) / max(mean_distance, 1e-8)
    transform = np.array(
        [
            [scale, 0.0, -scale * mean[0]],
            [0.0, scale, -scale * mean[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    normalized = np.column_stack(
        [
            points[:, 0] * scale - mean[0] * scale,
            points[:, 1] * scale - mean[1] * scale,
        ]
    )
    return normalized, transform


def weighted_homography(src_points, dst_points, weights):
    if len(src_points) < 4:
        return None

    src_norm, src_transform = normalize_points(src_points)
    dst_norm, dst_transform = normalize_points(dst_points)
    row_weights = np.sqrt(np.maximum(weights.astype(np.float64), 1e-8))

    rows = []
    for (x, y), (u, v), weight in zip(src_norm, dst_norm, row_weights):
        rows.append(weight * np.array([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]))
        rows.append(weight * np.array([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]))

    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homography = vt[-1].reshape(3, 3)
    homography = np.linalg.inv(dst_transform) @ homography @ src_transform
    if abs(homography[2, 2]) < 1e-12:
        return None
    homography = homography / homography[2, 2]
    if not np.isfinite(homography).all():
        return None
    return homography.astype(np.float64)


def is_cv_image(value):
    return isinstance(value, np.ndarray)


def read_bgr_image(image_or_path, label):
    if is_cv_image(image_or_path):
        return image_or_path

    import cv2

    image = cv2.imread(str(image_or_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {label} image: {image_or_path}")
    return image


def estimate_thermal_to_color_homography(
    matcher,
    color_input,
    thermal_input,
    ransac_reproj_threshold,
    min_matches,
    min_inliers,
    homography_center_ratio,
    center_fallback_to_all,
    center_weight_strength,
    center_weight_sigma,
):
    if is_cv_image(color_input):
        match_res = matcher(color_input, thermal_input)
    else:
        match_res = matcher(str(color_input), str(thermal_input))
    mkpts_color = np.asarray(match_res["mkpts0"], dtype=np.float32)
    mkpts_thermal = np.asarray(match_res["mkpts1"], dtype=np.float32)
    raw_match_count = int(len(mkpts_color))
    metrics = {
        "raw_matches": raw_match_count,
        "matches": raw_match_count,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "reproj_mean_px": np.nan,
        "reproj_median_px": np.nan,
        "homography_center_ratio": homography_center_ratio,
        "center_fallback": 0,
        "center_weight_strength": center_weight_strength,
        "center_weight_sigma": center_weight_sigma,
        "weighted_refit": 0,
    }

    img0 = match_res.get("img0")
    if img0 is None and (homography_center_ratio < 1.0 or center_weight_strength > 0.0):
        img0 = read_bgr_image(color_input, "color")

    if homography_center_ratio < 1.0:
        center_mask = center_region_mask_for_points(mkpts_color, img0.shape, homography_center_ratio)
        if int(center_mask.sum()) >= min_matches:
            mkpts_color = mkpts_color[center_mask]
            mkpts_thermal = mkpts_thermal[center_mask]
        elif not center_fallback_to_all:
            mkpts_color = mkpts_color[center_mask]
            mkpts_thermal = mkpts_thermal[center_mask]
        else:
            metrics["center_fallback"] = 1

        metrics["matches"] = int(len(mkpts_color))

    if len(mkpts_color) < min_matches:
        return None, metrics

    import cv2

    homography, inliers = cv2.findHomography(
        mkpts_thermal,
        mkpts_color,
        cv2.RANSAC,
        ransac_reproj_threshold,
    )
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    metrics["inliers"] = inlier_count
    metrics["inlier_ratio"] = inlier_count / max(len(mkpts_color), 1)

    if homography is None or inlier_count < min_inliers:
        return None, metrics

    inlier_mask = inliers.ravel().astype(bool)
    if center_weight_strength > 0.0 and img0 is not None and int(inlier_mask.sum()) >= 4:
        weights = center_weights_for_points(
            mkpts_color[inlier_mask],
            img0.shape,
            center_weight_strength,
            center_weight_sigma,
        )
        weighted_h = weighted_homography(
            mkpts_thermal[inlier_mask],
            mkpts_color[inlier_mask],
            weights,
        )
        if weighted_h is not None:
            homography = weighted_h
            metrics["weighted_refit"] = 1

    projected = cv2.perspectiveTransform(mkpts_thermal.reshape(-1, 1, 2), homography).reshape(-1, 2)
    reproj_errors = np.linalg.norm(projected - mkpts_color, axis=1)
    inlier_errors = reproj_errors[inlier_mask]
    if len(inlier_errors) > 0:
        metrics["reproj_mean_px"] = float(np.mean(inlier_errors))
        metrics["reproj_median_px"] = float(np.median(inlier_errors))

    return homography, metrics


def make_edges(image, canny_low, canny_high):
    import cv2

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(gray, canny_low, canny_high) > 0


def compute_edge_alignment_metrics(
    color_img,
    warped_thermal,
    warped_mask,
    canny_low,
    canny_high,
    edge_dilate,
    metric_center_ratio,
):
    import cv2

    valid_mask = warped_mask > 0
    valid_mask &= center_region_mask_for_image(valid_mask.shape, metric_center_ratio)
    color_edges = make_edges(color_img, canny_low, canny_high) & valid_mask
    thermal_edges = make_edges(warped_thermal, canny_low, canny_high) & valid_mask

    if edge_dilate > 0:
        kernel_size = edge_dilate * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        color_tolerant = cv2.dilate(color_edges.astype(np.uint8), kernel) > 0
        thermal_tolerant = cv2.dilate(thermal_edges.astype(np.uint8), kernel) > 0
    else:
        color_tolerant = color_edges
        thermal_tolerant = thermal_edges

    thermal_count = int(thermal_edges.sum())
    color_count = int(color_edges.sum())
    precision_hits = int((thermal_edges & color_tolerant).sum())
    recall_hits = int((color_edges & thermal_tolerant).sum())

    precision = precision_hits / thermal_count if thermal_count else np.nan
    recall = recall_hits / color_count if color_count else np.nan
    if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = np.nan

    edge_union = (color_tolerant | thermal_tolerant) & valid_mask
    edge_intersection = (color_tolerant & thermal_tolerant) & valid_mask
    edge_iou = edge_intersection.sum() / edge_union.sum() if edge_union.any() else np.nan

    return {
        "edge_precision": float(precision) if np.isfinite(precision) else np.nan,
        "edge_recall": float(recall) if np.isfinite(recall) else np.nan,
        "edge_f1": float(f1) if np.isfinite(f1) else np.nan,
        "edge_iou": float(edge_iou) if np.isfinite(edge_iou) else np.nan,
        "color_edge_pixels": color_count,
        "thermal_edge_pixels": thermal_count,
        "warped_coverage": float(valid_mask.sum() / valid_mask.size),
        "metric_center_ratio": metric_center_ratio,
    }


def make_mixing_from_images(
    color_img,
    thermal_img,
    homography,
    alpha,
    modal_fill=True,
    boundary_feather=15,
    thermal_border_crop=10,
):
    import cv2

    height, width = color_img.shape[:2]
    warped_thermal = cv2.warpPerspective(thermal_img, homography, (width, height))

    thermal_mask = make_source_border_mask(thermal_img.shape[:2], thermal_border_crop)
    warped_mask = cv2.warpPerspective(thermal_mask, homography, (width, height))
    feathered_mask = make_feathered_mask(warped_mask, boundary_feather)

    if modal_fill:
        modal_color = get_modal_bgr_color(thermal_img, thermal_border_crop)
        thermal_layer = np.empty_like(warped_thermal)
        thermal_layer[:, :] = modal_color
        thermal_layer = (
            thermal_layer.astype(np.float32) * (1.0 - feathered_mask[..., None])
            + warped_thermal.astype(np.float32) * feathered_mask[..., None]
        )
    else:
        thermal_layer = warped_thermal.astype(np.float32)

    if alpha >= 1.0:
        mixed = thermal_layer
    else:
        if modal_fill:
            alpha_map = np.full((height, width, 1), alpha, dtype=np.float32)
        else:
            alpha_map = (feathered_mask * alpha)[..., None]
        mixed = (
            color_img.astype(np.float32) * (1.0 - alpha_map)
            + thermal_layer.astype(np.float32) * alpha_map
        )
    return np.clip(mixed, 0, 255).astype(np.uint8), color_img, warped_thermal, warped_mask


def make_mixing_image(
    color_input,
    thermal_input,
    homography,
    alpha,
    modal_fill=True,
    boundary_feather=15,
    thermal_border_crop=10,
):
    color_img = read_bgr_image(color_input, "color")
    thermal_img = read_bgr_image(thermal_input, "thermal")
    return make_mixing_from_images(
        color_img,
        thermal_img,
        homography,
        alpha,
        modal_fill,
        boundary_feather,
        thermal_border_crop,
    )


def save_mixing_image(
    matcher,
    color_path,
    thermal_path,
    output_path,
    alpha,
    ransac_reproj_threshold,
    min_matches,
    min_inliers,
    canny_low,
    canny_high,
    edge_dilate,
    homography_center_ratio,
    metric_center_ratio,
    center_fallback_to_all,
    center_weight_strength,
    center_weight_sigma,
    modal_fill=True,
    boundary_feather=15,
    thermal_border_crop=10,
):
    import cv2

    homography, metrics = estimate_thermal_to_color_homography(
        matcher,
        color_path,
        thermal_path,
        ransac_reproj_threshold,
        min_matches,
        min_inliers,
        homography_center_ratio,
        center_fallback_to_all,
        center_weight_strength,
        center_weight_sigma,
    )
    if homography is None:
        return False, metrics

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mixed, color_img, warped_thermal, warped_mask = make_mixing_image(
        color_path,
        thermal_path,
        homography,
        alpha,
        modal_fill,
        boundary_feather,
        thermal_border_crop,
    )
    metrics["modal_fill"] = int(modal_fill)
    metrics["boundary_feather"] = int(boundary_feather)
    metrics["thermal_border_crop"] = int(thermal_border_crop)
    metrics.update(
        compute_edge_alignment_metrics(
            color_img,
            warped_thermal,
            warped_mask,
            canny_low,
            canny_high,
            edge_dilate,
            metric_center_ratio,
        )
    )
    if not cv2.imwrite(str(output_path), mixed):
        raise ValueError(f"Could not write output image: {output_path}")

    return True, metrics


def make_mixing_frame_with_fallback(
    matcher,
    color_path,
    thermal_path,
    alpha,
    ransac_reproj_threshold,
    min_matches,
    min_inliers,
    canny_low,
    canny_high,
    edge_dilate,
    homography_center_ratio,
    metric_center_ratio,
    center_fallback_to_all,
    center_weight_strength,
    center_weight_sigma,
    fallback_homography=None,
    modal_fill=True,
    boundary_feather=15,
    thermal_border_crop=10,
):
    homography, metrics = estimate_thermal_to_color_homography(
        matcher,
        color_path,
        thermal_path,
        ransac_reproj_threshold,
        min_matches,
        min_inliers,
        homography_center_ratio,
        center_fallback_to_all,
        center_weight_strength,
        center_weight_sigma,
    )
    fallback_reason = ""
    if homography is None and fallback_homography is not None:
        homography = fallback_homography
        fallback_reason = "previous_homography"
    if homography is None:
        metrics["fallback_reason"] = "color_frame"
        return None, metrics, fallback_homography

    mixed, color_img, warped_thermal, warped_mask = make_mixing_image(
        color_path,
        thermal_path,
        homography,
        alpha,
        modal_fill,
        boundary_feather,
        thermal_border_crop,
    )
    metrics["modal_fill"] = int(modal_fill)
    metrics["boundary_feather"] = int(boundary_feather)
    metrics["thermal_border_crop"] = int(thermal_border_crop)
    metrics.update(
        compute_edge_alignment_metrics(
            color_img,
            warped_thermal,
            warped_mask,
            canny_low,
            canny_high,
            edge_dilate,
            metric_center_ratio,
        )
    )
    metrics["fallback_reason"] = fallback_reason
    return mixed, metrics, homography if fallback_reason == "" else fallback_homography


def resolve_minima_paths(args):
    minima_root = Path(args.minima_root).expanduser()
    if hasattr(args, "ckpt"):
        ckpt = Path(args.ckpt).expanduser()
        if not ckpt.is_file() and not ckpt.is_absolute():
            candidate = minima_root / ckpt
            if candidate.is_file():
                args.ckpt = str(candidate)


class FfmpegVideoReader:
    """Sequential video reader backed by ffmpeg rawvideo pipes.

    OpenCV often fails to decode AV1 on machines without matching hardware
    acceleration.  ffmpeg's CLI can still decode these files in software, so
    LeRobot video mode uses this small pipe wrapper instead of cv2.VideoCapture.
    """

    def __init__(self, path, width, height, start_s=0.0):
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.frame_size = self.width * self.height * 3
        command = ["ffmpeg", "-v", "error"]
        if start_s > 0:
            command.extend(["-ss", f"{start_s:.6f}"])
        command.extend(
            [
                "-i",
                str(self.path),
                "-map",
                "0:v:0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ]
        )
        self.proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.frame_size * 4,
        )

    def read(self):
        if self.proc.stdout is None:
            return False, None
        data = self.proc.stdout.read(self.frame_size)
        if len(data) != self.frame_size:
            return False, None
        frame = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        return True, frame.copy()

    def release(self):
        if self.proc.stdout is not None:
            self.proc.stdout.close()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


class FfmpegVideoWriter:
    """Sequential mp4 writer backed by ffmpeg rawvideo pipes."""

    def __init__(self, path, width, height, frame_rate, output_vcodec):
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.frame_size = self.width * self.height * 3
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(frame_rate),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            output_vcodec,
            "-pix_fmt",
            "yuv420p",
        ]
        if output_vcodec == "libx264":
            command.extend(["-preset", "medium", "-crf", "18"])
        command.extend(["-movflags", "+faststart", str(self.path)])
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.frame_size * 4,
        )

    def write(self, frame):
        if self.proc.stdin is None:
            raise ValueError(f"ffmpeg writer stdin is closed for: {self.path}")
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Output frame shape {frame.shape[:2]} does not match writer size "
                f"{self.height}x{self.width}: {self.path}"
            )
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self):
        stderr = b""
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        if self.proc.stderr is not None:
            stderr = self.proc.stderr.read()
        return_code = self.proc.wait()
        if return_code != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed while writing {self.path}: {message}")


def process_lerobot_video_job(
    matcher,
    job,
    args,
    remaining_frames,
    progress_bar=None,
):
    rgb_path, thermal_path, output_path, thermal_start_s = job
    rgb_info = probe_video(rgb_path)
    thermal_info = probe_video(thermal_path)
    if rgb_info["frame_rate"] != thermal_info["frame_rate"]:
        raise ValueError(
            f"RGB and thermal fps differ for {rgb_path} / {thermal_path}: "
            f"{rgb_info['frame_rate']} vs {thermal_info['frame_rate']}"
        )

    fps = float(rgb_info["frame_rate"])
    frame_offset = int(round(thermal_start_s * fps))
    width = int(rgb_info["width"])
    height = int(rgb_info["height"])
    thermal_width = int(thermal_info["width"])
    thermal_height = int(thermal_info["height"])
    frame_count = rgb_info["frame_count"]

    if output_path.exists() and not args.overwrite:
        return {
            "saved": 0,
            "failed": 0,
            "skipped_existing": 1,
            "rows": [],
            "processed": 0,
        }

    rgb_reader = FfmpegVideoReader(rgb_path, width, height)
    thermal_reader = FfmpegVideoReader(thermal_path, thermal_width, thermal_height, thermal_start_s)
    writer = FfmpegVideoWriter(
        output_path,
        width,
        height,
        rgb_info["frame_rate"],
        args.output_vcodec,
    )

    saved = 0
    failed = 0
    rows = []
    processed = 0
    previous_homography = None
    max_frames_for_job = frame_count if frame_count is not None else int(1e18)
    if remaining_frames is not None:
        max_frames_for_job = min(max_frames_for_job, remaining_frames)

    for frame_index in range(max_frames_for_job):
        ok_rgb, color_frame = rgb_reader.read()
        ok_thermal, thermal_frame = thermal_reader.read()
        if not ok_rgb or not ok_thermal:
            break

        row = {
            "status": "error",
            "color_path": f"{rgb_path}#{frame_index}",
            "thermal_path": f"{thermal_path}#{frame_offset + frame_index}",
            "output_path": str(output_path),
            "video_path": str(output_path),
            "frame_index": frame_index,
        }
        try:
            reuse_previous_homography = (
                args.match_every_n_frames > 1
                and previous_homography is not None
                and frame_index % args.match_every_n_frames != 0
            )
            if reuse_previous_homography:
                mixed_frame, _, _, _ = make_mixing_image(
                    color_frame,
                    thermal_frame,
                    previous_homography,
                    args.alpha,
                    not args.no_modal_fill,
                    args.boundary_feather,
                    args.thermal_border_crop,
                )
                metrics = {
                    "raw_matches": 0,
                    "matches": 0,
                    "inliers": 0,
                    "fallback_reason": "reused_homography",
                    "modal_fill": int(not args.no_modal_fill),
                    "boundary_feather": int(args.boundary_feather),
                    "thermal_border_crop": int(args.thermal_border_crop),
                }
            else:
                mixed_frame, metrics, previous_homography = make_mixing_frame_with_fallback(
                    matcher,
                    color_frame,
                    thermal_frame,
                    args.alpha,
                    args.ransac_reproj_threshold,
                    args.min_matches,
                    args.min_inliers,
                    args.edge_canny_low,
                    args.edge_canny_high,
                    args.edge_dilate,
                    args.homography_center_ratio,
                    args.metric_center_ratio,
                    not args.no_center_fallback,
                    args.center_weight_strength,
                    args.center_weight_sigma,
                    fallback_homography=previous_homography,
                    modal_fill=not args.no_modal_fill,
                    boundary_feather=args.boundary_feather,
                    thermal_border_crop=args.thermal_border_crop,
                )
        except Exception as exc:
            mixed_frame = None
            metrics = {"fallback_reason": f"error:{exc}"}

        row.update(metrics)
        if mixed_frame is None:
            writer.write(color_frame)
            failed += 1
            row["status"] = "fallback_color"
        else:
            writer.write(mixed_frame)
            saved += 1
            row["status"] = "saved" if not metrics.get("fallback_reason") else "fallback_previous"
        rows.append(row)
        processed += 1
        if progress_bar is not None:
            progress_bar.update(1)

    writer.release()
    rgb_reader.release()
    thermal_reader.release()
    return {
        "saved": saved,
        "failed": failed,
        "skipped_existing": 0,
        "rows": rows,
        "processed": processed,
    }


def estimate_lerobot_job_frame_count(job):
    rgb_path, thermal_path, _, thermal_start_s = job
    rgb_info = probe_video(rgb_path)
    thermal_info = probe_video(thermal_path)
    if rgb_info["frame_rate"] != thermal_info["frame_rate"]:
        raise ValueError(
            f"RGB and thermal fps differ for {rgb_path} / {thermal_path}: "
            f"{rgb_info['frame_rate']} vs {thermal_info['frame_rate']}"
        )

    fps = float(rgb_info["frame_rate"])
    rgb_frames = rgb_info["frame_count"]
    if rgb_frames is None:
        rgb_frames = int(round(float(rgb_info["duration"]) * fps))

    thermal_frames = thermal_info["frame_count"]
    if thermal_frames is None:
        thermal_frames = int(round(float(thermal_info["duration"]) * fps))

    thermal_start_frame = int(round(thermal_start_s * fps))
    thermal_available_frames = max(0, int(thermal_frames) - thermal_start_frame)
    return max(0, min(int(rgb_frames), thermal_available_frames))


def metric_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def mean_metric(rows, key):
    values = metric_values(rows, key)
    return float(np.mean(values)) if values else np.nan


def fmt_metric(value, precision=3):
    return f"{value:.{precision}f}" if np.isfinite(value) else "nan"


def write_metrics_csv(metrics_csv, rows):
    metrics_csv = Path(metrics_csv)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    method_parser = argparse.ArgumentParser(add_help=False)
    method_parser.add_argument(
        "--method",
        type=str,
        default="sp_lg",
        choices=["sp_lg", "roma"],
        help="Matching method to use.",
    )
    method_args, _ = method_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description=(
            "Match RGB images with thermal images and save thermal-on-RGB mixing images."
        )
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Dataset root, for example: data/cold_water_bottle_selection",
    )
    parser.add_argument(
        "--input_format",
        "--format",
        dest="input_format",
        type=str,
        default="unitree",
        choices=["unitree", "lerobot"],
        help=(
            "Input layout. 'unitree' keeps the original episode*/images(or colors)+thermography "
            "image workflow; 'lerobot' reads LeRobot v3 videos and writes a mixed video feature."
        ),
    )
    parser.add_argument(
        "--minima_root",
        type=str,
        default="./MINIMA",
        help="Path to the MINIMA project root containing load_model.py and weights/.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=method_args.method,
        choices=["sp_lg", "roma"],
        help="Matching method to use.",
    )
    add_matching_arguments(parser, method_args.method)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Opacity for the warped thermal image. 1.0 outputs the warped thermal directly, skipping blending.",
    )
    parser.add_argument(
        "--boundary-feather",
        type=int,
        default=50,
        help=(
            "Feather radius in pixels for the warped thermal boundary after "
            "modal-color filling. 0 disables extra boundary smoothing."
        ),
    )
    parser.add_argument(
        "--thermal-border-crop",
        type=int,
        default=30,
        help=(
            "Ignore this many outer pixels from the source thermal image before "
            "warping/blending, useful for removing dark thermal sensor borders."
        ),
    )
    parser.add_argument(
        "--match-every-n-frames",
        type=int,
        default=1,
        help=(
            "LeRobot video speed knob: run MINIMA matching every N frames and "
            "reuse the previous homography for intermediate frames. 1 matches every frame."
        ),
    )
    parser.add_argument(
        "--no-modal-fill",
        action="store_true",
        help=(
            "Disable filling uncovered warped-thermal areas with the modal "
            "thermal color before blending."
        ),
    )
    parser.add_argument(
        "--ransac_reproj_threshold",
        type=float,
        default=5.0,
        help="RANSAC reprojection threshold passed to cv2.findHomography.",
    )
    parser.add_argument(
        "--min_matches",
        type=int,
        default=4,
        help="Minimum raw matches required before homography estimation.",
    )
    parser.add_argument(
        "--min_inliers",
        type=int,
        default=4,
        help="Minimum RANSAC inliers required to save a result.",
    )
    parser.add_argument(
        "--edge_canny_low",
        type=int,
        default=50,
        help="Lower Canny threshold for contour-overlap metrics.",
    )
    parser.add_argument(
        "--edge_canny_high",
        type=int,
        default=150,
        help="Upper Canny threshold for contour-overlap metrics.",
    )
    parser.add_argument(
        "--edge_dilate",
        type=int,
        default=3,
        help="Pixel tolerance for contour overlap metrics.",
    )
    parser.add_argument(
        "--metrics_csv",
        type=str,
        default=None,
        help="Optional CSV path for per-image alignment metrics.",
    )
    parser.add_argument(
        "--homography_center_ratio",
        type=float,
        default=1.0,
        help="Use only target RGB center matches for homography. 1.0 means all matches.",
    )
    parser.add_argument(
        "--metric_center_ratio",
        type=float,
        default=1.0,
        help="Evaluate edge metrics only in the output center region. 1.0 means full image.",
    )
    parser.add_argument(
        "--no_center_fallback",
        action="store_true",
        help="Do not fall back to all matches when the center region has too few matches.",
    )
    parser.add_argument(
        "--center_weight_strength",
        type=float,
        default=0.0,
        help="Soft center attention for homography refit. 0 disables it; try 0.3 for mild attention.",
    )
    parser.add_argument(
        "--center_weight_sigma",
        type=float,
        default=0.6,
        help="Spread of soft center attention. Larger values make the attention more gradual.",
    )
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default="colors_thermography_mixing",
        help="Episode-level output directory name.",
    )
    parser.add_argument(
        "--rgb-feature",
        type=str,
        default=DEFAULT_RGB_FEATURE,
        help=f"LeRobot RGB video feature name (default: {DEFAULT_RGB_FEATURE}).",
    )
    parser.add_argument(
        "--thermal-feature",
        type=str,
        default=DEFAULT_THERMAL_FEATURE,
        help=f"LeRobot thermal video feature name (default: {DEFAULT_THERMAL_FEATURE}).",
    )
    parser.add_argument(
        "--output-feature",
        type=str,
        default=DEFAULT_OUTPUT_FEATURE,
        help=f"LeRobot output mixed video feature name (default: {DEFAULT_OUTPUT_FEATURE}).",
    )
    parser.add_argument(
        "--output_vcodec",
        type=str,
        default="libx264",
        help="ffmpeg video codec used for LeRobot mixed mp4 output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process at most this many image pairs in unitree mode, or this many "
            "video frames in lerobot mode. Kept as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Process at most this many frames in lerobot video mode. In unitree "
            "image mode this is treated the same as --limit."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in colors_thermography_mixing.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only list how many pairs would be processed.",
    )
    return parser


def validate_args(args):
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1.")
    if args.boundary_feather < 0:
        raise ValueError("--boundary-feather must be >= 0.")
    if args.thermal_border_crop < 0:
        raise ValueError("--thermal-border-crop must be >= 0.")
    if args.match_every_n_frames <= 0:
        raise ValueError("--match-every-n-frames must be a positive integer.")
    if not 0.0 < args.homography_center_ratio <= 1.0:
        raise ValueError("--homography_center_ratio must be in (0, 1].")
    if not 0.0 < args.metric_center_ratio <= 1.0:
        raise ValueError("--metric_center_ratio must be in (0, 1].")
    if args.center_weight_strength < 0.0:
        raise ValueError("--center_weight_strength must be >= 0.")
    if args.center_weight_sigma <= 0.0:
        raise ValueError("--center_weight_sigma must be > 0.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be a positive integer.")
    if args.limit is not None and args.max_frames is not None and args.limit != args.max_frames:
        raise ValueError("Please use either --limit or --max-frames, or give them the same value.")
    if args.input_format == "lerobot":
        if shutil.which("ffprobe") is None:
            raise RuntimeError("ffprobe is required for --input_format=lerobot.")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for --input_format=lerobot.")


def resolve_max_frames(args):
    return args.max_frames if args.max_frames is not None else args.limit


def print_center_mode(args):
    if (
        args.homography_center_ratio < 1.0
        or args.metric_center_ratio < 1.0
        or args.center_weight_strength > 0.0
    ):
        print(
            "Center mode: "
            f"homography_center_ratio={args.homography_center_ratio}, "
            f"metric_center_ratio={args.metric_center_ratio}, "
            f"fallback_to_all={not args.no_center_fallback}, "
            f"center_weight_strength={args.center_weight_strength}, "
            f"center_weight_sigma={args.center_weight_sigma}"
        )


def print_mixing_layer_mode(args):
    print(
        "Thermal layer: "
        f"modal_fill={not args.no_modal_fill}, "
        f"boundary_feather={args.boundary_feather}px, "
        f"thermal_border_crop={args.thermal_border_crop}px"
    )


def print_lerobot_speed_mode(args):
    print(f"LeRobot speed: match_every_n_frames={args.match_every_n_frames}")


def load_matcher(args, use_path=True):
    resolve_minima_paths(args)
    _, torch, load_model = import_runtime_dependencies(args.minima_root)
    patch_runtime_compatibility(torch)
    matcher_input = "path" if use_path else "memory"
    print(f"Loading matcher: {args.method} ({matcher_input} input)")
    return load_model(args.method, args, use_path=use_path)


def print_summary(saved, failed, total_matches, total_inliers, metric_rows, elapsed, args):
    successful_rows = [row for row in metric_rows if row.get("status") == "saved"]
    print(
        "Done. "
        f"Saved: {saved}, failed/skipped by matching: {failed}, "
        f"avg matches: {total_matches / max(saved + failed, 1):.1f}, "
        f"avg inliers: {total_inliers / max(saved + failed, 1):.1f}, "
        f"elapsed: {elapsed:.1f}s"
    )
    if successful_rows:
        print(
            "Alignment metrics on saved outputs: "
            f"avg inlier ratio={fmt_metric(mean_metric(successful_rows, 'inlier_ratio'))}, "
            f"avg reproj mean px={fmt_metric(mean_metric(successful_rows, 'reproj_mean_px'))}, "
            f"avg edge F1={fmt_metric(mean_metric(successful_rows, 'edge_f1'))}, "
            f"avg edge IoU={fmt_metric(mean_metric(successful_rows, 'edge_iou'))}, "
            f"avg warped coverage={fmt_metric(mean_metric(successful_rows, 'warped_coverage'))}"
        )
        if args.homography_center_ratio < 1.0 or args.center_weight_strength > 0.0:
            print(
                "Center homography diagnostics: "
                f"avg raw matches={fmt_metric(mean_metric(successful_rows, 'raw_matches'), 1)}, "
                f"avg used matches={fmt_metric(mean_metric(successful_rows, 'matches'), 1)}, "
                f"center fallback frames={int(sum(row.get('center_fallback', 0) for row in successful_rows))}, "
                f"weighted refit frames={int(sum(row.get('weighted_refit', 0) for row in successful_rows))}"
            )
    if args.metrics_csv is not None:
        write_metrics_csv(args.metrics_csv, metric_rows)
        print(f"Saved per-image metrics to: {args.metrics_csv}")


def run_unitree_images(args, root_dir):
    pairs = []
    missing_thermal = 0
    skipped_existing = 0
    max_pairs = resolve_max_frames(args)
    for color_path, thermal_path, output_path in iter_image_pairs(root_dir, args.output_dir_name):
        if not thermal_path.is_file():
            missing_thermal += 1
            continue
        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue
        pairs.append((color_path, thermal_path, output_path))
        if max_pairs is not None and len(pairs) >= max_pairs:
            break

    print(f"Found {len(pairs)} pairs to process under {root_dir}")
    print(f"Missing thermal files: {missing_thermal}")
    print(f"Skipped existing outputs: {skipped_existing}")
    print_mixing_layer_mode(args)
    print_center_mode(args)

    if args.dry_run or not pairs:
        return

    matcher = load_matcher(args)

    saved = 0
    failed = 0
    total_matches = 0
    total_inliers = 0
    metric_rows = []
    iterator = tqdm(pairs, desc="Mixing RGB/Thermal") if tqdm is not None else pairs
    start_time = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for color_path, thermal_path, output_path in iterator:
            row = {
                "status": "error",
                "color_path": str(color_path),
                "thermal_path": str(thermal_path),
                "output_path": str(output_path),
            }
            try:
                ok, metrics = save_mixing_image(
                    matcher,
                    color_path,
                    thermal_path,
                    output_path,
                    args.alpha,
                    args.ransac_reproj_threshold,
                    args.min_matches,
                    args.min_inliers,
                    args.edge_canny_low,
                    args.edge_canny_high,
                    args.edge_dilate,
                    args.homography_center_ratio,
                    args.metric_center_ratio,
                    not args.no_center_fallback,
                    args.center_weight_strength,
                    args.center_weight_sigma,
                    modal_fill=not args.no_modal_fill,
                    boundary_feather=args.boundary_feather,
                    thermal_border_crop=args.thermal_border_crop,
                )
            except Exception as exc:
                failed += 1
                row["status"] = "error"
                metric_rows.append(row)
                print(f"[ERROR] {color_path} -> {thermal_path}: {exc}")
                continue

            row.update(metrics)
            total_matches += metrics.get("matches", 0)
            total_inliers += metrics.get("inliers", 0)
            if ok:
                saved += 1
                row["status"] = "saved"
            else:
                failed += 1
                row["status"] = "failed_matching"
            metric_rows.append(row)

    elapsed = time.time() - start_time
    print_summary(saved, failed, total_matches, total_inliers, metric_rows, elapsed, args)


def run_lerobot_videos(args, root_dir):
    jobs = iter_lerobot_video_jobs(
        root_dir,
        args.rgb_feature,
        args.thermal_feature,
        args.output_feature,
    )
    skipped_existing = sum(1 for _, _, output_path, _ in jobs if output_path.exists() and not args.overwrite)
    runnable_jobs = [
        job for job in jobs if args.overwrite or not job[2].exists()
    ]
    runnable_frame_counts = [estimate_lerobot_job_frame_count(job) for job in runnable_jobs]
    max_frames = resolve_max_frames(args)
    available_frames = sum(runnable_frame_counts)
    frames_to_process = available_frames if max_frames is None else min(available_frames, max_frames)

    print(f"Found {len(jobs)} LeRobot video job(s) under {root_dir}")
    print(f"RGB feature: {args.rgb_feature}")
    print(f"Thermal feature: {args.thermal_feature}")
    print(f"Output mixed feature: {args.output_feature}")
    print(f"Skipped existing outputs: {skipped_existing}")
    print(f"Frames to process: {frames_to_process}" + ("" if max_frames is None else f" / requested {max_frames}"))
    print_mixing_layer_mode(args)
    print_lerobot_speed_mode(args)
    for rgb_path, thermal_path, output_path, thermal_start_s in jobs:
        print(f"  RGB={rgb_path}")
        print(f"  thermal={thermal_path} (start={thermal_start_s:.6f}s)")
        print(f"  output={output_path}")

    print_center_mode(args)
    if args.dry_run or not runnable_jobs:
        return

    matcher = load_matcher(args, use_path=False)

    saved = 0
    failed = 0
    skipped = 0
    total_matches = 0
    total_inliers = 0
    metric_rows = []
    remaining_frames = max_frames
    progress_bar = (
        tqdm(total=frames_to_process, desc="Mixing LeRobot frames", unit="frame")
        if tqdm is not None
        else None
    )
    start_time = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            for job in runnable_jobs:
                if remaining_frames is not None and remaining_frames <= 0:
                    break
                result = process_lerobot_video_job(
                    matcher,
                    job,
                    args,
                    remaining_frames,
                    progress_bar,
                )
                skipped += result["skipped_existing"]
                saved += result["saved"]
                failed += result["failed"]
                for row in result["rows"]:
                    total_matches += row.get("matches", 0) or 0
                    total_inliers += row.get("inliers", 0) or 0
                metric_rows.extend(result["rows"])
                if remaining_frames is not None:
                    remaining_frames -= result["processed"]
        finally:
            if progress_bar is not None:
                progress_bar.close()

    elapsed = time.time() - start_time
    if skipped:
        print(f"Skipped existing video outputs during processing: {skipped}")
    print_summary(saved, failed, total_matches, total_inliers, metric_rows, elapsed, args)


def main():
    args = build_parser().parse_args()
    validate_args(args)

    root_dir = Path(args.root_dir)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    if args.input_format == "unitree":
        run_unitree_images(args, root_dir)
    elif args.input_format == "lerobot":
        run_lerobot_videos(args, root_dir)
    else:
        raise ValueError(f"Unknown input format: {args.input_format}")


if __name__ == "__main__":
    main()
