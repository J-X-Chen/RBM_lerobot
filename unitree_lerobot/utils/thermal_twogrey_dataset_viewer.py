#!/usr/bin/env python3
# python unitree_lerobot/utils/thermal_twogrey_dataset_viewer.py --input_path /path/to/lerobot_dataset     --output_path /path/to/export_result     --importance-color black
"""LeRobot v3 热成像 twogrey/matching 数据集浏览器。

默认 twogrey 模式显示四幅图像：

1. 头部 RGB；
2. 原始热成像；
3. 生成的冷图；
4. 生成的热图。

``thermal_input_type=matching`` 会预加载 MINIMA，增加一幅匹配到头部 RGB
坐标系的热成像，并从匹配后的热成像计算冷图/热图。为缩短逐帧时间，该模式
只做单应变换和可缓存的边界 mask。默认不裁剪、不向内羽化，只在热成像边框
外扩展最外圈像素；原始热成像区域不被修改。匹配后的彩色热成像向黑底渐变；
冷图、热图的空白默认保持白色；可通过参数启用边界亮度校准的 RGB 灰度填充。

视频位置和 episode 边界来自 ``meta/episodes/**/*.parquet``，因此即使不同
相机的视频分片边界不同，也能使用各自正确的视频文件和时间戳。

推荐在项目环境中启动：

    conda activate g1_lerobot
    python unitree_lerobot/utils/thermal_twogrey_dataset_viewer.py \
        lerobotv3_datasets/cold_water_bottle_selection_right \
        --thermal_input_type matching \
        --importance-color black

不启动界面、直接把本地数据集逐帧导出到另一个目录：

    python unitree_lerobot/utils/thermal_twogrey_dataset_viewer.py \
        --input-path lerobotv3_datasets/cold_water_bottle_selection_right \
        --output-path local_exports/cold_water_bottle_selection_right \
        --thermal-input-type matching \
        --importance-color black \
        --matching-every-n-frames 10 \
        --max-frames 1000

输出按 episode/组件保存无损 PNG，并附带 ``export_config.json``、
``manifest.jsonl`` 和 ``export_summary.json``。不传 ``--output-path`` 时仍按
原方式打开交互查看器。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCloseEvent, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


DEFAULT_BACKGROUND_ROI = (0, 0, 64, 64)
DEFAULT_BACKGROUND_VALUE = 80.0
DEFAULT_BACKGROUND_OUTLIER_THRESHOLD = 12.0
DEFAULT_BACKGROUND_OUTLIER_RATIO = 2.0
IMPORTANCE_COLOR_MODES = ("original", "black")
THERMAL_INPUT_TYPES = ("twogrey", "matching")
MATCHING_BLANK_FILL_MODES = ("white", "rgbgray")
LOCAL_OUTPUT_COMPONENTS = (
    "head_rgb",
    "thermal_rgb",
    "matched_thermal",
    "cold",
    "hot",
)
BACKGROUND_REGION_NAMES = ("左上", "中上", "右上")
DEFAULT_MINIMA_ROOT = "./MINIMA"
DEFAULT_MINIMA_CHECKPOINT = "./weights/minima_lightglue.pth"
DEFAULT_MATCHING_RANSAC_THRESHOLD = 5.0
DEFAULT_MATCHING_MIN_MATCHES = 4
DEFAULT_MATCHING_MIN_INLIERS = 4
DEFAULT_MATCHING_THERMAL_BORDER_CROP = 0
DEFAULT_MATCHING_BOUNDARY_FEATHER = 0
DEFAULT_MATCHING_OUTER_PADDING = 30
DEFAULT_MATCHING_BLANK_FILL = "white"
DEFAULT_MATCHING_CACHE_SIZE = 512


@dataclass(frozen=True)
class VideoSlice:
    chunk_index: int
    file_index: int
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    length: int
    tasks: tuple[str, ...]
    streams: dict[str, VideoSlice]


@dataclass(frozen=True)
class BackgroundEstimate:
    value: float
    region_medians: tuple[float, float, float]
    used_region_indices: tuple[int, ...]
    excluded_region_index: int | None
    region_boxes: tuple[
        tuple[int, int, int, int],
        tuple[int, int, int, int],
        tuple[int, int, int, int],
    ]


@dataclass(frozen=True)
class HomographyCacheEntry:
    homography: np.ndarray | None
    metrics: dict[str, Any]
    match_ms: float


@dataclass(frozen=True)
class MatchingFrame:
    thermal_rgb: np.ndarray
    valid_mask: np.ndarray
    feather_mask: np.ndarray
    homography: np.ndarray | None
    metrics: dict[str, Any]
    status: str
    match_ms: float
    warp_ms: float


@dataclass(frozen=True)
class ProcessedThermalFrame:
    cold_rgb: np.ndarray
    hot_rgb: np.ndarray
    matched_display_rgb: np.ndarray | None
    split_source_rgb: np.ndarray
    background: BackgroundEstimate
    matching_frame: MatchingFrame | None
    blank_fill_ms: float
    cold_fill_calibration: tuple[float, float]
    hot_fill_calibration: tuple[float, float]


@dataclass(frozen=True)
class LocalExportSummary:
    output_path: Path
    episode_count: int
    frame_count: int
    image_count: int
    elapsed_s: float


@dataclass(frozen=True)
class DatasetIndex:
    root: Path
    fps: float
    video_path_template: str
    video_keys: tuple[str, ...]
    head_keys: tuple[str, ...]
    thermal_keys: tuple[str, ...]
    episodes: tuple[EpisodeRecord, ...]

    def video_path(self, episode: EpisodeRecord, video_key: str) -> Path:
        stream = episode.streams[video_key]
        relative_path = self.video_path_template.format(
            video_key=video_key,
            episode_index=episode.episode_index,
            chunk_index=stream.chunk_index,
            file_index=stream.file_index,
        )
        return self.root / relative_path


def _video_column(video_key: str, field: str) -> str:
    return f"videos/{video_key}/{field}"


def _is_thermal_key(key: str) -> bool:
    lowered = key.lower()
    return "thermal" in lowered or "thermograph" in lowered


def _head_key_score(key: str) -> tuple[int, str]:
    """Lower scores are preferred when choosing the initial head RGB stream."""
    lowered = key.lower()
    if lowered == "observation.images.cam_left_high":
        rank = 0
    elif "head" in lowered:
        rank = 1
    elif "high" in lowered:
        rank = 2
    else:
        rank = 3
    return rank, key


def load_dataset_index(root: str | Path) -> DatasetIndex:
    """Read the small v3 metadata files without loading the dataset parquet data."""
    root = Path(root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"未找到 LeRobot 数据集元数据：{info_path}")

    with info_path.open("r", encoding="utf-8") as file:
        info = json.load(file)

    if str(info.get("codebase_version", "")).lower() not in {"v3.0", "3.0"}:
        raise ValueError(
            "当前工具读取 LeRobot v3 数据集；"
            f"info.json 中的 codebase_version={info.get('codebase_version')!r}。"
        )

    fps = float(info.get("fps", 0))
    if fps <= 0:
        raise ValueError(f"数据集 fps 必须大于 0，当前为 {fps!r}。")

    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("info.json 缺少 features 字典。")

    video_keys = tuple(
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    )
    thermal_keys = tuple(key for key in video_keys if _is_thermal_key(key))
    head_keys = tuple(
        sorted(
            (
                key
                for key in video_keys
                if not _is_thermal_key(key) and "wrist" not in key.lower()
            ),
            key=_head_key_score,
        )
    )
    if not head_keys:
        raise ValueError(
            "没有找到头部 RGB 视频。工具会排除名称中包含 wrist 或 thermal 的视频特征。"
        )
    if not thermal_keys:
        raise ValueError("没有找到名称中包含 thermal/thermograph 的热成像视频特征。")

    metadata_paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not metadata_paths:
        raise FileNotFoundError(
            f"未找到 episode 元数据：{root / 'meta' / 'episodes'}/**/*.parquet"
        )

    required_base_columns = {"episode_index", "length"}
    optional_base_columns = {"tasks"}
    requested_video_columns = {
        _video_column(key, field)
        for key in video_keys
        for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
    }

    rows: list[dict[str, Any]] = []
    for metadata_path in metadata_paths:
        parquet_file = pq.ParquetFile(metadata_path)
        available = set(parquet_file.schema_arrow.names)
        missing = required_base_columns - available
        if missing:
            raise ValueError(
                f"{metadata_path} 缺少必要字段：{', '.join(sorted(missing))}"
            )
        selected_columns = sorted(
            required_base_columns
            | (optional_base_columns & available)
            | (requested_video_columns & available)
        )
        rows.extend(parquet_file.read(columns=selected_columns).to_pylist())

    episodes: list[EpisodeRecord] = []
    for row in rows:
        streams: dict[str, VideoSlice] = {}
        for video_key in video_keys:
            fields = {
                field: row.get(_video_column(video_key, field))
                for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
            }
            if any(value is None for value in fields.values()):
                continue
            streams[video_key] = VideoSlice(
                chunk_index=int(fields["chunk_index"]),
                file_index=int(fields["file_index"]),
                from_timestamp=float(fields["from_timestamp"]),
                to_timestamp=float(fields["to_timestamp"]),
            )

        tasks_value = row.get("tasks") or []
        tasks = tuple(str(task) for task in tasks_value)
        length = int(row["length"])
        if length <= 0:
            continue
        episodes.append(
            EpisodeRecord(
                episode_index=int(row["episode_index"]),
                length=length,
                tasks=tasks,
                streams=streams,
            )
        )

    episodes.sort(key=lambda episode: episode.episode_index)
    if not episodes:
        raise ValueError("episode 元数据中没有长度大于 0 的 episode。")
    episode_indices = [episode.episode_index for episode in episodes]
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError("episode 元数据中存在重复的 episode_index。")

    return DatasetIndex(
        root=root,
        fps=fps,
        video_path_template=str(
            info.get(
                "video_path",
                "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            )
        ),
        video_keys=video_keys,
        head_keys=head_keys,
        thermal_keys=thermal_keys,
        episodes=tuple(episodes),
    )


def _clip_background_roi(
    background_roi: tuple[int, int, int, int],
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    if len(background_roi) != 4:
        raise ValueError("背景 ROI 必须是 (top, left, height, width)。")
    top, left, roi_height, roi_width = (int(value) for value in background_roi)
    if roi_height <= 0 or roi_width <= 0:
        raise ValueError("背景 ROI 的 height 和 width 必须大于 0。")
    if height <= 0 or width <= 0:
        raise ValueError(f"热成像尺寸不合法：{height}x{width}。")

    top = max(0, min(top, height - 1))
    left = max(0, min(left, width - 1))
    bottom = max(top + 1, min(top + roi_height, height))
    right = max(left + 1, min(left + roi_width, width))
    return top, left, bottom, right


def _three_top_background_rois(
    background_roi: tuple[int, int, int, int],
    height: int,
    width: int,
) -> tuple[
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
]:
    """Build top-left, top-center, and top-right ROIs of equal size.

    ``background_roi`` is interpreted as ``(top, side_margin, height, width)``.
    The side margin locates the left ROI and is mirrored for the right ROI;
    the center ROI is always horizontally centered.
    """
    if len(background_roi) != 4:
        raise ValueError("背景 ROI 必须是 (top, side_margin, height, width)。")
    top, side_margin, roi_height, roi_width = (
        int(value) for value in background_roi
    )
    if side_margin < 0:
        raise ValueError("背景 ROI 的 side_margin 不能为负数。")

    left_positions = (
        side_margin,
        max(0, (width - roi_width) // 2),
        max(0, width - side_margin - roi_width),
    )
    return tuple(
        _clip_background_roi(
            (top, left, roi_height, roi_width),
            height,
            width,
        )
        for left in left_positions
    )


def estimate_three_region_background(
    gray_f32: np.ndarray,
    *,
    background_roi: tuple[int, int, int, int],
    outlier_threshold: float,
    outlier_ratio: float,
) -> BackgroundEstimate:
    """Estimate background from three top ROIs, rejecting one clear outlier.

    The closest pair of ROI medians is treated as the candidate consensus. The
    third ROI is excluded only when its distance from that pair is both:

    - at least ``outlier_threshold`` grayscale levels; and
    - at least ``outlier_ratio`` times the difference inside the pair.

    When no region satisfies both conditions, pixels from all three ROIs are
    used. The final value is the median of all pixels in the retained regions.
    """
    if gray_f32.ndim != 2:
        raise ValueError(f"背景估计需要二维灰度图，当前 shape={gray_f32.shape}。")
    if outlier_threshold < 0:
        raise ValueError("背景异常绝对差值不能为负数。")
    if outlier_ratio < 1:
        raise ValueError("背景异常相对倍率必须大于或等于 1。")

    boxes = _three_top_background_rois(
        background_roi,
        gray_f32.shape[0],
        gray_f32.shape[1],
    )
    region_pixels = tuple(
        gray_f32[top:bottom, left:right].reshape(-1)
        for top, left, bottom, right in boxes
    )
    medians = tuple(float(np.median(pixels)) for pixels in region_pixels)

    pairs = ((0, 1), (0, 2), (1, 2))
    pair = min(pairs, key=lambda indices: abs(medians[indices[0]] - medians[indices[1]]))
    outlier_index = next(index for index in range(3) if index not in pair)
    pair_difference = abs(medians[pair[0]] - medians[pair[1]])
    pair_center = 0.5 * (medians[pair[0]] + medians[pair[1]])
    outlier_distance = abs(medians[outlier_index] - pair_center)
    relative_baseline = max(pair_difference, 1e-6)

    excluded_region_index: int | None = None
    if (
        outlier_distance >= float(outlier_threshold)
        and outlier_distance >= float(outlier_ratio) * relative_baseline
    ):
        used_indices = tuple(sorted(pair))
        excluded_region_index = outlier_index
    else:
        used_indices = (0, 1, 2)

    retained_pixels = np.concatenate(
        [region_pixels[index] for index in used_indices]
    )
    return BackgroundEstimate(
        value=float(np.median(retained_pixels)),
        region_medians=medians,
        used_region_indices=used_indices,
        excluded_region_index=excluded_region_index,
        region_boxes=boxes,
    )


def _split_normalized_twogrey(
    normalized: np.ndarray,
    *,
    boundary: int,
    importance_color: str,
) -> tuple[np.ndarray, np.ndarray]:
    if importance_color not in IMPORTANCE_COLOR_MODES:
        raise ValueError(
            f"importance_color 必须是 {IMPORTANCE_COLOR_MODES} 之一，"
            f"当前为 {importance_color!r}。"
        )
    boundary = int(np.clip(boundary, 0, 255))
    cold_gray = np.minimum(normalized, boundary).astype(np.uint8, copy=False)
    original_hot_gray = np.maximum(normalized, boundary).astype(
        np.uint8,
        copy=False,
    )
    if importance_color == "black":
        # Training-friendly, full-range symmetric mapping:
        #
        # - shared background -> white (255);
        # - strongest cold/hot deviation -> black (0);
        # - each side of the boundary independently uses the full 0..255 range.
        #
        # Mapping the two sides independently avoids the saturation caused by
        # reflecting around the boundary (for boundary=80 that old mapping
        # collapsed every hot value >=160 to exactly 0).
        if boundary > 0:
            cold_gray = np.rint(
                cold_gray.astype(np.float32) * (255.0 / boundary)
            ).astype(np.uint8)
        else:
            # There is no representable cold interval below a zero boundary.
            cold_gray = np.full_like(cold_gray, 255)

        if boundary < 255:
            hot_gray = np.rint(
                (255.0 - original_hot_gray.astype(np.float32))
                * (255.0 / (255 - boundary))
            ).astype(np.uint8)
        else:
            # There is no representable hot interval above a 255 boundary.
            hot_gray = np.full_like(original_hot_gray, 255)
    else:
        hot_gray = original_hot_gray
    return cold_gray, hot_gray


def thermal_to_twogrey(
    image: np.ndarray,
    *,
    background_roi: tuple[int, int, int, int] = DEFAULT_BACKGROUND_ROI,
    target_background: float = DEFAULT_BACKGROUND_VALUE,
    outlier_threshold: float = DEFAULT_BACKGROUND_OUTLIER_THRESHOLD,
    outlier_ratio: float = DEFAULT_BACKGROUND_OUTLIER_RATIO,
    importance_color: str = "original",
    background_estimate: BackgroundEstimate | None = None,
    valid_mask: np.ndarray | None = None,
    feather_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, BackgroundEstimate]:
    """Create cold/hot images using robust three-region background estimation.

    A decoded RGB thermal frame is deliberately treated as a single-channel
    signal by taking channel 0, matching PI0.5 preprocessing. The background
    extension uses three top ROIs and can reject one clear outlier.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array
    elif (
        array.ndim == 3
        and array.shape[0] in {1, 3, 4}
        and array.shape[-1] not in {1, 3, 4}
    ):
        gray = array[0]
    elif array.ndim == 3 and array.shape[-1] in {1, 3, 4}:
        gray = array[..., 0]
    else:
        raise ValueError(f"热成像必须是 HW、HWC 或 CHW，当前 shape={array.shape}。")

    gray_f32 = gray.astype(np.float32, copy=False)
    if np.issubdtype(gray.dtype, np.floating):
        gray_f32 = gray_f32 * 255.0

    background = background_estimate
    if background is None:
        background = estimate_three_region_background(
            gray_f32,
            background_roi=background_roi,
            outlier_threshold=outlier_threshold,
            outlier_ratio=outlier_ratio,
        )
    normalized = np.clip(
        gray_f32 - background.value + float(target_background),
        0,
        255,
    ).astype(np.uint8)

    boundary = int(np.uint8(np.clip(float(target_background), 0, 255)))
    cold_gray, hot_gray = _split_normalized_twogrey(
        normalized,
        boundary=boundary,
        importance_color=importance_color,
    )
    checked_valid_mask: np.ndarray | None = None
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != cold_gray.shape:
            raise ValueError(
                "匹配有效区域 mask 必须与热成像同尺寸，"
                f"当前 mask={valid_mask.shape}，热成像={cold_gray.shape}。"
            )
        checked_valid_mask = valid_mask
    if feather_mask is not None:
        feather_mask = np.asarray(feather_mask)
        if feather_mask.shape != cold_gray.shape:
            raise ValueError(
                "匹配羽化 mask 必须与热成像同尺寸，"
                f"当前 mask={feather_mask.shape}，热成像={cold_gray.shape}。"
            )
        if np.issubdtype(feather_mask.dtype, np.floating):
            alpha_u8 = np.rint(
                np.clip(feather_mask, 0.0, 1.0) * 255.0
            ).astype(np.uint8)
        else:
            alpha_u8 = np.clip(feather_mask, 0, 255).astype(
                np.uint8,
                copy=False,
            )
        if checked_valid_mask is not None:
            alpha_u8 = alpha_u8.copy()
            alpha_u8[~checked_valid_mask] = 0
        cold_gray = _blend_to_white(cold_gray, alpha_u8)
        hot_gray = _blend_to_white(hot_gray, alpha_u8)
    elif checked_valid_mask is not None:
        cold_gray = cold_gray.copy()
        hot_gray = hot_gray.copy()
        cold_gray[~checked_valid_mask] = 255
        hot_gray[~checked_valid_mask] = 255
    cold_rgb = np.repeat(cold_gray[..., None], 3, axis=-1)
    hot_rgb = np.repeat(hot_gray[..., None], 3, axis=-1)
    return cold_rgb, hot_rgb, background


def estimate_thermal_background(
    image: np.ndarray,
    *,
    background_roi: tuple[int, int, int, int],
    outlier_threshold: float,
    outlier_ratio: float,
) -> BackgroundEstimate:
    """Estimate the raw thermal background once before matching/warping."""
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array
    elif (
        array.ndim == 3
        and array.shape[0] in {1, 3, 4}
        and array.shape[-1] not in {1, 3, 4}
    ):
        gray = array[0]
    elif array.ndim == 3 and array.shape[-1] in {1, 3, 4}:
        gray = array[..., 0]
    else:
        raise ValueError(f"热成像必须是 HW、HWC 或 CHW，当前 shape={array.shape}。")
    gray_f32 = gray.astype(np.float32, copy=False)
    if np.issubdtype(gray.dtype, np.floating):
        gray_f32 = gray_f32 * 255.0
    return estimate_three_region_background(
        gray_f32,
        background_roi=background_roi,
        outlier_threshold=outlier_threshold,
        outlier_ratio=outlier_ratio,
    )


def _blend_to_white(image: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    """Composite a uint8 gray/RGB image onto white using a uint8 alpha mask."""
    image = np.asarray(image, dtype=np.uint8)
    alpha_u8 = np.asarray(alpha_u8, dtype=np.uint8)
    if image.shape[:2] != alpha_u8.shape:
        raise ValueError(
            f"图像与 alpha mask 尺寸不一致：{image.shape} / {alpha_u8.shape}。"
        )
    alpha_u16 = alpha_u8.astype(np.uint16)
    if image.ndim == 3:
        alpha_u16 = alpha_u16[..., None]
    blended = (
        image.astype(np.uint16) * alpha_u16
        + 255 * (255 - alpha_u16)
        + 127
    ) // 255
    return blended.astype(np.uint8)


def _blend_to_black(image: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    """Composite a uint8 gray/RGB image onto black using a uint8 alpha mask."""
    image = np.asarray(image, dtype=np.uint8)
    alpha_u8 = np.asarray(alpha_u8, dtype=np.uint8)
    if image.shape[:2] != alpha_u8.shape:
        raise ValueError(
            f"图像与 alpha mask 尺寸不一致：{image.shape} / {alpha_u8.shape}。"
        )
    alpha_u16 = alpha_u8.astype(np.uint16)
    if image.ndim == 3:
        alpha_u16 = alpha_u16[..., None]
    return (
        (image.astype(np.uint16) * alpha_u16 + 127) // 255
    ).astype(np.uint8)


def _fill_twogrey_blank_with_rgb_gray(
    cold_rgb: np.ndarray,
    hot_rgb: np.ndarray,
    head_rgb: np.ndarray,
    alpha_u8: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[float, float],
    tuple[float, float],
]:
    """Fill cold/hot blank pixels with separately calibrated RGB grayscale.

    Cold and hot pixels are preserved exactly wherever ``alpha_u8 == 255``.
    The outward padding band cross-fades each image to its own adjusted RGB
    grayscale fill, while the matched color thermal preview remains untouched.
    """
    import cv2

    cold_rgb = np.asarray(cold_rgb, dtype=np.uint8)
    hot_rgb = np.asarray(hot_rgb, dtype=np.uint8)
    head_rgb = np.asarray(head_rgb, dtype=np.uint8)
    alpha_u8 = np.asarray(alpha_u8, dtype=np.uint8)
    if cold_rgb.shape != head_rgb.shape or hot_rgb.shape != head_rgb.shape:
        raise ValueError(
            "冷图、热图与头部 RGB 必须同尺寸，"
            f"当前为 {cold_rgb.shape} / {hot_rgb.shape} / {head_rgb.shape}。"
        )
    if alpha_u8.shape != head_rgb.shape[:2]:
        raise ValueError(
            f"alpha mask 尺寸不匹配：{alpha_u8.shape} / {head_rgb.shape}。"
        )

    rgb_gray = cv2.cvtColor(head_rgb, cv2.COLOR_RGB2GRAY)
    boundary_mask = (alpha_u8 >= 32) & (alpha_u8 <= 223)
    if int(boundary_mask.sum()) < 64:
        boundary_mask = (alpha_u8 > 0) & (alpha_u8 < 255)

    if boundary_mask.any():
        rgb_boundary = rgb_gray[boundary_mask].astype(np.float32)
        rgb_median = float(np.median(rgb_boundary))
        rgb_quartiles = np.percentile(rgb_boundary, (25.0, 75.0))
        rgb_iqr = float(rgb_quartiles[1] - rgb_quartiles[0])
    else:
        rgb_median = 0.0
        rgb_iqr = 1.0

    alpha_u16 = alpha_u8.astype(np.uint16)

    def fill_one(target_rgb: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        target_gray = target_rgb[..., 0]
        if boundary_mask.any():
            target_boundary = target_gray[boundary_mask].astype(np.float32)
            target_median = float(np.median(target_boundary))
            target_quartiles = np.percentile(
                target_boundary,
                (25.0, 75.0),
            )
            target_iqr = float(target_quartiles[1] - target_quartiles[0])
            contrast_scale = float(
                np.clip(target_iqr / max(rgb_iqr, 1.0), 0.5, 2.0)
            )
            offset = target_median - contrast_scale * rgb_median
        else:
            contrast_scale = 1.0
            offset = 0.0

        adjusted_gray = np.clip(
            rgb_gray.astype(np.float32) * contrast_scale + offset,
            0.0,
            255.0,
        ).astype(np.uint8)
        filled_gray = (
            target_gray.astype(np.uint16) * alpha_u16
            + adjusted_gray.astype(np.uint16) * (255 - alpha_u16)
            + 127
        ) // 255
        filled_rgb = np.repeat(
            filled_gray.astype(np.uint8)[..., None],
            3,
            axis=-1,
        )
        return filled_rgb, (contrast_scale, offset)

    filled_cold, cold_calibration = fill_one(cold_rgb)
    filled_hot, hot_calibration = fill_one(hot_rgb)
    return filled_cold, filled_hot, cold_calibration, hot_calibration


def _resolve_viewer_path(path: str | Path, *, base: Path | None = None) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path]
    if base is not None:
        candidates.append(base / path)
    project_root = Path(__file__).resolve().parents[2]
    candidates.append(project_root / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


class MinimaThermalMatcher:
    """Preloaded MINIMA matcher with a small homography-only LRU cache."""

    def __init__(
        self,
        *,
        minima_root: str | Path,
        checkpoint: str | Path,
        ransac_threshold: float,
        min_matches: int,
        min_inliers: int,
        match_every_n_frames: int,
        thermal_border_crop: int,
        boundary_feather: int,
        outer_padding: int,
        cache_size: int,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        project_root_text = str(project_root)
        if project_root_text not in sys.path:
            sys.path.insert(0, project_root_text)

        from unitree_lerobot.utils.match_and_merge_rgb_and_thermography import (
            estimate_thermal_to_color_homography,
            load_matcher,
        )

        import cv2

        self.cv2 = cv2
        self._estimate_homography = estimate_thermal_to_color_homography
        self.ransac_threshold = float(ransac_threshold)
        self.min_matches = int(min_matches)
        self.min_inliers = int(min_inliers)
        self.match_every_n_frames = max(1, int(match_every_n_frames))
        self.thermal_border_crop = max(0, int(thermal_border_crop))
        self.boundary_feather = max(0, int(boundary_feather))
        self.outer_padding = max(0, int(outer_padding))
        self.cache_size = max(1, int(cache_size))
        self.cache: OrderedDict[tuple[Any, ...], HomographyCacheEntry] = OrderedDict()
        self.previous_homography: np.ndarray | None = None
        self.previous_sequence_key: tuple[Any, ...] | None = None
        self.previous_frame_index: int | None = None
        self._source_feather_masks: dict[
            tuple[int, int, int, int, int],
            np.ndarray,
        ] = {}

        resolved_minima_root = _resolve_viewer_path(minima_root)
        if not resolved_minima_root.is_dir():
            raise FileNotFoundError(f"MINIMA 目录不存在：{resolved_minima_root}")
        resolved_checkpoint = _resolve_viewer_path(
            checkpoint,
            base=resolved_minima_root,
        )
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(f"MINIMA checkpoint 不存在：{resolved_checkpoint}")

        args = SimpleNamespace(
            method="sp_lg",
            minima_root=str(resolved_minima_root),
            ckpt=str(resolved_checkpoint),
        )
        load_started = time.perf_counter()
        try:
            self.matcher = load_matcher(args, use_path=False)
        except ModuleNotFoundError as error:
            missing_name = error.name or str(error)
            raise RuntimeError(
                "MINIMA 依赖未安装完整："
                f"{missing_name}。请在启动查看器的环境中安装 "
                "`MINIMA/requirements.txt` 后重试。"
            ) from error
        self.load_ms = (time.perf_counter() - load_started) * 1000.0
        self.minima_root = resolved_minima_root
        self.checkpoint = resolved_checkpoint

    def reset_sequence(self) -> None:
        """Reset only temporal fallback state; cached frame homographies stay reusable."""
        self.previous_homography = None
        self.previous_sequence_key = None
        self.previous_frame_index = None

    def clear_cache(self) -> None:
        self.cache.clear()
        self.reset_sequence()

    def _remember(
        self,
        key: tuple[Any, ...],
        entry: HomographyCacheEntry,
    ) -> None:
        self.cache[key] = entry
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def _fast_warp(
        self,
        thermal_rgb: np.ndarray,
        homography: np.ndarray,
        output_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        output_height, output_width = output_shape
        source_height, source_width = thermal_rgb.shape[:2]
        crop = min(
            self.thermal_border_crop,
            max(0, (min(source_height, source_width) - 1) // 2),
        )
        mask_key = (
            source_height,
            source_width,
            crop,
            self.boundary_feather,
            self.outer_padding,
        )
        source_mask = self._source_feather_masks.get(mask_key)
        if source_mask is None:
            rows = np.minimum(
                np.arange(source_height) - crop + 1,
                source_height - crop - np.arange(source_height),
            )
            columns = np.minimum(
                np.arange(source_width) - crop + 1,
                source_width - crop - np.arange(source_width),
            )
            border_distance = np.minimum(rows[:, None], columns[None, :])
            if self.boundary_feather > 0:
                inner_mask = np.rint(
                    np.clip(
                        border_distance / float(self.boundary_feather),
                        0.0,
                        1.0,
                    )
                    * 255.0
                ).astype(np.uint8)
            else:
                inner_mask = np.where(
                    border_distance > 0,
                    255,
                    0,
                ).astype(np.uint8)
            if self.outer_padding > 0:
                padded_height = source_height + 2 * self.outer_padding
                padded_width = source_width + 2 * self.outer_padding
                padded_rows = np.minimum(
                    np.arange(padded_height),
                    padded_height - 1 - np.arange(padded_height),
                )
                padded_columns = np.minimum(
                    np.arange(padded_width),
                    padded_width - 1 - np.arange(padded_width),
                )
                outward_distance = np.minimum(
                    padded_rows[:, None],
                    padded_columns[None, :],
                )
                source_mask = np.rint(
                    np.clip(
                        outward_distance / float(self.outer_padding),
                        0.0,
                        1.0,
                    )
                    * 255.0
                ).astype(np.uint8)
                inner_slice = (
                    slice(self.outer_padding, self.outer_padding + source_height),
                    slice(self.outer_padding, self.outer_padding + source_width),
                )
                source_mask[inner_slice] = np.minimum(
                    source_mask[inner_slice],
                    inner_mask,
                )
            else:
                source_mask = inner_mask
            self._source_feather_masks[mask_key] = source_mask

        if self.outer_padding > 0:
            warp_source = self.cv2.copyMakeBorder(
                thermal_rgb,
                self.outer_padding,
                self.outer_padding,
                self.outer_padding,
                self.outer_padding,
                self.cv2.BORDER_REPLICATE,
            )
            padded_to_original = np.array(
                [
                    [1.0, 0.0, -float(self.outer_padding)],
                    [0.0, 1.0, -float(self.outer_padding)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            warp_homography = homography @ padded_to_original
        else:
            warp_source = thermal_rgb
            warp_homography = homography

        warped_rgb = self.cv2.warpPerspective(
            warp_source,
            warp_homography,
            (output_width, output_height),
            flags=self.cv2.INTER_LINEAR,
            borderMode=self.cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_mask = self.cv2.warpPerspective(
            source_mask,
            warp_homography,
            (output_width, output_height),
            flags=(
                self.cv2.INTER_LINEAR
                if self.boundary_feather > 0 or self.outer_padding > 0
                else self.cv2.INTER_NEAREST
            ),
            borderMode=self.cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid_mask = warped_mask > 0
        warped_rgb[~valid_mask] = 0
        return warped_rgb, valid_mask, warped_mask

    def align(
        self,
        head_rgb: np.ndarray,
        thermal_rgb: np.ndarray,
        *,
        cache_key: tuple[Any, ...],
        frame_index: int,
    ) -> MatchingFrame:
        metrics: dict[str, Any]
        homography: np.ndarray | None
        match_ms = 0.0
        status = ""

        cached = self.cache.get(cache_key)
        sequence_key = cache_key[:-1]
        can_reuse_previous = (
            self.previous_homography is not None
            and self.previous_sequence_key == sequence_key
            and self.previous_frame_index is not None
            and frame_index == self.previous_frame_index + 1
        )
        if cached is not None:
            self.cache.move_to_end(cache_key)
            homography = cached.homography
            metrics = dict(cached.metrics)
            metrics["cached_match_ms"] = cached.match_ms
            status = "缓存"
        elif (
            self.match_every_n_frames > 1
            and can_reuse_previous
            and frame_index % self.match_every_n_frames != 0
        ):
            homography = self.previous_homography
            metrics = {"fallback_reason": "reused_homography"}
            status = "复用上一单应矩阵"
        else:
            color_bgr = self.cv2.cvtColor(head_rgb, self.cv2.COLOR_RGB2BGR)
            thermal_bgr = self.cv2.cvtColor(thermal_rgb, self.cv2.COLOR_RGB2BGR)
            match_started = time.perf_counter()
            try:
                homography, metrics = self._estimate_homography(
                    self.matcher,
                    color_bgr,
                    thermal_bgr,
                    self.ransac_threshold,
                    self.min_matches,
                    self.min_inliers,
                    1.0,
                    True,
                    0.0,
                    0.6,
                )
            except Exception as error:
                homography = None
                metrics = {"fallback_reason": f"error:{error}"}
            match_ms = (time.perf_counter() - match_started) * 1000.0
            if homography is not None:
                self.previous_homography = homography
                status = "MINIMA 匹配"
            elif can_reuse_previous:
                homography = self.previous_homography
                metrics["fallback_reason"] = "previous_homography"
                status = "匹配失败，复用上一单应矩阵"
            else:
                status = "匹配失败，使用 resize"

        if cached is None:
            self._remember(
                cache_key,
                HomographyCacheEntry(
                    homography=None if homography is None else homography.copy(),
                    metrics=dict(metrics),
                    match_ms=match_ms,
                ),
            )
        if homography is not None:
            self.previous_homography = homography
            self.previous_sequence_key = sequence_key
            self.previous_frame_index = frame_index

        warp_started = time.perf_counter()
        if homography is None:
            output_height, output_width = head_rgb.shape[:2]
            matched_rgb = self.cv2.resize(
                thermal_rgb,
                (output_width, output_height),
                interpolation=self.cv2.INTER_LINEAR,
            )
            valid_mask = np.ones((output_height, output_width), dtype=bool)
            feather_mask = np.full(
                (output_height, output_width),
                255,
                dtype=np.uint8,
            )
        else:
            matched_rgb, valid_mask, feather_mask = self._fast_warp(
                thermal_rgb,
                homography,
                head_rgb.shape[:2],
            )
        warp_ms = (time.perf_counter() - warp_started) * 1000.0
        return MatchingFrame(
            thermal_rgb=matched_rgb,
            valid_mask=valid_mask,
            feather_mask=feather_mask,
            homography=homography,
            metrics=metrics,
            status=status,
            match_ms=match_ms,
            warp_ms=warp_ms,
        )


def process_thermal_frame(
    head_rgb: np.ndarray,
    thermal_rgb: np.ndarray,
    *,
    thermal_input_type: str,
    background_roi: tuple[int, int, int, int],
    background_value: float,
    background_outlier_threshold: float,
    background_outlier_ratio: float,
    importance_color: str,
    matching_blank_fill: str,
    matching_backend: MinimaThermalMatcher | None = None,
    matching_cache_key: tuple[Any, ...] | None = None,
    frame_index: int = 0,
) -> ProcessedThermalFrame:
    """Apply exactly the same processing used by GUI display and local export."""
    if thermal_input_type not in THERMAL_INPUT_TYPES:
        raise ValueError(
            f"thermal_input_type 必须是 {THERMAL_INPUT_TYPES} 之一。"
        )
    if matching_blank_fill not in MATCHING_BLANK_FILL_MODES:
        raise ValueError(
            f"matching_blank_fill 必须是 {MATCHING_BLANK_FILL_MODES} 之一。"
        )

    matching_frame: MatchingFrame | None = None
    matched_display_rgb: np.ndarray | None = None
    blank_fill_ms = 0.0
    cold_fill_calibration = (1.0, 0.0)
    hot_fill_calibration = (1.0, 0.0)

    if thermal_input_type == "matching":
        if matching_backend is None:
            raise RuntimeError("matching 模式尚未加载 MINIMA。")
        if matching_cache_key is None:
            raise ValueError("matching 模式需要 matching_cache_key。")

        # Estimate on the unwarped thermal frame. Empty target-canvas regions
        # must not participate in the three-region background estimate.
        source_background = estimate_thermal_background(
            thermal_rgb,
            background_roi=background_roi,
            outlier_threshold=background_outlier_threshold,
            outlier_ratio=background_outlier_ratio,
        )
        matching_frame = matching_backend.align(
            head_rgb,
            thermal_rgb,
            cache_key=matching_cache_key,
            frame_index=frame_index,
        )
        split_source = matching_frame.thermal_rgb
        matched_display_rgb = _blend_to_black(
            matching_frame.thermal_rgb,
            matching_frame.feather_mask,
        )
        if matching_blank_fill == "rgbgray":
            cold_rgb, hot_rgb, background = thermal_to_twogrey(
                split_source,
                background_roi=background_roi,
                target_background=background_value,
                outlier_threshold=background_outlier_threshold,
                outlier_ratio=background_outlier_ratio,
                importance_color=importance_color,
                background_estimate=source_background,
            )
            blank_fill_started = time.perf_counter()
            (
                cold_rgb,
                hot_rgb,
                cold_fill_calibration,
                hot_fill_calibration,
            ) = _fill_twogrey_blank_with_rgb_gray(
                cold_rgb,
                hot_rgb,
                head_rgb,
                matching_frame.feather_mask,
            )
            blank_fill_ms = (
                time.perf_counter() - blank_fill_started
            ) * 1000.0
        else:
            cold_rgb, hot_rgb, background = thermal_to_twogrey(
                split_source,
                background_roi=background_roi,
                target_background=background_value,
                outlier_threshold=background_outlier_threshold,
                outlier_ratio=background_outlier_ratio,
                importance_color=importance_color,
                background_estimate=source_background,
                valid_mask=matching_frame.valid_mask,
                feather_mask=matching_frame.feather_mask,
            )
    else:
        split_source = thermal_rgb
        cold_rgb, hot_rgb, background = thermal_to_twogrey(
            split_source,
            background_roi=background_roi,
            target_background=background_value,
            outlier_threshold=background_outlier_threshold,
            outlier_ratio=background_outlier_ratio,
            importance_color=importance_color,
        )

    return ProcessedThermalFrame(
        cold_rgb=cold_rgb,
        hot_rgb=hot_rgb,
        matched_display_rgb=matched_display_rgb,
        split_source_rgb=split_source,
        background=background,
        matching_frame=matching_frame,
        blank_fill_ms=blank_fill_ms,
        cold_fill_calibration=cold_fill_calibration,
        hot_fill_calibration=hot_fill_calibration,
    )


class AvFrameReader:
    """Small seekable PyAV reader optimized for sequential GUI playback."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self.container: av.container.InputContainer | None = None
        self.stream: av.video.stream.VideoStream | None = None
        self.iterator: Any = None
        self.last_timestamp: float | None = None
        self.last_frame: np.ndarray | None = None
        self.fps = 30.0

    def open(self, path: Path) -> None:
        path = path.resolve()
        if self.path == path and self.container is not None:
            return
        self.close()
        if not path.is_file():
            raise FileNotFoundError(f"视频文件不存在：{path}")

        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        if self.stream.average_rate is not None:
            self.fps = float(self.stream.average_rate)
        self.path = path
        self.iterator = None
        self.last_timestamp = None
        self.last_frame = None

    def close(self) -> None:
        if self.container is not None:
            self.container.close()
        self.path = None
        self.container = None
        self.stream = None
        self.iterator = None
        self.last_timestamp = None
        self.last_frame = None

    def _seek(self, timestamp: float) -> None:
        if self.container is None or self.stream is None:
            raise RuntimeError("视频尚未打开。")
        time_base = float(self.stream.time_base)
        one_frame = 1.0 / max(self.fps, 1.0)
        seek_timestamp = max(0.0, timestamp - one_frame)
        offset = max(0, round(seek_timestamp / time_base))
        self.container.seek(
            offset,
            stream=self.stream,
            backward=True,
            any_frame=False,
        )
        self.iterator = self.container.decode(self.stream)
        self.last_timestamp = None
        self.last_frame = None

    def read_rgb(self, timestamp: float) -> np.ndarray:
        if self.container is None or self.stream is None:
            raise RuntimeError("视频尚未打开。")

        half_frame = 0.5 / max(self.fps, 1.0)
        max_sequential_gap = max(0.25, 8.0 / max(self.fps, 1.0))
        needs_seek = (
            self.iterator is None
            or self.last_timestamp is None
            or timestamp < self.last_timestamp - half_frame
            or timestamp - self.last_timestamp > max_sequential_gap
        )
        if needs_seek:
            self._seek(timestamp)

        if (
            self.last_timestamp is not None
            and self.last_frame is not None
            and abs(self.last_timestamp - timestamp) <= half_frame
        ):
            return self.last_frame

        assert self.iterator is not None
        for frame in self.iterator:
            if frame.pts is None:
                continue
            frame_timestamp = float(frame.pts * self.stream.time_base)
            frame_rgb = frame.to_ndarray(format="rgb24")
            self.last_timestamp = frame_timestamp
            self.last_frame = frame_rgb
            if frame_timestamp >= timestamp - half_frame:
                return frame_rgb

        if self.last_frame is not None:
            return self.last_frame
        raise RuntimeError(f"无法在 {timestamp:.3f}s 解码视频帧：{self.path}")


def _select_export_video_key(
    available_keys: tuple[str, ...],
    preferred_key: str | None,
    *,
    label: str,
) -> str:
    if preferred_key is None:
        return available_keys[0]
    if preferred_key not in available_keys:
        raise ValueError(
            f"指定的{label}特征 {preferred_key!r} 不存在；"
            f"可选值：{', '.join(available_keys)}"
        )
    return preferred_key


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_rgb_png(
    path: Path,
    image_rgb: np.ndarray,
    *,
    compression: int,
) -> None:
    import cv2

    image_rgb = np.asarray(image_rgb)
    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError(f"保存图像必须是 HWC RGB，当前 shape={image_rgb.shape}。")
    image_u8 = np.clip(image_rgb, 0, 255).astype(np.uint8, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(
        str(path),
        image_bgr,
        [cv2.IMWRITE_PNG_COMPRESSION, int(compression)],
    ):
        raise OSError(f"图像保存失败：{path}")


def export_dataset_locally(
    input_path: str | Path,
    output_path: str | Path,
    *,
    background_roi: tuple[int, int, int, int],
    background_value: float,
    background_outlier_threshold: float,
    background_outlier_ratio: float,
    importance_color: str,
    thermal_input_type: str,
    minima_root: str | Path,
    minima_checkpoint: str | Path,
    matching_ransac_threshold: float,
    matching_min_matches: int,
    matching_min_inliers: int,
    matching_every_n_frames: int,
    matching_thermal_border_crop: int,
    matching_boundary_feather: int,
    matching_outer_padding: int,
    matching_blank_fill: str,
    matching_cache_size: int,
    preferred_head_key: str | None,
    preferred_thermal_key: str | None,
    episode_index: int | None,
    max_frames: int | None,
    max_frames_per_episode: int | None,
    save_components: tuple[str, ...],
    png_compression: int,
    overwrite: bool,
) -> LocalExportSummary:
    """Decode a local LeRobot dataset and save the displayed views as PNGs."""
    dataset = load_dataset_index(input_path)
    output_root = Path(output_path).expanduser().resolve()
    if output_root == dataset.root:
        raise ValueError("输出目录不能与输入数据集根目录相同。")
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(f"输出路径不是目录：{output_root}")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"输出目录非空：{output_root}。如需覆盖同名导出文件，请加 --overwrite。"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    head_key = _select_export_video_key(
        dataset.head_keys,
        preferred_head_key,
        label="头部 RGB",
    )
    thermal_key = _select_export_video_key(
        dataset.thermal_keys,
        preferred_thermal_key,
        label="热成像",
    )
    components = tuple(dict.fromkeys(save_components))
    unknown_components = set(components) - set(LOCAL_OUTPUT_COMPONENTS)
    if unknown_components:
        raise ValueError(
            f"未知保存组件：{', '.join(sorted(unknown_components))}。"
        )
    if thermal_input_type != "matching":
        components = tuple(
            component
            for component in components
            if component != "matched_thermal"
        )
    if not components:
        raise ValueError("当前模式下 --save-components 没有可保存的图像组件。")

    episodes = dataset.episodes
    if episode_index is not None:
        episodes = tuple(
            episode
            for episode in episodes
            if episode.episode_index == episode_index
        )
        if not episodes:
            raise ValueError(f"数据集中不存在 episode_index={episode_index}。")

    matching_backend = None
    if thermal_input_type == "matching":
        print("正在预加载 MINIMA……", flush=True)
        matching_backend = MinimaThermalMatcher(
            minima_root=minima_root,
            checkpoint=minima_checkpoint,
            ransac_threshold=matching_ransac_threshold,
            min_matches=matching_min_matches,
            min_inliers=matching_min_inliers,
            match_every_n_frames=matching_every_n_frames,
            thermal_border_crop=matching_thermal_border_crop,
            boundary_feather=matching_boundary_feather,
            outer_padding=matching_outer_padding,
            cache_size=matching_cache_size,
        )
        print(f"MINIMA 已加载：{matching_backend.load_ms:.1f} ms", flush=True)

    export_config = {
        "schema_version": 1,
        "input_path": str(dataset.root),
        "fps": dataset.fps,
        "head_key": head_key,
        "thermal_key": thermal_key,
        "thermal_input_type": thermal_input_type,
        "importance_color": importance_color,
        "background_roi": list(background_roi),
        "background_value": background_value,
        "background_outlier_threshold": background_outlier_threshold,
        "background_outlier_ratio": background_outlier_ratio,
        "matching_every_n_frames": matching_every_n_frames,
        "matching_thermal_border_crop": matching_thermal_border_crop,
        "matching_boundary_feather": matching_boundary_feather,
        "matching_outer_padding": matching_outer_padding,
        "matching_blank_fill": matching_blank_fill,
        "save_components": list(components),
        "png_compression": png_compression,
        "episode_filter": episode_index,
        "max_frames": max_frames,
        "max_frames_per_episode": max_frames_per_episode,
    }
    (output_root / "export_config.json").write_text(
        json.dumps(export_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    head_reader = AvFrameReader()
    thermal_reader = AvFrameReader()
    manifest_path = output_root / "manifest.jsonl"
    export_started = time.perf_counter()
    total_frames = 0
    total_images = 0
    exported_episode_count = 0
    try:
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            for episode_position, episode in enumerate(episodes, start=1):
                if max_frames is not None and total_frames >= max_frames:
                    break
                missing = [
                    key
                    for key in (head_key, thermal_key)
                    if key not in episode.streams
                ]
                if missing:
                    raise KeyError(
                        f"Episode {episode.episode_index} 缺少视频元数据："
                        f"{', '.join(missing)}"
                    )

                head_slice = episode.streams[head_key]
                thermal_slice = episode.streams[thermal_key]
                head_reader.open(dataset.video_path(episode, head_key))
                thermal_reader.open(dataset.video_path(episode, thermal_key))
                if matching_backend is not None:
                    matching_backend.reset_sequence()

                frame_count = episode.length
                if max_frames_per_episode is not None:
                    frame_count = min(frame_count, max_frames_per_episode)
                if max_frames is not None:
                    frame_count = min(frame_count, max_frames - total_frames)
                if frame_count <= 0:
                    break
                exported_episode_count += 1
                print(
                    f"[{episode_position}/{len(episodes)}] Episode "
                    f"{episode.episode_index:04d}：导出 {frame_count} 帧",
                    flush=True,
                )

                for frame_index in range(frame_count):
                    relative_timestamp = frame_index / dataset.fps
                    head_timestamp = head_slice.from_timestamp + relative_timestamp
                    thermal_timestamp = (
                        thermal_slice.from_timestamp + relative_timestamp
                    )
                    head_rgb = head_reader.read_rgb(head_timestamp)
                    thermal_rgb = thermal_reader.read_rgb(thermal_timestamp)
                    frame_started = time.perf_counter()
                    processed = process_thermal_frame(
                        head_rgb,
                        thermal_rgb,
                        thermal_input_type=thermal_input_type,
                        background_roi=background_roi,
                        background_value=background_value,
                        background_outlier_threshold=(
                            background_outlier_threshold
                        ),
                        background_outlier_ratio=background_outlier_ratio,
                        importance_color=importance_color,
                        matching_blank_fill=matching_blank_fill,
                        matching_backend=matching_backend,
                        matching_cache_key=(
                            str(dataset.root),
                            episode.episode_index,
                            head_key,
                            thermal_key,
                            frame_index,
                        ),
                        frame_index=frame_index,
                    )

                    arrays = {
                        "head_rgb": head_rgb,
                        "thermal_rgb": thermal_rgb,
                        "matched_thermal": processed.matched_display_rgb,
                        "cold": processed.cold_rgb,
                        "hot": processed.hot_rgb,
                    }
                    relative_paths: dict[str, str] = {}
                    episode_dir = (
                        output_root
                        / "episodes"
                        / f"episode_{episode.episode_index:06d}"
                    )
                    for component in components:
                        image = arrays[component]
                        if image is None:
                            continue
                        image_path = (
                            episode_dir
                            / component
                            / f"frame_{frame_index:06d}.png"
                        )
                        _write_rgb_png(
                            image_path,
                            image,
                            compression=png_compression,
                        )
                        relative_paths[component] = image_path.relative_to(
                            output_root
                        ).as_posix()
                        total_images += 1

                    background = processed.background
                    record: dict[str, Any] = {
                        "episode_index": episode.episode_index,
                        "frame_index": frame_index,
                        "relative_timestamp_s": relative_timestamp,
                        "head_timestamp_s": (
                            head_reader.last_timestamp
                            if head_reader.last_timestamp is not None
                            else head_timestamp
                        ),
                        "thermal_timestamp_s": (
                            thermal_reader.last_timestamp
                            if thermal_reader.last_timestamp is not None
                            else thermal_timestamp
                        ),
                        "tasks": list(episode.tasks),
                        "paths": relative_paths,
                        "background": {
                            "value": background.value,
                            "region_medians": background.region_medians,
                            "used_region_indices": background.used_region_indices,
                            "excluded_region_index": (
                                background.excluded_region_index
                            ),
                            "region_boxes": background.region_boxes,
                        },
                        "processing_ms": (
                            time.perf_counter() - frame_started
                        )
                        * 1000.0,
                    }
                    if processed.matching_frame is not None:
                        matching_frame = processed.matching_frame
                        record["matching"] = {
                            "status": matching_frame.status,
                            "match_ms": matching_frame.match_ms,
                            "warp_ms": matching_frame.warp_ms,
                            "valid_coverage": float(
                                matching_frame.valid_mask.mean()
                            ),
                            "soft_coverage": float(
                                matching_frame.feather_mask.mean()
                            )
                            / 255.0,
                            "metrics": matching_frame.metrics,
                        }
                        if matching_blank_fill == "rgbgray":
                            record["matching"]["blank_fill_ms"] = (
                                processed.blank_fill_ms
                            )
                            record["matching"]["cold_fill_calibration"] = (
                                processed.cold_fill_calibration
                            )
                            record["matching"]["hot_fill_calibration"] = (
                                processed.hot_fill_calibration
                            )
                    manifest_file.write(
                        json.dumps(
                            _json_compatible(record),
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    total_frames += 1
                    if (frame_index + 1) % 100 == 0 or frame_index + 1 == frame_count:
                        elapsed = time.perf_counter() - export_started
                        print(
                            f"  已完成 {frame_index + 1}/{frame_count} 帧；"
                            f"累计 {total_frames} 帧，耗时 {elapsed:.1f}s",
                            flush=True,
                        )
    finally:
        head_reader.close()
        thermal_reader.close()

    summary = LocalExportSummary(
        output_path=output_root,
        episode_count=exported_episode_count,
        frame_count=total_frames,
        image_count=total_images,
        elapsed_s=time.perf_counter() - export_started,
    )
    (output_root / "export_summary.json").write_text(
        json.dumps(_json_compatible(summary.__dict__), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


class ScaledImageLabel(QLabel):
    def __init__(self, placeholder: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._placeholder = placeholder
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 230)
        self.setStyleSheet(
            """
            QLabel {
                background: #111111;
                color: #dddddd;
                border: 1px solid #444444;
            }
            """
        )
        self.clear_image(placeholder)

    def clear_image(self, text: str | None = None) -> None:
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(text or self._placeholder)

    def set_rgb_array(self, image: np.ndarray) -> None:
        image = np.ascontiguousarray(image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"显示图像必须是 HWC RGB，当前 shape={image.shape}。")
        height, width, _ = image.shape
        qimage = QImage(
            image.data,
            width,
            height,
            int(image.strides[0]),
            QImage.Format_RGB888,
        ).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self.setText("")
        self._rescale()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is not None and not self._pixmap.isNull():
            self.setPixmap(
                self._pixmap.scaled(
                    self.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )


class ImagePanel(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = QLabel(title)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setWordWrap(True)
        self.title.setMinimumWidth(0)
        self.title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.title.setStyleSheet(
            "font-size: 17px; font-weight: bold; color: #222222; padding: 3px;"
        )
        self.image = ScaledImageLabel(f"{title}\n等待数据")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.title)
        layout.addWidget(self.image, 1)

    def set_title(self, title: str) -> None:
        self.title.setText(title)


class ClickSlider(QSlider):
    """Horizontal slider that jumps directly to a left-clicked position."""

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton and self.orientation() == Qt.Horizontal:
            ratio = max(0.0, min(1.0, event.x() / max(1, self.width())))
            self.setValue(
                self.minimum()
                + round(ratio * (self.maximum() - self.minimum()))
            )
        super().mousePressEvent(event)


class ThermalTwogreyViewer(QWidget):
    def __init__(
        self,
        *,
        root: str | Path | None = None,
        background_roi: tuple[int, int, int, int] = DEFAULT_BACKGROUND_ROI,
        background_value: float = DEFAULT_BACKGROUND_VALUE,
        background_outlier_threshold: float = DEFAULT_BACKGROUND_OUTLIER_THRESHOLD,
        background_outlier_ratio: float = DEFAULT_BACKGROUND_OUTLIER_RATIO,
        importance_color: str = "original",
        thermal_input_type: str = "twogrey",
        minima_root: str | Path = DEFAULT_MINIMA_ROOT,
        minima_checkpoint: str | Path = DEFAULT_MINIMA_CHECKPOINT,
        matching_ransac_threshold: float = DEFAULT_MATCHING_RANSAC_THRESHOLD,
        matching_min_matches: int = DEFAULT_MATCHING_MIN_MATCHES,
        matching_min_inliers: int = DEFAULT_MATCHING_MIN_INLIERS,
        matching_every_n_frames: int = 1,
        matching_thermal_border_crop: int = DEFAULT_MATCHING_THERMAL_BORDER_CROP,
        matching_boundary_feather: int = DEFAULT_MATCHING_BOUNDARY_FEATHER,
        matching_outer_padding: int = DEFAULT_MATCHING_OUTER_PADDING,
        matching_blank_fill: str = DEFAULT_MATCHING_BLANK_FILL,
        matching_cache_size: int = DEFAULT_MATCHING_CACHE_SIZE,
        preferred_head_key: str | None = None,
        preferred_thermal_key: str | None = None,
        initial_episode: int | None = None,
        start_paused: bool = False,
    ) -> None:
        super().__init__()
        self.dataset: DatasetIndex | None = None
        self.current_episode_position = 0
        self.current_frame_index = 0
        self.is_playing = not start_paused
        self.was_playing_before_scrub = False
        self.preferred_head_key = preferred_head_key
        self.preferred_thermal_key = preferred_thermal_key
        self.initial_episode = initial_episode
        self.background_roi = tuple(int(value) for value in background_roi)
        self.background_value = float(background_value)
        self.background_outlier_threshold = float(background_outlier_threshold)
        self.background_outlier_ratio = float(background_outlier_ratio)
        if importance_color not in IMPORTANCE_COLOR_MODES:
            raise ValueError(
                f"importance_color 必须是 {IMPORTANCE_COLOR_MODES} 之一。"
            )
        self.importance_color = importance_color
        if thermal_input_type not in THERMAL_INPUT_TYPES:
            raise ValueError(
                f"thermal_input_type 必须是 {THERMAL_INPUT_TYPES} 之一。"
            )
        self.thermal_input_type = thermal_input_type
        if matching_blank_fill not in MATCHING_BLANK_FILL_MODES:
            raise ValueError(
                "matching_blank_fill 必须是 "
                f"{MATCHING_BLANK_FILL_MODES} 之一。"
            )
        self.matching_blank_fill = matching_blank_fill
        self.matching_backend: MinimaThermalMatcher | None = None
        self.matching_config = {
            "minima_root": minima_root,
            "checkpoint": minima_checkpoint,
            "ransac_threshold": matching_ransac_threshold,
            "min_matches": matching_min_matches,
            "min_inliers": matching_min_inliers,
            "match_every_n_frames": matching_every_n_frames,
            "thermal_border_crop": matching_thermal_border_crop,
            "boundary_feather": matching_boundary_feather,
            "outer_padding": matching_outer_padding,
            "cache_size": matching_cache_size,
        }

        self.head_reader = AvFrameReader()
        self.thermal_reader = AvFrameReader()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_next_frame)

        self._build_ui()
        self._install_shortcuts()
        self._set_empty_state("请选择一个 LeRobot v3 数据集")
        if self.thermal_input_type == "matching":
            self.info_label.setText("正在预加载 MINIMA，请稍候……")
            QApplication.processEvents()
            self.matching_backend = MinimaThermalMatcher(**self.matching_config)
            self.info_label.setText(
                f"MINIMA 已加载（{self.matching_backend.load_ms:.0f} ms），"
                "请选择数据集。"
            )

        if root:
            self.load_root(root, show_error=True)

    def _build_ui(self) -> None:
        self.setWindowTitle(
            f"G1 热成像数据集浏览器 · thermal_input_type={self.thermal_input_type}"
        )
        self.resize(1820 if self.thermal_input_type == "matching" else 1480, 1030)

        self.select_root_button = QPushButton("选择数据集路径")
        self.select_root_button.clicked.connect(self.select_root)
        self.select_root_button.setStyleSheet(
            """
            QPushButton {
                font-size: 15px;
                font-weight: bold;
                padding: 9px 16px;
                border-radius: 7px;
                color: white;
                background: #2e86de;
            }
            QPushButton:hover { background: #1f6fc1; }
            """
        )
        self.root_label = QLabel("当前数据集路径：未选择")
        self.root_label.setWordWrap(True)
        self.root_label.setStyleSheet(
            "padding: 8px; border: 1px solid #cccccc; background: #fafafa;"
        )
        root_layout = QHBoxLayout()
        root_layout.addWidget(self.select_root_button)
        root_layout.addWidget(self.root_label, 1)

        self.head_key_combo = QComboBox()
        self.thermal_key_combo = QComboBox()
        self.head_key_combo.currentIndexChanged.connect(self.on_video_key_changed)
        self.thermal_key_combo.currentIndexChanged.connect(self.on_video_key_changed)

        settings_layout = QGridLayout()
        settings_layout.addWidget(QLabel("头部 RGB："), 0, 0)
        settings_layout.addWidget(self.head_key_combo, 0, 1)
        settings_layout.addWidget(QLabel("热成像："), 0, 2)
        settings_layout.addWidget(self.thermal_key_combo, 0, 3)
        settings_layout.addWidget(QLabel("输入模式："), 0, 4)
        mode_label = QLabel(self.thermal_input_type)
        mode_label.setStyleSheet("font-weight: bold; color: #8e44ad;")
        settings_layout.addWidget(mode_label, 0, 5)
        settings_layout.setColumnStretch(1, 1)
        settings_layout.setColumnStretch(3, 1)

        roi_settings_layout = QHBoxLayout()
        roi_settings_layout.addStretch()
        roi_settings_layout.addWidget(
            QLabel("顶部三 ROI (top, 边距, height, width)：")
        )

        self.roi_spinboxes: list[QSpinBox] = []
        for index, value in enumerate(self.background_roi):
            spinbox = QSpinBox()
            spinbox.setRange(1 if index >= 2 else 0, 10000)
            spinbox.setValue(value)
            spinbox.setFixedWidth(72)
            self.roi_spinboxes.append(spinbox)
            roi_settings_layout.addWidget(spinbox)

        roi_settings_layout.addWidget(QLabel("背景/边界值："))
        self.background_value_spinbox = QDoubleSpinBox()
        self.background_value_spinbox.setRange(0.0, 255.0)
        self.background_value_spinbox.setDecimals(1)
        self.background_value_spinbox.setValue(self.background_value)
        self.background_value_spinbox.setFixedWidth(80)
        roi_settings_layout.addWidget(self.background_value_spinbox)
        roi_settings_layout.addStretch()
        settings_layout.addLayout(roi_settings_layout, 1, 0, 1, 4)

        robust_settings_layout = QHBoxLayout()
        robust_settings_layout.addStretch()
        robust_settings_layout.addWidget(QLabel("异常差值："))
        self.outlier_threshold_spinbox = QDoubleSpinBox()
        self.outlier_threshold_spinbox.setRange(0.0, 255.0)
        self.outlier_threshold_spinbox.setDecimals(1)
        self.outlier_threshold_spinbox.setValue(
            self.background_outlier_threshold
        )
        self.outlier_threshold_spinbox.setFixedWidth(72)
        robust_settings_layout.addWidget(self.outlier_threshold_spinbox)

        robust_settings_layout.addWidget(QLabel("异常倍率："))
        self.outlier_ratio_spinbox = QDoubleSpinBox()
        self.outlier_ratio_spinbox.setRange(1.0, 100.0)
        self.outlier_ratio_spinbox.setDecimals(1)
        self.outlier_ratio_spinbox.setValue(self.background_outlier_ratio)
        self.outlier_ratio_spinbox.setFixedWidth(68)
        robust_settings_layout.addWidget(self.outlier_ratio_spinbox)

        robust_settings_layout.addWidget(QLabel("重要区域颜色："))
        self.importance_color_combo = QComboBox()
        self.importance_color_combo.addItem("original（冷暗 / 热亮）", "original")
        self.importance_color_combo.addItem(
            "black（训练版：背景白 / 重要黑）",
            "black",
        )
        initial_color_index = self.importance_color_combo.findData(
            self.importance_color
        )
        self.importance_color_combo.setCurrentIndex(initial_color_index)
        robust_settings_layout.addWidget(self.importance_color_combo)

        self.apply_settings_button = QPushButton("应用")
        self.apply_settings_button.clicked.connect(self.apply_twogrey_settings)
        robust_settings_layout.addWidget(self.apply_settings_button)
        robust_settings_layout.addStretch()
        settings_layout.addLayout(robust_settings_layout, 2, 0, 1, 4)

        self.previous_episode_button = QPushButton("◀ 上一个 Episode")
        self.next_episode_button = QPushButton("下一个 Episode ▶")
        self.previous_episode_button.clicked.connect(self.previous_episode)
        self.next_episode_button.clicked.connect(self.next_episode)

        self.episode_combo = QComboBox()
        self.episode_combo.setMinimumWidth(260)
        self.episode_combo.currentIndexChanged.connect(self.on_episode_selected)
        self.episode_label = QLabel("Episode：未加载")
        self.episode_label.setAlignment(Qt.AlignCenter)
        self.episode_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; padding: 4px;"
        )

        episode_layout = QHBoxLayout()
        episode_layout.addWidget(self.previous_episode_button)
        episode_layout.addStretch()
        episode_layout.addWidget(QLabel("跳转："))
        episode_layout.addWidget(self.episode_combo)
        episode_layout.addWidget(self.episode_label, 1)
        episode_layout.addStretch()
        episode_layout.addWidget(self.next_episode_button)

        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-size: 14px; color: #333333;")

        self.head_panel = ImagePanel("头部 RGB")
        self.thermal_panel = ImagePanel("原始热成像")
        self.matched_panel = (
            ImagePanel("MINIMA 匹配后热成像")
            if self.thermal_input_type == "matching"
            else None
        )
        self.cold_panel = ImagePanel("twogrey 冷图")
        self.hot_panel = ImagePanel("twogrey 热图")
        image_grid = QGridLayout()
        image_grid.setSpacing(8)
        image_grid.addWidget(self.head_panel, 0, 0)
        image_grid.addWidget(self.thermal_panel, 0, 1)
        if self.matched_panel is not None:
            image_grid.addWidget(self.matched_panel, 0, 2)
        image_grid.addWidget(self.cold_panel, 1, 0)
        image_grid.addWidget(self.hot_panel, 1, 1)
        image_grid.setRowStretch(0, 1)
        image_grid.setRowStretch(1, 1)
        image_grid.setColumnStretch(0, 1)
        image_grid.setColumnStretch(1, 1)
        if self.matched_panel is not None:
            image_grid.setColumnStretch(2, 1)

        self.frame_slider = ClickSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.sliderPressed.connect(self.on_slider_pressed)
        self.frame_slider.sliderReleased.connect(self.on_slider_released)
        self.frame_slider.valueChanged.connect(self.on_slider_value_changed)

        self.frame_label = QLabel("帧：0 / 0")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.stats_label = QLabel("")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("color: #444444;")

        self.previous_frame_button = QPushButton("◀ 单帧")
        self.play_pause_button = QPushButton("暂停")
        self.next_frame_button = QPushButton("单帧 ▶")
        self.previous_frame_button.clicked.connect(lambda: self.step_frame(-1))
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.next_frame_button.clicked.connect(lambda: self.step_frame(1))
        for button in (
            self.previous_frame_button,
            self.play_pause_button,
            self.next_frame_button,
        ):
            button.setMinimumWidth(110)
            button.setStyleSheet(
                """
                QPushButton {
                    font-size: 15px;
                    padding: 8px 14px;
                    color: white;
                    background: #555555;
                    border-radius: 7px;
                }
                QPushButton:hover:!disabled { background: #3f3f3f; }
                QPushButton:disabled { background: #999999; }
                """
            )

        playback_layout = QHBoxLayout()
        playback_layout.addStretch()
        playback_layout.addWidget(self.previous_frame_button)
        playback_layout.addWidget(self.play_pause_button)
        playback_layout.addWidget(self.next_frame_button)
        playback_layout.addStretch()

        shortcut_label = QLabel(
            "快捷键：Space 播放/暂停；←/→ 单帧；PageUp/PageDown 切换 Episode"
        )
        shortcut_label.setAlignment(Qt.AlignCenter)
        shortcut_label.setStyleSheet("color: #666666;")

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(root_layout)
        main_layout.addLayout(settings_layout)
        main_layout.addLayout(episode_layout)
        main_layout.addWidget(self.info_label)
        main_layout.addLayout(image_grid, 1)
        main_layout.addWidget(self.frame_slider)
        main_layout.addWidget(self.frame_label)
        main_layout.addWidget(self.stats_label)
        main_layout.addLayout(playback_layout)
        main_layout.addWidget(shortcut_label)

    def _install_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_play_pause)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self.step_frame(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self.step_frame(1))
        QShortcut(QKeySequence(Qt.Key_PageUp), self, activated=self.previous_episode)
        QShortcut(QKeySequence(Qt.Key_PageDown), self, activated=self.next_episode)

    def _image_panels(self) -> tuple[ImagePanel, ...]:
        panels = [self.head_panel, self.thermal_panel]
        if self.matched_panel is not None:
            panels.append(self.matched_panel)
        panels.extend((self.cold_panel, self.hot_panel))
        return tuple(panels)

    def _set_empty_state(self, message: str) -> None:
        self.dataset = None
        self.timer.stop()
        self.root_label.setText("当前数据集路径：未加载")
        self.episode_label.setText("Episode：未加载")
        self.info_label.setText(message)
        self.frame_label.setText("帧：0 / 0")
        self.stats_label.setText("")
        for panel in self._image_panels():
            panel.image.clear_image(message)
        for widget in (
            self.previous_episode_button,
            self.next_episode_button,
            self.episode_combo,
            self.head_key_combo,
            self.thermal_key_combo,
            self.importance_color_combo,
            self.outlier_threshold_spinbox,
            self.outlier_ratio_spinbox,
            self.frame_slider,
            self.previous_frame_button,
            self.play_pause_button,
            self.next_frame_button,
            self.apply_settings_button,
        ):
            widget.setEnabled(False)

    @property
    def head_key(self) -> str:
        return self.head_key_combo.currentText()

    @property
    def thermal_key(self) -> str:
        return self.thermal_key_combo.currentText()

    @property
    def current_episode(self) -> EpisodeRecord:
        if self.dataset is None:
            raise RuntimeError("数据集尚未加载。")
        return self.dataset.episodes[self.current_episode_position]

    def select_root(self) -> None:
        start_dir = (
            str(self.dataset.root)
            if self.dataset is not None
            else str(Path.cwd())
        )
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择 LeRobot v3 数据集根目录",
            start_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if selected:
            self.load_root(selected, show_error=True)

    @staticmethod
    def _select_combo_text(
        combo: QComboBox,
        preferred: str | None,
    ) -> None:
        if preferred and combo.findText(preferred) >= 0:
            combo.setCurrentText(preferred)
        elif combo.count() > 0:
            combo.setCurrentIndex(0)

    def load_root(self, root: str | Path, *, show_error: bool) -> bool:
        was_playing = self.is_playing
        self.timer.stop()
        self.head_reader.close()
        self.thermal_reader.close()
        if self.matching_backend is not None:
            self.matching_backend.clear_cache()
        try:
            dataset = load_dataset_index(root)
        except Exception as error:
            self._set_empty_state(f"数据集加载失败：{error}")
            if show_error:
                QMessageBox.critical(self, "数据集加载失败", str(error))
            return False

        self.dataset = dataset
        self.root_label.setText(f"当前数据集路径：{dataset.root}")

        self.head_key_combo.blockSignals(True)
        self.thermal_key_combo.blockSignals(True)
        self.episode_combo.blockSignals(True)

        self.head_key_combo.clear()
        self.head_key_combo.addItems(dataset.head_keys)
        self._select_combo_text(self.head_key_combo, self.preferred_head_key)

        self.thermal_key_combo.clear()
        self.thermal_key_combo.addItems(dataset.thermal_keys)
        self._select_combo_text(self.thermal_key_combo, self.preferred_thermal_key)

        self.episode_combo.clear()
        initial_position = 0
        for position, episode in enumerate(dataset.episodes):
            self.episode_combo.addItem(
                f"Episode {episode.episode_index:04d} · {episode.length} 帧",
                position,
            )
            if episode.episode_index == self.initial_episode:
                initial_position = position
        self.episode_combo.setCurrentIndex(initial_position)

        self.head_key_combo.blockSignals(False)
        self.thermal_key_combo.blockSignals(False)
        self.episode_combo.blockSignals(False)

        for widget in (
            self.episode_combo,
            self.head_key_combo,
            self.thermal_key_combo,
            self.importance_color_combo,
            self.outlier_threshold_spinbox,
            self.outlier_ratio_spinbox,
            self.frame_slider,
            self.previous_frame_button,
            self.play_pause_button,
            self.next_frame_button,
            self.apply_settings_button,
        ):
            widget.setEnabled(True)

        self.is_playing = was_playing
        self.current_episode_position = initial_position
        if not self.load_episode(initial_position, show_error=show_error):
            return False
        return True

    def _validate_selected_streams(self, episode: EpisodeRecord) -> None:
        missing = [
            key
            for key in (self.head_key, self.thermal_key)
            if key not in episode.streams
        ]
        if missing:
            raise KeyError(
                f"Episode {episode.episode_index} 缺少视频元数据：{', '.join(missing)}"
            )

    def _update_twogrey_panel_titles(self) -> None:
        source_prefix = (
            "matching 后"
            if self.thermal_input_type == "matching"
            else "twogrey "
        )
        if self.importance_color == "black":
            self.cold_panel.set_title(
                f"{source_prefix}冷图（背景白，越冷越黑）"
            )
            self.hot_panel.set_title(
                f"{source_prefix}热图（背景白，越热越黑）"
            )
        else:
            self.cold_panel.set_title(
                f"{source_prefix}冷图（≤ 背景边界）"
            )
            self.hot_panel.set_title(
                f"{source_prefix}热图（≥ 背景边界）"
            )

    def load_episode(self, position: int, *, show_error: bool = True) -> bool:
        if self.dataset is None:
            return False
        position = max(0, min(position, len(self.dataset.episodes) - 1))
        episode = self.dataset.episodes[position]

        try:
            self._validate_selected_streams(episode)
            head_path = self.dataset.video_path(episode, self.head_key)
            thermal_path = self.dataset.video_path(episode, self.thermal_key)
            self.head_reader.open(head_path)
            self.thermal_reader.open(thermal_path)
        except Exception as error:
            self.timer.stop()
            self.is_playing = False
            self.update_play_button()
            self.info_label.setText(f"Episode 加载失败：{error}")
            if show_error:
                QMessageBox.critical(self, "Episode 加载失败", str(error))
            return False

        self.current_episode_position = position
        self.current_frame_index = 0
        if self.matching_backend is not None:
            self.matching_backend.reset_sequence()
        self.episode_combo.blockSignals(True)
        self.episode_combo.setCurrentIndex(position)
        self.episode_combo.blockSignals(False)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, episode.length - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)

        self.previous_episode_button.setEnabled(position > 0)
        self.next_episode_button.setEnabled(position < len(self.dataset.episodes) - 1)
        self.episode_label.setText(
            f"Episode {episode.episode_index:04d} "
            f"（{position + 1}/{len(self.dataset.episodes)}）"
        )
        self.head_panel.set_title(f"头部 RGB · {self.head_key}")
        self.thermal_panel.set_title(f"原始热成像 · {self.thermal_key}")
        if self.matched_panel is not None:
            self.matched_panel.set_title("MINIMA 匹配后热成像")
        self._update_twogrey_panel_titles()

        self.timer.setInterval(max(1, round(1000.0 / self.dataset.fps)))
        self.show_frame(0)
        self.update_play_button()
        if self.is_playing:
            self.timer.start()
        return True

    def show_frame(self, frame_index: int) -> None:
        if self.dataset is None:
            return
        episode = self.current_episode
        frame_index = max(0, min(frame_index, episode.length - 1))
        head_slice = episode.streams[self.head_key]
        thermal_slice = episode.streams[self.thermal_key]
        relative_timestamp = frame_index / self.dataset.fps
        head_timestamp = head_slice.from_timestamp + relative_timestamp
        thermal_timestamp = thermal_slice.from_timestamp + relative_timestamp

        frame_started = time.perf_counter()
        try:
            head_rgb = self.head_reader.read_rgb(head_timestamp)
            thermal_rgb = self.thermal_reader.read_rgb(thermal_timestamp)
            processed = process_thermal_frame(
                head_rgb,
                thermal_rgb,
                thermal_input_type=self.thermal_input_type,
                background_roi=self.background_roi,
                background_value=self.background_value,
                background_outlier_threshold=self.background_outlier_threshold,
                background_outlier_ratio=self.background_outlier_ratio,
                importance_color=self.importance_color,
                matching_blank_fill=self.matching_blank_fill,
                matching_backend=self.matching_backend,
                matching_cache_key=(
                    str(self.dataset.root),
                    episode.episode_index,
                    self.head_key,
                    self.thermal_key,
                    frame_index,
                ),
                frame_index=frame_index,
            )
        except Exception as error:
            self.timer.stop()
            self.is_playing = False
            self.update_play_button()
            self.info_label.setText(f"帧处理失败：{error}")
            return
        matching_frame = processed.matching_frame
        split_source = processed.split_source_rgb
        cold_rgb = processed.cold_rgb
        hot_rgb = processed.hot_rgb
        background = processed.background
        blank_fill_ms = processed.blank_fill_ms
        cold_fill_calibration = processed.cold_fill_calibration
        hot_fill_calibration = processed.hot_fill_calibration
        self.head_panel.image.set_rgb_array(head_rgb)
        self.thermal_panel.image.set_rgb_array(thermal_rgb)
        if self.matched_panel is not None and processed.matched_display_rgb is not None:
            self.matched_panel.image.set_rgb_array(processed.matched_display_rgb)
        self.cold_panel.image.set_rgb_array(cold_rgb)
        self.hot_panel.image.set_rgb_array(hot_rgb)
        frame_ms = (time.perf_counter() - frame_started) * 1000.0

        self.current_frame_index = frame_index
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_index)
        self.frame_slider.blockSignals(False)
        head_actual_timestamp = (
            self.head_reader.last_timestamp
            if self.head_reader.last_timestamp is not None
            else head_timestamp
        )
        thermal_actual_timestamp = (
            self.thermal_reader.last_timestamp
            if self.thermal_reader.last_timestamp is not None
            else thermal_timestamp
        )
        head_relative_timestamp = head_actual_timestamp - head_slice.from_timestamp
        thermal_relative_timestamp = (
            thermal_actual_timestamp - thermal_slice.from_timestamp
        )
        sync_error_ms = (
            abs(head_relative_timestamp - thermal_relative_timestamp) * 1000.0
        )
        self.frame_label.setText(
            f"帧：{frame_index} / {episode.length - 1}    "
            f"目标时间：{relative_timestamp:.3f}s    "
            f"实际 RGB/热：{head_relative_timestamp:.3f}s/"
            f"{thermal_relative_timestamp:.3f}s    "
            f"时间差：{sync_error_ms:.1f} ms    FPS：{self.dataset.fps:g}"
        )

        task_text = "；".join(episode.tasks) if episode.tasks else "未记录"
        self.info_label.setText(
            f"任务：{task_text}    "
            f"头部视频：chunk-{head_slice.chunk_index:03d}/file-{head_slice.file_index:03d}    "
            f"热成像视频：chunk-{thermal_slice.chunk_index:03d}/file-{thermal_slice.file_index:03d}"
        )
        thermal_channel = split_source[..., 0]
        medians_text = " / ".join(
            f"{name}={median:.1f}"
            for name, median in zip(
                BACKGROUND_REGION_NAMES,
                background.region_medians,
                strict=True,
            )
        )
        if background.excluded_region_index is None:
            selection_text = "采用三个区域"
        else:
            excluded_name = BACKGROUND_REGION_NAMES[
                background.excluded_region_index
            ]
            selection_text = f"排除{excluded_name}，采用其他两个区域"
        stats_text = (
            f"顶部 ROI 中位数：{medians_text}    {selection_text}    "
            f"背景原始值={background.value:.1f}\n"
            f"显示模式={self.importance_color}    "
            f"背景/分界值={self.background_value:.1f}    "
            f"{'匹配后' if matching_frame is not None else '原始'}第0通道范围="
            f"{int(thermal_channel.min())}~{int(thermal_channel.max())}    "
            f"冷图范围={int(cold_rgb.min())}~{int(cold_rgb.max())}    "
            f"热图范围={int(hot_rgb.min())}~{int(hot_rgb.max())}"
        )
        if matching_frame is not None:
            metrics = matching_frame.metrics
            cached_ms = metrics.get("cached_match_ms")
            match_time_text = (
                f"缓存（首次 {float(cached_ms):.1f} ms）"
                if cached_ms is not None
                else f"{matching_frame.match_ms:.1f} ms"
            )
            valid_coverage = float(matching_frame.valid_mask.mean()) * 100.0
            soft_coverage = (
                float(matching_frame.feather_mask.mean()) / 255.0 * 100.0
            )
            raw_matches = int(metrics.get("raw_matches", 0))
            inliers = int(metrics.get("inliers", 0))
            reproj_median = metrics.get("reproj_median_px", np.nan)
            reproj_text = (
                f"{float(reproj_median):.2f} px"
                if np.isfinite(reproj_median)
                else "—"
            )
            fallback_reason = str(metrics.get("fallback_reason") or "")
            fallback_text = (
                f"    fallback={fallback_reason}"
                if fallback_reason
                else ""
            )
            stats_text += (
                f"\nMINIMA：{matching_frame.status}    匹配={match_time_text}    "
                f"warp={matching_frame.warp_ms:.1f} ms    "
                f"匹配点/内点={raw_matches}/{inliers}    "
                f"重投影中位数={reproj_text}    "
                f"有效/软覆盖={valid_coverage:.1f}%/{soft_coverage:.1f}%"
                f"{fallback_text}"
            )
            if self.matching_blank_fill == "rgbgray":
                cold_scale, cold_offset = cold_fill_calibration
                hot_scale, hot_offset = hot_fill_calibration
                stats_text += (
                    f"\n冷/热空白填充=RGB 灰度    耗时={blank_fill_ms:.1f} ms    "
                    f"冷校准={cold_scale:.2f}/{cold_offset:+.1f}    "
                    f"热校准={hot_scale:.2f}/{hot_offset:+.1f}"
                )
        stats_text += f"\n本帧总处理={frame_ms:.1f} ms"
        self.stats_label.setText(stats_text)

    def update_play_button(self) -> None:
        self.play_pause_button.setText("暂停" if self.is_playing else "播放")

    def toggle_play_pause(self) -> None:
        if self.dataset is None:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.timer.start()
        else:
            self.timer.stop()
        self.update_play_button()

    def play_next_frame(self) -> None:
        if self.dataset is None or not self.is_playing:
            return
        next_frame = self.current_frame_index + 1
        if next_frame >= self.current_episode.length:
            next_frame = 0
        self.show_frame(next_frame)

    def step_frame(self, delta: int) -> None:
        if self.dataset is None:
            return
        self.is_playing = False
        self.timer.stop()
        self.update_play_button()
        target = max(
            0,
            min(self.current_frame_index + delta, self.current_episode.length - 1),
        )
        self.show_frame(target)

    def on_slider_pressed(self) -> None:
        self.was_playing_before_scrub = self.is_playing
        self.timer.stop()

    def on_slider_value_changed(self, value: int) -> None:
        if self.dataset is not None:
            self.show_frame(value)

    def on_slider_released(self) -> None:
        if self.was_playing_before_scrub and self.is_playing:
            self.timer.start()

    def on_episode_selected(self, combo_index: int) -> None:
        if self.dataset is None or combo_index < 0:
            return
        position = self.episode_combo.itemData(combo_index)
        if position is not None and int(position) != self.current_episode_position:
            self.load_episode(int(position))

    def previous_episode(self) -> None:
        if self.dataset is not None and self.current_episode_position > 0:
            self.load_episode(self.current_episode_position - 1)

    def next_episode(self) -> None:
        if (
            self.dataset is not None
            and self.current_episode_position < len(self.dataset.episodes) - 1
        ):
            self.load_episode(self.current_episode_position + 1)

    def on_video_key_changed(self) -> None:
        if (
            self.dataset is not None
            and self.head_key_combo.currentIndex() >= 0
            and self.thermal_key_combo.currentIndex() >= 0
        ):
            self.load_episode(self.current_episode_position)

    def apply_twogrey_settings(self) -> None:
        self.background_roi = tuple(
            spinbox.value() for spinbox in self.roi_spinboxes
        )
        self.background_value = self.background_value_spinbox.value()
        self.background_outlier_threshold = self.outlier_threshold_spinbox.value()
        self.background_outlier_ratio = self.outlier_ratio_spinbox.value()
        self.importance_color = str(self.importance_color_combo.currentData())
        self._update_twogrey_panel_titles()
        if self.dataset is not None:
            self.show_frame(self.current_frame_index)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.timer.stop()
        self.head_reader.close()
        self.thermal_reader.close()
        super().closeEvent(event)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "打开 LeRobot v3 数据集，逐 episode 展示头部 RGB、原始热成像和 "
            "冷/热图；matching 模式还会展示 MINIMA 匹配后的热成像。"
        )
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default=None,
        help=(
            "LeRobot v3 数据集根目录；省略时可在界面中选择。"
            "本地批量导出也可改用 --input-path"
        ),
    )
    parser.add_argument(
        "--input-path",
        "--input_path",
        "--input-dir",
        "--input",
        default=None,
        help="本地 LeRobot v3 数据集根目录；可替代位置参数 dataset_root",
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        "--output-dir",
        "--output",
        default=None,
        help=(
            "本地导出目录；指定后不启动 GUI，而是批量保存各帧图像、"
            "export_config.json 和 manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--save-components",
        "--save_components",
        nargs="+",
        choices=LOCAL_OUTPUT_COMPONENTS,
        default=LOCAL_OUTPUT_COMPONENTS,
        metavar="COMPONENT",
        help=(
            "本地模式保存的图像组件，可选 head_rgb thermal_rgb "
            "matched_thermal cold hot；默认全部保存。twogrey 模式自动忽略 "
            "matched_thermal"
        ),
    )
    parser.add_argument(
        "--max-frames",
        "--max_frames",
        type=int,
        default=None,
        help="本地模式累计导出到 N 帧后立即结束并保存；默认不限制",
    )
    parser.add_argument(
        "--max-frames-per-episode",
        "--max_frames_per_episode",
        type=int,
        default=None,
        help="本地模式每个 episode 最多导出的帧数；默认不单独限制",
    )
    parser.add_argument(
        "--png-compression",
        "--png_compression",
        type=int,
        default=1,
        help="本地 PNG 压缩级别 0~9；默认 1，优先导出速度",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许本地模式覆盖输出目录中的同名文件；不会删除其他文件",
    )
    parser.add_argument(
        "--thermal-input-type",
        "--thermal_input_type",
        choices=THERMAL_INPUT_TYPES,
        default="twogrey",
        help=(
            "twogrey 直接从原始热成像生成冷/热图；matching 启动时预加载 "
            "MINIMA，将热成像匹配到头部 RGB 后再生成冷/热图。默认：twogrey"
        ),
    )
    parser.add_argument(
        "--background-roi",
        "--roi",
        nargs=4,
        type=int,
        metavar=("TOP", "SIDE_MARGIN", "HEIGHT", "WIDTH"),
        default=DEFAULT_BACKGROUND_ROI,
        help=(
            "顶部三个背景 ROI 的 top、左右边距、height、width；"
            "中间 ROI 自动水平居中。默认：0 0 64 64"
        ),
    )
    parser.add_argument(
        "--background-value",
        type=float,
        default=DEFAULT_BACKGROUND_VALUE,
        help="背景归一化目标值和冷/热分界值，默认：80",
    )
    parser.add_argument(
        "--background-outlier-threshold",
        type=float,
        default=DEFAULT_BACKGROUND_OUTLIER_THRESHOLD,
        help=(
            "排除异常背景区域所需的最小绝对中位数差值，"
            f"默认：{DEFAULT_BACKGROUND_OUTLIER_THRESHOLD:g}"
        ),
    )
    parser.add_argument(
        "--background-outlier-ratio",
        type=float,
        default=DEFAULT_BACKGROUND_OUTLIER_RATIO,
        help=(
            "异常区域差值相对于另外两个区域差值的最小倍率，"
            f"默认：{DEFAULT_BACKGROUND_OUTLIER_RATIO:g}"
        ),
    )
    parser.add_argument(
        "--importance-color",
        "--color-mode",
        choices=IMPORTANCE_COLOR_MODES,
        default="black",
        help=(
            "original 保持冷图越冷越暗、热图越热越亮；"
            "black 将阈值两侧分别线性映射到完整 0~255，"
            "使两图均为背景白、温差越显著越黑"
        ),
    )
    parser.add_argument(
        "--head-key",
        default=None,
        help="初始头部 RGB 视频特征名，默认自动选择 cam_left_high/head/high",
    )
    parser.add_argument(
        "--thermal-key",
        default=None,
        help="初始热成像视频特征名，默认自动选择包含 thermal 的特征",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help=(
            "GUI 启动后打开的 episode_index；本地模式只导出该 episode。"
            "本地模式省略时导出全部 episode"
        ),
    )
    parser.add_argument(
        "--minima-root",
        "--minima_root",
        default=DEFAULT_MINIMA_ROOT,
        help=f"MINIMA 代码目录，默认：{DEFAULT_MINIMA_ROOT}",
    )
    parser.add_argument(
        "--minima-checkpoint",
        "--minima_checkpoint",
        default=DEFAULT_MINIMA_CHECKPOINT,
        help=(
            "MINIMA checkpoint；相对路径会同时尝试相对于 MINIMA 目录解析，"
            f"默认：{DEFAULT_MINIMA_CHECKPOINT}"
        ),
    )
    parser.add_argument(
        "--matching-ransac-threshold",
        "--matching_ransac_threshold",
        type=float,
        default=DEFAULT_MATCHING_RANSAC_THRESHOLD,
        help=(
            "matching 单应矩阵的 RANSAC 重投影阈值，默认："
            f"{DEFAULT_MATCHING_RANSAC_THRESHOLD:g}"
        ),
    )
    parser.add_argument(
        "--matching-min-matches",
        "--matching_min_matches",
        type=int,
        default=DEFAULT_MATCHING_MIN_MATCHES,
        help=f"计算单应矩阵所需的最少匹配点，默认：{DEFAULT_MATCHING_MIN_MATCHES}",
    )
    parser.add_argument(
        "--matching-min-inliers",
        "--matching_min_inliers",
        type=int,
        default=DEFAULT_MATCHING_MIN_INLIERS,
        help=f"接受单应矩阵所需的最少内点，默认：{DEFAULT_MATCHING_MIN_INLIERS}",
    )
    parser.add_argument(
        "--matching-every-n-frames",
        "--matching_every_n_frames",
        type=int,
        default=1,
        help=(
            "每 N 帧运行一次 MINIMA，其余顺序播放帧复用上一单应矩阵；"
            "默认 1，即每帧匹配"
        ),
    )
    parser.add_argument(
        "--matching-thermal-border-crop",
        "--matching_thermal_border_crop",
        type=int,
        default=DEFAULT_MATCHING_THERMAL_BORDER_CROP,
        help=(
            "warp 前从热成像有效 mask 四边排除的像素数，默认："
            f"{DEFAULT_MATCHING_THERMAL_BORDER_CROP}"
        ),
    )
    parser.add_argument(
        "--matching-boundary-feather",
        "--matching_boundary_feather",
        type=int,
        default=DEFAULT_MATCHING_BOUNDARY_FEATHER,
        help=(
            "在原热成像内部做边缘衰减的宽度；0 表示不修改内部边缘，默认："
            f"{DEFAULT_MATCHING_BOUNDARY_FEATHER}"
        ),
    )
    parser.add_argument(
        "--matching-outer-padding",
        "--matching_outer_padding",
        type=int,
        default=DEFAULT_MATCHING_OUTER_PADDING,
        help=(
            "在热成像边框外复制最外圈像素并向外渐变到 0 的宽度；"
            f"0 表示关闭，默认：{DEFAULT_MATCHING_OUTER_PADDING}"
        ),
    )
    parser.add_argument(
        "--matching-blank-fill",
        "--matching_blank_fill",
        choices=MATCHING_BLANK_FILL_MODES,
        default=DEFAULT_MATCHING_BLANK_FILL,
        help=(
            "rgbgray 使用分别经过边界亮度校准的头部 RGB 灰度填充冷/热图"
            "空白；white 保持冷/热图空白为白色。彩色热成像不受此项影响。"
            f"默认：{DEFAULT_MATCHING_BLANK_FILL}"
        ),
    )
    parser.add_argument(
        "--matching-cache-size",
        "--matching_cache_size",
        type=int,
        default=DEFAULT_MATCHING_CACHE_SIZE,
        help=(
            "缓存的逐帧单应矩阵数量，回看缓存帧不会再次运行 MINIMA，默认："
            f"{DEFAULT_MATCHING_CACHE_SIZE}"
        ),
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="启动后暂停在第一帧",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input_path or args.dataset_root
    if args.input_path is not None and args.dataset_root is not None:
        positional_root = Path(args.dataset_root).expanduser().resolve()
        option_root = Path(args.input_path).expanduser().resolve()
        if positional_root != option_root:
            raise SystemExit(
                "位置参数 dataset_root 与 --input-path 指向不同目录，请只保留一个。"
            )
    roi = tuple(args.background_roi)
    if roi[0] < 0 or roi[1] < 0 or roi[2] <= 0 or roi[3] <= 0:
        raise SystemExit(
            "--background-roi 需要非负的 TOP/SIDE_MARGIN 和大于 0 的 HEIGHT/WIDTH。"
        )
    if not 0.0 <= args.background_value <= 255.0:
        raise SystemExit("--background-value 必须在 [0, 255] 范围内。")
    if args.background_outlier_threshold < 0:
        raise SystemExit("--background-outlier-threshold 不能为负数。")
    if args.background_outlier_ratio < 1:
        raise SystemExit("--background-outlier-ratio 必须大于或等于 1。")
    if args.matching_ransac_threshold <= 0:
        raise SystemExit("--matching-ransac-threshold 必须大于 0。")
    if args.matching_min_matches < 4:
        raise SystemExit("--matching-min-matches 必须大于或等于 4。")
    if args.matching_min_inliers < 4:
        raise SystemExit("--matching-min-inliers 必须大于或等于 4。")
    if args.matching_every_n_frames < 1:
        raise SystemExit("--matching-every-n-frames 必须大于或等于 1。")
    if args.matching_thermal_border_crop < 0:
        raise SystemExit("--matching-thermal-border-crop 不能为负数。")
    if args.matching_boundary_feather < 0:
        raise SystemExit("--matching-boundary-feather 不能为负数。")
    if args.matching_outer_padding < 0:
        raise SystemExit("--matching-outer-padding 不能为负数。")
    if args.matching_cache_size < 1:
        raise SystemExit("--matching-cache-size 必须大于或等于 1。")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames 必须大于或等于 1。")
    if args.max_frames_per_episode is not None and args.max_frames_per_episode < 1:
        raise SystemExit("--max-frames-per-episode 必须大于或等于 1。")
    if not 0 <= args.png_compression <= 9:
        raise SystemExit("--png-compression 必须在 [0, 9] 范围内。")
    if args.output_path is not None and input_path is None:
        raise SystemExit("本地导出模式需要通过 --input-path 或 dataset_root 指定输入。")

    if args.output_path is not None:
        try:
            summary = export_dataset_locally(
                input_path,
                args.output_path,
                background_roi=roi,
                background_value=args.background_value,
                background_outlier_threshold=args.background_outlier_threshold,
                background_outlier_ratio=args.background_outlier_ratio,
                importance_color=args.importance_color,
                thermal_input_type=args.thermal_input_type,
                minima_root=args.minima_root,
                minima_checkpoint=args.minima_checkpoint,
                matching_ransac_threshold=args.matching_ransac_threshold,
                matching_min_matches=args.matching_min_matches,
                matching_min_inliers=args.matching_min_inliers,
                matching_every_n_frames=args.matching_every_n_frames,
                matching_thermal_border_crop=args.matching_thermal_border_crop,
                matching_boundary_feather=args.matching_boundary_feather,
                matching_outer_padding=args.matching_outer_padding,
                matching_blank_fill=args.matching_blank_fill,
                matching_cache_size=args.matching_cache_size,
                preferred_head_key=args.head_key,
                preferred_thermal_key=args.thermal_key,
                episode_index=args.episode,
                max_frames=args.max_frames,
                max_frames_per_episode=args.max_frames_per_episode,
                save_components=tuple(args.save_components),
                png_compression=args.png_compression,
                overwrite=args.overwrite,
            )
        except Exception as error:
            print(f"本地导出失败：{error}", file=sys.stderr)
            return 2
        print(
            "本地导出完成："
            f"{summary.episode_count} 个 episode，"
            f"{summary.frame_count} 帧，{summary.image_count} 张图，"
            f"耗时 {summary.elapsed_s:.1f}s；输出：{summary.output_path}"
        )
        return 0

    app = QApplication(sys.argv)
    app.setApplicationName("G1 thermal dataset viewer")
    try:
        viewer = ThermalTwogreyViewer(
            root=input_path,
            background_roi=roi,
            background_value=args.background_value,
            background_outlier_threshold=args.background_outlier_threshold,
            background_outlier_ratio=args.background_outlier_ratio,
            importance_color=args.importance_color,
            thermal_input_type=args.thermal_input_type,
            minima_root=args.minima_root,
            minima_checkpoint=args.minima_checkpoint,
            matching_ransac_threshold=args.matching_ransac_threshold,
            matching_min_matches=args.matching_min_matches,
            matching_min_inliers=args.matching_min_inliers,
            matching_every_n_frames=args.matching_every_n_frames,
            matching_thermal_border_crop=args.matching_thermal_border_crop,
            matching_boundary_feather=args.matching_boundary_feather,
            matching_outer_padding=args.matching_outer_padding,
            matching_blank_fill=args.matching_blank_fill,
            matching_cache_size=args.matching_cache_size,
            preferred_head_key=args.head_key,
            preferred_thermal_key=args.thermal_key,
            initial_episode=args.episode,
            start_paused=args.paused,
        )
    except Exception as error:
        message = f"查看器启动失败：{error}"
        print(message, file=sys.stderr)
        QMessageBox.critical(None, "查看器启动失败", message)
        return 2
    viewer.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
