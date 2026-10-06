#!/usr/bin/env python3
#旧版本： python unitree_lerobot/utils/merge_rgb_and_thermography.py   ./two_bottle_selection_right_with_thermography   --d -85  --max-frames 300 --thermal-width 600 --overwrite
"""为 LeRobot 数据集生成预混合 RGB/thermal 视频特征。

默认会读取数据集里的：

    videos/observation.images.cam_left_high/...
    videos/observation.images.cam_thermal/...

并生成：

    videos/observation.images.cam_rgb_thermal_mixing/...

输出会复用 RGB 视频的 chunk/file 结构，例如：

    cold_water_bottle_selection/videos/observation.images.cam_rgb_thermal_mixing/chunk-000/file-000.mp4
    cold_water_bottle_selection/videos/observation.images.cam_rgb_thermal_mixing/chunk-000/file-001.mp4

示例：
    python unitree_lerobot/utils/merge_rgb_and_thermography.py \
        ./cold_water_bottle_selection --overwrite

调参示例：
    python unitree_lerobot/utils/merge_rgb_and_thermography.py \
        ./two_bottle_selection_right_with_thermography \
        --d -85 --max-frames 300 --thermal-width 600 --overwrite

当输入视频宽度为 W 时，d > 0 表示热成像相对 RGB 向左移动，
d < 0 表示向右移动；dy > 0 表示向上移动，dy < 0 表示向下移动。
热成像先被水平缩放至指定宽度，然后与 RGB 融合；超出 RGB 画布
的部分会被裁掉，未被热成像覆盖的区域使用热成像首帧的众数颜色
填充，热成像边界默认会做羽化过渡，因此输出尺寸始终与 RGB 相同。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path


DEFAULT_RGB_FEATURE = "observation.images.cam_left_high"
DEFAULT_THERMAL_FEATURE = "observation.images.cam_thermal"
DEFAULT_OUTPUT_FEATURE = "observation.images.cam_rgb_thermal_mixing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按 80% 热成像和 20% RGB 为 LeRobot 数据集生成 "
            "observation.images.cam_rgb_thermal_mixing 视频特征。"
        )
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="数据集根目录，例如 ./cold_water_bottle_selection",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=-85,
        help="热成像相对 RGB 的水平偏移量：正数向左，负数向右（默认：-85）。",
    )
    parser.add_argument(
        "--dy",
        "--vertical-offset",
        dest="dy",
        type=int,
        default=0,
        help="热成像的垂直偏移量：正数向上，负数向下（默认：0）。",
    )
    parser.add_argument(
        "--thermal-width",
        type=int,
        default=600,
        help="热成像画面的目标宽度（默认：600），高度保持与 RGB 相同。",
    )
    parser.add_argument(
        "--boundary-feather",
        type=int,
        default=15,
        help="热成像覆盖边界的羽化半径，单位像素；0 表示关闭边界平滑（默认：15）。",
    )
    parser.add_argument(
        "--thermal-border-crop",
        type=int,
        default=10,
        help="融合前忽略热成像最外圈的像素数，用于去掉热成像自带深色边框（默认：10）。",
    )
    parser.add_argument(
        "--rgb-feature",
        default=DEFAULT_RGB_FEATURE,
        help=f"RGB 视频特征名（默认：{DEFAULT_RGB_FEATURE}）。",
    )
    parser.add_argument(
        "--thermal-feature",
        default=DEFAULT_THERMAL_FEATURE,
        help=f"thermal 视频特征名（默认：{DEFAULT_THERMAL_FEATURE}）。",
    )
    parser.add_argument(
        "--output-feature",
        default=DEFAULT_OUTPUT_FEATURE,
        help=f"输出 mixed 视频特征名（默认：{DEFAULT_OUTPUT_FEATURE}）。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="每个视频只处理开头的指定帧数；默认处理完整视频。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的 mixed 视频。",
    )
    return parser.parse_args()


def get_feature_video_dir(dataset_path: Path, feature: str) -> Path:
    """返回 LeRobot 视频特征目录。"""
    return dataset_path / "videos" / feature


def iter_video_pairs(
    dataset_path: Path,
    rgb_feature: str,
    thermal_feature: str,
    output_feature: str,
) -> list[tuple[Path, Path, Path, float]]:
    """按 RGB 的 chunk/file 结构生成 RGB、thermal、output 路径和 thermal 起始秒数。"""
    rgb_root = get_feature_video_dir(dataset_path, rgb_feature)
    thermal_root = get_feature_video_dir(dataset_path, thermal_feature)
    output_root = get_feature_video_dir(dataset_path, output_feature)

    if not rgb_root.is_dir():
        raise FileNotFoundError(f"找不到 RGB 视频目录：{rgb_root}")
    if not thermal_root.is_dir():
        raise FileNotFoundError(f"找不到热成像视频目录：{thermal_root}")

    rgb_files = sorted(rgb_root.rglob("*.mp4"))
    thermal_files = sorted(thermal_root.rglob("*.mp4"))
    if not rgb_files:
        raise FileNotFoundError(f"RGB 视频目录下没有 mp4 文件：{rgb_root}")
    if not thermal_files:
        raise FileNotFoundError(f"热成像视频目录下没有 mp4 文件：{thermal_root}")

    pairs = []
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

        pairs.append((rgb_path, thermal_path, output_path, thermal_start_s))
        cumulative_rgb_duration_s += float(probe_video(rgb_path)["duration"])

    if missing_thermal_files:
        preview = "\n".join(f"  - {path}" for path in missing_thermal_files[:10])
        suffix = "" if len(missing_thermal_files) <= 10 else f"\n  ... 以及 {len(missing_thermal_files) - 10} 个"
        raise FileNotFoundError(
            "thermal 目录没有与 RGB 对应的 chunk/file 视频，且 thermal 不是单文件时间轴：\n"
            f"{preview}{suffix}"
        )

    return pairs


def probe_video(path: Path) -> dict[str, int | Fraction]:
    """使用 ffprobe 获取视频流的基本参数。"""
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
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"没有在文件中找到视频流：{path}")

    stream = streams[0]
    frame_rate = Fraction(stream["avg_frame_rate"])
    if frame_rate <= 0:
        raise ValueError(f"无法读取有效帧率：{path}")
    duration = float(stream.get("duration") or 0.0)
    if duration <= 0 and stream.get("nb_frames") not in (None, "N/A"):
        duration = int(stream["nb_frames"]) / float(frame_rate)
    if duration <= 0:
        raise ValueError(f"无法读取有效时长：{path}")

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": frame_rate,
        "duration": duration,
    }


def get_modal_color(path: Path, start_s: float = 0.0, border_crop: int = 0) -> tuple[int, int, int]:
    """统计热成像首帧中出现次数最多的 RGB 颜色。"""
    command = [
        "ffmpeg",
        "-v",
        "error",
    ]
    if start_s > 0:
        command.extend(["-ss", f"{start_s:.6f}"])
    command.extend([
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ])
    frame = subprocess.run(command, check=True, capture_output=True).stdout
    if not frame or len(frame) % 3 != 0:
        raise ValueError(f"无法从热成像视频首帧统计众数颜色：{path}")

    if border_crop > 0:
        info = probe_video(path)
        width = int(info["width"])
        height = int(info["height"])
        if border_crop * 2 < min(width, height):
            cropped_colors = []
            row_size = width * 3
            for y in range(border_crop, height - border_crop):
                row = frame[y * row_size : (y + 1) * row_size]
                x0 = border_crop * 3
                x1 = (width - border_crop) * 3
                cropped_colors.extend(zip(row[x0:x1:3], row[x0 + 1 : x1 : 3], row[x0 + 2 : x1 : 3]))
            frame_colors = cropped_colors
        else:
            frame_colors = zip(frame[0::3], frame[1::3], frame[2::3])
    else:
        frame_colors = zip(frame[0::3], frame[1::3], frame[2::3])

    colors = Counter(frame_colors)
    return colors.most_common(1)[0][0]


def make_filter(
    height: int,
    d: int,
    dy: int,
    thermal_width: int,
    modal_color: tuple[int, int, int],
    boundary_feather: int = 15,
    thermal_border_crop: int = 10,
) -> str:
    """缩放并偏移热成像，用众数颜色补齐，边界羽化后以 80% 权重叠加。"""
    thermal_x = -d
    thermal_y = -dy
    fill_color = "0x{:02x}{:02x}{:02x}".format(*modal_color)
    if thermal_border_crop > 0:
        crop = int(thermal_border_crop)
        cropped_width = thermal_width - crop * 2
        cropped_height = height - crop * 2
        scaled_thermal = (
            f"[1:v]scale={thermal_width}:{height}:flags=lanczos,format=rgba,"
            f"crop={cropped_width}:{cropped_height}:{crop}:{crop},"
            f"pad={thermal_width}:{height}:{crop}:{crop}:color=black@0"
        )
    else:
        scaled_thermal = f"[1:v]scale={thermal_width}:{height}:flags=lanczos,format=rgba"

    if boundary_feather <= 0:
        thermal_filter = (
            f"{scaled_thermal}[thermal];"
            f"[fill][thermal]overlay=x={thermal_x}:y={thermal_y}:"
            "eof_action=pass:shortest=1[thermal_filled];"
        )
    else:
        feather = int(boundary_feather)
        padded_width = thermal_width + feather * 2
        padded_height = height + feather * 2
        thermal_filter = (
            f"{scaled_thermal},"
            f"pad={padded_width}:{padded_height}:{feather}:{feather}:"
            "color=black@0[thermal_padded];"
            "[thermal_padded]split[thermal_rgba][thermal_alpha_src];"
            f"[thermal_alpha_src]alphaextract,boxblur=luma_radius={feather}:"
            "luma_power=1[softmask];"
            "[thermal_rgba]format=rgb24[thermal_rgb];"
            "[thermal_rgb][softmask]alphamerge[thermal_feathered];"
            f"[fill][thermal_feathered]overlay=x={thermal_x - feather}:"
            f"y={thermal_y - feather}:eof_action=pass:shortest=1[thermal_filled];"
        )

    return (
        "[0:v]format=rgba,split=2[rgb][fill_source];"
        f"[fill_source]drawbox=x=0:y=0:w=iw:h=ih:"
        f"color={fill_color}:t=fill[fill];"
        f"{thermal_filter}"
        "[thermal_filled]colorchannelmixer=aa=0.8[thermal_layer];"
        "[rgb][thermal_layer]overlay=x=0:y=0:"
        "eof_action=pass:shortest=1,format=yuv420p[out]"
    )


def merge_videos(
    rgb_path: Path,
    thermal_path: Path,
    output_path: Path,
    d: int,
    overwrite: bool,
    max_frames: int | None = None,
    thermal_width: int = 600,
    dy: int = 0,
    thermal_start_s: float = 0.0,
    boundary_feather: int = 15,
    thermal_border_crop: int = 10,
) -> None:
    rgb_info = probe_video(rgb_path)
    thermal_info = probe_video(thermal_path)

    if rgb_info["frame_rate"] != thermal_info["frame_rate"]:
        raise ValueError(
            "两段视频帧率必须相同，"
            f"当前 RGB={rgb_info['frame_rate']}，"
            f"热成像={thermal_info['frame_rate']}。"
        )
    if thermal_start_s < 0:
        raise ValueError(f"thermal_start_s 不能为负数，当前值为 {thermal_start_s}。")
    if thermal_start_s >= float(thermal_info["duration"]):
        raise ValueError(
            f"thermal_start_s={thermal_start_s:.6f} 已超出热成像视频时长 "
            f"{float(thermal_info['duration']):.6f}：{thermal_path}"
        )

    width = int(rgb_info["width"])
    height = int(rgb_info["height"])
    if thermal_width <= 0:
        raise ValueError(
            f"--thermal-width 必须是正整数，当前值为 {thermal_width}。"
        )
    if boundary_feather < 0:
        raise ValueError(
            f"--boundary-feather 必须大于等于 0，当前值为 {boundary_feather}。"
        )
    if thermal_border_crop < 0:
        raise ValueError(
            f"--thermal-border-crop 必须大于等于 0，当前值为 {thermal_border_crop}。"
        )
    if thermal_border_crop * 2 >= min(thermal_width, height):
        raise ValueError(
            f"--thermal-border-crop={thermal_border_crop} 对热成像尺寸 "
            f"{thermal_width}x{height} 过大。"
        )
    if not -width <= d <= thermal_width:
        raise ValueError(
            f"d 必须位于 [-{width}, {thermal_width}]，当前值为 {d}；"
            "超出此范围后热成像将完全位于输出画面之外。"
        )
    if not -height <= dy <= height:
        raise ValueError(
            f"dy 必须位于 [-{height}, {height}]，当前值为 {dy}；"
            "超出此范围后热成像将完全位于输出画面之外。"
        )
    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"--max-frames 必须是正整数，当前值为 {max_frames}。")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"输出文件已经存在：{output_path}。如需覆盖，请添加 --overwrite。"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    modal_color = get_modal_color(thermal_path, thermal_start_s, thermal_border_crop)

    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(rgb_path),
    ]
    if thermal_start_s > 0:
        command.extend(["-ss", f"{thermal_start_s:.6f}"])
    command.extend([
        "-i",
        str(thermal_path),
        "-filter_complex",
        make_filter(
            height,
            d,
            dy,
            thermal_width,
            modal_color,
            boundary_feather,
            thermal_border_crop,
        ),
        "-map",
        "[out]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-shortest",
    ])
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    command.append(str(output_path))

    print(f"RGB 视频：  {rgb_path}")
    print(f"热成像视频：{thermal_path}")
    print(f"输出视频：  {output_path}")
    print(f"热成像宽度：{thermal_width}")
    print(f"热成像起点：{thermal_start_s:.6f}s")
    print(f"填充众数颜色：RGB{modal_color}")
    print(f"边界羽化半径：{boundary_feather}px")
    print(f"热成像外圈裁剪：{thermal_border_crop}px")
    print(f"输出尺寸：  {width}x{height}（d={d}, dy={dy}）")
    print(f"处理帧数：  {max_frames if max_frames is not None else '全部'}")
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()

    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"未找到 {executable}，请先安装 FFmpeg。")

    dataset_path = args.dataset_path.expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"找不到数据集目录：{dataset_path}")

    video_pairs = iter_video_pairs(
        dataset_path,
        args.rgb_feature,
        args.thermal_feature,
        args.output_feature,
    )
    print(f"数据集：{dataset_path}")
    print(f"RGB 特征：{args.rgb_feature}")
    print(f"thermal 特征：{args.thermal_feature}")
    print(f"输出 mixed 特征：{args.output_feature}")
    print(f"待生成视频数：{len(video_pairs)}")
    print()

    for index, (rgb_path, thermal_path, output_path, thermal_start_s) in enumerate(video_pairs, start=1):
        print(f"[{index}/{len(video_pairs)}]")
        merge_videos(
            rgb_path,
            thermal_path,
            output_path,
            args.d,
            args.overwrite,
            args.max_frames,
            args.thermal_width,
            args.dy,
            thermal_start_s,
            args.boundary_feather,
            args.thermal_border_crop,
        )
        print()


if __name__ == "__main__":
    main()
