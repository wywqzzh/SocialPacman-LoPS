#!/usr/bin/env python3
"""把已渲染的 Pacman JPG 帧合成为 MP4 视频。

输入固定为：
    data/pacman_video/frame_images/{subject}/{game}/00001.jpg

输出固定为：
    data/pacman_video/video_data/{subject}/{game}.mp4

每个 game 子文件夹会生成一个独立 MP4。脚本使用 ffmpeg 编码视频，不再引入
额外 Python 视频写入依赖。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


FFMPEG_FALLBACK = Path("/usr/local/fsl/bin/ffmpeg")


class VideoBuildError(RuntimeError):
    """图片帧合成视频失败时抛出的明确异常。"""


def parse_args() -> argparse.Namespace:
    """解析视频合成脚本的干净 CLI。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", help="被试名前缀，例如 041122-403。")
    parser.add_argument("frame_root", type=Path, help="JPG 图片帧根目录。")
    parser.add_argument("video_root", type=Path, help="MP4 视频输出根目录。")
    parser.add_argument("--fps", type=float, default=30.0, help="输出视频帧率，默认 30。")
    parser.add_argument("--crf", type=int, default=18, help="H.264 质量参数，0-51，越小质量越高。默认 18。")
    parser.add_argument(
        "--preset",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        default="medium",
        help="ffmpeg libx264 编码速度/压缩率预设，默认 medium。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已经存在的 mp4。默认跳过已有视频。")
    return parser.parse_args()


def main() -> None:
    """命令行入口：遍历 subject 下每个 game 文件夹并生成 MP4。"""

    args = parse_args()
    validate_args(args)

    ffmpeg = find_ffmpeg()
    game_dirs = find_game_dirs(args.subject, args.frame_root)
    output_dir = args.video_root / args.subject
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"被试：{args.subject}")
    print(f"game 数量：{len(game_dirs)}")
    print(f"输出目录：{output_dir}")
    print(f"视频参数：fps={format_rate(args.fps)}, crf={args.crf}, preset={args.preset}")

    built = 0
    skipped = 0
    for game_dir in game_dirs:
        result = build_game_video(
            ffmpeg=ffmpeg,
            game_dir=game_dir,
            output_path=output_dir / f"{game_dir.name}.mp4",
            fps=args.fps,
            crf=args.crf,
            preset=args.preset,
            overwrite=args.overwrite,
        )
        if result["status"] == "built":
            built += 1
        else:
            skipped += 1
        print(f"{result['status']}: {result['game']} frames={result['frames']} output={result['output']}")

    print(f"完成：built={built}, skipped={skipped}")


def validate_args(args: argparse.Namespace) -> None:
    """检查视频参数范围，避免把非法参数传给 ffmpeg 后才失败。"""

    if args.fps <= 0:
        raise VideoBuildError("--fps 必须大于 0。")
    if not 0 <= args.crf <= 51:
        raise VideoBuildError("--crf 必须在 0 到 51 之间。")


def find_ffmpeg() -> str:
    """定位 ffmpeg 可执行文件。"""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    if FFMPEG_FALLBACK.exists():
        return str(FFMPEG_FALLBACK)
    raise VideoBuildError("找不到 ffmpeg；请确认 ffmpeg 在 PATH 中，或存在 /usr/local/fsl/bin/ffmpeg。")


def find_game_dirs(subject: str, frame_root: Path) -> list[Path]:
    """读取 subject 图片目录下包含 JPG 帧的所有 game 子文件夹。"""

    subject_dir = frame_root / subject
    if not subject_dir.exists():
        raise VideoBuildError(f"找不到图片帧目录：{subject_dir}")

    game_dirs = [
        path
        for path in subject_dir.iterdir()
        if path.is_dir() and any(iter_frame_paths(path))
    ]
    if not game_dirs:
        raise VideoBuildError(f"{subject_dir} 下没有包含 JPG 帧的 game 子文件夹。")
    return sorted(game_dirs, key=natural_key)


def build_game_video(
    *,
    ffmpeg: str,
    game_dir: Path,
    output_path: Path,
    fps: float,
    crf: int,
    preset: str,
    overwrite: bool,
) -> dict[str, object]:
    """把一个 game 文件夹下的 JPG 序列合成为一个 MP4。"""

    frame_paths = list(iter_frame_paths(game_dir))
    if not frame_paths:
        return {"status": "skipped", "game": game_dir.name, "frames": 0, "output": str(output_path)}
    if output_path.exists() and not overwrite:
        return {"status": "skipped", "game": game_dir.name, "frames": len(frame_paths), "output": str(output_path)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd, temporary_file = build_ffmpeg_command(
        ffmpeg=ffmpeg,
        game_dir=game_dir,
        frame_paths=frame_paths,
        output_path=output_path,
        fps=fps,
        crf=crf,
        preset=preset,
        overwrite=overwrite,
    )
    try:
        run_ffmpeg(cmd)
    finally:
        if temporary_file is not None:
            temporary_file.unlink(missing_ok=True)
    return {"status": "built", "game": game_dir.name, "frames": len(frame_paths), "output": str(output_path)}


def iter_frame_paths(game_dir: Path) -> Iterable[Path]:
    """按自然顺序返回一个 game 文件夹内的 JPG/JPEG 帧。"""

    paths = list(game_dir.glob("*.jpg")) + list(game_dir.glob("*.jpeg")) + list(game_dir.glob("*.JPG")) + list(game_dir.glob("*.JPEG"))
    yield from sorted(paths, key=natural_key)


def build_ffmpeg_command(
    *,
    ffmpeg: str,
    game_dir: Path,
    frame_paths: list[Path],
    output_path: Path,
    fps: float,
    crf: int,
    preset: str,
    overwrite: bool,
) -> tuple[list[str], Path | None]:
    """构造 ffmpeg 命令。

    标准渲染输出是连续的 ``00001.jpg``、``00002.jpg``。这种情况使用 image2
    pattern，速度最快；如果图片命名不连续，则自动退回 concat list，保证仍能
    按自然排序合成视频。
    """

    common_args = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
    ]
    encoding_args = [
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    sequence = numeric_sequence(frame_paths)
    if sequence is not None:
        start_number, _end_number = sequence
        return (
            common_args + [
                "-framerate",
                format_rate(fps),
                "-start_number",
                str(start_number),
                "-i",
                str(game_dir / "%05d.jpg"),
            ] + encoding_args,
            None,
        )

    concat_file = write_concat_file(frame_paths, fps)
    return (
        common_args + [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-r",
            format_rate(fps),
        ] + encoding_args,
        concat_file,
    )


def numeric_sequence(frame_paths: list[Path]) -> tuple[int, int] | None:
    """判断图片是否是连续的五位数字 JPG 序列。"""

    numbers: list[int] = []
    for path in frame_paths:
        if path.suffix.lower() != ".jpg" or not path.stem.isdigit() or len(path.stem) != 5:
            return None
        if path.name != f"{int(path.stem):05d}.jpg":
            return None
        numbers.append(int(path.stem))

    if not numbers:
        return None
    expected = list(range(numbers[0], numbers[-1] + 1))
    if numbers != expected:
        return None
    return numbers[0], numbers[-1]


def write_concat_file(frame_paths: list[Path], fps: float) -> Path:
    """为非标准文件名写 ffmpeg concat demuxer 列表。"""

    duration = 1.0 / fps
    handle = tempfile.NamedTemporaryFile("w", suffix=".ffconcat", encoding="utf-8", delete=False)
    with handle:
        handle.write("ffconcat version 1.0\n")
        for path in frame_paths:
            handle.write(f"file '{escape_ffconcat_path(path.resolve())}'\n")
            handle.write(f"duration {duration:.10f}\n")
        handle.write(f"file '{escape_ffconcat_path(frame_paths[-1].resolve())}'\n")
    return Path(handle.name)


def run_ffmpeg(cmd: list[str]) -> None:
    """执行 ffmpeg，并在失败时给出清晰错误。"""

    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise VideoBuildError(f"ffmpeg 失败：{message}") from exc


def escape_ffconcat_path(path: Path) -> str:
    """转义 concat 文件中的单引号。"""

    return str(path).replace("'", "'\\''")


def natural_key(path: Path) -> list[object]:
    """自然排序 key，使 2-... 排在 10-... 前面，00002 排在 00010 前面。"""

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def format_rate(value: float) -> str:
    """把 fps 浮点数格式化成 ffmpeg 友好的字符串。"""

    return f"{value:g}"


if __name__ == "__main__":
    try:
        main()
    except VideoBuildError as exc:
        raise SystemExit(str(exc)) from exc
