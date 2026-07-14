#!/usr/bin/env python3
"""运行 Pacman JPG 图片帧到 MP4 视频的合成。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.pacman_video.video_renderer import (
    VideoBuildError,
    build_game_video,
    find_ffmpeg,
    find_game_dirs,
    format_rate,
    validate_args,
)


def parse_args() -> argparse.Namespace:
    """解析视频合成脚本参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", nargs="?", default="041122-403")
    parser.add_argument("--frame-root", type=Path, default=PROJECT_ROOT / "data/pacman_video/frame_images")
    parser.add_argument("--video-root", type=Path, default=PROJECT_ROOT / "data/pacman_video/video_data")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument(
        "--preset",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        default="medium",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """把每个 game 子目录下的 JPG 序列合成为 MP4。"""

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


if __name__ == "__main__":
    try:
        main()
    except VideoBuildError as exc:
        raise SystemExit(str(exc)) from exc
