#!/usr/bin/env python3
"""把当前 frame JPG 图片序列批量合成为 MP4 视频。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.pacman_video.video_renderer import (  # noqa: E402
    VideoBuildError,
    build_game_video,
    find_ffmpeg,
    find_game_dirs,
    format_rate,
)


def normalize_session_name(value: str) -> str:
    """把 session 参数规范成不带后缀的单层目录名。

    输入语义：允许传入 stem 或 ``.pkl`` 文件名。
    输出语义：返回用于图片和视频嵌套目录的 session 名。
    关键约束：拒绝路径分隔符，避免绕过独立的 task 参数。
    """

    raw = str(value).strip()
    if Path(raw).name != raw:
        raise argparse.ArgumentTypeError(f"session 不能包含路径：{value!r}")
    return raw[:-4] if raw.endswith(".pkl") else raw


def parse_args() -> argparse.Namespace:
    """解析 frame 视频合成参数。

    输入语义：调用方指定显示模式、task、session 和编码参数。
    输出语义：返回可直接驱动 ffmpeg 合成流程的参数对象。
    关键约束：输入输出都包含显示模式，三种视频不会互相覆盖。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--display-mode", choices=["none", "strategy", "grammar"], default="strategy")
    parser.add_argument("--task", required=True, help="任务目录，例如 comp 或 coop。")
    parser.add_argument("--session", required=True, type=normalize_session_name, help="session 文件名或 stem。")
    parser.add_argument("--frame-root", type=Path, default=PROJECT_ROOT / "data/pacman_video/frame_images")
    parser.add_argument("--video-root", type=Path, default=PROJECT_ROOT / "data/pacman_video/frame_video")
    parser.add_argument("--fps", type=float, default=60.0, help="输出视频帧率，默认 60（原 30 FPS 的两倍）。")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument(
        "--preset",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        default="medium",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """校验视频帧率和 H.264 质量参数。

    输入语义：``args`` 来自 :func:`parse_args`。
    输出语义：参数合法时无返回；非法时抛出 VideoBuildError。
    关键约束：在启动任何 ffmpeg 子进程前完成校验，避免生成部分输出。
    """

    if args.fps <= 0:
        raise VideoBuildError("--fps 必须大于 0。")
    if not 0 <= args.crf <= 51:
        raise VideoBuildError("--crf 必须在 0 到 51 之间。")


def main() -> None:
    """把当前模式和 session 下的每个 DayTrial 图片目录分别合成为 MP4。"""

    args = parse_args()
    validate_args(args)
    ffmpeg = find_ffmpeg()
    relative_subject = f"{args.display_mode}/{args.task}/{args.session}"
    game_dirs = find_game_dirs(relative_subject, args.frame_root)
    output_dir = args.video_root / args.display_mode / args.task / args.session
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"模式：{args.display_mode}")
    print(f"task/session：{args.task}/{args.session}")
    print(f"game 数量：{len(game_dirs)}")
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
    print(f"输出目录：{output_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except VideoBuildError as exc:
        raise SystemExit(str(exc)) from exc
