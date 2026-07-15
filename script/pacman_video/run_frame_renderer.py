#!/usr/bin/env python3
"""把当前 frame render table 绘制为单人或双人 JPG 图片帧。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.pacman_video.frame_renderer import (  # noqa: E402
    BAR_TYPE_CHOICES,
    PacmanRenderer,
    iter_output_paths,
    load_render_rows,
    sanitize_name,
)


def normalize_session_name(value: str) -> str:
    """规范 session 参数并拒绝嵌入路径的值。

    输入语义：值可以带可选的 ``.pkl`` 后缀。
    输出语义：返回不带后缀的 session stem。
    关键约束：task 和 session 必须分开传入，避免输出目录出现意外层级。
    """

    raw = str(value).strip()
    if Path(raw).name != raw:
        raise argparse.ArgumentTypeError(f"session 不能包含路径：{value!r}")
    return raw[:-4] if raw.endswith(".pkl") else raw


def parse_args() -> argparse.Namespace:
    """解析逐帧图片渲染参数。

    输入语义：调用方指定 render table 的 task/session、可选 trial 和显示模式。
    输出语义：返回图片渲染所需的完整参数对象。
    关键约束：图片输出路径包含显示模式，none/strategy/grammar 不会互相覆盖。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="任务目录，例如 comp 或 coop。")
    parser.add_argument("--session", required=True, type=normalize_session_name, help="session 文件名或 stem。")
    parser.add_argument("--render-table-root", type=Path, default=PROJECT_ROOT / "data/pacman_video/frame_render_data")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data/pacman_video/frame_images")
    parser.add_argument("--trial", default=None, help="可选：只渲染指定 DayTrial。")
    parser.add_argument("--start", type=int, default=0, help="在筛选 trial 后跳过前 N 个原始帧。")
    parser.add_argument("--max-frames", type=int, default=None, help="可选：最多渲染 N 个原始帧。")
    parser.add_argument("--aa", type=int, default=3, help="抗锯齿超采样倍数；1 表示关闭。")
    parser.add_argument(
        "--display-mode",
        choices=sorted(BAR_TYPE_CHOICES),
        default="strategy",
        help="none=不显示标注；strategy=显示 P1/P2 策略；grammar=保留的 grammar 模式。",
    )
    return parser.parse_args()


def validate_display_columns(rows, display_mode: str) -> None:
    """检查所选显示模式需要的标注字段。

    输入语义：``rows`` 是筛选后的 render table，``display_mode`` 是三种显示模式之一。
    输出语义：验证成功时无返回；缺少字段时抛出 ValueError。
    关键约束：strategy 模式对存在的每位玩家分别校验，不能让 P2 画面静默缺少策略。
    """

    if display_mode == "none":
        return
    if display_mode == "grammar":
        missing = sorted({"gram", "gram_num"} - set(rows.columns))
        if missing:
            raise ValueError(f"grammar 模式缺少字段：{missing}；当前 11 数据尚未接入逐帧时间轴。")
        return

    players = ["p1"]
    if all(column in rows.columns for column in ("p2_ppX", "p2_ppY", "p2_pDir", "p2_pFrame")):
        players.append("p2")
    missing = [f"{player}_display_strategy" for player in players if f"{player}_display_strategy" not in rows.columns]
    if missing:
        raise ValueError(f"strategy 模式缺少字段：{missing}；请先用 strategy 模式生成 render table。")


def clean_selected_output_dirs(rows, output_dir: Path) -> None:
    """删除本轮 trial 目录中遗留的旧 JPG。

    输入语义：``rows`` 决定本轮会写入哪些 DayTrial，``output_dir`` 已包含模式/task/session。
    输出语义：只删除这些目标 trial 目录中的 JPG 文件。
    关键约束：不会删除其它模式、session、trial 或非 JPG 文件。
    """

    for trial in rows["DayTrial"].drop_duplicates().astype(str):
        trial_dir = output_dir / sanitize_name(trial)
        if not trial_dir.is_dir():
            continue
        for old_frame in trial_dir.glob("*.jpg"):
            old_frame.unlink()


def main() -> None:
    """读取 render table，渲染图片并打印输出摘要。"""

    args = parse_args()
    subject = f"{args.task}/{args.session}"
    rows = load_render_rows(
        render_table_dir=args.render_table_root,
        subject=subject,
        trial=args.trial,
        start=args.start,
        max_frames=args.max_frames,
    )
    if rows.empty:
        raise SystemExit("没有可渲染的帧。")
    validate_display_columns(rows, args.display_mode)

    output_dir = args.output_root / args.display_mode / args.task / args.session
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_selected_output_dirs(rows, output_dir)
    renderer = PacmanRenderer(str(rows.iloc[0]["Map"]), aa=max(1, args.aa), bar_type=args.display_mode)

    total = len(rows)
    for index, row, path in iter_output_paths(rows, output_dir, subject=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = renderer.render(row)
        frame.save(path, quality=95, subsampling=0)
        if (index + 1) % 100 == 0 or index == 0 or index + 1 == total:
            print(f"rendered {index + 1}/{total}: {path}")

    print("frame 图片渲染完成")
    print(f"模式：{args.display_mode}")
    print(f"帧数：{total}")
    print(f"输出目录：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
