#!/usr/bin/env python3
"""从当前 02/07 主流程数据生成 frame 视频逐帧渲染表。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.pacman_video.render_table import DataProcessingError, build_current_render_table  # noqa: E402


def normalize_session_name(value: str) -> str:
    """把命令行 session 参数规范成不带 pickle 后缀的文件名。

    输入语义：``value`` 可以是 session stem 或 ``session.pkl``。
    输出语义：返回可安全拼接到数据根目录下的单层文件名。
    关键约束：session 不能包含路径分隔符；task 目录由独立参数提供。
    """

    raw = str(value).strip()
    if Path(raw).name != raw:
        raise argparse.ArgumentTypeError(f"session 不能包含路径：{value!r}")
    name = raw[:-4] if raw.endswith(".pkl") else raw
    if not name:
        raise argparse.ArgumentTypeError("session 不能为空。")
    return name


def parse_args() -> argparse.Namespace:
    """解析当前 frame render table 入口参数。

    输入语义：调用方指定 task、session 和显示模式，也可以覆盖 02、07 和输出根目录。
    输出语义：返回可直接交给 :func:`build_current_render_table` 的参数对象。
    关键约束：none 模式不要求 07 文件；strategy 模式必须存在同 task/session 的 07 文件。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="任务目录，例如 comp 或 coop。")
    parser.add_argument("--session", required=True, type=normalize_session_name, help="session 文件名或 stem。")
    parser.add_argument(
        "--display-mode",
        choices=["none", "strategy", "grammar"],
        default="strategy",
        help="none=仅游戏画面；strategy=显示 07 修正策略；grammar=保留接口，暂未接入 11。",
    )
    parser.add_argument("--frame-root", type=Path, default=PROJECT_ROOT / "data/02_frame_data")
    parser.add_argument("--strategy-root", type=Path, default=PROJECT_ROOT / "data/07_revised_strategy_data")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data/pacman_video/frame_render_data")
    return parser.parse_args()


def main() -> None:
    """生成指定 task/session 的逐帧渲染表并打印对齐摘要。"""

    args = parse_args()
    relative_path = Path(args.task) / f"{args.session}.pkl"
    frame_path = args.frame_root / relative_path
    strategy_path = args.strategy_root / relative_path if args.display_mode == "strategy" else None
    output_path = args.output_root / relative_path

    try:
        summary = build_current_render_table(
            frame_path,
            output_path,
            display_mode=args.display_mode,
            strategy_path=strategy_path,
        )
    except DataProcessingError as exc:
        raise SystemExit(f"frame render table 生成失败：{exc}") from exc

    print("frame render table 生成完成")
    print(f"模式：{summary.display_mode}")
    print(f"玩家数量：{summary.player_count}")
    print(f"逐帧行数：{summary.frame_rows}")
    print(f"trial 数量：{summary.trial_count}")
    print(f"策略缺失 trial：{summary.missing_strategy_trials}")
    print(f"策略缺失 frame：{summary.missing_strategy_frames}")
    print(f"输出：{summary.output_path}")


if __name__ == "__main__":
    main()
