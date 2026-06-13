#!/usr/bin/env python3
"""运行 Pacman render table 生成流程。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.pacman_video.render_table import DataProcessingError, find_subject_paths, process_subject


def parse_args() -> argparse.Namespace:
    """解析 render table 生成脚本参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", nargs="?", default="041122-403")
    parser.add_argument("--frame-data-dir", type=Path, default=PROJECT_ROOT / "data/02_frame_data")
    parser.add_argument("--grammar-dir", type=Path, default=PROJECT_ROOT / "data/pacman_video/grammar_data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/pacman_video/render_data")
    parser.add_argument("--ghost-count", choices=["2", "4", "all"], default="2")
    return parser.parse_args()


def main() -> None:
    """生成指定被试的 render table 并打印摘要。"""

    args = parse_args()
    gram_pkl = args.grammar_dir / f"{args.subject}-gram.pkl"
    try:
        paths = find_subject_paths(
            args.subject,
            project_root=PROJECT_ROOT,
            output_root=args.output_dir,
            frame_data_dir=args.frame_data_dir,
            gram_pkl=gram_pkl,
        )
        summary = process_subject(paths, ghost_count=args.ghost_count)
    except DataProcessingError as exc:
        raise SystemExit(f"render table 生成失败：{exc}") from exc

    print("render table 生成完成")
    print(f"被试：{summary.subject}")
    print(f"ghost trial 过滤：{summary.ghost_count_filter}")
    print(f"输入 trial 数量：{summary.input_trial_count}")
    print(f"保留 trial 数量：{summary.trial_count}")
    print(f"逐帧行数：{summary.frame_rows}")
    for name, path in summary.outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
