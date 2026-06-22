#!/usr/bin/env python3
"""运行 frame_data 标准分析字段预处理。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.pacman_preprocess.frame_data_preprocess import preprocess_frame_data_directory  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输入语义：允许调用方覆盖 frame_data 输入目录、标准化输出目录和并行数。
    输出语义：返回 argparse Namespace。
    关键约束：默认路径指向当前仓库的 data 主流程目录。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data/02_frame_data",
        help="raw frame_data 输入目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/03_preprocessed_frame_data",
        help="标准化 frame_data 输出目录。",
    )
    parser.add_argument("--workers", type=int, default=min(34, os.cpu_count() or 1), help="并行进程数。")
    parser.add_argument("files", nargs="*", help="可选：只处理这些 frame_data 文件名。")
    return parser.parse_args()


def main() -> None:
    """命令行入口：批量生成标准化 frame_data 并打印摘要。"""

    args = parse_args()
    summaries = preprocess_frame_data_directory(
        args.input_dir,
        args.output_dir,
        files=args.files or None,
        workers=args.workers,
    )
    print("frame_data_preprocess 生成完成")
    print(f"文件数：{len(summaries)}")
    print(f"总行数：{sum(item['rows'] for item in summaries)}")
    print(f"输出目录：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
