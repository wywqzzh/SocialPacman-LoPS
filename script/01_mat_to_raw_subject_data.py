#!/usr/bin/env python3
"""运行 Pacman raw_mat_data 到 raw_subject_data 的转换。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.pacman_preprocess.mat_to_raw_subject_data import convert_mat_root_to_raw_subject_data


def parse_args() -> argparse.Namespace:
    """解析 raw_subject_data 生成脚本参数。

    输入语义：默认路径指向当前 LoPS 仓库的 `data`，也允许调用方显式覆盖。
    输出语义：返回包含输入目录、输出目录、session 白名单和并行数的命令行参数。
    关键约束：默认路径只在脚本层存在，正式模块不内置数据目录。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data/00_raw_mat_data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/01_raw_subject_data")
    parser.add_argument("--tasks", nargs="*", default=None, help="可选：只处理这些任务目录，例如 comp coop。")
    parser.add_argument("--workers", type=int, default=34)
    parser.add_argument("sessions", nargs="*", help="可选：只处理这些 session。")
    return parser.parse_args()


def main() -> None:
    """执行 raw_subject_data 转换并打印摘要。"""

    args = parse_args()
    results = convert_mat_root_to_raw_subject_data(
        args.raw_root,
        output_dir=args.output_dir,
        selected_subjects=args.sessions or None,
        selected_tasks=args.tasks,
        workers=args.workers,
    )
    print("raw_subject_data 生成完成")
    print(f"session 数量：{len(results)}")
    print(f"trial 数量：{sum(item['trials'] for item in results)}")
    print(f"逐帧行数：{sum(item['rows'] for item in results)}")
    print(f"输出目录：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
