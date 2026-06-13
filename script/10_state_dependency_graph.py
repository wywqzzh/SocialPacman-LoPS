#!/usr/bin/env python3
"""运行 strategy_sequence 到状态依赖图的学习。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.state_dependency_graph import (  # noqa: E402
    DEFAULT_STATE_NAMES,
    process_state_dependency_graph_directory,
)


def parse_args() -> argparse.Namespace:
    """解析状态依赖图学习脚本的命令行参数。

    输入语义：调用方可以显式传入数据目录、状态列和算法先验参数。
    输出语义：返回可直接驱动批处理和验证流程的参数对象。
    关键约束：默认路径只存在于脚本层，正式模块不内置任何数据目录。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data/09_strategy_sequence",
        help="human_fmri_data_preprocess 生成的 strategy_sequence 输入目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/10_state_dependency_graph_data",
        help="状态依赖图输出目录。",
    )
    parser.add_argument(
        "--state-names",
        nargs="+",
        default=list(DEFAULT_STATE_NAMES),
        help="参与图学习的状态列名。",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet 先验总强度。")
    return parser.parse_args()


def main() -> None:
    """命令行入口：批量学习状态依赖图并打印生成摘要。"""

    args = parse_args()

    summaries = process_state_dependency_graph_directory(
        args.input_dir,
        args.output_dir,
        state_names=args.state_names,
        alpha=args.alpha,
    )
    print("state_dependency_graph 生成完成")
    print(f"subject 数量：{len(summaries)}")
    print(f"样本帧总数：{sum(item['sample_count'] for item in summaries)}")
    print(f"总无向边数：{sum(item['edge_count'] for item in summaries)}")
    print(f"输出目录：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
