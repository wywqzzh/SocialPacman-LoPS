#!/usr/bin/env python3
"""运行人类 fMRI 动态策略权重拟合。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.dynamic_strategy_fitting import (  # noqa: E402
    DEFAULT_AGENTS,
    DynamicStrategyFittingConfig,
    process_dynamic_strategy_directory,
)


def parse_args() -> argparse.Namespace:
    """解析动态策略拟合命令行参数。

    输入语义：允许覆盖集中 utility 输入目录、输出目录、邻接表、随机种子、并行数和 GA 参数。
    输出语义：返回可直接构造配置并驱动目录批处理的参数对象。
    关键约束：默认路径只指向当前 LoPS 仓库 data 目录，不依赖旧项目路径。
    """

    data_root = PROJECT_ROOT / "pipeline_data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=data_root / "calculate_utility" / "utility_data",
        help="calculate_utility 输出的拟合输入目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_root / "dynamic_strategy_fitting" / "weight_data",
        help="动态拟合 WeightData 输出目录。",
    )
    parser.add_argument(
        "--adjacent-map",
        type=Path,
        default=data_root / "constant_data" / "adjacent_map_fmri.csv",
        help="fMRI 邻接表 CSV。",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="文件级并行进程数。")
    parser.add_argument("--segment-workers", type=int, default=1, help="单文件内部段落级并行进程数。")
    parser.add_argument(
        "--use-segment-seed",
        action="store_true",
        help="为每个段落使用 seed+file_index+segment_index 的固定随机种子。",
    )
    parser.add_argument("--seed", type=int, default=20260610, help="随机种子；每个文件会加上排序序号。")
    parser.add_argument("--stay-length", type=int, default=6, help="判定 stay 段的最小连续长度。")
    parser.add_argument("--ga-population-size", type=int, default=100, help="GA 种群大小。")
    parser.add_argument("--ga-iterations", type=int, default=500, help="GA 迭代次数。")
    parser.add_argument("--ga-mutation-probability", type=float, default=0.01, help="GA 变异概率。")
    parser.add_argument("--ga-precision", type=float, default=1e-3, help="GA 权重精度。")
    parser.add_argument("--weight-penalty", type=float, default=0.1, help="权重 L1 惩罚系数。")
    parser.add_argument("--vague-threshold", type=float, default=0.51, help="单策略正确率低于该阈值时标记 vague。")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DynamicStrategyFittingConfig:
    """把命令行参数转换成正式模块配置对象。

    输入语义：args 来自 parse_args。
    输出语义：返回 DynamicStrategyFittingConfig。
    关键约束：正式输出只包含 two-ghost 数据需要的 7 个策略；模块内部会按需补齐
    临时兼容维度，以复现旧随机优化路径。
    """

    return DynamicStrategyFittingConfig(
        agents=DEFAULT_AGENTS,
        stay_length=args.stay_length,
        ga_population_size=args.ga_population_size,
        ga_iterations=args.ga_iterations,
        ga_mutation_probability=args.ga_mutation_probability,
        ga_precision=args.ga_precision,
        weight_penalty=args.weight_penalty,
        vague_accuracy_threshold=args.vague_threshold,
        random_seed=args.seed,
        segment_workers=args.segment_workers,
        use_segment_seed=args.use_segment_seed or args.segment_workers > 1,
    )


def main() -> None:
    """命令行入口：批量运行动态策略权重拟合并打印摘要。"""

    args = parse_args()
    config = build_config(args)
    summaries = process_dynamic_strategy_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        adjacent_map_path=args.adjacent_map,
        config=config,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "processed_files": len(summaries),
                "total_rows": sum(item["rows"] for item in summaries),
                "output_dir": str(args.output_dir),
                "workers": args.workers,
                "config": asdict(config),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
