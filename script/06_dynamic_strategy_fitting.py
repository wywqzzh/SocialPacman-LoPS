#!/usr/bin/env python3
"""运行 Social Pacman 动态策略权重拟合。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

    输入语义：允许覆盖集中 utility 输入目录、输出目录、随机种子、并行数和 GA 参数。
    输出语义：返回可直接构造配置并驱动目录批处理的参数对象。
    关键约束：默认路径只指向当前 LoPS 仓库 data 目录，不依赖旧项目路径。
    """

    data_root = PROJECT_ROOT / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=data_root / "05_utility_data",
        help="calculate_utility 输出的拟合输入目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_root / "06_weight_data",
        help="动态拟合 WeightData 输出目录。",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="文件级并行进程数。")
    parser.add_argument("--segment-workers", type=int, default=1, help="单文件内部段落级并行进程数。")
    parser.add_argument(
        "--use-segment-seed",
        action="store_true",
        help="为每个段落使用 seed+file_index+segment_index 的固定随机种子。",
    )
    parser.add_argument("--seed", type=int, default=20260610, help="随机种子；每个文件会加上排序序号。")
    parser.add_argument("--stay-length", type=int, default=4, help="判定 stay 段的最小连续长度。")
    parser.add_argument("--ga-population-size", type=int, default=100, help="GA 种群大小。")
    parser.add_argument("--ga-iterations", type=int, default=500, help="GA 迭代次数。")
    parser.add_argument("--ga-mutation-probability", type=float, default=0.01, help="GA 变异概率。")
    parser.add_argument("--ga-precision", type=float, default=1e-3, help="GA 权重精度。")
    parser.add_argument("--weight-penalty", type=float, default=0.1, help="权重 L1 惩罚系数。")
    parser.add_argument("--vague-threshold", type=float, default=0.51, help="单策略正确率低于该阈值时标记 vague。")
    parser.add_argument("--min-effective-action-count", type=int, default=4, help="普通段落最少有效动作数，低于该值直接标为 vague。")
    parser.add_argument(
        "--min-effective-action-ratio",
        type=float,
        default=0.5,
        help="普通段落有效动作占比下限，低于该值直接标为 vague。",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DynamicStrategyFittingConfig:
    """把命令行参数转换成正式模块配置对象。

    输入语义：args 来自 parse_args。
    输出语义：返回 DynamicStrategyFittingConfig。
    关键约束：拟合维度直接等于 agents 数量，后续新增策略时只需要扩展配置。
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
        min_effective_action_count=args.min_effective_action_count,
        min_effective_action_ratio=args.min_effective_action_ratio,
    )


def main() -> None:
    """命令行入口：批量运行动态策略权重拟合并打印摘要。"""

    args = parse_args()
    config = build_config(args)
    summaries = process_dynamic_strategy_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
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
