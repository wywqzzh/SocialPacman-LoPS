#!/usr/bin/env python3
"""运行事件硬边界版 06 动态策略拟合。

本脚本读取 05 cluster global utility 输出，使用 ``dynamic_strategy_event_context``
中的新 context 划分方法拟合 p1/p2 策略权重，并保存为 07 可直接读取的
WeightData 表。默认输出到 ``data/06_cluster_global_event_context_weight_data``，
避免覆盖旧版 06 结果。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.dynamic_strategy_event_context import (
    process_dynamic_strategy_event_context_directory,
    process_dynamic_strategy_event_context_file,
)
from LoPS.dynamic_strategy_fitting import DEFAULT_AGENTS, DynamicStrategyFittingConfig


def parse_args() -> argparse.Namespace:
    """解析事件 context 版 06 的命令行参数。

    输出语义：返回输入、输出、并行和 GA 参数。
    关键约束：默认路径全部位于当前 LoPS 仓库 data 目录；``--single-file`` 用于
    只处理一个文件，避免调试时误跑全部数据。
    """

    data_root = PROJECT_ROOT / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=data_root / "05_cluster_global_utility_data")
    parser.add_argument("--output-dir", type=Path, default=data_root / "06_cluster_global_event_context_weight_data")
    parser.add_argument(
        "--single-file",
        type=Path,
        default=None,
        help="可选：只处理某个 05 utility pkl。相对路径会基于 --input-dir 解析。",
    )
    parser.add_argument("--workers", type=int, default=1, help="文件级并行进程数。")
    parser.add_argument("--segment-workers", type=int, default=min(8, os.cpu_count() or 1), help="单文件内部段落级并行进程数。")
    parser.add_argument("--stay-length", type=int, default=4, help="长 stay 硬边界长度阈值。")
    parser.add_argument("--ga-population-size", type=int, default=100)
    parser.add_argument("--ga-iterations", type=int, default=500)
    parser.add_argument("--ga-mutation-probability", type=float, default=0.01)
    parser.add_argument("--ga-precision", type=float, default=1e-3)
    parser.add_argument("--weight-penalty", type=float, default=0.1)
    parser.add_argument("--vague-accuracy-threshold", type=float, default=0.51)
    parser.add_argument("--random-seed", type=int, default=20260610)
    parser.add_argument("--no-segment-seed", action="store_true", help="关闭每段派生随机种子。")
    parser.add_argument(
        "--bean-event-suppression-window",
        type=int,
        default=3,
        help="强事件前后取消普通豆起止边界的 tile 窗口，默认 3。",
    )
    parser.add_argument(
        "--ghost-stay-suppression-window",
        type=int,
        default=5,
        help="吃 ghost 事件前后取消长 stay 切段作用的 tile 窗口，默认 5。",
    )
    parser.add_argument("--min-effective-action-count", type=int, default=4, help="普通段落最少有效动作数，低于该值直接标为 vague。")
    parser.add_argument(
        "--min-effective-action-ratio",
        type=float,
        default=0.5,
        help="普通段落有效动作占比下限，低于该值直接标为 vague。",
    )
    return parser.parse_args()


def resolve_single_file(input_dir: Path, value: Path) -> Path:
    """解析单文件输入路径。

    输入语义：value 可以是绝对路径、相对当前目录路径，或相对 input_dir 的路径。
    输出语义：返回存在的 pkl 路径。
    关键约束：不做模糊搜索；路径不存在时直接报错，避免误处理其它文件。
    """

    candidates = [value]
    if not value.is_absolute():
        candidates.append(input_dir / value)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"找不到 single-file：{value}")


def main() -> None:
    """命令行入口：执行事件 context 版 06 拟合。"""

    args = parse_args()
    config = DynamicStrategyFittingConfig(
        agents=DEFAULT_AGENTS,
        stay_length=args.stay_length,
        ga_population_size=args.ga_population_size,
        ga_iterations=args.ga_iterations,
        ga_mutation_probability=args.ga_mutation_probability,
        ga_precision=args.ga_precision,
        weight_penalty=args.weight_penalty,
        vague_accuracy_threshold=args.vague_accuracy_threshold,
        random_seed=args.random_seed,
        segment_workers=args.segment_workers,
        use_segment_seed=not args.no_segment_seed,
        bean_event_suppression_window=args.bean_event_suppression_window,
        ghost_stay_suppression_window=args.ghost_stay_suppression_window,
        min_effective_action_count=args.min_effective_action_count,
        min_effective_action_ratio=args.min_effective_action_ratio,
    )
    if args.single_file is not None:
        input_file = resolve_single_file(args.input_dir, args.single_file)
        output_file = args.output_dir / input_file.relative_to(args.input_dir)
        summary = process_dynamic_strategy_event_context_file(
            input_file,
            output_file,
            config,
            file_index=0,
        )
        print(summary)
        return

    summaries = process_dynamic_strategy_event_context_directory(
        args.input_dir,
        args.output_dir,
        config,
        workers=args.workers,
    )
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
