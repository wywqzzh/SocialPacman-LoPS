#!/usr/bin/env python3
"""运行 Social Pacman 范围版 utility 计算阶段。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.range_utility import (  # noqa: E402
    RangeUtilityConfig,
    config_to_dict,
    load_range_map_data,
    process_range_utility_directory,
    summarize_players,
)


def parse_args() -> argparse.Namespace:
    """解析范围版 utility 计算阶段的命令行参数。

    输入语义：调用方可以覆盖输入输出目录、常量目录、并行数和范围版策略参数。
    输出语义：返回可直接构造配置并驱动目录处理的参数对象。
    关键约束：默认路径指向新的 `data/05_range_utility_data`，不会覆盖当前 05 输出。
    """

    data_root = PROJECT_ROOT / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=data_root / "04_corrected_tile_data",
        help="corrected tile 输入目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_root / "05_range_utility_data",
        help="范围版 utility 输出目录。",
    )
    parser.add_argument(
        "--constant-dir",
        type=Path,
        default=data_root / "constant_data",
        help="包含 map_constants.pkl 的常量目录。",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="文件级并行进程数。")
    parser.add_argument("--local-radius", type=int, default=10, help="Local 资源统计半径。")
    parser.add_argument("--global-radius", type=int, default=60, help="Global 远处 bean 统计半径。")
    parser.add_argument("--global-ignore-radius", type=int, default=10, help="Global 忽略近处 bean 的距离阈值。")
    parser.add_argument("--evade-radius", type=int, default=10, help="Evade 危险 ghost 惩罚半径。")
    parser.add_argument("--approach-radius", type=int, default=34, help="Approach 非死亡 ghost 奖励半径。")
    parser.add_argument("--energizer-radius", type=int, default=10, help="Energizer 能量豆奖励半径。")
    parser.add_argument("--no-energizer-radius", type=int, default=12, help="NoEnergizer 能量豆惩罚半径。")
    parser.add_argument("--local-decay", type=float, default=0.90, help="Local 距离衰减系数。")
    parser.add_argument("--global-decay", type=float, default=0.97, help="Global 距离衰减系数。")
    parser.add_argument("--evade-decay", type=float, default=0.80, help="Evade 距离衰减系数。")
    parser.add_argument("--approach-decay", type=float, default=0.95, help="Approach 距离衰减系数。")
    parser.add_argument("--energizer-decay", type=float, default=0.90, help="Energizer 距离衰减系数。")
    parser.add_argument("--no-energizer-decay", type=float, default=0.90, help="NoEnergizer 距离衰减系数。")
    parser.add_argument("--bean-reward", type=float, default=2.0, help="普通豆子基础奖励。")
    parser.add_argument("--energizer-reward", type=float, default=4.0, help="能量豆基础奖励。")
    parser.add_argument("--ghost-reward", type=float, default=8.0, help="Approach 非死亡 ghost 基础奖励。")
    parser.add_argument("--ghost-penalty", type=float, default=8.0, help="危险 ghost 基础惩罚。")
    parser.add_argument("--energizer-penalty", type=float, default=4.0, help="NoEnergizer 中能量豆基础惩罚。")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RangeUtilityConfig:
    """根据命令行参数构造范围版 utility 配置。

    输入语义：args 来自 `parse_args`。
    输出语义：返回 `RangeUtilityConfig`。
    关键约束：所有参数都有显式默认值，便于复现实验配置。
    """

    return RangeUtilityConfig(
        local_radius=args.local_radius,
        global_radius=args.global_radius,
        global_ignore_radius=args.global_ignore_radius,
        evade_radius=args.evade_radius,
        approach_radius=args.approach_radius,
        energizer_radius=args.energizer_radius,
        no_energizer_radius=args.no_energizer_radius,
        local_decay=args.local_decay,
        global_decay=args.global_decay,
        evade_decay=args.evade_decay,
        approach_decay=args.approach_decay,
        energizer_decay=args.energizer_decay,
        no_energizer_decay=args.no_energizer_decay,
        bean_reward=args.bean_reward,
        energizer_reward=args.energizer_reward,
        ghost_reward=args.ghost_reward,
        ghost_penalty=args.ghost_penalty,
        energizer_penalty=args.energizer_penalty,
    )


def main() -> None:
    """命令行入口：批量生成范围版 utility 数据并打印 JSON 摘要。"""

    args = parse_args()
    config = build_config(args)
    map_data = load_range_map_data(args.constant_dir)
    summaries = process_range_utility_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        map_data=map_data,
        config=config,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "processed_files": len(summaries),
                "total_input_rows": sum(int(item["input_rows"]) for item in summaries),
                "total_output_rows": sum(int(item["output_rows"]) for item in summaries),
                "players": summarize_players(summaries),
                "output_dir": str(args.output_dir),
                "workers": args.workers,
                "config": config_to_dict(config),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
