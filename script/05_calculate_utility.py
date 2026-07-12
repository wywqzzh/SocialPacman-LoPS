#!/usr/bin/env python3
"""运行 Social Pacman 集中 utility 计算阶段。"""

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

from LoPS.calculate_utility import (  # noqa: E402
    CalculateUtilityConfig,
    load_calculate_utility_maps,
    process_calculate_utility_directory,
    process_calculate_utility_file,
)
from LoPS.hierarchical_utility import UtilityConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析集中 utility 计算阶段的命令行参数。

    输入语义：调用方可以覆盖输入目录、输出目录、常量目录、并行数和 raw Q 策略参数。
    输出语义：返回可直接构造配置并驱动目录处理的参数对象。
    关键约束：默认路径指向当前仓库的 data 主流程目录，不依赖旧项目路径。
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
        default=data_root / "05_cluster_global_utility_data",
        help="集中 utility 输出目录。",
    )
    parser.add_argument(
        "--constant-dir",
        type=Path,
        default=data_root / "constant_data",
        help="包含 map_constants.pkl 的常量目录。",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="文件级并行进程数。")
    parser.add_argument("--randomness-coeff", type=float, default=0.0, help="Q 随机扰动系数。")
    parser.add_argument("--laziness-coeff", type=float, default=0.0, help="沿用上一方向的惰性系数。")
    parser.add_argument("--global-depth", type=int, default=15, help="Global 策略深度参数。")
    parser.add_argument(
        "--global-ignore-depth",
        type=int,
        default=10,
        help="旧 Global 区域搜索的近距离 bean 忽略深度；不限制 cluster Global。",
    )
    parser.add_argument("--global-cluster-radius", type=int, default=60, help="Cluster Global 可考虑的最远资源团距离。")
    parser.add_argument("--global-cluster-distance-threshold", type=int, default=3, help="Cluster Global 中资源点聚类的地图距离阈值。")
    parser.add_argument("--local-depth", type=int, default=10, help="Local 路径树深度。")
    parser.add_argument(
        "--evade-depth",
        type=int,
        default=6,
        help="Evade 路径树深度；默认只把 6 步内的正常 ghost 视为即时威胁。",
    )
    parser.add_argument(
        "--approach-depth",
        type=int,
        default=20,
        help="Approach 路径树深度；默认允许20步范围内的远程追鬼行为提供方向证据。",
    )
    parser.add_argument("--energizer-depth", type=int, default=10, help="Energizer 路径树深度。")
    parser.add_argument("--no-energizer-depth", type=int, default=8, help="NoEnergizer 路径树深度。")
    parser.add_argument(
        "--single-file",
        type=Path,
        default=None,
        help="可选：只处理某个 corrected tile pkl。相对路径会基于 --input-dir 解析。",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> CalculateUtilityConfig:
    """根据命令行参数构造集中 utility 配置。

    输入语义：args 来自 ``parse_args``。
    输出语义：返回 ``CalculateUtilityConfig``。
    关键约束：默认随机和惰性系数为 0，保证 utility 输出可复现。
    """

    utility_config = UtilityConfig(
        randomness_coeff=args.randomness_coeff,
        laziness_coeff=args.laziness_coeff,
        global_depth=args.global_depth,
        global_ignore_depth=args.global_ignore_depth,
        local_depth=args.local_depth,
        evade_depth=args.evade_depth,
        approach_depth=args.approach_depth,
        energizer_depth=args.energizer_depth,
        no_energizer_depth=args.no_energizer_depth,
    )
    return CalculateUtilityConfig(
        utility_config=utility_config,
        global_cluster_radius=args.global_cluster_radius,
        global_cluster_distance_threshold=args.global_cluster_distance_threshold,
    )


def resolve_single_file(input_dir: Path, value: Path) -> Path:
    """解析单文件输入路径。

    输入语义：value 可以是绝对路径、相对当前工作目录路径，或相对 input_dir 的路径。
    输出语义：返回存在的输入 pickle 路径。
    关键约束：不做模糊匹配；找不到时直接报错，避免把同名文件误跑到错误目录。
    """

    candidates = [value]
    if not value.is_absolute():
        candidates.append(input_dir / value)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"找不到 single-file：{value}")


def main() -> None:
    """命令行入口：批量生成集中 utility 数据并打印 JSON 摘要。"""

    args = parse_args()
    map_data, adjacent_map = load_calculate_utility_maps(args.constant_dir)
    config = build_config(args)
    if args.single_file is not None:
        input_file = resolve_single_file(args.input_dir, args.single_file)
        output_file = args.output_dir / input_file.relative_to(args.input_dir)
        summary = process_calculate_utility_file(
            input_file,
            output_file,
            map_data=map_data,
            adjacent_map=adjacent_map,
            config=config,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    summaries = process_calculate_utility_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        map_data=map_data,
        adjacent_map=adjacent_map,
        config=config,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "processed_files": len(summaries),
                "total_input_rows": sum(item["input_rows"] for item in summaries),
                "total_output_rows": sum(item["output_rows"] for item in summaries),
                "total_changed_cells": sum(item["changed_cells"] for item in summaries),
                "players": summarize_players(summaries),
                "output_dir": str(args.output_dir),
                "workers": args.workers,
                "global_cluster_radius": args.global_cluster_radius,
                "global_cluster_distance_threshold": args.global_cluster_distance_threshold,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def summarize_players(summaries: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    """汇总每个玩家在 05 阶段的计算行数。

    输入语义：summaries 来自 ``process_calculate_utility_directory``。
    输出语义：返回以玩家前缀为键的总行数、计算行数、跳过行数和 Q 修正单元数。
    关键约束：该摘要只用于命令行日志，不影响任何保存数据。
    """

    result: dict[str, dict[str, int]] = {}
    for summary in summaries:
        players = summary.get("players", {})
        if not isinstance(players, dict):
            continue
        for player, player_summary in players.items():
            if not isinstance(player_summary, dict):
                continue
            accumulator = result.setdefault(
                str(player),
                {
                    "input_rows": 0,
                    "computed_rows": 0,
                    "skipped_rows": 0,
                    "changed_cells": 0,
                },
            )
            for key in accumulator:
                accumulator[key] += int(player_summary.get(key, 0))
    return result


if __name__ == "__main__":
    main()
