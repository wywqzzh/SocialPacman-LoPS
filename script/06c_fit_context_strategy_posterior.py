#!/usr/bin/env python3
"""运行 06c Context 潜在策略后验拟合。"""

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

from LoPS.context_strategy_posterior import (  # noqa: E402
    ContextStrategyPosteriorConfig,
    process_context_strategy_posterior_directory,
    process_context_strategy_posterior_file,
)


def parse_args() -> argparse.Namespace:
    """解析 06c 命令行参数。

    输入语义：允许覆盖输入输出、context 参数、beta 搜索、CV 和文件级并行。
    输出语义：返回可构造 ContextStrategyPosteriorConfig 的 argparse Namespace。
    关键约束：默认目录与 06b 分离；单文件调试不会误处理整个输入目录。
    """

    data_root = PROJECT_ROOT / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=data_root / "05_cluster_global_utility_data")
    parser.add_argument("--output-dir", type=Path, default=data_root / "06c_context_strategy_posterior_data")
    parser.add_argument(
        "--single-file",
        type=Path,
        default=None,
        help="可选：只处理一个输入 pkl；相对路径会基于 --input-dir 解析。",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="文件级并行数。")
    parser.add_argument("--stay-length", type=int, default=4)
    parser.add_argument(
        "--bean-event-suppression-window",
        type=int,
        default=3,
        help="强事件前后取消普通豆起止边界的 tile 窗口。",
    )
    parser.add_argument(
        "--ghost-stay-suppression-window",
        type=int,
        default=5,
        help="吃 ghost 事件前后取消长 stay 切段作用的 tile 窗口。",
    )
    parser.add_argument("--beta-min", type=float, default=0.05)
    parser.add_argument("--beta-max", type=float, default=20.0)
    parser.add_argument("--beta-grid-size", type=int, default=81)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--posterior-threshold", type=float, default=0.70)
    parser.add_argument(
        "--min-information-coverage",
        type=float,
        default=0.50,
        help="策略在一个 context 内能够区分合法方向的最小有效动作比例。",
    )
    parser.add_argument(
        "--information-epsilon",
        type=float,
        default=1e-12,
        help="判断归一化 Q 是否具有方向差异时使用的浮点容差。",
    )
    parser.add_argument(
        "--no-information-penalty",
        type=float,
        default=2.0,
        help="合法方向 Q 全相等时，在均匀动作 log-likelihood 上增加的固定损失。",
    )
    parser.add_argument("--random-seed", type=int, default=20260610)
    return parser.parse_args()


def resolve_single_file(input_dir: Path, value: Path) -> Path:
    """解析单文件参数为实际存在的输入路径。

    输入语义：value 可以是绝对路径、当前目录相对路径或 input_dir 相对路径。
    输出语义：返回第一个实际存在的文件。
    关键约束：不做模糊文件名搜索，避免处理错误被试。
    """

    candidates = [value]
    if not value.is_absolute():
        candidates.append(input_dir / value)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"找不到 single-file：{value}")


def build_config(args: argparse.Namespace) -> ContextStrategyPosteriorConfig:
    """把命令行参数转换成 06c 正式配置。

    输入语义：args 来自 parse_args。
    输出语义：返回不可变 ContextStrategyPosteriorConfig。
    关键约束：策略顺序使用模块默认七策略，不由命令行临时改变。
    """

    return ContextStrategyPosteriorConfig(
        stay_length=args.stay_length,
        bean_event_suppression_window=args.bean_event_suppression_window,
        ghost_stay_suppression_window=args.ghost_stay_suppression_window,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        beta_grid_size=args.beta_grid_size,
        cv_folds=args.cv_folds,
        posterior_threshold=args.posterior_threshold,
        min_information_coverage=args.min_information_coverage,
        information_epsilon=args.information_epsilon,
        no_information_penalty=args.no_information_penalty,
        random_seed=args.random_seed,
    )


def main() -> None:
    """命令行入口：运行单文件或嵌套目录 06c 并打印 JSON 摘要。"""

    args = parse_args()
    config = build_config(args)
    if args.single_file is not None:
        input_file = resolve_single_file(args.input_dir, args.single_file)
        try:
            relative_path = input_file.relative_to(args.input_dir)
        except ValueError:
            # 绝对路径位于 input-dir 外时只保留文件名，避免把绝对目录复制到输出层级。
            relative_path = Path(input_file.name)
        summary = process_context_strategy_posterior_file(
            input_file,
            args.output_dir / relative_path,
            config,
            file_index=0,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    summaries = process_context_strategy_posterior_directory(
        args.input_dir,
        args.output_dir,
        config,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "processed_files": len(summaries),
                "output_dir": str(args.output_dir),
                "workers": args.workers,
                "files": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
