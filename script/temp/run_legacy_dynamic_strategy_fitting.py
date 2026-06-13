#!/usr/bin/env python3
"""运行旧版动态策略拟合的临时流水线入口。

该脚本复用当前仓库已有的动态策略拟合验证工具，只把输出目录固定到本次深度
验证要求的 ``data_temp`` 流水线目录。旧版源码会被临时补丁为“所有段落拟合”并
同步 GA 参数和随机种子；补丁文件在运行结束后清理。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from script.dynamic_strategy_fitting.validate_dynamic_strategy_fitting import (  # noqa: E402
    TEMP_ROOT,
    run_legacy_outputs,
    write_temp_legacy_files,
)


def parse_args() -> argparse.Namespace:
    """解析旧版动态策略拟合临时入口参数。

    输入语义：所有输入、输出、旧项目路径和拟合参数均由命令行显式给出。
    输出语义：返回可传给既有验证辅助函数的参数对象。
    关键约束：本脚本不写入正式模块目录，只写 output-dir、report 和短暂临时代码。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adjacent-map", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--legacy-script",
        type=Path,
        default=Path("/home/zzh/project/Pacman/Language-of-Problem-Solving/BasicStrategy/FittingWeightHuman.py"),
    )
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=Path("/home/zzh/project/Pacman/Language-of-Problem-Solving"),
    )
    parser.add_argument(
        "--legacy-cwd",
        type=Path,
        default=Path("/home/zzh/project/Pacman/Language-of-Problem-Solving/BasicStrategy"),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--segment-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--stay-length", type=int, default=6)
    parser.add_argument("--ga-population-size", type=int, default=100)
    parser.add_argument("--ga-iterations", type=int, default=500)
    parser.add_argument("--ga-mutation-probability", type=float, default=0.01)
    parser.add_argument("--ga-precision", type=float, default=1e-3)
    parser.add_argument("--weight-penalty", type=float, default=0.1)
    parser.add_argument("--vague-threshold", type=float, default=0.51)
    return parser.parse_args()


def main() -> None:
    """生成临时旧脚本、运行旧版动态策略拟合并写出阶段报告。"""

    args = parse_args()
    # 旧项目工具模块导入时会修改当前工作目录；所有路径先绝对化，避免子进程中找不到输入。
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.adjacent_map = args.adjacent_map.resolve()
    args.report = args.report.resolve()
    args.legacy_script = args.legacy_script.resolve()
    args.legacy_root = args.legacy_root.resolve()
    args.legacy_cwd = args.legacy_cwd.resolve()
    input_paths = sorted(args.input_dir.glob("*.pkl"))
    if not input_paths:
        raise FileNotFoundError(f"旧版动态策略拟合输入目录中没有 pkl：{args.input_dir}")

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    run_dir = args.report.parent / "04_legacy_dynamic_strategy_fitting_logs"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        legacy_script_path, legacy_worker_path = write_temp_legacy_files(args)
        task_summaries = run_legacy_outputs(
            validation_input_dir=args.input_dir,
            legacy_output_dir=args.output_dir,
            run_dir=run_dir,
            args=args,
            legacy_script_path=legacy_script_path,
            legacy_worker_path=legacy_worker_path,
        )
    finally:
        # 旧脚本补丁只用于本轮验证，结束后必须清理。
        if TEMP_ROOT.exists():
            shutil.rmtree(TEMP_ROOT)

    output_files = sorted(args.output_dir.glob("*.pkl"))
    total_rows = 0
    for output_path in output_files:
        # 只在报告阶段读取行数，方便确认旧流程是否完整覆盖所有输入。
        import pandas as pd

        total_rows += int(pd.read_pickle(output_path).shape[0])

    report = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "file_count": len(output_files),
        "total_rows": total_rows,
        "workers": args.workers,
        "segment_workers": args.segment_workers,
        "seed": args.seed,
        "tasks": task_summaries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("file_count", "total_rows", "workers", "segment_workers")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
