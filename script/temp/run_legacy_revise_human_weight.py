#!/usr/bin/env python3
"""运行旧版 reviseHuman 权重修正逻辑的临时入口。

该脚本只用于深度验证：从旧项目导入 ``reviseMain``，对本轮旧链路的
WeightData 目录逐文件运行旧规则，并把输出写入 ``data_temp``。旧函数中
``predict_dir`` 和 ``revise_is_correct`` 可能受并列最大值随机选择影响，后续比较会
按项目既有规则把它们视作诊断列。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd


STRATEGY_NUMBER = {
    "global": 0,
    "local": 1,
    "evade_blinky": 2,
    "evade_clyde": 3,
    "evade_3": 4,
    "evade_4": 5,
    "approach": 6,
    "energizer": 7,
    "no_energizer": 8,
    "vague": 9,
    "stay": 10,
}
AGENTS = [
    "global",
    "local",
    "evade_blinky",
    "evade_clyde",
    "evade_ghost3",
    "evade_ghost4",
    "approach",
    "energizer",
    "no_energizer",
]
SUFFIX = "_Q_norm"
AGENT_Q_COLUMNS = [f"{agent}{SUFFIX}" for agent in AGENTS]


def run_one_file(input_path: Path, output_dir: Path, legacy_root: Path, scared_time: int) -> dict[str, Any]:
    """调用旧版 reviseMain 处理单个 WeightData 文件。

    输入语义：input_path 是旧链路第 4 步输出，output_dir 是第 5 步旧输出目录。
    输出语义：旧函数写出同名 pickle，本函数返回行数摘要。
    关键约束：旧项目路径只加入当前子进程的 sys.path，不写入正式模块。
    """

    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))
    from BasicStrategy.reviseWeight.utility import reviseMain

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(output_dir) + os.sep
    reviseMain(
        str(input_path),
        savePath=save_path,
        strategy_number=STRATEGY_NUMBER,
        agents=AGENTS,
        agents_list=AGENT_Q_COLUMNS,
        agent_num=len(AGENTS),
        scared_time=scared_time,
        suffix=SUFFIX,
    )
    output_path = output_dir / input_path.name
    return {"input_file": input_path.name, "rows": int(pd.read_pickle(output_path).shape[0])}


def parse_args() -> argparse.Namespace:
    """解析旧版 revise human weight 临时入口参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--scared-time", type=int, default=63)
    return parser.parse_args()


def main() -> None:
    """批量运行旧版 revise human weight 并写出验证报告。"""

    args = parse_args()
    args.legacy_root = args.legacy_root.resolve()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.report = args.report.resolve()

    input_paths = sorted(args.input_dir.glob("*.pkl"))
    if not input_paths:
        raise FileNotFoundError(f"旧版 revise human weight 输入目录中没有 pkl：{args.input_dir}")

    worker = partial(
        run_one_file,
        output_dir=args.output_dir,
        legacy_root=args.legacy_root,
        scared_time=args.scared_time,
    )
    process_count = max(1, min(args.processes, len(input_paths)))
    if process_count == 1:
        summaries = [worker(path) for path in input_paths]
    else:
        # 文件之间互不依赖，可直接进程并行。
        with mp.Pool(processes=process_count) as pool:
            summaries = pool.map(worker, input_paths)

    report = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "file_count": len(summaries),
        "total_rows": int(sum(item["rows"] for item in summaries)),
        "files": summaries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("file_count", "total_rows")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
