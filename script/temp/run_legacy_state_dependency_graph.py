#!/usr/bin/env python3
"""运行旧版状态依赖图学习逻辑的临时验证入口。

旧版入口 ``GrammarInduction/LearnStateHuman.py`` 固定读取旧项目 ``HumanData/seq`` 并写回
``HumanData/state``。本脚本只复用旧版 PC skeleton 算法，把输入输出路径改为本轮
深度验证目录，避免污染旧项目数据。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd


STATE_NAMES = ["IS1", "IS2", "PG1", "PG2", "PE", "BN5"]


def list_pickle_files(data_dir: Path) -> list[Path]:
    """列出目录中的 pkl 文件路径，并使用稳定顺序返回。"""

    return sorted(data_dir.glob("*.pkl"))


def load_legacy_module(legacy_root: Path) -> Any:
    """导入旧版 ``LearnStateHuman.py`` 模块。

    输入语义：legacy_root 是旧项目根目录。
    输出语义：返回包含旧版 ``PC`` 函数的模块对象。
    关键约束：旧模块依赖 ``PGM`` 包路径，因此导入前把旧项目根目录加入 ``sys.path``。
    """

    legacy_root = legacy_root.resolve()
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))

    module_path = legacy_root / "GrammarInduction" / "LearnStateHuman.py"
    spec = importlib.util.spec_from_file_location("legacy_learn_state_human", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入旧版 LearnStateHuman：{module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_one_file(module: Any, input_path: Path, output_path: Path) -> dict[str, Any]:
    """使用旧版 PC 函数处理一个 strategy sequence 文件。

    输入语义：input_path 指向旧版 strategy sequence pickle。
    输出语义：output_path 写出旧版状态图结构，并返回本文件摘要。
    关键约束：旧版状态矩阵取指定状态列转置后整体加 1，状态名不包含 ``BN10``。
    """

    sequence = pd.read_pickle(input_path)
    states = sequence["state"]
    data = states[STATE_NAMES].values.T + 1
    graph = module.PC(data, data.shape[1])
    result = {
        "G": graph,
        "stateNames": STATE_NAMES,
        "data": data,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(result, file)
    return {
        "input_file": input_path.name,
        "output_file": output_path.name,
        "sample_count": int(data.shape[1]),
        "edge_count": int(graph.sum() // 2),
    }


def parse_args() -> argparse.Namespace:
    """解析旧版状态依赖图临时入口参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """批量运行旧版状态依赖图学习并写出摘要报告。"""

    args = parse_args()
    module = load_legacy_module(args.legacy_root)
    input_files = list_pickle_files(args.input_dir)
    if not input_files:
        raise FileNotFoundError(f"旧版状态依赖图输入目录中没有 pkl：{args.input_dir}")

    summaries = [
        process_one_file(module, input_path, args.output_dir / input_path.name)
        for input_path in input_files
    ]
    report = {
        "legacy_root": str(args.legacy_root),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "file_count": len(summaries),
        "total_samples": int(sum(item["sample_count"] for item in summaries)),
        "total_edges": int(sum(item["edge_count"] for item in summaries)),
        "files": summaries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("file_count", "total_samples", "total_edges")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
