#!/usr/bin/env python3
"""运行旧版 grammar induction 逻辑的临时验证入口。

旧版 ``GrammarInductionHuman.py`` 固定读取 ``HumanData/seq`` 和 ``HumanData/state``，
并把结果写入旧项目目录。本脚本复用旧版 ``Chunk``、``skip_gram`` 和条件图读取逻辑，
只接管输入输出路径，确保深度验证过程中旧版结果保存到 ``data_temp``。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STATE_NAMES = ["IS1", "IS2", "PG1", "PG2", "PE", "BN5"]


def list_pickle_files(data_dir: Path) -> list[str]:
    """列出目录中的 pkl 文件名，并返回稳定排序后的列表。"""

    return sorted(path.name for path in data_dir.glob("*.pkl"))


def load_legacy_module(legacy_root: Path) -> Any:
    """导入旧版 ``GrammarInductionHuman.py`` 模块。

    输入语义：legacy_root 是旧项目根目录。
    输出语义：返回已导入的旧版模块对象。
    关键约束：旧模块依赖 ``PGM`` 包路径，因此需要把旧项目根目录加入 ``sys.path``。
    """

    legacy_root = legacy_root.resolve()
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))

    module_path = legacy_root / "GrammarInduction" / "GrammarInductionHuman.py"
    spec = importlib.util.spec_from_file_location("legacy_grammar_induction_human", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入旧版 GrammarInductionHuman：{module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_one_file(
    module: Any,
    sequence_dir: Path,
    state_graph_dir: Path,
    output_dir: Path,
    file_name: str,
    alpha: float,
) -> dict[str, Any]:
    """使用旧版 Chunk 流程处理一个 strategy sequence 文件。

    输入语义：sequence_dir 和 state_graph_dir 分别提供同名的序列与状态图文件。
    输出语义：output_dir 写出旧版 grammar pickle，返回本文件摘要。
    关键约束：旧版学习前会从序列中删除 ``N``，对应状态行也同步删除；skip-gram 检测
    使用删除前的 ``N`` 位置。
    """

    chunk = module.Chunk()
    sequence_path = sequence_dir / file_name
    state_path = state_graph_dir / file_name
    output_path = output_dir / file_name

    result = pd.read_pickle(sequence_path)
    sequence = result["seq"]
    strategy_symbols = result["S"]
    states = result["state"][STATE_NAMES]
    states.reset_index(inplace=True, drop=True)

    # 旧版 grammar 学习阶段移除 N，skip-gram 阶段再根据原始位置检测 N->EA 关系。
    n_positions = np.where((np.array(list(sequence)) == "N"))[0]
    sequence_without_n = sequence.replace("N", "")
    states_without_n = states.drop(n_positions)

    cluster_file_names = result["fileNames"]
    condition = module.getConditionGraph(str(state_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    sets, _, _, _ = chunk.Chunking(
        sequence_without_n,
        strategy_symbols,
        state=states_without_n,
        condition=condition,
        save_name=str(output_path),
        clusterFileNames=cluster_file_names,
        alpha=alpha,
    )

    # 旧脚本只打印排序后的 sets，不改变最终保存结果；这里保留读取、追加 skipGram 字段的语义。
    with output_path.open("rb") as file:
        grammar_result = pickle.load(file)

    skip_gram_found, skip_gram_count = chunk.skip_gram(grammar_result, n_positions, alpha)
    grammar_result["skipGram"] = bool(skip_gram_found)
    grammar_result["skipGramNum"] = skip_gram_count
    with output_path.open("wb") as file:
        pickle.dump(grammar_result, file)

    return {
        "input_file": file_name,
        "grammar_count": len(sets),
        "skip_gram": bool(skip_gram_found),
        "skip_gram_count": int(skip_gram_count),
    }


def parse_args() -> argparse.Namespace:
    """解析旧版 generate grammar 临时入口参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--strategy-sequence-dir", type=Path, required=True)
    parser.add_argument("--state-graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    """批量运行旧版 grammar induction 并写出摘要报告。"""

    args = parse_args()
    module = load_legacy_module(args.legacy_root)
    file_names = list_pickle_files(args.strategy_sequence_dir)
    if not file_names:
        raise FileNotFoundError(f"旧版 grammar 输入目录中没有 pkl：{args.strategy_sequence_dir}")

    summaries = [
        process_one_file(
            module=module,
            sequence_dir=args.strategy_sequence_dir,
            state_graph_dir=args.state_graph_dir,
            output_dir=args.output_dir,
            file_name=file_name,
            alpha=args.alpha,
        )
        for file_name in file_names
    ]
    report = {
        "legacy_root": str(args.legacy_root),
        "strategy_sequence_dir": str(args.strategy_sequence_dir),
        "state_graph_dir": str(args.state_graph_dir),
        "output_dir": str(args.output_dir),
        "file_count": len(summaries),
        "skip_gram_files": int(sum(1 for item in summaries if item["skip_gram"])),
        "files": summaries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("file_count", "skip_gram_files")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
