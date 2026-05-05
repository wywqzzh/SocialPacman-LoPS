from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from LoPS.generate_grammar.config import (
    GenerateGrammarConfig,
    GrammarLearningParams,
)
from LoPS.generate_grammar.pipeline import run_generate_grammar


def compare_values(old_value: Any, new_value: Any, path: str) -> list[str]:
    # 验证目标是 legacy 与旧 grammar 基准精确一致；不同类型分别使用最严格的比较方式。
    if isinstance(old_value, np.ndarray) or isinstance(new_value, np.ndarray):
        if not isinstance(old_value, np.ndarray) or not isinstance(new_value, np.ndarray):
            return [f"{path}: type mismatch {type(old_value).__name__} != {type(new_value).__name__}"]
        # ndarray 不使用容差；概率数组也必须逐元素完全一致。
        if not np.array_equal(old_value, new_value):
            return [f"{path}: ndarray mismatch"]
        return []

    if isinstance(old_value, pd.DataFrame) or isinstance(new_value, pd.DataFrame):
        if not isinstance(old_value, pd.DataFrame) or not isinstance(new_value, pd.DataFrame):
            return [f"{path}: type mismatch {type(old_value).__name__} != {type(new_value).__name__}"]
        try:
            # check_exact=True 明确禁止浮点容差，符合本轮“完全一致”的验证要求。
            pd.testing.assert_frame_equal(old_value, new_value, check_exact=True)
        except AssertionError as error:
            return [f"{path}: DataFrame mismatch: {error}"]
        return []

    if isinstance(old_value, Mapping) or isinstance(new_value, Mapping):
        if not isinstance(old_value, Mapping) or not isinstance(new_value, Mapping):
            return [f"{path}: type mismatch {type(old_value).__name__} != {type(new_value).__name__}"]
        differences = []
        for key, value in old_value.items():
            # 只要求旧输出已有 key 在新 legacy 中存在；structured 额外字段不参与此比较。
            key_path = f"{path}.{key}"
            if key not in new_value:
                differences.append(f"{key_path}: missing key")
                continue
            differences.extend(compare_values(value, new_value[key], key_path))
        return differences

    if isinstance(old_value, (list, tuple)) or isinstance(new_value, (list, tuple)):
        if not isinstance(old_value, (list, tuple)) or not isinstance(new_value, (list, tuple)):
            return [f"{path}: type mismatch {type(old_value).__name__} != {type(new_value).__name__}"]
        # 列表长度先比较，再逐项递归，方便报告精确到 key path 和 index。
        if len(old_value) != len(new_value):
            return [f"{path}: length mismatch {len(old_value)} != {len(new_value)}"]
        differences = []
        for index, old_item in enumerate(old_value):
            differences.extend(compare_values(old_item, new_value[index], f"{path}[{index}]"))
        return differences

    if old_value != new_value:
        return [f"{path}: value mismatch {old_value!r} != {new_value!r}"]
    return []


def compare_legacy_dict(old: Mapping[str, Any], new: Mapping[str, Any], file_name: str) -> list[str]:
    # 单文件 legacy 比较：以旧 pickle 的字段为准，逐 key 递归进入 compare_values。
    differences = []
    for key, old_value in old.items():
        key_path = f"{file_name}.{key}"
        if key not in new:
            differences.append(f"{key_path}: missing key")
            continue
        differences.extend(compare_values(old_value, new[key], key_path))
    return differences


def validate_outputs(config: GenerateGrammarConfig) -> int:
    # 验证脚本会先运行新 pipeline，再读取新输出中的 legacy 字典与固定旧基准比较。
    if config.baseline_grammar_dir is None:
        raise ValueError("baseline_grammar_dir is required for validation")

    output_paths = run_generate_grammar(config)
    differences = []
    for output_path in output_paths:
        file_name = output_path.name
        baseline_path = config.baseline_grammar_dir / file_name
        old_output = pd.read_pickle(baseline_path)
        new_output = pd.read_pickle(output_path)
        # 新 pickle 顶层有 legacy/structured；旧结果一致性只比较 legacy。
        legacy_output = new_output["legacy"]
        differences.extend(compare_legacy_dict(old_output, legacy_output, file_name))

    if differences:
        print("Validation failed:")
        for difference in differences:
            print(difference)
        return 1

    print(f"Validation passed for {len(output_paths)} files.")
    return 0


def parse_args() -> argparse.Namespace:
    # 参数与 run_generate_grammar.py 保持一致；默认值直接写 data 下的固定目录字符串。
    parser = argparse.ArgumentParser(description="Validate LoPS generate_grammar output against legacy grammar baseline.")
    parser.add_argument("--strategy-sequence-dir", type=Path, default="data/generate_grammar/input/strategy_sequence")
    parser.add_argument("--state-graph-dir", type=Path, default="data/generate_grammar/input/state_graph")
    parser.add_argument("--baseline-grammar-dir", type=Path, default="data/generate_grammar/baseline/grammar")
    parser.add_argument("--output-dir", type=Path, default="data/generate_grammar/refactored-output/grammar")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-iterations", type=int, default=100000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # 验证默认使用旧入口 alpha=0.5，并同步应用到 chunk、condition 和 skip-gram 三处。
    learning = replace(
        GrammarLearningParams(),
        chunk_alpha=args.alpha,
        condition_alpha=args.alpha,
        skip_gram_alpha=args.alpha,
        max_iterations=args.max_iterations,
    )
    config = GenerateGrammarConfig(
        strategy_sequence_dir=args.strategy_sequence_dir,
        state_graph_dir=args.state_graph_dir,
        output_dir=args.output_dir,
        baseline_grammar_dir=args.baseline_grammar_dir,
        learning=learning,
    )
    return validate_outputs(config)


if __name__ == "__main__":
    sys.exit(main())
