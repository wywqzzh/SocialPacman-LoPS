#!/usr/bin/env python3
"""验证人类权重修正结果和最终 grammar 不变量。"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.generate_grammar.config import GenerateGrammarConfig, GrammarLearningParams  # noqa: E402
from LoPS.generate_grammar.pipeline import run_generate_grammar  # noqa: E402
from script.extract_features_human.run_extract_features_human import (  # noqa: E402
    process_extract_features_human,
)
from script.human_fmri_data_preprocess.run_human_fmri_data_preprocess import (  # noqa: E402
    process_human_fmri_data,
)
from script.revise_human_weight.run_revise_human_weight import (  # noqa: E402
    RANDOM_DIAGNOSTIC_COLUMNS,
    process_revise_human_weight,
)
from script.state_dependency_graph.run_state_dependency_graph import (  # noqa: E402
    process_state_dependency_graph_directory,
)


def list_pickle_names(data_dir: Path) -> list[str]:
    """列出目录下的 pickle 文件名。

    输入语义：data_dir 是扁平数据目录。
    输出语义：返回排序后的 `.pkl` 文件名；目录不存在时返回空列表。
    关键约束：用于比较报告时不静默假定目录存在。
    """

    if not data_dir.is_dir():
        return []
    return sorted(path.name for path in data_dir.glob("*.pkl"))


def count_series_differences(left: pd.Series, right: pd.Series) -> int:
    """统计两个 Series 的逐行差异数量。

    输入语义：left/right 是同名列，索引已经对齐。
    输出语义：返回不同元素数量，NaN 与 NaN 视为相同。
    关键约束：用于随机诊断列报告，不作为硬失败。
    """

    diff_count = 0
    for left_value, right_value in zip(left.reset_index(drop=True), right.reset_index(drop=True)):
        if pd.isna(left_value) and pd.isna(right_value):
            continue
        try:
            same = bool(left_value == right_value)
        except ValueError:
            same = bool(np.array_equal(left_value, right_value))
        if not same:
            diff_count += 1
    return diff_count


def compare_corrected_weight_file(output_path: Path, baseline_path: Path) -> dict[str, Any]:
    """比较一个 corrected weight 文件。

    输入语义：output_path 是新脚本输出，baseline_path 是旧脚本 baseline。
    输出语义：返回结构、非随机列差异和随机诊断列差异数量。
    关键约束：`predict_dir`/`revise_is_correct` 不参与硬失败。
    """

    output = pd.read_pickle(output_path)
    baseline = pd.read_pickle(baseline_path)
    hard_mismatches: list[str] = []
    diagnostic_diffs: dict[str, int] = {}

    if output.shape != baseline.shape:
        hard_mismatches.append("shape")
    if list(output.columns) != list(baseline.columns):
        hard_mismatches.append("columns")
    if not output.index.equals(baseline.index):
        hard_mismatches.append("index")

    common_columns = [column for column in baseline.columns if column in output.columns]
    for column in common_columns:
        if column in RANDOM_DIAGNOSTIC_COLUMNS:
            diagnostic_diffs[column] = count_series_differences(output[column], baseline[column])
            continue
        try:
            assert_series_equal(output[column], baseline[column], check_exact=True, check_names=False)
        except AssertionError:
            hard_mismatches.append(column)

    return {
        "hard_mismatches": sorted(set(hard_mismatches)),
        "diagnostic_diffs": diagnostic_diffs,
        "passed": len(hard_mismatches) == 0,
    }


def compare_corrected_weight_directory(output_dir: Path, baseline_dir: Path) -> dict[str, Any]:
    """比较 corrected weight 输出目录。

    输入语义：output_dir 是新输出，baseline_dir 是旧结果。
    输出语义：返回文件数、失败文件和随机诊断列汇总。
    关键约束：缺失文件和额外文件都视为硬失败。
    """

    output_files = list_pickle_names(output_dir)
    baseline_files = list_pickle_names(baseline_dir)
    failed_files: dict[str, list[str]] = {}
    diagnostic_summary: dict[str, dict[str, int]] = {}

    for file_name in baseline_files:
        output_path = output_dir / file_name
        baseline_path = baseline_dir / file_name
        if not output_path.exists():
            failed_files[file_name] = ["missing_output"]
            continue
        comparison = compare_corrected_weight_file(output_path, baseline_path)
        diagnostic_summary[file_name] = comparison["diagnostic_diffs"]
        if not comparison["passed"]:
            failed_files[file_name] = comparison["hard_mismatches"]

    for file_name in sorted(set(output_files) - set(baseline_files)):
        failed_files[file_name] = ["extra_output"]

    total_diagnostic_diffs = {
        column: sum(item.get(column, 0) for item in diagnostic_summary.values())
        for column in RANDOM_DIAGNOSTIC_COLUMNS
    }
    return {
        "output_files": len(output_files),
        "baseline_files": len(baseline_files),
        "failed_files": failed_files,
        "diagnostic_diffs_by_file": diagnostic_summary,
        "total_diagnostic_diffs": total_diagnostic_diffs,
        "is_exact_match_excluding_random_diagnostics": len(failed_files) == 0,
    }


def load_pickle(path: Path) -> Any:
    """读取 pickle 对象。

    输入语义：path 指向 pickle 文件。
    输出语义：返回反序列化对象。
    关键约束：用于 grammar 结果递归比较。
    """

    with path.open("rb") as file:
        return pickle.load(file)


def compare_values(left: Any, right: Any, path: str) -> list[str]:
    """递归比较两个 grammar 输出值。

    输入语义：left/right 可以是标量、数组、DataFrame、Series、字典或序列。
    输出语义：返回差异路径列表；空列表表示完全一致。
    关键约束：不使用容差，所有字段必须精确一致。
    """

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return [f"{path}: type"]
        return [] if np.array_equal(left, right) else [path]

    if isinstance(left, pd.DataFrame) or isinstance(right, pd.DataFrame):
        if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
            return [f"{path}: type"]
        try:
            assert_frame_equal(left, right, check_exact=True)
        except AssertionError:
            return [path]
        return []

    if isinstance(left, pd.Series) or isinstance(right, pd.Series):
        if not isinstance(left, pd.Series) or not isinstance(right, pd.Series):
            return [f"{path}: type"]
        try:
            assert_series_equal(left, right, check_exact=True)
        except AssertionError:
            return [path]
        return []

    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return [f"{path}: type"]
        if set(left.keys()) != set(right.keys()):
            return [f"{path}: keys"]
        differences: list[str] = []
        for key in left:
            differences.extend(compare_values(left[key], right[key], f"{path}.{key}"))
        return differences

    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return [f"{path}: type"]
        if len(left) != len(right):
            return [f"{path}: length"]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(compare_values(left_item, right_item, f"{path}[{index}]"))
        return differences

    return [] if left == right else [path]


def compare_pickle_directory(left_dir: Path, right_dir: Path) -> dict[str, Any]:
    """比较两个 grammar 输出目录。

    输入语义：left_dir/right_dir 是两条链路的 grammar 目录。
    输出语义：返回文件数、失败文件和是否完全一致。
    关键约束：对象字段完全一致即可，不要求 pickle 字节哈希一致。
    """

    left_files = list_pickle_names(left_dir)
    right_files = list_pickle_names(right_dir)
    failed_files: dict[str, list[str]] = {}
    for file_name in sorted(set(left_files) | set(right_files)):
        left_path = left_dir / file_name
        right_path = right_dir / file_name
        if not left_path.exists():
            failed_files[file_name] = ["missing_left"]
            continue
        if not right_path.exists():
            failed_files[file_name] = ["missing_right"]
            continue
        differences = compare_values(load_pickle(left_path), load_pickle(right_path), file_name)
        if differences:
            failed_files[file_name] = differences[:20]

    return {
        "left_files": len(left_files),
        "right_files": len(right_files),
        "failed_files": failed_files,
        "is_exact_match": len(failed_files) == 0,
    }


def run_full_chain_from_corrected_weight(
    run_dir: Path,
    corrected_weight_dir: Path,
    constant_dir: Path,
    *,
    processes: int,
) -> Path:
    """从 corrected weight 开始运行当前完整主链路。

    输入语义：corrected_weight_dir 是一条链路的修正权重输出。
    输出语义：生成 extractor、human preprocess、state graph 和 grammar，返回 grammar 目录。
    关键约束：所有中间产物写入 run_dir，不覆盖仓库正式输出。
    """

    feature_dir = run_dir / "feature_data"
    discrete_dir = run_dir / "discrete_feature_data"
    process_extract_features_human(
        input_dir=corrected_weight_dir,
        constant_dir=constant_dir,
        feature_output_dir=feature_dir,
        discrete_output_dir=discrete_dir,
        processes=processes,
    )

    sequence_dir = run_dir / "strategy_sequence"
    process_human_fmri_data(
        raw_discrete_dir=discrete_dir,
        ghost2_discrete_dir=run_dir / "fmri_discrete_feature_data_ghost2",
        formed_ghost2_dir=run_dir / "fmri_formed_data_ghost2",
        strategy_sequence_dir=sequence_dir,
    )

    state_dir = run_dir / "state_graph"
    process_state_dependency_graph_directory(sequence_dir, state_dir)

    grammar_dir = run_dir / "grammar"
    run_generate_grammar(
        GenerateGrammarConfig(
            strategy_sequence_dir=sequence_dir,
            state_graph_dir=state_dir,
            output_dir=grammar_dir,
            learning=replace(
                GrammarLearningParams(),
                chunk_alpha=0.5,
                condition_alpha=0.5,
                skip_gram_alpha=0.5,
                max_iterations=100000,
            ),
        )
    )
    return grammar_dir


def validate_grammar_invariant(
    run_root: Path,
    baseline_corrected_weight_dir: Path,
    new_corrected_weight_dir: Path,
    constant_dir: Path,
    *,
    processes: int,
) -> dict[str, Any]:
    """验证新 corrected weight 接入主链路后 grammar 是否不变。

    输入语义：baseline_corrected_weight_dir 是旧结果，新目录是重构输出。
    输出语义：返回两条链路最终 grammar 的比较报告。
    关键约束：两条链路都从 corrected weight 重新生成全部下游阶段。
    """

    baseline_grammar_dir = run_full_chain_from_corrected_weight(
        run_root / "baseline_chain",
        baseline_corrected_weight_dir,
        constant_dir,
        processes=processes,
    )
    new_grammar_dir = run_full_chain_from_corrected_weight(
        run_root / "new_chain",
        new_corrected_weight_dir,
        constant_dir,
        processes=processes,
    )
    comparison = compare_pickle_directory(baseline_grammar_dir, new_grammar_dir)
    comparison["baseline_grammar_dir"] = str(baseline_grammar_dir)
    comparison["new_grammar_dir"] = str(new_grammar_dir)
    return comparison


def validate_revise_human_weight(
    input_dir: Path,
    output_dir: Path,
    baseline_dir: Path,
    constant_dir: Path,
    validation_root: Path,
    *,
    processes: int,
    scared_time: int,
    skip_revise: bool,
) -> dict[str, Any]:
    """执行本轮两重验证。

    输入语义：目录均为扁平目录；skip_revise 可跳过重新生成输出。
    输出语义：返回报告并写入 `validation_root/run_*/report.json`。
    关键约束：非随机列或最终 grammar 任一失败都会导致总体验证失败。
    """

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_root = validation_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    revise_summary: list[dict[str, Any]] = []
    if not skip_revise:
        revise_summary = process_revise_human_weight(
            input_dir=input_dir,
            output_dir=output_dir,
            processes=processes,
            scared_time=scared_time,
        )

    corrected_weight_report = compare_corrected_weight_directory(output_dir, baseline_dir)
    grammar_report = validate_grammar_invariant(
        run_root / "grammar_invariant",
        baseline_corrected_weight_dir=baseline_dir,
        new_corrected_weight_dir=output_dir,
        constant_dir=constant_dir,
        processes=processes,
    )

    report = {
        "run_id": run_id,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "baseline_dir": str(baseline_dir),
        "constant_dir": str(constant_dir),
        "revise_summary": revise_summary,
        "corrected_weight": corrected_weight_report,
        "grammar_invariant": grammar_report,
        "is_passed": corrected_weight_report["is_exact_match_excluding_random_diagnostics"]
        and grammar_report["is_exact_match"],
    }
    report_path = run_root / "report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    """解析验证脚本参数。

    输入语义：允许覆盖输入、输出、baseline、常量和验证目录。
    输出语义：返回完整验证参数对象。
    关键约束：默认路径全部使用 LoPS 仓库内扁平目录。
    """

    data_root = PROJECT_ROOT / "data" / "revise_human_weight"
    parser = argparse.ArgumentParser(description="验证 revise_human_weight 输出和 grammar 不变量。")
    parser.add_argument("--input-dir", type=Path, default=data_root / "input" / "weight_data")
    parser.add_argument("--output-dir", type=Path, default=data_root / "corrected_weight_data")
    parser.add_argument("--baseline-dir", type=Path, default=data_root / "baseline" / "corrected_weight_data")
    parser.add_argument(
        "--constant-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "extract_features_human" / "input" / "constant_data",
    )
    parser.add_argument("--validation-root", type=Path, default=data_root / "validation")
    parser.add_argument("--processes", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--scared-time", type=int, default=63)
    parser.add_argument("--skip-revise", action="store_true", help="跳过重新生成 corrected weight，只验证已有输出。")
    return parser.parse_args()


def main() -> None:
    """命令行入口：执行两重验证并在失败时退出。"""

    args = parse_args()
    report = validate_revise_human_weight(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        baseline_dir=args.baseline_dir,
        constant_dir=args.constant_dir,
        validation_root=args.validation_root,
        processes=args.processes,
        scared_time=args.scared_time,
        skip_revise=args.skip_revise,
    )
    print(
        "revise_human_weight 验证完成 "
        f"corrected_exact_excluding_random={report['corrected_weight']['is_exact_match_excluding_random_diagnostics']} "
        f"grammar_exact={report['grammar_invariant']['is_exact_match']} "
        f"random_diffs={report['corrected_weight']['total_diagnostic_diffs']} "
        f"report={report['report_path']}"
    )
    if not report["is_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
