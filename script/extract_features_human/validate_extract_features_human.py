#!/usr/bin/env python3
"""验证人类特征提取结果与当前主链路 grammar 不变量。"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
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
from script.human_fmri_data_preprocess.run_human_fmri_data_preprocess import (  # noqa: E402
    process_human_fmri_data,
)
from script.state_dependency_graph.run_state_dependency_graph import (  # noqa: E402
    process_state_dependency_graph_directory,
)

try:
    from script.extract_features_human.run_extract_features_human import (  # noqa: E402
        process_extract_features_human,
    )
except ModuleNotFoundError:
    from run_extract_features_human import process_extract_features_human  # type: ignore  # noqa: E402


def list_pickle_names(data_dir: Path) -> list[str]:
    """列出目录中的 pickle 文件名。

    输入语义：data_dir 是待比较或待处理的扁平数据目录。
    输出语义：返回按文件名排序的 `.pkl` 文件名列表。
    关键约束：目录不存在时返回空列表，方便报告缺失输出。
    """

    if not data_dir.is_dir():
        return []
    return sorted(path.name for path in data_dir.glob("*.pkl"))


def compare_dataframe_file(output_path: Path, baseline_path: Path) -> list[str]:
    """比较两个 DataFrame pickle 文件是否完全一致。

    输入语义：output_path 是新脚本输出，baseline_path 是旧脚本 baseline。
    输出语义：返回差异标签列表；空列表表示完全一致。
    关键约束：比较 shape、列顺序、索引、dtype 和值，不使用容差。
    """

    output = pd.read_pickle(output_path)
    baseline = pd.read_pickle(baseline_path)
    mismatches: list[str] = []
    if output.shape != baseline.shape:
        mismatches.append("shape")
    if list(output.columns) != list(baseline.columns):
        mismatches.append("columns")
    if not output.index.equals(baseline.index):
        mismatches.append("index")
    if list(output.dtypes.astype(str)) != list(baseline.dtypes.astype(str)):
        mismatches.append("dtype")
    try:
        assert_frame_equal(output, baseline, check_exact=True)
    except AssertionError:
        mismatches.append("values")
    return sorted(set(mismatches))


def compare_dataframe_directory(output_dir: Path, baseline_dir: Path) -> dict[str, Any]:
    """比较两个目录下的 DataFrame pickle 文件。

    输入语义：output_dir 是待验目录，baseline_dir 是旧脚本结果目录。
    输出语义：返回文件数、失败文件和是否完全一致的报告。
    关键约束：额外文件和缺失文件都视为失败。
    """

    output_files = list_pickle_names(output_dir)
    baseline_files = list_pickle_names(baseline_dir)
    failed_files: dict[str, list[str]] = {}

    for file_name in baseline_files:
        output_path = output_dir / file_name
        baseline_path = baseline_dir / file_name
        if not output_path.exists():
            failed_files[file_name] = ["missing_output"]
            continue
        mismatches = compare_dataframe_file(output_path, baseline_path)
        if mismatches:
            failed_files[file_name] = mismatches

    for file_name in sorted(set(output_files) - set(baseline_files)):
        failed_files[file_name] = ["extra_output"]

    return {
        "output_files": len(output_files),
        "baseline_files": len(baseline_files),
        "failed_files": failed_files,
        "is_exact_match": len(failed_files) == 0,
    }


def load_pickle(path: Path) -> Any:
    """读取 pickle 文件并返回对象。

    输入语义：path 指向一个 pickle 文件。
    输出语义：返回反序列化后的 Python 对象。
    关键约束：调用方负责保证文件存在。
    """

    with path.open("rb") as file:
        return pickle.load(file)


def compare_values(left: Any, right: Any, path: str) -> list[str]:
    """递归比较 grammar 输出对象。

    输入语义：left 和 right 可以是标量、字典、列表、ndarray、DataFrame 或 Series。
    输出语义：返回差异路径列表；空列表表示当前值完全一致。
    关键约束：所有数组和表格都使用精确比较，不使用数值容差。
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
    """比较两个目录下的 grammar pickle 文件是否完全一致。

    输入语义：left_dir 和 right_dir 是两条链路生成的 grammar 输出目录。
    输出语义：返回逐文件差异路径和总体是否一致。
    关键约束：比较对象内容，不要求 pickle 字节哈希一致。
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


def copy_discrete_input(source_dir: Path, target_dir: Path) -> None:
    """把离散特征输入复制到临时链路目录。

    输入语义：source_dir 是 extractor 输出目录，target_dir 是临时 raw-discrete 目录。
    输出语义：复制所有 `.pkl` 文件到 target_dir。
    关键约束：后续 human_fmri_data_preprocess 会把这些 two-ghost 离散特征整理成
    ghost2 formed 数据，不再生成四鬼分支。
    """

    target_dir.mkdir(parents=True, exist_ok=True)
    for file_name in list_pickle_names(source_dir):
        shutil.copy2(source_dir / file_name, target_dir / file_name)


def run_current_chain(run_dir: Path, raw_discrete_dir: Path) -> Path:
    """运行当前主链路并返回 grammar 输出目录。

    输入语义：run_dir 是本次验证的临时目录，raw_discrete_dir 是当前主链路离散特征输入。
    输出语义：生成 strategy_sequence、state graph 和 grammar，返回 grammar 目录。
    关键约束：每一阶段都写入 run_dir 下，避免覆盖仓库当前正式输出。
    """

    sequence_dir = run_dir / "strategy_sequence"
    state_dir = run_dir / "state_graph"
    grammar_dir = run_dir / "grammar"
    process_human_fmri_data(
        raw_discrete_dir=raw_discrete_dir,
        ghost2_discrete_dir=run_dir / "fmri_discrete_feature_data_ghost2",
        formed_ghost2_dir=run_dir / "fmri_formed_data_ghost2",
        strategy_sequence_dir=sequence_dir,
    )
    process_state_dependency_graph_directory(sequence_dir, state_dir)
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


def validate_chain_invariant(
    run_root: Path,
    current_raw_discrete_dir: Path,
    new_discrete_dir: Path,
) -> dict[str, Any]:
    """验证新 extractor 接入主链路后最终 grammar 是否不变。

    输入语义：current_raw_discrete_dir 是当前主链路基准输入，new_discrete_dir 是新 extractor 输出。
    输出语义：返回 grammar 目录比较报告。
    关键约束：两条链路都从离散特征开始重新生成所有下游阶段。
    """

    current_grammar_dir = run_current_chain(run_root / "current_chain", current_raw_discrete_dir)

    new_raw_dir = run_root / "new_extractor_chain" / "raw_discrete_feature_data"
    copy_discrete_input(new_discrete_dir, new_raw_dir)
    new_grammar_dir = run_current_chain(run_root / "new_extractor_chain", new_raw_dir)

    comparison = compare_pickle_directory(current_grammar_dir, new_grammar_dir)
    comparison["current_grammar_dir"] = str(current_grammar_dir)
    comparison["new_grammar_dir"] = str(new_grammar_dir)
    return comparison


def validate_extract_features_human(
    input_dir: Path,
    constant_dir: Path,
    feature_output_dir: Path,
    discrete_output_dir: Path,
    baseline_feature_dir: Path,
    baseline_discrete_dir: Path,
    current_raw_discrete_dir: Path,
    validation_root: Path,
    *,
    processes: int,
    skip_extract: bool,
) -> dict[str, Any]:
    """执行 extractor 双重验证并写出完整报告。

    输入语义：各目录均为扁平目录；skip_extract 控制是否跳过重新生成输出。
    输出语义：返回包含两重验证结果的报告字典。
    关键约束：任一验证失败都应导致命令行入口返回非零退出码。
    """

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_root = validation_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    extraction_summary: list[dict[str, Any]] = []
    if not skip_extract:
        extraction_summary = process_extract_features_human(
            input_dir=input_dir,
            constant_dir=constant_dir,
            feature_output_dir=feature_output_dir,
            discrete_output_dir=discrete_output_dir,
            processes=processes,
        )

    feature_report = compare_dataframe_directory(feature_output_dir, baseline_feature_dir)
    discrete_report = compare_dataframe_directory(discrete_output_dir, baseline_discrete_dir)
    chain_report = validate_chain_invariant(
        run_root=run_root / "chain_invariant",
        current_raw_discrete_dir=current_raw_discrete_dir,
        new_discrete_dir=discrete_output_dir,
    )

    report = {
        "run_id": run_id,
        "input_dir": str(input_dir),
        "feature_output_dir": str(feature_output_dir),
        "discrete_output_dir": str(discrete_output_dir),
        "baseline_feature_dir": str(baseline_feature_dir),
        "baseline_discrete_dir": str(baseline_discrete_dir),
        "current_raw_discrete_dir": str(current_raw_discrete_dir),
        "extraction_summary": extraction_summary,
        "feature_data": feature_report,
        "discrete_feature_data": discrete_report,
        "grammar_invariant": chain_report,
        "is_exact_match": feature_report["is_exact_match"]
        and discrete_report["is_exact_match"]
        and chain_report["is_exact_match"],
    }
    report_path = run_root / "report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    """解析验证脚本命令行参数。

    输入语义：允许覆盖 extractor 输入输出、baseline 和当前主链路输入。
    输出语义：返回完整验证所需参数。
    关键约束：默认路径全部使用 LoPS 仓库内数据。
    """

    data_root = PROJECT_ROOT / "data" / "extract_features_human"
    parser = argparse.ArgumentParser(description="验证 ExtractFeaturesHuman 重构结果和最终 grammar 不变量。")
    parser.add_argument("--input-dir", type=Path, default=data_root / "input" / "corrected_weight_data")
    parser.add_argument("--constant-dir", type=Path, default=data_root / "input" / "constant_data")
    parser.add_argument("--feature-output-dir", type=Path, default=data_root / "feature_data")
    parser.add_argument("--discrete-output-dir", type=Path, default=data_root / "discrete_feature_data")
    parser.add_argument("--baseline-feature-dir", type=Path, default=data_root / "baseline" / "feature_data")
    parser.add_argument("--baseline-discrete-dir", type=Path, default=data_root / "baseline" / "discrete_feature_data")
    parser.add_argument(
        "--current-raw-discrete-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "human_fmri_data_preprocess" / "fmri_discrete_feature_data",
    )
    parser.add_argument("--validation-root", type=Path, default=data_root / "validation")
    parser.add_argument("--processes", type=int, default=min(34, os.cpu_count() or 1))
    parser.add_argument("--skip-extract", action="store_true", help="跳过 extractor 重新生成，只验证已有输出。")
    return parser.parse_args()


def main() -> None:
    """命令行入口：执行双重验证并在失败时退出。"""

    args = parse_args()
    report = validate_extract_features_human(
        input_dir=args.input_dir,
        constant_dir=args.constant_dir,
        feature_output_dir=args.feature_output_dir,
        discrete_output_dir=args.discrete_output_dir,
        baseline_feature_dir=args.baseline_feature_dir,
        baseline_discrete_dir=args.baseline_discrete_dir,
        current_raw_discrete_dir=args.current_raw_discrete_dir,
        validation_root=args.validation_root,
        processes=args.processes,
        skip_extract=args.skip_extract,
    )
    print(
        "extract_features_human 验证完成 "
        f"feature_exact={report['feature_data']['is_exact_match']} "
        f"discrete_exact={report['discrete_feature_data']['is_exact_match']} "
        f"grammar_exact={report['grammar_invariant']['is_exact_match']} "
        f"report={report['report_path']}"
    )
    if not report["is_exact_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
