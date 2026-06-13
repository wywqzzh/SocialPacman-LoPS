"""验证完整 human fMRI 数据预处理流程的输出一致性。"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from run_human_fmri_data_preprocess import list_pickle_files, project_root


def load_pickle(path: Path) -> Any:
    """读取 pickle 文件并返回保存的对象。"""

    with path.open("rb") as file:
        return pickle.load(file)


def compare_dataframe_pair(output_path: Path, baseline_path: Path) -> list[str]:
    """比较两个 DataFrame pickle 文件，返回字段级差异标签。"""

    output = pd.read_pickle(output_path)
    baseline = pd.read_pickle(baseline_path)
    mismatches: list[str] = []

    if output.shape != baseline.shape:
        mismatches.append("shape")
    if list(output.columns) != list(baseline.columns):
        mismatches.append("columns")
    if not output.index.equals(baseline.index):
        mismatches.append("index")

    try:
        assert_frame_equal(output, baseline, check_exact=True)
    except AssertionError:
        mismatches.append("values")
    return mismatches


def compare_dataframe_directory(output_dir: Path, baseline_dir: Path) -> dict[str, Any]:
    """比较两个目录下的 DataFrame pickle 文件是否完全一致。"""

    baseline_files = list_pickle_files(baseline_dir)
    output_files = list_pickle_files(output_dir)
    failed_files: dict[str, list[str]] = {}

    for file_name in baseline_files:
        output_path = output_dir / file_name
        baseline_path = baseline_dir / file_name
        if not output_path.exists():
            failed_files[file_name] = ["missing_output"]
            continue

        mismatches = compare_dataframe_pair(output_path, baseline_path)
        if mismatches:
            failed_files[file_name] = mismatches

    for file_name in sorted(set(output_files) - set(baseline_files)):
        failed_files[file_name] = ["extra_output"]

    return {
        "baseline_files": len(baseline_files),
        "output_files": len(output_files),
        "failed_files": failed_files,
        "is_exact_match": len(failed_files) == 0,
    }


def compare_strategy_sequence_pair(output_path: Path, baseline_path: Path) -> list[str]:
    """比较单个 StrategySequence 结果文件是否完全一致。"""

    output = load_pickle(output_path)
    baseline = load_pickle(baseline_path)
    mismatches: list[str] = []

    if set(output.keys()) != set(baseline.keys()):
        mismatches.append("keys")
    if output.get("seq") != baseline.get("seq"):
        mismatches.append("seq")
    if output.get("S") != baseline.get("S"):
        mismatches.append("S")
    if [str(name) for name in output.get("fileNames", [])] != [str(name) for name in baseline.get("fileNames", [])]:
        mismatches.append("fileNames")

    try:
        assert_frame_equal(output["state"], baseline["state"], check_exact=True)
    except AssertionError:
        mismatches.append("state")

    try:
        assert_frame_equal(output["strategy"], baseline["strategy"], check_exact=True)
    except AssertionError:
        mismatches.append("strategy")

    try:
        assert_series_equal(output["strategyLabel"], baseline["strategyLabel"], check_exact=True)
    except AssertionError:
        mismatches.append("strategyLabel")
    return mismatches


def compare_strategy_sequence_directory(output_dir: Path, baseline_dir: Path) -> dict[str, Any]:
    """比较两个目录下的 StrategySequence pickle 文件是否完全一致。"""

    baseline_files = list_pickle_files(baseline_dir)
    output_files = list_pickle_files(output_dir)
    failed_files: dict[str, list[str]] = {}

    for file_name in baseline_files:
        output_path = output_dir / file_name
        baseline_path = baseline_dir / file_name
        if not output_path.exists():
            failed_files[file_name] = ["missing_output"]
            continue

        mismatches = compare_strategy_sequence_pair(output_path, baseline_path)
        if mismatches:
            failed_files[file_name] = mismatches

    for file_name in sorted(set(output_files) - set(baseline_files)):
        failed_files[file_name] = ["extra_output"]

    return {
        "baseline_files": len(baseline_files),
        "output_files": len(output_files),
        "failed_files": failed_files,
        "is_exact_match": len(failed_files) == 0,
    }


def validate_human_fmri_outputs(
    ghost2_discrete_dir: Path,
    formed_ghost2_dir: Path,
    strategy_sequence_dir: Path,
    baseline_root: Path,
    report_path: Path,
) -> dict[str, Any]:
    """验证完整 human fMRI 数据预处理各阶段输出是否与 baseline 一致。

    输入语义：当前主流程只保留 two-ghost 数据，因此只比较 ghost2 离散特征、
    formed 数据和最终 strategy sequence。
    输出语义：写出 JSON 报告并返回同一报告对象。
    关键约束：这里不再检查四鬼目录，避免验证逻辑重新引入四鬼分支。
    """

    stage_reports = {
        "fmri_discrete_feature_data_ghost2": compare_dataframe_directory(
            ghost2_discrete_dir,
            baseline_root / "fmri_discrete_feature_data_ghost2",
        ),
        "fmri_formed_data_ghost2": compare_dataframe_directory(
            formed_ghost2_dir,
            baseline_root / "fmri_formed_data_ghost2",
        ),
        "strategy_sequence": compare_strategy_sequence_directory(
            strategy_sequence_dir,
            baseline_root / "strategy_sequence",
        ),
    }

    report = {
        "baseline_root": str(baseline_root),
        "stage_reports": stage_reports,
        "is_exact_match": all(stage["is_exact_match"] for stage in stage_reports.values()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    """解析验证脚本参数，允许外部覆盖各阶段输出和 baseline 目录。"""

    data_root = project_root() / "data" / "human_fmri_data_preprocess"
    parser = argparse.ArgumentParser(description="验证完整 human fMRI 数据预处理输出。")
    parser.add_argument(
        "--ghost2-discrete-dir",
        type=Path,
        default=data_root / "fmri_discrete_feature_data_ghost2",
        help="待验证的 ghost2 离散特征数据目录。",
    )
    parser.add_argument(
        "--formed-ghost2-dir",
        type=Path,
        default=data_root / "fmri_formed_data_ghost2",
        help="待验证的 ghost2 formed 数据目录。",
    )
    parser.add_argument(
        "--strategy-sequence-dir",
        type=Path,
        default=data_root / "strategy_sequence",
        help="待验证的最终 StrategySequence 目录。",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=data_root / "baseline",
        help="当前旧脚本输出 baseline 根目录。",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=data_root / "validation" / "human_fmri_data_preprocess_validation_report.json",
        help="验证报告 JSON 输出路径。",
    )
    return parser.parse_args()


def main() -> None:
    """运行完整流程输出验证，并在任何阶段失败时返回非零退出码。"""

    args = parse_args()
    report = validate_human_fmri_outputs(
        ghost2_discrete_dir=args.ghost2_discrete_dir,
        formed_ghost2_dir=args.formed_ghost2_dir,
        strategy_sequence_dir=args.strategy_sequence_dir,
        baseline_root=args.baseline_root,
        report_path=args.validation_report,
    )

    failed_stage_count = sum(not stage["is_exact_match"] for stage in report["stage_reports"].values())
    print(
        "validation "
        f"stages={len(report['stage_reports'])} "
        f"failed_stages={failed_stage_count} "
        f"report={args.validation_report}"
    )
    if not report["is_exact_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
