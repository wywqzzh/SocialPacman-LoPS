#!/usr/bin/env python3
"""运行 strategy_sequence 到状态依赖图的学习与一致性验证。"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.state_dependency_graph import (  # noqa: E402
    DEFAULT_STATE_NAMES,
    build_state_matrix,
    convert_to_legacy_state_graph,
    learn_state_dependency_graph,
    process_state_dependency_graph_directory,
)


def parse_args() -> argparse.Namespace:
    """解析状态依赖图学习脚本的命令行参数。

    输入语义：调用方可以显式传入数据目录、状态列和算法先验参数。
    输出语义：返回可直接驱动批处理和验证流程的参数对象。
    关键约束：默认路径只存在于脚本层，正式模块不内置任何数据目录。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "pipeline_data/human_fmri_data_preprocess/strategy_sequence",
        help="human_fmri_data_preprocess 生成的 strategy_sequence 输入目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "pipeline_data/state_dependency_graph/state_dependency_graph_data",
        help="状态依赖图输出目录。",
    )
    parser.add_argument(
        "--state-names",
        nargs="+",
        default=list(DEFAULT_STATE_NAMES),
        help="参与图学习的状态列名。",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet 先验总强度。")
    parser.add_argument("--validate", action="store_true", help="生成后与旧结果基准进行完全一致性验证。")
    parser.add_argument("--validate-only", action="store_true", help="跳过生成步骤，只验证已有输出。")
    parser.add_argument(
        "--legacy-dir",
        type=Path,
        default=PROJECT_ROOT / "data/generate_grammar/input/state_graph",
        help="当前仓库内的旧结果基准目录，仅用于显式验证。",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=PROJECT_ROOT / "data/state_dependency_graph/validation",
        help="验证报告和过程 trace 输出目录。",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口：批量学习状态依赖图，并按需执行一致性验证。"""

    args = parse_args()

    if not args.validate_only:
        summaries = process_state_dependency_graph_directory(
            args.input_dir,
            args.output_dir,
            state_names=args.state_names,
            alpha=args.alpha,
        )
        print("state_dependency_graph 生成完成")
        print(f"subject 数量：{len(summaries)}")
        print(f"样本帧总数：{sum(item['sample_count'] for item in summaries)}")
        print(f"总无向边数：{sum(item['edge_count'] for item in summaries)}")
        print(f"输出目录：{args.output_dir.resolve()}")

    if args.validate or args.validate_only:
        report = validate_state_dependency_graph_outputs(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            legacy_dir=args.legacy_dir,
            validation_dir=args.validation_dir,
            state_names=args.state_names,
            alpha=args.alpha,
        )
        print("state_dependency_graph 验证完成")
        print(f"验证文件数：{report['checked_files']}")
        print(f"失败文件数：{report['failed_files']}")
        print(f"验证报告：{report['report_path']}")
        if report["failed_files"]:
            raise SystemExit(1)


def validate_state_dependency_graph_outputs(
    input_dir: Path,
    output_dir: Path,
    legacy_dir: Path,
    validation_dir: Path,
    state_names: list[str],
    alpha: float,
) -> dict[str, Any]:
    """比较新状态依赖图输出与旧结果基准，并保存过程 trace。

    输入语义：input_dir 提供原始 strategy_sequence，output_dir 提供新输出，legacy_dir 提供旧结果基准。
    输出语义：返回验证摘要，并把 JSON 报告写入 validation_dir。
    关键约束：矩阵比较使用完全相等，不使用数值容差。
    """

    validation_dir.mkdir(parents=True, exist_ok=True)
    output_files = sorted(output_dir.glob("*.pkl"))
    if not output_files:
        raise FileNotFoundError(f"找不到可验证的新输出：{output_dir}")

    comparisons = []
    traces = []
    for output_file in output_files:
        legacy_file = legacy_dir / output_file.name
        input_file = input_dir / output_file.name
        comparison = _compare_one_output(output_file, legacy_file)
        comparisons.append(comparison)
        traces.append(_trace_one_input(input_file, state_names=state_names, alpha=alpha))

    report = {
        "checked_files": len(comparisons),
        "failed_files": sum(0 if item["passed"] else 1 for item in comparisons),
        "comparisons": comparisons,
        "trace_summary": traces,
    }
    report_path = validation_dir / "state_dependency_graph_validation_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    report["report_path"] = str(report_path.resolve())
    return report


def _compare_one_output(output_file: Path, legacy_file: Path) -> dict[str, Any]:
    """比较一个新输出文件和一个旧结果基准文件。

    输入语义：output_file 使用新结构，legacy_file 使用旧字段结构。
    输出语义：返回包含三个字段级比较结果的字典。
    关键约束：如果文件缺失或字段不一致，结果会标记为失败而不是静默跳过。
    """

    if not legacy_file.exists():
        return {
            "file": output_file.name,
            "passed": False,
            "reason": f"缺少旧结果基准：{legacy_file}",
        }

    with output_file.open("rb") as file:
        new_result = pickle.load(file)
    with legacy_file.open("rb") as file:
        legacy_result = pickle.load(file)

    legacy_from_new = convert_to_legacy_state_graph(new_result)
    state_names_equal = list(legacy_from_new["stateNames"]) == list(legacy_result["stateNames"])
    state_matrix_equal = np.array_equal(legacy_from_new["data"], legacy_result["data"])
    adjacency_equal = np.array_equal(legacy_from_new["G"], legacy_result["G"])
    passed = state_names_equal and state_matrix_equal and adjacency_equal

    return {
        "file": output_file.name,
        "passed": bool(passed),
        "state_names_equal": bool(state_names_equal),
        "state_matrix_equal": bool(state_matrix_equal),
        "adjacency_equal": bool(adjacency_equal),
        "state_matrix_shape": list(np.asarray(legacy_from_new["data"]).shape),
        "adjacency_shape": list(np.asarray(legacy_from_new["G"]).shape),
        "edge_count": int(np.sum(legacy_from_new["G"]) // 2),
    }


def _trace_one_input(input_file: Path, state_names: list[str], alpha: float) -> dict[str, Any]:
    """对一个输入文件重新运行学习过程并记录关键中间指标。

    输入语义：input_file 指向 strategy_sequence 数据。
    输出语义：返回状态矩阵 hash、状态数、检验次数、删边次数和最终图 hash。
    关键约束：trace 重新运行同一确定性学习过程，只用于验证过程可追踪性。
    """

    strategy_sequence = pd.read_pickle(input_file)
    state_matrix = build_state_matrix(strategy_sequence, state_names)
    events: list[dict[str, Any]] = []
    result = learn_state_dependency_graph(
        strategy_sequence,
        state_names=state_names,
        alpha=alpha,
        trace_callback=events.append,
    )
    test_events = [event for event in events if event["event"] == "test"]
    remove_events = [event for event in events if event["event"] == "remove_edge"]
    start_event = next(event for event in events if event["event"] == "start")
    return {
        "file": input_file.name,
        "state_matrix_sha256": _array_sha256(state_matrix),
        "adjacency_sha256": _array_sha256(result["adjacency_matrix"]),
        "nstates": start_event["nstates"],
        "test_count": len(test_events),
        "remove_edge_count": len(remove_events),
        "tests": test_events,
        "removed_edges": remove_events,
    }


def _array_sha256(array: np.ndarray) -> str:
    """计算 numpy 数组内容和形状的稳定 SHA256 摘要。

    输入语义：array 是需要记录的矩阵。
    输出语义：返回同时包含 dtype、shape 和原始字节内容的摘要。
    关键约束：摘要用于验证报告，避免在摘要字段中重复保存大型矩阵。
    """

    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(value.shape).encode("utf-8"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


if __name__ == "__main__":
    main()
