#!/usr/bin/env python3
"""深度验证阶段输出比较工具。

该临时脚本比较两个目录中的 pickle 输出。比较目标是逻辑语义一致，而不是
pickle 字节一致：位置字符串会被解析为 tuple/list，numpy 数组会逐元素比较，
当前数据中语义为空的 fruit 列默认忽略。
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


IGNORED_COLUMNS = {"fruitPos", "fruitType"}


def normalize_value(value: Any) -> Any:
    """把单元格值规范成可比较的语义值。

    输入语义：value 可能来自旧版字符串字段、当前 tuple/list 字段、numpy 数组或 NaN。
    输出语义：返回结构化 Python 值；NaN 用统一哨兵表示。
    关键约束：数组比较仍保持精确比较，不使用数值容差。
    """

    if value is None:
        return "__MISSING__"
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return "__MISSING__"
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "nan":
            return "__MISSING__"
        try:
            return normalize_value(ast.literal_eval(stripped))
        except (ValueError, SyntaxError):
            return value
    if isinstance(value, np.ndarray):
        return normalize_value(value.tolist())
    if isinstance(value, tuple):
        return tuple(normalize_value(item) for item in value)
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def values_equal(left: Any, right: Any) -> bool:
    """比较两个规范化后的语义值是否一致。"""

    left_value = normalize_value(left)
    right_value = normalize_value(right)
    if isinstance(left_value, float) and isinstance(right_value, int):
        return left_value == right_value
    if isinstance(left_value, int) and isinstance(right_value, float):
        return left_value == right_value
    return left_value == right_value


def comparable_columns(current: pd.DataFrame, legacy: pd.DataFrame) -> tuple[list[str], list[str]]:
    """计算两个 DataFrame 可比较列和列级差异。"""

    current_columns = [column for column in current.columns if column not in IGNORED_COLUMNS]
    legacy_columns = [column for column in legacy.columns if column not in IGNORED_COLUMNS]
    missing_in_legacy = [column for column in current_columns if column not in legacy_columns]
    missing_in_current = [column for column in legacy_columns if column not in current_columns]
    if missing_in_legacy or missing_in_current:
        return [], [f"columns missing_in_legacy={missing_in_legacy} missing_in_current={missing_in_current}"]
    return current_columns, []


def compare_dataframe(current_path: Path, legacy_path: Path) -> dict[str, Any]:
    """比较单个 DataFrame pickle 文件。"""

    current = pd.read_pickle(current_path)
    legacy = pd.read_pickle(legacy_path)
    if current.shape[0] != legacy.shape[0]:
        return {
            "file": current_path.name,
            "passed": False,
            "reason": "row_count",
            "current_rows": int(current.shape[0]),
            "legacy_rows": int(legacy.shape[0]),
        }

    columns, column_errors = comparable_columns(current, legacy)
    if column_errors:
        return {"file": current_path.name, "passed": False, "reason": "columns", "details": column_errors}

    for column in columns:
        current_values = current[column].tolist()
        legacy_values = legacy[column].tolist()
        for row_index, (current_value, legacy_value) in enumerate(zip(current_values, legacy_values)):
            if not values_equal(current_value, legacy_value):
                return {
                    "file": current_path.name,
                    "passed": False,
                    "reason": "value",
                    "row": int(row_index),
                    "column": column,
                    "current_value": repr(current_value),
                    "legacy_value": repr(legacy_value),
                    "current_normalized": repr(normalize_value(current_value)),
                    "legacy_normalized": repr(normalize_value(legacy_value)),
                }
    return {"file": current_path.name, "passed": True, "rows": int(current.shape[0]), "columns": len(columns)}


def compare_directories(current_dir: Path, legacy_dir: Path) -> dict[str, Any]:
    """比较两个目录下同名 pickle DataFrame。"""

    current_files = sorted(path.name for path in current_dir.glob("*.pkl"))
    legacy_files = sorted(path.name for path in legacy_dir.glob("*.pkl"))
    results: list[dict[str, Any]] = []
    for file_name in sorted(set(current_files) | set(legacy_files)):
        current_path = current_dir / file_name
        legacy_path = legacy_dir / file_name
        if not current_path.exists() or not legacy_path.exists():
            results.append(
                {
                    "file": file_name,
                    "passed": False,
                    "reason": "missing_file",
                    "missing_current": not current_path.exists(),
                    "missing_legacy": not legacy_path.exists(),
                }
            )
            continue
        results.append(compare_dataframe(current_path, legacy_path))

    failed = [item for item in results if not item["passed"]]
    return {
        "current_dir": str(current_dir),
        "legacy_dir": str(legacy_dir),
        "current_files": len(current_files),
        "legacy_files": len(legacy_files),
        "checked_files": len(results),
        "failed_files": failed,
        "is_exact_semantic_match": len(failed) == 0,
    }


def parse_args() -> argparse.Namespace:
    """解析比较参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-dir", type=Path, required=True)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """运行目录比较并按结果设置退出码。"""

    args = parse_args()
    report = compare_directories(args.current_dir, args.legacy_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("checked_files", "is_exact_semantic_match")}, ensure_ascii=False))
    if not report["is_exact_semantic_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
