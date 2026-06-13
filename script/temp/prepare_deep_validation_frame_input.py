#!/usr/bin/env python3
"""为深度验证准备 frame_data 输入。

该临时脚本只服务本轮新旧流程一致性验证：当给定的 frame data 缺少
``Unnamed: 0`` 时，按 ``DayTrial`` 的数字前缀和 ``Step`` 数值排序后补出
排序行号。这个字段在旧 corrected tile 逻辑中表示原始 frame 行位置，
不参与游戏状态计算。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def daytrial_key(value: Any) -> tuple[int, int, str]:
    """把 DayTrial 转成数字排序键。

    输入语义：value 形如 ``1-2-subject``。
    输出语义：返回前两个数字段和原字符串，确保 ``2-1`` 排在 ``10-1`` 前。
    关键约束：前两个字段必须可转成整数，否则说明输入不符合当前 fMRI 数据约定。
    """

    text = str(value)
    parts = text.split("-")
    if len(parts) < 2:
        raise ValueError(f"DayTrial 缺少前两个数字段：{value!r}")
    return int(parts[0]), int(parts[1]), text


def normalize_one_file(input_path: Path, output_path: Path) -> dict[str, Any]:
    """规范化单个 frame data 文件并写出。

    输入语义：input_path 指向原始 frame pickle，output_path 是规范化输出。
    输出语义：写出带 ``Unnamed: 0`` 的 DataFrame，并返回摘要。
    关键约束：若原文件已有 ``Unnamed: 0``，仍会按同一排序规则重建该列，
    保证本轮验证的 frameIndex 语义稳定。
    """

    data = pd.read_pickle(input_path)
    required = {"DayTrial", "Step"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{input_path.name} 缺少必要列：{missing}")

    working = data.copy()
    sort_keys = working["DayTrial"].map(daytrial_key)
    working["_day_num"] = sort_keys.map(lambda item: item[0])
    working["_trial_num"] = sort_keys.map(lambda item: item[1])
    working["_step_num"] = pd.to_numeric(working["Step"], errors="raise")
    working = working.sort_values(["_day_num", "_trial_num", "_step_num"], kind="mergesort")
    working = working.drop(columns=["_day_num", "_trial_num", "_step_num"])
    working = working.reset_index(drop=True)
    if "Unnamed: 0" in working.columns:
        working = working.drop(columns=["Unnamed: 0"])
    working.insert(0, "Unnamed: 0", range(len(working)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    working.to_pickle(output_path)
    return {
        "file": input_path.name,
        "rows": int(len(working)),
        "columns": int(len(working.columns)),
        "output": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """批量规范化 frame data 并写出 JSON 报告。"""

    args = parse_args()
    input_paths = sorted(args.input_dir.glob("*.pkl"))
    if not input_paths:
        raise FileNotFoundError(f"没有找到 frame data：{args.input_dir}")

    summaries = [normalize_one_file(path, args.output_dir / path.name) for path in input_paths]
    report = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "file_count": len(summaries),
        "total_rows": sum(item["rows"] for item in summaries),
        "summaries": summaries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"file_count": report["file_count"], "total_rows": report["total_rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
