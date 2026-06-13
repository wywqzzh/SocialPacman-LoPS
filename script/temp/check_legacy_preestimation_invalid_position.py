#!/usr/bin/env python3
"""检查旧版 PreEstimation_fmri 是否会在指定异常位置报错。

该临时脚本只用于本轮深度验证中定位问题：从失败的 corrected tile 文件中
截取包含异常 Pacman 位置的小片段，按旧脚本需要的空 fruit 字段补齐后，
调用旧版 ``PreEstimation_fmri._individualEstimation``。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def patch_numpy_array(old_module: Any) -> None:
    """兼容旧代码在新版 numpy 下可能出现的 ragged array 行为。"""

    original_array = old_module.np.array

    def compatible_array(*args: Any, **kwargs: Any) -> Any:
        """原始 np.array 失败且是 ragged 输入时回退到 dtype=object。"""

        try:
            return original_array(*args, **kwargs)
        except ValueError as exc:
            if "setting an array element with a sequence" not in str(exc) or "dtype" in kwargs:
                raise
            patched_kwargs = dict(kwargs)
            patched_kwargs["dtype"] = object
            return original_array(*args, **patched_kwargs)

    old_module.np.array = compatible_array


def build_probe_data(input_path: Path, target: tuple[int, int], before: int, after: int) -> pd.DataFrame:
    """从 corrected tile 文件中截取包含目标位置的片段。

    输入语义：input_path 是当前流程失败的 corrected tile 文件。
    输出语义：返回包含目标行前后若干行的 DataFrame，并补齐旧脚本需要的空 fruit 字段。
    关键约束：不修改坐标和地图状态，只补语义为空的兼容列。
    """

    data = pd.read_pickle(input_path).reset_index(drop=True)
    positions = data["pacmanPos"].map(lambda value: tuple(int(x) for x in value))
    indices = list(np.where(positions == target)[0])
    if not indices:
        raise ValueError(f"未在 {input_path} 中找到目标位置 {target}")
    start = max(0, indices[0] - before)
    end = min(len(data), indices[0] + after + 1)
    probe = data.iloc[start:end].copy().reset_index(drop=True)
    if "fruitPos" not in probe.columns:
        probe["fruitPos"] = np.nan
    if "fruitType" not in probe.columns:
        probe["fruitType"] = np.nan
    return probe


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--constant-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-x", type=int, default=13)
    parser.add_argument("--target-y", type=int, default=16)
    parser.add_argument("--before", type=int, default=2)
    parser.add_argument("--after", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """运行旧版 probe 并把异常类型写入报告。"""

    args = parse_args()
    if str(args.legacy_root) not in sys.path:
        sys.path.insert(0, str(args.legacy_root))

    from Utils.FileUtils_fmri import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath
    from Behavior_Analysis.HierarchicalModel import PreEstimation_fmri as old

    patch_numpy_array(old)
    target = (args.target_x, args.target_y)
    probe = build_probe_data(args.input_path, target, args.before, args.after)
    adjacent_data = readAdjacentMap(str(args.constant_dir / "adjacent_map_fmri.csv"))
    locs_df = readLocDistance(str(args.constant_dir / "dij_distance_map_fmri.csv"))
    adjacent_path = readAdjacentPath(str(args.constant_dir / "dij_distance_map_fmri.csv"))
    reward_amount = readRewardAmount()

    report: dict[str, Any] = {
        "legacy_root": str(args.legacy_root),
        "input_path": str(args.input_path),
        "target": list(target),
        "probe_rows": int(len(probe)),
        "probe_positions": [repr(value) for value in probe["pacmanPos"].tolist()],
        "status": "unknown",
    }
    try:
        old._individualEstimation(probe, adjacent_data, locs_df, adjacent_path, reward_amount, str(args.input_path))
        report["status"] = "passed"
    except Exception as exc:  # noqa: BLE001 - 临时验证脚本需要记录旧代码任意异常。
        report["status"] = "failed"
        report["exception_type"] = type(exc).__name__
        report["exception_message"] = str(exc)
        report["traceback_tail"] = traceback.format_exc().splitlines()[-20:]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "exception_type", "exception_message") if key in report}, ensure_ascii=False))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
