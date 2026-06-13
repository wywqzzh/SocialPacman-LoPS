#!/usr/bin/env python3
"""运行旧版 PreEstimation_fmri 的临时包装。

该脚本用于深度验证第二阶段：从 corrected tile data 生成 utility data。
旧版脚本会读取 fruit 字段；当前 two-ghost 数据没有 fruit，因此这里仅在
传给旧代码的内存视图中补充空 fruit 列，不改变输入文件本身。
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


POSITION_COLUMNS = ["pacmanPos", "ghost1Pos", "ghost2Pos", "ghost3Pos", "ghost4Pos", "beans", "energizers"]


def patch_numpy_array(old_module: Any) -> None:
    """兼容旧代码在新版 numpy 下的 ragged ghost array 行为。"""

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


def normalize_position_value(value: Any) -> Any:
    """把当前位置字段规范成旧代码历史输入使用的整数坐标。

    输入语义：当前重构流程里坐标可能是 float tuple，例如 ``(15.0, 9.0)``。
    输出语义：坐标 tuple/list 转成整数坐标；坐标列表递归处理；空列表保持空。
    关键约束：只改变数值类型，不改变坐标语义。
    """

    if isinstance(value, tuple) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    if isinstance(value, list):
        if len(value) == 0:
            return []
        if len(value) == 2 and all(isinstance(item, (int, float, np.integer, np.floating)) for item in value):
            return (int(value[0]), int(value[1]))
        return [normalize_position_value(item) for item in value]
    return value


def legacy_input_view(input_path: Path) -> pd.DataFrame:
    """读取 corrected tile 并补齐旧脚本需要的空 fruit 字段。

    输入语义：input_path 是当前流程生成的 corrected tile pickle。
    输出语义：返回传给旧版 PreEstimation 的 DataFrame。
    关键约束：fruit 在当前数据中不存在；补充 NaN 只用于兼容旧代码分支。
    """

    data = pd.read_pickle(input_path).reset_index(drop=True)
    for column in POSITION_COLUMNS:
        if column in data.columns:
            data[column] = data[column].map(normalize_position_value)
    if "fruitPos" not in data.columns:
        data["fruitPos"] = np.nan
    if "fruitType" not in data.columns:
        data["fruitType"] = np.nan
    return data


def process_file(task: tuple[str, str, str, str]) -> dict[str, Any]:
    """处理单个 corrected tile 文件并写出旧版 utility 输出。"""

    legacy_root, input_path, output_dir, constant_dir = task
    legacy_root_path = Path(legacy_root)
    if str(legacy_root_path) not in sys.path:
        sys.path.insert(0, str(legacy_root_path))

    from Utils.FileUtils_fmri import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath
    from Behavior_Analysis.HierarchicalModel import PreEstimation_fmri as old

    patch_numpy_array(old)
    input_path_obj = Path(input_path)
    output_dir_obj = Path(output_dir)
    constant_dir_obj = Path(constant_dir)
    adjacent_data = readAdjacentMap(str(constant_dir_obj / "adjacent_map_fmri.csv"))
    locs_df = readLocDistance(str(constant_dir_obj / "dij_distance_map_fmri.csv"))
    adjacent_path = readAdjacentPath(str(constant_dir_obj / "dij_distance_map_fmri.csv"))
    reward_amount = readRewardAmount()

    frame_data = legacy_input_view(input_path_obj)
    utility_data = old._individualEstimation(
        frame_data,
        adjacent_data,
        locs_df,
        adjacent_path,
        reward_amount,
        str(input_path_obj),
    )
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_obj / f"{input_path_obj.stem}-with_Q.pkl"
    with output_path.open("wb") as file:
        pickle.dump(utility_data, file)
    return {"input_file": input_path_obj.name, "output_file": output_path.name, "row_count": int(utility_data.shape[0])}


def parse_args() -> argparse.Namespace:
    """解析临时旧流程参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--constant-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """批量运行旧版 utility 预估并写出报告。"""

    args = parse_args()
    input_paths = sorted(args.input_dir.glob("*.pkl"))
    if not input_paths:
        raise FileNotFoundError(f"没有找到 corrected tile data：{args.input_dir}")
    tasks = [
        (
            str(args.legacy_root.resolve()),
            str(path.resolve()),
            str(args.output_dir.resolve()),
            str(args.constant_dir.resolve()),
        )
        for path in input_paths
    ]
    if args.workers <= 1:
        summaries = [process_file(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            summaries = list(executor.map(process_file, tasks))

    report = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "file_count": len(summaries),
        "total_rows": sum(item["row_count"] for item in summaries),
        "summaries": summaries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"file_count": report["file_count"], "total_rows": report["total_rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
