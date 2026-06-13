#!/usr/bin/env python3
"""运行旧版 CorrectUtilityHuman 逻辑的临时验证入口。

该脚本只服务于深度验证流程：读取旧版邻接表工具，按旧版规则把不可走方向的
Q 值改成 ``-np.inf``，并把结果写入指定目录。由于本轮流水线中上一阶段已经把
位置字段规范成 tuple/list，本入口会先把位置字段统一解析成整数坐标，避免旧脚本
对字符串 ``eval`` 的输入假设影响验证。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DIRECTION_NAMES = ("left", "right", "up", "down")
Q_COLUMNS = (
    "global_Q",
    "local_Q",
    "evade_blinky_Q",
    "evade_clyde_Q",
    "evade_ghost3_Q",
    "evade_ghost4_Q",
    "approach_Q",
    "energizer_Q",
    "no_energizer_Q",
)


def parse_position(value: Any) -> tuple[int, int]:
    """把旧版或当前版的位置字段解析成整数坐标。

    输入语义：value 可以是 ``"(x, y)"`` 字符串、tuple 或 list。
    输出语义：返回旧版邻接表可使用的 ``(x, y)`` 坐标。
    关键约束：这里只做验证适配，不改变正式流水线的数据结构。
    """

    if isinstance(value, tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, list) and len(value) == 2:
        return int(value[0]), int(value[1])
    parsed = eval(str(value), {"__builtins__": {}})
    if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
        raise ValueError(f"无法解析位置字段：{value!r}")
    return int(parsed[0]), int(parsed[1])


def is_blocked_adjacent(value: Any) -> bool:
    """判断旧版邻接表中的某个方向是否不可走。

    输入语义：value 是旧版 ``readAdjacentMap`` 返回的方向值，tuple 表示可走，NaN/None
    表示墙或不存在的相邻格。
    输出语义：不可走返回 True。
    关键约束：保持旧版 ``None or float`` 的判断语义，同时兼容 numpy 浮点 NaN。
    """

    return value is None or isinstance(value, (float, np.floating))


def correct_one_file(input_path: Path, output_dir: Path, adjacent_data: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    """按旧版规则修正单个 utility pickle。

    输入语义：input_path 指向上一阶段旧流程产生的 utility 数据。
    输出语义：在 output_dir 写出同名 pickle，并返回处理摘要。
    关键约束：只修改 Q 数组中不可走方向的位置，其余列和值保持原样。
    """

    data = pd.read_pickle(input_path)
    changed_cells = 0
    positions = data["pacmanPos"].map(parse_position).tolist()

    # 逐列复刻旧脚本的处理方式；每个 Q 列中 4 个元素对应 left/right/up/down。
    for column in Q_COLUMNS:
        corrected_values = []
        for position, q_value in zip(positions, data[column]):
            if position not in adjacent_data:
                raise KeyError(f"旧版邻接表中找不到 Pacman 位置：{position}")
            q_array = np.array(q_value, copy=True)
            for direction_index, direction in enumerate(DIRECTION_NAMES):
                if is_blocked_adjacent(adjacent_data[position][direction]):
                    if not np.isneginf(q_array[direction_index]):
                        changed_cells += 1
                    q_array[direction_index] = -np.inf
            corrected_values.append(q_array)
        data[column] = corrected_values

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name
    data.to_pickle(output_path)
    return {"file": input_path.name, "rows": int(len(data)), "changed_cells": int(changed_cells)}


def load_legacy_adjacent_map(legacy_root: Path, adjacent_map: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """调用旧项目的邻接表读取函数。

    输入语义：legacy_root 是旧项目根目录，adjacent_map 是旧版使用的 CSV 文件。
    输出语义：返回旧工具函数生成的邻接字典。
    关键约束：旧工具模块导入时会修改当前工作目录，因此调用后恢复原工作目录。
    """

    original_cwd = os.getcwd()
    sys.path.insert(0, str(legacy_root))
    try:
        from Utils.FileUtils_fmri import readAdjacentMap

        adjacent_data = readAdjacentMap(str(adjacent_map))
    finally:
        os.chdir(original_cwd)
        try:
            sys.path.remove(str(legacy_root))
        except ValueError:
            pass
    return adjacent_data


def parse_args() -> argparse.Namespace:
    """解析旧版 correct utility 临时入口参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adjacent-map", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """批量运行旧版 correct utility 逻辑并写出验证报告。"""

    args = parse_args()
    input_paths = sorted(args.input_dir.glob("*.pkl"))
    if not input_paths:
        raise FileNotFoundError(f"旧版 correct utility 输入目录中没有 pkl：{args.input_dir}")

    adjacent_data = load_legacy_adjacent_map(args.legacy_root, args.adjacent_map)
    worker_count = max(1, min(args.workers, len(input_paths)))
    task = partial(correct_one_file, output_dir=args.output_dir, adjacent_data=adjacent_data)
    if worker_count == 1:
        summaries = [task(path) for path in input_paths]
    else:
        # 每个被试文件独立处理，适合用进程池充分利用服务器 CPU。
        with mp.Pool(processes=worker_count) as pool:
            summaries = pool.map(task, input_paths)

    report = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "file_count": len(summaries),
        "total_rows": int(sum(item["rows"] for item in summaries)),
        "changed_cells": int(sum(item["changed_cells"] for item in summaries)),
        "files": summaries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("file_count", "total_rows", "changed_cells")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
