#!/usr/bin/env python3
"""运行旧版 DataPreProcessHuman 语义的临时包装。

该脚本用于深度验证第一阶段：从 frame data 生成 tile data 和 corrected
tile data。旧脚本读取的是字符串形式坐标，而当前验证输入里坐标通常已经是
tuple/list；因此这里先构造旧脚本输入视图，再执行旧版抽样、插点和方向修正
逻辑。输出只用于与当前重构流程做语义比较。
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


POSITION_COLUMNS = ["pacmanPos", "ghost1Pos", "ghost2Pos", "ghost3Pos", "ghost4Pos", "beans", "energizers"]
GHOST_FIXES = {(14, 20): (14, 19), (15, 20): (15, 19), (16, 20): (16, 19)}


def parse_literal(value: Any) -> Any:
    """解析旧版字符串字段。

    输入语义：value 可以是字符串字面量，也可以已经是 tuple/list/NaN。
    输出语义：尽量返回 Python 原生结构；无法解析时返回原值。
    关键约束：使用 ``ast.literal_eval``，不执行任意表达式。
    """

    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def to_legacy_text(value: Any) -> Any:
    """把当前结构化字段转成旧脚本使用的字符串字段。

    输入语义：坐标、坐标列表保存在 tuple/list/ndarray 中。
    输出语义：tuple/list 转成 ``repr`` 字符串；空值保持 NaN/None。
    关键约束：只转换旧脚本会 ``eval`` 的位置字段，其它数值列不改。
    """

    if value is None:
        return value
    if isinstance(value, float) and np.isnan(value):
        return value
    if isinstance(value, np.ndarray):
        return repr(value.tolist())
    if isinstance(value, (tuple, list)):
        return repr(value)
    return value


def legacy_frame_view(frame_data: pd.DataFrame) -> pd.DataFrame:
    """构造旧 DataPreProcessHuman 可读取的 frame data 视图。

    输入语义：frame_data 是当前验证输入，通常包含 tuple/list 结构字段。
    输出语义：返回字段值风格接近旧 ``fmriFrameData`` 的 DataFrame。
    关键约束：fruit 在当前数据中不存在；为兼容后续旧脚本，只补空列，不引入
    任何奖励语义。
    """

    data = frame_data.copy(deep=True)
    for column in POSITION_COLUMNS:
        if column in data.columns:
            data[column] = data[column].map(to_legacy_text)
    if "fruitPos" not in data.columns:
        data["fruitPos"] = np.nan
    if "fruitType" not in data.columns:
        data["fruitType"] = np.nan
    return data


def get_dir(pos1: tuple[int, int], pos2: tuple[int, int]) -> str | float:
    """复现旧版 getDir：根据相邻 Pacman 坐标计算方向。"""

    if pos1 == (0, 18) and pos2 == (30, 18):
        offset = (-1, 0)
    elif pos1 == (30, 18) and pos2 == (0, 18):
        offset = (1, 0)
    else:
        offset = (pos2[0] - pos1[0], pos2[1] - pos1[1])
    return {(-1, 0): "left", (1, 0): "right", (0, -1): "up", (0, 1): "down", (0, 0): np.nan}[offset]


def correct_dir(data: pd.DataFrame) -> pd.DataFrame:
    """复现旧版 CorrectDir：重算 pacman_dir 列。"""

    pacman_pos = [parse_literal(value) for value in data["pacmanPos"]]
    directions: list[str | float] = [np.nan]
    for index in range(1, len(data)):
        directions.append(get_dir(tuple(pacman_pos[index - 1]), tuple(pacman_pos[index])))
    data["pacman_dir"] = directions
    return data


def correct_position(data: pd.DataFrame, frame_data: pd.DataFrame) -> pd.DataFrame:
    """复现旧版 CorrectPosition：修正 ghost 坐标并插入漏采 Pacman 点。"""

    keys = ["ghost1Pos", "ghost2Pos", "ghost3Pos", "ghost4Pos"]
    new_series_list: list[list[Any]] = []
    for row_offset in range(len(data)):
        # 旧逻辑会直接在分组内尝试修正三个 ghost home 门口异常坐标。
        for key in keys:
            value = parse_literal(data[key].iloc[row_offset])
            if isinstance(value, (tuple, list)) and tuple(value) in GHOST_FIXES:
                data.loc[data.index[row_offset], key] = repr(GHOST_FIXES[tuple(value)])

        if row_offset == 0 or row_offset == len(data) - 1:
            continue

        previous_position = parse_literal(data["pacmanPos"].iloc[row_offset - 1])
        current_position = parse_literal(data["pacmanPos"].iloc[row_offset])
        if len(current_position) == 0:
            continue
        previous_position = tuple(previous_position)
        current_position = tuple(current_position)

        if abs(previous_position[0] - current_position[0]) + abs(previous_position[1] - current_position[1]) == 1:
            continue
        if previous_position == (0, 18) and current_position == (30, 18):
            continue
        if previous_position == (30, 18) and current_position == (0, 18):
            continue

        previous_frame_index = data["frameIndex"].iloc[row_offset - 1]
        current_frame_index = data["frameIndex"].iloc[row_offset]
        all_frame_index = list(frame_data.index)
        index1 = all_frame_index.index(previous_frame_index)
        index2 = all_frame_index.index(current_frame_index)
        frame_index = all_frame_index[index1:index2]

        new_series = None
        inserted_position = []
        for frame_label in frame_index:
            temp_position = parse_literal(frame_data["pacmanPos"].loc[frame_label])
            if (
                temp_position != previous_position
                and temp_position != current_position
                and temp_position != (-1, 18)
                and temp_position != (31, 18)
                and temp_position not in inserted_position
            ):
                new_series = copy.deepcopy(frame_data.loc[frame_label])
                inserted_position.append(copy.deepcopy(temp_position))
        if new_series is not None:
            new_series_list.append([new_series, row_offset + data.index[0]])

    corrected = copy.deepcopy(data)
    for new_series, insert_position in new_series_list:
        new_series["frameIndex"] = copy.deepcopy(new_series["Unnamed: 0"])
        corrected = pd.concat([corrected.loc[: insert_position - 1], new_series.to_frame().T, corrected.loc[insert_position:]])
    return corrected


def extract_tile_from_frame(frame_data: pd.DataFrame, sample_rate: int) -> pd.DataFrame:
    """复现旧版 ExtractTileFromFrame：每个 DayTrial 每 25 帧抽一点并保留末帧。"""

    new_data = []
    for _, group in frame_data.groupby("DayTrial"):
        indices = np.arange(0, group.shape[0], sample_rate)
        if indices[-1] != group.shape[0] - 1:
            indices = np.append(indices, group.shape[0] - 1)
        sampled = group.iloc[indices].copy()
        sampled.reset_index(drop=True, inplace=True)
        new_data.append(sampled)

    tile_data = pd.concat(new_data, axis=0)
    tile_data.reset_index(drop=True, inplace=True)
    pacman_pos = tile_data["pacmanPos"].map(parse_literal)
    invalid = pacman_pos.map(lambda value: tuple(value) in {(-1, 18), (31, 18)})
    if invalid.any():
        tile_data = tile_data.drop(tile_data.index[invalid.to_numpy()])
        tile_data.reset_index(drop=True, inplace=True)
    return tile_data


def correct_tile_data(tile_data: pd.DataFrame, frame_data: pd.DataFrame) -> pd.DataFrame:
    """复现旧版 CorrectTileData：按 DayTrial 插点并重算方向。"""

    data = tile_data.copy(deep=True)
    data.reset_index(drop=True, inplace=True)
    data["frameIndex"] = data["Unnamed: 0"]

    frame = frame_data.copy(deep=True)
    frame.reset_index(drop=True, inplace=True)

    corrected_groups = []
    for day_trial, group in data.groupby("DayTrial"):
        frame_group = frame[frame.DayTrial == day_trial]
        corrected_group = correct_position(group, frame_group)
        corrected_group = correct_dir(corrected_group)
        corrected_groups.append(corrected_group)

    corrected = pd.concat(corrected_groups)
    corrected.reset_index(drop=True, inplace=True)
    return corrected


def process_one_file(input_path: Path, tile_dir: Path, corrected_dir: Path, sample_rate: int) -> dict[str, Any]:
    """处理单个文件，写出旧流程 tile 和 corrected tile。"""

    frame_data = legacy_frame_view(pd.read_pickle(input_path))
    tile_data = extract_tile_from_frame(frame_data, sample_rate)
    corrected_data = correct_tile_data(tile_data, frame_data)

    tile_path = tile_dir / input_path.name
    corrected_path = corrected_dir / input_path.name
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    corrected_path.parent.mkdir(parents=True, exist_ok=True)
    tile_data.to_pickle(tile_path)
    corrected_data.to_pickle(corrected_path)
    print(input_path.name, flush=True)
    return {
        "file": input_path.name,
        "tile_rows": int(len(tile_data)),
        "corrected_rows": int(len(corrected_data)),
    }


def parse_args() -> argparse.Namespace:
    """解析临时旧流程包装参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--tile-dir", type=Path, required=True)
    parser.add_argument("--corrected-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=25)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """批量运行旧版 tile 预处理语义并保存摘要。"""

    args = parse_args()
    input_paths = sorted(args.frame_dir.glob("*.pkl"))
    if not input_paths:
        raise FileNotFoundError(f"没有找到 frame data：{args.frame_dir}")

    worker = partial(
        process_one_file,
        tile_dir=args.tile_dir,
        corrected_dir=args.corrected_dir,
        sample_rate=args.sample_rate,
    )
    if args.workers <= 1:
        summaries = [worker(path) for path in input_paths]
    else:
        with multiprocessing.Pool(processes=min(args.workers, len(input_paths))) as pool:
            summaries = pool.map(worker, input_paths)

    report = {
        "frame_dir": str(args.frame_dir),
        "tile_dir": str(args.tile_dir),
        "corrected_dir": str(args.corrected_dir),
        "file_count": len(summaries),
        "total_tile_rows": sum(item["tile_rows"] for item in summaries),
        "total_corrected_rows": sum(item["corrected_rows"] for item in summaries),
        "summaries": summaries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"file_count": report["file_count"], "total_corrected_rows": report["total_corrected_rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
