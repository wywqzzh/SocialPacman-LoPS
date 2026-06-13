#!/usr/bin/env python3
"""从人类 fMRI frame data 生成 tile data 和 corrected tile data。

本脚本是 ``DataPreProcessHuman.py`` 的当前项目重构入口，只负责数据处理，
不依赖旧项目代码或旧项目数据目录。输入、输出路径由命令行参数控制，默认
指向当前仓库 ``data/human_tile_data_preprocess`` 下的数据目录。
"""

from __future__ import annotations

import argparse
import ast
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_ROOT = PROJECT_ROOT / "pipeline_data"
TUNNEL_LEFT = (0, 18)
TUNNEL_RIGHT = (30, 18)
INVALID_PACMAN_POSITIONS = {(-1, 18), (31, 18)}
DIRECTION_NAMES = ("left", "right", "up", "down")
TWO_GHOST_DROP_COLUMNS = (
    "ghost3Pos",
    "ghost4Pos",
    "ifscared3",
    "ifscared4",
    "g3pX",
    "g3pY",
    "g3Dir",
    "g3ModeR",
    "g3Scared",
    "g3Frame",
    "g4pX",
    "g4pY",
    "g4Dir",
    "g4ModeR",
    "g4Scared",
    "g4Frame",
)
GHOST_POSITION_FIXES = {
    (14, 20): (14, 19),
    (15, 20): (15, 19),
    (16, 20): (16, 19),
}
BASE_TILE_COLUMNS = [
    "frame_id",
    "DayTrial",
    "game_id",
    "Step",
    "pacmanPos",
    "ghost1Pos",
    "ghost2Pos",
    "ifscared1",
    "ifscared2",
    "beans",
    "energizers",
]
CORRECTED_TILE_COLUMNS = BASE_TILE_COLUMNS + ["action_dir", "available_dir"]


def parse_grid_position(value: Any) -> tuple[int, int]:
    """解析 Pacman 或 ghost 的格点坐标。

    输入语义：value 可以是旧数据中的 ``"(x, y)"`` 字符串，也可以已经是
    tuple/list 或 numpy 标量组成的坐标。
    输出语义：返回 ``(x, y)`` 整数 tuple。
    关键约束：只接受长度为 2 的坐标字面量，不使用 ``eval`` 执行任意代码。
    """

    if isinstance(value, tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, list) and len(value) == 2:
        return int(value[0]), int(value[1])
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
        raise ValueError(f"无法解析坐标：{value!r}")
    return int(parsed[0]), int(parsed[1])


def load_adjacent_map(path: Path) -> dict[tuple[int, int], dict[str, tuple[int, int] | float]]:
    """读取 fMRI 地图四方向邻接表。

    输入语义：path 指向包含 ``pos/left/right/up/down`` 列的 CSV。
    输出语义：返回位置到四方向相邻位置的字典；不可走方向用 ``np.nan`` 表示。
    关键约束：保留旧流程对 tunnel 两端的邻接补丁，确保 available_dir 与动态拟合一致。
    """

    adjacent_frame = pd.read_csv(path)
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]] = {}
    for _, row in adjacent_frame.iterrows():
        position = parse_grid_position(row["pos"])
        adjacent_map[position] = {}
        for direction in DIRECTION_NAMES:
            value = row[direction]
            adjacent_map[position][direction] = np.nan if pd.isna(value) else parse_grid_position(value)

    adjacent_map.setdefault((0, 18), {})
    adjacent_map.setdefault((30, 18), {})
    adjacent_map[(0, 18)].update({"left": (30, 18), "right": (1, 18), "up": np.nan, "down": np.nan})
    adjacent_map[(30, 18)].update({"left": (29, 18), "right": (0, 18), "up": np.nan, "down": np.nan})
    return adjacent_map


def normalize_tunnel_position(position: tuple[int, int]) -> tuple[int, int]:
    """把 tunnel 边界位置映射到旧拟合逻辑使用的内部位置。

    输入语义：position 是当前 Pacman 坐标。
    输出语义：返回用于邻接合法性判断的坐标。
    关键约束：该规则与 dynamic_strategy_fitting 中 available_dir 的历史语义一致。
    """

    if position in {(-1, 18), (0, 18)}:
        return (1, 18)
    if position in {(31, 18), (30, 18)}:
        return (29, 18)
    return position


def is_available_direction(
    position: tuple[int, int],
    direction: Any,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
) -> bool:
    """判断当前真实动作是否是合法可走方向。

    输入语义：position 是当前 Pacman 坐标，direction 是本行 ``action_dir``。
    输出语义：direction 缺失、不是四方向字符串或对应墙方向时返回 False。
    关键约束：该字段是单个 bool，不是四方向可用数组。
    """

    if not isinstance(direction, str) or direction not in DIRECTION_NAMES:
        return False
    adjacent_position = adjacent_map[normalize_tunnel_position(position)]
    adjacent_value = adjacent_position[direction]
    return adjacent_value is not None and not isinstance(adjacent_value, float)


def is_empty_position_marker(value: Any) -> bool:
    """判断位置字段是否是空列表。

    输入语义：value 通常来自 ghost 位置字段，可能是 list、tuple 或字符串。
    输出语义：空列表返回 True，其它坐标返回 False。
    关键约束：旧流程只在 ``len(curPosition) == 0`` 时跳过连续性检查。
    """

    if isinstance(value, (list, tuple)):
        return len(value) == 0
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, str):
        return value.strip() == "[]"
    return False


def infer_move_direction(previous: tuple[int, int], current: tuple[int, int]) -> str | float:
    """根据相邻 Pacman 坐标计算移动方向。

    输入语义：previous/current 是相邻 tile 点的 Pacman 坐标。
    输出语义：返回 ``left/right/up/down``，若坐标未移动则返回 ``np.nan``。
    关键约束：保留迷宫横向 tunnel 的特殊方向规则。
    """

    if previous == TUNNEL_LEFT and current == TUNNEL_RIGHT:
        offset = (-1, 0)
    elif previous == TUNNEL_RIGHT and current == TUNNEL_LEFT:
        offset = (1, 0)
    else:
        offset = (current[0] - previous[0], current[1] - previous[1])

    directions: dict[tuple[int, int], str | float] = {
        (-1, 0): "left",
        (1, 0): "right",
        (0, -1): "up",
        (0, 1): "down",
        (0, 0): np.nan,
    }
    if offset not in directions:
        raise ValueError(f"无法从非相邻坐标推断动作方向：{previous!r} -> {current!r}")
    return directions[offset]


def is_continuous_pacman_step(previous: tuple[int, int], current: tuple[int, int]) -> bool:
    """判断两个 Pacman 坐标是否在 tile 轨迹上连续。

    输入语义：previous/current 是相邻抽样点坐标。
    输出语义：相邻、原地横向 tunnel 特例均返回 True。
    关键约束：与旧流程一致，横向 tunnel 只接受 ``0 <-> 30``、y=18 的跨越。
    """

    if previous == TUNNEL_LEFT and current == TUNNEL_RIGHT:
        return True
    if previous == TUNNEL_RIGHT and current == TUNNEL_LEFT:
        return True
    return abs(previous[0] - current[0]) + abs(previous[1] - current[1]) == 1


def sample_tile_rows_from_frame_data(subject_frame_data: pd.DataFrame, sample_rate: int = 25) -> pd.DataFrame:
    """从单个 frame DataFrame 抽取 tile-level 数据。

    输入语义：subject_frame_data 是一个被试/session 的逐帧数据，必须包含 ``DayTrial``
    和 ``pacmanPos``。
    输出语义：返回抽样后的 tile DataFrame，保留原字段和旧流程的行顺序。
    关键约束：每个 ``DayTrial`` 每隔 sample_rate 取一帧，并强制保留最后一帧。
    """

    if sample_rate <= 0:
        raise ValueError("--sample-rate 必须大于 0。")
    required = {"frame_id", "DayTrial", "pacmanPos"}
    missing = required - set(subject_frame_data.columns)
    if missing:
        raise ValueError(f"frame data 缺少必要字段：{sorted(missing)}")

    sampled_groups: list[pd.DataFrame] = []
    # pandas groupby 默认 sort=True；旧脚本没有显式关闭排序，这里保留同样行为。
    for _, trial_frame_rows in subject_frame_data.groupby("DayTrial"):
        sample_indices = np.arange(0, trial_frame_rows.shape[0], sample_rate)
        if sample_indices[-1] != trial_frame_rows.shape[0] - 1:
            sample_indices = np.append(sample_indices, trial_frame_rows.shape[0] - 1)
        trial_tile_rows = trial_frame_rows.iloc[sample_indices].copy()
        trial_tile_rows.reset_index(drop=True, inplace=True)
        sampled_groups.append(trial_tile_rows)

    subject_tile_data = pd.concat(sampled_groups, axis=0)
    subject_tile_data.reset_index(drop=True, inplace=True)

    # 删除旧流程认为无效的 tunnel 外位置，避免后续方向计算出现非地图坐标。
    pacman_positions = subject_tile_data["pacmanPos"].map(parse_grid_position)
    invalid_mask = pacman_positions.map(lambda position: position in INVALID_PACMAN_POSITIONS)
    if invalid_mask.any():
        subject_tile_data = subject_tile_data.drop(subject_tile_data.index[invalid_mask.to_numpy()])
        subject_tile_data.reset_index(drop=True, inplace=True)
    return normalize_tile_schema(subject_tile_data, include_action_fields=False)


def normalize_tile_schema(data: pd.DataFrame, *, include_action_fields: bool) -> pd.DataFrame:
    """按标准 tile schema 规整列顺序和基础 dtype。

    输入语义：data 是 tile 或 corrected tile 表；include_action_fields 表示是否要求动作字段。
    输出语义：返回列顺序稳定、id/状态字段 dtype 收紧后的 DataFrame。
    关键约束：插入 Series 后 pandas 可能把整数列放宽成 object，本函数在保存前统一收紧。
    """

    columns = CORRECTED_TILE_COLUMNS if include_action_fields else BASE_TILE_COLUMNS
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"tile 数据缺少标准字段：{missing}")

    result = data.loc[:, columns].copy()
    result["frame_id"] = pd.to_numeric(result["frame_id"], errors="raise").astype("int64")
    result["DayTrial"] = result["DayTrial"].astype(str)
    result["game_id"] = result["game_id"].astype(str)
    result["Step"] = pd.to_numeric(result["Step"], errors="raise").astype("int64")
    result["ifscared1"] = pd.to_numeric(result["ifscared1"], errors="raise").astype("int8")
    result["ifscared2"] = pd.to_numeric(result["ifscared2"], errors="raise").astype("int8")
    if include_action_fields:
        result["available_dir"] = result["available_dir"].astype(bool)
    return result


def sample_tile_rows_for_all_subjects(frame_dir: Path, tile_dir: Path, sample_rate: int = 25) -> list[dict[str, Any]]:
    """批量从 frame data 目录生成 tile data。

    输入语义：frame_dir 是扁平的 pkl 目录，tile_dir 是输出目录。
    输出语义：写出同名 pkl 文件，并返回每个文件的处理摘要。
    关键约束：文件排序只用于稳定运行日志，不改变单个文件内部的分组顺序。
    """

    if not frame_dir.is_dir():
        raise FileNotFoundError(f"frame data 输入目录不存在：{frame_dir}")
    input_paths = sorted(frame_dir.glob("*.pkl"))
    if not input_paths:
        raise FileNotFoundError(f"frame data 输入目录没有 pkl 文件：{frame_dir}")

    tile_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for frame_path in input_paths:
        subject_frame_data = pd.read_pickle(frame_path)
        subject_tile_data = sample_tile_rows_from_frame_data(subject_frame_data, sample_rate=sample_rate)
        output_path = tile_dir / frame_path.name
        subject_tile_data.to_pickle(output_path)
        summaries.append(
            {
                "file": frame_path.name,
                "frame_rows": int(len(subject_frame_data)),
                "tile_rows": int(len(subject_tile_data)),
                "output": str(output_path),
            }
        )
        print(frame_path.name)
    return summaries


def repair_known_ghost_position_errors(trial_tile_rows: pd.DataFrame) -> None:
    """就地修正旧数据中 ghost 位置记录错误。

    输入语义：trial_tile_rows 是某个 ``DayTrial`` 的 tile 数据分组。
    输出语义：直接修改 trial_tile_rows 中 ghost1Pos/ghost2Pos 的错误坐标。
    关键约束：只修正三个旧流程明确列出的坐标，不推断其它异常情况。
    """

    for row_label in trial_tile_rows.index:
        for column in ("ghost1Pos", "ghost2Pos"):
            if column not in trial_tile_rows.columns:
                continue
            value = trial_tile_rows.at[row_label, column]
            position = normalize_optional_ghost_fix_position(value)
            if position in GHOST_POSITION_FIXES:
                trial_tile_rows.at[row_label, column] = GHOST_POSITION_FIXES[position]


def normalize_optional_ghost_fix_position(value: Any) -> tuple[int, int] | None:
    """把 ghost 位置字段规范成可用于错误修正表查询的坐标。

    输入语义：value 可以是 tuple/list 坐标、空列表标记或浮点缺失值。
    输出语义：合法坐标返回 ``(x, y)``；空 ghost、缺失值或不可解析值返回 None。
    关键约束：该函数只修复当前新 frame 中出现的 list/tuple 表示；字符串坐标保持旧流程行为，
    避免改变既有字符串格式输入的 corrected tile 结果。
    """

    if value is None or is_empty_position_marker(value):
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, np.ndarray) and value.size == 2:
        flattened = value.reshape(-1)
        return int(flattened[0]), int(flattened[1])
    return None


def restore_missing_pacman_path_rows(trial_tile_rows: pd.DataFrame, trial_frame_rows: pd.DataFrame) -> pd.DataFrame:
    """为一个 ``DayTrial`` 插入被抽样遗漏的 Pacman 中间位置。

    输入语义：trial_tile_rows 是单个 ``DayTrial`` 的抽样数据，trial_frame_rows 是同一
    ``DayTrial`` 的逐帧数据。
    输出语义：返回插入中间点后的 DataFrame。
    关键约束：插入候选只来自前后两个 tile 点之间的 frame 区间，并跳过
    ``(-1, 18)``、``(31, 18)`` 以及已插入过的位置。
    """

    repair_known_ghost_position_errors(trial_tile_rows)
    rows_to_insert: list[tuple[pd.Series, int]] = []
    trial_frame_ids = list(trial_frame_rows["frame_id"])
    trial_frame_by_id = trial_frame_rows.set_index("frame_id", drop=False)

    for row_offset in range(len(trial_tile_rows)):
        if row_offset == 0 or row_offset == len(trial_tile_rows) - 1:
            continue

        previous_position_value = trial_tile_rows["pacmanPos"].iloc[row_offset - 1]
        current_position_value = trial_tile_rows["pacmanPos"].iloc[row_offset]
        if is_empty_position_marker(current_position_value):
            continue

        previous_pacman_position = parse_grid_position(previous_position_value)
        current_pacman_position = parse_grid_position(current_position_value)
        if is_continuous_pacman_step(previous_pacman_position, current_pacman_position):
            continue

        # 走到这里说明相邻两个 tile 抽样点之间 Pacman 不是相邻移动。
        # 旧流程的解释是：25 帧抽样可能跳过了中间格子，因此要回到同一个
        # DayTrial 的逐帧 frame data 中，在这两个 tile 点之间找被漏抽的帧。
        previous_frame_id = trial_tile_rows["frame_id"].iloc[row_offset - 1]
        current_frame_id = trial_tile_rows["frame_id"].iloc[row_offset]
        start_frame_offset = trial_frame_ids.index(previous_frame_id)
        end_frame_offset = trial_frame_ids.index(current_frame_id)

        # 注意这里不包含 current_frame_id 对应帧，保持旧实现的半开区间
        # 行为：候选只来自 previous tile 之后、current tile 之前的 frame。
        candidate_frame_ids = trial_frame_ids[start_frame_offset:end_frame_offset]

        inserted_positions: list[tuple[int, int]] = []
        inserted_frame_row: pd.Series | None = None
        for frame_id in candidate_frame_ids:
            candidate_position = parse_grid_position(trial_frame_by_id.at[frame_id, "pacmanPos"])
            # 起点和终点已经存在于 tile 数据中，不能重复插入。
            if candidate_position in (previous_pacman_position, current_pacman_position):
                continue
            # (-1, 18)/(31, 18) 是横向 tunnel 外的过渡坐标，旧流程不把它们
            # 作为 tile 轨迹点保存。
            if candidate_position in INVALID_PACMAN_POSITIONS:
                continue
            # 同一个 frame 区间里可能多帧停在同一格，只插入第一次出现的格点。
            if candidate_position in inserted_positions:
                continue
            # 旧实现会持续扫描整个区间，并把最后一个符合条件的候选帧作为
            # 插入行；这里保留这个行为，不能提前 break。
            inserted_frame_row = copy.deepcopy(trial_frame_by_id.loc[frame_id])
            inserted_positions.append(candidate_position)

        if inserted_frame_row is not None:
            if len(inserted_positions) > 1:
                print("=================" * 10)
            # groupby 后的分组保留原 DataFrame 标签；旧实现用 row_offset 加上
            # 分组起始标签，得到插入位置标签，而不是简单的位置序号。
            insert_label = row_offset + trial_tile_rows.index[0]
            rows_to_insert.append((inserted_frame_row, insert_label))

    corrected_trial_rows = copy.deepcopy(trial_tile_rows)
    for inserted_frame_row, insert_label in rows_to_insert:
        # 旧代码用 DataFrame.append 链式插入；这里用等价的 concat，以兼容当前 pandas。
        corrected_trial_rows = pd.concat(
            [
                corrected_trial_rows.loc[: insert_label - 1],
                inserted_frame_row.to_frame().T,
                corrected_trial_rows.loc[insert_label:],
            ]
        )
    return corrected_trial_rows


def add_action_fields(
    trial_tile_rows: pd.DataFrame,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
) -> pd.DataFrame:
    """根据修正后的 Pacman 坐标生成 action_dir 和 available_dir。

    输入语义：trial_tile_rows 是一个 ``DayTrial`` 修正后的 tile 数据。
    输出语义：返回同一个 DataFrame，并追加/覆盖 ``action_dir`` 与 ``available_dir``。
    关键约束：action_dir 表示当前行走到下一行的动作；最后一行没有下一步，记为 ``np.nan``。
    """

    pacman_positions = [parse_grid_position(value) for value in trial_tile_rows["pacmanPos"]]
    actions: list[str | float] = []
    for index in range(len(pacman_positions) - 1):
        actions.append(infer_move_direction(pacman_positions[index], pacman_positions[index + 1]))
    actions.append(np.nan)

    trial_tile_rows["action_dir"] = actions
    trial_tile_rows["available_dir"] = [
        is_available_direction(position, action, adjacent_map)
        for position, action in zip(pacman_positions, actions)
    ]
    return trial_tile_rows


def correct_subject_tile_data(
    subject_tile_data: pd.DataFrame,
    subject_frame_data: pd.DataFrame,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
) -> pd.DataFrame:
    """修正单个被试/session 的 tile data。

    输入语义：subject_tile_data 来自抽帧结果，subject_frame_data 是同一文件的原始逐帧数据。
    输出语义：返回 corrected tile data，包含 ``action_dir`` 和 ``available_dir``。
    关键约束：按旧流程在每个 ``DayTrial`` 内独立修正，最后保持分组拼接顺序。
    """

    if "frame_id" not in subject_tile_data.columns:
        raise ValueError("tile data 缺少 frame_id，无法回到原始 frame 区间补点。")

    working_tile = subject_tile_data.copy(deep=True)
    working_frame = subject_frame_data.copy(deep=True)
    working_tile.reset_index(drop=True, inplace=True)
    working_frame.reset_index(drop=True, inplace=True)

    # 当前主流程只分析 two-ghost 数据；从 tile 阶段开始删除 3/4 鬼字段，
    # 后续模块不再携带空的三鬼/四鬼占位列。
    working_tile.drop(columns=[column for column in TWO_GHOST_DROP_COLUMNS if column in working_tile.columns], inplace=True)
    working_frame.drop(columns=[column for column in TWO_GHOST_DROP_COLUMNS if column in working_frame.columns], inplace=True)

    corrected_groups: list[pd.DataFrame] = []
    for day_trial_id, trial_tile_rows in working_tile.groupby("DayTrial"):
        trial_frame_rows = working_frame[working_frame.DayTrial == day_trial_id]
        corrected_trial_rows = restore_missing_pacman_path_rows(trial_tile_rows, trial_frame_rows)
        corrected_trial_rows = add_action_fields(corrected_trial_rows, adjacent_map)
        corrected_groups.append(corrected_trial_rows)

    if not corrected_groups:
        return working_tile.iloc[0:0].copy()

    corrected_tile_data = pd.concat(corrected_groups)
    corrected_tile_data.reset_index(drop=True, inplace=True)
    return normalize_tile_schema(corrected_tile_data, include_action_fields=True)


def correct_tile_data_for_all_subjects(
    tile_dir: Path,
    frame_dir: Path,
    corrected_dir: Path,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
) -> list[dict[str, Any]]:
    """批量生成 corrected tile data。

    输入语义：tile_dir 和 frame_dir 必须包含同名 pkl 文件。
    输出语义：在 corrected_dir 写出同名 pkl，并返回每个文件摘要。
    关键约束：每个文件独立处理，不跨被试共享状态。
    """

    if not tile_dir.is_dir():
        raise FileNotFoundError(f"tile data 输入目录不存在：{tile_dir}")
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"frame data 输入目录不存在：{frame_dir}")
    tile_paths = sorted(tile_dir.glob("*.pkl"))
    if not tile_paths:
        raise FileNotFoundError(f"tile data 输入目录没有 pkl 文件：{tile_dir}")

    corrected_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for tile_path in tile_paths:
        frame_path = frame_dir / tile_path.name
        if not frame_path.exists():
            raise FileNotFoundError(f"找不到 {tile_path.name} 对应的 frame data：{frame_path}")
        subject_tile_data = pd.read_pickle(tile_path)
        subject_frame_data = pd.read_pickle(frame_path)
        corrected_tile_data = correct_subject_tile_data(subject_tile_data, subject_frame_data, adjacent_map)
        output_path = corrected_dir / tile_path.name
        corrected_tile_data.to_pickle(output_path)
        summaries.append(
            {
                "file": tile_path.name,
                "tile_rows": int(len(subject_tile_data)),
                "corrected_rows": int(len(corrected_tile_data)),
                "output": str(output_path),
            }
        )
        print(tile_path.name)
    return summaries


def run_human_tile_data_preprocess(
    frame_dir: Path,
    tile_dir: Path,
    corrected_dir: Path,
    adjacent_map_path: Path,
    sample_rate: int = 25,
) -> dict[str, Any]:
    """执行完整的人类 tile 数据预处理流程。

    输入语义：frame_dir 是当前仓库内的 frame data pkl 目录。
    输出语义：先写 tile_dir，再写 corrected_dir，并返回整体摘要。
    关键约束：这个函数不读取旧项目路径，也不修改输入 frame data。
    """

    adjacent_map = load_adjacent_map(adjacent_map_path)
    tile_summaries = sample_tile_rows_for_all_subjects(frame_dir, tile_dir, sample_rate=sample_rate)
    corrected_summaries = correct_tile_data_for_all_subjects(tile_dir, frame_dir, corrected_dir, adjacent_map)
    return {
        "frame_dir": str(frame_dir),
        "tile_dir": str(tile_dir),
        "corrected_dir": str(corrected_dir),
        "sample_rate": sample_rate,
        "file_count": len(corrected_summaries),
        "total_tile_rows": int(sum(item["tile_rows"] for item in corrected_summaries)),
        "total_corrected_rows": int(sum(item["corrected_rows"] for item in corrected_summaries)),
        "tile_summaries": tile_summaries,
        "corrected_summaries": corrected_summaries,
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输出语义：返回包含输入目录、输出目录和抽样率的 argparse 命名空间。
    关键约束：默认路径全部位于当前 LoPS 仓库 ``data`` 下。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-dir", type=Path, default=DEFAULT_PIPELINE_ROOT / "pacman_data/preprocessed_frame_data")
    parser.add_argument("--tile-dir", type=Path, default=DEFAULT_PIPELINE_ROOT / "human_tile_data_preprocess/tile_data")
    parser.add_argument(
        "--corrected-dir",
        type=Path,
        default=DEFAULT_PIPELINE_ROOT / "human_tile_data_preprocess/corrected_tile_data",
    )
    parser.add_argument(
        "--adjacent-map",
        type=Path,
        default=DEFAULT_PIPELINE_ROOT / "constant_data/adjacent_map_fmri.csv",
        help="用于计算 available_dir 的 fMRI 邻接表。",
    )
    parser.add_argument("--sample-rate", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    """运行完整预处理流程并打印摘要。"""

    args = parse_args()
    summary = run_human_tile_data_preprocess(
        frame_dir=args.frame_dir,
        tile_dir=args.tile_dir,
        corrected_dir=args.corrected_dir,
        adjacent_map_path=args.adjacent_map,
        sample_rate=args.sample_rate,
    )
    print("human tile data preprocess 完成")
    print(f"文件数：{summary['file_count']}")
    print(f"tile 总行数：{summary['total_tile_rows']}")
    print(f"corrected tile 总行数：{summary['total_corrected_rows']}")
    print(f"tile 输出目录：{args.tile_dir.resolve()}")
    print(f"corrected 输出目录：{args.corrected_dir.resolve()}")


if __name__ == "__main__":
    main()
