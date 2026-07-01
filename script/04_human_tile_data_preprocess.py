#!/usr/bin/env python3
"""从标准 frame data 生成 tile data 和 corrected tile data。

整体流程分成两个清晰层次：

1. ``04_tile_data`` 保存“联合位置变化候选帧”。本阶段读取
   ``03_preprocessed_frame_data/{task}/{session}.pkl``，在每个 trial 内比较
   两个 Pacman（单人数据只有 p1）和两个 ghost 的联合位置状态；如果当前帧
   与上一帧的联合位置完全相同，则删除当前帧，否则保留。
2. ``04_corrected_tile_data`` 在候选帧基础上处理“异步位置切换区间”。真实数据
   中不同对象的 tile 切换可能相差几帧，因此候选帧会在一个短区间内连续出现。
   corrected 阶段以 13 帧为区间，在不破坏“上一条保留行 -> 下一条候选行”的
   全对象位置连续性的前提下删除中间候选帧，然后重新计算 Pacman 的动作方向和
   动作合法性。

这里的 corrected 不再表示旧流程中的补帧；它表示候选帧已经经过异步切换压缩，
并且 Pacman 动作方向已经按压缩后的序列重新计算。本阶段不再修正 ghost 坐标，
坐标修正应在 02 阶段完成。
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
TUNNEL_LEFT = (0, 18)
TUNNEL_RIGHT = (29, 18)
DIRECTION_NAMES = ("left", "right", "up", "down")
ASYNC_TRANSITION_INTERVAL_FRAMES = 13

BASE_TILE_COLUMNS = (
    "frame_id",
    "DayTrial",
    "game_id",
    "Step",
    "p1_pos",
    "p1_mode",
    "p1_alive",
)
OPTIONAL_P2_TILE_COLUMNS = ("p2_pos", "p2_mode", "p2_alive")
TAIL_TILE_COLUMNS = (
    "ghost1Pos",
    "ghost2Pos",
    "ifscared1",
    "ifscared2",
    "beans",
    "energizers",
)
P1_ACTION_COLUMNS = ("p1_action_dir",)
P2_ACTION_COLUMNS = ("p2_action_dir",)
P1_AVAILABLE_COLUMNS = ("p1_available_dir",)
P2_AVAILABLE_COLUMNS = ("p2_available_dir",)


class HumanTileDataPreprocessError(RuntimeError):
    """04 阶段 tile 数据预处理无法继续时抛出的明确异常。"""


def parse_grid_position(value: Any) -> tuple[int, int]:
    """解析 Pacman 或 ghost 的格点坐标。

    输入语义：value 可以是 tuple/list/numpy 数组，也可以是字符串形式的坐标。
    输出语义：返回 ``(x, y)`` 整数 tuple。
    关键约束：只接受长度为 2 的坐标，不使用 eval 执行任意代码。
    """

    if isinstance(value, tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, list) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, np.ndarray) and value.size == 2:
        flattened = value.reshape(-1)
        return int(flattened[0]), int(flattened[1])
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
        raise HumanTileDataPreprocessError(f"无法解析坐标：{value!r}")
    return int(parsed[0]), int(parsed[1])


def infer_move_direction(previous: tuple[int, int], current: tuple[int, int]) -> str | float:
    """根据相邻 tile 点坐标推断 Pacman 的运动方向。

    输入语义：previous/current 是同一玩家在相邻保留行中的坐标。
    输出语义：返回 ``left/right/up/down``；如果该玩家位置未变化或发生死亡、
    重置等非相邻跳转，则返回 ``np.nan``。
    关键约束：横向 tunnel 使用当前修正后的坐标 ``(0,18) <-> (29,18)``；本阶段
    不再补帧，因此不能把非相邻跳转强行解释为一步动作。
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
    return directions.get(offset, np.nan)


def load_adjacent_map(path: Path) -> dict[tuple[int, int], dict[str, tuple[int, int] | float]]:
    """从 map_constants.pkl 读取四方向邻接表。

    输入语义：path 指向 ``script/constant_map/generate_map_constants.py`` 生成的
    ``map_constants.pkl``。
    输出语义：返回以位置 tuple 为键、四方向邻居为值的字典。
    关键约束：邻接表必须来自当前 28x36 地图常量；不可走方向保留为 ``np.nan``。
    """

    if not path.is_file():
        raise FileNotFoundError(f"找不到地图常量文件：{path}")
    constants = pd.read_pickle(path)
    if not isinstance(constants, dict) or "adjacent_map" not in constants:
        raise HumanTileDataPreprocessError(f"地图常量文件缺少 adjacent_map：{path}")

    adjacent_frame = constants["adjacent_map"]
    required_columns = {"pos", *DIRECTION_NAMES}
    missing = required_columns - set(adjacent_frame.columns)
    if missing:
        raise HumanTileDataPreprocessError(f"adjacent_map 缺少字段：{sorted(missing)}")

    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]] = {}
    for _, row in adjacent_frame.iterrows():
        position = parse_grid_position(row["pos"])
        adjacent_map[position] = {}
        for direction in DIRECTION_NAMES:
            value = row[direction]
            adjacent_map[position][direction] = np.nan if pd.isna(value) else parse_grid_position(value)
    return adjacent_map


def is_available_direction(
    position: tuple[int, int],
    direction: Any,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
) -> bool:
    """判断当前动作方向是否是该位置的合法可走方向。

    输入语义：position 是 Pacman 当前位置，direction 是本行到下一行的动作方向。
    输出语义：合法方向返回 True；方向缺失、非四方向或地图上不可走时返回 False。
    关键约束：坐标已经在 02 阶段修正为当前地图常量使用的坐标。
    """

    if not isinstance(direction, str) or direction not in DIRECTION_NAMES:
        return False
    if position not in adjacent_map:
        raise HumanTileDataPreprocessError(f"位置不在地图邻接表中：{position}")
    adjacent_value = adjacent_map[position][direction]
    return adjacent_value is not None and not isinstance(adjacent_value, float)


def has_second_player(data: pd.DataFrame) -> bool:
    """判断当前 DataFrame 是否包含第二个 Pacman。

    输入语义：data 是 03 阶段输出的标准 frame/tile 表。
    输出语义：包含 ``p2_pos`` 时返回 True。
    关键约束：单人数据不会补出空的 ``p2_pos`` 列。
    """

    return "p2_pos" in data.columns


def tile_columns(has_p2: bool, *, include_action_fields: bool) -> tuple[str, ...]:
    """返回当前玩家结构对应的 tile 输出列顺序。

    输入语义：has_p2 表示是否存在第二个 Pacman；include_action_fields 表示是否追加动作列。
    输出语义：返回稳定的列顺序。
    关键约束：单人数据不保存任何 p2 字段。
    """

    columns = BASE_TILE_COLUMNS
    if has_p2:
        columns += OPTIONAL_P2_TILE_COLUMNS
    columns += TAIL_TILE_COLUMNS
    if include_action_fields:
        columns += P1_ACTION_COLUMNS
        columns += P1_AVAILABLE_COLUMNS
        if has_p2:
            columns += P2_ACTION_COLUMNS
            columns += P2_AVAILABLE_COLUMNS
    return columns


def required_frame_columns(has_p2: bool) -> set[str]:
    """返回抽帧所需的输入字段集合。

    输入语义：has_p2 表示输入是否是双人数据。
    输出语义：返回必须存在的列名集合。
    关键约束：联合状态只包含位置字段，不包含 ifscared 字段。
    """

    required = {
        "frame_id",
        "DayTrial",
        "game_id",
        "Step",
        "p1_pos",
        "p1_mode",
        "p1_alive",
        "ghost1Pos",
        "ghost2Pos",
        "ifscared1",
        "ifscared2",
        "beans",
        "energizers",
    }
    if has_p2:
        required.update({"p2_pos", "p2_mode", "p2_alive"})
    return required


def build_joint_position_state(row: pd.Series, has_p2: bool) -> tuple[tuple[int, int], ...]:
    """构造用于抽帧比较的联合位置状态。

    输入语义：row 是单帧数据；has_p2 控制是否纳入第二个 Pacman。
    输出语义：返回由 p1、可选 p2、ghost1、ghost2 坐标组成的 tuple。
    关键约束：只比较位置，不比较 ifscared、beans 或其它状态字段。
    """

    columns = ["p1_pos"]
    if has_p2:
        columns.append("p2_pos")
    columns.extend(["ghost1Pos", "ghost2Pos"])
    return tuple(parse_grid_position(row[column]) for column in columns)


def joint_position_columns(has_p2: bool) -> tuple[str, ...]:
    """返回联合位置状态使用的位置字段顺序。

    输入语义：has_p2 表示当前数据是否包含第二个 Pacman。
    输出语义：返回用于比较和连续性检查的位置字段名。
    关键约束：字段顺序必须与 ``build_joint_position_state`` 完全一致，保证不同
    行之间同一索引总是同一个对象。
    """

    columns = ["p1_pos"]
    if has_p2:
        columns.append("p2_pos")
    columns.extend(["ghost1Pos", "ghost2Pos"])
    return tuple(columns)


def joint_sampling_columns(has_p2: bool) -> tuple[str, ...]:
    """返回 04_tile_data 候选帧抽取使用的联合状态字段。

    输入语义：has_p2 表示当前数据是否包含第二个 Pacman。
    输出语义：返回用于判断“当前帧是否成为候选 tile 行”的字段名。
    关键约束：Pacman 的 mode 必须参与比较，因为死亡/刷新状态可能在位置不变时
    发生；alive 是 mode 的派生字段，不重复参与比较。
    """

    columns = ["p1_pos", "p1_mode"]
    if has_p2:
        columns.extend(["p2_pos", "p2_mode"])
    columns.extend(["ghost1Pos", "ghost2Pos"])
    return tuple(columns)


def joint_mode_columns(has_p2: bool) -> tuple[str, ...]:
    """返回 Pacman mode 字段顺序。

    输入语义：has_p2 表示当前数据是否包含第二个 Pacman。
    输出语义：返回当前数据中需要保护的 mode 字段名。
    关键约束：mode 状态用于避免 corrected 压缩删除只有一帧的死亡/刷新事件。
    """

    columns = ["p1_mode"]
    if has_p2:
        columns.append("p2_mode")
    return tuple(columns)


def has_unique_mode_state(
    previous_row: pd.Series,
    current_row: pd.Series,
    next_row: pd.Series,
    mode_columns: tuple[str, ...],
) -> bool:
    """判断当前候选行是否携带前后都没有的 Pacman mode 状态。

    输入语义：previous_row 是上一条已保留行，current_row 是尝试删除的候选行，
    next_row 是删除后会直接连接的下一条候选行。
    输出语义：如果任一玩家的当前 mode 同时不同于前后两行，则返回 True。
    关键约束：这种一帧独有 mode 常见于死亡/刷新边界，不能被异步位置压缩误删。
    """

    for column in mode_columns:
        previous_mode = int(previous_row[column])
        current_mode = int(current_row[column])
        next_mode = int(next_row[column])
        if current_mode != previous_mode and current_mode != next_mode:
            return True
    return False


def is_same_or_adjacent_position(
    previous: tuple[int, int],
    current: tuple[int, int],
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
) -> bool:
    """判断单个对象的两个位置是否连续。

    输入语义：previous/current 是同一对象在两条保留候选行上的位置。
    输出语义：原地不动或移动到地图邻接点时返回 True。
    关键约束：本函数使用当前地图常量中的邻接表，包含横向 tunnel 和 ghost house
    合法位置；如果位置不在地图中，说明上游坐标修正或地图常量存在问题，直接报错。
    """

    if previous == current:
        return True
    if previous not in adjacent_map:
        raise HumanTileDataPreprocessError(f"上一位置不在地图邻接表中：{previous}")
    if current not in adjacent_map:
        raise HumanTileDataPreprocessError(f"当前位置不在地图邻接表中：{current}")

    # 邻接表中的不可走方向是 np.nan；这里只比较真实 tuple 邻居。
    for neighbor in adjacent_map[previous].values():
        if isinstance(neighbor, tuple) and neighbor == current:
            return True
    return False


def are_joint_positions_continuous(
    previous_state: tuple[tuple[int, int], ...],
    current_state: tuple[tuple[int, int], ...],
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
) -> bool:
    """判断两个联合位置状态之间是否对所有对象都连续。

    输入语义：previous_state/current_state 是按 ``joint_position_columns`` 顺序构造
    的多对象位置 tuple。
    输出语义：每个对象都原地不动或移动一步时返回 True。
    关键约束：删除候选帧前必须通过这个检查，避免压缩后把真实多步移动伪装成
    单步 tile 行为。
    """

    if len(previous_state) != len(current_state):
        raise HumanTileDataPreprocessError("联合位置状态长度不一致，无法判断连续性。")
    return all(
        is_same_or_adjacent_position(previous, current, adjacent_map)
        for previous, current in zip(previous_state, current_state)
    )


def compress_async_transition_rows(
    trial_tile_rows: pd.DataFrame,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
    *,
    interval_frames: int = ASYNC_TRANSITION_INTERVAL_FRAMES,
) -> pd.DataFrame:
    """压缩一个 trial 内由异步位置切换造成的短区间候选帧。

    输入语义：trial_tile_rows 是当前 04 tile data 中某个 trial 的联合位置变化候选帧。
    输出语义：返回删除冗余中间候选帧后的 trial 数据，尚未追加动作字段。
    关键约束：
    - 第一行永远保留，保证每个 trial 有明确起点。
    - 一个异步切换区间从当前区间起始帧开始，覆盖 ``interval_frames`` 帧以内的候选；
      例如默认 13 帧时，起始帧 0 的区间是 0-12，帧 13 已经属于区间外。
    - 删除当前候选行前，必须确认“上一条已保留行 -> 下一条候选行”的所有对象位置
      都连续；如果不连续，当前候选行必须保留，并从下一条候选行开启新区间。
    """

    if interval_frames <= 0:
        raise HumanTileDataPreprocessError(f"异步切换区间必须为正数：{interval_frames}")
    if len(trial_tile_rows) <= 1:
        return trial_tile_rows.copy()

    has_p2 = has_second_player(trial_tile_rows)
    position_columns = joint_position_columns(has_p2)
    mode_columns = joint_mode_columns(has_p2)
    frames = pd.to_numeric(trial_tile_rows["frame_id"], errors="raise").astype("int64").to_numpy()
    if np.any(np.diff(frames) < 0):
        raise HumanTileDataPreprocessError("同一 trial 内 frame_id 不是递增顺序，无法压缩异步切换区间。")

    states = [
        tuple(parse_grid_position(row[column]) for column in position_columns)
        for _, row in trial_tile_rows.iterrows()
    ]

    keep_positions: list[int] = [0]
    previous_kept_position = 0
    interval_start_frame = int(frames[0])
    current_position = 1

    while current_position < len(trial_tile_rows) - 1:
        next_position = current_position + 1
        current_frame = int(frames[current_position])
        next_frame = int(frames[next_position])

        # 如果当前候选行已经落在上一异步区间之外，它自然成为新区间的起点；
        # 后续只判断它和下一行是否还处于同一个短切换区间。
        if current_frame - interval_start_frame >= interval_frames:
            interval_start_frame = current_frame

        # 下一行已经超出当前区间时，当前行是这个区间最后一个候选，必须保留。
        if next_frame - interval_start_frame >= interval_frames:
            if keep_positions[-1] != current_position:
                keep_positions.append(current_position)
            previous_kept_position = current_position
            interval_start_frame = next_frame
            current_position = next_position
            continue

        # 下一行仍在当前 13 帧区间内。只有当跳过当前行后，上一条保留行能够
        # 直接连续到下一行，当前行才是可删除的异步中间帧。
        previous_kept_state = states[previous_kept_position]
        next_state = states[next_position]
        if (
            are_joint_positions_continuous(previous_kept_state, next_state, adjacent_map)
            and not has_unique_mode_state(
                trial_tile_rows.iloc[previous_kept_position],
                trial_tile_rows.iloc[current_position],
                trial_tile_rows.iloc[next_position],
                mode_columns,
            )
        ):
            current_position = next_position
            continue

        # 如果删除当前行会破坏连续性，当前行承载了真实一步移动，必须保留；
        # 下一行作为新区间起点继续向后检查。
        if keep_positions[-1] != current_position:
            keep_positions.append(current_position)
        previous_kept_position = current_position
        interval_start_frame = next_frame
        current_position = next_position

    # 最后一行没有下一行可用于删除判定；它代表 trial 末端状态，必须保留。
    last_position = len(trial_tile_rows) - 1
    if keep_positions[-1] != last_position:
        keep_positions.append(last_position)

    compressed_rows = trial_tile_rows.iloc[keep_positions].copy()
    compressed_rows.reset_index(drop=True, inplace=True)
    return compressed_rows


def sample_tile_rows_from_frame_data(subject_frame_data: pd.DataFrame) -> pd.DataFrame:
    """按联合位置状态变化从单个 session 的 frame data 中抽取 tile 行。

    输入语义：subject_frame_data 是 03 阶段的标准逐帧表。
    输出语义：返回抽帧后的 tile DataFrame，第一帧总是保留，后续仅保留联合位置变化帧。
    关键约束：比较必须在每个 ``DayTrial`` 内独立进行，不能跨 trial 合并状态。
    """

    has_p2 = has_second_player(subject_frame_data)
    missing = sorted(required_frame_columns(has_p2) - set(subject_frame_data.columns))
    if missing:
        raise HumanTileDataPreprocessError(f"frame data 缺少必要字段：{missing}")

    if subject_frame_data.empty:
        empty = subject_frame_data.iloc[0:0].copy()
        return normalize_tile_schema(empty, include_action_fields=False)

    sampling_columns = list(joint_sampling_columns(has_p2))

    # 03 阶段已经把坐标规整成 tuple，mode 也规整成 int8，因此这里可以直接
    # 用 pandas 比较对象列和数值列。每个 trial 的第一帧保留；后续只要任一
    # Pacman 位置、Pacman mode 或 ghost 位置变化，就成为 04_tile_data 候选帧。
    shifted_positions = subject_frame_data.groupby("DayTrial", sort=False)[sampling_columns].shift()
    is_trial_first_row = subject_frame_data.groupby("DayTrial", sort=False).cumcount().eq(0)
    has_position_change = subject_frame_data[sampling_columns].ne(shifted_positions).any(axis=1)
    keep_mask = is_trial_first_row | has_position_change

    subject_tile_data = subject_frame_data.loc[keep_mask].copy()
    subject_tile_data.reset_index(drop=True, inplace=True)
    return normalize_tile_schema(subject_tile_data, include_action_fields=False)


def add_action_fields(
    trial_tile_rows: pd.DataFrame,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
) -> pd.DataFrame:
    """为一个 trial 的 tile 行生成 Pacman 运动方向和合法性字段。

    输入语义：trial_tile_rows 是同一 ``DayTrial`` 的抽帧结果。
    输出语义：返回副本并追加 ``p1_action_dir/p1_available_dir``，双人数据追加
    ``p2_action_dir/p2_available_dir``。
    关键约束：动作表示当前行到下一行的移动；最后一行没有下一步，记为 ``np.nan``。
    """

    has_p2 = has_second_player(trial_tile_rows)
    result = trial_tile_rows.copy()

    player_fields = [
        ("p1_pos", "p1_action_dir", "p1_available_dir"),
        ("p2_pos", "p2_action_dir", "p2_available_dir"),
    ]
    for player_column, action_column, available_column in player_fields:
        if player_column not in result.columns:
            continue
        player_positions = [parse_grid_position(value) for value in result[player_column]]
        actions: list[str | float] = []
        for index in range(len(player_positions) - 1):
            actions.append(infer_move_direction(player_positions[index], player_positions[index + 1]))
        actions.append(np.nan)
        result[action_column] = actions
        result[available_column] = [
            is_available_direction(position, action, adjacent_map)
            for position, action in zip(player_positions, actions)
        ]

    return normalize_tile_schema(result, include_action_fields=True)


def correct_subject_tile_data(
    subject_tile_data: pd.DataFrame,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
    *,
    interval_frames: int = ASYNC_TRANSITION_INTERVAL_FRAMES,
) -> pd.DataFrame:
    """为单个 session 的 tile data 生成 corrected tile data。

    输入语义：subject_tile_data 是联合状态变化抽帧后的数据。
    输出语义：返回先压缩异步位置切换区间、再追加 Pacman 动作字段的 corrected tile data。
    关键约束：本阶段不补帧、不修正 ghost 坐标；动作方向必须在压缩后重新计算。
    """

    corrected_groups: list[pd.DataFrame] = []
    for _, trial_tile_rows in subject_tile_data.groupby("DayTrial", sort=False):
        # corrected 阶段的核心是先清理短时间异步切换造成的候选帧，再基于
        # 压缩后的序列重新计算动作；顺序不能反过来，否则动作会对应旧候选序列。
        compressed_rows = compress_async_transition_rows(
            trial_tile_rows,
            adjacent_map,
            interval_frames=interval_frames,
        )
        corrected_groups.append(add_action_fields(compressed_rows, adjacent_map))

    if not corrected_groups:
        return normalize_tile_schema(subject_tile_data.iloc[0:0].copy(), include_action_fields=True)

    corrected_tile_data = pd.concat(corrected_groups, axis=0)
    corrected_tile_data.reset_index(drop=True, inplace=True)
    return normalize_tile_schema(corrected_tile_data, include_action_fields=True)


def normalize_tile_schema(data: pd.DataFrame, *, include_action_fields: bool) -> pd.DataFrame:
    """按当前单人/双人 schema 规整 tile 表列顺序和基础 dtype。

    输入语义：data 是 tile 或 corrected tile 表；include_action_fields 表示是否要求动作字段。
    输出语义：返回列顺序稳定、id/状态字段 dtype 收紧后的 DataFrame。
    关键约束：单人数据不保存 p2 字段；动作字段保留字符串或 ``np.nan``。
    """

    has_p2 = has_second_player(data)
    columns = tile_columns(has_p2, include_action_fields=include_action_fields)
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise HumanTileDataPreprocessError(f"tile 数据缺少标准字段：{missing}")

    result = data.loc[:, columns].copy()
    result["frame_id"] = pd.to_numeric(result["frame_id"], errors="raise").astype("int64")
    result["DayTrial"] = result["DayTrial"].astype(str)
    result["game_id"] = result["game_id"].astype(str)
    result["Step"] = pd.to_numeric(result["Step"], errors="raise").astype("int64")
    result["p1_mode"] = pd.to_numeric(result["p1_mode"], errors="raise").astype("int8")
    result["p1_alive"] = result["p1_alive"].astype(bool)
    if "p2_mode" in result.columns:
        result["p2_mode"] = pd.to_numeric(result["p2_mode"], errors="raise").astype("int8")
    if "p2_alive" in result.columns:
        result["p2_alive"] = result["p2_alive"].astype(bool)
    result["ifscared1"] = pd.to_numeric(result["ifscared1"], errors="raise").astype("int8")
    result["ifscared2"] = pd.to_numeric(result["ifscared2"], errors="raise").astype("int8")
    for column in ("p1_available_dir", "p2_available_dir"):
        if column in result.columns:
            result[column] = result[column].astype(bool)
    return result


def collect_nested_pkl_entries(source_dir: Path, files: Iterable[str] | None = None) -> list[tuple[str, Path]]:
    """收集当前阶段的嵌套输入 pkl 文件。

    输入语义：source_dir 必须使用 ``task/session.pkl`` 两层结构；files 可写文件名、
    session stem 或 ``task/session``。
    输出语义：返回 ``(task_name, pkl_path)`` 列表。
    关键约束：不兼容扁平目录，避免旧结构数据混入当前流程。
    """

    if not source_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{source_dir}")

    entries: list[tuple[str, Path]] = []
    for task_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        for path in sorted(task_dir.glob("*.pkl")):
            entries.append((task_dir.name, path))

    selected = set(files or [])
    if not selected:
        return entries

    matched: set[str] = set()
    filtered: list[tuple[str, Path]] = []
    for task_name, path in entries:
        stem = path.stem
        keys = {path.name, stem, f"{task_name}/{path.name}", f"{task_name}/{stem}"}
        if keys & selected:
            matched.update(keys & selected)
            filtered.append((task_name, path))

    missing = selected - matched
    if missing:
        raise HumanTileDataPreprocessError(f"找不到指定 pkl 文件：{sorted(missing)}")
    return filtered


def _sample_tile_worker(task: tuple[str, str, str]) -> dict[str, Any]:
    """单个进程任务：从一个 03 frame_data 文件生成一个 04 tile_data 文件。

    输入语义：task 包含 ``task_name``、输入 pkl 路径和 tile 输出根目录的字符串。
    输出语义：写出对应 ``tile_dir/task_name/session.pkl``，并返回行数摘要。
    关键约束：每个 session 独立处理；worker 内不读取或写入其它 session，保证
    多进程并行不会改变科研逻辑。
    """

    task_name, frame_path_text, tile_dir_text = task
    frame_path = Path(frame_path_text)
    tile_dir = Path(tile_dir_text)

    subject_frame_data = pd.read_pickle(frame_path)
    subject_tile_data = sample_tile_rows_from_frame_data(subject_frame_data)
    output_parent = tile_dir / task_name
    output_parent.mkdir(parents=True, exist_ok=True)
    output_path = output_parent / frame_path.name
    subject_tile_data.to_pickle(output_path)
    return {
        "file": f"{task_name}/{frame_path.name}",
        "frame_rows": int(len(subject_frame_data)),
        "tile_rows": int(len(subject_tile_data)),
        "output": str(output_path),
    }


def _correct_tile_worker(
    task: tuple[
        str,
        str,
        str,
        dict[tuple[int, int], dict[str, tuple[int, int] | float]],
        int,
    ],
) -> dict[str, Any]:
    """单个进程任务：从一个 04 tile_data 文件生成 corrected tile_data 文件。

    输入语义：task 包含 ``task_name``、tile pkl 路径、corrected 输出根目录、
    只读邻接表和异步区间长度。
    输出语义：写出对应 ``corrected_dir/task_name/session.pkl``，并返回行数摘要。
    关键约束：adjacent_map 只读使用；不同 session 的 corrected 压缩互相独立，
    因此可以安全按文件并行。
    """

    task_name, tile_path_text, corrected_dir_text, adjacent_map, interval_frames = task
    tile_path = Path(tile_path_text)
    corrected_dir = Path(corrected_dir_text)

    subject_tile_data = pd.read_pickle(tile_path)
    corrected_tile_data = correct_subject_tile_data(
        subject_tile_data,
        adjacent_map,
        interval_frames=interval_frames,
    )
    output_parent = corrected_dir / task_name
    output_parent.mkdir(parents=True, exist_ok=True)
    output_path = output_parent / tile_path.name
    corrected_tile_data.to_pickle(output_path)
    return {
        "file": f"{task_name}/{tile_path.name}",
        "tile_rows": int(len(subject_tile_data)),
        "corrected_rows": int(len(corrected_tile_data)),
        "output": str(output_path),
    }


def sample_tile_rows_for_all_subjects(
    frame_dir: Path,
    tile_dir: Path,
    *,
    files: Iterable[str] | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """批量从 03 frame data 生成 04 tile data。

    输入语义：frame_dir 是 ``task/session.pkl`` 嵌套目录。
    输出语义：在 tile_dir 下按同样 task 层级写出抽帧结果，并返回摘要。
    关键约束：每个文件独立处理，不跨 session 比较联合状态；workers 大于 1 时
    按 session 文件并行。
    """

    input_entries = collect_nested_pkl_entries(frame_dir, files)
    if not input_entries:
        raise FileNotFoundError(f"frame data 输入目录没有嵌套 pkl 文件：{frame_dir}")
    if workers < 1:
        raise HumanTileDataPreprocessError("workers 必须大于等于 1。")

    tasks = [(task_name, str(frame_path), str(tile_dir)) for task_name, frame_path in input_entries]
    if workers == 1:
        summaries = []
        for task in tasks:
            summary = _sample_tile_worker(task)
            summaries.append(summary)
            print(summary["file"])
        return summaries

    summaries: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_sample_tile_worker, task): f"{task[0]}/{Path(task[1]).name}" for task in tasks}
        for future in as_completed(futures):
            file_label = futures[future]
            try:
                summary = future.result()
            except Exception as exc:
                raise HumanTileDataPreprocessError(f"{file_label} 生成 tile_data 失败：{exc}") from exc
            summaries.append(summary)
            print(summary["file"])
    summaries.sort(key=lambda item: str(item["file"]))
    return summaries


def correct_tile_data_for_all_subjects(
    tile_dir: Path,
    corrected_dir: Path,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
    *,
    files: Iterable[str] | None = None,
    interval_frames: int = ASYNC_TRANSITION_INTERVAL_FRAMES,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """批量从 tile data 生成 corrected tile data。

    输入语义：tile_dir 是上一步生成的 ``task/session.pkl`` 嵌套目录。
    输出语义：在 corrected_dir 下按同样 task 层级写出异步切换压缩并追加动作方向后的结果。
    关键约束：这里的 corrected 不补帧；它只压缩候选帧并重算动作方向和合法性。
    workers 大于 1 时按 session 文件并行。
    """

    tile_entries = collect_nested_pkl_entries(tile_dir, files)
    if not tile_entries:
        raise FileNotFoundError(f"tile data 输入目录没有嵌套 pkl 文件：{tile_dir}")
    if workers < 1:
        raise HumanTileDataPreprocessError("workers 必须大于等于 1。")

    tasks = [
        (task_name, str(tile_path), str(corrected_dir), adjacent_map, int(interval_frames))
        for task_name, tile_path in tile_entries
    ]
    if workers == 1:
        summaries = []
        for task in tasks:
            summary = _correct_tile_worker(task)
            summaries.append(summary)
            print(summary["file"])
        return summaries

    summaries: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_correct_tile_worker, task): f"{task[0]}/{Path(task[1]).name}" for task in tasks}
        for future in as_completed(futures):
            file_label = futures[future]
            try:
                summary = future.result()
            except Exception as exc:
                raise HumanTileDataPreprocessError(f"{file_label} 生成 corrected_tile_data 失败：{exc}") from exc
            summaries.append(summary)
            print(summary["file"])
    summaries.sort(key=lambda item: str(item["file"]))
    return summaries


def run_human_tile_data_preprocess(
    frame_dir: Path,
    tile_dir: Path,
    corrected_dir: Path,
    map_constants_path: Path,
    *,
    files: Iterable[str] | None = None,
    interval_frames: int = ASYNC_TRANSITION_INTERVAL_FRAMES,
    workers: int = 1,
) -> dict[str, Any]:
    """执行完整的人类 tile 数据预处理流程。

    输入语义：frame_dir 是 03 阶段输出的嵌套 pkl 目录。
    输出语义：先写 tile_dir，再写 corrected_dir，并返回整体摘要。
    关键约束：地图常量用于 corrected 阶段的位置连续性检查和动作合法性判断；
    workers 控制两个文件级阶段的并行进程数。
    """

    if workers < 1:
        raise HumanTileDataPreprocessError("workers 必须大于等于 1。")
    adjacent_map = load_adjacent_map(map_constants_path)
    tile_summaries = sample_tile_rows_for_all_subjects(frame_dir, tile_dir, files=files, workers=workers)
    corrected_summaries = correct_tile_data_for_all_subjects(
        tile_dir,
        corrected_dir,
        adjacent_map,
        files=files,
        interval_frames=interval_frames,
        workers=workers,
    )
    return {
        "frame_dir": str(frame_dir),
        "tile_dir": str(tile_dir),
        "corrected_dir": str(corrected_dir),
        "map_constants": str(map_constants_path),
        "interval_frames": int(interval_frames),
        "workers": int(workers),
        "file_count": len(corrected_summaries),
        "total_tile_rows": int(sum(item["tile_rows"] for item in corrected_summaries)),
        "total_corrected_rows": int(sum(item["corrected_rows"] for item in corrected_summaries)),
        "tile_summaries": tile_summaries,
        "corrected_summaries": corrected_summaries,
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输出语义：返回包含输入目录和输出目录的 argparse 命名空间。
    关键约束：默认路径全部位于当前 LoPS 仓库 ``data`` 下，且使用 task 嵌套结构。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-dir", type=Path, default=DEFAULT_DATA_ROOT / "03_preprocessed_frame_data")
    parser.add_argument("--tile-dir", type=Path, default=DEFAULT_DATA_ROOT / "04_tile_data")
    parser.add_argument(
        "--corrected-dir",
        type=Path,
        default=DEFAULT_DATA_ROOT / "04_corrected_tile_data",
    )
    parser.add_argument(
        "--map-constants",
        type=Path,
        default=DEFAULT_DATA_ROOT / "constant_data" / "map_constants.pkl",
        help="generate_map_constants.py 生成的地图常量 pickle。",
    )
    parser.add_argument(
        "--async-interval-frames",
        type=int,
        default=ASYNC_TRANSITION_INTERVAL_FRAMES,
        help=(
            "corrected 阶段压缩异步位置切换候选帧时使用的区间长度。"
            "默认 13 表示起始帧后的 0-12 帧属于同一区间，差值达到 13 则进入新区间。"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="文件级并行进程数；默认 8，避免大型 pkl 同时读写造成过高 I/O 和内存压力。",
    )
    parser.add_argument("files", nargs="*", help="可选：只处理这些 session，支持 session 或 task/session。")
    return parser.parse_args()


def main() -> None:
    """运行完整预处理流程并打印摘要。"""

    args = parse_args()
    summary = run_human_tile_data_preprocess(
        frame_dir=args.frame_dir,
        tile_dir=args.tile_dir,
        corrected_dir=args.corrected_dir,
        map_constants_path=args.map_constants,
        files=args.files or None,
        interval_frames=args.async_interval_frames,
        workers=args.workers,
    )
    print("human tile data preprocess 完成")
    print(f"文件数：{summary['file_count']}")
    print(f"并行进程数：{summary['workers']}")
    print(f"异步切换区间：{summary['interval_frames']} 帧")
    print(f"tile 总行数：{summary['total_tile_rows']}")
    print(f"corrected tile 总行数：{summary['total_corrected_rows']}")
    print(f"tile 输出目录：{args.tile_dir.resolve()}")
    print(f"corrected 输出目录：{args.corrected_dir.resolve()}")


if __name__ == "__main__":
    main()
