"""Social Pacman utility 的集中计算、修正和归一化流程。

本模块为 corrected tile 数据中的每个玩家分别计算七个旧 hierarchical utility
策略的 Q 值。输入是一行保存公共状态与多个玩家状态的 joint-state 表，输出仍
保持一行 joint-state，只新增 ``p1_*_Q``、``p1_*_Q_norm``、``p2_*_Q`` 等玩家
前缀字段，避免破坏合作/竞争分析需要的同一时刻对齐关系。
"""

from __future__ import annotations

import ast
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from LoPS.hierarchical_utility import (
    Q_COLUMNS,
    MapData,
    UtilityConfig,
    estimate_utility_for_dataframe,
    load_map_data,
    load_map_data_from_directory,
)


DIRECTION_NAMES: tuple[str, ...] = ("left", "right", "up", "down")
Q_NORM_COLUMNS: tuple[str, ...] = tuple(f"{column}_norm" for column in Q_COLUMNS)
PLAYER_PREFIXES: tuple[str, ...] = ("p1", "p2")
PARSED_POSITION_COLUMNS: tuple[str, ...] = (
    "pacmanPos",
    "ghost1Pos",
    "ghost2Pos",
    "beans",
    "energizers",
)
LEGACY_STATUS_COLUMNS: tuple[str, ...] = ("ifscared1", "ifscared2")


@dataclass(frozen=True)
class CalculateUtilityConfig:
    """保存集中 utility 计算阶段的配置。

    输入语义：utility_config 控制 raw Q 的策略深度等参数。
    输出语义：配置对象被文件级和目录级处理函数共享。
    关键约束：当前阶段不引入随机拟合参数，只包装 Q 计算本身的配置。
    """

    utility_config: UtilityConfig = UtilityConfig()


def parse_literal_if_needed(value: Any) -> Any:
    """解析数据中可能以字符串保存的 Python 字面量。

    输入语义：value 可以是 ``"(x, y)"``、``"[(x, y)]"`` 等字符串，也可以已经是对象。
    输出语义：字符串使用 ``ast.literal_eval`` 解析，其它值原样返回。
    关键约束：不使用 ``eval``，避免把数据解析和代码执行混在一起。
    """

    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def parse_position(value: Any) -> tuple[int, int]:
    """把位置字段解析成整数坐标。

    输入语义：value 可以是长度为 2 的 tuple/list，也可以是字符串形式的位置。
    输出语义：返回 ``(x, y)`` 整数坐标。
    关键约束：空方向或墙方向不应传入该函数。
    """

    parsed = parse_literal_if_needed(value)
    if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
        raise ValueError(f"无法解析位置字段：{value!r}")
    return int(parsed[0]), int(parsed[1])


def load_adjacent_map(path: str | Path) -> dict[tuple[int, int], dict[str, tuple[int, int] | float]]:
    """从统一地图常量 pickle 读取邻接表。

    输入语义：path 指向 ``script/constant_map/generate_map_constants.py`` 生成的
    ``map_constants.pkl``。
    输出语义：返回位置到四方向相邻位置的字典，不可走方向用 ``np.nan`` 表示。
    关键约束：地图连通性只以 pkl 内容为准，本函数不再补充或覆盖任何方向。
    """

    return load_map_data(path).adjacent_by_position


def load_calculate_utility_maps(
    constant_dir: str | Path,
) -> tuple[MapData, dict[tuple[int, int], dict[str, tuple[int, int] | float]]]:
    """读取集中 utility 阶段需要的全部地图常量。

    输入语义：constant_dir 包含 ``map_constants.pkl``。
    输出语义：返回 raw Q 计算使用的 MapData，以及修正/归一化使用的邻接表。
    关键约束：地图内容只从统一 pkl 读取，读取后不再做任何地图信息修正。
    """

    constant_dir = Path(constant_dir)
    map_data = load_map_data_from_directory(constant_dir)
    return map_data, map_data.adjacent_by_position


def correct_unavailable_q_values(
    data: pd.DataFrame,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
) -> tuple[pd.DataFrame, int]:
    """把不可走方向的 raw Q 值修正为 ``-np.inf``。

    输入语义：data 是已经追加 raw ``*_Q`` 的单被试 DataFrame。
    输出语义：返回修正后的 DataFrame 和被写入 ``-np.inf`` 的单元数量。
    关键约束：只修改 Q 数组中的墙方向，不改变行数、索引和非 Q 字段。
    """

    if "pacmanPos" not in data.columns:
        raise ValueError("utility 数据缺少 pacmanPos 列。")

    corrected = data.copy(deep=True)
    missing_columns = [column for column in Q_COLUMNS if column not in corrected.columns]
    if missing_columns:
        raise ValueError(f"utility 数据缺少 Q 列：{missing_columns}")

    unavailable_by_row: list[list[int]] = []
    for value in corrected["pacmanPos"]:
        position = parse_position(value)
        if position not in adjacent_map:
            raise KeyError(f"邻接表中找不到 Pacman 位置：{position}")
        adjacent = adjacent_map[position]
        unavailable_by_row.append(
            [
                direction_index
                for direction_index, direction in enumerate(DIRECTION_NAMES)
                if not isinstance(adjacent[direction], tuple)
            ]
        )

    changed_cells = 0
    for column in Q_COLUMNS:
        new_values: list[np.ndarray] = []
        for q_value, unavailable_indices in zip(corrected[column], unavailable_by_row):
            q_array = np.array(q_value, copy=True)
            if q_array.shape[0] != len(DIRECTION_NAMES):
                raise ValueError(f"{column} 中存在长度不是 4 的 Q 数组：shape={q_array.shape}")
            for direction_index in unavailable_indices:
                if not np.isneginf(q_array[direction_index]):
                    changed_cells += 1
                q_array[direction_index] = -np.inf
            new_values.append(q_array)
        corrected[column] = new_values

    return corrected, changed_cells


def normalize_with_inf(values: Any) -> np.ndarray:
    """按旧拟合规则归一化可能包含 ``-inf`` 的四方向 Q 值。

    输入语义：values 是长度为 4 的数组或列表，墙方向可能为 ``-inf``。
    输出语义：返回归一化后的 numpy 数组，有限值全为 0 时保持 0。
    关键约束：最大值只从有限方向中计算，墙方向不参与归一化。
    """

    source = np.asarray(values)
    result = source.copy()
    finite_indices = np.where(~np.isinf(source))[0]
    if set(source[finite_indices]) == {0}:
        result[finite_indices] = 0
    else:
        result[finite_indices] = result[finite_indices] / np.max(result[finite_indices])
    return result


def make_evade_q_non_negative(
    q_values: np.ndarray,
    offset: float,
    position: tuple[int, int],
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
) -> np.ndarray:
    """把 evade/no_energizer 类 Q 值平移到非负尺度并归一化。

    输入语义：q_values 是单帧四方向 Q 数组，offset 是该列全局有限最小值。
    输出语义：返回归一化后的数组。
    关键约束：该函数会原地修改 q_values；这是为了复现当前 weight_data 中保存的 raw Q。
    """

    available_indices: list[int] = []
    for direction in DIRECTION_NAMES:
        adjacent_value = adjacent_map[position][direction]
        if adjacent_value is not None and not isinstance(adjacent_value, float):
            available_indices.append(DIRECTION_NAMES.index(direction))
    q_values[available_indices] = q_values[available_indices] - offset
    return normalize_with_inf(q_values)


def prepare_standard_analysis_columns(data: pd.DataFrame) -> pd.DataFrame:
    """校验并整理单个玩家视角的 utility 临时表。

    输入语义：data 是从 joint-state 表中抽出的单玩家视角表，必须包含
    ``DayTrial/pacmanPos/action_dir/available_dir``。
    输出语义：返回行顺序不变、位置和方向字段已整理的 DataFrame。
    关键约束：本阶段不能删除或重排 joint 行，否则会破坏两个玩家之间的时间对齐。
    """

    required_columns = {"DayTrial", "pacmanPos", "action_dir", "available_dir"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"计算 utility 缺少标准分析字段：{missing_columns}")

    result = data.reset_index(drop=True).copy()
    for column in PARSED_POSITION_COLUMNS:
        if column in result.columns:
            result[column] = result[column].apply(parse_literal_if_needed)

    # action_dir 缺失统一用 NaN 表示；是否过滤无动作行交给后续拟合阶段按玩家决定。
    result["action_dir"] = result["action_dir"].apply(lambda value: value if value is not None else np.nan)
    result["available_dir"] = result["available_dir"].astype(bool)
    return result


def add_temporary_arrive_direction(data: pd.DataFrame) -> pd.DataFrame:
    """为 hierarchical utility 内部补充旧 arrive direction。

    输入语义：data 是标准 corrected tile 表，包含 ``DayTrial`` 和 ``action_dir``。
    输出语义：返回临时 DataFrame，其中 ``pacman_dir`` 等于同一 DayTrial 上一行的
    ``action_dir``。
    关键约束：``pacman_dir`` 只用于复现 Local 等策略的历史 Q 计算，不写入本阶段输出。
    """

    if "DayTrial" not in data.columns or "action_dir" not in data.columns:
        raise ValueError("计算临时 arrive direction 需要 DayTrial 和 action_dir 字段。")
    result = data.copy(deep=True)
    result["pacman_dir"] = result.groupby("DayTrial", sort=False)["action_dir"].shift(1)
    return result


def build_utility_estimation_input(data: pd.DataFrame) -> pd.DataFrame:
    """构造只供 Q 估计器使用的临时输入表。

    输入语义：data 是新 schema 的 corrected tile 表，ghost 状态字段已经是 int8。
    输出语义：返回带临时 ``pacman_dir`` 的 DataFrame，并把 ifscared 字段临时转为
    float。
    关键约束：旧 Q 结果是在 ifscared 为 float 的输入上生成的，而历史风险判断会把
    float 状态当作缺失标记；为了保证科研结果一致，这个兼容只发生在估计器入口，
    正式输出会恢复为新 schema 的 int8 状态码。
    """

    result = add_temporary_arrive_direction(data)
    missing_columns = [column for column in LEGACY_STATUS_COLUMNS if column not in result.columns]
    if missing_columns:
        raise ValueError(f"计算 utility 缺少 ghost 状态字段：{missing_columns}")
    for column in LEGACY_STATUS_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    return result


def restore_standard_input_columns(estimated_utility: pd.DataFrame, standard_input: pd.DataFrame) -> pd.DataFrame:
    """把 Q 估计后的非 Q 字段恢复为标准数据流格式。

    输入语义：estimated_utility 是估计器输出，可能携带临时 float 状态；standard_input
    是进入 utility 阶段的新 schema 输入。
    输出语义：返回 Q 列保持不变、标准字段 dtype 和取值恢复后的 DataFrame。
    关键约束：只恢复调用方已经提供的标准字段，不生成或保留旧流程字段。
    """

    result = estimated_utility.copy(deep=True)
    for column in standard_input.columns:
        if column in result.columns:
            # 使用原输入列覆盖估计器临时列，确保保存到下游的是新 schema。
            result[column] = standard_input[column].to_numpy()
    return result


def add_row_id(data: pd.DataFrame) -> pd.DataFrame:
    """为 utility 输出生成稳定行号 row_id。

    输入语义：data 是保持 joint-state 行序的输出表。
    输出语义：返回首列为 ``row_id`` 的 DataFrame。
    关键约束：row_id 只表示当前文件内的输出行号，不承载原始 frame id 语义。
    """

    result = data.copy(deep=True)
    if "row_id" in result.columns:
        result.drop(columns=["row_id"], inplace=True)
    result.insert(0, "row_id", np.arange(len(result), dtype=np.int64))
    return result


def append_normalized_q_columns(
    data: pd.DataFrame,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
) -> pd.DataFrame:
    """为修正后的 ``*_Q`` 追加 ``*_Q_norm`` 字段。

    输入语义：data 是单玩家视角表，已经过不可走方向修正，并完成标准字段整理。
    输出语义：返回追加 Q_norm 后的 DataFrame。
    关键约束：evade/no_energizer 类字段会在当前玩家内按列级最小有限值平移；
    p1 和 p2 分开调用该函数，因此归一化尺度互不影响。
    """

    result = data.copy(deep=True)
    for column in Q_COLUMNS:
        if ("evade" not in column) and ("no_energizer" not in column):
            result[f"{column}_norm"] = result[column].apply(normalize_with_inf)
            continue

        flat_values = result[column].explode().values
        finite_values = flat_values[flat_values != -np.inf]
        if len(finite_values) == 0:
            raise ValueError(f"{column} 没有有限 Q 值，无法计算归一化 offset。")
        offset = np.min(finite_values)
        result[f"{column}_norm"] = result[[column, "pacmanPos"]].apply(
            lambda row: make_evade_q_non_negative(row[column], offset, row.pacmanPos, adjacent_map)
            if set(row[column]) != {0}
            else [0, 0, 0, 0],
            axis=1,
        )
    return result


def prepare_calculated_utility_dataframe(
    corrected_utility: pd.DataFrame,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
) -> pd.DataFrame:
    """把单玩家视角的修正后 utility 表整理成可写回 joint-state 的结果。

    输入语义：corrected_utility 已包含修正后的 raw ``*_Q`` 字段。
    输出语义：返回追加 ``*_Q_norm`` 后的单玩家视角 DataFrame。
    关键约束：不删除无动作 trial，也不改变行数；玩家级过滤留给后续拟合阶段。
    """

    prepared = prepare_standard_analysis_columns(corrected_utility)
    return append_normalized_q_columns(prepared, adjacent_map)


def discover_player_prefixes(data: pd.DataFrame) -> list[str]:
    """识别当前文件中实际存在的玩家字段前缀。

    输入语义：data 是 04 corrected tile 输出的 joint-state 表。
    输出语义：返回存在完整 ``<player>_pos/action_dir/available_dir`` 字段的玩家前缀。
    关键约束：未来单人数据如果没有 ``p2_*`` 列，会自然跳过 p2，不生成 p2 Q 字段。
    """

    players: list[str] = []
    for player in PLAYER_PREFIXES:
        required_columns = {
            f"{player}_pos",
            f"{player}_action_dir",
            f"{player}_available_dir",
        }
        if required_columns.isdisjoint(data.columns):
            continue
        missing_columns = sorted(required_columns - set(data.columns))
        if missing_columns:
            raise ValueError(f"{player} 玩家字段不完整，缺少：{missing_columns}")
        players.append(player)
    if not players:
        raise ValueError("未找到任何玩家字段，至少需要 p1_pos/p1_action_dir/p1_available_dir。")
    return players


def build_player_alive_mask(data: pd.DataFrame, player: str) -> pd.Series:
    """构造某个玩家需要计算 Q 的行掩码。

    输入语义：data 是 joint-state 表，player 是 ``p1`` 或 ``p2``。
    输出语义：返回布尔 Series，True 表示该行玩家处于可计算状态。
    关键约束：死亡行仍保留在最终输出中，但该玩家的 Q 字段写为 NaN。
    """

    position_column = f"{player}_pos"
    mask = data[position_column].notna()
    alive_column = f"{player}_alive"
    if alive_column in data.columns:
        mask &= data[alive_column].astype(bool)
    return mask


def build_player_view(data: pd.DataFrame, player: str, row_mask: pd.Series) -> pd.DataFrame:
    """把 joint-state 表转换为单个玩家的临时 utility 输入表。

    输入语义：data 是完整 joint-state 表，row_mask 指明需要计算 Q 的行。
    输出语义：返回只包含可计算行的 DataFrame，其中玩家字段被映射为旧估计器使用的
    ``pacmanPos/action_dir/available_dir``。
    关键约束：该表只在 05 内部使用，保存结果时会改回玩家前缀字段。
    """

    view = data.loc[row_mask].copy()
    view["pacmanPos"] = view[f"{player}_pos"]
    view["action_dir"] = view[f"{player}_action_dir"]
    view["available_dir"] = view[f"{player}_available_dir"]
    return view


def prefixed_q_columns(player: str) -> list[str]:
    """返回某个玩家在输出表中对应的全部 Q 字段名。

    输入语义：player 是 ``p1`` 或 ``p2``。
    输出语义：返回 raw Q 和 Q_norm 的玩家前缀字段名。
    关键约束：字段顺序固定为 raw Q 在前、norm Q 在后，便于人工检查输出。
    """

    return [f"{player}_{column}" for column in (*Q_COLUMNS, *Q_NORM_COLUMNS)]


def calculate_player_utility(
    frame_data: pd.DataFrame,
    player: str,
    map_data: MapData,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
    config: CalculateUtilityConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """为 joint-state 表中的单个玩家计算七个策略 Q。

    输入语义：frame_data 是完整 joint-state 表，player 指定要计算的玩家。
    输出语义：返回只包含 ``<player>_*_Q`` 字段的 DataFrame，以及该玩家的处理摘要。
    关键约束：死亡行和缺失玩家位置行不进入 Q 估计器，但会在返回表中保留 NaN。
    """

    row_mask = build_player_alive_mask(frame_data, player)
    output = pd.DataFrame(index=frame_data.index)
    for column in prefixed_q_columns(player):
        output[column] = pd.Series([np.nan] * len(frame_data), index=frame_data.index, dtype=object)

    if not row_mask.any():
        return output, {
            "input_rows": int(frame_data.shape[0]),
            "computed_rows": 0,
            "skipped_rows": int((~row_mask).sum()),
            "changed_cells": 0,
        }

    player_view = build_player_view(frame_data, player, row_mask)
    # Q 估计器内部仍需要少量历史输入语义；这些临时字段不会写入正式输出。
    utility_input = build_utility_estimation_input(player_view)
    raw_utility = estimate_utility_for_dataframe(utility_input, map_data, config.utility_config)
    raw_utility.drop(columns=["pacman_dir"], inplace=True, errors="ignore")
    raw_utility = restore_standard_input_columns(raw_utility, player_view)
    corrected_utility, changed_cells = correct_unavailable_q_values(raw_utility, adjacent_map)
    calculated_utility = prepare_calculated_utility_dataframe(corrected_utility, adjacent_map)

    target_indices = frame_data.index[row_mask]
    for source_column in (*Q_COLUMNS, *Q_NORM_COLUMNS):
        target_column = f"{player}_{source_column}"
        # calculated_utility 已 reset index，因此这里按顺序写回原 joint 行。
        for target_index, value in zip(target_indices, calculated_utility[source_column].to_numpy()):
            output.at[target_index, target_column] = value

    return output, {
        "input_rows": int(frame_data.shape[0]),
        "computed_rows": int(row_mask.sum()),
        "skipped_rows": int((~row_mask).sum()),
        "changed_cells": int(changed_cells),
    }


def calculate_utility_for_dataframe(
    frame_data: pd.DataFrame,
    map_data: MapData,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
    config: CalculateUtilityConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """对单个 corrected tile joint-state DataFrame 执行完整 utility 计算。

    输入语义：frame_data 是 04 corrected tile 输出，包含公共状态和一个或两个玩家状态。
    输出语义：返回原 joint-state 字段加玩家前缀 Q 字段的 DataFrame 和处理摘要。
    关键约束：不拆文件、不展开成长表、不删除 joint 行，确保合作/竞争状态对齐。
    """

    config = CalculateUtilityConfig() if config is None else config
    result = add_row_id(frame_data.reset_index(drop=True))
    player_summaries: dict[str, dict[str, Any]] = {}
    changed_cells = 0

    for player in discover_player_prefixes(frame_data):
        player_output, player_summary = calculate_player_utility(
            frame_data=frame_data,
            player=player,
            map_data=map_data,
            adjacent_map=adjacent_map,
            config=config,
        )
        player_summaries[player] = player_summary
        changed_cells += int(player_summary["changed_cells"])
        for column in player_output.columns:
            result[column] = player_output[column].to_numpy()

    summary = {
        "input_rows": int(frame_data.shape[0]),
        "output_rows": int(result.shape[0]),
        "changed_cells": int(changed_cells),
        "players": player_summaries,
        "column_count": int(result.shape[1]),
    }
    return result, summary


def process_calculate_utility_file(
    input_path: str | Path,
    output_path: str | Path,
    map_data: MapData,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
    config: CalculateUtilityConfig | None = None,
) -> dict[str, Any]:
    """处理单个 corrected tile pickle 并保存集中 utility 输出。

    输入语义：input_path 是单被试 corrected tile 数据，output_path 是目标 pickle。
    输出语义：写出包含 ``*_Q`` 和 ``*_Q_norm`` 的 DataFrame，并返回摘要。
    关键约束：输出文件名由调用方决定，标准运行脚本沿用输入文件名。
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    with input_path.open("rb") as file:
        frame_data = pickle.load(file)
    calculated_utility, summary = calculate_utility_for_dataframe(frame_data, map_data, adjacent_map, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(calculated_utility, file)
    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        **summary,
    }


def process_calculate_utility_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    map_data: MapData,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
    config: CalculateUtilityConfig | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """批量处理 corrected tile 嵌套目录并生成集中 utility 数据。

    输入语义：input_dir 是包含 ``comp/*.pkl``、``coop/*.pkl`` 等任务子目录的目录。
    输出语义：每个输入文件按相同相对路径写到 output_dir，返回文件摘要列表。
    关键约束：只支持当前主流程的嵌套结构，不再兼容旧扁平目录。
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    config = CalculateUtilityConfig() if config is None else config
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")
    input_paths = sorted(path for path in input_dir.glob("*/*.pkl") if path.is_file())
    if not input_paths:
        raise FileNotFoundError(f"输入目录中没有嵌套 pickle 文件：{input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (input_path, output_dir / input_path.relative_to(input_dir), map_data, adjacent_map, config)
        for input_path in input_paths
    ]
    if workers <= 1:
        return [_process_calculate_utility_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        return list(executor.map(_process_calculate_utility_task, tasks))


def _process_calculate_utility_task(
    task: tuple[
        Path,
        Path,
        MapData,
        dict[tuple[int, int], dict[str, tuple[int, int] | float]],
        CalculateUtilityConfig,
    ],
) -> dict[str, Any]:
    """执行目录级并行中的单个集中 utility 任务。

    输入语义：task 包含输入路径、输出路径、地图数据、邻接表和配置。
    输出语义：返回 ``process_calculate_utility_file`` 的摘要。
    关键约束：保持顶层函数，便于 multiprocessing pickle。
    """

    input_path, output_path, map_data, adjacent_map, config = task
    return process_calculate_utility_file(input_path, output_path, map_data, adjacent_map, config)
