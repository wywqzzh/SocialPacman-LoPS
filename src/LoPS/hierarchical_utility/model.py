"""fMRI hierarchical utility 预计算的数据模型与输入解析。"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DIRECTIONS: tuple[str, str, str, str] = ("left", "right", "up", "down")
DIRECTION_TO_INDEX = {direction: index for index, direction in enumerate(DIRECTIONS)}
GHOST_NAMES: tuple[str, str] = ("blinky", "clyde")


@dataclass(frozen=True)
class MapData:
    """保存 utility 计算所需的 Pacman 地图常量。

    输入语义：由 `script/constant_map/generate_map_constants.py` 生成的
    `map_constants.pkl` 解析得到。
    输出语义：策略计算时通过位置 tuple 查询四方向相邻位置和任意两点距离。
    关键约束：该结构只保存正式计算需要的数据；所有地图连通性修正都必须已经在
    地图常量生成阶段完成，读取阶段不再补边或改写距离。
    """

    adjacent_by_position: dict[tuple[int, int], dict[str, Any]]
    distance_by_position: dict[tuple[int, int], dict[tuple[int, int], int | float]]
    reward_amount: dict[int, int]


@dataclass(frozen=True)
class CompiledMapData:
    """保存面向高频路径搜索的地图整数化表示。

    输入语义：由 `MapData` 编译得到，位置 tuple 被映射为紧凑整数 id。
    输出语义：策略搜索时直接读取邻接 id、区域 bitmask 和奖励表。
    关键约束：墙体和不可达方向统一编码为 -1，避免在内层循环反复判断 NaN。
    """

    position_to_id: dict[tuple[int, int], int]
    id_to_position: tuple[tuple[int, int], ...]
    neighbor_ids: tuple[tuple[int, int, int, int], ...]
    global_region_masks: tuple[tuple[int, int, int, int], ...]
    reward_amount: dict[int, int]


@dataclass(frozen=True)
class UtilityConfig:
    """保存 hierarchical utility 预计算的所有可调参数。

    输入语义：调用方可覆盖旧脚本中写死的深度、阈值、随机和惰性系数。
    输出语义：策略构造时从该配置读取参数，默认值复现 fMRI 目标路径。
    关键约束：默认随机系数和惰性系数均为 0，因此保存的 Q 值不受随机数影响。
    Local 默认使用 0.90 的逐步奖励衰减：第 j 步资源奖励乘以 ``0.9**(j-1)``，
    使更直接到达的最佳路径优于绕路后汇合到同一资源的路径。
    Evade 默认只搜索 6 步，把它限定为对近距离正常 ghost 的即时躲避策略，避免把
    7--10 步外、恰好与资源目标反向的远处 ghost 误解释成主动躲避目标。
    Approach 默认搜索 20 步，用于表达玩家在 energizer 后跨较长地图距离追逐 scared
    ghost 的行为；该范围明显长于 Evade，二者分别对应远程追逐和近程避险。
    """

    randomness_coeff: float = 0.0
    laziness_coeff: float = 0.0
    global_depth: int = 15
    global_ignore_depth: int = 10
    global_ghost_attractive_thr: int = 34
    global_ghost_repulsive_thr: int = 34
    local_depth: int = 10
    local_discount_factor: float = 0.90
    local_ghost_attractive_thr: int = 10
    local_ghost_repulsive_thr: int = 10
    evade_depth: int = 6
    evade_ghost_attractive_thr: int = 0
    evade_ghost_repulsive_thr: int = 0
    approach_depth: int = 20
    approach_ghost_attractive_thr: int = 0
    approach_ghost_repulsive_thr: int = 0
    energizer_depth: int = 10
    energizer_ghost_attractive_thr: int = 0
    energizer_ghost_repulsive_thr: int = 0
    no_energizer_depth: int = 8
    no_energizer_ghost_attractive_thr: int = 0
    no_energizer_ghost_repulsive_thr: int = 0


@dataclass(frozen=True)
class FrameState:
    """保存单帧游戏状态的规范化表示。

    输入语义：由 corrected tile DataFrame 的一行解析得到。
    输出语义：每个策略只读取该结构，不再直接处理 DataFrame 字段或字符串字面量。
    关键约束：当前主流程只分析 two-ghost 数据，因此只保存 blinky/clyde 两只 ghost。
    """

    pacman_position: tuple[int, int]
    energizers: list[tuple[int, int]] | float
    beans: list[tuple[int, int]] | float
    ghost_positions: list[tuple[int, int] | tuple[()]]
    ghost_status: list[Any]
    last_direction: str | None


@dataclass(frozen=True)
class CompiledFrameState:
    """保存单帧状态的整数 id 与 bitmask 表示。

    输入语义：由 `FrameState` 和 `CompiledMapData` 编译得到。
    输出语义：共享路径搜索引擎使用该结构进行快速 membership、移除和碰撞判断。
    关键约束：缺失位置统一使用 -1，缺失对象集合统一使用空 bitmask。
    """

    pacman_id: int
    bean_mask: int
    energizer_mask: int
    ghost_ids: tuple[int, ...]
    ghost_status: tuple[Any, ...]
    last_direction_id: int | None


def load_map_data(map_constants_path: str | Path) -> MapData:
    """读取地图相邻关系、距离表和奖励常量。

    输入语义：map_constants_path 指向当前项目生成的 `map_constants.pkl`。
    输出语义：返回可被所有策略复用的 `MapData`。
    关键约束：本函数只读取和转换 pkl 中已有的地图常量，不再进行 tunnel、
    鬼屋或其它坐标连通性的读取后修正。
    """

    map_constants = _read_map_constants(Path(map_constants_path))
    adjacent_by_position = _read_adjacent_map(map_constants["adjacent_map"])
    distance_by_position = _read_distance_map(map_constants["dij_distance_map"])
    reward_amount = {
        1: 2,
        2: 4,
        8: 8,
        9: 8,
    }
    return MapData(adjacent_by_position, distance_by_position, reward_amount)


def compile_map_data(map_data: MapData) -> CompiledMapData:
    """把地图常量编译成路径搜索使用的整数结构。

    输入语义：map_data 是标准地图常量，包含 tuple 位置的邻接表和距离表。
    输出语义：返回整数 id 邻接表，以及 Global 策略使用的预计算区域 bitmask。
    关键约束：编译结果只依赖地图常量，可在一个文件或一个进程内重复复用。
    """

    positions: set[tuple[int, int]] = set(map_data.adjacent_by_position)
    positions.update(map_data.distance_by_position)
    for adjacent_positions in map_data.adjacent_by_position.values():
        for value in adjacent_positions.values():
            if not _is_float_marker(value):
                positions.add(tuple(value))
    for distance_targets in map_data.distance_by_position.values():
        positions.update(distance_targets)

    id_to_position = tuple(sorted(positions))
    position_to_id = {position: index for index, position in enumerate(id_to_position)}
    neighbor_rows: list[tuple[int, int, int, int]] = []
    for position in id_to_position:
        adjacent_positions = map_data.adjacent_by_position.get(position, {})
        row: list[int] = []
        for direction in DIRECTIONS:
            next_position = adjacent_positions.get(direction, np.nan)
            if _is_float_marker(next_position):
                row.append(-1)
            else:
                row.append(position_to_id[tuple(next_position)])
        neighbor_rows.append(tuple(row))  # type: ignore[arg-type]

    global_region_masks = tuple(
        _build_global_region_masks(position, position_to_id, map_data.distance_by_position)
        for position in id_to_position
    )
    return CompiledMapData(
        position_to_id=position_to_id,
        id_to_position=id_to_position,
        neighbor_ids=tuple(neighbor_rows),
        global_region_masks=global_region_masks,
        reward_amount=map_data.reward_amount,
    )


def compile_frame_state(frame_state: FrameState, compiled_map: CompiledMapData) -> CompiledFrameState:
    """把单帧游戏状态编译成整数 id 与 bitmask。

    输入语义：frame_state 是行级规范状态，compiled_map 提供位置 id 映射。
    输出语义：返回共享路径搜索可以直接使用的 `CompiledFrameState`。
    关键约束：不在地图中的对象位置不会被路径访问到，因此不会写入 bitmask。
    """

    return CompiledFrameState(
        pacman_id=compiled_map.position_to_id[frame_state.pacman_position],
        bean_mask=_positions_to_mask(frame_state.beans, compiled_map.position_to_id),
        energizer_mask=_positions_to_mask(frame_state.energizers, compiled_map.position_to_id),
        ghost_ids=tuple(
            -1 if position == () else compiled_map.position_to_id.get(position, -1)
            for position in frame_state.ghost_positions
        ),  # type: ignore[arg-type]
        ghost_status=tuple(frame_state.ghost_status),
        last_direction_id=None if frame_state.last_direction is None else DIRECTION_TO_INDEX[frame_state.last_direction],
    )


def parse_frame_state(row: pd.Series, columns: pd.Index | list[str] | tuple[str, ...]) -> FrameState:
    """把 corrected tile 的一行解析为策略计算使用的规范状态。

    输入语义：row 是单行游戏状态，columns 是该 DataFrame 的列名集合。
    输出语义：返回已经解析坐标、对象列表、状态字段和 last direction 的 `FrameState`。
    关键约束：只兼容旧 fMRI 目标路径实际使用的字段变体，不额外猜测无关格式。
    """

    column_set = set(columns)
    pacman_position = _parse_required_position(row["pacmanPos"])
    energizers = _parse_position_list(row["energizers"])
    beans = _parse_position_list(row["beans"])
    ghost_positions = [_parse_optional_ghost_position(row[f"ghost{index}Pos"]) for index in range(1, 3)]

    if "ghost1_status" in column_set or "ghost2_status" in column_set:
        ghost_status = [row["ghost1_status"], row["ghost2_status"]]
    else:
        ghost_status = [row[f"ifscared{index}"] for index in range(1, 3)]

    # 新标准分析流删除了 arrive direction（旧 pacman_dir）。该字段只在非零 laziness
    # 系数下影响 Q；当前 fMRI utility 默认 laziness 为 0，因此缺失时保持 None。
    if "pacman_dir" in column_set and not pd.isna(row["pacman_dir"]):
        last_direction = row["pacman_dir"]
    else:
        last_direction = None
    return FrameState(
        pacman_position=pacman_position,
        energizers=energizers,
        beans=beans,
        ghost_positions=ghost_positions,
        ghost_status=ghost_status,
        last_direction=last_direction,
    )


def _read_map_constants(path: Path) -> dict[str, pd.DataFrame]:
    """读取统一地图常量 pickle 并校验基本结构。

    输入语义：path 指向 `data/constant_data/map_constants.pkl` 或同结构文件。
    输出语义：返回包含 `adjacent_map` 和 `dij_distance_map` 两张 DataFrame 的字典。
    关键约束：该函数只负责加载和结构校验，不对地图内容做任何补丁。
    """

    if not path.is_file():
        raise FileNotFoundError(f"找不到地图常量文件：{path}")

    constants = pd.read_pickle(path)
    if not isinstance(constants, dict):
        raise TypeError(f"地图常量文件应保存为 dict：{path}")

    required_keys = {"adjacent_map", "dij_distance_map"}
    missing_keys = sorted(required_keys - set(constants))
    if missing_keys:
        raise KeyError(f"地图常量文件缺少字段：{missing_keys}")

    for key in required_keys:
        if not isinstance(constants[key], pd.DataFrame):
            raise TypeError(f"地图常量 {key} 应为 DataFrame，实际为 {type(constants[key]).__name__}")
    return constants


def _read_adjacent_map(adjacent_data: pd.DataFrame) -> dict[tuple[int, int], dict[str, Any]]:
    """从统一地图常量表读取四方向相邻关系。

    输入语义：adjacent_data 是 `map_constants.pkl` 中的 `adjacent_map` DataFrame。
    输出语义：返回以位置 tuple 为键、四方向为二级键的相邻关系字典。
    关键约束：NaN 墙体保持为浮点 NaN；读取阶段不再新增 tunnel 边或改写方向。
    """

    required_columns = {"pos", *DIRECTIONS}
    missing_columns = sorted(required_columns - set(adjacent_data.columns))
    if missing_columns:
        raise KeyError(f"adjacent_map 缺少列：{missing_columns}")

    adjacent_by_position: dict[tuple[int, int], dict[str, Any]] = {}
    for _, item in adjacent_data.iterrows():
        position = _coerce_position(item["pos"])
        adjacent_by_position[position] = {
            direction: np.nan if _is_float_marker(item[direction]) else _coerce_position(item[direction])
            for direction in DIRECTIONS
        }
    return adjacent_by_position


def _read_distance_map(distance_data: pd.DataFrame) -> dict[tuple[int, int], dict[tuple[int, int], int | float]]:
    """从统一地图常量表读取任意两点的最短距离。

    输入语义：distance_data 是 `map_constants.pkl` 中的 `dij_distance_map` DataFrame。
    输出语义：返回 `distance_by_position[pos1][pos2] = dis` 的嵌套字典。
    关键约束：距离表必须来自地图常量生成阶段；读取阶段不再手动补充距离。
    """

    required_columns = {"pos1", "pos2", "dis"}
    missing_columns = sorted(required_columns - set(distance_data.columns))
    if missing_columns:
        raise KeyError(f"dij_distance_map 缺少列：{missing_columns}")

    distance_by_position: dict[tuple[int, int], dict[tuple[int, int], int | float]] = {}
    for _, item in distance_data.iterrows():
        pos1 = _coerce_position(item["pos1"])
        pos2 = _coerce_position(item["pos2"])
        distance_by_position.setdefault(pos1, {})[pos2] = item["dis"]
    return distance_by_position


def _build_global_region_masks(
    current_position: tuple[int, int],
    position_to_id: dict[tuple[int, int], int],
    distance_by_position: dict[tuple[int, int], dict[tuple[int, int], int | float]],
) -> tuple[int, int, int, int]:
    """预计算当前位置四个方向的远距离区域 bitmask。

    输入语义：current_position 是 Pacman 当前位置，position_to_id 是地图位置映射。
    输出语义：返回四方向 bitmask，bit 为 1 表示该位置属于对应方向的远距离区域。
    关键约束：区域边界和距离阈值与 Global 策略保持一致。
    """

    masks: list[int] = []
    distances = distance_by_position.get(current_position, {})
    for direction in DIRECTIONS:
        upper_left, lower_right = _direction_area(current_position, direction)
        mask = 0
        for x in range(upper_left[0], lower_right[0] + 1):
            for y in range(upper_left[1], lower_right[1] + 1):
                position = (x, y)
                if distances.get(position, 0) <= 10:
                    continue
                position_id = position_to_id.get(position)
                if position_id is not None:
                    mask |= 1 << position_id
        masks.append(mask)
    return tuple(masks)  # type: ignore[return-value]


def _direction_area(position: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    """返回 Global 策略中一个方向对应的矩形区域。

    输入语义：position 是 Pacman 当前位置，direction 是四方向之一。
    输出语义：返回 `[upper_left, lower_right]` 两个角点。
    关键约束：边界固定为 fMRI 地图使用的 x=1..28、y=1..33。
    """

    left_bound = 1
    right_bound = 28
    upper_bound = 1
    lower_bound = 33
    if direction == "left":
        return [(left_bound, upper_bound), (max(1, position[0] - 1), lower_bound)]
    if direction == "right":
        return [(min(right_bound, position[0] + 1), upper_bound), (right_bound, lower_bound)]
    if direction == "up":
        return [(left_bound, upper_bound), (right_bound, min(lower_bound, position[1] + 1))]
    if direction == "down":
        return [(left_bound, min(lower_bound, position[1] + 1)), (right_bound, lower_bound)]
    raise ValueError(f"未知方向：{direction}")


def _positions_to_mask(value: list[tuple[int, int]] | float, position_to_id: dict[tuple[int, int], int]) -> int:
    """把位置列表转换为整数 bitmask。

    输入语义：value 是 beans 或 energizers 的位置列表，也可能是浮点缺失标记。
    输出语义：返回位置集合 bitmask，缺失或空列表返回 0。
    关键约束：不在地图中的位置不会被路径访问，因此直接忽略。
    """

    if _is_float_marker(value):
        return 0
    mask = 0
    for position in value:
        position_id = position_to_id.get(tuple(position))
        if position_id is not None:
            mask |= 1 << position_id
    return mask


def _coerce_position(value: Any) -> tuple[int, int]:
    """把地图常量中的位置值转换为标准整数 tuple。

    输入语义：value 来自 `map_constants.pkl`，通常已经是 tuple，也兼容字符串形式。
    输出语义：返回 `(x, y)` 整数坐标。
    关键约束：该函数只做类型转换，不改变坐标含义，也不执行任何地图修正。
    """

    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
        raise ValueError(f"无法解析地图坐标：{value!r}")
    return int(parsed[0]), int(parsed[1])


def _parse_required_position(value: Any) -> tuple[int, int]:
    """解析必须存在的 Pacman 坐标。

    输入语义：value 可以是 tuple/list，也可以是 `"(x, y)"` 字符串。
    输出语义：返回整数 tuple 坐标。
    关键约束：该函数只用于 Pacman 位置，空位置不是合法输入。
    """

    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    return int(parsed[0]), int(parsed[1])


def _parse_optional_ghost_position(value: Any) -> tuple[int, int] | tuple[()]:
    """解析 ghost 坐标或空 ghost 标记。

    输入语义：value 可以是坐标字符串、tuple/list 坐标，也可以是旧数据中的空列表。
    输出语义：坐标返回 `(x, y)`，空 ghost 返回空 tuple。
    关键约束：空 tuple 对应旧代码中 `tuple([])` 的后续行为。
    """

    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if isinstance(parsed, list) and len(parsed) == 0:
        return ()
    return int(parsed[0]), int(parsed[1])


def _parse_position_list(value: Any) -> list[tuple[int, int]] | float:
    """解析 beans 或 energizers 位置列表。

    输入语义：value 可以是列表、字符串列表或 NaN。
    输出语义：返回坐标 tuple 列表；缺失时返回 `np.nan`。
    关键约束：列表内元素顺序保持不变，因为旧策略会按列表状态逐步移除对象。
    """

    if value is None or _is_float_missing(value):
        return np.nan
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    return [tuple(item) for item in parsed]


def _is_float_marker(value: Any) -> bool:
    """判断一个值是否属于旧代码用来表示缺失的浮点标记。

    输入语义：value 来自 DataFrame 或解析过程。
    输出语义：Python float 和 numpy floating 返回 True。
    关键约束：整数状态值不能被当作浮点缺失标记。
    """

    return isinstance(value, (float, np.floating))


def _is_float_missing(value: Any) -> bool:
    """判断一个值是否是浮点 NaN 缺失值。

    输入语义：value 可能是 Python float、numpy floating 或其它对象。
    输出语义：只有浮点 NaN 返回 True。
    关键约束：普通数值型水果类型不能被当作缺失值。
    """

    return _is_float_marker(value) and bool(np.isnan(value))
