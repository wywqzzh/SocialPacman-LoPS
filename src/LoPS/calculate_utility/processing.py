"""Social Pacman utility 的集中计算、修正和归一化流程。

本模块为 corrected tile 数据中的每个玩家分别计算七种行为策略的 Q 值。Global、
Energizer 和 Approach 另外保存逐行目标候选：Global 的目标是一团资源，Energizer
的目标是一个明确的 energizer 坐标，Approach 的目标是一只身份稳定的 ghost。
输入是一行保存公共状态与多个玩家状态的 joint-state 表，输出仍保持一行 joint-state，
避免破坏合作/竞争分析需要的同一时刻对齐关系。
"""

from __future__ import annotations

import ast
import pickle
from collections import deque
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

    输入语义：utility_config 控制旧七策略 raw Q 的深度等参数；
    global_cluster_min_distance/global_cluster_radius 控制单个资源点参与 cluster Global
    距离计算的最近距离，以及有效 Global 目标的最远距离；
    global_cluster_distance_threshold 控制候选资源点的聚类半径。
    utility_config 中旧 Global 使用的 global_ignore_depth 不直接限制 cluster Global。
    输出语义：配置对象被文件级和目录级处理函数共享。
    关键约束：当前阶段只生成逐行候选 utility，不根据 context 选择 best Global 或
    best Energizer；目标选择发生在后续 context 拟合阶段。
    """

    utility_config: UtilityConfig = UtilityConfig()
    global_cluster_min_distance: int = 2
    global_cluster_radius: int = 60
    global_cluster_distance_threshold: int = 2


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


def parse_position_list(value: Any) -> list[tuple[int, int]]:
    """把资源列表字段解析为坐标列表。

    输入语义：value 通常来自 ``beans`` 或 ``energizers``，可以是字符串列表、
    Python list/tuple、空列表或缺失值。
    输出语义：返回去除非法元素后的 ``(x, y)`` 坐标列表。
    关键约束：本函数用于 cluster global 候选生成；缺失资源按空列表处理，
    单个非法元素会被跳过，避免一条脏资源记录中断整个文件计算。
    """

    if value is None or (isinstance(value, (float, np.floating)) and pd.isna(value)):
        return []
    parsed = parse_literal_if_needed(value)
    if parsed is None or (isinstance(parsed, (float, np.floating)) and pd.isna(parsed)):
        return []
    if isinstance(parsed, np.ndarray):
        parsed = parsed.tolist()
    if not isinstance(parsed, (list, tuple)):
        return []

    positions: list[tuple[int, int]] = []
    for item in parsed:
        try:
            positions.append(parse_position(item))
        except (TypeError, ValueError):
            continue
    return positions


def map_distance(map_data: MapData, first: tuple[int, int], second: tuple[int, int]) -> float:
    """读取两个 tile 之间的地图最短路距离。

    输入语义：first/second 是 tile 坐标，map_data 来自统一地图常量。
    输出语义：返回最短路距离，无法到达或缺失时返回 ``np.inf``。
    关键约束：必须使用地图距离表而不是坐标差；这样 tunnel 两端例如
    ``(0, 18)`` 和 ``(29, 18)`` 会被正确视为相邻。
    """

    if first == second:
        return 0.0
    return float(map_data.distance_by_position.get(first, {}).get(second, np.inf))


def cluster_resources_by_distance(
    resources: list[tuple[int, int]],
    map_data: MapData,
    distance_threshold: int,
) -> list[set[tuple[int, int]]]:
    """按地图最短路距离把资源点聚成多个 global 目标团。

    输入语义：resources 是当前行剩余 ``beans + energizers``，distance_threshold
    是两个资源点可被合并为同一团的最大地图距离。
    输出语义：返回资源坐标集合列表，每个集合对应一个候选 global 目标。
    关键约束：cluster size 允许为 1；聚类用 union-find 连接所有距离不超过阈值的
    资源对，因此 tunnel 连通性完全由 map_data 控制。
    """

    unique_resources = sorted(set(resources))
    parent = {position: position for position in unique_resources}

    def find(position: tuple[int, int]) -> tuple[int, int]:
        """查找 union-find 根节点，并压缩路径。"""

        while parent[position] != position:
            parent[position] = parent[parent[position]]
            position = parent[position]
        return position

    def union(first: tuple[int, int], second: tuple[int, int]) -> None:
        """合并两个资源点所在的 cluster。"""

        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first_index, first in enumerate(unique_resources):
        for second in unique_resources[first_index + 1 :]:
            if map_distance(map_data, first, second) <= distance_threshold:
                union(first, second)

    groups: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for position in unique_resources:
        groups.setdefault(find(position), set()).add(position)
    return sorted(groups.values(), key=lambda cluster: sorted(cluster))


def cluster_min_distance(
    position: tuple[int, int],
    cluster: set[tuple[int, int]],
    map_data: MapData,
) -> float:
    """计算当前位置到一个资源 cluster 的最短距离。

    输入语义：position 是 Pacman 当前位置，cluster 是一团资源坐标。
    输出语义：返回到该 cluster 中最近资源点的地图最短路距离。
    关键约束：如果 cluster 为空或不可达，返回 ``np.inf``，下游会把该候选视为无信息。
    """

    if not cluster:
        return float("inf")
    return float(min(map_distance(map_data, position, resource) for resource in cluster))


def nearest_cluster_resource(
    position: tuple[int, int],
    cluster: set[tuple[int, int]],
    map_data: MapData,
) -> tuple[int, int] | None:
    """返回 cluster 中离当前位置最近的资源点。

    输入语义：position 是 Pacman 当前位置，cluster 是候选资源团。
    输出语义：返回最近资源点；cluster 为空时返回 None。
    关键约束：该字段只用于解释 best global 目标，不参与 Q 计算。
    """

    if not cluster:
        return None
    return min(cluster, key=lambda resource: map_distance(map_data, position, resource))


def global_cluster_q_for_row(
    row: pd.Series,
    map_data: MapData,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
    config: CalculateUtilityConfig,
) -> tuple[list[list[float]], list[list[float]], list[dict[str, Any]]]:
    """为单行生成多个 cluster global 候选 utility。

    输入语义：row 是单玩家视角的一行，包含 ``pacmanPos/beans/energizers``；
    map_data/adjacent_map 来自统一地图常量；config 提供聚类阈值和 global 距离范围。
    输出语义：返回 raw utility 矩阵、normalized utility 矩阵和与矩阵行对齐的 meta。
    关键约束：这里不选择 best cluster。每个 cluster 的四方向 raw Q 表示“走该方向后
    到可参与 Global 的资源子集距离减少量 × 子集资源数”；普通豆和 energizer 同权重。
    距离小于 ``global_cluster_min_distance`` 的眼前资源只从本行距离计算中排除，不删除
    完整 cluster/meta；只要同一资源团还有较远资源，Global 就继续提供方向信息。只有
    整团都落入局部范围时，合法方向才全部置零，避免单颗近豆令整个远程目标突然失效。
    """

    position = parse_position(row["pacmanPos"])
    beans = parse_position_list(row.get("beans", []))
    energizers = parse_position_list(row.get("energizers", []))
    energizer_set = set(energizers)
    clusters = cluster_resources_by_distance(
        beans + energizers,
        map_data,
        config.global_cluster_distance_threshold,
    )

    raw_matrix: list[list[float]] = []
    meta_values: list[dict[str, Any]] = []
    for cluster_id, cluster in enumerate(clusters):
        min_distance = cluster_min_distance(position, cluster, map_data)
        # 完整 cluster 继续用于跨行目标匹配；Global Q 只使用当前位置至少相距阈值
        # 步数的资源。过滤集合在当前行固定，计算相邻位置距离时不能再次过滤，否则
        # 朝距离2资源前进一步后会把该目标移除，反而无法得到正向推进 utility。
        global_resources: set[tuple[int, int]] = set()
        for resource in cluster:
            resource_distance = map_distance(map_data, position, resource)
            if np.isfinite(resource_distance) and resource_distance >= config.global_cluster_min_distance:
                global_resources.add(resource)
        global_min_distance = cluster_min_distance(position, global_resources, map_data)
        raw_q = [float("-inf")] * len(DIRECTION_NAMES)
        # 距离 0/1 的资源由 Local 表达；只要过滤后仍有距离 2 及以上的资源，Global
        # 就朝剩余资源继续计算。半径上限仍针对过滤后的最近 Global 资源。
        in_global_range = (
            np.isfinite(global_min_distance)
            and global_min_distance <= config.global_cluster_radius
        )

        for direction_index, direction in enumerate(DIRECTION_NAMES):
            adjacent_value = adjacent_map[position][direction]
            if not isinstance(adjacent_value, tuple):
                continue
            if not in_global_range:
                raw_q[direction_index] = 0.0
                continue
            next_distance = cluster_min_distance(adjacent_value, global_resources, map_data)
            if np.isfinite(next_distance):
                # 一个普通豆和一个 energizer 在 Global 中贡献相同。这里按真正参与本行
                # Global 距离计算的资源数缩放；后续 06 使用归一化 Q，raw 尺度主要用于解释。
                raw_q[direction_index] = (
                    global_min_distance - next_distance
                ) * len(global_resources)
            else:
                raw_q[direction_index] = 0.0

        raw_matrix.append(raw_q)
        resources = sorted(cluster)
        meta_values.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": len(resources),
                "resource_positions": resources,
                "nearest_resource": nearest_cluster_resource(position, cluster, map_data),
                "contains_energizer": any(resource in energizer_set for resource in resources),
                "min_distance": min_distance,
                # 以下字段明确区分完整 cluster 与本行真正参与 Global 距离计算的子集。
                # ``resource_positions`` 保持完整，避免近资源过滤导致跨行 cluster 身份漂移。
                "global_resource_positions": sorted(global_resources),
                "global_resource_count": len(global_resources),
                "ignored_near_resource_count": len(cluster) - len(global_resources),
                "nearest_global_resource": nearest_cluster_resource(position, global_resources, map_data),
                "global_min_distance": global_min_distance,
            }
        )

    norm_matrix = [normalize_global_cluster_q(row_values).tolist() for row_values in raw_matrix]
    return raw_matrix, norm_matrix, meta_values


def energizer_target_q_for_row(
    row: pd.Series,
    map_data: MapData,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
) -> tuple[list[list[float]], list[list[float]], list[dict[str, Any]]]:
    """为单行生成以每个剩余 energizer 为明确目标的候选 utility。

    输入语义：row 是单玩家视角的一行，至少包含 ``pacmanPos`` 和 ``energizers``；
    map_data/adjacent_map 来自统一地图常量。
    输出语义：返回 ``目标数×4`` raw utility、对应归一化矩阵和逐目标 meta。
    关键约束：候选不设置搜索半径。方向 utility 等于移动前后到目标的地图最短路
    距离减少量，因此远处 energizer 也能持续提供目标导向信息；墙方向保持 ``-inf``。
    ``target_position`` 是跨行稳定身份，不能使用每行重新编号的列表下标匹配目标。
    """

    position = parse_position(row["pacmanPos"])
    targets = sorted(set(parse_position_list(row.get("energizers", []))))
    raw_matrix: list[list[float]] = []
    meta_values: list[dict[str, Any]] = []

    for target_position in targets:
        current_distance = map_distance(map_data, position, target_position)
        raw_q = [float("-inf")] * len(DIRECTION_NAMES)
        for direction_index, direction in enumerate(DIRECTION_NAMES):
            adjacent_value = adjacent_map[position][direction]
            if not isinstance(adjacent_value, tuple):
                continue
            next_distance = map_distance(map_data, adjacent_value, target_position)
            if np.isfinite(current_distance) and np.isfinite(next_distance):
                # 正值表示第一步接近目标，负值表示远离目标。最短路距离来自地图
                # 常量，因此左右 tunnel 的连接会自然参与目标方向判断。
                raw_q[direction_index] = current_distance - next_distance
            else:
                raw_q[direction_index] = 0.0

        raw_matrix.append(raw_q)
        meta_values.append(
            {
                "target_id": target_position,
                "target_position": target_position,
                "min_distance": current_distance,
            }
        )

    norm_matrix = [normalize_global_cluster_q(row_values).tolist() for row_values in raw_matrix]
    return raw_matrix, norm_matrix, meta_values


def parse_ghost_status(value: Any) -> int:
    """把 ghost 状态整理为可用于候选过滤的整数。

    输入语义：value 来自 ``ghostN_status`` 或 ``ifscaredN``，允许 Python/numpy 整数
    以及表示整数的有限浮点数。
    输出语义：返回对应整数状态码。
    关键约束：缺失、无穷或非整数状态直接报错；状态 3 表示死亡 ghost，不能作为
    Approach 目标，其余已有状态继续沿用当前“正常鬼也可被主动追逐”的研究定义。
    """

    if value is None or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"非法 ghost 状态：{value!r}")
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"非法 ghost 状态：{value!r}")
    return int(numeric)


def parse_optional_position(value: Any) -> tuple[int, int] | None:
    """尝试把可缺失位置解析为坐标。

    输入语义：value 来自 ghost 位置字段，可能是坐标、空 tuple 或缺失标记。
    输出语义：合法坐标返回 ``(x, y)``，缺失或空位置返回 None。
    关键约束：本函数只容忍明确的缺失形式；其它畸形非空值仍由 ``parse_position``
    抛出异常，避免把数据损坏误当成 ghost 暂时不存在。
    """

    if value is None or (isinstance(value, (float, np.floating)) and pd.isna(value)):
        return None
    parsed = parse_literal_if_needed(value)
    if isinstance(parsed, (tuple, list)) and len(parsed) == 0:
        return None
    return parse_position(parsed)


def shortest_nonreversing_target_path_length(
    start_position: tuple[int, int],
    first_position: tuple[int, int],
    target_position: tuple[int, int],
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
    max_depth: int,
) -> int | None:
    """计算固定首步后首次到达目标的最短非立即折返路径。

    输入语义：start_position 是当前 Pacman 位置，first_position 是已经选定的下一格，
    target_position 是本候选 ghost；max_depth 按从 Pacman 当前格开始的移动步数计数。
    输出语义：在深度内可到达目标时返回总步数，否则返回 None。
    关键约束：搜索状态保存有向边 ``(上一格, 当前格)``，每一步禁止立即回到上一格，
    与现有共享路径树一致。其它 ghost 不作为静态障碍：它们会逐帧移动，当前坐标不能
    阻断未来整条路径；同时本候选只在到达目标 ghost 时结算奖励，因此不会累计其它鬼。
    """

    if max_depth < 1:
        return None
    if first_position == target_position:
        return 1

    # 同一张稀疏地图中最短非折返路径无需重复经过相同有向边；用 visited 控制循环，
    # 既保持 tunnel/环路可搜索，也避免深度较大时枚举指数数量的重复路径。
    frontier: deque[tuple[tuple[int, int], tuple[int, int], int]] = deque(
        [(start_position, first_position, 1)]
    )
    visited = {(start_position, first_position)}
    while frontier:
        previous, current, depth = frontier.popleft()
        if depth >= max_depth:
            continue
        for next_position in adjacent_map[current].values():
            if not isinstance(next_position, tuple) or next_position == previous:
                continue
            next_depth = depth + 1
            if next_position == target_position:
                return next_depth
            directed_state = (current, next_position)
            if directed_state in visited:
                continue
            visited.add(directed_state)
            frontier.append((current, next_position, next_depth))
    return None


def approach_target_q_for_row(
    row: pd.Series,
    map_data: MapData,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
    config: CalculateUtilityConfig,
) -> tuple[list[list[float]], list[list[float]], list[dict[str, Any]]]:
    """为单行生成按 ghost 身份分开的目标导向 Approach 候选。

    输入语义：row 是单玩家视角的一行，包含 ``pacmanPos``、两只 ghost 的位置与状态；
    config 中的 Approach 深度和折扣控制目标可见范围及距离衰减。
    输出语义：返回 ``非死亡 ghost 数×4`` raw utility、归一化矩阵和逐目标 meta。
    关键约束：每个候选只奖励命中自己的目标，另一只 ghost 不提供奖励、也不作为静态
    障碍。Ghost1/Ghost2 身份是跨行匹配键，位置允许随帧移动。05只生成候选，不读取
    真实动作，也不决定 context 的最终追逐目标。
    """

    position = parse_position(row["pacmanPos"])
    target_specs: list[tuple[str, tuple[int, int], int]] = []
    for ghost_number in range(1, 3):
        position_value = parse_optional_position(row.get(f"ghost{ghost_number}Pos"))
        status_column = (
            f"ghost{ghost_number}_status"
            if f"ghost{ghost_number}_status" in row.index
            else f"ifscared{ghost_number}"
        )
        status = parse_ghost_status(row[status_column])
        if position_value is None or position_value not in adjacent_map or status == 3:
            continue
        target_specs.append((f"ghost{ghost_number}", position_value, status))

    raw_matrix: list[list[float]] = []
    meta_values: list[dict[str, Any]] = []
    approach_depth = int(config.utility_config.approach_depth)
    discount = float(config.utility_config.approach_discount_factor)
    reward = float(map_data.reward_amount[8])

    for target_id, target_position, target_status in target_specs:
        raw_q = [float("-inf")] * len(DIRECTION_NAMES)
        first_hit_depths: list[int | None] = [None] * len(DIRECTION_NAMES)
        for direction_index, direction in enumerate(DIRECTION_NAMES):
            adjacent_value = adjacent_map[position][direction]
            if not isinstance(adjacent_value, tuple):
                continue
            path_length = shortest_nonreversing_target_path_length(
                position,
                adjacent_value,
                target_position,
                adjacent_map,
                approach_depth,
            )
            first_hit_depths[direction_index] = path_length
            if path_length is None:
                raw_q[direction_index] = 0.0
            else:
                raw_q[direction_index] = reward * (discount ** (path_length - 1))

        raw_matrix.append(raw_q)
        meta_values.append(
            {
                "target_id": target_id,
                "target_position": target_position,
                "target_status": target_status,
                "min_distance": map_distance(map_data, position, target_position),
                "first_hit_depths": first_hit_depths,
                "approach_depth": approach_depth,
                "discount_factor": discount,
            }
        )

    norm_matrix = [normalize_global_cluster_q(row_values).tolist() for row_values in raw_matrix]
    return raw_matrix, norm_matrix, meta_values


def normalize_global_cluster_q(values: Any) -> np.ndarray:
    """归一化单个 cluster 的四方向 global Q。

    输入语义：values 是长度为 4 的 raw Q，墙方向为 ``-inf``，可走方向可能为正、
    负或 0。
    输出语义：若存在正向推进，则用最大正值归一化；若没有任何正向推进，则所有
    可走方向都置为 0，墙方向保持 ``-inf``。
    关键约束：这个规则会把“只是没有变远”与“真正接近目标 cluster”区分开；
    06 选择 best global 时也会据此把全 0 候选视为无预测信息。
    """

    source = np.asarray(values, dtype=float)
    result = source.copy()
    finite_indices = np.where(~np.isinf(source))[0]
    if len(finite_indices) == 0:
        return result
    positive_values = source[finite_indices][source[finite_indices] > 0]
    if len(positive_values) == 0:
        result[finite_indices] = 0.0
        return result
    result[finite_indices] = result[finite_indices] / np.max(positive_values)
    return result


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
    关键约束：平移和归一化只能作用于输入副本，不能修改 DataFrame 中保存的 raw Q。
    """

    # DataFrame 的 deep copy 不会递归复制 object 单元格中的 numpy 数组。如果直接
    # 修改 q_values，生成 Q_norm 的同时会悄悄改写同一行的 raw Q。因此这里必须
    # 显式创建数值副本，并让后续全部操作只发生在副本上。
    working_q = np.asarray(q_values, dtype=float).copy()
    available_indices: list[int] = []
    for direction in DIRECTION_NAMES:
        adjacent_value = adjacent_map[position][direction]
        if adjacent_value is not None and not isinstance(adjacent_value, float):
            available_indices.append(DIRECTION_NAMES.index(direction))
    working_q[available_indices] = working_q[available_indices] - offset
    return normalize_with_inf(working_q)


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


def normalize_ghost_status_column(values: pd.Series, column: str) -> pd.Series:
    """把一列 ghost 状态规范为经过验证的整数。

    输入语义：values 是 04 阶段生成的 ifscared 状态列，column 用于错误信息定位。
    输出语义：返回与输入索引一致的 int64 Series。
    关键约束：缺失、无穷和非整数状态都直接报错；有限整值 float 可以安全转换，
    但绝不能因为其 dtype 是 float 就被解释成缺失状态。
    """

    numeric = pd.to_numeric(values, errors="raise")
    # 使用 float64 临时视图只用于有限性和整数性校验；最终返回值仍是整数。
    numeric_array = numeric.to_numpy(dtype=float, na_value=np.nan)
    invalid_finite = ~np.isfinite(numeric_array)
    if np.any(invalid_finite):
        invalid_indices = values.index[invalid_finite].tolist()[:5]
        raise ValueError(f"{column} 包含缺失或无穷状态，示例索引：{invalid_indices}")

    non_integer = numeric_array != np.floor(numeric_array)
    if np.any(non_integer):
        invalid_indices = values.index[non_integer].tolist()[:5]
        invalid_values = numeric_array[non_integer][:5].tolist()
        raise ValueError(
            f"{column} 包含非整数 ghost 状态，示例索引和值："
            f"{list(zip(invalid_indices, invalid_values))}"
        )

    return numeric.astype(np.int64)


def build_utility_estimation_input(data: pd.DataFrame) -> pd.DataFrame:
    """构造只供 Q 估计器使用的临时输入表。

    输入语义：data 是新 schema 的 corrected tile 表，ghost 状态字段已经是 int8。
    输出语义：返回带临时 ``pacman_dir`` 的 DataFrame，并把 ifscared 字段规范为
    经过有限性和整数性验证的整数。
    关键约束：当前标准 schema 已经使用整数状态；不得转换为 float，否则风险函数
    可能把有限浮点状态误判为缺失值，进而改变所有 ghost 相关路径的终止条件。
    """

    result = add_temporary_arrive_direction(data)
    missing_columns = [column for column in LEGACY_STATUS_COLUMNS if column not in result.columns]
    if missing_columns:
        raise ValueError(f"计算 utility 缺少 ghost 状态字段：{missing_columns}")
    for column in LEGACY_STATUS_COLUMNS:
        result[column] = normalize_ghost_status_column(result[column], column)
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


def global_cluster_candidate_columns() -> tuple[str, str, str]:
    """返回 05 阶段新增的 cluster global 候选字段。

    输入语义：无。
    输出语义：返回 raw 候选矩阵、归一化候选矩阵和候选 meta 三个无玩家前缀字段名。
    关键约束：这些字段只是候选池，不直接进入 06 的 GA 拟合；06b 会先选择 best
    cluster，再覆盖正式 ``global_Q/global_Q_norm`` 字段。
    """

    return ("global_utility_k", "global_utility_k_norm", "global_utility_k_meta")


def energizer_target_candidate_columns() -> tuple[str, str, str]:
    """返回05阶段新增的目标导向 Energizer 候选字段。

    输入语义：无。
    输出语义：返回 raw 候选矩阵、归一化候选矩阵和目标 meta 三个字段名。
    关键约束：一个矩阵行只对应一个明确 energizer 坐标；05不在多个目标间做选择。
    """

    return ("energizer_utility_k", "energizer_utility_k_norm", "energizer_utility_k_meta")


def approach_target_candidate_columns() -> tuple[str, str, str]:
    """返回05阶段新增的目标导向 Approach 候选字段。

    输入语义：无。
    输出语义：返回 raw 候选矩阵、归一化候选矩阵和 ghost 目标 meta 三个字段名。
    关键约束：矩阵每一行对应稳定 ghost 身份，而不是随位置或列表顺序变化的临时编号。
    """

    return ("approach_utility_k", "approach_utility_k_norm", "approach_utility_k_meta")


def utility_candidate_columns() -> tuple[str, ...]:
    """返回05保存的全部逐行目标候选字段。

    输入语义：无。
    输出语义：按 Global、Energizer、Approach 的稳定顺序返回字段。
    关键约束：该顺序只用于初始化和写回，不表示两种策略的优先级。
    """

    return (
        *global_cluster_candidate_columns(),
        *energizer_target_candidate_columns(),
        *approach_target_candidate_columns(),
    )


def prefixed_global_cluster_candidate_columns(player: str) -> list[str]:
    """返回某个玩家对应的 cluster global 候选字段名。

    输入语义：player 是 ``p1`` 或 ``p2``。
    输出语义：返回带玩家前缀的候选字段名列表。
    关键约束：候选字段与普通 Q 字段一起保存在 05 输出，供 06b context 预处理读取。
    """

    return [f"{player}_{column}" for column in global_cluster_candidate_columns()]


def prefixed_utility_candidate_columns(player: str) -> list[str]:
    """返回某个玩家的全部逐行目标候选字段名。

    输入语义：player 是 ``p1`` 或 ``p2``。
    输出语义：返回 Global、Energizer 与 Approach 候选的玩家前缀字段。
    关键约束：单人文件只调用现有玩家，不会创建缺失玩家的候选列。
    """

    return [f"{player}_{column}" for column in utility_candidate_columns()]


def append_global_cluster_candidate_columns(
    data: pd.DataFrame,
    map_data: MapData,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
    config: CalculateUtilityConfig,
) -> pd.DataFrame:
    """为单玩家视角追加 cluster global 候选字段。

    输入语义：data 是已经完成普通 Q 和 Q_norm 计算的单玩家视角表。
    输出语义：返回追加 ``global_utility_k*`` 三个字段的新 DataFrame。
    关键约束：本函数不修改 ``global_Q``；旧 global 仍保留作兼容检查，06b 会在
    context 级选择 best cluster 后覆盖正式 global 字段。
    """

    result = data.copy(deep=True)
    raw_values: list[list[list[float]]] = []
    norm_values: list[list[list[float]]] = []
    meta_values: list[list[dict[str, Any]]] = []
    for _, row in result.iterrows():
        raw_matrix, norm_matrix, meta = global_cluster_q_for_row(row, map_data, adjacent_map, config)
        raw_values.append(raw_matrix)
        norm_values.append(norm_matrix)
        meta_values.append(meta)

    result["global_utility_k"] = raw_values
    result["global_utility_k_norm"] = norm_values
    result["global_utility_k_meta"] = meta_values
    return result


def append_energizer_target_candidate_columns(
    data: pd.DataFrame,
    map_data: MapData,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
) -> pd.DataFrame:
    """为单玩家视角追加目标导向 Energizer 候选字段。

    输入语义：data 是已经完成普通 Q 和 Global 候选计算的单玩家视角表。
    输出语义：返回追加 ``energizer_utility_k*`` 三个字段的新 DataFrame。
    关键约束：不在05内根据动作选择目标，也不把是否最终吃到 energizer 混入 utility；
    context 级目标选择和事件结果修正分别属于06c与07c。
    """

    result = data.copy(deep=True)
    raw_values: list[list[list[float]]] = []
    norm_values: list[list[list[float]]] = []
    meta_values: list[list[dict[str, Any]]] = []
    for _, row in result.iterrows():
        raw_matrix, norm_matrix, meta = energizer_target_q_for_row(row, map_data, adjacent_map)
        raw_values.append(raw_matrix)
        norm_values.append(norm_matrix)
        meta_values.append(meta)

    result["energizer_utility_k"] = raw_values
    result["energizer_utility_k_norm"] = norm_values
    result["energizer_utility_k_meta"] = meta_values
    return result


def append_approach_target_candidate_columns(
    data: pd.DataFrame,
    map_data: MapData,
    adjacent_map: dict[tuple[int, int], dict[str, tuple[int, int] | float]],
    config: CalculateUtilityConfig,
) -> pd.DataFrame:
    """为单玩家视角追加按 ghost 身份分开的 Approach 候选字段。

    输入语义：data 已完成普通 Q、Global 和 Energizer 候选计算；config 提供当前正式
    Approach 搜索深度与距离衰减。
    输出语义：返回追加 ``approach_utility_k*`` 三个字段的新 DataFrame。
    关键约束：不覆盖旧 ``approach_Q``，使05输出仍可用于回归诊断；06c会在 context
    级选定目标后，把候选写入自己的正式 Approach 拟合视图。
    """

    result = data.copy(deep=True)
    raw_values: list[list[list[float]]] = []
    norm_values: list[list[list[float]]] = []
    meta_values: list[list[dict[str, Any]]] = []
    for _, row in result.iterrows():
        raw_matrix, norm_matrix, meta = approach_target_q_for_row(
            row,
            map_data,
            adjacent_map,
            config,
        )
        raw_values.append(raw_matrix)
        norm_values.append(norm_matrix)
        meta_values.append(meta)

    result["approach_utility_k"] = raw_values
    result["approach_utility_k_norm"] = norm_values
    result["approach_utility_k_meta"] = meta_values
    return result


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
    for column in (*prefixed_q_columns(player), *prefixed_utility_candidate_columns(player)):
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
    calculated_utility = append_global_cluster_candidate_columns(
        calculated_utility,
        map_data,
        adjacent_map,
        config,
    )
    calculated_utility = append_energizer_target_candidate_columns(
        calculated_utility,
        map_data,
        adjacent_map,
    )
    calculated_utility = append_approach_target_candidate_columns(
        calculated_utility,
        map_data,
        adjacent_map,
        config,
    )

    target_indices = frame_data.index[row_mask]
    for source_column in (*Q_COLUMNS, *Q_NORM_COLUMNS, *utility_candidate_columns()):
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
