"""基于前向 BFS 范围的新版 Pacman utility 计算。

本模块提供一套与当前 05 阶段并列的实验性 utility 计算方式。它不枚举路径树，
而是在每个候选方向上先让 Pacman 从当前位置走到下一格，然后从该下一格开始
做前向 BFS：当前位置和下一格都会被标记为已访问，后续搜索不能立刻回头，也
不能重复访问同一 tile。各策略在 BFS 访问到的节点上按深度汇总奖励或惩罚。
这样做的目标是让搜索半径可以显著加长，同时避免旧版“下一格周围最短路半径
求和”把身后资源也算进当前方向收益的问题。
"""

from __future__ import annotations

import ast
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from LoPS.hierarchical_utility import Q_COLUMNS, MapData, load_map_data


DIRECTION_NAMES: tuple[str, ...] = ("left", "right", "up", "down")
Q_NORM_COLUMNS: tuple[str, ...] = tuple(f"{column}_norm" for column in Q_COLUMNS)
PLAYER_PREFIXES: tuple[str, ...] = ("p1", "p2")

_WORKER_MAP_DATA: "RangeMapData | None" = None
_WORKER_CONFIG: "RangeUtilityConfig | None" = None


@dataclass(frozen=True)
class RangeUtilityConfig:
    """保存范围版 utility 的所有半径、衰减和奖励参数。

    输入语义：每个 radius 控制最短路距离的纳入范围，每个 decay 控制远处对象的
    影响衰减，reward/penalty 控制不同对象的基础价值。
    输出语义：配置对象会被 DataFrame 级和文件级处理函数共享。
    关键约束：这是新版 utility 定义，不要求与旧路径树 utility 数值一致。
    """

    local_radius: int = 10
    global_radius: int = 60
    global_ignore_radius: int = 10
    evade_radius: int = 10
    approach_radius: int = 34
    energizer_radius: int = 10
    no_energizer_radius: int = 12
    local_decay: float = 0.90
    global_decay: float = 0.97
    evade_decay: float = 0.80
    approach_decay: float = 0.95
    energizer_decay: float = 0.90
    no_energizer_decay: float = 0.90
    bean_reward: float = 2.0
    energizer_reward: float = 4.0
    ghost_reward: float = 8.0
    ghost_penalty: float = 8.0
    energizer_penalty: float = 4.0


@dataclass(frozen=True)
class RangeMapData:
    """保存范围版 utility 需要的地图快表。

    输入语义：由 `data/constant_data/map_constants.pkl` 读取并整理得到。
    输出语义：计算时可通过位置查询四方向邻居和任意目标点最短路距离。
    关键约束：本结构只读取地图常量，不在读取后做任何 tunnel、鬼屋或其它地图修正。
    """

    adjacent_by_position: dict[tuple[int, int], dict[str, Any]]
    distance_by_position: dict[tuple[int, int], dict[tuple[int, int], int | float]]


def load_range_map_data(constant_dir: str | Path) -> RangeMapData:
    """从当前项目常量目录读取范围版 utility 所需地图信息。

    输入语义：constant_dir 必须包含 `map_constants.pkl`。
    输出语义：返回 `RangeMapData`，其中包含四方向邻接和最短路距离。
    关键约束：所有地图合法性修正必须已经发生在地图常量生成阶段。
    """

    constant_dir = Path(constant_dir)
    map_data: MapData = load_map_data(constant_dir / "map_constants.pkl")
    distance_by_position = {
        position: dict(distance_row)
        for position, distance_row in map_data.distance_by_position.items()
    }
    # 距离表通常只保存不同点之间的最短路；范围汇总需要把当前位置到自身视为 0，
    # 这样“下一格正好有豆子/能量豆”的即时收益可以被纳入。
    for position in map_data.adjacent_by_position:
        distance_by_position.setdefault(position, {})[position] = 0
    return RangeMapData(
        adjacent_by_position=map_data.adjacent_by_position,
        distance_by_position=distance_by_position,
    )


def parse_literal_if_needed(value: Any) -> Any:
    """解析数据中可能以字符串形式保存的 Python 字面量。

    输入语义：value 可以是字符串坐标、字符串列表，也可以已经是 tuple/list。
    输出语义：字符串通过 `ast.literal_eval` 安全解析，其它值原样返回。
    关键约束：不使用 `eval`，避免把数据解析变成任意代码执行。
    """

    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def is_missing_scalar(value: Any) -> bool:
    """判断一个值是否表示缺失标记。

    输入语义：value 来自 pandas 行字段，可能是 None、NaN 或普通对象。
    输出语义：缺失返回 True，其它返回 False。
    关键约束：list/tuple 不能直接交给 `pd.isna`，否则会得到数组而不是布尔值。
    """

    return value is None or isinstance(value, (float, np.floating)) and pd.isna(value)


def parse_position(value: Any) -> tuple[int, int]:
    """把单个坐标字段解析为 `(x, y)` 整数 tuple。

    输入语义：value 可以是 tuple/list，也可以是字符串形式坐标。
    输出语义：返回长度为 2 的整数坐标。
    关键约束：缺失坐标不能传入本函数，调用方应先判断可计算行。
    """

    parsed = parse_literal_if_needed(value)
    if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
        raise ValueError(f"无法解析位置字段：{value!r}")
    return int(parsed[0]), int(parsed[1])


def parse_position_list(value: Any) -> list[tuple[int, int]]:
    """把对象列表字段解析为坐标列表。

    输入语义：value 可以是 `[(x, y), ...]`、字符串列表、空列表或缺失值。
    输出语义：返回坐标 tuple 列表；缺失值按空列表处理。
    关键约束：单个 `(x, y)` 不会被误当成两个对象，需要显式转换成一项列表。
    """

    if is_missing_scalar(value):
        return []
    parsed = parse_literal_if_needed(value)
    if is_missing_scalar(parsed):
        return []
    if isinstance(parsed, tuple) and len(parsed) == 2 and all(_looks_like_number(item) for item in parsed):
        return [parse_position(parsed)]
    if not isinstance(parsed, (tuple, list)):
        raise ValueError(f"无法解析对象列表字段：{value!r}")
    return [parse_position(item) for item in parsed]


def _looks_like_number(value: Any) -> bool:
    """判断对象是否像坐标中的数字。

    输入语义：value 是 tuple 中的一个元素。
    输出语义：整数、浮点整数和 numpy 数字返回 True。
    关键约束：该函数只用于区分单坐标 tuple 与坐标列表，不承担严格类型校验。
    """

    return isinstance(value, (int, float, np.integer, np.floating))


def discover_player_prefixes(data: pd.DataFrame) -> list[str]:
    """识别当前 joint-state 表中实际存在的玩家字段。

    输入语义：data 是 04 corrected tile 输出，可能包含 p1，也可能同时包含 p2。
    输出语义：返回存在完整位置、动作和可行动作列的玩家前缀。
    关键约束：单人数据缺失 p2 时自然跳过，不生成全 NaN 的 p2 utility 列。
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
    """构造某个玩家需要计算 utility 的行掩码。

    输入语义：data 是 joint-state 表，player 是 `p1` 或 `p2`。
    输出语义：True 表示该玩家在该行有合法位置且处于存活状态。
    关键约束：死亡行保留在输出中，但该玩家的 Q 字段保持 NaN。
    """

    position_column = f"{player}_pos"
    mask = data[position_column].notna()
    alive_column = f"{player}_alive"
    if alive_column in data.columns:
        mask &= data[alive_column].map(coerce_bool)
    return mask


def coerce_bool(value: Any) -> bool:
    """把数据字段中的布尔标记转换为 Python bool。

    输入语义：value 可能是 bool、0/1、字符串或缺失值。
    输出语义：返回是否为真。
    关键约束：字符串 `"False"`、`"0"`、`"nan"` 应被视为 False。
    """

    if is_missing_scalar(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "nan", "none"}
    return bool(value)


def prefixed_q_columns(player: str) -> list[str]:
    """返回某个玩家在输出表中对应的 raw Q 和 Q_norm 字段名。

    输入语义：player 是玩家前缀。
    输出语义：字段顺序为 raw Q 在前、Q_norm 在后。
    关键约束：字段名与当前 05 输出保持一致，便于下游替换输入目录。
    """

    return [f"{player}_{column}" for column in (*Q_COLUMNS, *Q_NORM_COLUMNS)]


def iter_forward_bfs_positions(
    current_position: tuple[int, int],
    first_step_position: tuple[int, int],
    map_data: RangeMapData,
    radius: int,
):
    """从候选下一格开始生成前向 BFS 可达位置。

    输入语义：current_position 是 Pacman 当前 tile，first_step_position 是某个候选方向
    的下一 tile，radius 是从 current_position 出发的最大步数。
    输出语义：逐个产出 ``(position, depth)``，其中 first_step_position 的 depth 为 1。
    关键约束：current_position 和 first_step_position 初始就被视为已访问。这样
    Pacman 从位置 1 走到位置 2 后，BFS 不会立即走回位置 1，也不会重复访问位置 2
    或之后访问过的任何 tile。这个约束让范围版 utility 更接近旧路径树“不立即
    回头”的几何语义，同时保留一次访问一个 tile 的快速搜索方式。
    """

    if radius <= 0:
        return

    visited = {current_position, first_step_position}
    queue: list[tuple[tuple[int, int], int]] = [(first_step_position, 1)]
    head = 0
    while head < len(queue):
        position, depth = queue[head]
        head += 1
        yield position, depth
        if depth >= radius:
            continue

        adjacent_positions = map_data.adjacent_by_position.get(position, {})
        for direction in DIRECTION_NAMES:
            next_position = adjacent_positions.get(direction)
            if not isinstance(next_position, tuple):
                continue
            if next_position in visited:
                continue
            visited.add(next_position)
            queue.append((next_position, depth + 1))


def calculate_range_utility_for_dataframe(
    frame_data: pd.DataFrame,
    map_data: RangeMapData,
    config: RangeUtilityConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """为一个 corrected tile joint-state 表计算范围版 utility。

    输入语义：frame_data 是 04 输出的一局或一个被试文件，map_data 是地图快表。
    输出语义：返回保留原行序、追加玩家前缀 Q 字段的 DataFrame，以及处理摘要。
    关键约束：不拆成长表、不删除死亡行，保证两个玩家仍在同一行表达同一局面。
    """

    config = RangeUtilityConfig() if config is None else config
    result = frame_data.reset_index(drop=True).copy()
    if "row_id" in result.columns:
        result.drop(columns=["row_id"], inplace=True)
    result.insert(0, "row_id", np.arange(len(result), dtype=np.int64))

    player_summaries: dict[str, dict[str, int]] = {}
    for player in discover_player_prefixes(frame_data):
        player_output, player_summary = calculate_player_range_utility(
            frame_data=frame_data.reset_index(drop=True),
            player=player,
            map_data=map_data,
            config=config,
        )
        player_summaries[player] = player_summary
        for column in player_output.columns:
            result[column] = player_output[column].to_numpy()

    summary = {
        "input_rows": int(frame_data.shape[0]),
        "output_rows": int(result.shape[0]),
        "players": player_summaries,
        "column_count": int(result.shape[1]),
    }
    return result, summary


def calculate_player_range_utility(
    frame_data: pd.DataFrame,
    player: str,
    map_data: RangeMapData,
    config: RangeUtilityConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """为 joint-state 表中的单个玩家计算范围版七策略 Q。

    输入语义：frame_data 是完整 joint-state 表，player 指定当前玩家。
    输出语义：返回只包含 `<player>_*_Q` 和 `<player>_*_Q_norm` 的 DataFrame。
    关键约束：所有公共对象字段按同一行读取，死亡或缺失位置行不计算 Q。
    """

    row_mask = build_player_alive_mask(frame_data, player)
    output = pd.DataFrame(index=frame_data.index)
    for column in prefixed_q_columns(player):
        output[column] = pd.Series([np.nan] * len(frame_data), index=frame_data.index, dtype=object)

    computed_rows = 0
    for row_index, row in frame_data.loc[row_mask].iterrows():
        row_q_values = calculate_row_range_q_values(row, player, map_data, config)
        computed_rows += 1
        for column in Q_COLUMNS:
            output.at[row_index, f"{player}_{column}"] = row_q_values[column]
            output.at[row_index, f"{player}_{column}_norm"] = normalize_q_values(row_q_values[column])

    return output, {
        "input_rows": int(frame_data.shape[0]),
        "computed_rows": int(computed_rows),
        "skipped_rows": int((~row_mask).sum()),
    }


def calculate_row_range_q_values(
    row: pd.Series,
    player: str,
    map_data: RangeMapData,
    config: RangeUtilityConfig,
) -> dict[str, np.ndarray]:
    """计算单行、单玩家在四个候选方向上的范围版 Q。

    输入语义：row 是 joint-state 的一行，player 决定读取哪个 Pacman 位置。
    输出语义：返回七个策略名到四方向 numpy 数组的映射。
    关键约束：不可走方向统一为 `-np.inf`，有限方向才参与后续归一化。
    """

    pacman_position = parse_position(row[f"{player}_pos"])
    if pacman_position not in map_data.adjacent_by_position:
        raise KeyError(f"地图邻接表中找不到 Pacman 位置：{pacman_position}")

    beans = parse_position_list(row.get("beans", []))
    energizers = parse_position_list(row.get("energizers", []))
    ghosts = [
        parse_optional_position(row.get("ghost1Pos")),
        parse_optional_position(row.get("ghost2Pos")),
    ]
    ghost_status = [row.get("ifscared1"), row.get("ifscared2")]

    q_values = {column: np.full(len(DIRECTION_NAMES), -np.inf, dtype=float) for column in Q_COLUMNS}
    for direction_index, direction in enumerate(DIRECTION_NAMES):
        next_position = map_data.adjacent_by_position[pacman_position][direction]
        if not isinstance(next_position, tuple):
            continue
        q_values["local_Q"][direction_index] = calculate_local_score(
            pacman_position,
            next_position,
            beans,
            energizers,
            map_data,
            config,
        )
        q_values["global_Q"][direction_index] = calculate_global_score(
            pacman_position,
            next_position,
            beans,
            map_data,
            config,
        )
        q_values["energizer_Q"][direction_index] = calculate_resource_score(
            current_position=pacman_position,
            first_step_position=next_position,
            targets=energizers,
            map_data=map_data,
            radius=config.energizer_radius,
            reward=config.energizer_reward,
            decay=config.energizer_decay,
        )
        q_values["approach_Q"][direction_index] = calculate_approach_score(
            pacman_position,
            next_position,
            ghosts,
            ghost_status,
            map_data,
            config,
        )
        q_values["evade_blinky_Q"][direction_index] = calculate_evade_score(
            pacman_position,
            next_position,
            ghosts[0],
            ghost_status[0],
            map_data,
            config,
        )
        q_values["evade_clyde_Q"][direction_index] = calculate_evade_score(
            pacman_position,
            next_position,
            ghosts[1],
            ghost_status[1],
            map_data,
            config,
        )
        q_values["no_energizer_Q"][direction_index] = calculate_no_energizer_score(
            pacman_position,
            next_position,
            energizers,
            map_data,
            config,
        )
    return q_values


def parse_optional_position(value: Any) -> tuple[int, int] | None:
    """解析可缺失的 ghost 位置字段。

    输入语义：value 可能是坐标、字符串坐标、空 tuple 或缺失值。
    输出语义：合法坐标返回 tuple；缺失或空 tuple 返回 None。
    关键约束：None 表示该 ghost 不参与当前策略得分。
    """

    if is_missing_scalar(value):
        return None
    parsed = parse_literal_if_needed(value)
    if parsed in ((), []):
        return None
    return parse_position(parsed)


def calculate_local_score(
    current_position: tuple[int, int],
    first_step_position: tuple[int, int],
    beans: list[tuple[int, int]],
    energizers: list[tuple[int, int]],
    map_data: RangeMapData,
    config: RangeUtilityConfig,
) -> float:
    """计算 Local 策略在候选方向前方 BFS 区域内的资源价值。

    输入语义：current_position 是当前位置，first_step_position 是候选方向下一格，
    beans/energizers 是当前剩余资源。
    输出语义：返回从 first_step_position 开始、不能回头和不能重复访问的 BFS
    区域内豆子与能量豆的距离衰减奖励和。
    关键约束：first_step_position 的资源会以 depth=1 计入奖励；当前位置不会
    被计入，也不会被后续 BFS 重新访问。
    """

    bean_score = calculate_resource_score(
        current_position=current_position,
        first_step_position=first_step_position,
        targets=beans,
        map_data=map_data,
        radius=config.local_radius,
        reward=config.bean_reward,
        decay=config.local_decay,
    )
    energizer_score = calculate_resource_score(
        current_position=current_position,
        first_step_position=first_step_position,
        targets=energizers,
        map_data=map_data,
        radius=config.local_radius,
        reward=config.energizer_reward,
        decay=config.local_decay,
    )
    return bean_score + energizer_score


def calculate_global_score(
    current_position: tuple[int, int],
    first_step_position: tuple[int, int],
    beans: list[tuple[int, int]],
    map_data: RangeMapData,
    config: RangeUtilityConfig,
) -> float:
    """计算 Global 策略在候选方向前方 BFS 区域内的远处豆子价值。

    输入语义：current_position 是当前位置，first_step_position 是候选方向下一格，
    beans 是剩余豆子坐标。
    输出语义：返回 BFS 前方区域中，深度超过 `global_ignore_radius` 且不超过
    `global_radius` 的豆子奖励和。
    关键约束：Global 仍保留“忽略近处资源”的策略定义，因此第一步资源通常不会
    被 Global 计入；前向 BFS 只改变可达区域的几何约束。
    """

    return calculate_resource_score(
        current_position=current_position,
        first_step_position=first_step_position,
        targets=beans,
        map_data=map_data,
        radius=config.global_radius,
        reward=config.bean_reward,
        decay=config.global_decay,
        min_exclusive_distance=config.global_ignore_radius,
    )


def calculate_approach_score(
    current_position: tuple[int, int],
    first_step_position: tuple[int, int],
    ghosts: list[tuple[int, int] | None],
    ghost_status: list[Any],
    map_data: RangeMapData,
    config: RangeUtilityConfig,
) -> float:
    """计算 Approach 策略在候选方向前方 BFS 区域内的 ghost 接近价值。

    输入语义：current_position 是当前位置，first_step_position 是候选方向下一格，
    ghosts/status 描述两只 ghost 当前状态。
    输出语义：返回前向 BFS 半径内非死亡 ghost 的距离衰减奖励和。
    关键约束：Approach 表达“追鬼/靠近鬼”，不是只追可吃 ghost；因此正常、
    危险和 scared ghost 都是目标，只有死亡状态 3 不计入。
    """

    score = 0.0
    target_status_by_position: dict[tuple[int, int], list[Any]] = {}
    for ghost_position, status in zip(ghosts, ghost_status):
        if ghost_position is None or not is_approach_target_ghost(status):
            continue
        target_status_by_position.setdefault(ghost_position, []).append(status)
    if not target_status_by_position:
        return 0.0

    for position, depth in iter_forward_bfs_positions(
        current_position,
        first_step_position,
        map_data,
        config.approach_radius,
    ):
        if position not in target_status_by_position:
            continue
        # 两只 ghost 理论上可以处在同一 tile；这种情况下两只都计入接近价值。
        score += len(target_status_by_position[position]) * decayed_value(
            depth,
            config.ghost_reward,
            config.approach_decay,
        )
    return score


def calculate_evade_score(
    current_position: tuple[int, int],
    first_step_position: tuple[int, int],
    ghost_position: tuple[int, int] | None,
    status: Any,
    map_data: RangeMapData,
    config: RangeUtilityConfig,
) -> float:
    """计算单只 ghost 对候选方向前方 BFS 区域的 Evade 惩罚。

    输入语义：current_position 是当前位置，first_step_position 是候选方向下一格，
    ghost_position/status 描述某一只 ghost。
    输出语义：危险 ghost 出现在前向 BFS 半径内时返回负惩罚，否则返回 0。
    关键约束：状态 1/2 视为危险；scared 或 dead ghost 不产生 evade 惩罚。
    """

    if ghost_position is None or not is_dangerous_ghost(status):
        return 0.0
    for position, depth in iter_forward_bfs_positions(
        current_position,
        first_step_position,
        map_data,
        config.evade_radius,
    ):
        if position == ghost_position:
            return -decayed_value(depth, config.ghost_penalty, config.evade_decay)
    return 0.0


def calculate_no_energizer_score(
    current_position: tuple[int, int],
    first_step_position: tuple[int, int],
    energizers: list[tuple[int, int]],
    map_data: RangeMapData,
    config: RangeUtilityConfig,
) -> float:
    """计算 NoEnergizer 策略对候选方向前方 energizer 的回避价值。

    输入语义：current_position 是当前位置，first_step_position 是候选方向下一格，
    energizers 是剩余能量豆坐标。
    输出语义：前向 BFS 半径内能量豆越近，返回越大的负惩罚。
    关键约束：该策略不额外奖励普通豆子，用来表达“避免吃能量豆”的倾向。
    """

    return -calculate_resource_score(
        current_position=current_position,
        first_step_position=first_step_position,
        targets=energizers,
        map_data=map_data,
        radius=config.no_energizer_radius,
        reward=config.energizer_penalty,
        decay=config.no_energizer_decay,
    )


def calculate_resource_score(
    current_position: tuple[int, int],
    first_step_position: tuple[int, int],
    targets: list[tuple[int, int]],
    map_data: RangeMapData,
    radius: int,
    reward: float,
    decay: float,
    min_exclusive_distance: int = -1,
) -> float:
    """按前向 BFS 深度汇总一组目标对象的距离衰减价值。

    输入语义：current_position 是当前位置，first_step_position 是候选方向下一格，
    targets 是目标坐标列表，radius 是最大 BFS 深度。
    输出语义：返回所有满足深度条件目标的奖励和。
    关键约束：`min_exclusive_distance` 用于排除过近对象，默认不排除任何合法距离。
    depth=1 的 first_step_position 会被计入，因此从位置 1 走到位置 2 这一步
    吃到的资源不会丢失。
    """

    target_positions = set(targets)
    if not target_positions:
        return 0.0

    score = 0.0
    for position, depth in iter_forward_bfs_positions(
        current_position,
        first_step_position,
        map_data,
        radius,
    ):
        if min_exclusive_distance < depth <= radius and position in target_positions:
            score += decayed_value(depth, reward, decay)
    return score


def shortest_distance(
    source: tuple[int, int],
    target: tuple[int, int],
    map_data: RangeMapData,
) -> float:
    """查询两个地图坐标之间的最短路距离。

    输入语义：source/target 是地图坐标，map_data 保存距离表。
    输出语义：返回有限距离；不可达或缺失目标返回 `np.inf`。
    关键约束：不临时搜索地图，只使用常量文件已经生成的最短路表。
    """

    if source == target:
        return 0.0
    return float(map_data.distance_by_position.get(source, {}).get(target, np.inf))


def decayed_value(distance: float, reward: float, decay: float) -> float:
    """把对象基础奖励转换为距离衰减后的价值。

    输入语义：distance 是最短路距离，reward 是基础奖励，decay 是每步衰减系数。
    输出语义：返回 `reward * decay ** max(distance - 1, 0)`。
    关键约束：距离 0 和 1 都视为即时附近对象，不再额外放大奖励。
    """

    return float(reward) * (float(decay) ** max(float(distance) - 1.0, 0.0))


def is_approach_target_ghost(status: Any) -> bool:
    """判断 ghost 是否是 Approach 策略的目标。

    输入语义：status 是 ifscared 状态码，可能为 numpy 数字或缺失值。
    输出语义：非缺失且状态不等于 3 时返回 True。
    关键约束：这里对齐旧版 Approach 语义，追正常 ghost 导致死亡也仍然算作
    approach 目标；状态 3 表示死亡 ghost，不参与接近价值计算。
    """

    if is_missing_scalar(status):
        return False
    return int(status) != 3


def is_dangerous_ghost(status: Any) -> bool:
    """判断 ghost 是否处于需要回避的危险状态。

    输入语义：status 是 ifscared 状态码。
    输出语义：状态 1 或 2 返回 True。
    关键约束：缺失、死亡和 scared 状态均不产生 evade 惩罚。
    """

    if is_missing_scalar(status):
        return False
    return int(status) in {1, 2}


def normalize_q_values(values: Any) -> np.ndarray:
    """把四方向 raw Q 转换成下游拟合使用的 0-1 归一化 Q。

    输入语义：values 是长度为 4 的 Q 数组，墙方向为 `-np.inf`。
    输出语义：返回同形状数组，有限方向非负归一化，墙方向保持 `-np.inf`。
    关键约束：如果有限方向全为 0，则保持全 0，避免制造伪信息。
    """

    source = np.asarray(values, dtype=float)
    result = source.copy()
    finite_mask = np.isfinite(source)
    if not finite_mask.any():
        return result

    finite_values = source[finite_mask]
    if np.allclose(finite_values, 0.0):
        result[finite_mask] = 0.0
        return result

    # 对包含负惩罚的策略，先把有限方向整体平移到非负区间；正向奖励策略不平移。
    shifted_values = finite_values.copy()
    min_value = float(np.min(shifted_values))
    if min_value < 0:
        shifted_values = shifted_values - min_value
    max_value = float(np.max(shifted_values))
    result[finite_mask] = shifted_values / max_value if max_value > 0 else 0.0
    return result


def process_range_utility_file(
    input_path: str | Path,
    output_path: str | Path,
    map_data: RangeMapData,
    config: RangeUtilityConfig | None = None,
) -> dict[str, Any]:
    """处理单个 04 corrected tile pickle 并保存范围版 utility 输出。

    输入语义：input_path 是一个嵌套目录中的 `.pkl` 文件，output_path 是目标文件。
    输出语义：写出同名 DataFrame，并返回文件摘要。
    关键约束：保存路径由调用方决定，函数本身不假设任务类型目录名。
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    config = RangeUtilityConfig() if config is None else config
    with input_path.open("rb") as file:
        frame_data = pickle.load(file)
    calculated_data, summary = calculate_range_utility_for_dataframe(frame_data, map_data, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(calculated_data, file)
    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        **summary,
    }


def process_range_utility_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    map_data: RangeMapData,
    config: RangeUtilityConfig | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """批量处理 corrected tile 嵌套目录并生成范围版 utility 数据。

    输入语义：input_dir 包含 `comp/*.pkl`、`coop/*.pkl` 等任务子目录。
    输出语义：每个输入文件按相同相对路径写入 output_dir。
    关键约束：只支持当前嵌套目录结构；workers 大于 1 时进行文件级并行。
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    config = RangeUtilityConfig() if config is None else config
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    input_paths = sorted(path for path in input_dir.glob("*/*.pkl") if path.is_file())
    if not input_paths:
        raise FileNotFoundError(f"输入目录中没有嵌套 pickle 文件：{input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(input_path, output_dir / input_path.relative_to(input_dir)) for input_path in input_paths]
    if workers <= 1:
        return [process_range_utility_file(input_path, output_path, map_data, config) for input_path, output_path in tasks]

    with ProcessPoolExecutor(
        max_workers=min(workers, len(tasks)),
        initializer=_init_range_worker,
        initargs=(map_data, config),
    ) as executor:
        return list(executor.map(_process_range_utility_task, tasks))


def _init_range_worker(map_data: RangeMapData, config: RangeUtilityConfig) -> None:
    """初始化文件级并行 worker 中的只读地图和配置。

    输入语义：map_data/config 由主进程在创建进程池时传入。
    输出语义：写入进程内全局变量，后续任务只传文件路径。
    关键约束：这些对象在 worker 内只读使用，避免跨文件状态污染。
    """

    global _WORKER_MAP_DATA, _WORKER_CONFIG
    _WORKER_MAP_DATA = map_data
    _WORKER_CONFIG = config


def _process_range_utility_task(task: tuple[Path, Path]) -> dict[str, Any]:
    """执行并行池中的单个文件任务。

    输入语义：task 包含输入路径和输出路径。
    输出语义：返回 `process_range_utility_file` 的摘要。
    关键约束：保持为顶层函数，保证 multiprocessing 可以 pickle 调用。
    """

    if _WORKER_MAP_DATA is None or _WORKER_CONFIG is None:
        raise RuntimeError("range utility worker 尚未初始化地图和配置。")
    input_path, output_path = task
    return process_range_utility_file(input_path, output_path, _WORKER_MAP_DATA, _WORKER_CONFIG)


def summarize_players(summaries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """汇总目录处理结果中的玩家级行数统计。

    输入语义：summaries 来自 `process_range_utility_directory`。
    输出语义：返回每个玩家的 input/computed/skipped 总行数。
    关键约束：该函数只服务命令行日志，不影响保存数据。
    """

    result: dict[str, dict[str, int]] = {}
    for summary in summaries:
        players = summary.get("players", {})
        if not isinstance(players, dict):
            continue
        for player, player_summary in players.items():
            if not isinstance(player_summary, dict):
                continue
            accumulator = result.setdefault(str(player), {"input_rows": 0, "computed_rows": 0, "skipped_rows": 0})
            for key in accumulator:
                accumulator[key] += int(player_summary.get(key, 0))
    return result


def config_to_dict(config: RangeUtilityConfig) -> dict[str, Any]:
    """把范围版 utility 配置转换为可 JSON 序列化的字典。

    输入语义：config 是 dataclass 配置对象。
    输出语义：返回普通字典。
    关键约束：该函数用于命令行摘要，避免上层脚本直接依赖 dataclasses 细节。
    """

    return asdict(config)
