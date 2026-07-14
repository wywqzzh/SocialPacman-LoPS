"""事件硬边界版 Social Pacman 动态策略拟合。

本模块是 06 动态策略拟合的实验新版。它复用旧 06 的权重拟合器和输出字段，
只替换 context 划分方式：先按玩家行为事件与公共吃鬼事件生成不可跨越的硬边界，
再把队友吃 Energizer 造成的公共环境变化作为软边界，最后在不跨硬边界的前提下
合并软边界造成的过短段。玩家掉头只作为段内动作参与策略 likelihood，不再触发
context 切分。
"""

from __future__ import annotations

import ast
import copy
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from LoPS.dynamic_strategy_fitting import (
    DEFAULT_AGENTS,
    DynamicStrategyFittingConfig,
    all_directions_nan,
    discover_player_prefixes,
    fit_all_segments,
    initialize_player_result,
    prepare_fitting_dataframe,
)
OPPOSITE_DIRECTIONS: dict[str, str] = {
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up",
}
DIRECTION_NAMES: tuple[str, ...] = ("left", "right", "up", "down")
DIRECTION_TO_INDEX: dict[str, int] = {direction: index for index, direction in enumerate(DIRECTION_NAMES)}
PLAYER_PREFIXES: tuple[str, ...] = ("p1", "p2")


def parse_position_collection(value: Any) -> list[Any]:
    """解析 beans/energizers 这种坐标列表字段。

    输入语义：value 可以是 list/tuple、字符串形式列表、空值或 NaN。
    输出语义：返回列表；缺失值返回空列表。
    关键约束：本函数只用于统计资源数量，不关心坐标是否合法，因此不把每个元素
    强制转换为坐标 tuple。
    """

    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "nan", "none"}:
            return []
        parsed = ast.literal_eval(stripped)
    else:
        parsed = value
    if isinstance(parsed, np.ndarray):
        parsed = parsed.tolist()
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return []


def parse_position_or_none(value: Any) -> tuple[int, int] | None:
    """把单个坐标字段解析成整数 tuple。

    输入语义：value 可以是字符串 ``"(x, y)"``、tuple/list、None 或 NaN。
    输出语义：合法坐标返回 ``(x, y)``，缺失或无法解析时返回 None。
    关键约束：本函数用于事件归因，宁可跳过缺失坐标，也不把错误坐标强行归属给玩家。
    """

    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if isinstance(parsed, np.ndarray):
        parsed = parsed.tolist()
    if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
        return None
    try:
        return int(parsed[0]), int(parsed[1])
    except (TypeError, ValueError):
        return None


def parse_position_set(value: Any) -> set[tuple[int, int]]:
    """把资源列表字段解析成坐标集合。

    输入语义：value 来自 ``beans`` 或 ``energizers``，可能是字符串列表、真实列表或缺失值。
    输出语义：返回资源坐标集合；无法解析的元素会被跳过。
    关键约束：事件归因只需要比较消失坐标是否等于玩家到达坐标，因此使用 set 做差。
    """

    positions: set[tuple[int, int]] = set()
    for item in parse_position_collection(value):
        position = parse_position_or_none(item)
        if position is not None:
            positions.add(position)
    return positions


def parse_global_cluster_matrix(value: Any) -> np.ndarray:
    """解析 05 保存的 cluster global 候选 utility 矩阵。

    输入语义：value 来自 ``<player>_global_utility_k`` 或
    ``<player>_global_utility_k_norm``，通常是 ``n×4`` list，也可能是 NaN。
    输出语义：返回二维 numpy 数组；缺失时返回形状 ``(0, 4)`` 的空矩阵。
    关键约束：矩阵第二维必须是四方向；若数据损坏则直接报错，避免 06 静默错位。
    """

    if value is None or isinstance(value, float) and pd.isna(value):
        return np.empty((0, len(DIRECTION_NAMES)), dtype=float)
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if parsed is None or isinstance(parsed, float) and pd.isna(parsed):
        return np.empty((0, len(DIRECTION_NAMES)), dtype=float)
    matrix = np.asarray(parsed, dtype=float)
    if matrix.size == 0:
        return np.empty((0, len(DIRECTION_NAMES)), dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(DIRECTION_NAMES):
        raise ValueError(f"cluster global utility 矩阵必须是 n×4，实际 shape={matrix.shape}")
    return matrix


def parse_global_cluster_meta(value: Any) -> list[dict[str, Any]]:
    """解析 05 保存的 cluster global 候选 meta。

    输入语义：value 来自 ``<player>_global_utility_k_meta``，与候选矩阵行一一对应。
    输出语义：返回 meta 字典列表；缺失时返回空列表。
    关键约束：meta 中的 resource_positions 会被标准化为 set，用于在 context 后续行中
    匹配同一目标 cluster。
    """

    if value is None or isinstance(value, float) and pd.isna(value):
        return []
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if parsed is None or isinstance(parsed, float) and pd.isna(parsed):
        return []
    if isinstance(parsed, np.ndarray):
        parsed = parsed.tolist()
    if not isinstance(parsed, list):
        return []
    return [dict(item) for item in parsed if isinstance(item, dict)]


def meta_resource_set(meta: dict[str, Any]) -> set[tuple[int, int]]:
    """从单个 cluster meta 中提取资源坐标集合。

    输入语义：meta 是 05 生成的候选 cluster 描述。
    输出语义：返回 ``resource_positions`` 对应的坐标集合。
    关键约束：资源坐标用于跨行匹配同一目标 cluster；无法解析的坐标会被跳过。
    """

    return {
        position
        for position in (parse_position_or_none(item) for item in meta.get("resource_positions", []))
        if position is not None
    }


def q_like_zero(value: Any) -> list[float]:
    """根据已有 Q 的墙方向构造无信息四方向 Q。

    输入语义：value 是当前行已有的 ``global_Q`` 或 ``global_Q_norm``。
    输出语义：可走方向置 0，墙方向保持 ``-inf``。
    关键约束：当某个 context 没有可用 best global 候选时，不能把墙方向也置为 0，
    否则后续拟合会把不可走方向当作并列预测方向。
    """

    q_values = np.asarray(value, dtype=float)
    if q_values.ndim != 1 or q_values.shape[0] != len(DIRECTION_NAMES):
        return [0.0] * len(DIRECTION_NAMES)
    return [float("-inf") if np.isneginf(each) else 0.0 for each in q_values]


def positive_prediction_indices(q_values: np.ndarray) -> list[int]:
    """从一个四方向 Q 中提取有正向信息的最大方向集合。

    输入语义：q_values 是 normalized cluster global Q。
    输出语义：返回并列最大且最大值大于 0 的方向下标；若全 0、全负或全墙则返回空。
    关键约束：这正是 06 选择 best global 时“无正向推进不算有效预测”的规则。
    """

    finite_indices = np.where(~np.isinf(q_values))[0]
    if len(finite_indices) == 0:
        return []
    finite_values = q_values[finite_indices]
    max_value = float(np.max(finite_values))
    if max_value <= 0:
        return []
    return [int(index) for index in finite_indices if q_values[index] == max_value]


def match_context_cluster_index(
    row_meta: list[dict[str, Any]],
    start_resource_positions: set[tuple[int, int]],
) -> int | None:
    """在某一行候选 cluster 中匹配 context 起点选中的目标 cluster。

    输入语义：row_meta 是该行候选 cluster meta，start_resource_positions 是 context
    起点 best 候选的资源集合。
    输出语义：返回该行中最可能对应同一目标 cluster 的矩阵行号；无法匹配时返回 None。
    关键约束：05 每行都会根据剩余资源重新聚类，行内 cluster_id 不是跨行稳定主键；
    因此这里优先用资源坐标重叠匹配，避免 cluster 顺序变化导致 best global 错位。
    """

    best_index: int | None = None
    best_key: tuple[int, int, float] | None = None
    for index, meta in enumerate(row_meta):
        resources = meta_resource_set(meta)
        overlap = len(resources & start_resource_positions)
        if overlap <= 0:
            continue
        cluster_size = int(meta.get("cluster_size", len(resources)))
        min_distance = float(meta.get("min_distance", np.inf))
        # overlap 越大越好；cluster_size 作为次级稳定项；距离越近越好，因此取负数。
        key = (overlap, cluster_size, -min_distance)
        if best_key is None or key > best_key:
            best_index = index
            best_key = key
    return best_index


def score_context_global_candidate(
    data: pd.DataFrame,
    context: tuple[int, int],
    player: str,
    start_meta: dict[str, Any],
) -> dict[str, Any]:
    """计算一个 context 起点 cluster 候选对真实动作的解释能力。

    输入语义：data 是单玩家拟合临时表，context 是全局半开区间，start_meta 是
    context 起点某个候选 cluster 的 meta。
    输出语义：返回概率准确率、集合准确率和用于破平的 cluster 属性。
    关键约束：只有真实动作行进入分母；若候选 Q 全 0 或没有正向推进，该动作贡献 0。
    """

    start, end = context
    start_resource_positions = meta_resource_set(start_meta)
    valid_actions = 0
    probability_credit = 0.0
    set_hits = 0
    matrix_column = f"{player}_global_utility_k_norm"
    meta_column = f"{player}_global_utility_k_meta"

    for row_index in range(start, end):
        action = data.at[row_index, "action_dir"]
        if not isinstance(action, str):
            continue
        valid_actions += 1
        row_meta = parse_global_cluster_meta(data.at[row_index, meta_column])
        matched_index = match_context_cluster_index(row_meta, start_resource_positions)
        if matched_index is None:
            continue
        matrix = parse_global_cluster_matrix(data.at[row_index, matrix_column])
        if matched_index >= matrix.shape[0]:
            continue
        prediction_indices = positive_prediction_indices(matrix[matched_index])
        if not prediction_indices:
            continue
        true_index = DIRECTION_TO_INDEX[action]
        if true_index in prediction_indices:
            set_hits += 1
            probability_credit += 1.0 / len(prediction_indices)

    cluster_size = int(start_meta.get("cluster_size", len(start_resource_positions)))
    start_distance = float(start_meta.get("min_distance", np.inf))
    return {
        "cluster_id": start_meta.get("cluster_id"),
        "cluster_size": cluster_size,
        "start_distance": start_distance,
        "valid_actions": valid_actions,
        "prob_accuracy": probability_credit / valid_actions if valid_actions else 0.0,
        "set_accuracy": set_hits / valid_actions if valid_actions else 0.0,
        "meta": start_meta,
    }


def choose_best_global_candidate(
    data: pd.DataFrame,
    context: tuple[int, int],
    player: str,
) -> dict[str, Any] | None:
    """为一个 player-context 选择解释力最强的 global cluster。

    输入语义：data 是单玩家拟合临时表，context 是半开区间，player 是 ``p1`` 或 ``p2``。
    输出语义：返回 best cluster 的评分字典；没有候选时返回 None。
    关键约束：best 选择发生在 GA 拟合前，后续拟合仍只看到一个普通 ``global_Q_norm``。
    破平顺序为概率准确率、集合准确率、cluster size、起点距离、cluster_id。
    """

    start, _ = context
    meta_column = f"{player}_global_utility_k_meta"
    if meta_column not in data.columns:
        return None
    start_meta_values = parse_global_cluster_meta(data.at[start, meta_column])
    if not start_meta_values:
        return None

    scored_candidates = [
        score_context_global_candidate(data, context, player, start_meta)
        for start_meta in start_meta_values
    ]
    if not scored_candidates:
        return None
    return max(
        scored_candidates,
        key=lambda item: (
            item["prob_accuracy"],
            item["set_accuracy"],
            item["cluster_size"],
            -item["start_distance"],
            -int(item["cluster_id"]) if item["cluster_id"] is not None else 0,
        ),
    )


def apply_best_global_candidates(
    data: pd.DataFrame,
    contexts: list[tuple[int, int]],
    player: str,
) -> pd.DataFrame:
    """在 context 级选择 best global，并覆盖正式 global Q 字段。

    输入语义：data 是 ``prepare_fitting_dataframe`` 生成的单玩家临时表，contexts 是
    已经划分好的全局半开区间。
    输出语义：返回新 DataFrame，其中 ``global_Q/global_Q_norm`` 和玩家前缀
    ``<player>_global_Q/<player>_global_Q_norm`` 均已替换为 best cluster 候选。
    关键约束：如果输入没有 05 cluster global 候选字段，本函数保持原 global Q 不变，
    但仍添加 best cluster 解释列，便于兼容旧数据调试。
    """

    result = data.copy(deep=True)
    # prepare_fitting_dataframe 为 GA 只构造了 ``global_Q_norm`` 这类归一化列；
    # cluster global 预处理还需要同步替换 raw ``global_Q``，因此这里从玩家前缀
    # 字段补出临时无前缀列，保持后续写回逻辑集中。
    prefixed_global_column = f"{player}_global_Q"
    prefixed_global_norm_column = f"{player}_global_Q_norm"
    if "global_Q" not in result.columns and prefixed_global_column in result.columns:
        result["global_Q"] = result[prefixed_global_column].to_numpy()
    if "global_Q_norm" not in result.columns and prefixed_global_norm_column in result.columns:
        result["global_Q_norm"] = result[prefixed_global_norm_column].to_numpy()

    best_columns = {
        "best_global_cluster_id": np.nan,
        "best_global_cluster_prob_accuracy": np.nan,
        "best_global_cluster_set_accuracy": np.nan,
        "best_global_cluster_meta": np.nan,
    }
    for column, default_value in best_columns.items():
        result[column] = pd.Series([default_value] * len(result), index=result.index, dtype=object)

    required_columns = {
        f"{player}_global_utility_k",
        f"{player}_global_utility_k_norm",
        f"{player}_global_utility_k_meta",
    }
    if not required_columns.issubset(result.columns):
        return result

    raw_column = f"{player}_global_utility_k"
    norm_column = f"{player}_global_utility_k_norm"
    meta_column = f"{player}_global_utility_k_meta"

    for context in contexts:
        best_candidate = choose_best_global_candidate(result, context, player)
        start, end = context
        if best_candidate is None:
            continue
        start_resource_positions = meta_resource_set(best_candidate["meta"])
        for row_index in range(start, end):
            row_meta = parse_global_cluster_meta(result.at[row_index, meta_column])
            matched_index = match_context_cluster_index(row_meta, start_resource_positions)
            if matched_index is None:
                raw_q = q_like_zero(result.at[row_index, "global_Q"])
                norm_q = q_like_zero(result.at[row_index, "global_Q_norm"])
            else:
                raw_matrix = parse_global_cluster_matrix(result.at[row_index, raw_column])
                norm_matrix = parse_global_cluster_matrix(result.at[row_index, norm_column])
                if matched_index < raw_matrix.shape[0] and matched_index < norm_matrix.shape[0]:
                    raw_q = raw_matrix[matched_index].tolist()
                    norm_q = norm_matrix[matched_index].tolist()
                else:
                    raw_q = q_like_zero(result.at[row_index, "global_Q"])
                    norm_q = q_like_zero(result.at[row_index, "global_Q_norm"])

            result.at[row_index, "global_Q"] = raw_q
            result.at[row_index, "global_Q_norm"] = norm_q
            result.at[row_index, prefixed_global_column] = raw_q
            result.at[row_index, prefixed_global_norm_column] = norm_q
            result.at[row_index, "best_global_cluster_id"] = best_candidate["cluster_id"]
            result.at[row_index, "best_global_cluster_prob_accuracy"] = best_candidate["prob_accuracy"]
            result.at[row_index, "best_global_cluster_set_accuracy"] = best_candidate["set_accuracy"]
            result.at[row_index, "best_global_cluster_meta"] = best_candidate["meta"]

    return result


def append_player_event_columns(data: pd.DataFrame, players: list[str]) -> pd.DataFrame:
    """为 joint-state 表添加玩家私有事件标记。

    输入语义：data 是 05 utility 输出表，players 是实际存在的玩家前缀。
    输出语义：返回追加 ``p1_eat_bean/p1_eat_energizer/p1_eat_ghost`` 等列的副本。
    关键约束：事件列标在事件发生后的到达行。也就是说，若第 i 行动作让玩家在
    第 i+1 行吃到资源，则第 i+1 行对应玩家事件为 True。06b/06c 的 context 边界
    必须直接使用这个事件行，不能回推到动作行；07 的策略归因可按自身规则另行读取
    前一动作，但不得反向改变事件位置的定义。
    """

    result = data.copy(deep=True)
    for player in players:
        for event_name in ("eat_bean", "eat_energizer", "eat_ghost"):
            result[f"{player}_{event_name}"] = False

    for _, trial_data in result.groupby("DayTrial", sort=False):
        labels = list(trial_data.index)
        for previous_label, current_label in zip(labels[:-1], labels[1:]):
            # 资源消失表示上一行到当前行之间有人吃到了对应资源。只有到达坐标
            # 命中消失坐标的玩家才获得该事件，另一个玩家不再共享这个硬边界。
            eaten_beans = parse_position_set(result.at[previous_label, "beans"]) - parse_position_set(
                result.at[current_label, "beans"]
            )
            eaten_energizers = parse_position_set(result.at[previous_label, "energizers"]) - parse_position_set(
                result.at[current_label, "energizers"]
            )
            if eaten_beans or eaten_energizers:
                for player in players:
                    position_column = f"{player}_pos"
                    if position_column not in result.columns:
                        continue
                    arrival_position = parse_position_or_none(result.at[current_label, position_column])
                    if arrival_position in eaten_beans:
                        result.at[current_label, f"{player}_eat_bean"] = True
                    if arrival_position in eaten_energizers:
                        result.at[current_label, f"{player}_eat_energizer"] = True

        for ghost_index, status_column, position_column in (
            (1, "ifscared1", "ghost1Pos"),
            (2, "ifscared2", "ghost2Pos"),
        ):
            if status_column not in trial_data.columns or position_column not in trial_data.columns:
                continue
            status_values = pd.to_numeric(trial_data[status_column], errors="coerce")
            eaten_labels = status_values[(status_values == 3) & (status_values.diff() < 0)].index.tolist()
            for event_label in eaten_labels:
                ghost_position = parse_position_or_none(result.at[event_label, position_column])
                if ghost_position is None:
                    continue
                for player in players:
                    position_column_player = f"{player}_pos"
                    if position_column_player not in result.columns:
                        continue
                    player_position = parse_position_or_none(result.at[event_label, position_column_player])
                    if player_position == ghost_position:
                        result.at[event_label, f"{player}_eat_ghost"] = True

    return result


def resource_counts(trial_data: pd.DataFrame, column: str) -> pd.Series:
    """统计每一行剩余资源数量。

    输入语义：trial_data 是单个 DayTrial 的玩家临时拟合表，column 是 ``beans``
    或 ``energizers``。
    输出语义：返回与 trial_data 对齐的整数数量 Series。
    关键约束：数量下降表示上一行到下一行之间吃到了对应资源。
    """

    if column not in trial_data.columns:
        return pd.Series([0] * len(trial_data), index=trial_data.index, dtype="int64")
    return trial_data[column].apply(lambda value: len(parse_position_collection(value))).astype("int64")


def true_run_ranges(flags: list[bool]) -> list[tuple[int, int]]:
    """把布尔序列中的连续 True 区间转换为半开区间。

    输入语义：flags[i] 表示第 i 行动作到下一行之间发生某事件。
    输出语义：返回 ``(start, end)``，其中 end 是最后一个 True 下标 + 1。
    关键约束：对于连续吃豆子段，返回区间正好覆盖产生吃豆动作的那些行。
    """

    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        if (not flag or index == len(flags) - 1) and start is not None:
            end = index + 1 if flag and index == len(flags) - 1 else index
            ranges.append((start, end))
            start = None
    return ranges


def bean_run_ranges_allow_short_stay(
    bean_action_flags: list[bool],
    action_values: list[Any],
    stay_length: int,
) -> list[tuple[int, int]]:
    """识别允许短暂停顿连接的连续吃普通豆动作段。

    输入语义：``bean_action_flags[i]`` 表示第 i 行动作会让玩家在第 i+1 行吃到普通豆；
    ``action_values`` 是同一 trial 内该玩家的动作方向序列；``stay_length`` 是长 stay
    阈值。
    输出语义：返回普通豆连续事件段的半开区间 ``(start, end)``，区间仍以动作行为单位。
    关键约束：本函数只处理普通豆。短 ``NaN`` 行被视为透明 gap，不会打断前后连续吃豆；
    但有效动作如果没有导致吃豆，或 ``NaN`` 连续长度达到长 stay 阈值，就会结束当前
    吃豆段。这样可以修正“到达吃豆后的短 stay 把空间连续吃豆切碎”的问题，同时不把
    真正停止或改变目标的行为错误合并。
    """

    if len(bean_action_flags) != len(action_values):
        raise ValueError("bean_action_flags 和 action_values 长度不一致，无法识别连续吃豆段。")

    ranges: list[tuple[int, int]] = []
    run_start: int | None = None
    last_bean_action_index: int | None = None
    missing_gap_length = 0

    def close_current_run() -> None:
        """关闭当前连续吃豆段，右端停在最后一个真正导致吃豆的动作之后。"""

        nonlocal run_start, last_bean_action_index, missing_gap_length
        if run_start is not None and last_bean_action_index is not None:
            ranges.append((run_start, last_bean_action_index + 1))
        run_start = None
        last_bean_action_index = None
        missing_gap_length = 0

    for index, bean_action in enumerate(bean_action_flags):
        if bean_action:
            # 一旦再次出现“动作导致吃豆”，前面尚未达到长 stay 的 NaN gap 就被视作
            # 连接两个吃豆动作的短暂停顿，保留在同一个连续吃豆段内部。
            if run_start is None:
                run_start = index
            last_bean_action_index = index
            missing_gap_length = 0
            continue

        if run_start is None:
            continue

        if action_is_missing(action_values[index]):
            missing_gap_length += 1
            if missing_gap_length >= stay_length:
                # 长 stay 是明确行为段落，不能继续连接前后的吃豆动作。关闭时不把
                # stay 行吞进吃豆段，避免真正静止被误解释成连续采食。
                close_current_run()
            continue

        # 有效动作但没有吃普通豆，说明行为目标已经不再是连续采食，立即断开。
        close_current_run()

    close_current_run()
    return ranges


def action_is_missing(value: Any) -> bool:
    """判断动作方向是否缺失。

    输入语义：value 来自 ``action_dir``，可能是方向字符串或 NaN。
    输出语义：缺失返回 True。
    关键约束：字符串方向不能直接用 pandas.isna 后和布尔数组混用。
    """

    return not isinstance(value, str)


def suppress_bean_boundaries_near_events(
    bean_start_points: set[int],
    bean_end_points: set[int],
    directional_event_boundaries: set[int],
    window: int,
    *,
    symmetric_event_boundaries: set[int] | None = None,
) -> set[int]:
    """找出方向性边界或对称强事件附近应取消的普通豆边界。

    输入语义：bean_start_points/bean_end_points 分别是连续吃普通豆过程的首个和末个
    实际事件行；directional_event_boundaries 包含 trial 和长 stay 等只替代朝向自身一侧
    吃豆边界的事件点；symmetric_event_boundaries 包含生死、energizer 和吃 ghost 等会
    替代前后两侧任意吃豆边界的行为强事件；window 是时间窗口，单位为 tile。
    输出语义：返回需要从普通豆边界集合中删除的局部边界下标。
    关键约束：方向性边界只删除“朝向事件”的一侧，即事件前的吃豆结束点和事件后的
    吃豆开始点；对称强事件删除其前后 window 内的所有吃豆开始/结束点。函数绝不删除
    事件本身；距离使用同一 trial 内的 tile 下标，而不是地图空间距离。
    """

    if window < 0:
        raise ValueError("bean event suppression window 不能小于 0。")

    symmetric_boundaries = symmetric_event_boundaries or set()

    # Trial 或长 stay 只替代事件朝向一侧的普通豆边界：事件后的吃豆开始，以及事件前
    # 的吃豆结束。这保留了另一侧真实采食过程的开始/结束语义。
    directionally_suppressed_starts = {
        bean_start
        for bean_start in bean_start_points
        if any(
            0 <= bean_start - event_boundary <= window
            for event_boundary in directional_event_boundaries
        )
    }
    directionally_suppressed_ends = {
        bean_end
        for bean_end in bean_end_points
        if any(
            0 <= event_boundary - bean_end <= window
            for event_boundary in directional_event_boundaries
        )
    }

    # 生死、本人吃 energizer 和本人吃 ghost 会独立定义行为阶段。其前后短窗口内的普通
    # 豆开始/结束事件都属于同一强行为转变附近的局部碎片，因此不再区分事件位于哪侧、
    # 也不区分普通豆边界是开始还是结束。
    symmetrically_suppressed = {
        bean_boundary
        for bean_boundary in bean_start_points | bean_end_points
        if any(
            abs(bean_boundary - event_boundary) <= window
            for event_boundary in symmetric_boundaries
        )
    }
    return (
        directionally_suppressed_starts
        | directionally_suppressed_ends
        | symmetrically_suppressed
    )


def suppress_stay_ranges_near_ghost(
    stay_ranges: list[tuple[int, int]],
    eat_ghost_indices: list[int],
    window: int,
) -> set[tuple[int, int]]:
    """找出吃 ghost 事件前后应取消切段作用的长 stay 区间。

    输入语义：stay_ranges 是长 stay 的半开行区间，eat_ghost_indices 是当前玩家私有的
    吃 ghost 事件行，window 是前后时间窗口，单位为 tile。
    输出语义：返回整段不再加入硬边界的 stay 区间集合。
    关键约束：只取消 stay 的开始/结束硬边界，不删除原始无动作行，也不删除 eat ghost
    强事件。事件位于 stay 内部时距离为 0；位于外部时按事件行到最近 stay 行计算。
    """

    if window < 0:
        raise ValueError("ghost stay suppression window 不能小于 0。")

    suppressed: set[tuple[int, int]] = set()
    for start, end in stay_ranges:
        if end <= start:
            continue
        for event_index in eat_ghost_indices:
            if start <= event_index < end:
                distance = 0
            elif event_index < start:
                distance = start - event_index
            else:
                distance = event_index - (end - 1)
            if distance <= window:
                suppressed.add((start, end))
                break
    return suppressed


def hard_boundary_points(
    trial_data: pd.DataFrame,
    player: str,
    stay_length: int,
    bean_event_suppression_window: int = 3,
    ghost_stay_suppression_window: int = 5,
) -> set[int]:
    """生成单个 trial 内不可跨越的硬边界。

    输入语义：trial_data 是 reset index 后的单 trial 临时表，player 是当前拟合玩家，
    bean_event_suppression_window 指定普通豆起止边界在强事件前后的取消窗口；
    ghost_stay_suppression_window 指定吃 ghost 事件前后取消长 stay 切段作用的窗口。
    输出语义：返回局部行号边界集合，包含 0 和 len(trial_data)。
    关键约束：硬边界表达行为事件的语义变化；短段合并时绝对不能跨过这些边界。
    普通豆起止是可抑制的弱硬边界；trial、生死、本人 energizer 和任一玩家吃 ghost
    是始终保留的强硬边界。长 stay 默认也是强边界，但若处于公共吃 ghost 事件前后
    指定窗口，则取消该 stay 的切段作用，避免 ghost 交互动画附近产生碎段。
    """

    row_count = len(trial_data)
    # Trial 边界和保留下来的长 stay 采用方向性普通豆抑制；行为强事件采用前后对称抑制。
    # 两类事件最终都是不可跨越的硬边界，区分集合只用于普通豆弱边界的去重。
    directional_boundaries: set[int] = {0, row_count}
    symmetric_boundaries: set[int] = set()

    # Pacman 生死变化会改变动作意义，死亡/复活前后的段落不能合并。
    alive_column = f"{player}_alive"
    if alive_column in trial_data.columns:
        alive_values = trial_data[alive_column].astype(bool).tolist()
        for index in range(1, row_count):
            if alive_values[index] != alive_values[index - 1]:
                symmetric_boundaries.add(index)

    # 普通豆和 energizer 分开处理。事件列本身已经标在事件发生行，context 切段
    # 只能使用这些事件行，不能把导致事件的前一动作行当作事件边界。
    #
    # 连续吃豆的识别仍需要动作序列：短 NaN/stay 可以连接前后吃豆事件。这里先用
    # action flags 判断哪些 bean 事件属于同一过程，再把动作范围转换回首个和末个
    # 实际事件行。动作范围只是内部识别工具，不会直接进入最终边界集合。
    bean_action_flags: list[bool] = []
    eat_bean_column = f"{player}_eat_bean"
    eat_energizer_column = f"{player}_eat_energizer"
    for index in range(row_count):
        if index >= row_count - 1:
            bean_action_flags.append(False)
            continue
        bean_eaten = bool(trial_data[eat_bean_column].iloc[index + 1]) if eat_bean_column in trial_data.columns else False
        bean_action_flags.append(bool(bean_eaten))
    bean_action_ranges = bean_run_ranges_allow_short_stay(
        bean_action_flags,
        trial_data["action_dir"].tolist(),
        stay_length,
    )
    bean_start_points: set[int] = set()
    bean_end_points: set[int] = set()
    for action_start, action_end in bean_action_ranges:
        # action_start 的动作导致 action_start+1 行第一次记录 eat_bean；action_end-1
        # 的动作导致 action_end 行最后一次记录 eat_bean。因此实际事件点分别是
        # action_start+1 和 action_end，不能使用动作范围自身的左右边界。
        first_event = action_start + 1
        last_event = action_end
        if 0 <= first_event < row_count:
            bean_start_points.add(first_event)
        if 0 <= last_event < row_count:
            bean_end_points.add(last_event)

    # Energizer 是单点强事件。事件列已经标在吃到 energizer 后的到达行，因此直接
    # 加入 True 所在行；不再回推前一动作，也不人为构造一行宽的 [start, end) 区间。
    if eat_energizer_column in trial_data.columns:
        energizer_indices = trial_data.index[trial_data[eat_energizer_column].astype(bool)].tolist()
        symmetric_boundaries.update(int(index) for index in energizer_indices)

    # 任一玩家吃到 ghost 都会立即移除共享环境中的追逐目标，使两名玩家的 Approach
    # utility 在同一事件行发生结构变化。因此这里使用 P1/P2 私有事件列的并集，为两名
    # 玩家生成完全相同、且不能被短段合并删除的公共硬边界。原私有列保持不变，后续仍
    # 可用于判断究竟由谁吃到 ghost。
    eaten_indices: list[int] = []
    for event_player in PLAYER_PREFIXES:
        eat_ghost_column = f"{event_player}_eat_ghost"
        if eat_ghost_column not in trial_data.columns:
            continue
        event_indices = trial_data.index[trial_data[eat_ghost_column].astype(bool)].tolist()
        eaten_indices.extend(int(index) for index in event_indices)
    eaten_indices = sorted(set(eaten_indices))
    symmetric_boundaries.update(eaten_indices)

    # 长 stay 默认作为强边界；但 ghost 被任一玩家吃掉前后的停顿常来自交互/动画，
    # 而不是当前玩家主动 stay。若整段 stay 距公共 eat_ghost 行不超过窗口，则同时
    # 取消其起止边界，公共吃鬼硬边界本身仍始终保留。
    missing_flags = [action_is_missing(value) for value in trial_data["action_dir"]]
    long_stay_ranges = [
        (start, end)
        for start, end in true_run_ranges(missing_flags)
        if end - start >= stay_length
    ]
    suppressed_stay_ranges = suppress_stay_ranges_near_ghost(
        long_stay_ranges,
        eaten_indices,
        ghost_stay_suppression_window,
    )
    for start, end in long_stay_ranges:
        if (start, end) not in suppressed_stay_ranges:
            directional_boundaries.add(start)
            directional_boundaries.add(end)

    suppressed_bean_boundaries = suppress_bean_boundaries_near_events(
        bean_start_points,
        bean_end_points,
        directional_boundaries,
        bean_event_suppression_window,
        symmetric_event_boundaries=symmetric_boundaries,
    )
    bean_event_points = bean_start_points | bean_end_points
    retained_bean_boundaries = bean_event_points - suppressed_bean_boundaries
    strong_boundaries = directional_boundaries | symmetric_boundaries
    boundaries = strong_boundaries | retained_bean_boundaries
    return {boundary for boundary in boundaries if 0 <= boundary <= row_count}


def soft_turnaround_points(trial_data: pd.DataFrame) -> set[int]:
    """生成掉头位置，供诊断分析使用。

    输入语义：trial_data 是单 trial 临时表，``action_dir`` 已经把非法方向置为 NaN。
    输出语义：返回局部行号集合。
    关键约束：该函数不再参与正式 context 划分，只保留用于行为诊断和A/B比较；
    NaN 过渡仍使用前一个有效方向作为参照，便于稳定识别动作反转位置。
    """

    boundaries: set[int] = set()
    previous_direction: str | None = None
    for index, direction in enumerate(trial_data["action_dir"].tolist()):
        if not isinstance(direction, str):
            continue
        if previous_direction is not None and OPPOSITE_DIRECTIONS.get(previous_direction) == direction:
            boundaries.add(index)
        previous_direction = direction
    return boundaries


def soft_teammate_event_points(trial_data: pd.DataFrame, player: str) -> set[int]:
    """生成队友吃 Energizer 对应的公共环境软边界。

    输入语义：trial_data 是单 trial 临时表，player 是当前正在划分 context 的玩家。
    输出语义：返回另一名玩家吃 Energizer 的真实事件行集合。
    关键约束：队友吃 Energizer 改变公共 ghost 状态，但仍沿用可合并软边界规则；
    任一玩家吃 ghost 已在 ``hard_boundary_points`` 中统一生成公共硬边界，不得再次
    作为软边界。普通豆事件不共享，ghost 按计时器自然恢复也不在这里生成边界。
    """

    boundaries: set[int] = set()
    for teammate in ("p1", "p2"):
        if teammate == player:
            continue
        column = f"{teammate}_eat_energizer"
        if column not in trial_data.columns:
            continue
        event_indices = trial_data.index[trial_data[column].astype(bool)].tolist()
        boundaries.update(int(index) for index in event_indices)
    return boundaries


def merge_soft_short_contexts(
    contexts: list[tuple[int, int]],
    trial_data: pd.DataFrame,
    min_length: int,
) -> list[tuple[int, int]]:
    """在一个硬边界区间内部合并过短软段。

    输入语义：contexts 都位于同一个硬边界区间内，彼此相邻且覆盖完整区间。
    输出语义：返回合并后的段落。
    关键约束：函数只处理同一硬区间内部的软边界，因此不会跨行为事件硬边界。
    如果硬区间本身很短，只有一个 context，则保持原样。
    """

    merged = list(contexts)
    while len(merged) > 1:
        short_index: int | None = None
        for index, (start, end) in enumerate(merged):
            if end - start < min_length:
                short_index = index
                break
        if short_index is None:
            break

        if short_index == 0:
            left, right = merged[0], merged[1]
            merged[0:2] = [(left[0], right[1])]
            continue
        if short_index == len(merged) - 1:
            left, right = merged[-2], merged[-1]
            merged[-2:] = [(left[0], right[1])]
            continue

        previous_length = merged[short_index - 1][1] - merged[short_index - 1][0]
        next_length = merged[short_index + 1][1] - merged[short_index + 1][0]
        if previous_length <= next_length:
            left, right = merged[short_index - 1], merged[short_index]
            merged[short_index - 1 : short_index + 1] = [(left[0], right[1])]
        else:
            left, right = merged[short_index], merged[short_index + 1]
            merged[short_index : short_index + 2] = [(left[0], right[1])]

    return merged


def build_event_context_segments(
    prepared_data: pd.DataFrame,
    player: str,
    config: DynamicStrategyFittingConfig | None = None,
) -> tuple[list[tuple[int, int]], list[bool]]:
    """按玩家事件、公共吃鬼硬边界和队友 Energizer 软边界构造 context。

    输入语义：prepared_data 是 ``prepare_fitting_dataframe`` 生成的玩家临时表，
    player 是当前拟合玩家。
    输出语义：返回全局 row_id 半开区间列表，以及每个区间是否为 stay/all-NaN 段。
    关键约束：短段合并只发生在同一硬边界区间内部，绝不跨 trial、吃豆段、生死、
    本人 energizer、任一玩家吃 ghost 或长 stay 硬边界。软边界只来自队友的
    Energizer 事件，只有在两侧能形成足够长段落时才保留。掉头、普通转向、玩家到
    普通豆的距离和 ghost 计时恢复都不再生成 context 边界。
    """

    config = DynamicStrategyFittingConfig() if config is None else config
    all_contexts: list[tuple[int, int]] = []
    all_is_nan: list[bool] = []

    for trial_index, trial_name in enumerate(np.unique(prepared_data.DayTrial.values)):
        trial_data = prepared_data[prepared_data.DayTrial == trial_name].copy().reset_index(drop=True)
        print(f"| ({trial_index}) {trial_name} | Event context data shape {trial_data.shape}")

        level_offset = int(trial_data["row_id"].iloc[0])
        hard_boundaries = sorted(
            hard_boundary_points(
                trial_data,
                player,
                config.stay_length,
                bean_event_suppression_window=config.bean_event_suppression_window,
                ghost_stay_suppression_window=config.ghost_stay_suppression_window,
            )
        )
        # 玩家自己的掉头只作为段内动作进入后续 likelihood，不再用同一动作变量预先
        # 切段。队友吃 Energizer 仍作为候选软边界并经过 min_length 合并；任一玩家
        # 吃 ghost 已进入 hard_boundaries，不能在这里被短段合并撤销。
        soft_boundaries = soft_teammate_event_points(trial_data, player)

        trial_contexts: list[tuple[int, int]] = []
        for hard_start, hard_end in zip(hard_boundaries[:-1], hard_boundaries[1:]):
            if hard_end <= hard_start:
                continue
            local_soft = sorted(boundary for boundary in soft_boundaries if hard_start < boundary < hard_end)
            raw_contexts = list(zip([hard_start] + local_soft, local_soft + [hard_end]))
            trial_contexts.extend(
                merge_soft_short_contexts(raw_contexts, trial_data, min_length=config.stay_length)
            )

        for start, end in trial_contexts:
            all_contexts.append((start + level_offset, end + level_offset))
            all_is_nan.append(all_directions_nan(trial_data, (start, end)))

    return all_contexts, all_is_nan


def fit_player_strategy_event_context_dataframe(
    raw_data: pd.DataFrame,
    player: str,
    config: DynamicStrategyFittingConfig,
) -> pd.DataFrame:
    """使用事件 context 为单个玩家拟合动态策略权重。

    输入语义：raw_data 是 05 utility joint-state 表，player 指定当前玩家。
    输出语义：返回 ``<player>_weight``、``<player>_trial_context`` 等 06 兼容字段。
    关键约束：context 划分后会先选择每段 best cluster global，并覆盖临时表中的
    ``global_Q/global_Q_norm``；段落拟合、权重归一化和 07 所需输出字段保持原接口。
    best global 直接读取 05 已生成的候选 utility，06b 不再为 context 读取地图距离。
    """

    if config.random_seed is not None:
        np.random.seed(config.random_seed)

    print(f"=== Event Context Dynamic Strategy Fitting: {player} ====")
    fit_data = prepare_fitting_dataframe(raw_data, player, config)
    suffix = "_Q_norm"
    invalid_direction_indices = np.where(fit_data["available_dir"] == False)[0]
    fit_data.loc[fit_data.index[invalid_direction_indices], "action_dir"] = [np.nan] * len(invalid_direction_indices)

    contexts, is_nan = build_event_context_segments(fit_data, player, config)
    fit_data = apply_best_global_candidates(fit_data, contexts, player)
    result_list, _, is_correct, predicted_direction, is_vague = fit_all_segments(
        fit_data,
        contexts,
        is_nan,
        config,
        suffix=suffix,
    )

    output = initialize_player_result(raw_data.index, player)
    trial_weight: list[Any] = []
    trial_context: list[tuple[int, int]] = []
    trial_normalized_weight: list[Any] = []
    trial_is_stay: list[bool] = []
    for result_index, result in enumerate(result_list):
        weight = np.asarray(result[: len(config.agents)], dtype=float)
        start = result[-2]
        end = result[-1]
        for _ in range(start, end):
            trial_context.append((start, end))
            trial_weight.append(weight.tolist())
            trial_is_stay.append(is_nan[result_index])
            if is_nan[result_index] is False and np.sum(weight) != 0 and np.max(weight) != np.min(weight):
                normalized_weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))
                trial_normalized_weight.append(normalized_weight.tolist())
            else:
                trial_normalized_weight.append(copy.deepcopy(weight.tolist()))

    if len(trial_weight) != fit_data.shape[0]:
        raise RuntimeError(
            f"{player} event-context 拟合结果长度不一致："
            f"weights={len(trial_weight)}, rows={fit_data.shape[0]}"
        )

    output[f"{player}_weight"] = trial_weight
    output[f"{player}_normalized_weight"] = trial_normalized_weight
    output[f"{player}_prediction_correct"] = is_correct
    output[f"{player}_predict_dir"] = predicted_direction
    output[f"{player}_trial_context"] = trial_context
    output[f"{player}_is_stay"] = trial_is_stay
    output[f"{player}_is_vague"] = is_vague
    # 06b 在 context 级选出的 best global 是后续 07 和视频解释真正使用的
    # global utility，因此把覆盖后的正式 global_Q/global_Q_norm 写回玩家前缀字段。
    for source_column, target_column in (
        ("global_Q", f"{player}_global_Q"),
        ("global_Q_norm", f"{player}_global_Q_norm"),
        ("best_global_cluster_id", f"{player}_best_global_cluster_id"),
        ("best_global_cluster_prob_accuracy", f"{player}_best_global_cluster_prob_accuracy"),
        ("best_global_cluster_set_accuracy", f"{player}_best_global_cluster_set_accuracy"),
        ("best_global_cluster_meta", f"{player}_best_global_cluster_meta"),
    ):
        if source_column in fit_data.columns:
            output[target_column] = fit_data[source_column].to_numpy()
    print(np.sum(is_vague) / len(fit_data))
    print(f"Finished event-context fitting {player}.")
    return output


def fit_dynamic_strategy_event_context_dataframe(
    raw_data: pd.DataFrame,
    config: DynamicStrategyFittingConfig | None = None,
) -> pd.DataFrame:
    """对单个 joint-state 表执行事件 context 版动态策略拟合。

    输入语义：raw_data 是 05 utility 输出表。
    输出语义：返回保留原字段并追加 p1/p2 权重字段的 DataFrame。
    关键约束：输出字段完全兼容 07；新增逻辑只影响 ``*_trial_context`` 和拟合权重。
    """

    config = DynamicStrategyFittingConfig() if config is None else config
    base = raw_data.reset_index(drop=True).copy(deep=True)
    players = discover_player_prefixes(base, config)
    result = append_player_event_columns(base, players)
    for player in players:
        player_output = fit_player_strategy_event_context_dataframe(result, player, config)
        for column in player_output.columns:
            result[column] = player_output[column].to_numpy()
    return result


def process_dynamic_strategy_event_context_file(
    input_path: str | Path,
    output_path: str | Path,
    config: DynamicStrategyFittingConfig | None = None,
    file_index: int = 0,
) -> dict[str, Any]:
    """处理单个 05 utility 文件并保存事件 context 版 06 输出。

    输入语义：input_path 是 05 utility pickle，output_path 是目标 06b pickle。
    输出语义：写出事件 context 版 WeightData，并返回摘要。
    关键约束：若设置 random_seed，会按文件序号派生 seed，保持文件级并行可复现。
    """

    config = DynamicStrategyFittingConfig() if config is None else config
    input_file = Path(input_path)
    output_file = Path(output_path)
    with input_file.open("rb") as file:
        raw_data = pickle.load(file)

    if config.random_seed is not None:
        file_config = DynamicStrategyFittingConfig(
            agents=config.agents,
            stay_length=config.stay_length,
            ga_population_size=config.ga_population_size,
            ga_iterations=config.ga_iterations,
            ga_mutation_probability=config.ga_mutation_probability,
            ga_precision=config.ga_precision,
            weight_penalty=config.weight_penalty,
            vague_accuracy_threshold=config.vague_accuracy_threshold,
            random_seed=config.random_seed + file_index,
            segment_workers=config.segment_workers,
            use_segment_seed=config.use_segment_seed,
            bean_event_suppression_window=config.bean_event_suppression_window,
            ghost_stay_suppression_window=config.ghost_stay_suppression_window,
            min_effective_action_count=config.min_effective_action_count,
            min_effective_action_ratio=config.min_effective_action_ratio,
        )
    else:
        file_config = config

    result = fit_dynamic_strategy_event_context_dataframe(raw_data, file_config)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as file:
        pickle.dump(result, file)
    print("Finished saving event-context data.")
    return {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "rows": int(result.shape[0]),
        "columns": int(result.shape[1]),
        "seed": file_config.random_seed,
    }


def process_dynamic_strategy_event_context_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    config: DynamicStrategyFittingConfig | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """批量处理嵌套目录中的 05 utility 文件。

    输入语义：input_dir 是 ``comp/*.pkl``、``coop/*.pkl`` 结构，output_dir 是 06b 输出根目录。
    输出语义：按相同相对路径保存并返回摘要列表。
    关键约束：文件级并行和段落级并行不要同时开太大；单文件调试时建议只开段落级并行。
    """

    config = DynamicStrategyFittingConfig() if config is None else config
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    input_files = sorted(path for path in input_dir.glob("*/*.pkl") if path.is_file())
    if not input_files:
        raise FileNotFoundError(f"输入目录中没有嵌套 pickle 文件：{input_dir}")

    tasks = [
        (
            input_file,
            output_dir / input_file.relative_to(input_dir),
            config,
            file_index,
        )
        for file_index, input_file in enumerate(input_files)
    ]
    if workers <= 1:
        return [_process_event_context_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        return list(executor.map(_process_event_context_task, tasks))


def _process_event_context_task(
    task: tuple[Path, Path, DynamicStrategyFittingConfig, int],
) -> dict[str, Any]:
    """执行目录级并行中的单文件事件 context 拟合任务。"""

    input_path, output_path, config, file_index = task
    return process_dynamic_strategy_event_context_file(input_path, output_path, config, file_index=file_index)
