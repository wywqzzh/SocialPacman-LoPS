"""事件硬边界版 Social Pacman 动态策略拟合。

本模块是 06 动态策略拟合的实验新版。它复用旧 06 的权重拟合器和输出字段，
只替换 context 划分方式：先按玩家行为事件生成不可跨越的硬边界，再只把掉头
作为软方向边界，最后在不跨硬边界的前提下合并软边界造成的过短段。
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
from LoPS.hierarchical_utility import MapData


OPPOSITE_DIRECTIONS: dict[str, str] = {
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up",
}
DIRECTION_NAMES: tuple[str, ...] = ("left", "right", "up", "down")
DIRECTION_TO_INDEX: dict[str, int] = {direction: index for index, direction in enumerate(DIRECTION_NAMES)}


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
    第 i+1 行吃到资源，则第 i+1 行对应玩家事件为 True；06 会回推到第 i 行动作
    构造硬边界，07 也继续沿用旧规则中的 ``event_index - 1`` 时间对齐。
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


def action_arrival_position(
    trial_data: pd.DataFrame,
    player: str,
    action_index: int,
) -> tuple[int, int] | None:
    """读取某个动作完成后的玩家到达位置。

    输入语义：action_index 表示第 action_index 行到下一行之间的动作；资源事件列也标在
    下一行，因此到达位置读取 ``action_index + 1`` 行的 ``<player>_pos``。
    输出语义：合法坐标返回 ``(x, y)``，越界或缺失时返回 None。
    关键约束：本函数只用于判断连续采食事件的空间相邻性，不改变任何原始坐标。
    """

    arrival_index = action_index + 1
    position_column = f"{player}_pos"
    if arrival_index >= len(trial_data) or position_column not in trial_data.columns:
        return None
    return parse_position_or_none(trial_data[position_column].iloc[arrival_index])


def nearest_non_missing_action_index(action_values: list[Any], start_index: int, step: int) -> int | None:
    """从指定位置开始寻找最近的非缺失动作行。

    输入语义：action_values 是单 trial 动作序列，start_index 是搜索起点，step 为 -1
    表示向前找、1 表示向后找。
    输出语义：返回最近的有效动作下标；若一路都是 NaN/stay 则返回 None。
    关键约束：该函数只跳过缺失动作。长 stay 仍会由 hard_boundary_points 中独立的
    长 stay 硬边界阻断，不会因为这里跳过 NaN 而被实际跨越。
    """

    index = start_index
    while 0 <= index < len(action_values):
        if not action_is_missing(action_values[index]):
            return index
        index += step
    return None


def positions_are_adjacent(
    first_position: tuple[int, int] | None,
    second_position: tuple[int, int] | None,
    map_data: MapData | None,
) -> bool:
    """判断两个资源到达位置在地图上是否相邻。

    输入语义：first_position/second_position 是两次资源事件的到达坐标，map_data 提供
    最短路径距离。
    输出语义：两个位置之间地图最短距离为 1 时返回 True。
    关键约束：使用地图距离表而不是坐标差，保证 tunnel 和鬼屋修正后的连通性一致。
    """

    if first_position is None or second_position is None or map_data is None:
        return False
    distance_lookup = map_data.distance_by_position.get(first_position, {})
    return distance_lookup.get(second_position) == 1


def suppress_bean_boundaries_around_energizer(
    trial_data: pd.DataFrame,
    player: str,
    bean_ranges: list[tuple[int, int]],
    energizer_action_flags: list[bool],
    map_data: MapData | None,
) -> set[int]:
    """识别应从硬边界中删除的“普通豆-energizer”连接边界。

    输入语义：bean_ranges 是普通豆连续动作段，energizer_action_flags 标记每个动作是否
    吃到 energizer，map_data 用于判断资源位置是否相邻。
    输出语义：返回需要从硬边界集合中删除的局部边界下标。
    关键约束：energizer 常嵌在连续采食轨迹中。若 energizer 前一个有效动作是普通豆段
    终止，且普通豆到达位置与 energizer 到达位置相邻，则删除普通豆终止边界以及
    energizer 起点边界；后侧的普通豆开始边界同理。这样只移除“资源连续性”造成的
    人工切点，死亡、吃鬼、长 stay 等其它硬边界不会被本函数删除。
    """

    if map_data is None:
        return set()

    action_values = trial_data["action_dir"].tolist()
    # 记录普通豆连续段的第一个/最后一个真实吃豆动作。由于 bean_ranges 已经允许
    # 短暂 NaN 连接，start/end 本身正好表达“普通豆段开始/终止”的硬边界。
    bean_start_boundary_by_action = {start: start for start, _ in bean_ranges}
    bean_end_boundary_by_action = {end - 1: end for _, end in bean_ranges if end > 0}
    suppressed_boundaries: set[int] = set()

    for energizer_action_index, ate_energizer in enumerate(energizer_action_flags):
        if not ate_energizer:
            continue
        energizer_position = action_arrival_position(trial_data, player, energizer_action_index)

        previous_action_index = nearest_non_missing_action_index(
            action_values,
            energizer_action_index - 1,
            step=-1,
        )
        if previous_action_index in bean_end_boundary_by_action:
            previous_bean_position = action_arrival_position(trial_data, player, previous_action_index)
            if positions_are_adjacent(previous_bean_position, energizer_position, map_data):
                # 删除普通豆段终止边界；同时删除 energizer 起点边界，使“普通豆 -> energizer”
                # 在空间连续时保持为同一个资源采食段。
                suppressed_boundaries.add(bean_end_boundary_by_action[previous_action_index])
                suppressed_boundaries.add(energizer_action_index)

        next_action_index = nearest_non_missing_action_index(
            action_values,
            energizer_action_index + 1,
            step=1,
        )
        if next_action_index in bean_start_boundary_by_action:
            next_bean_position = action_arrival_position(trial_data, player, next_action_index)
            if positions_are_adjacent(energizer_position, next_bean_position, map_data):
                # 删除 energizer 终点边界和后侧普通豆开始边界，使“energizer -> 普通豆”
                # 在空间连续时不会被切成长度很短的独立段。
                suppressed_boundaries.add(energizer_action_index + 1)
                suppressed_boundaries.add(bean_start_boundary_by_action[next_action_index])

    return suppressed_boundaries


def hard_boundary_points(
    trial_data: pd.DataFrame,
    player: str,
    stay_length: int,
    map_data: MapData | None = None,
) -> set[int]:
    """生成单个 trial 内不可跨越的硬边界。

    输入语义：trial_data 是 reset index 后的单 trial 临时表，player 是当前拟合玩家，
    map_data 用于判断 energizer 前后普通豆事件是否空间连续。
    输出语义：返回局部行号边界集合，包含 0 和 len(trial_data)。
    关键约束：硬边界表达行为事件的语义变化；短段合并时绝对不能跨过这些边界。
    但当 energizer 嵌在连续普通豆采食轨迹中时，普通豆段在 energizer 前后的起止边界
    会被抑制，避免把一条连续采食路径切成多个长度为 1 的小段。
    """

    row_count = len(trial_data)
    boundaries: set[int] = {0, row_count}

    # Pacman 生死变化会改变动作意义，死亡/复活前后的段落不能合并。
    alive_column = f"{player}_alive"
    if alive_column in trial_data.columns:
        alive_values = trial_data[alive_column].astype(bool).tolist()
        for index in range(1, row_count):
            if alive_values[index] != alive_values[index - 1]:
                boundaries.add(index)

    # 普通豆和 energizer 分开处理。普通豆代表连续采食轨迹，允许中间夹短暂
    # NaN/stay；energizer 是强策略事件，仍然保持单独硬边界。
    bean_action_flags: list[bool] = []
    energizer_eat_flags: list[bool] = []
    eat_bean_column = f"{player}_eat_bean"
    eat_energizer_column = f"{player}_eat_energizer"
    for index in range(row_count):
        if index >= row_count - 1:
            bean_action_flags.append(False)
            energizer_eat_flags.append(False)
            continue
        bean_eaten = bool(trial_data[eat_bean_column].iloc[index + 1]) if eat_bean_column in trial_data.columns else False
        energizer_eaten = (
            bool(trial_data[eat_energizer_column].iloc[index + 1])
            if eat_energizer_column in trial_data.columns
            else False
        )
        bean_action_flags.append(bool(bean_eaten))
        energizer_eat_flags.append(bool(energizer_eaten))
    bean_ranges = bean_run_ranges_allow_short_stay(
        bean_action_flags,
        trial_data["action_dir"].tolist(),
        stay_length,
    )
    suppressed_resource_boundaries = suppress_bean_boundaries_around_energizer(
        trial_data,
        player,
        bean_ranges,
        energizer_eat_flags,
        map_data,
    )
    for start, end in bean_ranges:
        boundaries.add(start)
        boundaries.add(end)
    # 吃 energizer 是资源段的一种，但它也作为单独硬事件保留一行动作边界，
    # 避免单帧 energizer 事件被较长普通吃豆段完全吞掉。
    for start, end in true_run_ranges(energizer_eat_flags):
        boundaries.add(start)
        boundaries.add(end)
    boundaries.difference_update(suppressed_resource_boundaries)

    # 吃 ghost 也只使用当前玩家自己的事件列。事件列标在 ghost 进入 dead 的行，
    # 这里沿用上一版做法，只把状态转变行作为硬边界。
    eat_ghost_column = f"{player}_eat_ghost"
    if eat_ghost_column in trial_data.columns:
        eaten_indices = trial_data.index[trial_data[eat_ghost_column].astype(bool)].tolist()
        for index in eaten_indices:
            boundaries.add(int(index))

    # 长 stay 段是明确的静止事件，作为硬边界独立出来。短暂 NaN 不在这里硬切，
    # 后续会在同一硬区间内按软规则处理。
    missing_flags = [action_is_missing(value) for value in trial_data["action_dir"]]
    for start, end in true_run_ranges(missing_flags):
        if end - start >= stay_length:
            boundaries.add(start)
            boundaries.add(end)

    return {boundary for boundary in boundaries if 0 <= boundary <= row_count}


def soft_turnaround_points(trial_data: pd.DataFrame) -> set[int]:
    """生成只由掉头动作产生的软边界。

    输入语义：trial_data 是单 trial 临时表，``action_dir`` 已经把非法方向置为 NaN。
    输出语义：返回局部行号边界集合。
    关键约束：普通转向不切段；只在 left/right 或 up/down 直接反向时切段。
    NaN 过渡会使用前一个有效方向作为参照，避免短暂停顿掩盖掉头。
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


def nearest_ordinary_bean_distance(
    position: tuple[int, int] | None,
    bean_positions: set[tuple[int, int]],
    map_data: MapData | None,
) -> float:
    """计算当前位置到最近普通豆子的地图最短距离。

    输入语义：position 是当前玩家坐标，bean_positions 是当前行剩余普通豆集合，
    map_data 是地图常量中的最短路径距离表。
    输出语义：返回最近普通豆子的最短路径距离；缺少坐标、没有普通豆或无法查询距离时
    返回 ``np.inf``。
    关键约束：这里只看 ``beans``，不看 energizer。energizer 已经作为独立强事件参与
    context 划分，若再混入 local 范围边界，会让 local 与 energizer 的语义重叠。
    """

    if position is None or not bean_positions or map_data is None:
        return float("inf")
    distance_lookup = map_data.distance_by_position.get(position)
    if distance_lookup is None:
        return float("inf")
    distances = [distance_lookup.get(bean_position, float("inf")) for bean_position in bean_positions]
    if not distances:
        return float("inf")
    return float(min(distances))


def soft_local_bean_range_points(
    trial_data: pd.DataFrame,
    player: str,
    map_data: MapData | None,
    distance_threshold: int,
) -> set[int]:
    """生成进入或离开普通豆 local 范围的软边界。

    输入语义：trial_data 是单 trial 的玩家临时拟合表，player 指定当前玩家，
    map_data 提供地图最短路径距离，distance_threshold 通常等于旧 utility 的
    ``local_depth=10``。
    输出语义：返回局部行号边界集合；第 i 行边界表示第 i 行相对第 i-1 行发生
    ``最近普通豆距离 <= threshold`` 状态切换。
    关键约束：这是软边界，不是硬事件。后续仍会在同一硬边界区间内合并过短段，
    避免距离在阈值附近抖动时制造过多 1-2 行碎段。每个玩家使用自己的位置列，因此
    双人任务中 p1/p2 的 local 范围切点彼此独立。
    """

    if map_data is None or distance_threshold <= 0:
        return set()
    position_column = f"{player}_pos"
    if position_column not in trial_data.columns or "beans" not in trial_data.columns:
        return set()

    alive_column = f"{player}_alive"
    in_local_range: list[bool] = []
    for _, row in trial_data.iterrows():
        # 玩家死亡或缺失时不把当前位置解释为 local 采食范围，避免死亡停留点生成
        # 没有行为意义的软切点。
        if alive_column in trial_data.columns and not bool(row[alive_column]):
            in_local_range.append(False)
            continue
        position = parse_position_or_none(row[position_column])
        bean_positions = parse_position_set(row["beans"])
        distance = nearest_ordinary_bean_distance(position, bean_positions, map_data)
        in_local_range.append(distance <= distance_threshold)

    boundaries: set[int] = set()
    for index in range(1, len(in_local_range)):
        if in_local_range[index] != in_local_range[index - 1]:
            boundaries.add(index)
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
    map_data: MapData | None = None,
) -> tuple[list[tuple[int, int]], list[bool]]:
    """按事件硬边界和掉头软边界构造全局 context 段落。

    输入语义：prepared_data 是 ``prepare_fitting_dataframe`` 生成的玩家临时表，
    player 是当前拟合玩家，map_data 可选提供地图距离表以计算 local 范围软边界。
    输出语义：返回全局 row_id 半开区间列表，以及每个区间是否为 stay/all-NaN 段。
    关键约束：短段合并只发生在同一硬边界区间内部，绝不跨 trial、吃豆段、生死、
    吃 energizer、吃 ghost 或长 stay 边界。进入/离开普通豆 local 范围只作为软边界，
    允许在段落太短时被合并。
    """

    config = DynamicStrategyFittingConfig() if config is None else config
    all_contexts: list[tuple[int, int]] = []
    all_is_nan: list[bool] = []

    for trial_index, trial_name in enumerate(np.unique(prepared_data.DayTrial.values)):
        trial_data = prepared_data[prepared_data.DayTrial == trial_name].copy().reset_index(drop=True)
        print(f"| ({trial_index}) {trial_name} | Event context data shape {trial_data.shape}")

        level_offset = int(trial_data["row_id"].iloc[0])
        hard_boundaries = sorted(hard_boundary_points(trial_data, player, config.stay_length, map_data=map_data))
        soft_boundaries = soft_turnaround_points(trial_data)
        soft_boundaries.update(
            soft_local_bean_range_points(
                trial_data,
                player,
                map_data,
                config.local_bean_distance_threshold,
            )
        )

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
    map_data: MapData | None = None,
) -> pd.DataFrame:
    """使用事件 context 为单个玩家拟合动态策略权重。

    输入语义：raw_data 是 05 utility joint-state 表，player 指定当前玩家。
    输出语义：返回 ``<player>_weight``、``<player>_trial_context`` 等 06 兼容字段。
    关键约束：context 划分后会先选择每段 best cluster global，并覆盖临时表中的
    ``global_Q/global_Q_norm``；段落拟合、权重归一化和 07 所需输出字段保持原接口。
    map_data 只用于 context 切分，不参与 best global 选择；best global 直接读取
    05 已生成的候选 utility。
    """

    if config.random_seed is not None:
        np.random.seed(config.random_seed)

    print(f"=== Event Context Dynamic Strategy Fitting: {player} ====")
    fit_data = prepare_fitting_dataframe(raw_data, player, config)
    suffix = "_Q_norm"
    invalid_direction_indices = np.where(fit_data["available_dir"] == False)[0]
    fit_data.loc[fit_data.index[invalid_direction_indices], "action_dir"] = [np.nan] * len(invalid_direction_indices)

    contexts, is_nan = build_event_context_segments(fit_data, player, config, map_data=map_data)
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
    map_data: MapData | None = None,
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
        player_output = fit_player_strategy_event_context_dataframe(result, player, config, map_data=map_data)
        for column in player_output.columns:
            result[column] = player_output[column].to_numpy()
    return result


def process_dynamic_strategy_event_context_file(
    input_path: str | Path,
    output_path: str | Path,
    config: DynamicStrategyFittingConfig | None = None,
    file_index: int = 0,
    map_data: MapData | None = None,
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
            local_bean_distance_threshold=config.local_bean_distance_threshold,
            min_effective_action_count=config.min_effective_action_count,
            min_effective_action_ratio=config.min_effective_action_ratio,
        )
    else:
        file_config = config

    result = fit_dynamic_strategy_event_context_dataframe(raw_data, file_config, map_data=map_data)
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
    map_data: MapData | None = None,
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
            map_data,
        )
        for file_index, input_file in enumerate(input_files)
    ]
    if workers <= 1:
        return [_process_event_context_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        return list(executor.map(_process_event_context_task, tasks))


def _process_event_context_task(
    task: tuple[Path, Path, DynamicStrategyFittingConfig, int, MapData | None],
) -> dict[str, Any]:
    """执行目录级并行中的单文件事件 context 拟合任务。"""

    input_path, output_path, config, file_index, map_data = task
    return process_dynamic_strategy_event_context_file(input_path, output_path, config, file_index=file_index, map_data=map_data)
