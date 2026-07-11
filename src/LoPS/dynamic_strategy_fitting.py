"""Social Pacman 动态策略权重拟合。

本模块实现 05 utility 数据到 06 WeightData 的动态策略拟合流程。输入数据
保持 joint-state 结构，本阶段分别为每个玩家构造临时单人视角、划分 context、
拟合策略权重，再把玩家前缀结果写回同一份 joint-state 表。
"""

from __future__ import annotations

import copy
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DIRECTION_NAMES: tuple[str, ...] = ("left", "right", "up", "down")
DEFAULT_AGENTS: tuple[str, ...] = (
    "global",
    "local",
    "evade_blinky",
    "evade_clyde",
    "approach",
    "energizer",
    "no_energizer",
)
PLAYER_PREFIXES: tuple[str, ...] = ("p1", "p2")
PLAYER_RESULT_COLUMNS: tuple[str, ...] = (
    "weight",
    "normalized_weight",
    "prediction_correct",
    "predict_dir",
    "trial_context",
    "is_stay",
    "is_vague",
)


@dataclass(frozen=True)
class DynamicStrategyFittingConfig:
    """保存动态策略拟合所需的可调参数。

    输入语义：调用方可以覆盖 agent 列表、段落规则、GA 参数和随机种子。
    输出语义：配置对象会被传入 DataFrame、文件和目录级处理函数。
    关键约束：正式模块不写死任何数据路径；seed 为 None 时不主动重置随机状态。
    """

    agents: tuple[str, ...] = DEFAULT_AGENTS
    stay_length: int = 4
    ga_population_size: int = 100
    ga_iterations: int = 500
    ga_mutation_probability: float = 0.01
    ga_precision: float = 1e-3
    weight_penalty: float = 0.1
    vague_accuracy_threshold: float = 0.51
    random_seed: int | None = None
    segment_workers: int = 1
    use_segment_seed: bool = False
    local_bean_distance_threshold: int = 10
    min_effective_action_count: int = 4
    min_effective_action_ratio: float = 0.5


def choose_max_direction(probability: Any) -> int:
    """从四方向分数中选择最大值方向。

    输入语义：probability 是长度为 4 的方向分数或 one-hot 向量。
    输出语义：返回被选中的方向索引。
    关键约束：多个方向并列最大时使用 ``np.random.choice``，保留旧拟合的随机并列选择语义。
    """

    # 旧脚本中 copy_estimated 的负数修正没有参与返回值；这里保留选择逻辑本身。
    return int(np.random.choice([index for index, value in enumerate(probability) if value == max(probability)]))


def one_hot_direction(value: str) -> list[int]:
    """把方向字符串转换为长度为 4 的 one-hot 编码。

    输入语义：value 必须是 left/right/up/down 之一。
    输出语义：返回与 DIRECTION_NAMES 顺序对应的 one-hot 列表。
    关键约束：非法方向直接抛错，保持输入数据必须已清洗的假设。
    """

    if value not in DIRECTION_NAMES:
        raise ValueError(f"未知方向：{value}")
    if not isinstance(value, str):
        raise TypeError(f"未知方向类型：{type(value)}")
    result = [0, 0, 0, 0]
    result[DIRECTION_NAMES.index(value)] = 1
    return result


def prepare_fitting_dataframe(
    raw_data: pd.DataFrame,
    player: str,
    config: DynamicStrategyFittingConfig | None = None,
) -> pd.DataFrame:
    """构造单个玩家视角的动态拟合临时表。

    输入语义：raw_data 是 05 输出的 joint-state 表，player 是 ``p1`` 或 ``p2``。
    输出语义：返回包含旧拟合核心字段 ``action_dir/available_dir/*_Q_norm`` 的临时表。
    关键约束：不解析坐标、不读取地图、不做合法方向重判；这些信息均由上游阶段提供。
    """

    config = DynamicStrategyFittingConfig() if config is None else config
    required_columns = [
        "row_id",
        "DayTrial",
        "ifscared1",
        "ifscared2",
        "energizers",
        f"{player}_action_dir",
        f"{player}_available_dir",
    ]
    required_columns.extend(f"{player}_{agent}_Q_norm" for agent in config.agents)
    missing_columns = [column for column in required_columns if column not in raw_data.columns]
    if missing_columns:
        raise ValueError(
            f"{player} 动态拟合输入缺少 calculate_utility 阶段应生成的字段："
            f"{missing_columns}。请先运行 script/05_calculate_utility.py。"
        )

    data = raw_data.copy(deep=True).reset_index(drop=True)
    # 玩家字段只在临时视角中改名；最终输出仍写回 p1/p2 前缀字段。
    data["action_dir"] = data[f"{player}_action_dir"].apply(lambda value: value if value is not None else np.nan)
    data["available_dir"] = data[f"{player}_available_dir"].astype(bool)
    data["row_id"] = pd.to_numeric(data["row_id"], errors="raise").astype("int64")

    # 死亡或缺失玩家位置时保留 joint 行，但该玩家不参与有效动作拟合。
    alive_column = f"{player}_alive"
    if alive_column in data.columns:
        dead_mask = ~data[alive_column].astype(bool)
        data.loc[dead_mask, "action_dir"] = np.nan
        data.loc[dead_mask, "available_dir"] = False

    for agent in config.agents:
        data[f"{agent}_Q_norm"] = data[f"{player}_{agent}_Q_norm"]
    return data


def all_directions_nan(
    data: pd.DataFrame,
    context: tuple[int, int],
) -> bool:
    """判断一个段落是否没有任何可拟合方向行。

    输入语义：data 是单个 trial 表，context 是半开区间 ``(start, end)``。
    输出语义：若所有行都是 NaN 方向或非法方向，则返回 True。
    关键约束：该判断同时用于段落合并和普通拟合过滤。
    """

    start, end = context
    segment = data.iloc[start:end]
    temp_data = copy.deepcopy(segment)
    temp_data["nan_dir"] = temp_data.action_dir.apply(lambda value: isinstance(value, float))
    valid_data = segment[(temp_data.nan_dir == False) & (temp_data.available_dir == True)]
    return valid_data.shape[0] == 0


def change_direction_indices(directions: pd.Series) -> np.ndarray | list[int]:
    """找出 Pacman 方向发生变化的位置。

    输入语义：directions 是一个 trial 内的 ``action_dir`` 序列。
    输出语义：返回切点列表，最后一个切点总是 trial 长度。
    关键约束：连续相邻变化只保留间隔大于 1 的位置，保留旧切点定义。
    """

    changed = pd.Series((directions != directions.shift()).where(lambda value: value == True).dropna().index)
    changed = changed[(changed - changed.shift()) > 1].values
    if len(changed) > 0 and changed[-1] != len(directions):
        changed = np.array(list(changed) + [len(directions)])
    if len(changed) == 0:
        changed = [len(directions)]
    return changed


def merge_context(contexts: list[tuple[int, int]], cutoff_points: list[int]) -> list[int]:
    """把不可切开的上下文区间合并进切点序列。

    输入语义：contexts 是需要保持完整的半开区间，cutoff_points 是候选切点。
    输出语义：返回调整后的切点列表。
    关键约束：保留旧流程对全 NaN 段和吃鬼段不被切开的处理方式。
    """

    new_cutoff_points = [0]
    pointer = 0
    end = -1
    if len(contexts) > 0:
        for index in range(len(cutoff_points)):
            if cutoff_points[index] < contexts[pointer][0] and cutoff_points[index] > new_cutoff_points[-1]:
                new_cutoff_points.append(cutoff_points[index])
            elif cutoff_points[index] >= contexts[pointer][0] and cutoff_points[index] <= contexts[pointer][1]:
                new_cutoff_points.append(contexts[pointer][0])
                new_cutoff_points.append(contexts[pointer][1])
                pointer += 1
            elif cutoff_points[index] > contexts[pointer][1]:
                new_cutoff_points.append(contexts[pointer][0])
                new_cutoff_points.append(contexts[pointer][1])
                new_cutoff_points.append(cutoff_points[index])
                pointer += 1
            end = index
            if pointer >= len(contexts):
                break
    for index in range(end + 1, len(cutoff_points)):
        if cutoff_points[index] > new_cutoff_points[-1]:
            new_cutoff_points.append(cutoff_points[index])
    return new_cutoff_points[1:]


def add_event_cutoff_points(cutoff_points: np.ndarray | list[int], trial_data: pd.DataFrame, stay_length: int) -> tuple[list[int], list[int], list[int]]:
    """把吃 energizer 和吃 ghost 的事件位置加入段落切点。

    输入语义：cutoff_points 是方向变化切点，trial_data 是 reset index 后的单 trial 数据。
    输出语义：返回新切点、吃 energizer 位置和吃 ghost 位置。
    关键约束：长 NaN 段和吃鬼上下文会通过 merge_context 保持完整。
    """

    eat_ghost = (
        (
            ((trial_data.ifscared1 == 3) & (trial_data.ifscared1.diff() < 0))
            | ((trial_data.ifscared2 == 3) & (trial_data.ifscared2.diff() < 0))
        )
        .where(lambda value: value == True)
        .dropna()
        .index.tolist()
    )
    eat_energizers = (
        (
            trial_data.energizers.apply(lambda value: len(value) if not isinstance(value, float) else 0).diff() < 0
        )
        .where(lambda value: value == True)
        .dropna()
        .index.tolist()
    )
    merged_cutoffs = sorted(list(cutoff_points) + eat_ghost + eat_energizers)
    merged_cutoffs = list(set(merged_cutoffs))
    merged_cutoffs.sort()

    # 长连续 NaN 方向段不能被普通方向切点切开，否则后续 stay 标记会错位。
    temp_direction_flags = [0 if isinstance(value, float) else 1 for value in trial_data.action_dir]
    nan_indices = np.where(np.array(temp_direction_flags) == 0)[0]
    nan_contexts: list[tuple[int, int]] = []
    if len(nan_indices) > 0:
        start = nan_indices[0]
        count = 1
        for index in range(1, len(nan_indices)):
            if nan_indices[index] != nan_indices[index - 1] + 1:
                if count >= stay_length:
                    nan_contexts.append((start, nan_indices[index - 1] + 1))
                start = nan_indices[index]
                count = 1
            else:
                count += 1
    merged_cutoffs = merge_context(nan_contexts, merged_cutoffs)

    # 吃 energizer 后到最后一次吃 ghost 的区间也不能被切开。
    eat_ghost_contexts: list[tuple[int, int]] = []
    for index, eat_energizer in enumerate(eat_energizers):
        start = eat_energizer
        if index == len(eat_energizers) - 1:
            end = len(trial_data)
        else:
            end = eat_energizers[index + 1]
        last_eat_ghost = None
        for ghost_index in eat_ghost:
            if start < ghost_index < end:
                last_eat_ghost = ghost_index
        if last_eat_ghost is not None:
            eat_ghost_contexts.append((start, last_eat_ghost))
    merged_cutoffs = merge_context(eat_ghost_contexts, merged_cutoffs)
    return merged_cutoffs, eat_energizers, eat_ghost


def label_context_events(
    directions: pd.Series,
    contexts: list[tuple[int, int]],
    eat_energizers: list[int],
    eat_ghost: list[int],
) -> tuple[list[int], list[bool]]:
    """为每个段落标记事件类型。

    输入语义：contexts 是单 trial 段落；eat_energizers/eat_ghost 是 trial 内事件位置。
    输出语义：event 中 0=全 NaN，1=吃 energizer，2=吃 ghost，3=普通段。
    关键约束：事件指针只向前移动，保持旧脚本的优先级和边界规则。
    """

    events: list[int] = []
    is_nan: list[bool] = []
    energizer_pointer = 0
    ghost_pointer = 0
    for start, end in contexts:
        if np.all(directions.iloc[start:end].apply(lambda value: isinstance(value, float)) == True):
            is_nan.append(True)
            events.append(0)
            if (
                energizer_pointer < len(eat_energizers)
                and eat_energizers[energizer_pointer] > start
                and eat_energizers[energizer_pointer] <= end
            ):
                while energizer_pointer < len(eat_energizers) and eat_energizers[energizer_pointer] > start and eat_energizers[energizer_pointer] <= end:
                    energizer_pointer += 1
        else:
            if (
                energizer_pointer < len(eat_energizers)
                and eat_energizers[energizer_pointer] > start
                and eat_energizers[energizer_pointer] <= end
            ):
                events.append(1)
                while energizer_pointer < len(eat_energizers) and eat_energizers[energizer_pointer] > start and eat_energizers[energizer_pointer] <= end:
                    energizer_pointer += 1
            elif ghost_pointer < len(eat_ghost) and eat_ghost[ghost_pointer] > start and eat_ghost[ghost_pointer] <= end:
                events.append(2)
                while ghost_pointer < len(eat_ghost) and eat_ghost[ghost_pointer] > start and eat_ghost[ghost_pointer] <= end:
                    ghost_pointer += 1
            else:
                events.append(3)
            is_nan.append(False)
    return events, is_nan


def context_needs_merge(
    event: int,
    context: tuple[int, int],
    trial_data: pd.DataFrame,
    stay_length: int,
) -> bool:
    """判断一个段落是否需要并入相邻段。

    输入语义：event 是段落事件类型，context 是段落半开区间。
    输出语义：需要合并返回 True，否则 False。
    关键约束：这是段落合并规则的核心判断，必须和旧流程保持一致。
    """

    length = context[1] - context[0]
    if event == 0 and length >= stay_length:
        return False
    if event == 0 and length < stay_length:
        return True
    if event in (1, 2):
        return all_directions_nan(trial_data, context)
    if event == 3:
        return not (length > 3 and all_directions_nan(trial_data, context) == False)
    raise ValueError(f"未知段落事件类型：{event}")


def merge_short_contexts(
    contexts: list[tuple[int, int]],
    events: list[int],
    trial_data: pd.DataFrame,
    stay_length: int,
) -> tuple[list[tuple[int, int]], list[bool]]:
    """按事件类型和长度合并不可独立拟合的短段落。

    输入语义：contexts/events 是单 trial 的初始段落和事件标签。
    输出语义：返回合并后的段落，以及每段是否是 stay/all-nan 段。
    关键约束：相邻段能否接受当前段、合并方向选择和事件优先级都保留旧规则。
    """

    need_merge: list[bool] = []
    accept_merge: list[list[int]] = []
    event_acceptance = {0: [1, 2, 3], 1: [1], 2: [0, 2, 3], 3: [0, 1, 2, 3]}

    for index in range(len(contexts)):
        need_merge.append(context_needs_merge(events[index], contexts[index], trial_data, stay_length))
        if events[index] == 0 and (contexts[index][1] - contexts[index][0]) >= stay_length:
            accept_merge.append([])
        else:
            accept_merge.append(event_acceptance[events[index]])

    index = 0
    while index < len(contexts):
        if need_merge[index] is False:
            index += 1
            continue

        status = events[index]
        front_length = np.inf
        tail_length = np.inf
        can_be_accepted = False
        if index > 0 and status in accept_merge[index - 1]:
            front_length = contexts[index - 1][1] - contexts[index - 1][0]
            can_be_accepted = True
        if index != len(contexts) - 1 and status in accept_merge[index + 1]:
            tail_length = contexts[index + 1][1] - contexts[index + 1][0]
            can_be_accepted = True

        if can_be_accepted is False:
            if all_directions_nan(trial_data, contexts[index]) is False:
                need_merge[index] = False
                index += 1
                continue
            if index == len(contexts) - 1:
                front_length = contexts[index - 1][1] - contexts[index - 1][0]
            elif index == 0:
                tail_length = contexts[index + 1][1] - contexts[index + 1][0]
            elif events[index - 1] == 1:
                front_length = contexts[index - 1][1] - contexts[index - 1][0]
            elif events[index + 1] == 1:
                tail_length = contexts[index + 1][1] - contexts[index + 1][0]
            else:
                front_length = contexts[index - 1][1] - contexts[index - 1][0]
                tail_length = contexts[index + 1][1] - contexts[index + 1][0]
                print("front and tail is not!", "=" * 50)

        if front_length < tail_length:
            events[index - 1] = merge_event_labels(events[index - 1], events[index])
            contexts[index - 1] = (contexts[index - 1][0], contexts[index][1])
            accept_merge[index - 1] = event_acceptance[events[index - 1]]
            need_merge[index - 1] = context_needs_merge(
                events[index - 1], contexts[index - 1], trial_data, stay_length
            )
            contexts = contexts[:index] + contexts[index + 1 :]
            events = events[:index] + events[index + 1 :]
            accept_merge = accept_merge[:index] + accept_merge[index + 1 :]
            need_merge = need_merge[:index] + need_merge[index + 1 :]
            index -= 2
        else:
            contexts[index] = (contexts[index][0], contexts[index + 1][1])
            events[index] = merge_event_labels(events[index], events[index + 1])
            accept_merge[index] = event_acceptance[events[index]]
            need_merge[index] = context_needs_merge(events[index], contexts[index], trial_data, stay_length)
            contexts = contexts[: index + 1] + contexts[index + 2 :]
            events = events[: index + 1] + events[index + 2 :]
            accept_merge = accept_merge[: index + 1] + accept_merge[index + 2 :]
            need_merge = need_merge[: index + 1] + need_merge[index + 2 :]
            index -= 1
        index += 1

    is_nan = [event == 0 for event in events]
    cannot_fit: list[bool] = []
    for index in range(len(contexts)):
        if context_needs_merge(events[index], contexts[index], trial_data, stay_length) is False:
            cannot_fit.append(False)
        elif all_directions_nan(trial_data, contexts[index]) is False:
            cannot_fit.append(False)
        else:
            cannot_fit.append(True)
    if np.sum(cannot_fit) != 0:
        print("need_combine is not 0!", "=" * 50)

    for index in range(len(contexts) - 1):
        if contexts[index][1] != contexts[index + 1][0]:
            print("loss data!", "=" * 50)
    if contexts[-1][1] != len(trial_data):
        print("loss data!", "=" * 50)
    return contexts, is_nan


def merge_event_labels(left_event: int, right_event: int) -> int:
    """合并两个相邻段落的事件标签。

    输入语义：left_event/right_event 是旧事件编号。
    输出语义：返回合并后事件编号。
    关键约束：非 0 事件取较小编号，0 只在另一侧也是 0 时保留为 0。
    """

    if left_event != 0 and right_event != 0:
        return min(left_event, right_event)
    if left_event == 0 and right_event != 0:
        return right_event
    if left_event != 0 and right_event == 0:
        return left_event
    return 0


def build_context_segments(
    prepared_data: pd.DataFrame,
    config: DynamicStrategyFittingConfig | None = None,
) -> tuple[list[tuple[int, int]], list[bool], list[int], list[int]]:
    """为完整被试数据构造全局段落列表。

    输入语义：prepared_data 是 prepare_fitting_dataframe 后的数据。
    输出语义：返回全局坐标段落、stay 标记、吃 energizer 行号和吃 ghost 行号。
    关键约束：每个 trial 内先独立切段，再用 ``row_id`` 映射回完整 DataFrame 行号。
    """

    config = DynamicStrategyFittingConfig() if config is None else config
    all_contexts: list[tuple[int, int]] = []
    all_is_nan: list[bool] = []
    all_eat_energizers: list[int] = []
    all_eat_ghost: list[int] = []

    for trial_index, trial_name in enumerate(np.unique(prepared_data.DayTrial.values)):
        trial_data = prepared_data[prepared_data.DayTrial == trial_name]
        trial_data.reset_index(drop=True, inplace=True)
        print(f"| ({trial_index}) {trial_name} | Data shape {trial_data.shape}")

        cutoffs, eat_energizers, eat_ghost = add_event_cutoff_points(
            change_direction_indices(trial_data.action_dir),
            trial_data,
            config.stay_length,
        )
        level_offset = int(trial_data["row_id"].iloc[0])
        all_eat_energizers += [eat_index + level_offset for eat_index in eat_energizers]
        all_eat_ghost += [eat_index + level_offset for eat_index in eat_ghost]

        contexts = list(zip([0] + list(cutoffs[:-1]), cutoffs))
        events, is_nan = label_context_events(trial_data.action_dir, contexts, eat_energizers, eat_ghost)
        contexts, is_nan = merge_short_contexts(contexts, events, trial_data, config.stay_length)
        global_contexts = [(context[0] + level_offset, context[1] + level_offset) for context in contexts]
        all_is_nan += is_nan
        all_contexts += global_contexts

    return all_contexts, all_is_nan, all_eat_energizers, all_eat_ghost


def build_agent_q_values(data: pd.DataFrame, agents: tuple[str, ...], suffix: str) -> np.ndarray:
    """把每行每个 agent 的四方向 Q 值整理为三维数组。

    输入语义：data 是有效方向行，agent Q 列的每个单元是长度为 4 的数组。
    输出语义：返回形状 ``(样本数, 4, agent数)`` 的数组。
    关键约束：agent 顺序就是权重向量顺序，不能重排。
    """

    q_columns = [f"{agent}{suffix}" for agent in agents]
    # 每个 Q 单元都是长度为 4 的方向数组。这里一次性堆叠成
    # sample x direction x agent，避免在每个 context 内反复 Python 双层循环。
    return np.stack(
        [np.stack(data[column].to_numpy()).astype(float) for column in q_columns],
        axis=2,
    )


def weighted_direction_q(agent_q_values: np.ndarray, weights: Any) -> np.ndarray:
    """用策略权重合成四方向 Q。

    输入语义：agent_q_values 形状为 ``(样本数, 4, 策略数)``，weights 是策略权重。
    输出语义：返回形状为 ``(样本数, 4)`` 的合成方向 Q。
    关键约束：保持旧逻辑中 ``0 * -inf`` 先产生 NaN、再统一转为 ``-inf`` 的行为。
    """

    with np.errstate(invalid="ignore"):
        direction_q_values = agent_q_values @ weights
    direction_q_values[np.isnan(direction_q_values)] = -np.inf
    return direction_q_values


def expected_argmax_accuracy(direction_q_values: np.ndarray, true_direction: np.ndarray) -> float:
    """计算真实方向在最大 Q 并列集合中的期望正确数。

    输入语义：direction_q_values 是每个样本的四方向合成 Q，true_direction 是真实方向索引。
    输出语义：返回旧循环逻辑中的 accuracy 累加值；若真实方向与 k 个最大方向并列，
    贡献 ``1/k``。
    关键约束：真实方向为 ``-inf`` 时贡献 0，与旧实现保持一致。
    """

    if direction_q_values.shape[0] == 0:
        return 0.0
    row_indices = np.arange(direction_q_values.shape[0])
    target_values = direction_q_values[row_indices, true_direction]
    max_values = np.max(direction_q_values, axis=1)
    tie_counts = np.sum(direction_q_values == max_values[:, None], axis=1)
    correct_mask = (~np.isinf(target_values)) & (target_values == max_values)
    return float(np.sum(np.where(correct_mask, 1 / tie_counts, 0.0)))


def weighted_accuracy_objective(
    weights: Any,
    agent_q_values: np.ndarray,
    true_direction: np.ndarray,
    penalty: float,
) -> float:
    """计算单个 context 的权重拟合目标。

    输入语义：weights 是 GA 候选权重，agent_q_values/true_direction 是 context 内预计算数据。
    输出语义：返回旧 GA likelihood 使用的目标值，数值越小越好。
    关键约束：该函数只把逐样本循环改成 NumPy 向量化，不改变目标函数定义。
    """

    sample_count = agent_q_values.shape[0]
    if sample_count == 0:
        return np.inf
    direction_q_values = weighted_direction_q(agent_q_values, weights)
    accuracy = expected_argmax_accuracy(direction_q_values, true_direction)
    return -accuracy / sample_count + penalty * np.sum(np.abs(weights))


def negative_likelihood(
    weights: Any,
    data: pd.DataFrame,
    true_prob: pd.Series,
    agents: tuple[str, ...],
    suffix: str = "_Q",
    return_trajectory: bool = False,
) -> Any:
    """计算旧动态拟合使用的方向准确率目标。

    输入语义：weights 是 agent 权重，data 是有效方向行，true_prob 是真实方向 one-hot。
    输出语义：默认返回目标值；return_trajectory=True 时同时返回四方向综合 Q。
    关键约束：该函数名称沿用 likelihood 语义，但旧目标实际是准确率惩罚而非标准 log-likelihood。
    """

    if len(agents) == 0:
        raise ValueError("agents 不能为空。")
    agent_weights = [weights[index] for index in range(len(weights))]
    sample_count = data.shape[0]
    agent_q_values = build_agent_q_values(data, agents, suffix)
    direction_q_values = weighted_direction_q(agent_q_values, agent_weights)
    true_directions = true_prob.apply(choose_max_direction).values
    accuracy = expected_argmax_accuracy(direction_q_values, true_directions)
    objective = (1 - accuracy / sample_count) * 1000 + np.sum(np.abs(agent_weights))
    if return_trajectory:
        return objective, direction_q_values
    return objective


def calculate_correct_rate(
    weights: Any,
    data: pd.DataFrame,
    true_prob: pd.Series,
    agents: tuple[str, ...],
    suffix: str = "_Q",
) -> float:
    """计算拟合权重的平均方向预测正确率。

    输入语义：weights 是拟合得到的 agent 权重，data/true_prob 是有效方向样本。
    输出语义：返回 100 次随机并列选择下的平均正确率。
    关键约束：旧流程在并列最大方向上重复随机选择 100 次；这里保留这个随机诊断语义。
    """

    _, estimated_prob = negative_likelihood(weights, data, true_prob, agents, return_trajectory=True, suffix=suffix)
    true_direction = np.array([np.argmax(each) for each in true_prob])
    correct_rate_sum = 0
    for _ in range(100):
        estimated_direction = np.array([choose_max_direction(each) for each in estimated_prob])
        correct_rate_sum += np.sum(estimated_direction == true_direction) / len(estimated_direction)
    return correct_rate_sum / 100


def calculate_is_correct(
    weights: Any,
    data: pd.DataFrame,
    true_prob: pd.Series,
    agents: tuple[str, ...],
    suffix: str = "_Q",
) -> tuple[np.ndarray, np.ndarray]:
    """生成每个有效方向样本的预测是否正确和预测方向。

    输入语义：weights/data/true_prob 与 calculate_correct_rate 相同。
    输出语义：返回布尔正确数组和方向索引数组。
    关键约束：并列方向仍使用随机选择，因此验证时必须固定随机种子。
    """

    _, estimated_prob = negative_likelihood(weights, data, true_prob, agents, return_trajectory=True, suffix=suffix)
    true_direction = np.array([np.argmax(each) for each in true_prob])
    estimated_direction = np.array([choose_max_direction(each) for each in estimated_prob])
    return estimated_direction == true_direction, estimated_direction


def fit_one_segment(
    segment_index: int,
    contexts: list[tuple[int, int]],
    is_nan: list[bool],
    data: pd.DataFrame,
    config: DynamicStrategyFittingConfig | None = None,
    suffix: str = "_Q_norm",
    segment_seed: int | None = None,
) -> dict[str, Any] | None:
    """拟合一个动态策略段落的 agent 权重。

    输入语义：segment_index 指向 contexts/is_nan 中的一个段落。
    输出语义：返回旧流程需要写回的权重、有效行索引、预测结果和 vague 标记。
    关键约束：stay 段直接返回全 0 权重；普通段使用旧 GA 参数拟合。
    """

    config = DynamicStrategyFittingConfig() if config is None else config
    if segment_seed is not None:
        np.random.seed(segment_seed)
    start, end = contexts[segment_index]
    print(start, end)
    if is_nan[segment_index] is True:
        return {
            "resultlist": [0] * len(config.agents) + [0] + [start] + [end],
            "ind": None,
            "phase_is_correct": None,
            "predict_dir": None,
            "is_vague": False,
            "loss": None,
        }

    segment = data[start:end]
    temp_data = copy.deepcopy(segment)
    temp_data["nan_dir"] = temp_data.action_dir.apply(lambda value: isinstance(value, float))
    valid_data = segment[(temp_data.nan_dir == False) & (temp_data.available_dir == True)]
    if valid_data.shape[0] == 0:
        print(f"All the directions are nan from {start} to {end}!")
        return None

    valid_indices = np.where((temp_data.nan_dir == False) & (temp_data.available_dir == True))[0] + start
    effective_action_ratio = valid_data.shape[0] / max(end - start, 1)
    if (
        valid_data.shape[0] < config.min_effective_action_count
        or effective_action_ratio < config.min_effective_action_ratio
    ):
        # 有效动作太少或占比太低时，段落通常是短暂停顿、回摆或长 stay 边缘。
        # 这类段落即使某个策略偶然完全命中少数动作，也不应被解释成确定策略；
        # 因此直接标为 vague，并保留 0 权重，避免 GA 对低证据段过拟合。
        return {
            "resultlist": [0] * len(config.agents) + [0] + [start] + [end],
            "ind": None,
            "phase_is_correct": None,
            "predict_dir": None,
            "is_vague": True,
            "loss": None,
        }

    true_prob = valid_data.action_dir.ffill().apply(one_hot_direction)
    true_direction = true_prob.apply(choose_max_direction).values
    sample_count = valid_data.shape[0]
    agent_q_values = build_agent_q_values(valid_data, config.agents, suffix)

    def likelihood(agent_weights: Any) -> float:
        """计算 GA 优化器调用的段落目标值。"""

        return weighted_accuracy_objective(
            agent_weights,
            agent_q_values,
            true_direction,
            config.weight_penalty,
        )

    patch_multiprocessing_start_method_for_sko()
    from sko.GA import GA

    ga = GA(
        func=likelihood,
        n_dim=len(config.agents),
        size_pop=config.ga_population_size,
        max_iter=config.ga_iterations,
        prob_mut=config.ga_mutation_probability,
        lb=[0] * len(config.agents),
        ub=[1] * len(config.agents),
        precision=config.ga_precision,
    )
    weights, loss = ga.run(config.ga_iterations)
    correct_rate = calculate_correct_rate(weights, valid_data, true_prob, config.agents, suffix=suffix)
    max_agent_index = np.argmax(weights)
    single_agent_weight = [0] * len(config.agents)
    single_agent_weight[max_agent_index] = 1
    single_agent_correct_rate = calculate_correct_rate(np.array(single_agent_weight), valid_data, true_prob, config.agents, suffix=suffix)
    # 旧流程先进入 if 分支再赋值 Python True；这里显式转成 bool，避免 np.bool_ 在后续
    # ``is True`` 写回判断中被当成 False。
    is_vague = bool(single_agent_correct_rate <= config.vague_accuracy_threshold)
    phase_is_correct, predict_direction = calculate_is_correct(weights, valid_data, true_prob, config.agents, suffix=suffix)
    return {
        "resultlist": weights.tolist() + [correct_rate] + [start] + [end],
        "ind": valid_indices,
        "phase_is_correct": phase_is_correct,
        "predict_dir": predict_direction,
        "is_vague": is_vague,
        "loss": loss,
    }


def patch_multiprocessing_start_method_for_sko() -> None:
    """兼容 sko 导入期重复设置 multiprocessing start method 的行为。

    输入语义：无显式输入，补丁作用于当前进程的 multiprocessing 模块。
    输出语义：重复设置 start method 时忽略 ``context has already been set``。
    关键约束：只吞掉这个已知兼容错误，其它 RuntimeError 继续抛出。
    """

    if getattr(multiprocessing.set_start_method, "_lops_safe_patch", False):
        return
    original_set_start_method = multiprocessing.set_start_method

    def safe_set_start_method(method: str, force: bool = False) -> None:
        """安全调用 set_start_method，兼容 sko 在 worker 中的重复设置。"""

        try:
            original_set_start_method(method, force=force)
        except RuntimeError as exc:
            if "context has already been set" not in str(exc):
                raise

    safe_set_start_method._lops_safe_patch = True  # type: ignore[attr-defined]
    multiprocessing.set_start_method = safe_set_start_method


def fit_all_segments(
    data: pd.DataFrame,
    contexts: list[tuple[int, int]],
    is_nan: list[bool],
    config: DynamicStrategyFittingConfig | None = None,
    suffix: str = "_Q_norm",
) -> tuple[list[Any], Any, np.ndarray, np.ndarray, np.ndarray]:
    """顺序拟合一个被试的全部动态段落。

    输入语义：contexts/is_nan 覆盖 prepared DataFrame 的所有行。
    输出语义：返回 result_list、总 loss、逐行正确性、逐行预测方向和 vague 标记。
    关键约束：正式行为必须拟合全部段落；旧源码只拟合第一个段落是调试残留。
    """

    config = DynamicStrategyFittingConfig() if config is None else config
    result_list: list[Any] = []
    total_loss: Any = 0
    is_correct = np.zeros((data.shape[0],))
    is_correct[is_correct == 0] = np.nan
    predicted_direction = np.zeros((data.shape[0],))
    predicted_direction[predicted_direction == 0] = np.nan
    is_vague = np.array([False] * len(data))

    if config.segment_workers > 1:
        segment_results = fit_segments_in_parallel(data, contexts, is_nan, config, suffix)
    else:
        segment_results = []
        for index in range(len(contexts)):
            segment_seed = config.random_seed + index if config.random_seed is not None and config.use_segment_seed else None
            segment_results.append(
                fit_one_segment(index, contexts, is_nan, data, config, suffix=suffix, segment_seed=segment_seed)
            )

    for index, result in enumerate(segment_results):
        if result is None:
            continue
        result_list.append(result["resultlist"])
        valid_indices = result["ind"]
        if valid_indices is not None:
            is_correct[valid_indices] = result["phase_is_correct"]
            predicted_direction[valid_indices] = result["predict_dir"]
        if result["is_vague"] is True:
            start, end = contexts[index]
            is_vague[start:end] = [True] * (end - start)
        if result["loss"] is not None:
            total_loss += result["loss"]

    return result_list, total_loss, is_correct, predicted_direction, is_vague


_SEGMENT_DATA: pd.DataFrame | None = None
_SEGMENT_CONTEXTS: list[tuple[int, int]] | None = None
_SEGMENT_IS_NAN: list[bool] | None = None
_SEGMENT_CONFIG: DynamicStrategyFittingConfig | None = None
_SEGMENT_SUFFIX: str = "_Q_norm"


def fit_segments_in_parallel(
    data: pd.DataFrame,
    contexts: list[tuple[int, int]],
    is_nan: list[bool],
    config: DynamicStrategyFittingConfig,
    suffix: str,
) -> list[dict[str, Any] | None]:
    """并行拟合一个文件内的全部段落。

    输入语义：data/context/is_nan 是完整文件级拟合状态，config.segment_workers 控制进程数。
    输出语义：返回与 contexts 顺序一致的段落结果列表。
    关键约束：设置 random_seed 时每段使用 ``file_seed + segment_index``，保证并行顺序不影响结果。
    """

    tasks = [
        (index, config.random_seed + index if config.random_seed is not None else None)
        for index in range(len(contexts))
    ]
    with ProcessPoolExecutor(
        max_workers=min(config.segment_workers, len(tasks)),
        initializer=_init_segment_worker,
        initargs=(data, contexts, is_nan, config, suffix),
    ) as executor:
        return list(executor.map(_fit_segment_worker_task, tasks))


def _init_segment_worker(
    data: pd.DataFrame,
    contexts: list[tuple[int, int]],
    is_nan: list[bool],
    config: DynamicStrategyFittingConfig,
    suffix: str,
) -> None:
    """初始化段落级并行 worker 的只读上下文。

    输入语义：父进程传入完整文件数据、段落列表和配置。
    输出语义：写入 worker 进程全局变量，减少每个段落任务的重复 pickle。
    关键约束：worker 内只读这些对象，不跨段落写共享状态。
    """

    global _SEGMENT_DATA, _SEGMENT_CONTEXTS, _SEGMENT_IS_NAN, _SEGMENT_CONFIG, _SEGMENT_SUFFIX
    _SEGMENT_DATA = data
    _SEGMENT_CONTEXTS = contexts
    _SEGMENT_IS_NAN = is_nan
    _SEGMENT_CONFIG = config
    _SEGMENT_SUFFIX = suffix


def _fit_segment_worker_task(task: tuple[int, int | None]) -> dict[str, Any] | None:
    """执行一个段落级并行任务。

    输入语义：task 包含段落 index 和该段随机种子。
    输出语义：返回 fit_one_segment 的结果。
    关键约束：依赖 _init_segment_worker 已经设置的只读上下文。
    """

    if (
        _SEGMENT_DATA is None
        or _SEGMENT_CONTEXTS is None
        or _SEGMENT_IS_NAN is None
        or _SEGMENT_CONFIG is None
    ):
        raise RuntimeError("段落 worker 尚未初始化。")
    index, segment_seed = task
    return fit_one_segment(
        index,
        _SEGMENT_CONTEXTS,
        _SEGMENT_IS_NAN,
        _SEGMENT_DATA,
        _SEGMENT_CONFIG,
        suffix=_SEGMENT_SUFFIX,
        segment_seed=segment_seed,
    )

def discover_player_prefixes(data: pd.DataFrame, config: DynamicStrategyFittingConfig) -> list[str]:
    """识别当前 05 输出中可进行 06 拟合的玩家。

    输入语义：data 是 joint-state utility 表，config.agents 给出需要的策略名。
    输出语义：返回字段完整的玩家前缀列表，例如 ``["p1", "p2"]``。
    关键约束：单人数据没有 ``p2_*`` 时跳过 p2，不生成 p2 权重字段。
    """

    players: list[str] = []
    for player in PLAYER_PREFIXES:
        required_columns = {f"{player}_action_dir", f"{player}_available_dir"}
        required_columns.update(f"{player}_{agent}_Q_norm" for agent in config.agents)
        if required_columns.isdisjoint(data.columns):
            continue
        missing_columns = sorted(required_columns - set(data.columns))
        if missing_columns:
            raise ValueError(f"{player} 动态拟合字段不完整，缺少：{missing_columns}")
        players.append(player)
    if not players:
        raise ValueError("未找到可拟合玩家字段，至少需要 p1_action_dir/p1_available_dir/p1_*_Q_norm。")
    return players


def initialize_player_result(index: pd.Index, player: str) -> pd.DataFrame:
    """创建某个玩家的 06 输出字段空表。

    输入语义：index 与 joint-state 输入表一致，player 是 ``p1`` 或 ``p2``。
    输出语义：返回包含 ``p1_weight`` 等字段的 DataFrame，默认值为 NaN。
    关键约束：所有列使用 object dtype，便于保存 list、tuple、bool 和 NaN 混合值。
    """

    output = pd.DataFrame(index=index)
    for column in PLAYER_RESULT_COLUMNS:
        output[f"{player}_{column}"] = pd.Series([np.nan] * len(index), index=index, dtype=object)
    return output


def fit_player_strategy_dataframe(
    raw_data: pd.DataFrame,
    player: str,
    config: DynamicStrategyFittingConfig,
) -> pd.DataFrame:
    """为一个玩家拟合动态策略权重并返回玩家前缀结果列。

    输入语义：raw_data 是 05 输出的 joint-state 表，player 指定当前玩家。
    输出语义：返回 ``<player>_weight``、``<player>_trial_context`` 等结果列。
    关键约束：死亡和非法动作行不会被作为有效动作拟合，但 joint 行仍完整保留。
    """

    if config.random_seed is not None:
        np.random.seed(config.random_seed)

    print(f"=== Dynamic Strategy Fitting: {player} ====")
    fit_data = prepare_fitting_dataframe(raw_data, player, config)
    suffix = "_Q_norm"
    trial_names = np.unique(fit_data.DayTrial.values)
    print("The num of trials : ", len(trial_names))
    print("-" * 50)

    # context 划分和拟合阶段只把合法动作当作有效观测；保存结果时不改写原始玩家动作字段。
    invalid_direction_indices = np.where(fit_data["available_dir"] == False)[0]
    fit_data.loc[fit_data.index[invalid_direction_indices], "action_dir"] = [np.nan] * len(invalid_direction_indices)

    contexts, is_nan, _eat_energizers, _eat_ghost = build_context_segments(fit_data, config)
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

    if len(trial_weight) == fit_data.shape[0] and len(trial_weight) > 0:
        output[f"{player}_weight"] = trial_weight
        output[f"{player}_normalized_weight"] = trial_normalized_weight
        output[f"{player}_prediction_correct"] = is_correct
        output[f"{player}_predict_dir"] = predicted_direction
        output[f"{player}_trial_context"] = trial_context
        output[f"{player}_is_stay"] = trial_is_stay
        output[f"{player}_is_vague"] = is_vague
        print(np.sum(is_vague) / len(fit_data))

    print(f"Finished fitting {player}.")
    return output


def append_public_event_columns(data: pd.DataFrame) -> pd.DataFrame:
    """为 joint-state 输出添加公共局面事件标记。

    输入语义：data 是 05 utility 输出或已追加玩家权重字段的 joint-state 表。
    输出语义：返回追加 ``eat_energizer`` 和 ``eat_ghost`` 的 DataFrame。
    关键约束：这两个事件来自公共游戏状态，不归属于 p1 或 p2；它们会参与两个玩家
    各自的 context 划分，但保存时只保留一份，避免误解为玩家本人触发事件。
    """

    required_columns = {"DayTrial", "energizers", "ifscared1", "ifscared2"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"公共事件标记缺少字段：{missing_columns}")

    result = data.copy(deep=True)
    result["eat_energizer"] = False
    result["eat_ghost"] = False
    for _, trial_data in result.groupby("DayTrial", sort=False):
        energizer_count = trial_data["energizers"].apply(lambda value: len(value) if not isinstance(value, float) else 0)
        eat_energizer_indices = energizer_count.diff().where(lambda value: value < 0).dropna().index
        eat_ghost_indices = (
            (
                ((trial_data.ifscared1 == 3) & (trial_data.ifscared1.diff() < 0))
                | ((trial_data.ifscared2 == 3) & (trial_data.ifscared2.diff() < 0))
            )
            .where(lambda value: value == True)
            .dropna()
            .index
        )
        result.loc[eat_energizer_indices, "eat_energizer"] = True
        result.loc[eat_ghost_indices, "eat_ghost"] = True
    return result


def fit_dynamic_strategy_dataframe(
    raw_data: pd.DataFrame,
    config: DynamicStrategyFittingConfig | None = None,
) -> pd.DataFrame:
    """对单个 joint-state DataFrame 执行完整动态策略拟合。

    输入语义：raw_data 是 05 utility 输出表，包含 p1/p2 的 Q_norm 字段。
    输出语义：返回保留原 joint-state 字段、追加玩家前缀权重字段的 WeightData 表。
    关键约束：每个玩家分别构造 context 和拟合权重，不拆文件、不删除 joint 行。
    """

    config = DynamicStrategyFittingConfig() if config is None else config
    result = append_public_event_columns(raw_data.reset_index(drop=True).copy(deep=True))
    for player in discover_player_prefixes(result, config):
        player_output = fit_player_strategy_dataframe(result, player, config)
        for column in player_output.columns:
            result[column] = player_output[column].to_numpy()
    return result


def process_dynamic_strategy_file(
    input_path: str | Path,
    output_path: str | Path,
    config: DynamicStrategyFittingConfig | None = None,
    file_index: int = 0,
) -> dict[str, Any]:
    """处理单个集中 utility 文件并保存动态策略权重。

    输入语义：input_path 是 calculate_utility 输出 pickle，output_path 是目标 WeightData pickle。
    输出语义：写出拟合后的 DataFrame，并返回文件摘要。
    关键约束：若设置 random_seed，会按 ``random_seed + file_index`` 为每个文件设置独立种子。
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

    result = fit_dynamic_strategy_dataframe(raw_data, file_config)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as file:
        pickle.dump(result, file)
    print("Finished saving data.")
    return {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "rows": int(result.shape[0]),
        "columns": int(result.shape[1]),
        "seed": file_config.random_seed,
    }


def process_dynamic_strategy_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    config: DynamicStrategyFittingConfig | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """批量处理 05 utility 嵌套目录。

    输入语义：input_dir 是 ``comp/*.pkl``、``coop/*.pkl`` 等任务子目录结构。
    输出语义：每个输入文件按相同相对路径写到 output_dir，返回摘要列表。
    关键约束：文件间独立；设置 seed 时按排序后的文件序号派生文件级 seed。
    """

    config = DynamicStrategyFittingConfig() if config is None else config
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")
    input_files = sorted(path for path in input_dir.glob("*/*.pkl") if path.is_file())
    if not input_files:
        raise FileNotFoundError(f"输入目录中没有嵌套 pickle 文件：{input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
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
        return [_process_dynamic_strategy_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        return list(executor.map(_process_dynamic_strategy_task, tasks))


def _process_dynamic_strategy_task(
    task: tuple[
        Path,
        Path,
        DynamicStrategyFittingConfig,
        int,
    ],
) -> dict[str, Any]:
    """执行目录级并行中的单个文件任务。

    输入语义：task 包含输入路径、输出路径、配置和文件序号。
    输出语义：返回 ``process_dynamic_strategy_file`` 的摘要。
    关键约束：保持顶层函数，便于 multiprocessing 序列化。
    """

    input_path, output_path, config, file_index = task
    return process_dynamic_strategy_file(input_path, output_path, config, file_index=file_index)
