"""根据事件结果和方向证据修正 Social Pacman context 策略。

本模块接收单个玩家的 posterior 临时视图，按固定顺序执行 vague、Energizer、
Approach 和 scared-time 规则，并返回逐 context 的修正分数与策略。目录遍历、玩家
字段映射和结果写回由阶段入口负责，本模块不包含历史 GA 数据接口。
"""

from __future__ import annotations

import ast
import copy
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STRATEGY_NUMBER = {
    "global": 0,
    "local": 1,
    "evade_blinky": 2,
    "evade_clyde": 3,
    "approach": 6,
    "energizer": 7,
    "no_energizer": 8,
    "vague": 9,
    "stay": 10,
}
AGENTS = [
    "global",
    "local",
    "evade_blinky",
    "evade_clyde",
    "approach",
    "energizer",
    "no_energizer",
]
AGENT_INDEX = {agent: index for index, agent in enumerate(AGENTS)}
AGENT_INDEX_TO_STRATEGY_NUMBER = {index: STRATEGY_NUMBER[agent] for agent, index in AGENT_INDEX.items()}
SUFFIX = "_Q_norm"
AGENT_Q_COLUMNS = [f"{agent}{SUFFIX}" for agent in AGENTS]
DIRECTION_NAMES = ["left", "right", "up", "down"]
RANDOM_DIAGNOSTIC_COLUMNS = ["predict_dir", "revised_prediction_correct"]
VAGUE_REVISE_MIN_ACCURACY = 0.70
LOW_EVIDENCE_MIN_EFFECTIVE_ACTION_RATIO = 0.5
SCARED_GHOST_STATUS_MIN = 4
DEFAULT_REVISION_RELATIVE_ACCURACY_THRESHOLD = 0.8
ENERGIZER_FOLLOWUP_APPROACH_RELATIVE_ACCURACY_THRESHOLD = 0.75
ENERGIZER_OUTCOME_MIN_ACCURACY = 0.70
ENERGIZER_OUTCOME_RELATIVE_ACCURACY_THRESHOLD = 0.80


@dataclass
class ContextData:
    """保存一个规则段落中可参与方向比较的行。

    输入语义：segment 是完整段落，valid_data 是去掉无方向行后的段落。
    输出语义：valid_indices 和 nan_indices 保存原 DataFrame 标签，用于写回。
    关键约束：true_prob 只包含有效方向行，后续评分和诊断列都基于它计算。
    """

    segment: pd.DataFrame
    valid_data: pd.DataFrame
    valid_indices: np.ndarray
    nan_indices: np.ndarray
    true_prob: pd.Series


def one_hot_direction(value: str) -> list[int]:
    """把方向字符串转换为 one-hot 编码。

    输入语义：value 必须是 left/right/up/down 之一。
    输出语义：返回长度为 4 的 one-hot 列表。
    关键约束：非法方向直接抛错，保持旧脚本输入假设。
    """

    if value not in DIRECTION_NAMES:
        raise ValueError(f"未知方向：{value}")
    if not isinstance(value, str):
        raise TypeError(f"未知方向类型：{type(value)}")
    onehot_vec = [0, 0, 0, 0]
    onehot_vec[DIRECTION_NAMES.index(value)] = 1
    return onehot_vec


def choose_max_direction(probability: Any) -> int:
    """确定性选择最大 Q 值对应的方向。

    输入语义：probability 是长度为 4 的方向分数或 one-hot 向量。
    输出语义：返回最大值第一次出现的位置。
    关键约束：旧脚本这里使用无 seed 随机打破并列；新脚本只让诊断列确定化，
    下游核心列不依赖这个随机并列选择。
    """

    values = list(probability)
    max_value = max(values)
    return next(index for index, value in enumerate(values) if value == max_value)


def normalize_context_key(value: Any) -> tuple[int, int] | None:
    """把 trial_context 字段整理成可作为字典键的半开区间。

    输入语义：value 通常是 ``(start, end)``，也可能来自字符串化的元组或 numpy 数组。
    输出语义：返回整数 ``(start, end)``；无法解析时返回 None。
    关键约束：07 的优先级调整需要在整个 context 上统计 ghost scared 比例，
    因此必须把同一段落的所有行稳定归入同一个 key。
    """

    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "nan", "none"}:
            return None
        try:
            value = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return None
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def build_approach_priority_mask(data: pd.DataFrame) -> pd.Series:
    """标记哪些 context 在并列权重时应让 approach 优先。

    输入语义：data 是单个 trial 的单人临时视角，包含 ``trial_context`` 和
    ``ifscared1/ifscared2``。
    输出语义：返回与 data 同索引的 bool Series，True 表示该行所在 context 中超过一半
    tile 行至少有一只 ghost 处于 scared 状态。
    关键约束：该函数只提供“并列时 approach 可优先”的上下文条件，不直接改变权重；
    若 approach 本身不是最大权重之一，后续策略选择仍保持原规则。
    """

    priority_mask = pd.Series(False, index=data.index, dtype=bool)
    required_columns = {"trial_context", "ifscared1", "ifscared2"}
    if not required_columns <= set(data.columns):
        return priority_mask

    context_keys = data["trial_context"].apply(normalize_context_key)
    scared_rows = (
        pd.to_numeric(data["ifscared1"], errors="coerce").ge(SCARED_GHOST_STATUS_MIN)
        | pd.to_numeric(data["ifscared2"], errors="coerce").ge(SCARED_GHOST_STATUS_MIN)
    )

    # pandas 对 tuple 分组有时会把它解释成多层键；这里显式构造字典，保证
    # ``(start, end)`` 作为一个整体 context key 使用。
    context_to_labels: dict[tuple[int, int], list[Any]] = {}
    for label, context_key in context_keys.items():
        if context_key is None:
            continue
        context_to_labels.setdefault(context_key, []).append(label)

    for labels in context_to_labels.values():
        scared_ratio = float(scared_rows.loc[labels].mean())
        if scared_ratio > 0.5:
            priority_mask.loc[labels] = True
    return priority_mask


def weight_has_tied_approach_max(weight: Any) -> bool:
    """判断权重中 approach 是否与其它策略并列最大。

    输入语义：weight 是 7 个 agent 的权重向量，通常来自
    ``revised_normalized_weight``。
    输出语义：当 approach 位于最大权重集合，且最大集合至少包含两个策略时返回 True。
    关键约束：该函数用于保护“scared 多数 + approach 并列最大”的段落不被
    revise_approach 再改写为其它策略；如果 approach 不是最大并列项，则仍按原规则修正。
    """

    try:
        weight_array = np.asarray(weight, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    if weight_array.size <= AGENT_INDEX["approach"] or weight_array.size == 0:
        return False
    if np.sum(np.abs(weight_array)) == 0:
        return False
    max_value = np.max(weight_array)
    max_indices = np.where(weight_array == max_value)[0]
    return len(max_indices) > 1 and AGENT_INDEX["approach"] in max_indices


def context_has_scared_majority(segment: pd.DataFrame) -> bool:
    """判断一个 context 内是否超过一半行存在 scared ghost。

    输入语义：segment 是 ``data.loc[prev:end-1]`` 得到的同一 context 片段。
    输出语义：当至少一只 ghost 的状态码大于等于 4 的行比例超过 0.5 时返回 True。
    关键约束：ifscared=3 表示 dead，不算 scared；ifscared>=4 才表示 scared/flash scared。
    """

    if segment.empty or not {"ifscared1", "ifscared2"} <= set(segment.columns):
        return False
    scared_rows = (
        pd.to_numeric(segment["ifscared1"], errors="coerce").ge(SCARED_GHOST_STATUS_MIN)
        | pd.to_numeric(segment["ifscared2"], errors="coerce").ge(SCARED_GHOST_STATUS_MIN)
    )
    return float(scared_rows.mean()) > 0.5


def strategy_from_weight(
    weight: Any,
    is_stay: bool,
    is_vague: bool,
    strategy_to_number: dict[str, int],
    file_name: str,
    trial_name: str,
    prefer_approach_when_scared: bool = False,
) -> int:
    """根据权重向量和人工标记得到策略编号。

    输入语义：weight 是 7 个 agent 权重，is_stay/is_vague 是旧流程标记。
    输出语义：返回旧策略编号。
    关键约束：权重向量下标与旧策略编号不同，必须通过 AGENT_INDEX 显式映射。
    当当前 context 中 scared ghost 占多数且 approach 与其它策略并列最大时，
    approach 优先；否则保持原有并列优先级。
    """

    if is_stay is True:
        return strategy_to_number["stay"]
    if is_vague is True or np.sum(weight) == 0:
        return strategy_to_number["vague"]
    try:
        min_value = np.min(weight)
        if min_value < 0:
            weight = weight - min_value
        weight = weight / np.sum(weight)
        weight = list(weight)
        max_value = np.max(weight)
        max_indices = np.where(weight == max_value)[0]
        if len(max_indices) > 1:
            if prefer_approach_when_scared and AGENT_INDEX["approach"] in max_indices:
                return strategy_to_number["approach"]
            if AGENT_INDEX["local"] in max_indices:
                return strategy_to_number["local"]
            if AGENT_INDEX["global"] in max_indices:
                return strategy_to_number["global"]
            if (
                AGENT_INDEX["global"] not in max_indices
                and AGENT_INDEX["local"] not in max_indices
                and AGENT_INDEX["energizer"] not in max_indices
                and AGENT_INDEX["approach"] not in max_indices
                and AGENT_INDEX["no_energizer"] not in max_indices
            ):
                return AGENT_INDEX_TO_STRATEGY_NUMBER[int(max_indices[0])]
            return strategy_to_number["vague"]
        return AGENT_INDEX_TO_STRATEGY_NUMBER[int(max_indices[0])]
    except Exception:
        # 旧脚本只打印上下文后继续返回 index[0]；这里保留可定位的异常信息。
        print("=" * 120)
        print(file_name, trial_name)
        return AGENT_INDEX_TO_STRATEGY_NUMBER[int(max_indices[0])]


def recompute_strategy(data: pd.DataFrame, file_name: str, trial_name: str) -> None:
    """重新根据 revised_normalized_weight/is_stay/is_vague 写入 strategy 列。

    输入语义：data 是单个 trial 的工作表，file_name/trial_name 用于错误定位。
    输出语义：就地更新 data 的 `strategy` 列。
    关键约束：每个规则阶段后都重新计算 strategy。并列权重的基础规则沿用旧脚本；
    只有当一个 context 中 scared ghost 行数超过一半，且 approach 同为最大权重时，
    才把 approach 作为该 context 的优先显示策略。
    """

    approach_priority_mask = build_approach_priority_mask(data)
    data["strategy"] = data[["revised_normalized_weight", "is_stay", "is_vague"]].apply(
        lambda row: strategy_from_weight(
            row.revised_normalized_weight,
            row.is_stay,
            row.is_vague,
            STRATEGY_NUMBER,
            file_name,
            trial_name,
            bool(approach_priority_mask.loc[row.name]),
        ),
        axis=1,
    )


def extract_context_data(data: pd.DataFrame, prev: int, end: int) -> ContextData | None:
    """提取一个规则段落中的有效方向行。

    输入语义：prev/end 是旧 `trial_context` 中的半开区间，按 DataFrame 标签定位。
    输出语义：返回 ContextData；若段落没有可用方向行则返回 None。
    关键约束：无方向行不触发任何写回，保持旧脚本 `continue` 行为。
    """

    segment = copy.deepcopy(data.loc[prev : end - 1])
    temp_data = copy.deepcopy(segment)
    nan_dir = temp_data.action_dir.apply(lambda value: isinstance(value, float))
    valid_data = segment[nan_dir == False]
    if valid_data.shape[0] == 0:
        return None

    valid_indices = np.where(nan_dir == False)[0] + prev
    nan_indices = np.where(nan_dir == True)[0] + prev
    true_prob = valid_data.action_dir.ffill().apply(one_hot_direction)
    return ContextData(
        segment=segment,
        valid_data=valid_data,
        valid_indices=valid_indices,
        nan_indices=nan_indices,
        true_prob=true_prob,
    )


def build_agent_q_values(data: pd.DataFrame, q_columns: list[str]) -> np.ndarray:
    """把每个 agent 的四方向 Q 值整理为三维数组。

    输入语义：data 包含每个 agent 的 Q 列，每个单元格是四方向分数。
    输出语义：返回形状为 `(样本数, 4, agent 数)` 的数组。
    关键约束：列顺序决定 agent 编号，不能改变。
    """

    num_samples = data.shape[0]
    pre_estimation = data[q_columns].values
    agent_q_value = np.zeros((num_samples, 4, len(q_columns)))
    for sample_index in range(num_samples):
        for agent_index in range(len(q_columns)):
            agent_q_value[sample_index, :, agent_index] = pre_estimation[sample_index][agent_index]
    return agent_q_value


def is_uninformative_q(q_values: Any) -> bool:
    """判断一个四方向 Q 向量是否没有提供有效方向偏好。

    输入语义：q_values 是长度为 4 的方向分数，墙方向通常为 ``-inf``，也可能混入 ``nan``。
    输出语义：如果所有可行方向的有限 Q 都是 0，或不存在有限方向，则返回 True。
    关键约束：只有全 0 才表示策略没有给出有效预测；若最大可行方向并列但
    Q 不是全 0，例如 ``[1, 1, -inf, -inf]``，仍应按并列概率折算准确率。
    """

    q_array = np.asarray(q_values, dtype=float)
    finite_values = q_array[np.isfinite(q_array)]
    if len(finite_values) == 0:
        return True
    return bool(np.all(finite_values == 0))


def score_direction_q(q_values: Any, target_direction: int) -> tuple[int, bool, float]:
    """计算单行 Q 对真实方向的预测结果。

    输入语义：q_values 是四方向 Q，target_direction 是真实方向下标。
    输出语义：返回确定性预测方向、是否预测正确，以及并列折算后的准确率贡献。
    关键约束：全 0 无信息 Q 直接视为预测不准确，贡献 0；非零最大值并列仍按
    ``1 / 并列数`` 折算，保留策略确有偏好但无法区分并列方向时的不确定性。
    """

    q_array = np.asarray(q_values, dtype=float).copy()
    q_array[np.isnan(q_array)] = -np.inf
    estimated_direction = choose_max_direction(q_array)

    # 全 0 无信息预测只有“可走方向存在”，没有任何策略奖励或风险偏好，
    # 因此诊断列和 rate 都按错误处理。
    if is_uninformative_q(q_array):
        return estimated_direction, False, 0.0
    if target_direction < 0 or target_direction >= len(q_array) or np.isinf(q_array[target_direction]):
        return estimated_direction, False, 0.0

    max_value = np.max(q_array)
    max_indices = np.where(q_array == max_value)[0]
    if q_array[target_direction] == max_value:
        return estimated_direction, True, float(1 / len(max_indices))
    return estimated_direction, False, 0.0


def score_agent_accuracies(context: ContextData, agent_indices: list[int]) -> tuple[list[float], np.ndarray]:
    """计算指定 agent 在一个段落内的方向预测准确率。

    输入语义：context 是有效方向段落，agent_indices 是要评估的 agent 编号。
    输出语义：返回与 agent_indices 对齐的准确率列表，以及完整 Q 值数组。
    关键约束：有信息的并列最大方向仍按 `1 / 并列数` 计分；若一个样本所有
    可行方向 Q 都是 0，则该样本没有策略信息，直接记为预测不准确。
    """

    true_dir = context.true_prob.apply(choose_max_direction).values
    agent_q_value = build_agent_q_values(context.valid_data, AGENT_Q_COLUMNS)
    agent_accuracy: list[float] = []
    for agent_index in agent_indices:
        accuracy = 0.0
        dir_q_value = agent_q_value[:, :, agent_index]
        for sample_index in range(context.valid_data.shape[0]):
            sample_q = dir_q_value[sample_index]
            target_direction = true_dir[sample_index]
            _, _, credit = score_direction_q(sample_q, target_direction)
            accuracy += credit
        agent_accuracy.append(accuracy / context.valid_data.shape[0])
    return agent_accuracy, agent_q_value


def calculate_prediction_result(
    weight: np.ndarray,
    context: ContextData,
) -> tuple[np.ndarray, np.ndarray, float]:
    """计算一个修正权重在段落内的诊断预测结果。

    输入语义：weight 是 7 维 agent 权重，context 是有效方向段落。
    输出语义：返回 `prediction_correct`、`estimated_dir` 和按并列方向折算的 rate。
    关键约束：estimated_dir 使用确定性并列选择；rate 对有信息的并列方向做折算，
    但无信息 Q 直接按预测不准确处理。
    """

    agent_q_value = build_agent_q_values(context.valid_data, AGENT_Q_COLUMNS)
    # Q 矩阵中可能包含 nan，旧脚本允许它先参与矩阵乘法，再统一转成 -inf。
    # 这里仅屏蔽 numpy 的运行时提示，不改变 nan 的后续处理语义。
    with np.errstate(invalid="ignore"):
        dir_q_value = agent_q_value @ [weight[index] for index in range(len(weight))]
    dir_q_value[np.isnan(dir_q_value)] = -np.inf
    true_dir = np.array([np.argmax(each) for each in context.true_prob])
    prediction_results = [score_direction_q(q_values, int(true_dir[index])) for index, q_values in enumerate(dir_q_value)]
    estimated_dir = np.array([item[0] for item in prediction_results])
    is_correct = np.array([item[1] for item in prediction_results])
    rate = float(np.sum([item[2] for item in prediction_results]))
    rate /= len(dir_q_value)
    return is_correct, estimated_dir, rate


def assign_object_values(data: pd.DataFrame, labels: list[int], column: str, value: Any) -> None:
    """向 object 列按标签写入同一个复杂对象。

    输入语义：labels 是 DataFrame 标签，value 通常是权重列表。
    输出语义：就地写入 data[column]。
    关键约束：用 Series 避免 pandas 把列表权重展开成二维赋值。
    """

    data.loc[labels, column] = pd.Series([deepcopy(value) for _ in labels], index=labels, dtype=object)


def apply_revised_weight(
    data: pd.DataFrame,
    prev: int,
    end: int,
    context: ContextData,
    revise_weight: list[int] | list[float],
    *,
    update_predict_dir: bool,
    strategy_value: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """把一个段落修正规则写回数据表。

    输入语义：prev/end 指定写回区间，context 提供有效方向行，revise_weight 是新权重。
    输出语义：更新 revised_normalized_weight/revised_prediction_correct/predict_dir/is_vague，并返回诊断结果。
    关键约束：如果 update_predict_dir 为 False，则保持旧 `reviseWrongEnergizer` 不写 predict_dir 的行为。
    """

    labels = list(data.loc[prev : end - 1].index)
    assign_object_values(data, labels, "revised_normalized_weight", revise_weight)

    phase_is_correct, estimated_dir, rate = calculate_prediction_result(np.array(revise_weight), context)
    data.loc[list(context.valid_indices), "revised_prediction_correct"] = np.array(phase_is_correct, dtype=int)
    if len(context.nan_indices) > 0:
        data.loc[list(context.nan_indices), "revised_prediction_correct"] = [np.nan] * len(context.nan_indices)
    if update_predict_dir:
        data.loc[list(context.valid_indices), "predict_dir"] = np.array(estimated_dir)
    data.loc[labels, "is_vague"] = [False] * len(labels)
    if strategy_value is not None:
        data.loc[labels, "strategy"] = [strategy_value] * len(labels)
    return phase_is_correct, estimated_dir, rate


def revise_function(
    data: pd.DataFrame,
    contexts: list[tuple[int, int]],
    revise_weight: list[int],
    main_agent: int,
    relative_accuracy_threshold: float = DEFAULT_REVISION_RELATIVE_ACCURACY_THRESHOLD,
    include_relative_threshold: bool = False,
) -> None:
    """按指定主 agent 修正一组段落。

    输入语义：contexts 是半开区间列表，revise_weight 是目标 one-hot 权重；
    relative_accuracy_threshold 控制主策略相对最佳策略的最低比例，布尔参数指定边界
    是否允许等号。
    输出语义：满足准确率阈值的段落会被写回目标权重。
    关键约束：默认仍保留旧规则 ``main/max > 0.8 且 main > 0.6``。只有调用方显式
    覆盖阈值和等号语义时才改变，避免 Approach 调整连带影响 Energizer 等其它规则。
    """

    for prev, end in contexts:
        context = extract_context_data(data, prev, end)
        if context is None:
            continue
        agent_accuracy, _ = score_agent_accuracies(context, list(range(len(AGENTS))))
        main_agent_accuracy = agent_accuracy[main_agent]
        max_accuracy = np.max(agent_accuracy)
        # score_agent_accuracies 已经把全 0 无信息 Q 记为 0 分。若所有策略都是 0 分，
        # 说明当前段落没有任何策略提供有效方向证据，不能仅凭事件标签强行改写。
        if max_accuracy <= 0:
            continue
        relative_accuracy = main_agent_accuracy / max_accuracy
        relative_passed = (
            relative_accuracy >= relative_accuracy_threshold
            if include_relative_threshold
            else relative_accuracy > relative_accuracy_threshold
        )
        if relative_passed and main_agent_accuracy > 0.6:
            apply_revised_weight(data, prev, end, context, revise_weight, update_predict_dir=True)


def tied_best_accuracy_weight(agent_accuracy: list[float] | np.ndarray, *, tolerance: float = 1e-12) -> tuple[list[int], float]:
    """把最高准确率并列策略转换成多热权重。

    输入语义：agent_accuracy 是七个策略在同一 context 上的单策略预测准确率，
    tolerance 用于抵抗浮点误差。
    输出语义：返回 ``(revise_weight, max_accuracy)``，其中所有达到最高准确率的策略
    权重都为 1，其它为 0。
    关键约束：这里不做策略优先级选择；并列证据必须保留到权重里，最终显示策略统一交给
    ``strategy_from_weight`` 的优先级规则决定，避免 revise 阶段用 ``argmax`` 提前丢失并列信息。
    """

    accuracy_array = np.asarray(agent_accuracy, dtype=float)
    if accuracy_array.size == 0:
        return [0] * len(AGENTS), 0.0
    max_accuracy = float(np.max(accuracy_array))
    revise_weight = [
        1 if max_accuracy - float(value) <= tolerance else 0
        for value in accuracy_array
    ]
    return revise_weight, max_accuracy


def revise_vague(data: pd.DataFrame, contexts: list[tuple[int, int]]) -> None:
    """修正旧流程中标记为 vague 的段落。

    输入语义：contexts 来自 `is_vague=True` 的 trial_context。
    输出语义：只有至少一个单独策略能稳定预测真实方向时，才把 vague 段改写成这些策略。
    关键约束：本函数不再相信拟合权重中的唯一最大值。对于 vague 段，必须逐一计算
    七个单独策略的方向预测准确率；最优策略达到保守准确率阈值，且段落内有效动作
    比例足够时，才取消 vague。这里不设置有效动作绝对数量门槛，避免完整的短行为段
    仅因长度小于 4 而永远无法进入并列优先级。若多个策略并列最高，修正权重会同时
    保留这些策略，后续由统一优先级决定显示标签。
    """

    for prev, end in contexts:
        segment = copy.deepcopy(data.loc[prev : end - 1])
        if segment.empty:
            continue

        context = extract_context_data(data, prev, end)
        if context is None:
            continue

        effective_action_ratio = context.valid_data.shape[0] / max(end - prev, 1)
        if effective_action_ratio < LOW_EVIDENCE_MIN_EFFECTIVE_ACTION_RATIO:
            # 这里只限制有效动作在整个 context 中的占比。短 context 若每行都有动作，
            # 仍可进入准确率比较；大量停顿夹杂少数动作的段落继续保留 vague。
            continue

        agent_accuracy, _ = score_agent_accuracies(context, list(range(len(AGENTS))))
        revise_weight, max_accuracy = tied_best_accuracy_weight(agent_accuracy)

        # vague 段只有在最优单策略本身达到保守准确率阈值时才改写。
        # 这里不再要求它明显领先第二名；如果多个策略都能同样解释该段，就把
        # 这些并列最高策略都写入权重，交给统一策略优先级做最终显示。
        if max_accuracy < VAGUE_REVISE_MIN_ACCURACY:
            continue

        apply_revised_weight(data, prev, end, context, revise_weight, update_predict_dir=True)


def revise_energizer_by_outcome(data: pd.DataFrame) -> None:
    """结合单策略准确率和 context 结束边界事件修正 Energizer。

    输入语义：data 是单个 trial 的单玩家临时视图，包含 context、七策略 Q 和玩家
    私有 ``eat_energizer`` 事件。
    输出语义：若结束边界实际吃到 energizer，且 Energizer 单策略准确率不低于 0.70、
    同时达到最佳单策略准确率的 0.80，则整段修正为 Energizer。若结束边界没有吃到，
    仍只在 Energizer 与其它策略精确并列最优时移除 Energizer，再由统一优先级消歧。
    关键约束：成功事件只能确认已有足够行为证据的 Energizer，不能把低准确率策略
    强行覆盖到整段；失败事件也不否定单独解释能力唯一最高的 Energizer。事件定义
    使用半开区间 ``(start, end)`` 的 ``end`` 行，因为06事件列标在资源消失后的到达行。
    """

    contexts = unique_sorted_contexts(data["trial_context"].dropna().to_numpy())
    contexts = filter_contexts_to_trial(data, contexts)
    energizer_index = AGENT_INDEX["energizer"]
    for prev, end in contexts:
        context = extract_context_data(data, prev, end)
        if context is None:
            continue
        agent_accuracy, _ = score_agent_accuracies(context, list(range(len(AGENTS))))
        tied_weight, max_accuracy = tied_best_accuracy_weight(agent_accuracy)
        if max_accuracy <= 0:
            continue

        boundary_eats_energizer = (
            end in data.index
            and bool(data.at[end, "eat_energizer"])
        )
        if boundary_eats_energizer:
            energizer_accuracy = float(agent_accuracy[energizer_index])
            relative_accuracy = energizer_accuracy / float(max_accuracy)
            if (
                energizer_accuracy >= ENERGIZER_OUTCOME_MIN_ACCURACY
                and relative_accuracy >= ENERGIZER_OUTCOME_RELATIVE_ACCURACY_THRESHOLD
            ):
                # 成功吃到 Energizer 只能在绝对准确率与相对准确率同时达标时确认意图。
                # 这允许 Energizer 略低于另一策略，但避免单靠结果事件覆盖弱行为证据。
                revised_weight = [0] * len(AGENTS)
                revised_weight[energizer_index] = 1
                apply_revised_weight(
                    data,
                    prev,
                    end,
                    context,
                    revised_weight,
                    update_predict_dir=True,
                )
            continue

        tied_indices = [index for index, value in enumerate(tied_weight) if value == 1]
        if energizer_index not in tied_indices or len(tied_indices) < 2:
            continue

        # 没有实际吃到时只删除精确并列集合中的 Energizer，其余并列证据完整保留；
        # local/global/approach 等最终显示顺序仍由 strategy_from_weight 统一决定。
        revised_weight = list(tied_weight)
        revised_weight[energizer_index] = 0
        apply_revised_weight(
            data,
            prev,
            end,
            context,
            revised_weight,
            update_predict_dir=True,
        )


def set_weight(data: pd.DataFrame, contexts: list[tuple[int, int]], revise_weight: list[int]) -> None:
    """无额外阈值地把一组段落写成指定权重。

    输入语义：contexts 是需要合并或强制修正的半开区间列表。
    输出语义：每个有有效方向行的段落都会写入 revise_weight。
    关键约束：旧脚本用于合并 scared time 内 approach 段。
    """

    for prev, end in contexts:
        context = extract_context_data(data, prev, end)
        if context is None:
            continue
        apply_revised_weight(data, prev, end, context, revise_weight, update_predict_dir=True)


def unique_sorted_contexts(values: Any) -> list[tuple[int, int]]:
    """把 trial_context 值去重并按起点排序。

    输入语义：values 是若干 `(prev, end)` 元组。
    输出语义：返回稳定排序后的唯一上下文列表。
    关键约束：旧规则依赖段落顺序，必须按左端点排序。
    """

    contexts = list(set(list(values)))
    contexts.sort(key=lambda item: item[0])
    return contexts


def filter_contexts_to_trial(data: pd.DataFrame, contexts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """过滤不属于当前 trial 切片的 context。

    输入语义：data 是当前 trial 的 DataFrame，index 保留全局 row_id 标签；contexts 是全局半开区间。
    输出语义：只返回完整落在当前 trial index 范围内的 context。
    关键约束：06 的 context 使用全局行号；07 按 trial 分块运行规则，因此跨 trial 或
    指向其它 trial 的 context 不能在当前 trial 内修正。
    """

    if data.empty:
        return []
    valid_labels = set(data.index)
    filtered: list[tuple[int, int]] = []
    for prev, end in contexts:
        if prev >= end:
            continue
        if prev in valid_labels and (end - 1) in valid_labels:
            filtered.append((prev, end))
    return filtered


def process_trial(
    data: pd.DataFrame,
    input_path: Path,
    trial_name: str,
    scared_time: int,
) -> pd.DataFrame | None:
    """处理单个 trial 的全部手动规则。

    输入语义：data 是同一 `DayTrial` 的切片，保留原始标签；trial_name 是 trial 名。
    输出语义：返回修正后的 two-ghost trial 数据。
    关键约束：Energizer 必须同时满足准确率和实际事件结果；Approach 的未吃鬼二次
    否定规则不适用于双人任务，因此当前不执行。
    """

    recompute_strategy(data, str(input_path), trial_name)

    vague_index = np.where(data["is_vague"] == True)[0] + data["row_id"].iloc[0]
    vague_contexts = unique_sorted_contexts(np.array(data["trial_context"].loc[vague_index]))
    vague_contexts = filter_contexts_to_trial(data, vague_contexts)
    revise_vague(data, vague_contexts)
    recompute_strategy(data, str(input_path), trial_name)

    # 不执行“当前玩家没有亲自吃到 ghost 就否定 Approach”的二次修正。策略后验已经
    # 根据完整 Q 证据得到 Approach；
    # 双人任务中即使当前玩家没有亲自吃到 ghost，也可能是队友抢先吃掉，不能再把
    # “未亲自吃到”不能作为否定追鬼意图的充分证据。

    eat_energizer = np.where(data["eat_energizer"] == True)[0] - 1 + data["row_id"].iloc[0]
    eat_energizer_context = list(np.array(data["trial_context"].loc[eat_energizer]))
    eat_energizer_context = filter_contexts_to_trial(data, eat_energizer_context)
    revise_energizer_by_outcome(data)
    recompute_strategy(data, str(input_path), trial_name)

    eat_energizer_next = [end for _, end in eat_energizer_context]
    existing_index = list(np.array(data.index))
    eat_energizer_next = [index for index in eat_energizer_next if index in existing_index]
    eat_energizer_next_context = list(np.array(data["trial_context"].loc[eat_energizer_next]))
    eat_energizer_next_context = filter_contexts_to_trial(data, eat_energizer_next_context)
    for prev, end in deepcopy(eat_energizer_next_context):
        is1_values = list(data.loc[prev:end]["ifscared1"])
        is2_values = list(data.loc[prev:end]["ifscared2"])
        if (3 not in is1_values) and (3 not in is2_values):
            if end + 1 < data.iloc[-1]["row_id"] and data.loc[end + 1]["strategy"] != STRATEGY_NUMBER["approach"]:
                eat_energizer_next_context.remove((prev, end))
            elif end + 1 > data.iloc[-1]["row_id"]:
                eat_energizer_next_context.remove((prev, end))

    revise_weight = [0] * len(AGENTS)
    revise_weight[AGENT_INDEX["approach"]] = 1
    # Energizer 后第一段允许 Approach 达到最佳单策略准确率的 75% 即触发，并包含
    # 恰好 0.75 的边界。其它 revise_function 调用仍使用严格大于 0.8 的阈值。
    revise_function(
        data,
        eat_energizer_next_context,
        revise_weight,
        AGENT_INDEX["approach"],
        relative_accuracy_threshold=ENERGIZER_FOLLOWUP_APPROACH_RELATIVE_ACCURACY_THRESHOLD,
        include_relative_threshold=True,
    )
    recompute_strategy(data, str(input_path), trial_name)

    for eat_index_position, eat_index in enumerate(eat_energizer):
        prev = eat_index + 1
        if eat_index_position < len(eat_energizer) - 1:
            end = eat_energizer[eat_index_position + 1] + 1
        else:
            end = data["row_id"].iloc[-1] + 1
        approach_positions = np.where(data.loc[prev : end - 1]["strategy"] == STRATEGY_NUMBER["approach"])[0]
        index_context = list(set(list(data.loc[prev : end - 1]["trial_context"].iloc[approach_positions])))
        index_context.sort(key=lambda item: item[0])
        index_context = filter_contexts_to_trial(data, index_context)
        if len(index_context) <= 1:
            continue

        new_context = [index_context[0]]
        for context in index_context[1:]:
            if context[0] - new_context[-1][0] <= scared_time:
                new_context[-1] = (new_context[-1][0], context[1])
            else:
                new_context.append(context)
        revise_weight = [0] * len(AGENTS)
        revise_weight[AGENT_INDEX["approach"]] = 1
        set_weight(data, new_context, revise_weight)

    recompute_strategy(data, str(input_path), trial_name)

    return data


def revise_player_view(
    player_view: pd.DataFrame,
    input_path: Path,
    player: str,
    scared_time: int,
) -> pd.DataFrame:
    """对一个玩家的临时单人视角执行完整 07 修正。

    输入语义：player_view 已经由阶段入口映射为规则模块需要的通用字段。
    输出语义：返回与原 player_view 行顺序一致的修正后表。
    关键约束：process_trial 内部使用原始行标签和 row_id 定位 context；因此合并所有 trial
    后必须按原 index 排序。Energizer 始终使用准确率与实际结果联合判定。
    """

    required_columns = {"DayTrial", "row_id", "normalized_weight", "prediction_correct", "action_dir"}
    missing_columns = sorted(required_columns - set(player_view.columns))
    if missing_columns:
        raise ValueError(f"{input_path.name} {player} 缺少 revise_human_weight 输入字段：{missing_columns}")
    player_view["revised_normalized_weight"] = copy.deepcopy(np.array(player_view["normalized_weight"]))
    player_view["revised_prediction_correct"] = copy.deepcopy(np.array(player_view["prediction_correct"]))

    trial_name_list = np.unique(player_view.DayTrial.values)
    all_trial_record: list[pd.DataFrame] = []
    for trial_name in trial_name_list:
        trial_data = player_view[player_view.DayTrial == trial_name].copy()
        processed_trial = process_trial(
            trial_data,
            input_path,
            trial_name,
            scared_time,
        )
        if processed_trial is not None:
            all_trial_record.append(copy.deepcopy(processed_trial))

    corrected_data = pd.concat(all_trial_record)
    corrected_data.sort_index(inplace=True)
    if len(corrected_data) != len(player_view):
        raise ValueError(f"{input_path.name} {player} 修正后丢行：input={len(player_view)}, output={len(corrected_data)}")
    return corrected_data
