#!/usr/bin/env python3
"""按旧规则修正人类策略权重数据。"""

from __future__ import annotations

import argparse
import copy
import os
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from itertools import groupby
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


def project_root() -> Path:
    """返回 LoPS 仓库根目录。

    输入语义：无输入，通过脚本位置推导根目录。
    输出语义：返回用于构造默认数据目录的 Path。
    关键约束：默认路径只存在于脚本层，不写入正式业务逻辑。
    """

    return Path(__file__).resolve().parents[2]


def list_pickle_files(data_dir: Path) -> list[Path]:
    """列出扁平目录中的 pickle 文件。

    输入语义：data_dir 是 WeightData 输入目录。
    输出语义：返回按文件名排序的 `.pkl` 路径。
    关键约束：目录不存在或无输入时直接抛错，避免静默生成空结果。
    """

    if not data_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{data_dir}")
    file_paths = sorted(data_dir.glob("*.pkl"))
    if not file_paths:
        raise FileNotFoundError(f"输入目录中没有 pickle 文件：{data_dir}")
    return file_paths


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


def strategy_from_weight(
    weight: Any,
    is_stay: bool,
    is_vague: bool,
    strategy_to_number: dict[str, int],
    file_name: str,
    trial_name: str,
) -> int:
    """根据权重向量和人工标记得到策略编号。

    输入语义：weight 是 7 个 agent 权重，is_stay/is_vague 是旧流程标记。
    输出语义：返回旧策略编号。
    关键约束：权重向量下标与旧策略编号不同，必须通过 AGENT_INDEX 显式映射。
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
    关键约束：每个规则阶段后都按旧脚本重新计算 strategy。
    """

    data["strategy"] = data[["revised_normalized_weight", "is_stay", "is_vague"]].apply(
        lambda row: strategy_from_weight(
            row.revised_normalized_weight,
            row.is_stay,
            row.is_vague,
            STRATEGY_NUMBER,
            file_name,
            trial_name,
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


def score_agent_accuracies(context: ContextData, agent_indices: list[int]) -> tuple[list[float], np.ndarray]:
    """计算指定 agent 在一个段落内的方向预测准确率。

    输入语义：context 是有效方向段落，agent_indices 是要评估的 agent 编号。
    输出语义：返回与 agent_indices 对齐的准确率列表，以及完整 Q 值数组。
    关键约束：准确率公式保留旧脚本对并列最大方向按 `1 / 并列数` 计分的规则。
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
            if np.isinf(sample_q[target_direction]):
                accuracy += 0
            else:
                max_value = np.max(sample_q)
                max_indices = np.where(sample_q == max_value)[0]
                if sample_q[target_direction] == max_value:
                    accuracy += 1 / len(max_indices)
        agent_accuracy.append(accuracy / context.valid_data.shape[0])
    return agent_accuracy, agent_q_value


def calculate_prediction_result(
    weight: np.ndarray,
    context: ContextData,
) -> tuple[np.ndarray, np.ndarray, float]:
    """计算一个修正权重在段落内的诊断预测结果。

    输入语义：weight 是 7 维 agent 权重，context 是有效方向段落。
    输出语义：返回 `prediction_correct`、`estimated_dir` 和按并列方向折算的 rate。
    关键约束：estimated_dir 使用确定性并列选择；rate 保留旧脚本的并列折算逻辑。
    """

    agent_q_value = build_agent_q_values(context.valid_data, AGENT_Q_COLUMNS)
    # Q 矩阵中可能包含 nan，旧脚本允许它先参与矩阵乘法，再统一转成 -inf。
    # 这里仅屏蔽 numpy 的运行时提示，不改变 nan 的后续处理语义。
    with np.errstate(invalid="ignore"):
        dir_q_value = agent_q_value @ [weight[index] for index in range(len(weight))]
    dir_q_value[np.isnan(dir_q_value)] = -np.inf
    true_dir = np.array([np.argmax(each) for each in context.true_prob])
    estimated_dir = np.array([choose_max_direction(each) for each in dir_q_value])
    is_correct = estimated_dir == true_dir

    rate = 0.0
    for index, q_values in enumerate(dir_q_value):
        max_indices = np.where(q_values == np.max(q_values))[0]
        if true_dir[index] in list(max_indices):
            rate += 1 / len(max_indices)
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
) -> None:
    """按指定主 agent 修正一组段落。

    输入语义：contexts 是半开区间列表，revise_weight 是目标 one-hot 权重。
    输出语义：满足准确率阈值的段落会被写回目标权重。
    关键约束：阈值 `main/max > 0.8 且 main > 0.6` 保留旧规则。
    """

    for prev, end in contexts:
        context = extract_context_data(data, prev, end)
        if context is None:
            continue
        agent_accuracy, _ = score_agent_accuracies(context, list(range(len(AGENTS))))
        main_agent_accuracy = agent_accuracy[main_agent]
        max_accuracy = np.max(agent_accuracy)
        if main_agent_accuracy / max_accuracy > 0.8 and main_agent_accuracy > 0.6:
            apply_revised_weight(data, prev, end, context, revise_weight, update_predict_dir=True)


def revise_vague(data: pd.DataFrame, contexts: list[tuple[int, int]]) -> None:
    """修正旧流程中标记为 vague 的段落。

    输入语义：contexts 来自 `is_vague=True` 的 trial_context。
    输出语义：能明确归属单一策略的 vague 段会被改写权重和 is_vague。
    关键约束：只有最大权重并列时才进入准确率比较；单一最大值直接取消 vague。
    """

    for prev, end in contexts:
        segment = copy.deepcopy(data.loc[prev : end - 1])
        weight = segment["revised_normalized_weight"].iloc[0]
        if np.sum(weight) <= 0:
            continue

        if np.max(weight) < 1:
            # revised_normalized_weight 来自旧内部 9 维 normalized_weight 的投影；若投影后最大值小于 1，
            # 说明旧 3/4 鬼占位 agent 曾是唯一最大值。旧流程在 two-ghost 数据中会
            # 跳过这类 vague 段，不把它改写成可见策略，因此这里保留 vague 状态。
            continue

        max_indices = np.where(weight == np.max(weight))[0]
        if len(max_indices) == 1:
            labels = list(data.loc[prev : end - 1].index)
            data.loc[labels, "is_vague"] = [False] * len(labels)
            continue

        context = extract_context_data(data, prev, end)
        if context is None:
            continue
        agent_accuracy, agent_q_value = score_agent_accuracies(context, list(max_indices))

        max_accuracy_index = -1
        max_accuracy = -1.0
        for local_index, agent_index in enumerate(max_indices):
            if agent_accuracy[local_index] >= max_accuracy:
                max_accuracy_index = int(agent_index)
                max_accuracy = agent_accuracy[local_index]

        revise_weight = [0] * len(AGENTS)
        revise_weight[max_accuracy_index] = 1
        available_direction_count = 4 - np.sum(np.isinf(agent_q_value[0, :, 0]))
        if max_accuracy > 1 / available_direction_count:
            apply_revised_weight(data, prev, end, context, revise_weight, update_predict_dir=True)


def revise_approach(data: pd.DataFrame, contexts: list[tuple[int, int]]) -> None:
    """修正未实际吃到 ghost 的 approach 段落。

    输入语义：contexts 是当前被判定为 approach 且排除吃 ghost 后的段落。
    输出语义：如果其它策略预测表现接近或优于 approach，则改写为其它策略。
    关键约束：保留旧规则 `accuracyApproach == 0 或 max/approach > 0.8`。
    """

    for prev, end in contexts:
        context = extract_context_data(data, prev, end)
        if context is None:
            continue
        agent_accuracy, _ = score_agent_accuracies(context, list(range(len(AGENTS))))
        accuracy_approach = agent_accuracy[AGENT_INDEX["approach"]]
        agent_accuracy[AGENT_INDEX["approach"]] = 0
        max_index = int(np.argmax(agent_accuracy))
        if accuracy_approach == 0 or agent_accuracy[max_index] / accuracy_approach > 0.8:
            revise_weight = [0] * len(AGENTS)
            revise_weight[max_index] = 1
            apply_revised_weight(data, prev, end, context, revise_weight, update_predict_dir=True)


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


def revise_wrong_energizer(data: pd.DataFrame, energizer_contexts: list[tuple[int, int]]) -> None:
    """修正被误标为 energizer 的 local 段落。

    输入语义：energizer_contexts 是连续 energizer 标签区间，区间右端是闭区间。
    输出语义：满足后继 local 且准确率不过度下降的段落被改为 local。
    关键约束：该旧规则不写回 predict_dir，只更新 revised_prediction_correct 和 strategy。
    """

    for context_range in energizer_contexts:
        prev = context_range[0]
        end = context_range[1] + 1
        context = extract_context_data(data, prev, end)
        if context is None:
            continue

        temp_index = context_range[1] + 1
        if temp_index > data["row_id"].iloc[-1]:
            continue
        if data["strategy"].loc[temp_index] != STRATEGY_NUMBER["approach"] and data["strategy"].loc[temp_index] == STRATEGY_NUMBER["local"]:
            original_weight = deepcopy(data["revised_normalized_weight"].loc[prev])
            revise_weight = [0] * len(AGENTS)
            revise_weight[AGENT_INDEX["local"]] = 1

            # 旧脚本在判断失败前已经把整段临时写成 local；若准确率下降过多，
            # 再用连续 energizer 段第一行的原始权重回滚整段。这个“整段回滚”
            # 会传播第一行权重，属于旧输出的一部分，需要显式保留。
            labels = list(data.loc[prev : end - 1].index)
            assign_object_values(data, labels, "revised_normalized_weight", revise_weight)

            phase_is_correct, _, rate = calculate_prediction_result(np.array(revise_weight), context)
            _, _, original_rate = calculate_prediction_result(np.array(original_weight), context)
            if rate / original_rate < 0.8:
                assign_object_values(data, labels, "revised_normalized_weight", original_weight)
                continue

            data.loc[labels, "strategy"] = [STRATEGY_NUMBER["local"]] * len(labels)
            data.loc[list(context.valid_indices), "revised_prediction_correct"] = np.array(phase_is_correct, dtype=int)
            if len(context.nan_indices) > 0:
                data.loc[list(context.nan_indices), "revised_prediction_correct"] = [np.nan] * len(context.nan_indices)
            data.loc[labels, "is_vague"] = [False] * len(labels)


def unique_sorted_contexts(values: Any) -> list[tuple[int, int]]:
    """把 trial_context 值去重并按起点排序。

    输入语义：values 是若干 `(prev, end)` 元组。
    输出语义：返回稳定排序后的唯一上下文列表。
    关键约束：旧规则依赖段落顺序，必须按左端点排序。
    """

    contexts = list(set(list(values)))
    contexts.sort(key=lambda item: item[0])
    return contexts


def process_trial(data: pd.DataFrame, input_path: Path, trial_name: str, scared_time: int) -> pd.DataFrame | None:
    """处理单个 trial 的全部手动规则。

    输入语义：data 是同一 `DayTrial` 的切片，保留原始标签；trial_name 是 trial 名。
    输出语义：返回修正后的 two-ghost trial 数据。
    关键约束：规则执行顺序必须与旧 `reviseMain` 一致。
    """

    recompute_strategy(data, str(input_path), trial_name)

    vague_index = np.where(data["is_vague"] == True)[0] + data["row_id"].iloc[0]
    vague_contexts = unique_sorted_contexts(np.array(data["trial_context"].loc[vague_index]))
    revise_vague(data, vague_contexts)
    recompute_strategy(data, str(input_path), trial_name)

    context_approach = np.where(data["strategy"] == STRATEGY_NUMBER["approach"])[0]
    context_approach = list(set(list(data["trial_context"].iloc[context_approach])))
    eat_ghost = np.where(data["eat_ghost"] == True)[0] - 1 + data["row_id"].iloc[0]
    for prev, end in deepcopy(context_approach):
        is_eat_ghost = [1 if prev <= eat_index < end else 0 for eat_index in eat_ghost]
        if 1 in is_eat_ghost:
            context_approach.remove((prev, end))
    revise_approach(data, context_approach)
    recompute_strategy(data, str(input_path), trial_name)

    eat_energizer = np.where(data["eat_energizer"] == True)[0] - 1 + data["row_id"].iloc[0]
    eat_energizer_context = list(np.array(data["trial_context"].loc[eat_energizer]))
    revise_weight = [0] * len(AGENTS)
    revise_weight[AGENT_INDEX["energizer"]] = 1
    revise_function(data, eat_energizer_context, revise_weight, AGENT_INDEX["energizer"])
    recompute_strategy(data, str(input_path), trial_name)

    eat_energizer_next = [end for _, end in eat_energizer_context]
    existing_index = list(np.array(data.index))
    eat_energizer_next = [index for index in eat_energizer_next if index in existing_index]
    eat_energizer_next_context = list(np.array(data["trial_context"].loc[eat_energizer_next]))
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
    revise_function(data, eat_energizer_next_context, revise_weight, AGENT_INDEX["approach"])
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

    energizer_index = np.where(data["strategy"] == STRATEGY_NUMBER["energizer"])[0] + data["row_id"].iloc[0]
    groups = groupby(enumerate(energizer_index), lambda index_value: index_value[0] - index_value[1])
    energizer_context = [(group_items[0][1], group_items[-1][1]) for _, group_items in ((key, list(group)) for key, group in groups)]
    revise_wrong_energizer(data, energizer_context)
    # 旧脚本在该阶段后直接收集 trial 数据，没有再对收集结果重算 strategy。
    # 失败分支保持原 strategy，成功分支由 revise_wrong_energizer 直接写成 local。
    return data


def process_one_file(input_path: Path, output_dir: Path, scared_time: int = 63) -> dict[str, Any]:
    """处理一个 WeightData 文件并保存 CorrectedWeightData。

    输入语义：input_path 指向旧 WeightData pickle，output_dir 是扁平输出目录。
    输出语义：写出同名 corrected weight pickle，并返回摘要。
    关键约束：当前输入已经是 two-ghost trial；输出索引 reset，与旧脚本一致。
    """

    df = pd.read_pickle(input_path)
    required_columns = {"DayTrial", "row_id", "normalized_weight", "prediction_correct", "action_dir"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"{input_path.name} 缺少 revise_human_weight 输入字段：{missing_columns}")
    df["revised_normalized_weight"] = copy.deepcopy(np.array(df["normalized_weight"]))
    df["revised_prediction_correct"] = copy.deepcopy(np.array(df["prediction_correct"]))

    trial_name_list = np.unique(df.DayTrial.values)
    all_trial_record: list[pd.DataFrame] = []
    for trial_name in trial_name_list:
        trial_data = df[df.DayTrial == trial_name].copy()
        processed_trial = process_trial(trial_data, input_path, trial_name, scared_time)
        if processed_trial is not None:
            all_trial_record.append(copy.deepcopy(processed_trial))

    corrected_data = pd.concat(all_trial_record)
    corrected_data.reset_index(inplace=True, drop=True)
    output_path = output_dir / input_path.name
    corrected_data.to_pickle(output_path)

    return {
        "input_file": input_path.name,
        "output_file": output_path.name,
        "input_rows": int(len(df)),
        "output_rows": int(len(corrected_data)),
        "mean_revised_prediction_correct": float(np.nanmean(corrected_data["revised_prediction_correct"])),
    }


def process_revise_human_weight(
    input_dir: Path,
    output_dir: Path,
    *,
    processes: int,
    scared_time: int = 63,
) -> list[dict[str, Any]]:
    """批量执行人类权重修正流程。

    输入语义：input_dir/output_dir 都是扁平目录，processes 控制并行度。
    输出语义：返回所有文件的处理摘要。
    关键约束：规则无跨文件依赖，因此文件级并行不会改变结果。
    """

    input_paths = list_pickle_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    worker = partial(process_one_file, output_dir=output_dir, scared_time=scared_time)

    if processes <= 1:
        return [worker(input_path) for input_path in input_paths]

    process_count = min(processes, len(input_paths))
    import multiprocessing

    with multiprocessing.Pool(processes=process_count) as pool:
        return pool.map(worker, input_paths)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输入语义：允许覆盖输入、输出、并行度和 scared_time。
    输出语义：返回可驱动批处理的参数对象。
    关键约束：默认路径全部位于 LoPS 仓库内，不依赖旧项目。
    """

    data_root = project_root() / "pipeline_data"
    parser = argparse.ArgumentParser(description="按旧规则修正人类策略权重数据。")
    parser.add_argument("--input-dir", type=Path, default=data_root / "dynamic_strategy_fitting" / "weight_data")
    parser.add_argument("--output-dir", type=Path, default=data_root / "revise_human_weight" / "corrected_weight_data")
    parser.add_argument("--processes", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--scared-time", type=int, default=63)
    return parser.parse_args()


def main() -> None:
    """命令行入口：运行人类权重修正并打印摘要。"""

    args = parse_args()
    summaries = process_revise_human_weight(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        processes=args.processes,
        scared_time=args.scared_time,
    )
    print(
        "revise_human_weight 完成 "
        f"input_files={len(summaries)} "
        f"input_rows={sum(item['input_rows'] for item in summaries)} "
        f"output_rows={sum(item['output_rows'] for item in summaries)} "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
