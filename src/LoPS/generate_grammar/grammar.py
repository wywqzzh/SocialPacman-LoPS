from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from LoPS.generate_grammar.config import GrammarLearningParams
from LoPS.generate_grammar.scoring import bd_score, learn_state_condition_links
from LoPS.generate_grammar.state_graph import StateDependencyGraph
from LoPS.generate_grammar.token import combine_tokens, split_token, token_length, tokens_share_base_token


@dataclass
class OrganizedGrammarData:
    # _organize_discrete_data 的中间结果，字段命名对应旧 organize_data 的返回值。
    # data_child/data_parent 使用 1 表示未出现、2 表示出现；data_condition 保存状态值 + 1。
    data_child: pd.DataFrame
    data_parent: pd.DataFrame
    data_condition: pd.DataFrame
    condition_state: list[list[str]]


@dataclass
class GrammarLearningResult:
    # 核心学习结果全部使用新 token 表示，例如 "G-L"、"E-A"。
    # legacy.py 会负责把这些 token 转回旧输出需要的无分隔符字符串和占位符。
    grammar_tokens: list[str]
    probabilities: list[float]
    position_grammar: list[str]
    original_sequence: list[str]
    time_probabilities: np.ndarray
    frequencies: list[int]
    parsed_sequence: list[str]
    parsed_state_features: pd.DataFrame
    active_tokens: list[str]
    participant_file_names: list[str]
    participant_ids: list[str]
    components: list[list[str]]


@dataclass
class SkipGramResult:
    # 对应旧输出字段 skipGram 和 skipGramNum。
    found: bool
    count: int | float


def static_probability(tokens: Sequence[str], active_tokens: Sequence[str]) -> list[float]:
    # 复刻旧 Tools.static_pro：按 active_tokens 的顺序统计当前解析序列中每个 grammar 的出现概率。
    counts = {}
    for active_token in active_tokens:
        counts.update({active_token: 0})
    for token in tokens:
        counts[token] += 1
    total = np.sum(list(counts.values()))
    return list(np.array(list(counts.values())) / total)


def choose_candidate_chunks(
    ratios: list[float],
    chunks: list[str],
    components: list[list[str]],
    keep_ratio: float,
) -> tuple[list[str], list[float], list[list[str]]]:
    # 复刻旧 choice_max_n：先按 ratio 降序，再只保留 ratio > 1 且接近最大 ratio 的候选。
    ordered_indices = sorted(range(len(ratios)), key=lambda index: ratios[index], reverse=True)
    if len(ordered_indices) == 0:
        return [], [], []

    ordered = [
        (chunks[index], ratios[index], components[index])
        for index in ordered_indices
        if ratios[index] > 1
    ]
    if len(ordered) == 0:
        return [], [], []

    best_ratio = ordered[0][1]
    selected = [ordered[0]]
    for candidate in ordered[1:]:
        # 旧代码阈值是 0.85；这里通过 keep_ratio 暴露出来，但默认值仍保持旧行为。
        if candidate[1] / best_ratio > keep_ratio:
            selected.append(candidate)
        else:
            break

    selected_chunks, selected_ratios, selected_components = zip(*selected)
    return list(selected_chunks), list(selected_ratios), list(selected_components)


def kl_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    # 复刻旧 Tools.KL：若参考分布缺少某个 key，用 0.00001 作为平滑值。
    value = 0
    for key in p.keys():
        probability = p[key]
        if key in q:
            reference_probability = q[key]
        else:
            reference_probability = 0.00001
        value += probability * math.log2(probability / reference_probability)
    return value


class GrammarLearner:
    def __init__(self, params: GrammarLearningParams):
        # GrammarLearner 只持有算法参数，不持有路径，也不读写文件。
        self.params = params

    def _parse_longest(
        self,
        tokens: list[str],
        grammar_tokens: list[str],
        state_features: pd.DataFrame | None = None,
    ) -> tuple[list[str], pd.DataFrame | None]:
        # 对原始基础 token 序列做最长匹配解析。grammar_tokens 的顺序必须保持学习追加顺序；
        # 当多个 token 都能匹配当前位置时，选择基础 token 长度最长的那个，等价于旧 parse()。
        parsed_tokens = []
        parsed_state_rows = []
        pointer = 0
        while pointer < len(tokens):
            matched_index = 0
            matched_length = 0
            for index, grammar_token in enumerate(grammar_tokens):
                # token_length 计算基础 token 个数，不能使用字符串长度；"G-L" 的长度是 2。
                length = token_length(grammar_token)
                if tokens[pointer:pointer + length] == split_token(grammar_token) and length > matched_length:
                    matched_length = length
                    matched_index = index
            if matched_length == 0:
                raise ValueError(f"No grammar token matches sequence position {pointer}: {tokens[pointer:]}")

            parsed_tokens.append(grammar_tokens[matched_index])
            if state_features is not None:
                # 旧 parse() 会把一个 chunk 对齐到该片段首个基础 token 的状态行。
                parsed_state_rows.append(list(state_features.iloc[pointer]))
            pointer += matched_length

        if state_features is None:
            return parsed_tokens, None
        parsed_state_features = pd.DataFrame(parsed_state_rows, columns=state_features.columns)
        return parsed_tokens, parsed_state_features

    def _parse_probabilities(
        self,
        tokens: list[str],
        grammar_tokens: list[str],
    ) -> tuple[list[str], list[float], list[str], list[int]]:
        # 复刻旧 parse_pro：用 grammar_tokens 对原始序列重新解析，并统计每个 grammar 的频数和概率。
        cover_indices = []
        pointer = 0
        position_grammar = []
        # 旧代码这里使用最后一次循环的 L 值填充 position_gram；为保持 legacy 输出一致，保留该行为。
        last_grammar_length = token_length(grammar_tokens[-1])

        while pointer < len(tokens):
            matched_index = 0
            matched_length = 0
            for index, grammar_token in enumerate(grammar_tokens):
                length = token_length(grammar_token)
                if tokens[pointer:pointer + length] == split_token(grammar_token) and length > matched_length:
                    matched_length = length
                    matched_index = index
            if matched_length == 0:
                raise ValueError(f"No grammar token matches sequence position {pointer}: {tokens[pointer:]}")

            cover_indices.append(matched_index)
            pointer += matched_length
            position_grammar += [grammar_tokens[matched_index]] * last_grammar_length

        # frequencies_by_token 必须按 grammar_tokens 顺序初始化，保证输出顺序与旧 S/sets 对齐。
        frequencies_by_token = {}
        for grammar_token in grammar_tokens:
            frequencies_by_token.update({grammar_token: 0})
        for index in cover_indices:
            frequencies_by_token[grammar_tokens[index]] += 1

        frequencies = np.array(list(frequencies_by_token.values()))
        probabilities = frequencies / np.sum(frequencies)
        return list(grammar_tokens), list(probabilities), position_grammar, list(frequencies)

    def _organize_discrete_data(
        self,
        tokens: list[str],
        active_tokens: list[str],
        state_features: pd.DataFrame,
        state_dependencies: StateDependencyGraph,
    ) -> OrganizedGrammarData:
        # 复刻旧 organize_data：把解析后的 token 序列转为离散 parent/child/condition 矩阵。
        # 所有状态都使用 1/2 或 state+1 编码，因为旧 BDscore/count 假定状态从 1 开始。
        state_features = state_features.reset_index(drop=True)
        data_parent = {}
        data_child = {}
        for token in active_tokens:
            data_parent.update({token: np.ones(len(tokens) - 1)})
            data_child.update({token: np.ones(len(tokens) - 1)})

        data_condition = {}
        data_policy_condition = {}
        for state_name in state_features.columns:
            data_condition.update({state_name: np.ones(len(tokens) - 1)})
            data_policy_condition.update({state_name: np.ones(len(tokens) - 1)})

        for index in range(1, len(tokens)):
            # parent 使用上一个时间点的 token，child 使用当前时间点的 token。
            data_parent[tokens[index - 1]][index - 1] = 2
            data_child[tokens[index]][index - 1] = 2
            for state_name in state_features.columns:
                # data_condition 对齐 child 时刻，data_policy_condition 对齐 parent 时刻。
                data_condition[state_name][index - 1] = state_features[state_name].iloc[index] + 1
                data_policy_condition[state_name][index - 1] = state_features[state_name].iloc[index - 1] + 1

        data_parent_frame = pd.DataFrame(data_parent, dtype=int)
        data_child_frame = pd.DataFrame(data_child, dtype=int)
        data_condition_frame = pd.DataFrame(data_condition, dtype=int)
        data_policy_condition_frame = pd.DataFrame(data_policy_condition, dtype=int)

        # learn_state_condition_links 需要的 data 结构与旧 organize_data 完全一致：
        # 前半部分是状态条件变量，后半部分是 grammar parent 变量。
        data = pd.concat([data_policy_condition_frame, data_parent_frame], axis=1).values.T
        data = np.array(data, dtype=int)
        nstates = np.max(data, axis=1).T
        nstates = np.array(nstates, dtype=int)
        casual_num = data_policy_condition_frame.shape[1]
        effect_num = data_parent_frame.shape[1]
        block_message = {index: [index] for index in range(casual_num)}

        # 通过状态图学习每个 grammar token 需要附加哪些状态条件。
        learned_adjacency, _, _, _ = learn_state_condition_links(
            data=data,
            nstates=nstates,
            block_message=block_message,
            casual_num=casual_num,
            block_num=len(block_message),
            effect_num=effect_num,
            alpha=self.params.condition_alpha,
            conditions=state_dependencies.conditions_by_state,
        )
        condition_state = []
        names = np.array(list(data_condition_frame.columns))
        for index in range(casual_num, casual_num + effect_num):
            # 旧代码按 Alearn[:, i] == 1 找到状态列名；这里保持相同映射。
            condition_indices = np.where(learned_adjacency[:, index] == 1)[0]
            condition_state.append(list(names[condition_indices]))

        return OrganizedGrammarData(
            data_child=data_child_frame,
            data_parent=data_parent_frame,
            data_condition=data_condition_frame,
            condition_state=condition_state,
        )

    def learn(
        self,
        token_sequence: list[str],
        initial_tokens: list[str],
        state_features: pd.DataFrame,
        state_dependencies: StateDependencyGraph,
        participant_file_names: list[str],
        participant_ids: list[str],
    ) -> GrammarLearningResult:
        # original_sequence 是删除 N 后的基础 token 序列；后续每轮都重新从它做最长匹配，
        # 这与旧 Chunking 中 sequence 保持原始字符串、seq 表示当前解析结果的设计一致。
        original_sequence = list(token_sequence)
        active_tokens = list(initial_tokens)
        parsed_sequence = list(original_sequence)
        parsed_state_features = state_features.reset_index(drop=True).copy()
        probabilities = static_probability(parsed_sequence, active_tokens)
        components = [[token, ""] for token in active_tokens]

        predict_tokens, predict_probabilities, _, _ = self._parse_probabilities(original_sequence, active_tokens)
        previous_distribution = {
            token: predict_probabilities[index]
            for index, token in enumerate(predict_tokens)
        }
        kl_history = []

        for _ in range(self.params.max_iterations):
            # 每轮根据当前解析序列重新组织离散数据，再评估哪些 parent->child 组合值得合并。
            organized = self._organize_discrete_data(
                parsed_sequence,
                active_tokens,
                parsed_state_features,
                state_dependencies,
            )
            ratios = []
            chunks = []
            candidate_components = []

            for child_index, child_token in enumerate(active_tokens):
                # 旧代码排除 V、1、2、N、S、e 作为 child；这些 token 不参与 chunk 合并目标。
                if child_token in self.params.excluded_child_tokens:
                    continue

                data_child = organized.data_child[child_token].values
                nstates_child = int(np.max(data_child).T)
                condition_names = organized.condition_state[child_index]
                if len(condition_names) != 0:
                    # 如果状态图认为该 child 受状态条件影响，BDscore 要把这些状态作为 parent 条件。
                    data_condition = organized.data_condition[condition_names].values.T
                    nstates_condition = np.array(np.max(data_condition, 1).T, dtype=int)
                else:
                    data_condition = []
                    nstates_condition = []

                score_alpha = 1 if self.params.chunk_alpha < 0 else self.params.chunk_alpha
                # score_without_parent 是只考虑状态条件、不考虑候选 grammar parent 的得分。
                score_without_parent, _ = bd_score(
                    data_child,
                    data_condition,
                    nstates_child,
                    nstates_condition,
                    score_alpha,
                )

                for parent_index, parent_token in enumerate(active_tokens):
                    # 旧代码跳过 parent==child、V、N 以及基础 token 有交集的候选。
                    if parent_token == child_token or parent_token in self.params.excluded_parent_tokens:
                        continue
                    if self.params.reject_shared_base_tokens and tokens_share_base_token(parent_token, child_token):
                        continue

                    data_parent = organized.data_parent[parent_token].values.reshape(1, -1)
                    nstates_parent = int(np.max(data_parent).T)
                    if len(condition_names) != 0:
                        # 有状态条件时，候选 parent 和状态条件一起作为 BDscore 的 parent 集。
                        parent_and_condition_data = np.vstack((data_parent, data_condition))
                        parent_and_condition_data = np.array(parent_and_condition_data, dtype=int)
                        nstates_parent_and_condition = np.array(np.max(parent_and_condition_data, 1).T, dtype=int)
                    else:
                        parent_and_condition_data = np.array(data_parent, dtype=int)
                        nstates_parent_and_condition = nstates_parent

                    # score_with_parent 是加入候选 grammar parent 后的得分；旧算法用二者比值做候选 ratio。
                    score_with_parent, _ = bd_score(
                        data_child,
                        parent_and_condition_data,
                        nstates_child,
                        nstates_parent_and_condition,
                        score_alpha,
                    )
                    _, pair_posterior = bd_score(data_child, data_parent, 2, 2, 1)
                    pair_frequency = pair_posterior[1, 1] / len(parsed_sequence)
                    # 旧代码要求 parent-child 同现频率同时高于独立概率乘积和最小频率阈值。
                    if (
                        pair_frequency < probabilities[child_index] * probabilities[parent_index]
                        or pair_frequency < self.params.min_pair_frequency
                    ):
                        continue

                    ratios.append(score_without_parent / score_with_parent)
                    chunks.append(combine_tokens(parent_token, child_token))
                    candidate_components.append([parent_token, child_token])

            if len(ratios) == 0:
                break

            # 一轮可能选出多个接近最佳 ratio 的 chunk；它们按旧 choice_max_n 顺序追加到 active_tokens。
            selected_chunks, _, selected_components = choose_candidate_chunks(
                ratios,
                chunks,
                candidate_components,
                self.params.candidate_ratio_keep,
            )
            if len(selected_chunks) == 0:
                break

            added_any = False
            for index, chunk in enumerate(selected_chunks):
                # 新实现避免重复加入相同 token；正常旧流程不会重复，但这里防止无意义循环。
                if chunk in active_tokens:
                    continue
                active_tokens.append(chunk)
                components.append(list(selected_components[index]))
                added_any = True
            if not added_any:
                break

            # 每次加入新 chunk 后，都从 original_sequence 重新做最长匹配，更新解析序列和对齐状态。
            parsed_sequence, parsed_state_features_or_none = self._parse_longest(
                original_sequence,
                active_tokens,
                state_features,
            )
            if parsed_state_features_or_none is None:
                raise ValueError("state_features must be provided for grammar learning")
            parsed_state_features = parsed_state_features_or_none
            probabilities = static_probability(parsed_sequence, active_tokens)

            # 使用解析概率分布的 KL 均值作为收敛条件，保持旧代码最近 5 次均值 <= 0.05 的语义。
            predict_tokens, predict_probabilities, _, _ = self._parse_probabilities(original_sequence, active_tokens)
            current_distribution = {
                token: predict_probabilities[index]
                for index, token in enumerate(predict_tokens)
                if predict_probabilities[index] != 0
            }
            kl_history.append(kl_divergence(current_distribution, previous_distribution))
            previous_distribution = dict(current_distribution)
            if (
                len(kl_history) >= self.params.convergence_window
                and np.mean(kl_history[-self.params.convergence_window:]) <= self.params.convergence_kl_threshold
            ):
                break

        # 循环结束后，按最终 active_tokens 重新统计 sets/pro/gram/frequency。
        grammar_tokens, probabilities, position_grammar, frequencies = self._parse_probabilities(
            original_sequence,
            active_tokens,
        )
        # 旧代码会删除概率为 0 的 grammar，以保证输出只包含实际出现过的项。
        nonzero_indices = np.where(np.array(probabilities) != 0)[0]
        grammar_tokens = [grammar_tokens[index] for index in nonzero_indices]
        probabilities = [probabilities[index] for index in nonzero_indices]
        frequencies = [frequencies[index] for index in nonzero_indices]
        active_tokens = [active_tokens[index] for index in nonzero_indices]
        components = [components[index] for index in nonzero_indices]

        weighted_frequencies = np.array(frequencies, dtype=float)
        for index, grammar_token in enumerate(grammar_tokens):
            # time_pro 统计每个 grammar 覆盖的基础 token 数占比，因此频数要乘 token_length。
            weighted_frequencies[index] *= token_length(grammar_token)
        time_probabilities = weighted_frequencies / np.sum(weighted_frequencies)

        return GrammarLearningResult(
            grammar_tokens=grammar_tokens,
            probabilities=probabilities,
            position_grammar=position_grammar,
            original_sequence=original_sequence,
            time_probabilities=time_probabilities,
            frequencies=frequencies,
            parsed_sequence=parsed_sequence,
            parsed_state_features=parsed_state_features,
            active_tokens=active_tokens,
            participant_file_names=participant_file_names,
            participant_ids=participant_ids,
            components=components,
        )

    def detect_skip_gram(
        self,
        result: GrammarLearningResult,
        n_positions: np.ndarray,
    ) -> SkipGramResult:
        # skip-gram 检测要把之前删除的 N 插回当前解析序列中，再判断 N 后第 2 到第 5 个 token 是否为 E-A。
        parsed_sequence = result.parsed_sequence
        position_sum = -1
        n_pointer = 0
        sequence_with_n = []
        for token in parsed_sequence:
            # position_sum 以基础 token 数推进，用于把原始 N 位置映射回解析后的 chunk 序列。
            position_sum += token_length(token)
            sequence_with_n.append(token)
            if n_pointer < len(n_positions) and position_sum >= n_positions[n_pointer]:
                sequence_with_n.append(self.params.removed_token)
                position_sum += 1
                n_pointer += 1

        n_parent = np.array([1] * len(sequence_with_n))
        target_child = np.array([1] * len(sequence_with_n))
        for index, token in enumerate(sequence_with_n):
            if token != self.params.removed_token:
                continue
            n_parent[index] = 2
            for next_index in range(
                index + self.params.skip_gram_min_offset,
                min(index + self.params.skip_gram_max_offset + 1, len(sequence_with_n)),
            ):
                # 旧逻辑跳过 N，只要窗口内命中目标 EA；新核心目标 token 是 "E-A"。
                if sequence_with_n[next_index] != self.params.removed_token and (
                    sequence_with_n[next_index] == self.params.skip_gram_target
                ):
                    target_child[index] = 2
                    break

        # BDscore 输入仍使用 1/2 编码，和旧 skip_gram 中 N、EA 两个二值变量一致。
        target_child = target_child.reshape(-1, 1).T
        n_parent = n_parent.reshape(-1, 1).T
        target_states = int(np.max(target_child).T)
        n_states = int(np.max(n_parent).T)

        # score_without_parent/score_with_parent 的比值和 U[1,1] 频率阈值共同决定 skipGram。
        score_without_parent, _ = bd_score(
            target_child.reshape(-1, 1),
            [],
            target_states,
            [],
            self.params.skip_gram_alpha,
        )
        score_with_parent, posterior = bd_score(
            target_child,
            n_parent,
            target_states,
            [n_states],
            self.params.skip_gram_alpha,
        )
        if (
            score_without_parent / score_with_parent > 1
            and posterior[1, 1] / len(sequence_with_n) > self.params.skip_gram_min_frequency
        ):
            return SkipGramResult(True, posterior[1, 1])
        return SkipGramResult(False, 0)
