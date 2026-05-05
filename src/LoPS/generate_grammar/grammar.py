"""语法学习核心算法。

本模块只处理内存中的 token 序列、状态特征和概率评分，不负责文件读写或验证格式转换。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from LoPS.generate_grammar.config import GrammarLearningParams
from LoPS.generate_grammar.scoring import bd_score, learn_state_condition_links
from LoPS.generate_grammar.state_graph import StateDependencyGraph
from LoPS.generate_grammar.token import (
    GrammarToken,
    combine_tokens,
    format_grammar_token,
    parse_token_string,
    token_length,
    tokens_share_base_token,
)


@dataclass
class ParsedSequence:
    """保存一次最长匹配解析得到的全部派生信息。

    输入语义：由基础 token 序列和当前 grammar token 集合解析得到。
    输出语义：tokens 保存核心 tuple token，token_strings 保存对应输出字符串，span_starts/span_lengths
    记录每个解析片段覆盖的原始位置，计数、概率、时间占比和 position_grammar 保存后续学习或验证需要的派生指标。
    关键约束：该结构只描述解析结果，不保存候选筛选、pair posterior 或 BD score 结果。
    """

    tokens: list[GrammarToken]
    token_strings: list[str]
    span_starts: list[int]
    span_lengths: list[int]
    token_counts: dict[str, int]
    token_probabilities: dict[str, float]
    token_time: dict[str, float]
    position_grammar: list[str]


@dataclass
class OrganizedGrammarData:
    """保存离散化后的语法学习矩阵。

    输入语义：由解析后的 token 序列、有效 grammar token 列表、状态特征表和状态依赖图整理得到。
    输出语义：data_child/data_parent 使用 1 表示未出现、2 表示出现；data_condition 保存状态值 + 1；
    condition_state 记录每个 child token 需要附加评估的状态条件列名。
    关键约束：字段顺序必须与 active_tokens 和状态列顺序一致，供后续 BD score 计算按位置索引。
    """

    data_child: pd.DataFrame
    data_parent: pd.DataFrame
    data_condition: pd.DataFrame
    condition_state: list[list[str]]


@dataclass
class GrammarLearningResult:
    """保存一次 grammar 学习的完整内存结果。

    输入语义：由 GrammarLearner.learn 根据基础 token 序列、初始 token 集、状态特征和状态依赖图产生。
    输出语义：grammar_tokens、probabilities、frequencies 描述最终 token 分布；parsed_sequence 和
    parsed_state_features 描述最长匹配后的序列及其状态对齐；components 保存合并 token 的直接组成。
    关键约束：核心结果只使用模块内 token 表示，例如 "G-L"、"E-A"；外部格式映射应在适配层完成。
    """

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
    """记录 skip-gram 检测结果。

    输入语义：由最终解析序列和删除 token 的原始位置检测得到。
    输出语义：found 表示是否满足 skip-gram 判定；count 表示目标共现计数，不满足时为 0。
    关键约束：count 来自 BD score 后验矩阵中的二值共现单元，调用方不应假设它一定是整数类型。
    """

    found: bool
    count: int | float


def static_probability(tokens: Sequence[str], active_tokens: Sequence[str]) -> list[float]:
    """按 active_tokens 顺序统计当前解析序列中各 token 的出现概率。

    输入语义：tokens 是当前解析后的 token 序列；active_tokens 是需要统计的完整候选 token 顺序。
    输出语义：返回与 active_tokens 等长的概率列表，概率为对应 token 计数除以总计数。
    关键约束：tokens 中的每个元素都必须已存在于 active_tokens，否则计数表无法更新。
    """

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
    """从候选 chunk 中选择接近最佳得分的一组结果。

    输入语义：ratios、chunks、components 按相同索引描述候选得分、合并 token 和组成 token；
    keep_ratio 控制候选得分与最佳得分的最小相对比例。
    输出语义：返回被选中的 chunk、ratio 和组成列表，顺序按 ratio 从高到低排列。
    关键约束：只有 ratio > 1 的候选才表示加入 parent 后得分改善；输入三个列表必须等长。
    """

    # 先按 ratio 降序排序，便于从最佳候选开始做相对阈值筛选。
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
        # keep_ratio 表示候选与本轮最佳 ratio 的接近程度，低于阈值后停止扩展本轮合并集。
        if candidate[1] / best_ratio > keep_ratio:
            selected.append(candidate)
        else:
            break

    selected_chunks, selected_ratios, selected_components = zip(*selected)
    return list(selected_chunks), list(selected_ratios), list(selected_components)


def kl_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """计算两个离散概率分布之间的 KL 散度。

    输入语义：p 是当前分布，q 是参考分布，键为 token，值为非零概率。
    输出语义：返回以 log2 计算的 KL(p || q) 标量。
    关键约束：当参考分布缺少 p 中某个 token 时使用固定小概率平滑，避免除零或缺键。
    """

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
    """执行 grammar chunk 学习和 skip-gram 检测。

    输入语义：实例只接收 GrammarLearningParams，实际数据通过 learn 和 detect_skip_gram 传入。
    输出语义：learn 返回 GrammarLearningResult；detect_skip_gram 返回 SkipGramResult。
    关键约束：类本身不读写文件、不持有路径；所有 token 合并、概率统计和状态条件都在内存中完成。
    """

    def __init__(self, params: GrammarLearningParams):
        """初始化语法学习器。

        输入语义：params 提供迭代次数、候选筛选阈值、BD score 参数和 skip-gram 配置。
        输出语义：构造后的实例可复用同一组参数处理多个内存数据集。
        关键约束：初始化阶段不校验数据路径，也不产生随机状态。
        """

        # GrammarLearner 只持有算法参数，不持有路径，也不读写文件。
        self.params = params

    def _build_parsed_sequence(
        self,
        token_sequence: Sequence[str],
        grammar_tokens: Sequence[str],
    ) -> ParsedSequence:
        """一次性构建最长匹配解析及其概率派生信息。

        输入语义：token_sequence 是基础 token 序列；grammar_tokens 是当前有效 grammar token 顺序。
        输出语义：返回 ParsedSequence，包含 tuple token、字符串 token、原始跨度、频次、概率、时间占比
        和 position_grammar。
        关键约束：匹配规则必须与旧解析完全一致；同一位置多个候选命中时，只在长度更长时替换，
        相同长度保持 grammar_tokens 中先出现的候选。
        """

        # 边界输入仍是字符串，进入核心解析后立即展开为基础动作 tuple，避免在匹配循环中反复 split。
        token_sequence_list = list(token_sequence)
        base_tokens: GrammarToken = tuple(
            base_token
            for token in token_sequence_list
            for base_token in parse_token_string(token)
        )
        grammar_token_pairs = [
            (grammar_token, parse_token_string(grammar_token))
            for grammar_token in grammar_tokens
        ]
        last_grammar_length = len(grammar_token_pairs[-1][1])

        parsed_tokens: list[GrammarToken] = []
        token_strings: list[str] = []
        span_starts: list[int] = []
        span_lengths: list[int] = []
        position_grammar: list[str] = []
        pointer = 0

        while pointer < len(base_tokens):
            matched_index = 0
            matched_length = 0
            for index, (_, grammar_token) in enumerate(grammar_token_pairs):
                length = len(grammar_token)
                # 这里保留旧实现的稳定最长匹配语义：只有更长命中才替换，等长候选保持先出现者。
                if base_tokens[pointer:pointer + length] == grammar_token and length > matched_length:
                    matched_length = length
                    matched_index = index
            if matched_length == 0:
                raise ValueError(f"No grammar token matches sequence position {pointer}: {token_sequence_list[pointer:]}")

            _, matched_token = grammar_token_pairs[matched_index]
            matched_string = format_grammar_token(matched_token)
            parsed_tokens.append(matched_token)
            token_strings.append(matched_string)
            span_starts.append(pointer)
            span_lengths.append(matched_length)
            # position_grammar 必须保留旧 parse_pro 的固定填充规则：使用最后一个 grammar token 的长度。
            position_grammar += [matched_string] * last_grammar_length
            pointer += matched_length

        token_counts = {grammar_token: 0 for grammar_token in grammar_tokens}
        for token_string in token_strings:
            token_counts[token_string] += 1

        total_count = np.sum(list(token_counts.values()))
        token_probabilities = {
            token_string: count / total_count
            for token_string, count in token_counts.items()
        }

        # token_time 表示基础 token 覆盖占比；它只由频次和 token 基础长度决定。
        weighted_counts = {
            token_string: token_counts[token_string] * len(token_tuple)
            for token_string, token_tuple in grammar_token_pairs
        }
        total_weighted_count = np.sum(list(weighted_counts.values()))
        token_time = {
            token_string: weighted_counts[token_string] / total_weighted_count
            for token_string in token_counts
        }

        return ParsedSequence(
            tokens=parsed_tokens,
            token_strings=token_strings,
            span_starts=span_starts,
            span_lengths=span_lengths,
            token_counts=token_counts,
            token_probabilities=token_probabilities,
            token_time=token_time,
            position_grammar=position_grammar,
        )

    def _align_state_features_to_parsed_sequence(
        self,
        parsed: ParsedSequence,
        state_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """按解析片段起点把状态特征对齐到 ParsedSequence。

        输入语义：parsed 保存每个 chunk 在原始基础 token 序列中的起点；state_features 与基础 token 序列按行对齐。
        输出语义：返回与 parsed.token_strings 等长的状态表，每行取对应 chunk 起点的状态。
        关键约束：保持旧 `_parse_longest()` 行为，状态不做聚合、不取 chunk 末尾行。
        """

        # 一个 chunk 覆盖多个基础 token 时，旧实现只取该片段首个基础 token 的状态行。
        parsed_state_rows = [list(state_features.iloc[start]) for start in parsed.span_starts]
        return pd.DataFrame(parsed_state_rows, columns=state_features.columns)

    def _parse_longest(
        self,
        tokens: list[str],
        grammar_tokens: list[str],
        state_features: pd.DataFrame | None = None,
    ) -> tuple[list[str], pd.DataFrame | None]:
        """使用 grammar token 对基础 token 序列做最长匹配解析。

        输入语义：tokens 是基础 token 序列；grammar_tokens 是按学习顺序排列的可匹配 token；
        state_features 可选，若提供则必须与基础 token 序列按行对齐。
        输出语义：返回解析后的 token 序列，以及按每个解析片段首个基础 token 对齐的状态特征表。
        关键约束：grammar_tokens 必须至少覆盖 tokens 的每个位置；多个候选命中时选择基础 token 数最长者。
        """

        parsed = self._build_parsed_sequence(tokens, grammar_tokens)

        if state_features is None:
            return parsed.token_strings, None
        parsed_state_features = self._align_state_features_to_parsed_sequence(parsed, state_features)
        return parsed.token_strings, parsed_state_features

    def _parse_probabilities(
        self,
        tokens: list[str],
        grammar_tokens: list[str],
    ) -> tuple[list[str], list[float], list[str], list[int]]:
        """重新解析基础序列并统计 grammar token 的频数和概率。

        输入语义：tokens 是基础 token 序列；grammar_tokens 是当前有效 token 列表。
        输出语义：返回 token 列表、对应概率、位置展开后的 grammar 序列和对应频数。
        关键约束：输出顺序保持 grammar_tokens 顺序；position_grammar 的展开长度遵循固定填充长度约束。
        """

        parsed = self._build_parsed_sequence(tokens, grammar_tokens)
        frequencies = [parsed.token_counts[token] for token in grammar_tokens]
        probabilities = [parsed.token_probabilities[token] for token in grammar_tokens]
        return list(grammar_tokens), probabilities, parsed.position_grammar, frequencies

    def _organize_discrete_data(
        self,
        tokens: list[str],
        active_tokens: list[str],
        state_features: pd.DataFrame,
        state_dependencies: StateDependencyGraph,
    ) -> OrganizedGrammarData:
        """把解析序列和状态特征整理为 BD score 使用的离散矩阵。

        输入语义：tokens 是当前解析序列；active_tokens 是当前可用 grammar token；state_features 与
        tokens 按行对齐；state_dependencies 描述状态之间允许学习的条件关系。
        输出语义：返回 parent、child、condition 三类矩阵以及每个 child token 的状态条件列名。
        关键约束：二值 token 变量使用 1/2 编码；状态变量使用原状态值 + 1 编码，以满足计数矩阵从 1 开始。
        """

        # 把解析后的 token 序列转为离散 parent/child/condition 矩阵。
        # 所有状态都使用 1/2 或 state+1 编码，因为 BDscore/count 假定状态从 1 开始。
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

        # learn_state_condition_links 的 data 前半部分是状态条件变量，后半部分是 grammar parent 变量。
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
            # learned_adjacency[:, index] == 1 表示对应状态列被学习为该 grammar parent 的条件。
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
        """学习 grammar token 并返回最终解析、概率和组成信息。

        输入语义：token_sequence 是已完成输入清理的基础 token 序列；initial_tokens 是初始 token 集；
        state_features 与 token_sequence 按行对齐；state_dependencies 提供状态条件约束；
        participant_file_names 和 participant_ids 作为结果元数据透传。
        输出语义：返回包含最终 grammar token、概率、时间占比、解析序列和组成关系的 GrammarLearningResult。
        关键约束：每轮新增候选后都从 original_sequence 重新解析；收敛由解析概率分布 KL 均值决定。
        """

        # original_sequence 是删除 N 后的基础 token 序列；后续每轮都重新从它做最长匹配，
        # 保证候选 chunk 的加入不会累积破坏基础输入顺序。
        original_sequence = list(token_sequence)
        active_tokens = list(initial_tokens)
        parsed_sequence = list(original_sequence)
        parsed_state_features = state_features.reset_index(drop=True).copy()
        probabilities = static_probability(parsed_sequence, active_tokens)
        components = [[token, ""] for token in active_tokens]

        parsed_prediction = self._build_parsed_sequence(original_sequence, active_tokens)
        predict_tokens = list(active_tokens)
        predict_probabilities = [
            parsed_prediction.token_probabilities[token]
            for token in active_tokens
        ]
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
                # 配置中的 excluded_child_tokens 不参与 chunk 合并目标，避免特殊标记进入 child 评估。
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
                    # 候选 parent 不能与 child 相同，也不能属于排除集；共享基础 token 的组合可按参数拒绝。
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

                    # score_with_parent 是加入候选 grammar parent 后的得分；二者比值作为候选 ratio。
                    score_with_parent, _ = bd_score(
                        data_child,
                        parent_and_condition_data,
                        nstates_child,
                        nstates_parent_and_condition,
                        score_alpha,
                    )
                    _, pair_posterior = bd_score(data_child, data_parent, 2, 2, 1)
                    pair_frequency = pair_posterior[1, 1] / len(parsed_sequence)
                    # parent-child 同现频率必须同时高于独立概率乘积和最小频率阈值，过滤弱关联候选。
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

            # 一轮可能选出多个接近最佳 ratio 的 chunk；它们按候选筛选顺序追加到 active_tokens。
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
                # 防止重复加入相同 token，避免在候选重复时产生无意义循环。
                if chunk in active_tokens:
                    continue
                active_tokens.append(chunk)
                components.append(list(selected_components[index]))
                added_any = True
            if not added_any:
                break

            # 每次加入新 chunk 后，都从 original_sequence 重新做最长匹配，更新解析序列和对齐状态。
            parsed_result = self._build_parsed_sequence(
                original_sequence,
                active_tokens,
            )
            parsed_sequence = parsed_result.token_strings
            parsed_state_features = self._align_state_features_to_parsed_sequence(parsed_result, state_features)
            probabilities = [
                parsed_result.token_probabilities[token]
                for token in active_tokens
            ]

            # 使用解析概率分布的 KL 均值作为收敛条件；窗口和阈值由参数控制。
            current_distribution = {
                token: parsed_result.token_probabilities[token]
                for token in active_tokens
                if parsed_result.token_probabilities[token] != 0
            }
            kl_history.append(kl_divergence(current_distribution, previous_distribution))
            previous_distribution = dict(current_distribution)
            if (
                len(kl_history) >= self.params.convergence_window
                and np.mean(kl_history[-self.params.convergence_window:]) <= self.params.convergence_kl_threshold
            ):
                break

        # 循环结束后，按最终 active_tokens 重新统计 sets/pro/gram/frequency。
        final_parsed = self._build_parsed_sequence(
            original_sequence,
            active_tokens,
        )
        grammar_tokens = list(active_tokens)
        probabilities = [
            final_parsed.token_probabilities[token]
            for token in active_tokens
        ]
        position_grammar = final_parsed.position_grammar
        frequencies = [
            final_parsed.token_counts[token]
            for token in active_tokens
        ]
        # 删除概率为 0 的 grammar，保证输出只包含最终解析中实际出现过的项。
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
        """检测删除 token 与目标 token 的 skip-gram 关系。

        输入语义：result 提供最终解析序列；n_positions 是被删除 token 在原始基础序列中的位置。
        输出语义：返回是否检测到 skip-gram 以及对应共现计数。
        关键约束：检测窗口按解析后的 chunk 序列移动，但删除 token 的插回位置按基础 token 长度映射。
        """

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
                # 窗口内跳过被删除 token，只要命中配置的目标 token 即标记该 N 位置有效。
                if sequence_with_n[next_index] != self.params.removed_token and (
                    sequence_with_n[next_index] == self.params.skip_gram_target
                ):
                    target_child[index] = 2
                    break

        # BDscore 输入仍使用 1/2 编码，表示 N parent 与目标 child 两个二值变量。
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
