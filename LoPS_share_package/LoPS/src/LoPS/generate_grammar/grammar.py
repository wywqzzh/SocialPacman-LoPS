"""语法学习核心算法。

本模块只处理内存中的 token 序列、状态特征和概率评分，不负责文件读写或验证格式转换。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from LoPS.generate_grammar.config import GrammarLearningParams
from LoPS.structure_learning import StateDependencyGraph, bd_score, learn_condition_effect_links
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


@dataclass(frozen=True)
class DiscreteLearningData:
    """保存候选评分所需的离散父子变量、状态条件变量和状态依赖信息。

    输入语义：由 ParsedSequence、active token 顺序、状态矩阵和状态依赖图整理得到。
    输出语义：data_parent/data_child 的行按 token_names 排列、列按相邻解析片段排列；
    data_condition 的行按 state_names 排列、列按 child 时刻排列；condition_state 保存每个 child token
    需要附加的状态条件名称；learned_state_adjacency 保存状态条件学习得到的邻接矩阵。
    关键约束：token 二值变量继续使用 1/2 编码，状态变量继续使用原状态值 + 1 编码。
    """

    data_parent: np.ndarray
    data_child: np.ndarray
    data_condition: np.ndarray
    condition_state: list[list[str]]
    learned_state_adjacency: np.ndarray
    token_names: list[str]
    state_names: list[str]

    def parent_values(self, token: str) -> np.ndarray:
        """按 token 名称取 parent 二值变量行。

        输入语义：token 必须存在于 token_names。
        输出语义：返回该 token 作为前一时刻 parent 的 1/2 编码样本。
        关键约束：返回的是数组视图，调用方只读使用，不应原地修改。
        """

        return self.data_parent[self.token_names.index(token)]

    def child_values(self, token: str) -> np.ndarray:
        """按 token 名称取 child 二值变量行。

        输入语义：token 必须存在于 token_names。
        输出语义：返回该 token 作为当前时刻 child 的 1/2 编码样本。
        关键约束：行顺序由 token_names 明确约束，不再依赖 DataFrame 列名。
        """

        return self.data_child[self.token_names.index(token)]

    def condition_values(self, state_names: Sequence[str]) -> np.ndarray:
        """按状态名称取 condition 状态变量矩阵。

        输入语义：state_names 是需要作为 BD score parent 条件的状态列名。
        输出语义：返回形状为 条件状态数 x 样本数 的 1-based 状态矩阵。
        关键约束：空状态列表返回 0 行数组，供调用方按无条件路径处理。
        """

        if len(state_names) == 0:
            return np.empty((0, self.data_condition.shape[1]), dtype=int)
        condition_indices = [self.state_names.index(name) for name in state_names]
        return self.data_condition[condition_indices]


@dataclass(frozen=True)
class CandidateScore:
    """保存单个 parent-child 候选的评分过程结果。

    输入语义：由当前解析结果、离散学习矩阵、child token 和 parent token 计算得到。
    输出语义：记录候选 chunk、直接组成、无 parent 得分、有 parent 得分、pair posterior、
    pair frequency 和用于候选选择的 ratio。
    关键约束：这是单个候选行的过程快照，不做跨候选预筛选，也不改变候选遍历顺序。
    """

    parent_token: str
    child_token: str
    chunk: str
    components: list[str]
    score_without_parent: float
    score_with_parent: float
    pair_posterior: np.ndarray
    pair_frequency: float
    ratio: float


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


@dataclass(frozen=True)
class SkipGramCandidateTrace:
    """保存一次 skip-gram 检测的关键过程指标。

    输入语义：由最终解析序列和被删除 token 原始位置构造。
    输出语义：记录插回 N 后的序列、N 插入位置、BD score 输入变量、得分、posterior 和最终判定指标。
    关键约束：该结构只用于测试和过程解释，不进入正式结构化输出。
    """

    parent_token: str
    child_token: str
    sequence_with_n: list[str]
    n_insert_positions: list[int]
    n_parent: np.ndarray
    target_child: np.ndarray
    score_without_parent: float
    score_with_parent: float
    posterior: np.ndarray
    pair_frequency: float


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
        parsed: ParsedSequence,
        active_tokens: list[str],
        state_features: pd.DataFrame,
        state_dependencies: StateDependencyGraph,
    ) -> DiscreteLearningData:
        """把解析序列和状态特征整理为 BD score 使用的离散矩阵。

        输入语义：parsed 是当前解析结果；active_tokens 是当前可用 grammar token；
        state_features 与 parsed.token_strings 按行对齐；state_dependencies 描述状态之间允许学习的条件关系。
        输出语义：返回 ndarray 形式的 parent、child、condition 矩阵以及状态依赖学习结果。
        关键约束：二值 token 变量使用 1/2 编码；状态变量使用原状态值 + 1 编码；每次调用仍重新学习状态条件链接。
        """

        tokens = parsed.token_strings
        transition_count = len(tokens) - 1
        token_to_index = {token: index for index, token in enumerate(active_tokens)}
        state_names = list(state_features.columns)
        # DataFrame 只作为输入边界；核心矩阵构建使用 ndarray 和显式 state_names 顺序。
        state_values = np.asarray(state_features.reset_index(drop=True).values, dtype=int)

        data_parent = np.ones((len(active_tokens), transition_count), dtype=int)
        data_child = np.ones((len(active_tokens), transition_count), dtype=int)
        data_condition = np.ones((len(state_names), transition_count), dtype=int)
        data_policy_condition = np.ones((len(state_names), transition_count), dtype=int)

        for index in range(1, len(tokens)):
            # parent 使用上一个解析 token，child 使用当前解析 token；样本列对应相邻解析片段。
            data_parent[token_to_index[tokens[index - 1]], index - 1] = 2
            data_child[token_to_index[tokens[index]], index - 1] = 2
            # condition 对齐 child 时刻，policy_condition 对齐 parent 时刻，保持旧 BDscore 输入语义。
            data_condition[:, index - 1] = state_values[index] + 1
            data_policy_condition[:, index - 1] = state_values[index - 1] + 1

        # learn_condition_effect_links 的 data 前半部分是状态条件变量，后半部分是 grammar parent 变量。
        data = np.vstack((data_policy_condition, data_parent)).astype(int)
        nstates = np.max(data, axis=1).T.astype(int)
        casual_num = len(state_names)
        effect_num = len(active_tokens)
        block_message = {index: [index] for index in range(casual_num)}

        # 通过状态图学习每个 grammar token 需要附加哪些状态条件。
        learned_adjacency, _, _, _ = learn_condition_effect_links(
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
        for index in range(casual_num, casual_num + effect_num):
            # learned_adjacency[:, index] == 1 表示对应状态列被学习为该 grammar parent 的条件。
            condition_indices = np.where(learned_adjacency[:, index] == 1)[0]
            condition_state.append([state_names[condition_index] for condition_index in condition_indices])

        return DiscreteLearningData(
            data_parent=data_parent,
            data_child=data_child,
            data_condition=data_condition,
            condition_state=condition_state,
            learned_state_adjacency=learned_adjacency,
            token_names=list(active_tokens),
            state_names=state_names,
        )

    def _score_candidate_pair(
        self,
        organized: DiscreteLearningData,
        probabilities: Sequence[float],
        parsed_length: int,
        child_index: int,
        parent_index: int,
        child_token: str,
        parent_token: str,
        data_child: np.ndarray,
        data_condition: np.ndarray | list,
        nstates_child: int,
        nstates_condition: np.ndarray | list,
        score_without_parent: float,
    ) -> CandidateScore | None:
        """计算单个 parent-child 候选的 BD score 和过滤结果。

        输入语义：organized 提供离散 parent 矩阵；probabilities 是当前 active token 概率；
        child_index/parent_index 对应 active token 顺序；data_child 和状态条件得分上下文由 child 循环预先计算。
        输出语义：候选满足现有 pair posterior 过滤规则时返回 CandidateScore，否则返回 None。
        关键约束：函数内部保持旧评分顺序：先计算 score_with_parent，再计算 pair_posterior，
        最后按 pair_frequency 与独立概率乘积、最小频率阈值过滤。
        """

        # 候选 parent 不能与 child 相同，也不能属于排除集；共享基础 token 的组合可按参数拒绝。
        if parent_token == child_token or parent_token in self.params.excluded_parent_tokens:
            return None
        if self.params.reject_shared_base_tokens and tokens_share_base_token(parent_token, child_token):
            return None

        data_parent = organized.parent_values(parent_token).reshape(1, -1)
        nstates_parent = int(np.max(data_parent).T)
        if len(data_condition) != 0:
            # 有状态条件时，候选 parent 和状态条件一起作为 BDscore 的 parent 集。
            parent_and_condition_data = np.vstack((data_parent, data_condition))
            parent_and_condition_data = np.array(parent_and_condition_data, dtype=int)
            nstates_parent_and_condition = np.array(np.max(parent_and_condition_data, 1).T, dtype=int)
        else:
            parent_and_condition_data = np.array(data_parent, dtype=int)
            nstates_parent_and_condition = nstates_parent

        score_alpha = 1 if self.params.chunk_alpha < 0 else self.params.chunk_alpha
        # score_with_parent 是加入候选 grammar parent 后的得分；二者比值作为候选 ratio。
        score_with_parent, _ = bd_score(
            data_child,
            parent_and_condition_data,
            nstates_child,
            nstates_parent_and_condition,
            score_alpha,
        )
        # pair_posterior 必须继续来自 BD score 后验，其中包含 Dirichlet 先验，不是纯 raw count。
        _, pair_posterior = bd_score(data_child, data_parent, 2, 2, 1)
        pair_frequency = pair_posterior[1, 1] / parsed_length
        if (
            pair_frequency < probabilities[child_index] * probabilities[parent_index]
            or pair_frequency < self.params.min_pair_frequency
        ):
            return None

        ratio = score_without_parent / score_with_parent
        return CandidateScore(
            parent_token=parent_token,
            child_token=child_token,
            chunk=combine_tokens(parent_token, child_token),
            components=[parent_token, child_token],
            score_without_parent=score_without_parent,
            score_with_parent=score_with_parent,
            pair_posterior=pair_posterior,
            pair_frequency=pair_frequency,
            ratio=ratio,
        )

    def _select_next_chunk(
        self,
        candidate_scores: Sequence[CandidateScore],
    ) -> tuple[list[str], list[float], list[list[str]]]:
        """按当前候选选择规则挑选本轮要加入的 chunk。

        输入语义：candidate_scores 按主循环原始遍历顺序排列。
        输出语义：返回被选中的 chunk 字符串、ratio 和直接组成列表。
        关键约束：选择逻辑委托给既有 choose_candidate_chunks()，保留 ratio 降序、ratio > 1
        和 candidate_ratio_keep 的所有旧语义。
        """

        # 这里不重新排序候选行；choose_candidate_chunks 负责按旧规则排序和筛选。
        return choose_candidate_chunks(
            ratios=[candidate.ratio for candidate in candidate_scores],
            chunks=[candidate.chunk for candidate in candidate_scores],
            components=[candidate.components for candidate in candidate_scores],
            keep_ratio=self.params.candidate_ratio_keep,
        )

    def learn(
        self,
        token_sequence: list[str],
        initial_tokens: list[str],
        state_features: pd.DataFrame,
        state_dependencies: StateDependencyGraph,
        participant_file_names: list[str],
        participant_ids: list[str],
        progress_callback: Callable[[str, Mapping[str, object]], None] | None = None,
    ) -> GrammarLearningResult:
        """学习 grammar token 并返回最终解析、概率和组成信息。

        输入语义：token_sequence 是已完成输入清理的基础 token 序列；initial_tokens 是初始 token 集；
        state_features 与 token_sequence 按行对齐；state_dependencies 提供状态条件约束；
        participant_file_names 和 participant_ids 作为结果元数据透传；progress_callback 可选，用于向运行脚本报告
        学习过程，不参与算法计算。
        输出语义：返回包含最终 grammar token、概率、时间占比、解析序列和组成关系的 GrammarLearningResult。
        关键约束：每轮新增候选后都从 original_sequence 重新解析；收敛由解析概率分布 KL 均值决定。
        """

        # original_sequence 是删除 N 后的基础 token 序列；后续每轮都重新从它做最长匹配，
        # 保证候选 chunk 的加入不会累积破坏基础输入顺序。
        original_sequence = list(token_sequence)
        active_tokens = list(initial_tokens)
        parsed_result = self._build_parsed_sequence(original_sequence, active_tokens)
        parsed_sequence = parsed_result.token_strings
        parsed_state_features = self._align_state_features_to_parsed_sequence(parsed_result, state_features)
        probabilities = static_probability(parsed_sequence, active_tokens)
        components = [[token, ""] for token in active_tokens]

        predict_tokens = list(active_tokens)
        predict_probabilities = [
            parsed_result.token_probabilities[token]
            for token in active_tokens
        ]
        previous_distribution = {
            token: predict_probabilities[index]
            for index, token in enumerate(predict_tokens)
        }
        kl_history = []
        stop_reason = "max_iterations"
        iterations_completed = 0

        if progress_callback is not None:
            progress_callback(
                "learn_start",
                {
                    "sequence_length": len(original_sequence),
                    "initial_token_count": len(active_tokens),
                    "initial_parsed_length": len(parsed_sequence),
                    "state_count": len(state_features.columns),
                },
            )

        for iteration in range(1, self.params.max_iterations + 1):
            iterations_completed = iteration
            # 每轮根据当前解析序列重新组织离散数据，再评估哪些 parent->child 组合值得合并。
            organized = self._organize_discrete_data(
                parsed_result,
                active_tokens,
                parsed_state_features,
                state_dependencies,
            )
            candidate_scores = []

            for child_index, child_token in enumerate(active_tokens):
                # 配置中的 excluded_child_tokens 不参与 chunk 合并目标，避免特殊标记进入 child 评估。
                if child_token in self.params.excluded_child_tokens:
                    continue

                data_child = organized.child_values(child_token)
                nstates_child = int(np.max(data_child).T)
                condition_names = organized.condition_state[child_index]
                if len(condition_names) != 0:
                    # 如果状态图认为该 child 受状态条件影响，BDscore 要把这些状态作为 parent 条件。
                    data_condition = organized.condition_values(condition_names)
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
                    candidate_score = self._score_candidate_pair(
                        organized=organized,
                        probabilities=probabilities,
                        parsed_length=len(parsed_sequence),
                        child_index=child_index,
                        parent_index=parent_index,
                        child_token=child_token,
                        parent_token=parent_token,
                        data_child=data_child,
                        data_condition=data_condition,
                        nstates_child=nstates_child,
                        nstates_condition=nstates_condition,
                        score_without_parent=score_without_parent,
                    )
                    if candidate_score is None:
                        continue
                    candidate_scores.append(candidate_score)

            if len(candidate_scores) == 0:
                stop_reason = "no_candidates"
                if progress_callback is not None:
                    progress_callback(
                        "learn_iteration",
                        {
                            "iteration": iteration,
                            "active_tokens": list(active_tokens),
                            "active_token_count": len(active_tokens),
                            "parsed_length": len(parsed_sequence),
                            "candidate_count": 0,
                            "selected_chunks": [],
                            "selected_ratios": [],
                            "kl_divergence": None,
                            "convergence_mean": None,
                            "stop_reason": stop_reason,
                        },
                    )
                break

            # 一轮可能选出多个接近最佳 ratio 的 chunk；它们按候选筛选顺序追加到 active_tokens。
            selected_chunks, selected_ratios, selected_components = self._select_next_chunk(candidate_scores)
            if len(selected_chunks) == 0:
                stop_reason = "no_selected_chunks"
                if progress_callback is not None:
                    progress_callback(
                        "learn_iteration",
                        {
                            "iteration": iteration,
                            "active_tokens": list(active_tokens),
                            "active_token_count": len(active_tokens),
                            "parsed_length": len(parsed_sequence),
                            "candidate_count": len(candidate_scores),
                            "selected_chunks": [],
                            "selected_ratios": [],
                            "kl_divergence": None,
                            "convergence_mean": None,
                            "stop_reason": stop_reason,
                        },
                    )
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
                stop_reason = "duplicate_chunks"
                if progress_callback is not None:
                    progress_callback(
                        "learn_iteration",
                        {
                            "iteration": iteration,
                            "active_tokens": list(active_tokens),
                            "active_token_count": len(active_tokens),
                            "parsed_length": len(parsed_sequence),
                            "candidate_count": len(candidate_scores),
                            "selected_chunks": selected_chunks,
                            "selected_ratios": selected_ratios,
                            "kl_divergence": None,
                            "convergence_mean": None,
                            "stop_reason": stop_reason,
                        },
                    )
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
            kl_value = kl_divergence(current_distribution, previous_distribution)
            kl_history.append(kl_value)
            previous_distribution = dict(current_distribution)
            convergence_mean = None
            converged = False
            if len(kl_history) >= self.params.convergence_window:
                convergence_mean = float(np.mean(kl_history[-self.params.convergence_window:]))
                converged = convergence_mean <= self.params.convergence_kl_threshold
            if progress_callback is not None:
                progress_callback(
                    "learn_iteration",
                    {
                        "iteration": iteration,
                        "active_tokens": list(active_tokens),
                        "active_token_count": len(active_tokens),
                        "parsed_length": len(parsed_sequence),
                        "candidate_count": len(candidate_scores),
                        "selected_chunks": selected_chunks,
                        "selected_ratios": selected_ratios,
                        "kl_divergence": float(kl_value),
                        "convergence_mean": convergence_mean,
                        "stop_reason": "converged" if converged else None,
                    },
                )
            if converged:
                stop_reason = "converged"
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

        if progress_callback is not None:
            progress_callback(
                "learn_finished",
                {
                    "iterations": iterations_completed,
                    "stop_reason": stop_reason,
                    "grammar_token_count": len(grammar_tokens),
                    "parsed_length": len(parsed_sequence),
                    "nonzero_token_count": len(active_tokens),
                },
            )

        return GrammarLearningResult(
            grammar_tokens=grammar_tokens,
            probabilities=probabilities,
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
        progress_callback: Callable[[str, Mapping[str, object]], None] | None = None,
    ) -> SkipGramResult:
        """检测删除 token 与目标 token 的 skip-gram 关系。

        输入语义：result 提供最终解析序列；n_positions 是被删除 token 在原始基础序列中的位置。
        输出语义：返回是否检测到 skip-gram 以及对应共现计数。
        关键约束：检测窗口按解析后的 chunk 序列移动，但删除 token 的插回位置按基础 token 长度映射。
        """

        sequence_with_n, n_insert_positions = self._build_skip_gram_sequence(result.parsed_sequence, n_positions)
        trace = self._score_skip_gram_sequence(sequence_with_n, n_insert_positions)
        score_ratio = trace.score_without_parent / trace.score_with_parent
        found = score_ratio > 1 and trace.pair_frequency > self.params.skip_gram_min_frequency
        count = trace.posterior[1, 1] if found else 0
        if progress_callback is not None:
            progress_callback(
                "skip_gram",
                {
                    "n_count": len(n_positions),
                    "sequence_with_n_length": len(sequence_with_n),
                    "n_insert_count": len(n_insert_positions),
                    "score_ratio": float(score_ratio),
                    "pair_frequency": float(trace.pair_frequency),
                    "found": found,
                    "count": count,
                },
            )
        return SkipGramResult(found, count)

    def _build_skip_gram_sequence(
        self,
        parsed_sequence: Sequence[str],
        n_positions: np.ndarray,
    ) -> tuple[list[str], list[int]]:
        """把删除的 N 按旧位置映射规则插回解析序列。

        输入语义：parsed_sequence 是最终 chunk 解析序列；n_positions 是 N 在原始基础 token 序列中的位置。
        输出语义：返回插入 N 后的序列，以及 N 在新序列中的插入下标。
        关键约束：保持旧实现的单步 if 逻辑；每个解析 token 后最多插入一个 N，不改成 while 批量插入。
        """

        # skip-gram 检测要把之前删除的 N 插回当前解析序列中，再判断 N 后第 2 到第 5 个 token 是否为 E-A。
        position_sum = -1
        n_pointer = 0
        sequence_with_n = []
        n_insert_positions = []
        for token in parsed_sequence:
            # position_sum 以基础 token 数推进，用于把原始 N 位置映射回解析后的 chunk 序列。
            position_sum += token_length(token)
            sequence_with_n.append(token)
            if n_pointer < len(n_positions) and position_sum >= n_positions[n_pointer]:
                n_insert_positions.append(len(sequence_with_n))
                sequence_with_n.append(self.params.removed_token)
                position_sum += 1
                n_pointer += 1
        return sequence_with_n, n_insert_positions

    def _score_skip_gram_sequence(
        self,
        sequence_with_n: Sequence[str],
        n_insert_positions: Sequence[int],
    ) -> SkipGramCandidateTrace:
        """对插回 N 的序列计算 skip-gram BD score 和 posterior。

        输入语义：sequence_with_n 是 `_build_skip_gram_sequence()` 的输出序列；
        n_insert_positions 是该序列中 N 的下标列表。
        输出语义：返回 SkipGramCandidateTrace，包含二值变量、BD score、posterior 和 pair frequency。
        关键约束：目标 token 搜索窗口仍使用 `[min_offset, max_offset]`，并跳过窗口内的 N。
        """

        n_parent = np.array([1] * len(sequence_with_n))
        target_child = np.array([1] * len(sequence_with_n))
        for index in n_insert_positions:
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
        target_child_matrix = target_child.reshape(-1, 1).T
        n_parent_matrix = n_parent.reshape(-1, 1).T
        target_states = int(np.max(target_child_matrix).T)
        n_states = int(np.max(n_parent_matrix).T)

        # score_without_parent/score_with_parent 的比值和 U[1,1] 频率阈值共同决定 skipGram。
        score_without_parent, _ = bd_score(
            target_child_matrix.reshape(-1, 1),
            [],
            target_states,
            [],
            self.params.skip_gram_alpha,
        )
        score_with_parent, posterior = bd_score(
            target_child_matrix,
            n_parent_matrix,
            target_states,
            [n_states],
            self.params.skip_gram_alpha,
        )
        return SkipGramCandidateTrace(
            parent_token=self.params.removed_token,
            child_token=self.params.skip_gram_target,
            sequence_with_n=list(sequence_with_n),
            n_insert_positions=list(n_insert_positions),
            n_parent=n_parent_matrix,
            target_child=target_child_matrix,
            score_without_parent=score_without_parent,
            score_with_parent=score_with_parent,
            posterior=posterior,
            pair_frequency=posterior[1, 1] / len(sequence_with_n),
        )
