"""generate_grammar 概率评分测试。

覆盖离散状态计数、BD score 计算和状态条件连线学习。
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from LoPS.generate_grammar.config import DEFAULT_STATE_NAMES
from LoPS.generate_grammar.data import load_state_dependency_graph, load_strategy_state_data
from LoPS.structure_learning import bd_score, count_state_combinations, learn_condition_effect_links
from tests.generate_grammar_fixtures import (
    HAS_REPRESENTATIVE_GRAMMAR_INPUTS,
    STATE_GRAPH_DIR,
    STRATEGY_SEQUENCE_DIR,
)


def _build_real_condition_link_inputs() -> tuple[np.ndarray, np.ndarray, dict[int, list[int]], int, int, int, list[list[int]]]:
    """构造状态条件链接学习所需的真实数据矩阵。

    返回值依次为离散观测矩阵、每行状态数、条件块映射、条件变量数量、
    条件块数量、效果变量数量和状态依赖列表。矩阵使用 1-based 编码，
    以匹配 BD 评分函数的输入约束。
    """
    # 该夹具不仅覆盖小数组，也覆盖默认路径真实数据下的状态条件学习行为。
    record = load_strategy_state_data(
        STRATEGY_SEQUENCE_DIR / "031222-401-03-Dec-2022-1.pkl",
        DEFAULT_STATE_NAMES,
    )
    sequence = "".join(record.token_sequence)
    # 学习流程在进入 chunking 前删除 N，并同步删除对应状态行。
    index_n = np.where(np.array(list(sequence)) == "N")[0]
    sequence = sequence.replace("N", "")

    state = record.state_features.reset_index(drop=True)
    state = state.drop(index_n).reset_index(drop=True)

    # data_parent 和 data_policy_condition 均使用 1/2 或 state+1 编码，满足 BD 评分输入约束。
    data_parent = {token: np.ones(len(sequence) - 1) for token in record.initial_tokens}
    data_policy_condition = {state_name: np.ones(len(sequence) - 1) for state_name in state.columns}

    for index in range(1, len(sequence)):
        data_parent[sequence[index - 1]][index - 1] = 2
        for state_name in state.columns:
            data_policy_condition[state_name][index - 1] = state[state_name].iloc[index - 1] + 1

    data_parent_frame = pd.DataFrame(data_parent, dtype=int)
    data_policy_condition_frame = pd.DataFrame(data_policy_condition, dtype=int)
    data = pd.concat([data_policy_condition_frame, data_parent_frame], axis=1).values.T
    data = np.array(data, dtype=int)
    nstates = np.max(data, axis=1).T.astype(int)
    casual_num = data_policy_condition_frame.shape[1]
    effect_num = data_parent_frame.shape[1]
    block_message = {index: [index] for index in range(casual_num)}
    graph = load_state_dependency_graph(STATE_GRAPH_DIR / "031222-401-03-Dec-2022-1.pkl")
    return data, nstates, block_message, casual_num, len(block_message), effect_num, graph.conditions_by_state


class GenerateGrammarScoringTest(unittest.TestCase):
    """覆盖 generate_grammar 评分和状态条件链接学习逻辑。"""

    def test_count_state_combinations_matches_legacy_count(self) -> None:
        """验证状态组合计数遵守 1-based 编码和组合索引顺序。"""
        # 小数组测试锁定 count 的 1-based 编码和组合索引计算。
        data = np.array(
            [
                [1, 2, 1, 2, 2, 1],
                [1, 1, 2, 2, 1, 2],
            ],
            dtype=int,
        )
        nstates = np.array([2, 2], dtype=int)

        # 期望值来自固定小数组的行为快照，避免测试运行时依赖外部代码目录。
        expected = np.array([1.0, 2.0, 2.0, 1.0])
        actual = count_state_combinations(data, nstates)

        np.testing.assert_array_equal(actual, expected)

    def test_bd_score_matches_legacy_bd_score(self) -> None:
        """验证 BD score 和 posterior 矩阵在固定输入上保持稳定。"""
        # 固定 parent/child 数组直接比较 score 和 posterior，防止 gammaln 公式或 reshape 顺序漂移。
        data_v = np.array([1, 2, 1, 2, 2, 1], dtype=int)
        data_parents = np.array(
            [
                [1, 1, 2, 2, 1, 2],
                [2, 1, 2, 1, 1, 2],
            ],
            dtype=int,
        )
        nstates_v = 2
        nstates_parents = np.array([2, 2], dtype=int)
        alpha = 0.5 / (np.prod(nstates_v) * np.prod(nstates_parents))

        # 期望值来自固定输入的行为快照；概率后验必须逐元素一致。
        expected_score = -2.886905549919682
        expected_posterior = np.array(
            [
                [0.0625, 0.0625, 1.0625, 2.0625],
                [2.0625, 1.0625, 0.0625, 0.0625],
            ]
        )
        actual_score, actual_posterior = bd_score(data_v, data_parents, nstates_v, nstates_parents, alpha)

        self.assertEqual(actual_score, expected_score)
        np.testing.assert_array_equal(actual_posterior, expected_posterior)

    @unittest.skipUnless(
        HAS_REPRESENTATIVE_GRAMMAR_INPUTS,
        "保留的 08–12 集成测试数据尚未生成",
    )
    def test_learn_state_condition_links_matches_legacy_learn_bayes_net_block(self) -> None:
        """验证真实数据下状态条件链接学习输出固定邻接矩阵。"""
        # 真实文件测试使用迁移后的 LoPS/data 输入，并比较固定输出快照。
        data, nstates, block_message, casual_num, block_num, effect_num, conditions = _build_real_condition_link_inputs()

        expected_adjacency = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=int,
        )
        actual_adjacency, _, _, _ = learn_condition_effect_links(
            data=data,
            nstates=nstates,
            block_message=block_message,
            casual_num=casual_num,
            block_num=block_num,
            effect_num=effect_num,
            alpha=0.5,
            conditions=conditions,
        )

        np.testing.assert_array_equal(actual_adjacency, expected_adjacency)


if __name__ == "__main__":
    unittest.main()
