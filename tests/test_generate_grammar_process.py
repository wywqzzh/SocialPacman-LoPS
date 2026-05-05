"""generate_grammar 关键过程一致性测试。

这些测试锁定 Phase 3 优化前的中间指标，防止后续重构只保持最终输出一致，
却改变解析、离散矩阵或候选 posterior 等关键过程语义。
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from LoPS.generate_grammar.config import GrammarLearningParams
from LoPS.generate_grammar.grammar import GrammarLearner
from LoPS.generate_grammar.scoring import bd_score
from LoPS.generate_grammar.state_graph import StateDependencyGraph


class TestGenerateGrammarProcess(unittest.TestCase):
    """覆盖 grammar 学习过程中需要保持一致的关键中间指标。"""

    def setUp(self) -> None:
        """构造所有测试共享的小型确定性序列和学习器。

        输入语义：固定基础 token 序列、grammar token 顺序和状态表。
        输出语义：每个测试可直接复用 learner、tokens、grammar_tokens 和 state_features。
        关键约束：该夹具不读写外部文件，期望值是当前过程行为的内联快照。
        """

        self.learner = GrammarLearner(GrammarLearningParams())
        self.tokens = ["G", "L", "E", "A", "G", "L"]
        self.grammar_tokens = ["G-L", "E-A", "G", "L", "E", "A"]
        self.state_features = pd.DataFrame(
            {
                "IS1": [0, 1, 0, 1, 0, 1],
                "IS2": [1, 0, 1, 0, 1, 0],
                "PG1": [0, 0, 1, 1, 0, 0],
                "PG2": [1, 1, 0, 0, 1, 1],
                "PE": [0, 1, 1, 0, 0, 1],
                "BN5": [1, 0, 0, 1, 1, 0],
            }
        )

    def test_parse_longest_process_matches_snapshot(self) -> None:
        """验证最长匹配解析和状态行对齐过程保持固定。

        输入语义：基础 token 序列、复合 grammar token 和逐基础 token 对齐的状态表。
        输出语义：解析 token 序列与解析后状态行应匹配当前过程快照。
        关键约束：状态行使用 chunk 覆盖区间的首个基础 token 行，不能改为末尾或聚合状态。
        """

        parsed, parsed_state = self.learner._parse_longest(
            self.tokens,
            self.grammar_tokens,
            self.state_features,
        )

        self.assertEqual(parsed, ["G-L", "E-A", "G-L"])
        self.assertIsNotNone(parsed_state)
        self.assertEqual(
            parsed_state.to_dict("list"),
            {
                "IS1": [0, 0, 0],
                "IS2": [1, 1, 1],
                "PG1": [0, 1, 0],
                "PG2": [1, 0, 1],
                "PE": [0, 1, 0],
                "BN5": [1, 0, 1],
            },
        )

    def test_parse_probabilities_process_matches_snapshot(self) -> None:
        """验证解析概率、频次和位置级 grammar 展开保持固定。

        输入语义：基础 token 序列和当前 grammar token 顺序。
        输出语义：返回 grammar 顺序、概率、position_grammar 和频次快照。
        关键约束：概率和频次顺序必须严格跟随 grammar_tokens，而不是按出现顺序重新排序。
        """

        grammar_tokens, probabilities, position_grammar, frequencies = self.learner._parse_probabilities(
            self.tokens,
            self.grammar_tokens,
        )

        self.assertEqual(grammar_tokens, self.grammar_tokens)
        self.assertEqual(position_grammar, ["G-L", "E-A", "G-L"])
        self.assertEqual(frequencies, [2, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(
            np.array(probabilities, dtype=float),
            np.array([2 / 3, 1 / 3, 0, 0, 0, 0], dtype=float),
        )

    def test_build_parsed_sequence_matches_legacy_parse_outputs(self) -> None:
        """验证 ParsedSequence 与旧解析入口的过程结果完全一致。

        输入语义：基础 token 序列和 grammar token 顺序。
        输出语义：ParsedSequence 中的 tuple token、字符串 token、跨度、频次、概率和 position_grammar
        必须能还原旧 `_parse_longest()` 与 `_parse_probabilities()` 的返回值。
        关键约束：该测试保护 03-02 的共享解析实现，防止后续重新引入重复但不一致的解析逻辑。
        """

        parsed = self.learner._build_parsed_sequence(self.tokens, self.grammar_tokens)
        legacy_tokens, _ = self.learner._parse_longest(self.tokens, self.grammar_tokens)
        legacy_grammar, legacy_probabilities, legacy_position_grammar, legacy_frequencies = (
            self.learner._parse_probabilities(self.tokens, self.grammar_tokens)
        )

        self.assertEqual(parsed.tokens, [("G", "L"), ("E", "A"), ("G", "L")])
        self.assertEqual(parsed.token_strings, legacy_tokens)
        self.assertEqual(parsed.span_starts, [0, 2, 4])
        self.assertEqual(parsed.span_lengths, [2, 2, 2])
        self.assertEqual(parsed.position_grammar, legacy_position_grammar)
        self.assertEqual([parsed.token_counts[token] for token in legacy_grammar], legacy_frequencies)
        np.testing.assert_array_equal(
            np.array([parsed.token_probabilities[token] for token in legacy_grammar], dtype=float),
            np.array(legacy_probabilities, dtype=float),
        )
        np.testing.assert_array_equal(
            np.array([parsed.token_time[token] for token in legacy_grammar], dtype=float),
            np.array([2 / 3, 1 / 3, 0, 0, 0, 0], dtype=float),
        )

    def test_organize_discrete_data_process_matches_snapshot(self) -> None:
        """验证离散 parent/child/condition 矩阵和状态条件列表保持固定。

        输入语义：解析后的 token 序列、active token 顺序、解析后状态表和空状态依赖图。
        输出语义：离散矩阵使用 1/2 或 state+1 编码，状态条件列表按 active token 顺序排列。
        关键约束：parent 对齐前一解析 token，child 对齐当前解析 token，condition 对齐 child 时刻。
        """

        parsed_tokens, parsed_state = self.learner._parse_longest(
            self.tokens,
            self.grammar_tokens,
            self.state_features,
        )
        self.assertIsNotNone(parsed_state)
        parsed = self.learner._build_parsed_sequence(self.tokens, self.grammar_tokens)

        organized = self.learner._organize_discrete_data(
            parsed,
            self.grammar_tokens,
            parsed_state,
            StateDependencyGraph([[], [], [], [], [], []]),
        )

        self.assertEqual(parsed_tokens, ["G-L", "E-A", "G-L"])
        self.assertEqual(organized.token_names, self.grammar_tokens)
        self.assertEqual(organized.state_names, ["IS1", "IS2", "PG1", "PG2", "PE", "BN5"])
        np.testing.assert_array_equal(
            organized.data_parent,
            np.array(
                [
                    [2, 1],
                    [1, 2],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                ],
                dtype=int,
            ),
        )
        np.testing.assert_array_equal(
            organized.data_child,
            np.array(
                [
                    [1, 2],
                    [2, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                ],
                dtype=int,
            ),
        )
        np.testing.assert_array_equal(
            organized.data_condition,
            np.array(
                [
                    [1, 1],
                    [2, 2],
                    [2, 1],
                    [1, 2],
                    [2, 1],
                    [1, 2],
                ],
                dtype=int,
            ),
        )
        self.assertEqual(
            [[str(name) for name in names] for names in organized.condition_state],
            [
                ["PG1", "PG2", "PE", "BN5"],
                ["PG1", "PG2", "PE", "BN5"],
                [],
                [],
                [],
                [],
            ],
        )
        expected_adjacency = np.zeros((12, 12))
        expected_adjacency[2:6, 6] = 1
        expected_adjacency[2:6, 7] = 1
        np.testing.assert_array_equal(organized.learned_state_adjacency, expected_adjacency)

    def test_pair_posterior_comes_from_bd_score_not_raw_count(self) -> None:
        """验证候选 pair_posterior 来自 BD score 后验而不是纯频次。

        输入语义：使用离散矩阵中 `E-A -> G-L` 的 parent/child 二值变量。
        输出语义：posterior 包含 Dirichlet 先验平滑后的二维矩阵。
        关键约束：posterior[1, 1] 为 2.0，明显不同于该 pair 的纯 raw count 1。
        """

        _, parsed_state = self.learner._parse_longest(
            self.tokens,
            self.grammar_tokens,
            self.state_features,
        )
        self.assertIsNotNone(parsed_state)
        parsed = self.learner._build_parsed_sequence(self.tokens, self.grammar_tokens)
        organized = self.learner._organize_discrete_data(
            parsed,
            self.grammar_tokens,
            parsed_state,
            StateDependencyGraph([[], [], [], [], [], []]),
        )

        data_child = organized.child_values("G-L")
        data_parent = organized.parent_values("E-A").reshape(1, -1)
        _, pair_posterior = bd_score(data_child, data_parent, 2, 2, 1)
        raw_count = int(np.sum((data_child == 2) & (data_parent.reshape(-1) == 2)))

        np.testing.assert_array_equal(pair_posterior, np.array([[2.0, 1.0], [1.0, 2.0]]))
        self.assertEqual(raw_count, 1)
        self.assertNotEqual(pair_posterior[1, 1], raw_count)


if __name__ == "__main__":
    unittest.main()
