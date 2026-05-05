from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from LoPS.generate_grammar.config import (
    DEFAULT_STATE_NAMES,
    GrammarLearningParams,
)
from LoPS.generate_grammar.data_io import load_strategy_state_data
from LoPS.generate_grammar.grammar import (
    GrammarLearner,
    GrammarLearningResult,
    SkipGramResult,
    choose_candidate_chunks,
)
from LoPS.generate_grammar.state_graph import load_state_dependency_graph
from tests.generate_grammar_fixtures import STATE_GRAPH_DIR, STRATEGY_SEQUENCE_DIR


class GenerateGrammarCoreTest(unittest.TestCase):
    # grammar 核心测试聚焦新 token 表示和纯内存算法，不涉及文件写入和 legacy 占位符。
    def test_parse_longest_uses_composite_tokens(self) -> None:
        # 最长匹配应优先选择 "G-L"、"E-A"，而不是逐个基础 token 解析。
        learner = GrammarLearner(GrammarLearningParams())
        parsed, parsed_state = learner._parse_longest(
            ["G", "L", "E", "A"],
            ["G-L", "E-A", "G", "L", "E", "A"],
        )

        self.assertEqual(parsed, ["G-L", "E-A"])
        self.assertIsNone(parsed_state)

    def test_parse_probabilities_returns_probabilities_and_frequencies(self) -> None:
        # 该测试锁定 parse_pro 等价行为：返回 grammar 顺序、概率、位置 grammar 和频数。
        learner = GrammarLearner(GrammarLearningParams())

        grammar_tokens, probabilities, position_grammar, frequencies = learner._parse_probabilities(
            ["G", "L", "E", "A", "G"],
            ["G-L", "E-A", "G", "L", "E", "A"],
        )

        self.assertEqual(grammar_tokens, ["G-L", "E-A", "G", "L", "E", "A"])
        self.assertEqual(frequencies, [1, 1, 1, 0, 0, 0])
        self.assertEqual(position_grammar, ["G-L", "E-A", "G"])
        self.assertAlmostEqual(probabilities[0], 1 / 3)
        self.assertAlmostEqual(probabilities[1], 1 / 3)
        self.assertAlmostEqual(probabilities[2], 1 / 3)

    def test_choose_candidate_chunks_keeps_ratios_above_threshold_near_best(self) -> None:
        # 候选筛选必须保留 ratio > 1 且与最大 ratio 足够接近的 chunk。
        chunks, ratios, components = choose_candidate_chunks(
            ratios=[1.1, 2.0, 1.8, 1.6, 0.9],
            chunks=["A-B", "G-L", "E-A", "L-G", "S-E"],
            components=[["A", "B"], ["G", "L"], ["E", "A"], ["L", "G"], ["S", "E"]],
            keep_ratio=0.85,
        )

        self.assertEqual(chunks, ["G-L", "E-A"])
        self.assertEqual(ratios, [2.0, 1.8])
        self.assertEqual(components, [["G", "L"], ["E", "A"]])

    def test_detect_skip_gram_finds_constructed_n_to_ea_case(self) -> None:
        # 构造一个 N 后第 2 个 token 命中 E-A 的序列，验证 skip-gram 判定可触发。
        learner = GrammarLearner(GrammarLearningParams())
        result = GrammarLearningResult(
            grammar_tokens=["G", "E-A"],
            probabilities=[0.5, 0.5],
            position_grammar=[],
            original_sequence=["G", "G", "E", "A"],
            time_probabilities=np.array([0.5, 0.5]),
            frequencies=[2, 1],
            parsed_sequence=["G", "G", "E-A"],
            parsed_state_features=pd.DataFrame(),
            active_tokens=["G", "E-A"],
            participant_file_names=[],
            participant_ids=[],
            components=[["G", ""], ["E", "A"]],
        )

        skip_gram = learner.detect_skip_gram(result, np.array([0]))

        self.assertIsInstance(skip_gram, SkipGramResult)
        self.assertTrue(skip_gram.found)
        self.assertGreater(skip_gram.count, 0)

    def test_learn_returns_result_for_representative_real_file(self) -> None:
        # 代表性真实文件 smoke test：确保核心 learn 能处理旧数据并返回非空结果。
        record = load_strategy_state_data(
            STRATEGY_SEQUENCE_DIR / "031222-401.pkl",
            DEFAULT_STATE_NAMES,
        )
        sequence = "".join(record.token_sequence)
        n_positions = np.where(np.array(list(sequence)) == "N")[0]
        clean_sequence = list(sequence.replace("N", ""))
        state_features = record.state_features.reset_index(drop=True).drop(n_positions).reset_index(drop=True)
        state_dependencies = load_state_dependency_graph(STATE_GRAPH_DIR / "031222-401.pkl")

        learner = GrammarLearner(GrammarLearningParams())
        result = learner.learn(
            clean_sequence,
            record.initial_tokens,
            state_features,
            state_dependencies,
            record.participant_file_names,
            record.participant_ids,
        )

        self.assertIsInstance(result, GrammarLearningResult)
        self.assertTrue(result.grammar_tokens)
        self.assertTrue(result.probabilities)
        self.assertTrue(result.parsed_sequence)
        self.assertEqual(result.participant_file_names, record.participant_file_names)
        self.assertEqual(result.participant_ids, record.participant_ids)


if __name__ == "__main__":
    unittest.main()
