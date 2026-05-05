from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from LoPS.generate_grammar.config import (
    DEFAULT_STATE_NAMES,
    GenerateGrammarConfig,
)
from LoPS.generate_grammar.data_io import load_strategy_state_data
from LoPS.generate_grammar.state_graph import load_state_dependency_graph
from LoPS.generate_grammar.token import (
    combine_tokens,
    format_token,
    split_token,
    token_length,
    tokens_share_base_token,
)
from tests.generate_grammar_fixtures import STATE_GRAPH_DIR, STRATEGY_SEQUENCE_DIR


class GenerateGrammarFoundationTest(unittest.TestCase):
    # foundation 测试覆盖最底层的路径配置、token 表示和外部 pickle 读取。
    # 这些能力一旦漂移，后续 scoring、grammar、pipeline 的旧新对照都会失去基础。
    def test_token_helpers(self) -> None:
        # 复合 token 必须按基础 token 拆分和组合，不能按字符串字符数量理解。
        self.assertEqual(split_token("G"), ["G"])
        self.assertEqual(split_token("G-L-E-A"), ["G", "L", "E", "A"])
        self.assertEqual(format_token(["G", "L"]), "G-L")
        self.assertEqual(combine_tokens("G-L", "E-A"), "G-L-E-A")
        self.assertEqual(token_length("G-L-E-A"), 4)
        self.assertTrue(tokens_share_base_token("G-L", "L-E"))
        self.assertFalse(tokens_share_base_token("G-L", "E-A"))

    def test_config_validates_explicit_project_data_inputs(self) -> None:
        # 配置必须由调用方显式传入 LoPS/data 下的数据目录，src 层不再保存任何外部默认路径。
        with tempfile.TemporaryDirectory() as temp_dir:
            config = GenerateGrammarConfig(
                strategy_sequence_dir=STRATEGY_SEQUENCE_DIR,
                state_graph_dir=STATE_GRAPH_DIR,
                output_dir=Path(temp_dir),
            )
            config.validate()
            self.assertTrue(config.output_dir.exists())

    def test_load_strategy_state_data(self) -> None:
        # 使用代表性真实文件，验证旧字段 seq/S/state/fileNames 被转换为新数据结构。
        record = load_strategy_state_data(
            STRATEGY_SEQUENCE_DIR / "031222-401.pkl",
            DEFAULT_STATE_NAMES,
        )
        self.assertEqual(record.input_file_name, "031222-401.pkl")
        self.assertIsInstance(record.token_sequence, list)
        self.assertIsInstance(record.initial_tokens, list)
        self.assertEqual(list(record.state_features.columns), list(DEFAULT_STATE_NAMES))
        self.assertTrue(record.participant_file_names[0].endswith(".pkl"))
        self.assertFalse(record.participant_ids[0].endswith(".pkl"))

    def test_load_state_dependency_graph(self) -> None:
        # StateGraph 的 G 矩阵会被转换成每个状态的条件依赖列表。
        graph = load_state_dependency_graph(STATE_GRAPH_DIR / "031222-401.pkl")
        self.assertTrue(graph.conditions_by_state)
        self.assertIsInstance(graph.conditions_by_state[0], list)


if __name__ == "__main__":
    unittest.main()
