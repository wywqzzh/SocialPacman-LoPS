"""generate_grammar 基础能力测试。

覆盖 token 表示、路径配置校验、策略状态数据读取和状态依赖图读取。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from LoPS.generate_grammar.config import (
    DEFAULT_STATE_NAMES,
    GenerateGrammarConfig,
)
from LoPS.generate_grammar.data import load_state_dependency_graph, load_strategy_state_data
from LoPS.generate_grammar.token import (
    combine_tokens,
    format_token,
    split_token,
    token_length,
    tokens_share_base_token,
)
from tests.generate_grammar_fixtures import STATE_GRAPH_DIR, STRATEGY_SEQUENCE_DIR


class GenerateGrammarFoundationTest(unittest.TestCase):
    """覆盖 generate_grammar 基础设施的稳定性。

    这些测试检查路径配置、token 表示和 pickle 数据读取能力，确保上层
    scoring、grammar 和 pipeline 测试拥有可靠输入。
    """

    def test_token_helpers(self) -> None:
        """验证 token 拆分、组合、长度和基础 token 共享判断。"""
        # 复合 token 必须按基础 token 拆分和组合，不能按字符串字符数量理解。
        self.assertEqual(split_token("G"), ["G"])
        self.assertEqual(split_token("G-L-E-A"), ["G", "L", "E", "A"])
        self.assertEqual(format_token(["G", "L"]), "G-L")
        self.assertEqual(combine_tokens("G-L", "E-A"), "G-L-E-A")
        self.assertEqual(token_length("G-L-E-A"), 4)
        self.assertTrue(tokens_share_base_token("G-L", "L-E"))
        self.assertFalse(tokens_share_base_token("G-L", "E-A"))

    def test_config_validates_explicit_project_data_inputs(self) -> None:
        """验证配置对象会接受显式项目数据目录并创建输出目录。"""
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
        """验证策略序列 pickle 会被读取为正式数据记录对象。"""
        # 使用代表性真实文件，检查序列、状态特征和参与者文件名等核心字段的数据形态。
        record = load_strategy_state_data(
            STRATEGY_SEQUENCE_DIR / "031222-401-03-Dec-2022-1.pkl",
            DEFAULT_STATE_NAMES,
        )
        self.assertEqual(record.input_file_name, "031222-401-03-Dec-2022-1.pkl")
        self.assertIsInstance(record.token_sequence, list)
        self.assertIsInstance(record.initial_tokens, list)
        self.assertEqual(list(record.state_features.columns), list(DEFAULT_STATE_NAMES))
        self.assertTrue(record.participant_file_names[0].endswith(".pkl"))
        self.assertFalse(record.participant_ids[0].endswith(".pkl"))

    def test_load_state_dependency_graph(self) -> None:
        """验证状态依赖图 pickle 会生成按状态索引组织的条件依赖列表。"""
        # StateGraph 的 G 矩阵会被转换成每个状态的条件依赖列表。
        graph = load_state_dependency_graph(STATE_GRAPH_DIR / "031222-401-03-Dec-2022-1.pkl")
        self.assertTrue(graph.conditions_by_state)
        self.assertIsInstance(graph.conditions_by_state[0], list)


if __name__ == "__main__":
    unittest.main()
