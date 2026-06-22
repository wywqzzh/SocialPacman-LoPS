"""generate_grammar 文件级流水线测试。

覆盖输入数据准备、状态对齐和单文件结构化输出。
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
from LoPS.generate_grammar.pipeline import prepare_strategy_state_data, process_strategy_state_file
from tests.generate_grammar_fixtures import STATE_GRAPH_DIR, STRATEGY_SEQUENCE_DIR


class GenerateGrammarPipelineTest(unittest.TestCase):
    """覆盖 generate_grammar 的文件级 pipeline 编排行为。"""

    def test_prepare_strategy_state_data_removes_n_and_aligns_state_features(self) -> None:
        """验证预处理会删除 N token 并同步对齐状态特征行。"""
        # prepare 阶段必须保证 token_sequence 与 state_features 等长，否则后续状态条件会错位。
        record = load_strategy_state_data(
            STRATEGY_SEQUENCE_DIR / "031222-401-03-Dec-2022-1.pkl",
            DEFAULT_STATE_NAMES,
        )
        state_dependencies = load_state_dependency_graph(STATE_GRAPH_DIR / "031222-401-03-Dec-2022-1.pkl")

        prepared = prepare_strategy_state_data(record, state_dependencies)

        self.assertNotIn("N", prepared.token_sequence)
        self.assertEqual(len(prepared.token_sequence), len(prepared.state_features))
        self.assertTrue(len(prepared.n_positions) > 0)

    def test_process_strategy_state_file_returns_structured_output_only(self) -> None:
        """验证单文件处理返回结构化输出并包含核心分区字段。"""
        # 单文件处理不写真实输出目录，使用临时目录配置只验证内存结果结构。
        with tempfile.TemporaryDirectory() as temp_dir:
            config = GenerateGrammarConfig(
                strategy_sequence_dir=STRATEGY_SEQUENCE_DIR,
                state_graph_dir=STATE_GRAPH_DIR,
                output_dir=Path(temp_dir),
            )
            progress_events = []

            def capture_progress(event: str, payload: dict[str, object]) -> None:
                """收集单文件处理过程事件，便于验证运行脚本可打印学习进度。"""

                progress_events.append((event, dict(payload)))

            output = process_strategy_state_file("031222-401-03-Dec-2022-1.pkl", config, progress_callback=capture_progress)

        # 核心 pipeline 返回面向正式模块的结构化分区，格式转换由验证脚本单独处理。
        self.assertEqual(set(output.keys()), {"source", "parameters", "grammar", "parsed", "skip_gram"})
        structured = output
        self.assertEqual(set(structured.keys()), {"source", "parameters", "grammar", "parsed", "skip_gram"})
        self.assertIn("participant_file_names", structured["source"])
        self.assertIn("participant_ids", structured["source"])
        self.assertIn("original_sequence", structured["parsed"])
        self.assertTrue(structured["grammar"])
        event_names = [event for event, _ in progress_events]
        self.assertIn("file_start", event_names)
        self.assertIn("file_prepared", event_names)
        self.assertIn("learn_start", event_names)
        self.assertIn("learn_iteration", event_names)
        self.assertIn("learn_finished", event_names)
        self.assertIn("skip_gram", event_names)
        self.assertIn("file_finished", event_names)
        # 所有单文件事件都应携带输入文件名，运行脚本才能把过程信息归属到具体被试。
        self.assertTrue(all(payload["input_file_name"] == "031222-401-03-Dec-2022-1.pkl" for _, payload in progress_events))


if __name__ == "__main__":
    unittest.main()
