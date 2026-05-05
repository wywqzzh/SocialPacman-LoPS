from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from LoPS.generate_grammar.config import GenerateGrammarConfig
from LoPS.generate_grammar.pipeline import process_strategy_state_file
from script.validate_generate_grammar import compare_legacy_dict, compare_values
from tests.generate_grammar_fixtures import BASELINE_GRAMMAR_DIR, STATE_GRAPH_DIR, STRATEGY_SEQUENCE_DIR


class GenerateGrammarValidationTest(unittest.TestCase):
    # validation 测试锁定验证脚本的精确比较语义，不允许后续引入默认数值容差。
    def test_compare_values_accepts_equal_list_array_and_dataframe(self) -> None:
        # 相同 list、ndarray、DataFrame 应返回空差异列表。
        old_value = [
            "sets",
            np.array([1, 2, 3]),
            pd.DataFrame({"state": [1, 2]}),
        ]
        new_value = [
            "sets",
            np.array([1, 2, 3]),
            pd.DataFrame({"state": [1, 2]}),
        ]

        self.assertEqual(compare_values(old_value, new_value, "root"), [])

    def test_compare_values_reports_path_for_different_values(self) -> None:
        # 差异报告必须包含精确 key path，便于定位旧新输出不一致的位置。
        differences = compare_values({"pro": [1.0]}, {"pro": [2.0]}, "031222-401.pkl")

        self.assertTrue(differences)
        self.assertIn("031222-401.pkl.pro[0]", differences[0])

    def test_compare_legacy_dict_reports_missing_key(self) -> None:
        # 旧 pickle 中存在的 key 如果在新 legacy 缺失，必须明确报告。
        differences = compare_legacy_dict({"sets": [], "pro": []}, {"sets": []}, "031222-401.pkl")

        self.assertEqual(differences, ["031222-401.pkl.pro: missing key"])

    def test_representative_output_contains_all_legacy_baseline_keys(self) -> None:
        # 代表性真实文件检查新 legacy 至少覆盖旧基准中的全部字段。
        with tempfile.TemporaryDirectory() as temp_dir:
            config = GenerateGrammarConfig(
                strategy_sequence_dir=STRATEGY_SEQUENCE_DIR,
                state_graph_dir=STATE_GRAPH_DIR,
                output_dir=Path(temp_dir),
            )
            output = process_strategy_state_file("031222-401.pkl", config)

        old_output = pd.read_pickle(BASELINE_GRAMMAR_DIR / "031222-401.pkl")
        legacy = output["legacy"]
        for key in old_output.keys():
            self.assertIn(key, legacy)


if __name__ == "__main__":
    unittest.main()
