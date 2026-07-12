"""验证 07/07c 人工策略修正的短 context 规则。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "script" / "07_revise_human_weight.py"
MODULE_NAME = "test_revise_human_weight_module"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"无法加载 07 修正脚本：{SCRIPT_PATH}")
REVISION = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = REVISION
SPEC.loader.exec_module(REVISION)


class ReviseHumanWeightTests(unittest.TestCase):
    """覆盖短 vague 段的比例门槛和并列策略优先级。"""

    def test_three_valid_actions_are_revised_without_absolute_count_limit(self) -> None:
        """验证3行全有效 context 可修正，并由 Local 赢得并列优先级。

        输入语义：构造3行连续向上动作；Global、Local、Energizer 都唯一预测向上，
        其余策略使用全零 Q 表示无信息。
        输出语义：revise_vague 应保存三个并列最高策略，recompute_strategy 再按现有
        Local > Global 优先级输出 Local。
        关键约束：有效动作比例为 1.0；测试故意只给3行，防止重新引入数量至少4的限制。
        """

        row_count = 3
        data = pd.DataFrame(
            {
                "action_dir": ["up"] * row_count,
                "trial_context": [(0, row_count)] * row_count,
                "revised_normalized_weight": pd.Series(
                    [[1 / len(REVISION.AGENTS)] * len(REVISION.AGENTS) for _ in range(row_count)],
                    dtype=object,
                ),
                "revised_prediction_correct": pd.Series([np.nan] * row_count, dtype=object),
                "predict_dir": pd.Series([np.nan] * row_count, dtype=object),
                "is_stay": [False] * row_count,
                "is_vague": [True] * row_count,
                "ifscared1": [1] * row_count,
                "ifscared2": [1] * row_count,
                "strategy": [REVISION.STRATEGY_NUMBER["vague"]] * row_count,
            }
        )
        informative_q = [-np.inf, -np.inf, 1.0, 0.0]
        uninformative_q = [-np.inf, -np.inf, 0.0, 0.0]
        informative_agents = {"global", "local", "energizer"}
        for agent in REVISION.AGENTS:
            q_value = informative_q if agent in informative_agents else uninformative_q
            data[f"{agent}_Q_norm"] = pd.Series(
                [list(q_value) for _ in range(row_count)],
                dtype=object,
            )

        REVISION.revise_vague(data, [(0, row_count)])
        expected_weight = [1, 1, 0, 0, 0, 1, 0]
        for value in data["revised_normalized_weight"]:
            self.assertEqual(value, expected_weight)
        self.assertFalse(data["is_vague"].any())

        REVISION.recompute_strategy(data, "synthetic.pkl", "01-01-test")
        self.assertEqual(
            data["strategy"].tolist(),
            [REVISION.STRATEGY_NUMBER["local"]] * row_count,
        )


if __name__ == "__main__":
    unittest.main()
