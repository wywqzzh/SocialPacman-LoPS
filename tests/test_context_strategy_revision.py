"""验证 07 人工策略修正的短 context 规则。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from LoPS import context_strategy_revision as REVISION


class ContextStrategyRevisionTests(unittest.TestCase):
    """覆盖短 vague 段的比例门槛和并列策略优先级。"""

    def test_energizer_followup_approach_accepts_exactly_point_seventy_five(self) -> None:
        """验证 Energizer 后 Approach 在相对准确率恰好 0.75 时能够修正。

        输入语义：构造4行 left 动作；Global 四行全对，Approach 前三行正确、最后一行
        预测 up，其它策略均无信息。
        输出语义：以 0.75 且包含等号调用 revise_function 后，整段写成 Approach one-hot。
        关键约束：该测试只覆盖 Energizer 后调用采用的新边界；默认调用仍保持严格
        ``>0.8``，不能借此放宽其它人工规则。
        """

        row_count = 4
        data = pd.DataFrame(
            {
                "action_dir": ["left"] * row_count,
                "trial_context": [(0, row_count)] * row_count,
                "revised_normalized_weight": pd.Series(
                    [[1 / len(REVISION.AGENTS)] * len(REVISION.AGENTS) for _ in range(row_count)],
                    dtype=object,
                ),
                "revised_prediction_correct": pd.Series([np.nan] * row_count, dtype=object),
                "predict_dir": pd.Series([np.nan] * row_count, dtype=object),
                "is_stay": [False] * row_count,
                "is_vague": [True] * row_count,
            }
        )
        no_information_q = [0.0, 0.0, -np.inf, -np.inf]
        for agent in REVISION.AGENTS:
            data[f"{agent}_Q_norm"] = pd.Series(
                [list(no_information_q) for _ in range(row_count)],
                dtype=object,
            )
        data["global_Q_norm"] = pd.Series(
            [[1.0, 0.0, -np.inf, -np.inf] for _ in range(row_count)],
            dtype=object,
        )
        data["approach_Q_norm"] = pd.Series(
            [[1.0, 0.0, -np.inf, -np.inf]] * 3
            + [[0.0, 0.0, 1.0, -np.inf]],
            dtype=object,
        )
        approach_weight = [0] * len(REVISION.AGENTS)
        approach_weight[REVISION.AGENT_INDEX["approach"]] = 1

        REVISION.revise_function(
            data,
            [(0, row_count)],
            approach_weight,
            REVISION.AGENT_INDEX["approach"],
            relative_accuracy_threshold=0.75,
            include_relative_threshold=True,
        )

        for value in data["revised_normalized_weight"]:
            self.assertEqual(value, approach_weight)
        self.assertFalse(data["is_vague"].any())

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

    def test_energizer_tie_is_resolved_by_context_end_event(self) -> None:
        """验证 Energizer 只在并列歧义中由结束边界事件决定。

        输入语义：前三行构成一个 context，Global、Local、Energizer 都完整预测向右；
        第3行是下一 context 的事件边界。
        输出语义：边界吃到 energizer 时前三行变成 Energizer；没有吃到时 Energizer
        从并列集合移除，保留 Global+Local 并由现有优先级显示 Local。
        关键约束：事件读取半开区间 end 行，不回退到 end-1，也不把事件用于修改 utility。
        """

        def make_data(boundary_event: bool) -> pd.DataFrame:
            """构造包含一个待消歧 context 和一个事件边界行的测试表。"""

            row_count = 4
            data = pd.DataFrame(
                {
                    "action_dir": ["right", "right", "right", np.nan],
                    "trial_context": [(0, 3)] * 3 + [(3, 4)],
                    "revised_normalized_weight": pd.Series(
                        [[1 / len(REVISION.AGENTS)] * len(REVISION.AGENTS) for _ in range(row_count)],
                        dtype=object,
                    ),
                    "revised_prediction_correct": pd.Series([np.nan] * row_count, dtype=object),
                    "predict_dir": pd.Series([np.nan] * row_count, dtype=object),
                    "is_stay": [False, False, False, True],
                    "is_vague": [True, True, True, False],
                    "eat_energizer": [False, False, False, boundary_event],
                    "ifscared1": [1] * row_count,
                    "ifscared2": [1] * row_count,
                    "strategy": [REVISION.STRATEGY_NUMBER["vague"]] * row_count,
                }
            )
            informative_q = [-np.inf, 1.0, -np.inf, 0.0]
            uninformative_q = [-np.inf, 0.0, -np.inf, 0.0]
            for agent in REVISION.AGENTS:
                q_value = informative_q if agent in {"global", "local", "energizer"} else uninformative_q
                data[f"{agent}_Q_norm"] = pd.Series(
                    [list(q_value) for _ in range(row_count)],
                    dtype=object,
                )
            return data

        eaten = make_data(True)
        REVISION.revise_energizer_by_outcome(eaten)
        energizer_weight = [0] * len(REVISION.AGENTS)
        energizer_weight[REVISION.AGENT_INDEX["energizer"]] = 1
        for value in eaten.loc[0:2, "revised_normalized_weight"]:
            self.assertEqual(value, energizer_weight)

        not_eaten = make_data(False)
        REVISION.revise_energizer_by_outcome(not_eaten)
        expected_other_ties = [1, 1, 0, 0, 0, 0, 0]
        for value in not_eaten.loc[0:2, "revised_normalized_weight"]:
            self.assertEqual(value, expected_other_ties)
        REVISION.recompute_strategy(not_eaten, "synthetic.pkl", "01-01-test")
        self.assertEqual(
            not_eaten.loc[0:2, "strategy"].tolist(),
            [REVISION.STRATEGY_NUMBER["local"]] * 3,
        )

    def test_energizer_event_accepts_near_best_non_tied_accuracy(self) -> None:
        """验证成功事件允许准确率达标但未与最佳策略精确并列的 Energizer。

        输入语义：十个有效动作中 Global 命中八次、Energizer 命中七次，下一行记录
        吃到 energizer；因此 Energizer 的绝对准确率为 0.70，相对准确率为 0.875。
        输出语义：整个有效 context 被修正为 Energizer。
        关键约束：测试覆盖绝对准确率 0.70 的等号边界，并验证非精确并列也可触发。
        """

        action_count = 10
        row_count = action_count + 1
        data = pd.DataFrame(
            {
                "action_dir": ["right"] * action_count + [np.nan],
                "trial_context": [(0, action_count)] * action_count + [(action_count, row_count)],
                "revised_normalized_weight": pd.Series(
                    [[1 / len(REVISION.AGENTS)] * len(REVISION.AGENTS) for _ in range(row_count)],
                    dtype=object,
                ),
                "revised_prediction_correct": pd.Series([np.nan] * row_count, dtype=object),
                "predict_dir": pd.Series([np.nan] * row_count, dtype=object),
                "is_stay": [False] * action_count + [True],
                "is_vague": [True] * action_count + [False],
                "eat_energizer": [False] * action_count + [True],
                "ifscared1": [1] * row_count,
                "ifscared2": [1] * row_count,
                "strategy": [REVISION.STRATEGY_NUMBER["vague"]] * row_count,
            }
        )
        correct_q = [-np.inf, 1.0, -np.inf, 0.0]
        wrong_q = [1.0, 0.0, -np.inf, 0.0]
        no_information_q = [-np.inf, 0.0, -np.inf, 0.0]
        for agent in REVISION.AGENTS:
            if agent == "global":
                values = [correct_q] * 8 + [wrong_q] * 2 + [no_information_q]
            elif agent == "energizer":
                values = [correct_q] * 7 + [wrong_q] * 3 + [no_information_q]
            else:
                values = [no_information_q] * row_count
            data[f"{agent}_Q_norm"] = pd.Series(
                [list(value) for value in values],
                dtype=object,
            )

        REVISION.revise_energizer_by_outcome(data)
        expected_weight = [0] * len(REVISION.AGENTS)
        expected_weight[REVISION.AGENT_INDEX["energizer"]] = 1
        for value in data.loc[0 : action_count - 1, "revised_normalized_weight"]:
            self.assertEqual(value, expected_weight)


if __name__ == "__main__":
    unittest.main()
