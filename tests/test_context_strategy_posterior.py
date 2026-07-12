"""验证 06c context 策略后验的核心统计语义。"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from LoPS.context_strategy_posterior import (
    ContextObservation,
    ContextStrategyPosteriorConfig,
    batch_total_context_nll,
    build_observation_batch,
    build_grouped_folds,
    context_marginal_nll,
    context_strategy_log_likelihood,
    fit_full_beta_models,
    normalize_legal_q,
    posterior_from_log_likelihood,
)
from LoPS.dynamic_strategy_event_context import (
    apply_best_global_candidates,
    build_event_context_segments,
    hard_boundary_points,
    suppress_bean_boundaries_near_events,
    suppress_stay_ranges_near_ghost,
    soft_teammate_event_points,
)
from LoPS.dynamic_strategy_fitting import DynamicStrategyFittingConfig


class ContextStrategyPosteriorTests(unittest.TestCase):
    """覆盖 06c 归一化、概率模型、Global 选择和 beta 模型选择。"""

    def test_normalize_legal_q_ignores_walls_and_maps_to_unit_interval(self) -> None:
        """验证合法方向 Min-Max 不让墙方向参与最大值和最小值。"""

        actual = normalize_legal_q([-2.0, 0.0, 2.0, -np.inf])
        np.testing.assert_allclose(actual[:3], [0.0, 0.5, 1.0])
        self.assertTrue(np.isneginf(actual[3]))

    def test_bean_boundaries_are_suppressed_only_on_event_facing_side(self) -> None:
        """验证 3-tile 窗口只删除朝向强事件一侧的普通豆边界。"""

        suppressed = suppress_bean_boundaries_near_events(
            bean_start_points={2, 11},
            bean_end_points={5, 15},
            event_boundaries={0, 8, 20},
            window=3,
        )
        # 开始点 2/11 分别紧随强事件 0/8，结束点 5 紧邻其后的强事件 8；结束点 15
        # 离后续强事件 20 已超过窗口，因此保留。
        self.assertEqual(suppressed, {2, 5, 11})
        # 函数只返回 bean 边界，强事件 0/8/20 本身绝不会进入待删除集合。
        self.assertTrue(suppressed.isdisjoint({0, 8, 20}))

    def test_future_stay_does_not_suppress_earlier_bean_start(self) -> None:
        """验证未来 stay 只替代吃豆结束点，不会误删此前的吃豆开始点。"""

        suppressed = suppress_bean_boundaries_near_events(
            bean_start_points={62},
            bean_end_points={65},
            event_boundaries={65},
            window=3,
        )
        self.assertEqual(suppressed, {65})
        self.assertNotIn(62, suppressed)

    def test_hard_boundaries_use_event_rows_instead_of_previous_actions(self) -> None:
        """验证 energizer 只在事件行切段，邻近 bean 首尾事件按规则删除。"""

        trial_data = pd.DataFrame(
            {
                "action_dir": ["right"] * 10,
                "p1_alive": [True] * 10,
                "p1_eat_bean": [False, False, False, True, True, False, False, False, False, False],
                "p1_eat_energizer": [False, False, False, False, False, True, False, False, False, False],
                "p1_eat_ghost": [False] * 10,
            }
        )
        boundaries = hard_boundary_points(
            trial_data,
            "p1",
            stay_length=4,
            bean_event_suppression_window=3,
            ghost_stay_suppression_window=5,
        )

        # bean 开始事件 3 紧随 trial 起点强事件 0，因此删除；bean 结束事件 4 紧邻
        # 其后的 energizer 5，也被删除。energizer 仍只使用真实事件行 5。
        self.assertEqual(boundaries, {0, 5, 10})

    def test_context_does_not_split_on_distance_to_remaining_beans(self) -> None:
        """验证位置和豆集合变化本身不会再产生 10 步 local-range context。"""

        data = pd.DataFrame(
            {
                "row_id": list(range(6)),
                "DayTrial": ["01-01-test"] * 6,
                "action_dir": ["right"] * 6,
                "available_dir": [True] * 6,
                "p1_alive": [True] * 6,
                "p1_pos": [(0, 0), (1, 0), (2, 0), (20, 20), (21, 20), (22, 20)],
                "beans": [[(1, 0)], [(2, 0)], [], [(29, 35)], [(29, 35)], []],
                "p1_eat_bean": [False] * 6,
                "p1_eat_energizer": [False] * 6,
                "p1_eat_ghost": [False] * 6,
            }
        )
        contexts, is_stay = build_event_context_segments(
            data,
            "p1",
            DynamicStrategyFittingConfig(stay_length=4),
        )

        self.assertEqual(contexts, [(0, 6)])
        self.assertEqual(is_stay, [False])

    def test_teammate_high_impact_events_are_mergeable_soft_boundaries(self) -> None:
        """验证队友高影响事件只在不会制造短段时切分当前玩家 context。

        输入语义：构造10行无转向 trial，分别放置临近起点和位于中部的队友 energizer，
        并对照队友普通豆与当前玩家自己的 energizer。
        输出语义：队友事件在第1行时因前段长度1而合并，在第5行时保留；队友普通豆
        永不共享，当前玩家自己的 energizer 始终作为硬边界保留。
        关键约束：最短段长度沿用 stay_length=4，软边界合并不能跨越自己的硬边界。
        """

        def make_trial() -> pd.DataFrame:
            """构造包含双人私有事件列的最小 context 测试表。"""

            row_count = 10
            return pd.DataFrame(
                {
                    "row_id": list(range(row_count)),
                    "DayTrial": ["01-01-test"] * row_count,
                    "action_dir": ["right"] * row_count,
                    "available_dir": [True] * row_count,
                    "p1_alive": [True] * row_count,
                    "p1_eat_bean": [False] * row_count,
                    "p1_eat_energizer": [False] * row_count,
                    "p1_eat_ghost": [False] * row_count,
                    "p2_eat_bean": [False] * row_count,
                    "p2_eat_energizer": [False] * row_count,
                    "p2_eat_ghost": [False] * row_count,
                }
            )

        config = DynamicStrategyFittingConfig(stay_length=4)

        near_start = make_trial()
        near_start.loc[1, "p2_eat_energizer"] = True
        self.assertEqual(soft_teammate_event_points(near_start, "p1"), {1})
        contexts, _ = build_event_context_segments(near_start, "p1", config)
        self.assertEqual(contexts, [(0, 10)])

        middle = make_trial()
        middle.loc[5, "p2_eat_energizer"] = True
        contexts, _ = build_event_context_segments(middle, "p1", config)
        self.assertEqual(contexts, [(0, 5), (5, 10)])

        teammate_bean = make_trial()
        teammate_bean.loc[5, "p2_eat_bean"] = True
        self.assertEqual(soft_teammate_event_points(teammate_bean, "p1"), set())
        contexts, _ = build_event_context_segments(teammate_bean, "p1", config)
        self.assertEqual(contexts, [(0, 10)])

        own_energizer = make_trial()
        own_energizer.loc[1, "p1_eat_energizer"] = True
        contexts, _ = build_event_context_segments(own_energizer, "p1", config)
        self.assertEqual(contexts, [(0, 1), (1, 10)])

    def test_stay_event_is_suppressed_near_private_eat_ghost(self) -> None:
        """验证 ghost 前后 5 tile 的 stay 整段取消边界，但远处 stay 保留。"""

        suppressed = suppress_stay_ranges_near_ghost(
            stay_ranges=[(10, 15), (30, 40), (50, 55)],
            eat_ghost_indices=[16, 35],
            window=5,
        )
        # ghost=16 距第一段最后一行 14 为 2；ghost=35 位于第二段内部。
        self.assertEqual(suppressed, {(10, 15), (30, 40)})
        self.assertNotIn((50, 55), suppressed)

    def test_normalize_legal_q_equal_values_becomes_uninformative(self) -> None:
        """验证合法方向全相等时归一化为全零而非伪造最大方向。"""

        actual = normalize_legal_q([4.0, 4.0, -np.inf, 4.0])
        np.testing.assert_array_equal(actual[[0, 1, 3]], [0.0, 0.0, 0.0])
        self.assertTrue(np.isneginf(actual[2]))

    def test_context_likelihood_uses_softmax_and_mask(self) -> None:
        """用可手算的两方向动作验证 softmax context likelihood。"""

        q_values = np.asarray(
            [
                [
                    [1.0, 0.0, -np.inf, -np.inf],
                    [0.0, 0.0, -np.inf, -np.inf],
                ]
            ]
        )
        observation = ContextObservation(
            player="p1",
            trial_name="01-01-test",
            context=(0, 1),
            is_stay=False,
            row_indices=np.asarray([0]),
            action_indices=np.asarray([0]),
            q_values=q_values,
            null_log_likelihood=-math.log(2),
        )
        actual = context_strategy_log_likelihood(observation, beta=2.0)
        expected_preference = 2.0 - np.log(np.exp(2.0) + 1.0)
        np.testing.assert_allclose(actual, [expected_preference, -math.log(2)])

        posterior = posterior_from_log_likelihood(actual)
        self.assertAlmostEqual(float(np.sum(posterior)), 1.0)
        self.assertGreater(posterior[0], posterior[1])

    def test_vectorized_batch_loss_matches_context_loop(self) -> None:
        """验证批量优化只加速实现，不改变逐 context NLL。"""

        observations: list[ContextObservation] = []
        for context_index, actions in enumerate((np.asarray([0, 0]), np.asarray([1]))):
            q_values = np.repeat(
                np.asarray(
                    [
                        [1.0, 0.0, -np.inf, -np.inf],
                        [0.0, 1.0, -np.inf, -np.inf],
                    ]
                )[None, :, :],
                len(actions),
                axis=0,
            )
            observations.append(
                ContextObservation(
                    player="p1",
                    trial_name=f"trial-{context_index}",
                    context=(0, len(actions)),
                    is_stay=False,
                    row_indices=np.arange(len(actions)),
                    action_indices=actions,
                    q_values=q_values,
                    null_log_likelihood=-len(actions) * math.log(2),
                )
            )
        expected = sum(context_marginal_nll(item, beta=1.7) for item in observations)
        actual = batch_total_context_nll(build_observation_batch(observations), beta=1.7)
        self.assertAlmostEqual(actual, expected)

    def test_best_global_selection_matches_06b_probability_accuracy(self) -> None:
        """验证 06c 复用的 06b 规则会选择真实动作解释率更高的 cluster。"""

        meta = [
            {
                "cluster_id": 0,
                "cluster_size": 2,
                "resource_positions": [(1, 1), (2, 1)],
                "min_distance": 3,
            },
            {
                "cluster_id": 1,
                "cluster_size": 1,
                "resource_positions": [(9, 9)],
                "min_distance": 2,
            },
        ]
        data = pd.DataFrame(
            {
                "action_dir": ["right", "right"],
                "global_Q": [[0.0, 0.0, -np.inf, -np.inf]] * 2,
                "global_Q_norm": [[0.0, 0.0, -np.inf, -np.inf]] * 2,
                "p1_global_Q": [[0.0, 0.0, -np.inf, -np.inf]] * 2,
                "p1_global_Q_norm": [[0.0, 0.0, -np.inf, -np.inf]] * 2,
                "p1_global_utility_k": [
                    [[0.0, 2.0, -np.inf, -np.inf], [1.0, 0.0, -np.inf, -np.inf]],
                    [[0.0, 2.0, -np.inf, -np.inf], [1.0, 0.0, -np.inf, -np.inf]],
                ],
                "p1_global_utility_k_norm": [
                    [[0.0, 1.0, -np.inf, -np.inf], [1.0, 0.0, -np.inf, -np.inf]],
                    [[0.0, 1.0, -np.inf, -np.inf], [1.0, 0.0, -np.inf, -np.inf]],
                ],
                "p1_global_utility_k_meta": [meta, meta],
            }
        )
        selected = apply_best_global_candidates(data, [(0, 2)], "p1")
        self.assertEqual(selected.at[0, "best_global_cluster_id"], 0)
        self.assertEqual(selected.at[0, "best_global_cluster_prob_accuracy"], 1.0)
        np.testing.assert_array_equal(selected.at[1, "global_Q"][:2], [0.0, 2.0])

    def test_bic_selects_shared_and_separate_beta_on_synthetic_data(self) -> None:
        """构造同质和异质玩家动作，验证 BIC 能选择一个或两个 beta。"""

        config = ContextStrategyPosteriorConfig(
            agents=("local",),
            beta_min=0.05,
            beta_max=10.0,
            beta_grid_size=31,
        )

        def make_observation(player: str, actions: np.ndarray) -> ContextObservation:
            """构造所有行都偏好 left 的单策略 context。"""

            row_q = np.asarray([[1.0, 0.0, -np.inf, -np.inf]])
            q_values = np.repeat(row_q[None, :, :], len(actions), axis=0)
            return ContextObservation(
                player=player,
                trial_name=f"trial-{player}",
                context=(0, len(actions)),
                is_stay=False,
                row_indices=np.arange(len(actions)),
                action_indices=actions,
                q_values=q_values,
                null_log_likelihood=-len(actions) * math.log(2),
            )

        all_left = np.zeros(80, dtype=int)
        shared = fit_full_beta_models(
            {
                "p1": [make_observation("p1", all_left)],
                "p2": [make_observation("p2", all_left)],
            },
            config,
        )
        self.assertEqual(shared["selected_model"], "shared")

        alternating = np.tile([0, 1], 40)
        separate = fit_full_beta_models(
            {
                "p1": [make_observation("p1", all_left)],
                "p2": [make_observation("p2", alternating)],
            },
            config,
        )
        self.assertEqual(separate["selected_model"], "separate")
        self.assertGreater(separate["separate_beta"]["p1"], separate["separate_beta"]["p2"])

    def test_grouped_folds_keep_both_players_of_trial_together(self) -> None:
        """验证 fold 映射只由 DayTrial 决定，不会拆开双人 context。"""

        observations: list[ContextObservation] = []
        q_values = np.asarray([[[1.0, 0.0, -np.inf, -np.inf]]])
        for trial in ("01-01", "02-01", "03-01"):
            for player in ("p1", "p2"):
                observations.append(
                    ContextObservation(
                        player=player,
                        trial_name=trial,
                        context=(0, 1),
                        is_stay=False,
                        row_indices=np.asarray([0]),
                        action_indices=np.asarray([0]),
                        q_values=q_values,
                        null_log_likelihood=-math.log(2),
                    )
                )
        mapping = build_grouped_folds(observations, fold_count=5, random_seed=7)
        self.assertEqual(set(mapping), {"01-01", "02-01", "03-01"})
        self.assertEqual(len(set(mapping.values())), 3)


if __name__ == "__main__":
    unittest.main()
