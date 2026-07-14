"""验证 06c context 策略后验的核心统计语义。"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from LoPS.context_strategy_posterior import (
    ContextObservation,
    ContextStrategyPosteriorConfig,
    apply_best_approach_candidates,
    apply_best_energizer_candidates,
    batch_total_context_nll,
    build_observation_batch,
    build_grouped_folds,
    calculate_strategy_information_coverage,
    context_eligible_strategy_log_likelihood,
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

    def test_best_approach_can_select_farther_ghost_target(self) -> None:
        """验证 context 动作证据可以选择远鬼，而不是固定选择最近 ghost。

        输入语义：构造三个持续向右动作；近处 Ghost1 候选预测左，远处 Ghost2 候选
        预测右，并让两只 ghost 的位置逐行变化但身份保持稳定。
        输出语义：06c 选择 Ghost2，并把它的 Q 写入正式 Approach 拟合视图。
        关键约束：跨行匹配必须使用 target_id，不能使用动态位置或候选列表下标。
        """

        data = pd.DataFrame(
            {
                "action_dir": ["right", "right", "right"],
                "p1_approach_Q": [[0.0, 0.0, -np.inf, -np.inf]] * 3,
                "p1_approach_utility_k": [
                    [[1.0, 0.0, -np.inf, -np.inf], [0.0, 1.0, -np.inf, -np.inf]],
                    [[1.0, 0.0, -np.inf, -np.inf], [0.0, 1.0, -np.inf, -np.inf]],
                    [[1.0, 0.0, -np.inf, -np.inf], [0.0, 1.0, -np.inf, -np.inf]],
                ],
                "p1_approach_utility_k_norm": [
                    [[1.0, 0.0, -np.inf, -np.inf], [0.0, 1.0, -np.inf, -np.inf]],
                    [[1.0, 0.0, -np.inf, -np.inf], [0.0, 1.0, -np.inf, -np.inf]],
                    [[1.0, 0.0, -np.inf, -np.inf], [0.0, 1.0, -np.inf, -np.inf]],
                ],
                "p1_approach_utility_k_meta": [
                    [
                        {"target_id": "ghost1", "target_position": (1, 0), "min_distance": 1},
                        {"target_id": "ghost2", "target_position": (10, 0), "min_distance": 10},
                    ],
                    [
                        {"target_id": "ghost1", "target_position": (0, 0), "min_distance": 1},
                        {"target_id": "ghost2", "target_position": (9, 0), "min_distance": 9},
                    ],
                    [
                        {"target_id": "ghost1", "target_position": (-1, 0), "min_distance": 1},
                        {"target_id": "ghost2", "target_position": (8, 0), "min_distance": 8},
                    ],
                ],
            }
        )

        selected = apply_best_approach_candidates(data, [(0, 3)], "p1")

        self.assertEqual(selected.at[0, "best_approach_target_id"], "ghost2")
        self.assertEqual(selected.at[0, "best_approach_target_prob_accuracy"], 1.0)
        for row_index in range(3):
            np.testing.assert_array_equal(
                selected.at[row_index, "p1_approach_Q"],
                [0.0, 1.0, -np.inf, -np.inf],
            )

    def test_bean_boundaries_are_suppressed_only_on_event_facing_side(self) -> None:
        """验证 3-tile 窗口只删除朝向强事件一侧的普通豆边界。"""

        suppressed = suppress_bean_boundaries_near_events(
            bean_start_points={2, 11},
            bean_end_points={5, 15},
            directional_event_boundaries={0, 8, 20},
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
            directional_event_boundaries={65},
            window=3,
        )
        self.assertEqual(suppressed, {65})
        self.assertNotIn(62, suppressed)

    def test_behavioral_strong_event_suppresses_all_nearby_bean_boundaries(self) -> None:
        """验证行为强事件会对称删除前后窗口内任意类型的普通豆边界。"""

        suppressed = suppress_bean_boundaries_near_events(
            bean_start_points={7, 9, 13, 14},
            bean_end_points={6, 8, 11, 14},
            directional_event_boundaries={0, 20},
            symmetric_event_boundaries={10},
            window=3,
        )

        # 强事件 10 前后的开始点 7/9/13 和结束点 8/11 都会删除；距离为 4 的
        # 结束点 6 及开始/结束点 14 则保留。该规则不受边界类型和事件方向限制。
        self.assertEqual(suppressed, {7, 8, 9, 11, 13})

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

    def test_teammate_energizer_is_soft_but_teammate_ghost_is_hard_boundary(self) -> None:
        """验证队友 Energizer 可合并，而队友吃 ghost 是公共强边界。

        输入语义：构造10行无转向 trial，分别放置临近起点和位于中部的队友 Energizer，
        并对照队友普通豆、当前玩家 Energizer 与临近起点的队友吃 ghost。
        输出语义：队友事件在第1行时因前段长度1而合并，在第5行时保留；队友普通豆
        永不共享，当前玩家 Energizer 与任一玩家吃 ghost 始终作为硬边界保留。
        关键约束：最短段长度沿用 stay_length=4，但公共吃 ghost 边界不能参与合并。
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

        teammate_ghost = make_trial()
        teammate_ghost.loc[1, "p2_eat_ghost"] = True
        self.assertEqual(soft_teammate_event_points(teammate_ghost, "p1"), set())
        contexts, _ = build_event_context_segments(teammate_ghost, "p1", config)
        self.assertEqual(contexts, [(0, 1), (1, 10)])

    def test_turnaround_action_no_longer_splits_context(self) -> None:
        """验证持续掉头只作为段内动作，不再生成context边界。

        输入语义：构造10行无资源事件trial，前5个动作向右、后5个动作向左。
        输出语义：完整trial保持为一个context。
        关键约束：即使掉头后的反向动作持续时间超过stay_length，也不能仅凭动作反转
        切段；真正的策略变化应由行为事件或后续模型证据识别。
        """

        row_count = 10
        data = pd.DataFrame(
            {
                "row_id": list(range(row_count)),
                "DayTrial": ["01-01-test"] * row_count,
                "action_dir": ["right"] * 5 + ["left"] * 5,
                "available_dir": [True] * row_count,
                "p1_alive": [True] * row_count,
                "p1_eat_bean": [False] * row_count,
                "p1_eat_energizer": [False] * row_count,
                "p1_eat_ghost": [False] * row_count,
            }
        )

        contexts, is_stay = build_event_context_segments(
            data,
            "p1",
            DynamicStrategyFittingConfig(stay_length=4),
        )

        self.assertEqual(contexts, [(0, row_count)])
        self.assertEqual(is_stay, [False])

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
        expected_no_information = -math.log(2) - 2.0
        np.testing.assert_allclose(actual, [expected_preference, expected_no_information])

        posterior = posterior_from_log_likelihood(actual)
        self.assertAlmostEqual(float(np.sum(posterior)), 1.0)
        self.assertGreater(posterior[0], posterior[1])

    def test_no_information_penalty_does_not_depend_on_beta(self) -> None:
        """验证全零 Q 使用固定额外损失，而不是随 beta 变化的伪错误方向。

        输入语义：构造一个具有两个合法方向的全零策略，并用两个不同 beta 计算。
        输出语义：两次 log-likelihood 都等于 ``-log(2)-2``。
        关键约束：固定惩罚只作用于多于一个合法方向的无信息行。
        """

        observation = ContextObservation(
            player="p1",
            trial_name="01-01-test",
            context=(0, 1),
            is_stay=False,
            row_indices=np.asarray([0]),
            action_indices=np.asarray([0]),
            q_values=np.asarray([[[0.0, 0.0, -np.inf, -np.inf]]]),
            null_log_likelihood=-math.log(2),
        )

        low_beta = context_strategy_log_likelihood(observation, beta=0.2)
        high_beta = context_strategy_log_likelihood(observation, beta=8.0)
        np.testing.assert_allclose(low_beta, [-math.log(2)-2.0])
        np.testing.assert_allclose(high_beta, low_beta)

    def test_single_legal_direction_is_not_penalized_as_no_information(self) -> None:
        """验证只有一个合法动作时，即使 Q 无差异也保持零损失。"""

        observation = ContextObservation(
            player="p1",
            trial_name="01-01-test",
            context=(0, 1),
            is_stay=False,
            row_indices=np.asarray([0]),
            action_indices=np.asarray([0]),
            q_values=np.asarray([[[0.0, -np.inf, -np.inf, -np.inf]]]),
            null_log_likelihood=0.0,
        )

        actual = context_strategy_log_likelihood(observation, beta=2.0)
        np.testing.assert_allclose(actual, [0.0])

    def test_low_information_strategy_is_gated_without_null_competition(self) -> None:
        """验证少数命中不能让大部分时间无信息的策略获得虚高 posterior。

        输入语义：构造19个有效动作，Global 每行都有方向信息，Local 仅最后6行能够
        区分方向，复现真实 P2 80--99 context 的 6/19 coverage 结构。
        输出语义：Local coverage 低于0.5并被排除；候选数组中 Local 为负无穷，Global
        是唯一合格行为策略，因此整个 context 的 posterior 只归属于 Global。
        关键约束：无信息行仍在 coverage 分母中，不能只按6个有信息动作计算100%。
        """

        action_count = 19
        global_q = np.repeat(
            np.asarray([[1.0, 0.0, -np.inf, -np.inf]]),
            action_count,
            axis=0,
        )
        local_q = np.repeat(
            np.asarray([[0.0, 0.0, -np.inf, -np.inf]]),
            action_count,
            axis=0,
        )
        local_q[-6:, 0] = 1.0
        q_values = np.stack((global_q, local_q), axis=1)
        coverage = calculate_strategy_information_coverage(q_values)
        eligible = coverage >= 0.50
        observation = ContextObservation(
            player="p2",
            trial_name="01-01-test",
            context=(0, action_count),
            is_stay=False,
            row_indices=np.arange(action_count),
            action_indices=np.zeros(action_count, dtype=int),
            q_values=q_values,
            null_log_likelihood=-action_count * math.log(2),
            strategy_information_coverage=coverage,
            strategy_eligible=eligible,
        )

        np.testing.assert_allclose(coverage, [1.0, 6.0 / 19.0])
        np.testing.assert_array_equal(eligible, [True, False])
        candidates = context_eligible_strategy_log_likelihood(observation, beta=2.0)
        self.assertTrue(np.isfinite(candidates[0]))
        self.assertTrue(np.isneginf(candidates[1]))
        posterior = posterior_from_log_likelihood(candidates)
        np.testing.assert_allclose(posterior, [1.0, 0.0])
        self.assertTrue(np.isfinite(context_marginal_nll(observation, beta=2.0)))

    def test_posterior_accepts_ineligible_negative_infinity_candidate(self) -> None:
        """验证 posterior 将被门控候选置0，同时保留其余候选归一化。"""

        posterior = posterior_from_log_likelihood(np.asarray([-2.0, -np.inf, -1.0]))
        self.assertEqual(float(posterior[1]), 0.0)
        self.assertAlmostEqual(float(np.sum(posterior)), 1.0)
        self.assertGreater(posterior[2], posterior[0])

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

    def test_best_energizer_selection_tracks_target_position_across_rows(self) -> None:
        """验证06c按目标坐标选择并跨行匹配最佳 Energizer。

        输入语义：两个目标在候选列表中的顺序逐行交换；真实动作始终向右，第一个目标
        始终预测向右，第二个目标始终预测向左。
        输出语义：best target 必须保持为 ``(8,5)``，选中 Q 每行都预测向右。
        关键约束：候选列表下标不是稳定身份，不能因第二行顺序交换而错配目标。
        """

        first = {"target_id": (8, 5), "target_position": (8, 5), "min_distance": 12.0}
        second = {"target_id": (2, 5), "target_position": (2, 5), "min_distance": 4.0}
        right_q = [0.0, 1.0, -np.inf, -np.inf]
        left_q = [1.0, 0.0, -np.inf, -np.inf]
        data = pd.DataFrame(
            {
                "action_dir": ["right", "right"],
                "p1_energizer_Q": [[0.0, 0.0, -np.inf, -np.inf]] * 2,
                "p1_energizer_utility_k": [
                    [right_q, left_q],
                    [left_q, right_q],
                ],
                "p1_energizer_utility_k_norm": [
                    [right_q, left_q],
                    [left_q, right_q],
                ],
                "p1_energizer_utility_k_meta": [
                    [first, second],
                    [second, first],
                ],
            }
        )

        selected = apply_best_energizer_candidates(data, [(0, 2)], "p1")

        self.assertEqual(selected.at[0, "best_energizer_target_position"], (8, 5))
        self.assertEqual(selected.at[0, "best_energizer_target_prob_accuracy"], 1.0)
        np.testing.assert_array_equal(selected.at[0, "selected_energizer_Q"], right_q)
        np.testing.assert_array_equal(selected.at[1, "selected_energizer_Q"], right_q)

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
