"""验证 05 utility 的 ghost 状态语义和 raw Q 不变性。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from LoPS.calculate_utility.processing import (
    CalculateUtilityConfig,
    add_temporary_arrive_direction,
    append_normalized_q_columns,
    build_player_alive_mask,
    build_player_view,
    build_utility_estimation_input,
    cluster_resources_by_distance,
    correct_unavailable_q_values,
    energizer_target_q_for_row,
    global_cluster_q_for_row,
    load_calculate_utility_maps,
    make_evade_q_non_negative,
)
from LoPS.hierarchical_utility import MapData, Q_COLUMNS, UtilityConfig, estimate_utility_for_dataframe
from LoPS.hierarchical_utility.strategies import _status_value


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_TILE_FILE = (
    PROJECT_ROOT
    / "data"
    / "04_corrected_tile_data"
    / "comp"
    / "10001-10022-2025-07-15-JJJ-1.pkl"
)
CONSTANT_DIR = PROJECT_ROOT / "data" / "constant_data"
TARGET_TRIAL = "02-01-10001-10022-2025-07-15-JJJ"
EVADE_TARGET_TRIAL = "01-01-10001-10022-2025-07-15-JJJ"
EVADE_TARGET_FRAME_ID = 1407


class CalculateUtilityStatusAndNormalizationTests(unittest.TestCase):
    """覆盖状态类型修复、真实 utility 回归样例和归一化数组隔离。"""

    def test_cluster_global_allows_one_empty_tile_but_not_two(self) -> None:
        """验证资源聚类只允许两个豆子之间存在至多一个空格。

        输入语义：使用四格直线地图，分别构造地图距离为 2 和 3 的两个资源点。
        输出语义：距离 2 的资源属于同一连通分量，距离 3 的资源保持为两个候选团。
        关键约束：距离按移动边数计算，因此距离 2 对应一个中间空格，距离 3 对应
        两个中间空格；测试使用默认阈值，避免配置默认值以后悄然回退。
        """

        positions = [(index, 0) for index in range(4)]
        adjacent_map = {
            position: {
                "left": (position[0] - 1, 0) if position[0] > 0 else np.nan,
                "right": (position[0] + 1, 0) if position[0] < 3 else np.nan,
                "up": np.nan,
                "down": np.nan,
            }
            for position in positions
        }
        distance_map = {
            source: {target: abs(source[0] - target[0]) for target in positions}
            for source in positions
        }
        map_data = MapData(adjacent_map, distance_map, {1: 2, 2: 4, 8: 8, 9: 8})
        threshold = CalculateUtilityConfig().global_cluster_distance_threshold

        one_empty_tile = cluster_resources_by_distance([(0, 0), (2, 0)], map_data, threshold)
        two_empty_tiles = cluster_resources_by_distance([(0, 0), (3, 0)], map_data, threshold)

        self.assertEqual(threshold, 2)
        self.assertEqual(one_empty_tile, [{(0, 0), (2, 0)}])
        self.assertEqual(two_empty_tiles, [{(0, 0)}, {(3, 0)}])

    def test_energizer_target_utility_has_no_search_radius(self) -> None:
        """验证远处 Energizer 仍按最短路距离减少量提供方向信息。

        输入语义：构造一条长度为6的直线，Pacman 在起点，唯一 energizer 在5步外。
        输出语义：向右的 raw utility 为正，目标 meta 使用稳定坐标且候选只有一个。
        关键约束：测试距离故意小于真实地图但不传任何 depth；目标导向定义本身不得
        再读取旧 ``energizer_depth`` 或因超出固定半径而返回全0。
        """

        positions = [(index, 0) for index in range(6)]
        adjacent_map = {
            position: {
                "left": (position[0] - 1, 0) if position[0] > 0 else np.nan,
                "right": (position[0] + 1, 0) if position[0] < 5 else np.nan,
                "up": np.nan,
                "down": np.nan,
            }
            for position in positions
        }
        distance_map = {
            source: {target: abs(source[0] - target[0]) for target in positions}
            for source in positions
        }
        map_data = MapData(adjacent_map, distance_map, {1: 2, 2: 4, 8: 8, 9: 8})
        row = pd.Series({"pacmanPos": (0, 0), "energizers": [(5, 0)]})

        raw_matrix, normalized_matrix, metadata = energizer_target_q_for_row(
            row,
            map_data,
            adjacent_map,
        )

        self.assertEqual(len(raw_matrix), 1)
        self.assertEqual(metadata[0]["target_position"], (5, 0))
        self.assertEqual(metadata[0]["min_distance"], 5.0)
        self.assertEqual(raw_matrix[0][1], 1.0)
        self.assertEqual(normalized_matrix[0][1], 1.0)
        self.assertTrue(np.isneginf(raw_matrix[0][0]))

    def test_tunnel_side_beans_are_not_clustered_across_two_empty_tiles(self) -> None:
        """验证 tunnel 两侧豆子不会跨越坐标 0 和 29 合并。

        输入语义：读取正式地图，聚类 tunnel 两侧的 ``(1,18)`` 与 ``(28,18)``。
        输出语义：两点分别形成独立候选团。
        关键约束：地图仍保留 ``(0,18)`` 与 ``(29,18)`` 的移动连通；这里只验证两颗
        豆子的最短路为 3，超过默认聚类阈值 2，不修改 tunnel 行为拓扑。
        """

        map_data, _ = load_calculate_utility_maps(CONSTANT_DIR)
        threshold = CalculateUtilityConfig().global_cluster_distance_threshold
        clusters = cluster_resources_by_distance([(1, 18), (28, 18)], map_data, threshold)

        self.assertEqual(clusters, [{(1, 18)}, {(28, 18)}])

    def test_default_evade_depth_is_six_tiles(self) -> None:
        """验证正式 utility 默认只搜索 6 步内的 Evade 威胁。

        输入语义：直接构造默认 UtilityConfig，不通过命令行覆盖参数。
        输出语义：Blinky/Clyde 共用的 evade_depth 必须为 6。
        关键约束：该测试只约束正式 hierarchical utility；独立 05b range utility
        仍保留自己的实验半径配置。
        """

        self.assertEqual(UtilityConfig().evade_depth, 6)

    def test_default_approach_depth_is_twenty_tiles(self) -> None:
        """验证正式 utility 默认使用20步 Approach 搜索深度。

        输入语义：直接构造默认 UtilityConfig，不通过运行脚本覆盖参数。
        输出语义：Blinky/Clyde 共用的 approach_depth 必须为20。
        关键约束：该默认值只改变正式 hierarchical Approach，不修改 Evade 或独立
        05b range utility 的实验半径。
        """

        self.assertEqual(UtilityConfig().approach_depth, 20)

    def test_default_approach_discount_factor_is_point_nine_five(self) -> None:
        """验证正式 Approach 最佳命中路径使用较缓的 0.95 距离衰减。"""

        self.assertEqual(UtilityConfig().approach_discount_factor, 0.95)

    def test_real_p1_approach_prefers_distance_reducing_actions(self) -> None:
        """验证 P1 追第二只 scared ghost 时，最佳衰减路径修复四个旧误判帧。

        输入语义：读取 03-01 中旧路径均值曾误判的 0-based 帧 23、25、26、29。
        输出语义：每帧真实动作的 Approach Q 都严格高于旧误判方向。
        关键约束：该回归同时约束距离衰减和最大叶聚合；若恢复路径均值，至少一个断言
        会失败。frame_id 用于跨整个 04 文件稳定定位对应 tile。
        """

        checks = (
            (6165, 2, 3),
            (6200, 0, 3),
            (6214, 0, 1),
            (6251, 0, 3),
        )
        for frame_id, actual_direction, old_prediction in checks:
            utility = self._estimate_real_player_frame(
                player="p1",
                trial="03-01-10001-10022-2025-07-15-JJJ",
                frame_id=frame_id,
            )
            approach_q = np.asarray(utility.at[0, "approach_Q"], dtype=float)
            self.assertGreater(
                approach_q[actual_direction],
                approach_q[old_prediction],
                msg=f"frame_id={frame_id} Approach 仍偏向旧误判方向",
            )

    def test_default_local_discount_factor_is_point_nine(self) -> None:
        """验证 Local 最佳路径默认使用0.90逐步奖励衰减。

        输入语义：直接读取默认 UtilityConfig。
        输出语义：local_discount_factor 必须为0.90。
        关键约束：该参数只作用于 Local 路径资源奖励，不改变其它路径策略。
        """

        self.assertEqual(UtilityConfig().local_discount_factor, 0.90)

    def test_status_value_distinguishes_finite_float_from_missing(self) -> None:
        """验证有限整值 float 保留状态含义，只有真正缺失值映射为 0。"""

        for status in range(1, 6):
            self.assertEqual(_status_value(status), status)
            self.assertEqual(_status_value(np.int8(status)), status)
            self.assertEqual(_status_value(float(status)), status)
            self.assertEqual(_status_value(np.float64(status)), status)
        self.assertEqual(_status_value(None), 0)
        self.assertEqual(_status_value(np.nan), 0)

        with self.assertRaises(ValueError):
            _status_value(3.5)
        with self.assertRaises(ValueError):
            _status_value(np.inf)
        with self.assertRaises(TypeError):
            _status_value(True)

    def test_utility_input_keeps_valid_statuses_as_integers(self) -> None:
        """验证 utility 入口接受有限整值状态，但输出一定是整数列。"""

        data = pd.DataFrame(
            {
                "DayTrial": ["01-01-test", "01-01-test"],
                "action_dir": ["left", "right"],
                "ifscared1": [3.0, 4.0],
                "ifscared2": [np.int8(1), np.int8(5)],
            }
        )
        actual = build_utility_estimation_input(data)
        self.assertTrue(pd.api.types.is_integer_dtype(actual["ifscared1"]))
        self.assertTrue(pd.api.types.is_integer_dtype(actual["ifscared2"]))
        self.assertEqual(actual["ifscared1"].tolist(), [3, 4])
        self.assertEqual(actual["ifscared2"].tolist(), [1, 5])

    def test_utility_input_rejects_missing_or_fractional_status(self) -> None:
        """验证脏状态在进入搜索树前立即报错，而不是被静默转换为危险状态。"""

        base = {
            "DayTrial": ["01-01-test"],
            "action_dir": ["left"],
            "ifscared2": [1],
        }
        for invalid_status in (np.nan, np.inf, 2.5):
            with self.subTest(invalid_status=invalid_status):
                data = pd.DataFrame({**base, "ifscared1": [invalid_status]})
                with self.assertRaises(ValueError):
                    build_utility_estimation_input(data)

    def test_evade_normalization_does_not_mutate_input_array(self) -> None:
        """验证 Evade/NoEnergizer 的平移只发生在副本，不改写 raw Q。"""

        position = (1, 1)
        adjacent_map = {
            position: {
                "left": (0, 1),
                "right": (2, 1),
                "up": np.nan,
                "down": (1, 2),
            }
        }
        raw_q = np.asarray([-8.0, -4.0, -np.inf, 0.0])
        before = raw_q.copy()
        normalized = make_evade_q_non_negative(raw_q, -8.0, position, adjacent_map)

        np.testing.assert_array_equal(raw_q, before)
        np.testing.assert_allclose(normalized[[0, 1, 3]], [0.0, 0.5, 1.0])
        self.assertTrue(np.isneginf(normalized[2]))

    def test_dataframe_normalization_preserves_every_raw_q_column(self) -> None:
        """验证完整追加 Q_norm 流程不会通过 object 数组别名修改任何 raw Q。"""

        position = (1, 1)
        adjacent_map = {
            position: {
                "left": (0, 1),
                "right": (2, 1),
                "up": np.nan,
                "down": (1, 2),
            }
        }
        data = pd.DataFrame({"pacmanPos": [position]})
        before: dict[str, np.ndarray] = {}
        for column in Q_COLUMNS:
            if "evade" in column or "no_energizer" in column:
                value = np.asarray([-8.0, -4.0, -np.inf, 0.0])
            else:
                value = np.asarray([1.0, 2.0, -np.inf, 3.0])
            data[column] = pd.Series([value], dtype=object)
            before[column] = value.copy()

        result = append_normalized_q_columns(data, adjacent_map)
        for column in Q_COLUMNS:
            np.testing.assert_array_equal(data.at[0, column], before[column])
            np.testing.assert_array_equal(result.at[0, column], before[column])

    def test_cluster_global_excludes_distance_one_but_keeps_distance_two(self) -> None:
        """验证 Cluster Global 在距离 1 时无信息、距离 2 时恢复方向证据。

        输入语义：构造四格直线地图，分别让 Pacman 与唯一豆子相距 1 步和 2 步。
        输出语义：距离 1 的合法方向均为 0；距离 2 时向右接近为正、向左远离为负。
        关键约束：最小距离只区分紧邻资源的 Local 行为，不恢复旧版 10 步忽略范围。
        """

        positions = [(0, 0), (1, 0), (2, 0), (3, 0)]
        adjacent_map = {
            (0, 0): {"left": np.nan, "right": (1, 0), "up": np.nan, "down": np.nan},
            (1, 0): {"left": (0, 0), "right": (2, 0), "up": np.nan, "down": np.nan},
            (2, 0): {"left": (1, 0), "right": (3, 0), "up": np.nan, "down": np.nan},
            (3, 0): {"left": (2, 0), "right": np.nan, "up": np.nan, "down": np.nan},
        }
        distance_map = {
            source: {target: abs(source[0] - target[0]) for target in positions}
            for source in positions
        }
        map_data = MapData(adjacent_map, distance_map, {1: 2, 2: 4, 8: 8, 9: 8})
        adjacent_row = pd.Series({"pacmanPos": (1, 0), "beans": [(2, 0)], "energizers": []})

        raw_matrix, normalized_matrix, metadata = global_cluster_q_for_row(
            adjacent_row,
            map_data,
            adjacent_map,
            CalculateUtilityConfig(),
        )

        self.assertEqual(len(raw_matrix), 1)
        self.assertEqual(metadata[0]["min_distance"], 1.0)
        np.testing.assert_allclose(np.asarray(raw_matrix[0])[[0, 1]], [0.0, 0.0])
        np.testing.assert_allclose(np.asarray(normalized_matrix[0])[[0, 1]], [0.0, 0.0])

        distant_row = pd.Series({"pacmanPos": (1, 0), "beans": [(3, 0)], "energizers": []})
        raw_matrix, normalized_matrix, metadata = global_cluster_q_for_row(
            distant_row,
            map_data,
            adjacent_map,
            CalculateUtilityConfig(),
        )
        self.assertEqual(metadata[0]["min_distance"], 2.0)
        self.assertLess(raw_matrix[0][0], 0.0)
        self.assertGreater(raw_matrix[0][1], 0.0)
        self.assertEqual(normalized_matrix[0][1], 1.0)

    def test_real_frame_35_approach_reaches_scared_ghost_behind_dead_ghost(self) -> None:
        """验证 0-based 第35帧向左路径不会被中间的 dead ghost 错误终止。"""

        utility = self._estimate_real_player_frame(player="p2", trial=TARGET_TRIAL, frame_id=3455)
        approach_q = np.asarray(utility.at[0, "approach_Q"], dtype=float)
        # 该回归只验证 dead ghost 不会阻断通往后方 scared ghost 的左侧路径。20步
        # 搜索会额外发现向下的远程路径，因此不再把 left > down 当作状态语义约束。
        self.assertGreater(approach_q[0], 0.0)
        self.assertTrue(np.isneginf(approach_q[2]))

    def test_real_normal_ghost_produces_nonzero_evade_utility(self) -> None:
        """验证状态 1 的正常 ghost 能在真实地图搜索中产生 Evade 风险。"""

        # 该帧 P2 与状态 1 的 Blinky 地图距离为 4，位于新的 6 步 Evade 搜索范围内。
        utility = self._estimate_real_player_frame(
            player="p2",
            trial=EVADE_TARGET_TRIAL,
            frame_id=EVADE_TARGET_FRAME_ID,
        )
        evade_q = np.asarray(utility.at[0, "evade_blinky_Q"], dtype=float)
        self.assertLess(float(np.min(evade_q[np.isfinite(evade_q)])), 0.0)

    def test_real_frame_6_local_uses_best_leaf_path(self) -> None:
        """验证 Local 按首方向的最大叶路径奖励聚合，而不是对所有路径取平均。

        输入语义：01-01 trial 的 0-based 第6帧中，P2 位于 (13,27)；向左五条叶路径
        奖励为 [4,4,2,0,14]，向上为 [10,8,8,8,8]。
        输出语义：Local Q 必须取逐步衰减后的最大值，且正确偏好真实动作 left。
        关键约束：该回归只改变 Local；墙方向 down 仍保持负无穷。
        """

        utility = self._estimate_real_player_frame(
            player="p2",
            trial=EVADE_TARGET_TRIAL,
            frame_id=88,
        )
        local_q = np.asarray(utility.at[0, "local_Q"], dtype=float)
        gamma = UtilityConfig().local_discount_factor
        expected_left = 2.0 * sum(gamma**depth_index for depth_index in range(3, 10))
        expected_right = 2.0 * sum(gamma**depth_index for depth_index in range(8, 10))
        expected_up = 2.0 * sum(gamma**depth_index for depth_index in range(5, 10))
        np.testing.assert_allclose(local_q[:3], [expected_left, expected_right, expected_up])
        self.assertTrue(np.isneginf(local_q[3]))
        self.assertGreater(local_q[0], local_q[2])

    def _estimate_real_player_frame(self, player: str, trial: str, frame_id: int) -> pd.DataFrame:
        """读取一个真实 tile，按修复后的 05 输入路径计算并修正不可走方向。

        输入语义：player 指定玩家前缀，trial/frame_id 共同定位原始帧。
        输出语义：返回仅一行且包含七个 raw Q 的单玩家 DataFrame。
        关键约束：测试数据或地图常量缺失时跳过真实数据回归，不影响纯函数测试。
        """

        if not CURRENT_TILE_FILE.exists() or not (CONSTANT_DIR / "map_constants.pkl").exists():
            self.skipTest("当前仓库未提供真实 04 数据或地图常量。")

        source = pd.read_pickle(CURRENT_TILE_FILE)
        selected = source[
            source["DayTrial"].astype(str).eq(trial)
            & source["frame_id"].eq(frame_id)
        ].copy()
        self.assertEqual(len(selected), 1)

        row_mask = build_player_alive_mask(selected, player)
        player_view = build_player_view(selected, player, row_mask)
        utility_input = build_utility_estimation_input(player_view)
        map_data, adjacent_map = load_calculate_utility_maps(CONSTANT_DIR)
        estimated = estimate_utility_for_dataframe(utility_input, map_data, UtilityConfig())
        corrected, _ = correct_unavailable_q_values(estimated, adjacent_map)
        return corrected


if __name__ == "__main__":
    unittest.main()
