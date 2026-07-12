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
    correct_unavailable_q_values,
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

    def test_cluster_global_remains_informative_inside_legacy_ignore_depth(self) -> None:
        """验证目标团进入旧 10 步忽略范围后，cluster Global 仍指向该目标。

        输入语义：构造三格直线地图，Pacman 与唯一豆子只相距一步，并把旧 Global
        ignore depth 保持为默认 10。
        输出语义：向右接近豆子的 cluster Global Q 必须为正，向左远离时必须为负。
        关键约束：该测试只约束新增的 cluster Global；旧区域 Global 仍可使用自己的
        ignore depth，Local 的搜索范围也不受影响。
        """

        positions = [(0, 0), (1, 0), (2, 0)]
        adjacent_map = {
            (0, 0): {"left": np.nan, "right": (1, 0), "up": np.nan, "down": np.nan},
            (1, 0): {"left": (0, 0), "right": (2, 0), "up": np.nan, "down": np.nan},
            (2, 0): {"left": (1, 0), "right": np.nan, "up": np.nan, "down": np.nan},
        }
        distance_map = {
            source: {target: abs(source[0] - target[0]) for target in positions}
            for source in positions
        }
        map_data = MapData(adjacent_map, distance_map, {1: 2, 2: 4, 8: 8, 9: 8})
        row = pd.Series({"pacmanPos": (1, 0), "beans": [(2, 0)], "energizers": []})

        raw_matrix, normalized_matrix, metadata = global_cluster_q_for_row(
            row,
            map_data,
            adjacent_map,
            CalculateUtilityConfig(),
        )

        self.assertEqual(len(raw_matrix), 1)
        self.assertEqual(metadata[0]["min_distance"], 1.0)
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
