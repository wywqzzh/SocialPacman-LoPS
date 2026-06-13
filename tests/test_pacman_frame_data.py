"""pacman_data frame_data 转换的排序和行号测试。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from LoPS.pacman_data.raw_subject_data_to_frame_data import convert_raw_subject_data_to_frame_data


class PacmanFrameDataTest(unittest.TestCase):
    """验证 raw_subject_data 到 frame_data 的关键表结构约束。"""

    def test_frame_data_uses_numeric_daytrial_order_before_frame_id(self) -> None:
        """确认 DayTrial 按数字排序，并在排序后生成连续 frame_id。"""

        # 输入故意打乱 trial 顺序，并放入 10-1；如果按字符串排序，10-1 会排到 2-1 前面。
        raw_data = pd.DataFrame(
            {
                "Step": [1, 1, 1, 2],
                "DayTrial": [
                    "10-1-031222-401-03-Dec-2022",
                    "1-2-031222-401-03-Dec-2022",
                    "2-1-031222-401-03-Dec-2022",
                    "1-2-031222-401-03-Dec-2022",
                ],
                "Map": ["#" * (29 * 36)] * 4,
                "pacMan_1": [10, 1, 2, 1],
                "pacMan_2": [18, 18, 18, 19],
                "ghost1_1": [5, 5, 5, 5],
                "ghost1_2": [6, 6, 6, 6],
                "ghost1_3": [1, 1, 1, 1],
                "ghost2_1": [7, 7, 7, 7],
                "ghost2_2": [8, 8, 8, 8],
                "ghost2_3": [1, 1, 1, 1],
                "ghost3_1": [np.inf, np.inf, np.inf, np.inf],
                "ghost3_2": [np.inf, np.inf, np.inf, np.inf],
                "ghost3_3": [np.inf, np.inf, np.inf, np.inf],
                "ghost4_1": [np.inf, np.inf, np.inf, np.inf],
                "ghost4_2": [np.inf, np.inf, np.inf, np.inf],
                "ghost4_3": [np.inf, np.inf, np.inf, np.inf],
                "JoyStick": ["left", "right", "up", "down"],
                "pDir": ["left", "right", "up", "down"],
            }
        )

        frame_data = convert_raw_subject_data_to_frame_data(raw_data)

        self.assertNotIn("Unnamed: 0", frame_data.columns)
        self.assertEqual(frame_data["frame_id"].tolist(), [0, 1, 2, 3])
        self.assertEqual(
            frame_data[["DayTrial", "Step"]].apply(tuple, axis=1).tolist(),
            [
                ("1-2-031222-401-03-Dec-2022", 0),
                ("1-2-031222-401-03-Dec-2022", 1),
                ("2-1-031222-401-03-Dec-2022", 0),
                ("10-1-031222-401-03-Dec-2022", 0),
            ],
        )

    def test_frame_data_drops_four_ghost_trials_before_indexing(self) -> None:
        """确认 frame_data 生成阶段会按整局丢弃 four-ghost trial。"""

        raw_data = pd.DataFrame(
            {
                "Step": [1, 2, 1, 2],
                "DayTrial": [
                    "1-1-031222-401-03-Dec-2022",
                    "1-1-031222-401-03-Dec-2022",
                    "2-1-031222-401-03-Dec-2022",
                    "2-1-031222-401-03-Dec-2022",
                ],
                "Map": ["#" * (29 * 36)] * 4,
                "pacMan_1": [10, 10, 12, 12],
                "pacMan_2": [18, 19, 18, 19],
                "ghost1_1": [5, 5, 5, 5],
                "ghost1_2": [6, 6, 6, 6],
                "ghost1_3": [1, 1, 1, 1],
                "ghost2_1": [7, 7, 7, 7],
                "ghost2_2": [8, 8, 8, 8],
                "ghost2_3": [1, 1, 1, 1],
                # 第一个 trial 第三、第四个 ghost 全程为 inf，应被保留。
                # 第二个 trial 有真实 ghost3/ghost4 坐标，应整局删除。
                "ghost3_1": [np.inf, np.inf, 14, 14],
                "ghost3_2": [np.inf, np.inf, 17, 17],
                "ghost3_3": [np.inf, np.inf, 3, 3],
                "ghost4_1": [np.inf, np.inf, 22, 22],
                "ghost4_2": [np.inf, np.inf, 18, 18],
                "ghost4_3": [np.inf, np.inf, 4, 4],
                "JoyStick": ["left", "right", "up", "down"],
                "pDir": ["left", "right", "up", "down"],
            }
        )

        frame_data = convert_raw_subject_data_to_frame_data(raw_data)

        self.assertEqual(frame_data["DayTrial"].unique().tolist(), ["1-1-031222-401-03-Dec-2022"])
        self.assertEqual(frame_data["Step"].tolist(), [0, 1])
        self.assertNotIn("Unnamed: 0", frame_data.columns)
        self.assertEqual(frame_data["frame_id"].tolist(), [0, 1])
        self.assertEqual(frame_data["ghost3Pos"].tolist(), [[], []])
        self.assertEqual(frame_data["ghost4Pos"].tolist(), [[], []])


if __name__ == "__main__":
    unittest.main()
