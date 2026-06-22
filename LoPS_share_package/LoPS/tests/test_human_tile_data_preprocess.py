"""human_tile_data_preprocess 的 ghost 位置修正测试。"""

from __future__ import annotations

import unittest
import importlib.util
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_PATH = PROJECT_ROOT / "script" / "04_human_tile_data_preprocess.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("script_04_human_tile_data_preprocess", SCRIPT_PATH)
if SCRIPT_SPEC is None or SCRIPT_SPEC.loader is None:
    raise ImportError(f"无法加载测试目标脚本：{SCRIPT_PATH}")
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = SCRIPT_MODULE
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)

normalize_optional_ghost_fix_position = SCRIPT_MODULE.normalize_optional_ghost_fix_position
repair_known_ghost_position_errors = SCRIPT_MODULE.repair_known_ghost_position_errors


class HumanTileDataPreprocessTest(unittest.TestCase):
    """验证 corrected tile 阶段对 ghost 空位置和坐标修正的处理。"""

    def test_repair_known_ghost_position_errors_skips_empty_list_and_fixes_list_position(self) -> None:
        """空 list ghost 不应报错，非空 list/tuple 坐标应按修正表改成 tuple。"""

        trial_tile_rows = pd.DataFrame(
            {
                "ghost1Pos": [(14, 20), "not-a-position"],
                "ghost2Pos": [[15, 20], "[16, 20]"],
                "ghost3Pos": [[], "[]"],
                "ghost4Pos": [[12, 18], (11, 18)],
            }
        )

        repair_known_ghost_position_errors(trial_tile_rows)

        self.assertEqual(trial_tile_rows.at[0, "ghost1Pos"], (14, 19))
        self.assertEqual(trial_tile_rows.at[0, "ghost2Pos"], (15, 19))
        self.assertEqual(trial_tile_rows.at[1, "ghost2Pos"], "[16, 20]")
        self.assertEqual(trial_tile_rows.at[0, "ghost3Pos"], [])
        self.assertEqual(trial_tile_rows.at[1, "ghost3Pos"], "[]")
        self.assertEqual(trial_tile_rows.at[0, "ghost4Pos"], [12, 18])

    def test_normalize_optional_ghost_fix_position_returns_none_for_empty_markers(self) -> None:
        """空 ghost 标记应返回 None，避免进入 dict membership 判断。"""

        self.assertIsNone(normalize_optional_ghost_fix_position([]))
        self.assertIsNone(normalize_optional_ghost_fix_position("[]"))
        self.assertIsNone(normalize_optional_ghost_fix_position(()))


if __name__ == "__main__":
    unittest.main()
