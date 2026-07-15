"""当前单人/双人 frame 视频数据对齐与渲染测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.pacman_video.frame_renderer import PacmanRenderer, iter_output_paths
from LoPS.pacman_video.render_table import (
    DataProcessingError,
    align_current_strategies_to_frames,
    build_current_render_table,
    validate_current_frame_table,
)
from LoPS.pacman_video.video_renderer import numeric_sequence


def make_frame_table(*, include_p2: bool) -> pd.DataFrame:
    """构造覆盖两个 trial 的最小当前 02 frame 表。

    输入语义：``include_p2`` 控制是否加入完整 P2 绘图字段。
    输出语义：返回 Step 连续、Map 长度合法的合成逐帧表。
    关键约束：合成表只服务于接口测试，不模拟完整游戏物理过程。
    """

    data: dict[str, object] = {
        "DayTrial": ["01-trial"] * 3 + ["02-trial"] * 2,
        "Step": [0, 1, 2, 0, 1],
        "Map": [" " * (28 * 36)] * 5,
        "p1_ppX": [100.0] * 5,
        "p1_ppY": [100.0] * 5,
        "p1_pDir": ["right"] * 5,
        "p1_pFrame": [0] * 5,
        "p1_alive": [True] * 5,
    }
    if include_p2:
        data.update(
            {
                "p2_ppX": [200.0] * 5,
                "p2_ppY": [100.0] * 5,
                "p2_pDir": ["left"] * 5,
                "p2_pFrame": [0] * 5,
                "p2_alive": [True] * 5,
            }
        )
    return pd.DataFrame(data)


def make_strategy_table(*, include_p2: bool) -> pd.DataFrame:
    """构造与合成 frame 表对应的稀疏 07 策略表。

    输入语义：``include_p2`` 控制是否加入第二位玩家策略。
    输出语义：返回每个 trial 至少从 Step 0 开始的策略关键帧。
    关键约束：第一局在 Step 2 切换策略，用于检查右开区间对齐。
    """

    data: dict[str, object] = {
        "DayTrial": ["01-trial", "01-trial", "02-trial"],
        "Step": [0, 2, 0],
        "p1_revised_strategy_name": ["global", "local", "energizer"],
    }
    if include_p2:
        data["p2_revised_strategy_name"] = ["stay", "approach", "vague"]
    return pd.DataFrame(data)


class CurrentFrameVideoPipelineTests(unittest.TestCase):
    """验证当前 02/07 到 frame 视频接口的关键结构约束。"""

    def test_single_player_alignment_does_not_create_p2_columns(self) -> None:
        """单人输入应只生成 P1 策略，并在 trial 内按 Step 前向延续。"""

        frames = make_frame_table(include_p2=False)
        strategies = make_strategy_table(include_p2=False)
        aligned, missing = align_current_strategies_to_frames(
            frames,
            strategies,
            player_count=1,
            source_name="synthetic",
        )

        self.assertEqual(validate_current_frame_table(frames, "synthetic"), 1)
        self.assertNotIn("p2_display_strategy", aligned.columns)
        self.assertEqual(
            aligned["p1_display_strategy"].tolist(),
            ["global", "global", "local", "energizer", "energizer"],
        )
        self.assertEqual(missing, {"p1": []})

    def test_dual_player_alignment_keeps_player_labels_independent(self) -> None:
        """双人输入应分别对齐 P1/P2，且第二个 trial 不继承第一局末尾策略。"""

        frames = make_frame_table(include_p2=True)
        strategies = make_strategy_table(include_p2=True)
        aligned, _ = align_current_strategies_to_frames(
            frames,
            strategies,
            player_count=2,
            source_name="synthetic",
        )

        self.assertEqual(validate_current_frame_table(frames, "synthetic"), 2)
        self.assertEqual(aligned["p2_display_strategy"].tolist(), ["stay", "stay", "approach", "vague", "vague"])

    def test_partial_p2_fields_raise_clear_error(self) -> None:
        """只出现部分 P2 绘图字段时必须报错，不能静默降级成单人。"""

        frames = make_frame_table(include_p2=False)
        frames["p2_ppX"] = 200.0
        with self.assertRaises(DataProcessingError):
            validate_current_frame_table(frames, "synthetic")

    def test_build_none_mode_needs_no_strategy_file(self) -> None:
        """none 模式应只依赖 02，并保持输入行数和单人字段结构。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frame_path = root / "frame.pkl"
            output_path = root / "render.pkl"
            make_frame_table(include_p2=False).to_pickle(frame_path)

            summary = build_current_render_table(
                frame_path,
                output_path,
                display_mode="none",
            )
            output = pd.read_pickle(output_path)

        self.assertEqual(summary.player_count, 1)
        self.assertEqual(summary.frame_rows, 5)
        self.assertNotIn("p2_ppX", output.columns)
        self.assertNotIn("p1_display_strategy", output.columns)

    def test_renderer_draws_two_players_and_two_strategy_blocks(self) -> None:
        """双人 strategy 帧应正常生成固定尺寸图片并包含玩家策略字段。"""

        row = make_frame_table(include_p2=True).iloc[0].copy()
        row["p1_display_strategy"] = "global"
        row["p2_display_strategy"] = "local"
        for ghost in ("g1", "g2"):
            row[f"{ghost}pX"] = 300.0
            row[f"{ghost}pY"] = 300.0
            row[f"{ghost}Dir"] = "left"
            row[f"{ghost}ModeR"] = 1
            row[f"{ghost}Scared"] = False
            row[f"{ghost}Frame"] = 0

        image = PacmanRenderer(str(row["Map"]), aa=1, bar_type="strategy").render(row)
        self.assertEqual(image.size, (812, 1170))
        # 角色中心是原始嘴部三角形的起点，因此在身体上半部取样，避免把黑嘴误判为漏画。
        self.assertNotEqual(image.getpixel((143, 90)), (0, 0, 0))
        self.assertNotEqual(image.getpixel((243, 90)), (0, 0, 0))

    def test_output_frame_names_are_zero_based(self) -> None:
        """每个 trial 的图片文件名都应从 000000 开始独立编号。"""

        rows = make_frame_table(include_p2=False)
        paths = [path.name for _, _, path in iter_output_paths(rows, Path("frames"), subject=None)]
        self.assertEqual(paths, ["000000.jpg", "000001.jpg", "000002.jpg", "000000.jpg", "000001.jpg"])

    def test_video_sequence_accepts_six_digit_zero_based_frames(self) -> None:
        """视频合成器应识别六位 0-based 连续编号，避免 concat 重复末帧。"""

        paths = [Path(f"{index:06d}.jpg") for index in range(3)]
        self.assertEqual(numeric_sequence(paths), (0, 2, 6))


if __name__ == "__main__":
    unittest.main()
