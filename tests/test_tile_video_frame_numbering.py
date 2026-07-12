"""验证 tile 视频的文件名、顶部帧号和 context 全部使用 0-based 编号。"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RENDERER_PATH = PROJECT_ROOT / "script" / "pacman_video" / "run_tile_video_renderer.py"


def load_renderer_module():
    """从脚本路径加载 tile renderer，供编号纯函数回归测试使用。

    输入语义：固定读取当前仓库的正式视频入口脚本。
    输出语义：返回已经执行并可调用函数的 Python module。
    关键约束：注册到 sys.modules 后再执行，确保脚本内部类型和异常类稳定。
    """

    module_name = "test_run_tile_video_renderer"
    spec = importlib.util.spec_from_file_location(module_name, RENDERER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载视频渲染脚本：{RENDERER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


renderer = load_renderer_module()


class TileVideoFrameNumberingTests(unittest.TestCase):
    """覆盖 0-based counter、context 闭区间和 PNG 文件名。"""

    def test_frame_counter_uses_zero_and_last_index(self) -> None:
        """验证首尾帧显示为 0/N-1，而不是 1/N。"""

        self.assertEqual(renderer.format_video_frame_counter(0, 223), "video frame 0/222")
        self.assertEqual(renderer.format_video_frame_counter(222, 223), "video frame 222/222")
        with self.assertRaises(renderer.TileVideoRenderError):
            renderer.format_video_frame_counter(223, 223)

    def test_context_lookup_and_closed_interval_are_zero_based(self) -> None:
        """验证右开 row_id context 会转换成 0-based 视频闭区间。"""

        game_rows = pd.DataFrame({"row_id": [100, 101, 102]})
        lookup = renderer.build_context_video_frame_lookup(game_rows)
        self.assertEqual(lookup, {100: 0, 101: 1, 102: 2})

        row = pd.Series({"p1_trial_context": (100, 102)})
        self.assertEqual(
            renderer.format_player_context_video_frames(row, "p1", lookup),
            "P1 ctx [0,1]",
        )

    def test_saved_png_names_start_at_zero(self) -> None:
        """验证连续图片文件名严格从 000000 开始。"""

        frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            renderer.save_frame_images(frames, output_dir)
            self.assertEqual(
                sorted(path.name for path in output_dir.glob("*.png")),
                ["000000.png", "000001.png", "000002.png"],
            )


if __name__ == "__main__":
    unittest.main()
