#!/usr/bin/env python3
"""运行 Pacman render table 到 JPG 图片帧的渲染。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.pacman_video.frame_renderer import BAR_TYPE_CHOICES, PacmanRenderer, iter_output_paths, load_render_rows


def parse_args() -> argparse.Namespace:
    """解析图片帧渲染脚本参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", nargs="?", default="041122-403")
    parser.add_argument("--render-table-dir", type=Path, default=PROJECT_ROOT / "data/pacman_video/render_data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/pacman_video/frame_images")
    parser.add_argument("--trial", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--aa", type=int, default=3)
    parser.add_argument("--bar-type", choices=sorted(BAR_TYPE_CHOICES), default="grammar")
    return parser.parse_args()


def main() -> None:
    """逐帧渲染 JPG 图片并打印进度。"""

    args = parse_args()
    rows = load_render_rows(
        render_table_dir=args.render_table_dir,
        subject=args.subject,
        trial=args.trial,
        start=args.start,
        max_frames=args.max_frames,
    )
    if rows.empty:
        raise SystemExit("没有可渲染的帧。")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = PacmanRenderer(str(rows.iloc[0]["Map"]), aa=max(1, args.aa), bar_type=args.bar_type)
    total = len(rows)
    for index, row, path in iter_output_paths(rows, args.output_dir, subject=args.subject):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = renderer.render(row)
        frame.save(path, quality=95, subsampling=0)
        if (index + 1) % 50 == 0 or index == 0 or index + 1 == total:
            print(f"rendered {index + 1}/{total}: {path}")


if __name__ == "__main__":
    main()
