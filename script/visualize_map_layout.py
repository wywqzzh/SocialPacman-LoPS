#!/usr/bin/env python3
"""从 Pacman 数据中读取地图字段，并可视化墙和空位。

本脚本是辅助检查工具，不参与正式 01-12 分析流程。它可以读取
raw_subject_data 的 pickle 文件，也可以直接读取原始 `.mat` 文件中的
`data/gameMap/currentTiles`。输出图片只区分墙和空位：墙使用黑色格子，
其它字符（豆子、能量豆、空白等）统一视为可走位置并使用白色格子。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import h5py
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/01_raw_subject_data/comp/10001-10022-2025-07-15-JJJ-1.pkl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/map_visualization/map_layout.png"
DEFAULT_WALL_CHARS = "|_-"
DEFAULT_WIDTH = 28


def load_label_font(cell_size: int) -> ImageFont.ImageFont:
    """加载行列号字体。

    输入语义：cell_size 是单个地图格子的像素尺寸，用于估计合适字号。
    输出语义：返回 PIL 可用字体对象。
    关键约束：字体文件在不同机器上可能不存在，因此找不到字体时退回默认字体，
    保证脚本仍能生成地图检查图。
    """

    font_size = max(10, min(16, int(cell_size * 0.55)))
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        candidate = Path(font_path)
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), font_size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    """计算文本绘制尺寸。

    输入语义：draw 是当前图片的绘制上下文，text 是待绘制标签，font 是字体。
    输出语义：返回文本宽高，用于把行列号放到对应格子的中心位置。
    关键约束：使用 textbbox 而不是固定字符宽度，避免不同字体下标签偏移明显。
    """

    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输入语义：默认读取一个已经生成的 raw_subject_data pickle；调用方也可以显式
    指定任意包含 Map 字段的 `.pkl` 或包含 currentTiles 的 `.mat`。
    输出语义：返回输入路径、输出路径、地图宽度和绘图样式参数。
    关键约束：本脚本只输出墙/空位图，不绘制 Pacman、ghost、豆子或其它动态元素。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT, help="输入 .pkl 或 .mat 文件。")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT, help="输出 PNG 图片路径。")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="地图字符串每行字符数，当前数据默认为 28。")
    parser.add_argument("--cell-size", type=int, default=24, help="每个地图格子的像素尺寸。")
    parser.add_argument("--wall-chars", default=DEFAULT_WALL_CHARS, help="视为墙的地图字符集合。")
    parser.add_argument("--no-grid", action="store_true", help="不绘制墙格边线。")
    parser.add_argument("--zero-based-labels", action="store_true", help="行列号从 0 开始；默认从 1 开始以匹配数据中的 tile 坐标。")
    return parser.parse_args()


def load_map_string(input_path: Path) -> str:
    """从输入文件读取一帧地图字符串。

    输入语义：input_path 可以是 `.pkl` 或 `.mat`。`.pkl` 需要包含 `Map` 列，
    `.mat` 需要包含 `data/gameMap/currentTiles`。
    输出语义：返回第一帧地图的字符串表示。
    关键约束：只读取第一帧地图，因为墙体布局在同一任务内应保持不变。
    """

    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".pkl":
        data = pd.read_pickle(input_path)
        if "Map" not in data.columns:
            raise KeyError(f"{input_path} 中没有 Map 列。")
        return str(data["Map"].iloc[0])

    if suffix == ".mat":
        with h5py.File(input_path, "r") as mat:
            if "data/gameMap/currentTiles" not in mat:
                raise KeyError(f"{input_path} 中没有 data/gameMap/currentTiles。")
            current_tiles = mat["data/gameMap/currentTiles"]
            first_frame = current_tiles[0]
            return "".join(chr(int(value)) for value in first_frame)

    raise ValueError(f"暂不支持的输入格式：{input_path.suffix}")


def reshape_map(map_string: str, width: int) -> list[str]:
    """把一维地图字符串切分成二维文本行。

    输入语义：map_string 是 `.mat` 中展开后的地图字符序列，width 是每行字符数。
    输出语义：返回按行排列的字符串列表。
    关键约束：长度必须能被 width 整除，否则无法确定完整矩形地图。
    """

    if width <= 0:
        raise ValueError("--width 必须大于 0。")
    if len(map_string) % width != 0:
        raise ValueError(f"地图长度 {len(map_string)} 不能被 width={width} 整除。")
    return [map_string[start : start + width] for start in range(0, len(map_string), width)]


def render_wall_empty_map(
    rows: list[str],
    output_path: Path,
    *,
    wall_chars: Iterable[str],
    cell_size: int,
    draw_grid: bool,
    zero_based_labels: bool,
) -> None:
    """把二维地图行渲染成只包含墙和空位的 PNG 图片。

    输入语义：rows 是地图二维字符行；wall_chars 指定哪些字符视为墙。
    输出语义：在 output_path 写出 PNG 图片。
    关键约束：非墙字符全部视为空位，因此豆子、能量豆等动态/奖励元素不会被单独绘制。
    """

    if not rows:
        raise ValueError("地图行不能为空。")
    if cell_size <= 0:
        raise ValueError("--cell-size 必须大于 0。")

    height = len(rows)
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("所有地图行必须等长。")

    wall_set = set(wall_chars)
    label_margin = max(42, cell_size * 2)
    outer_padding = max(8, cell_size // 2)
    image_width = label_margin + width * cell_size + outer_padding
    image_height = label_margin + height * cell_size + outer_padding
    image = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(image)
    label_font = load_label_font(cell_size)

    wall_color = (0, 0, 0)
    walkable_color = (255, 255, 255)
    grid_color = (235, 235, 235)
    label_color = (0, 0, 0)

    # 坐标标注放在地图外侧，不占用实际地图格。默认使用 1-based 编号，
    # 因为数据中的 pacman tile 坐标按 1-based 解释时才和可走格完全一致。
    label_offset = 0 if zero_based_labels else 1
    for x in range(width):
        label = str(x + label_offset)
        x0 = label_margin + x * cell_size
        text_width, text_height = text_size(draw, label, label_font)
        # 列号放在对应格子的水平中心，便于从标签直接对齐到地图列。
        draw.text(
            (x0 + (cell_size - text_width) / 2, label_margin - cell_size + (cell_size - text_height) / 2),
            label,
            fill=label_color,
            font=label_font,
        )
    for y in range(height):
        label = str(y + label_offset)
        y0 = label_margin + y * cell_size
        text_width, text_height = text_size(draw, label, label_font)
        # 行号右对齐到地图左侧，避免一位数和两位数标签的末端不一致。
        draw.text(
            (label_margin - text_width - max(8, cell_size // 3), y0 + (cell_size - text_height) / 2),
            label,
            fill=label_color,
            font=label_font,
        )

    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            x0 = label_margin + x * cell_size
            y0 = label_margin + y * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            is_wall = char in wall_set
            fill = wall_color if is_wall else walkable_color
            draw.rectangle([x0, y0, x1, y1], fill=fill)
            # 参考图中白色通道是连续区域，只有黑色墙块带浅色格线；
            # 因此这里不在可走位置上画网格，避免把通道切成独立小方块。
            if draw_grid and is_wall:
                draw.rectangle([x0, y0, x1, y1], outline=grid_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    """命令行入口：读取地图、转换为空位/墙布局图并保存。"""

    args = parse_args()
    map_string = load_map_string(args.input_path)
    rows = reshape_map(map_string, args.width)
    render_wall_empty_map(
        rows,
        args.output_path,
        wall_chars=args.wall_chars,
        cell_size=args.cell_size,
        draw_grid=not args.no_grid,
        zero_based_labels=args.zero_based_labels,
    )
    print(f"地图尺寸：{args.width} 列 x {len(rows)} 行")
    print(f"墙字符：{args.wall_chars!r}")
    print(f"颜色说明：黑色=墙，白色=可走位置")
    print(f"行列编号：{'0-based' if args.zero_based_labels else '1-based'}")
    print(f"输出图片：{args.output_path.resolve()}")


if __name__ == "__main__":
    main()
