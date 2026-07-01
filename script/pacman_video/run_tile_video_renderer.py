#!/usr/bin/env python3
"""用 04 阶段 tile 数据绘制简易 Pacman 游戏视频。

本脚本直接读取 ``data/04_corrected_tile_data/{task}/{session}.pkl``，选择其中
一个 ``DayTrial`` 作为一个 game，把每条 tile 行渲染成一帧视频。地图背景来自
``data/constant_data/map_constants.pkl`` 的可走点集合：黑色表示墙，白色表示
可以走的位置；每帧额外绘制当前剩余的豆子、energizer、两个 Pacman 和两个 ghost。

这个脚本是检查 04 处理结果的轻量可视化工具，不依赖旧的 render table、grammar
或逐帧渲染链路。
"""

from __future__ import annotations

import argparse
import ast
import math
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.pacman_video.frame_renderer import DIR_DOWN, direction_enum, quadratic_path

DEFAULT_TILE_ROOT = PROJECT_ROOT / "data/04_corrected_tile_data"
DEFAULT_MAP_CONSTANTS = PROJECT_ROOT / "data/constant_data/map_constants.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/pacman_video/tile_video"
DEFAULT_FRAME_OUTPUT_DIR = PROJECT_ROOT / "data/pacman_video/tile_frame_images"
DEFAULT_ICON_FONT = PROJECT_ROOT / "data/pacman_video/assets/Font Awesome 5 Pro-Solid-900.otf"
DEFAULT_TASK = "comp"
DEFAULT_CELL_SIZE = 30
DEFAULT_FPS = 4.0
DEFAULT_AA = 3
DISPLAY_WALL_COLOR = (0, 0, 0)
DISPLAY_WALKABLE_COLOR = (255, 255, 255)
DISPLAY_LABEL_COLOR = (0, 0, 0)
DISPLAY_WALL_GRID_COLOR = (235, 235, 235)
POSITION_COLUMNS = ("p1_pos", "p2_pos", "ghost1Pos", "ghost2Pos")
ITEM_COLUMNS = ("beans", "energizers")
VIDEO_GHOST_HOUSE_DISPLAY_POSITIONS = {
    (14, 16),
    (15, 16),
    *{(x, y) for y in range(17, 20) for x in range(12, 18)},
}
SYMBOLS = {
    "ghost": "\uf6e2",
    "cookie": "\uf563",
    "eye": "\uf06e",
    "monkey": "\uf6fb",
    "apple": "\uf5d1",
}


class TileVideoRenderError(RuntimeError):
    """tile 视频渲染无法继续时抛出的明确异常。"""


def parse_grid_position(value: Any) -> tuple[int, int]:
    """解析 tile 数据或地图常量中的坐标。

    输入语义：value 可以是 tuple/list/numpy 数组，也可以是字符串形式的 ``(x, y)``。
    输出语义：返回整数坐标 ``(x, y)``。
    关键约束：只接受长度为 2 的坐标，不使用 eval 执行任意代码。
    """

    if isinstance(value, tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, list) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, np.ndarray) and value.size == 2:
        flattened = value.reshape(-1)
        return int(flattened[0]), int(flattened[1])
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
        raise TileVideoRenderError(f"无法解析坐标：{value!r}")
    return int(parsed[0]), int(parsed[1])


def parse_position_list(value: Any) -> list[tuple[int, int]]:
    """解析一组地图元素坐标。

    输入语义：value 通常来自 04 数据中的 ``beans`` 或 ``energizers`` 字段，可以是
    Python list，也可以是字符串形式的坐标列表。
    输出语义：返回坐标 tuple 列表；空值返回空列表。
    关键约束：豆子和 energizer 是逐帧状态，必须按当前行解析，不能只画初始状态。
    """

    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "nan", "none"}:
            return []
        value = ast.literal_eval(stripped)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TileVideoRenderError(f"无法解析坐标列表：{value!r}")
    return [parse_grid_position(item) for item in value]


def load_label_font(cell_size: int) -> ImageFont.ImageFont:
    """加载地图行列号和帧信息字体。

    输入语义：cell_size 是单个地图格子的像素大小。
    输出语义：返回 PIL 可用字体对象。
    关键约束：不同环境可能没有同一字体文件；找不到字体时退回默认字体。
    """

    font_size = max(10, min(16, int(cell_size * 0.52)))
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        candidate = Path(font_path)
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), font_size)
    return ImageFont.load_default()


def load_icon_fonts(icon_font_path: Path, cell_size: int) -> dict[str, ImageFont.ImageFont] | None:
    """加载 Font Awesome 图标字体。

    输入语义：icon_font_path 是从旧视频脚本迁移到当前项目的数据资产字体路径。
    输出语义：返回不同元素尺寸对应的字体；字体不存在时返回 None。
    关键约束：正式脚本不能依赖旧项目绝对路径，因此默认只读取当前项目 data 下字体。
    """

    if not icon_font_path.is_file():
        return None
    return {
        "actor": ImageFont.truetype(str(icon_font_path), max(12, int(cell_size * 0.82))),
        "dead_ghost": ImageFont.truetype(str(icon_font_path), max(7, int(cell_size * 0.48))),
    }


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    """计算文本尺寸，便于把标签放到格子中心。

    输入语义：draw 是当前绘图上下文，text/font 是待测文本和字体。
    输出语义：返回文本宽高。
    关键约束：使用 ``textbbox`` 适配不同字体，避免标签位置明显偏移。
    """

    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def draw_centered_icon(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    symbol: str,
    font: ImageFont.ImageFont,
    *,
    fill: tuple[int, int, int],
    stroke_fill: tuple[int, int, int] | None = None,
    stroke_width: int = 0,
) -> None:
    """以像素中心点为锚点绘制一个 Font Awesome 图标。

    输入语义：center 是图标中心点；symbol 是 Font Awesome 私有区字符。
    输出语义：直接修改绘图上下文。
    关键约束：Font Awesome 图标的 bbox 不是从字符左上角开始，必须用 textbbox
    反推左上角，否则图标会明显偏离 tile 中心。
    """

    left, top, right, bottom = draw.textbbox((0, 0), symbol, font=font, stroke_width=stroke_width)
    width = right - left
    height = bottom - top
    x = center[0] - width / 2 - left
    y = center[1] - height / 2 - top
    draw.text(
        (x, y),
        symbol,
        font=font,
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill or fill,
    )


def load_walkable_positions(map_constants_path: Path) -> set[tuple[int, int]]:
    """从地图常量读取可走位置集合。

    输入语义：map_constants_path 指向 ``generate_map_constants.py`` 生成的 pkl。
    输出语义：返回所有合法可走 tile 坐标，包括 tunnel 和 ghost house 位置。
    关键约束：视频背景只依赖当前项目地图常量，不从旧数据或旧路径读取地图。
    """

    if not map_constants_path.is_file():
        raise FileNotFoundError(f"找不到地图常量文件：{map_constants_path}")
    constants = pd.read_pickle(map_constants_path)
    if not isinstance(constants, dict) or "adjacent_map" not in constants:
        raise TileVideoRenderError(f"地图常量文件缺少 adjacent_map：{map_constants_path}")

    adjacent_map = constants["adjacent_map"]
    if "pos" not in adjacent_map.columns:
        raise TileVideoRenderError("adjacent_map 缺少 pos 字段。")
    walkable_positions = {parse_grid_position(value) for value in adjacent_map["pos"]}

    # 地图常量只保留分析需要的 ghost house 节点；视频背景则应该显示完整鬼屋内部
    # 为空白区域，否则鬼屋里没有出现过 ghost 的格子会被画成黑墙，看起来像地图错误。
    return walkable_positions | VIDEO_GHOST_HOUSE_DISPLAY_POSITIONS


def board_bounds(walkable_positions: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    """根据可走点集合推断视频画布中的地图坐标范围。

    输入语义：walkable_positions 是地图常量中的合法坐标集合。
    输出语义：返回 ``(x_min, x_max, y_min, y_max)``。
    关键约束：当前数据含 tunnel 坐标 x=0 和 x=29，因此视频默认展示 0-29 列；
    y 轴展示 1-36 行，和已有地图检查图的行号保持一致。
    """

    if not walkable_positions:
        raise TileVideoRenderError("可走点集合为空，无法绘制地图。")
    x_values = [position[0] for position in walkable_positions]
    y_values = [position[1] for position in walkable_positions]
    x_min = min(0, min(x_values))
    x_max = max(29, max(x_values))
    y_min = min(1, min(y_values))
    y_max = max(36, max(y_values))
    return x_min, x_max, y_min, y_max


def build_base_map_image(
    walkable_positions: set[tuple[int, int]],
    *,
    cell_size: int,
) -> tuple[Image.Image, dict[str, int], ImageFont.ImageFont]:
    """绘制静态地图背景。

    输入语义：walkable_positions 是可走坐标集合；cell_size 控制单个 tile 的像素大小。
    输出语义：返回地图底图、坐标变换元数据和字体对象。
    关键约束：黑色表示墙，白色表示可走位置；行列号直接使用数据坐标值。
    """

    if cell_size <= 0:
        raise TileVideoRenderError("--cell-size 必须大于 0。")

    x_min, x_max, y_min, y_max = board_bounds(walkable_positions)
    width_cells = x_max - x_min + 1
    height_cells = y_max - y_min + 1
    label_margin = max(46, cell_size * 2)
    right_padding = max(12, cell_size // 2)
    bottom_padding = max(12, cell_size // 2)
    header_height = max(28, int(cell_size * 1.2))
    image_width = label_margin + width_cells * cell_size + right_padding
    image_height = header_height + label_margin + height_cells * cell_size + bottom_padding
    # mp4 的 yuv420p 编码要求宽高为偶数；格子尺寸改变后必须显式对齐，
    # 否则 ffmpeg 会因为奇数画布尺寸拒绝写入视频。
    if image_width % 2:
        image_width += 1
    if image_height % 2:
        image_height += 1

    image = Image.new("RGB", (image_width, image_height), DISPLAY_WALKABLE_COLOR)
    draw = ImageDraw.Draw(image)
    font = load_label_font(cell_size)

    wall_color = DISPLAY_WALL_COLOR
    walkable_color = DISPLAY_WALKABLE_COLOR
    wall_grid_color = DISPLAY_WALL_GRID_COLOR
    label_color = DISPLAY_LABEL_COLOR
    origin_x = label_margin
    origin_y = header_height + label_margin

    # 列号和行号使用数据坐标本身。这样 tunnel 的 0/29 列也能直接检查，
    # 不会和已有 1-28 主地图列混淆。
    for x in range(x_min, x_max + 1):
        label = str(x)
        px = origin_x + (x - x_min) * cell_size
        text_width, text_height = text_size(draw, label, font)
        draw.text(
            (px + (cell_size - text_width) / 2, origin_y - cell_size + (cell_size - text_height) / 2),
            label,
            fill=label_color,
            font=font,
        )
    for y in range(y_min, y_max + 1):
        label = str(y)
        py = origin_y + (y - y_min) * cell_size
        text_width, text_height = text_size(draw, label, font)
        draw.text(
            (origin_x - text_width - max(8, cell_size // 3), py + (cell_size - text_height) / 2),
            label,
            fill=label_color,
            font=font,
        )

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            px = origin_x + (x - x_min) * cell_size
            py = origin_y + (y - y_min) * cell_size
            rectangle = [px, py, px + cell_size, py + cell_size]
            if (x, y) in walkable_positions:
                draw.rectangle(rectangle, fill=walkable_color)
            else:
                draw.rectangle(rectangle, fill=wall_color, outline=wall_grid_color)

    metadata = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "cell_size": cell_size,
    }
    return image, metadata, font


def position_to_pixel(position: tuple[int, int], metadata: dict[str, int]) -> tuple[float, float]:
    """把 tile 坐标转换为画布像素中心点。

    输入语义：position 是数据中的 ``(x, y)`` 坐标，metadata 来自底图构造函数。
    输出语义：返回该 tile 中心点像素坐标。
    关键约束：如果坐标落在当前画布范围外，说明数据或地图常量不一致，直接报错。
    """

    x, y = position
    if not (metadata["x_min"] <= x <= metadata["x_max"] and metadata["y_min"] <= y <= metadata["y_max"]):
        raise TileVideoRenderError(f"坐标超出视频地图范围：{position}")
    cell_size = metadata["cell_size"]
    px = metadata["origin_x"] + (x - metadata["x_min"]) * cell_size + cell_size / 2
    py = metadata["origin_y"] + (y - metadata["y_min"]) * cell_size + cell_size / 2
    return px, py


def scale_render_metadata(metadata: dict[str, int], factor: int) -> dict[str, int]:
    """把坐标变换元数据放大到超采样画布。

    输入语义：metadata 来自最终输出尺寸的底图；factor 是超采样倍数。
    输出语义：返回可用于高分辨率工作画布的新 metadata。
    关键约束：坐标范围 ``x_min/x_max/y_min/y_max`` 不变，只有像素单位字段放大。
    """

    if factor <= 1:
        return metadata
    scaled = metadata.copy()
    for key in ("origin_x", "origin_y", "cell_size"):
        scaled[key] = int(scaled[key] * factor)
    return scaled


def draw_actor(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    metadata: dict[str, int],
    *,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    label: str,
    font: ImageFont.ImageFont,
    icon_font: ImageFont.ImageFont | None,
    symbol: str,
    stroke_fill: tuple[int, int, int] | None = None,
    stroke_width: int = 0,
) -> None:
    """在单帧图像上绘制一个动态对象。

    输入语义：position 是对象 tile 坐标；fill/outline 控制颜色；label 是 fallback 标签。
    输出语义：直接修改传入的绘图上下文。
    关键约束：优先使用旧视频脚本中的 Font Awesome 元素；字体缺失时退回圆点标签。
    """

    px, py = position_to_pixel(position, metadata)
    if icon_font is not None:
        draw_centered_icon(
            draw,
            (px, py),
            symbol,
            icon_font,
            fill=fill,
            stroke_fill=stroke_fill,
            stroke_width=stroke_width,
        )
        return

    radius = max(5, int(metadata["cell_size"] * 0.28))
    draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=fill, outline=outline, width=2)

    text_width, text_height = text_size(draw, label, font)
    draw.text(
        (px - text_width / 2, py - text_height / 2),
        label,
        fill=(255, 255, 255),
        font=font,
    )


def draw_player_sprite(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    metadata: dict[str, int],
    *,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    direction: Any = None,
    mouth_fill: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """绘制旧 renderer 风格的 Pacman 玩家图形。

    输入语义：position 是玩家 tile 坐标，fill/outline 是玩家颜色和外轮廓颜色；
    direction 是当前 tile 到下一 tile 的动作方向，可为空；mouth_fill 是嘴部留空
    区域的背景色。
    输出语义：直接在当前帧上绘制一个固定开口的 Pacman。
    关键约束：这里迁移 ``PacmanRenderer._draw_pacman`` 的颜色、圆形主体和嘴部
    三角切口，但不使用动画帧；由于当前地图是 tile 级示意图，半径按单格宽度
    收敛到 0.50 个 tile，尽量提高可辨识度，同时不明显覆盖相邻格。
    """

    px, py = position_to_pixel(position, metadata)
    cell_size = metadata["cell_size"]
    radius = cell_size * 0.50
    line_width = max(1, int(cell_size * 0.04))
    bounds = [px - radius, py - radius, px + radius, py + radius]
    draw.ellipse(bounds, fill=fill, outline=outline, width=line_width)

    # 旧 renderer 用 pFrame 控制 30/60 度开口；tile 视频不做动画，因此固定为
    # 60 度开口。方向来自 04 阶段重算的 action_dir，缺失时默认朝右。
    dir_enum = direction_enum(direction)
    if dir_enum < 0:
        dir_enum = direction_enum("right")
    mouth = math.radians(60)
    direction_angle = {
        direction_enum("right"): 0.0,
        direction_enum("down"): math.pi / 2,
        direction_enum("left"): math.pi,
        direction_enum("up"): -math.pi / 2,
    }[dir_enum]
    point_a = (
        px + radius * 1.15 * math.cos(direction_angle - mouth / 2),
        py + radius * 1.15 * math.sin(direction_angle - mouth / 2),
    )
    point_b = (
        px + radius * 1.15 * math.cos(direction_angle + mouth / 2),
        py + radius * 1.15 * math.sin(direction_angle + mouth / 2),
    )
    # 当前地图道路是白色，黑嘴会显得很突兀；这里用道路背景色“挖空”，
    # 视觉上仍保留 Pacman 开口，但不会出现额外的黑色块。
    draw.polygon([(px, py), point_a, point_b], fill=mouth_fill)


def draw_ghost_sprite(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    metadata: dict[str, int],
    *,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    scared: bool = False,
    eye_only: bool = False,
    direction: Any = None,
) -> None:
    """绘制旧 renderer 风格 ghost。

    输入语义：position 是 ghost tile 坐标，fill/outline 控制主体颜色；eye_only
    表示只绘制回家状态的眼睛；scared 表示惊吓蓝色状态。
    输出语义：直接在当前帧上绘制 ghost 或眼睛。
    关键约束：迁移 ``PacmanRenderer._draw_ghost`` 的二次曲线圆顶、波浪底、白眼睛
    和蓝色瞳孔。04 tile 数据没有 ghost 方向列，因此缺失方向时使用向下眼神。
    """

    px, py = position_to_pixel(position, metadata)
    cell_size = metadata["cell_size"]
    # 旧 ghost 形状的主体宽度约为 13 个 scale 单位。这里用 cell/14 让视觉宽度
    # 接近 Pacman 的一个 tile 直径，同时留出少量边距，不让身体压住网格线。
    scale = cell_size / 14.0
    x = px - 6 * scale
    y = py - 6.5 * scale
    body_color = (33, 33, 255) if scared else fill

    if not eye_only:
        path = quadratic_path(x + 0.5 * scale, y + 6 * scale, x + 2 * scale, y, x + 7 * scale, y)
        path += quadratic_path(x + 7 * scale, y, x + 12 * scale, y, x + 13.5 * scale, y + 6 * scale)
        # 不做 ghost 帧动画，固定使用旧 renderer 偶数帧的波浪底坐标。
        coords = [13, 13, 11, 11, 9, 13, 8, 13, 8, 11, 5, 11, 5, 13, 4, 13, 2, 11, 0, 13]
        for index in range(0, len(coords), 2):
            path.append((x + (0.5 + coords[index]) * scale, y + (0.5 + coords[index + 1]) * scale))
        draw.polygon(path, fill=body_color)

    if scared:
        # 惊吓状态沿用旧 renderer 的蓝色身体、黄色眼睛和波浪嘴。
        face = (255, 255, 0)
        draw.rectangle([x + 4 * scale, y + 5 * scale, x + 6 * scale, y + 7 * scale], fill=face)
        draw.rectangle([x + 8 * scale, y + 5 * scale, x + 10 * scale, y + 7 * scale], fill=face)
        mouth_coords = [(1, 10), (2, 9), (3, 9), (4, 10), (5, 10), (6, 9), (7, 9), (8, 10), (9, 10), (10, 9), (11, 9), (12, 10)]
        mouth_points = [(x + (0.5 + a) * scale, y + (0.5 + b) * scale) for a, b in mouth_coords]
        draw.line(mouth_points, fill=face, width=max(1, int(scale)))
        return

    eye_coords = [(0, 1), (1, 0), (2, 0), (3, 1), (3, 3), (2, 4), (1, 4), (0, 3)]
    dir_enum = direction_enum(direction)
    if dir_enum < 0:
        dir_enum = DIR_DOWN
    if dir_enum == direction_enum("left"):
        xoff, yoff = -1, 0
        pxoff, pyoff = 5, 1
    elif dir_enum == direction_enum("right"):
        xoff, yoff = 1, 0
        pxoff, pyoff = 9, 1
    elif dir_enum == direction_enum("up"):
        xoff, yoff = 0, -1
        pxoff, pyoff = 7, -1
    else:
        xoff, yoff = 0, 1
        pxoff, pyoff = 7, 4

    for base in (2.5, 8.5):
        points = [(x + (xoff + base + a) * scale, y + (yoff + 3.5 + b) * scale) for a, b in eye_coords]
        draw.polygon(points, fill=(255, 255, 255))

    blue = (0, 55, 255)
    draw.rectangle([x + (2 + pxoff) * scale, y + (3 + pyoff) * scale, x + (4 + pxoff) * scale, y + (5 + pyoff) * scale], fill=blue)
    draw.rectangle([x + (-4 + pxoff) * scale, y + (3 + pyoff) * scale, x + (-2 + pxoff) * scale, y + (5 + pyoff) * scale], fill=blue)


def draw_items(
    draw: ImageDraw.ImageDraw,
    positions: list[tuple[int, int]],
    metadata: dict[str, int],
    *,
    radius: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
    kind: str,
) -> None:
    """在单帧图像上绘制旧 renderer 风格豆子或 energizer。

    输入语义：positions 是当前帧剩余的豆子/能量豆坐标；radius 是圆点半径；
    kind 标识普通豆子或 energizer。
    输出语义：直接修改传入的绘图上下文。
    关键约束：旧 renderer 中普通豆子和 energizer 都是纯色圆点，没有图标描边
    或高光；这里保留这个极简规则，让元素风格和旧图一致。
    """

    for position in positions:
        px, py = position_to_pixel(position, metadata)
        bounds = [px - radius, py - radius, px + radius, py + radius]
        draw.ellipse(bounds, fill=fill, outline=outline or fill)


def is_missing_position(value: Any) -> bool:
    """判断位置字段是否为空。

    输入语义：value 是某个动态对象的位置字段。
    输出语义：缺失值返回 True，其它坐标返回 False。
    关键约束：tuple 坐标不能直接传给 ``pd.isna`` 后做布尔判断，因为它会返回数组。
    """

    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none"}:
        return True
    return False


def render_tile_frame(
    base_image: Image.Image,
    metadata: dict[str, int],
    font: ImageFont.ImageFont,
    icon_fonts: dict[str, ImageFont.ImageFont] | None,
    row: pd.Series,
    *,
    tile_index: int,
    total_tiles: int,
    aa: int = DEFAULT_AA,
) -> np.ndarray:
    """把一条 04 corrected tile 行渲染为视频帧。

    输入语义：row 是某个 ``DayTrial`` 的一条 tile 记录。
    输出语义：返回 imageio 可写入的视频帧 numpy 数组。
    关键约束：本函数不改变底图；当 aa>1 时先在高分辨率副本上绘制，再缩回
    底图尺寸，以减少 Pacman、ghost 和豆子边缘锯齿。
    """

    if aa <= 0:
        raise TileVideoRenderError("--aa 必须大于 0。")
    if aa > 1:
        frame = base_image.resize((base_image.width * aa, base_image.height * aa), Image.Resampling.NEAREST)
        active_metadata = scale_render_metadata(metadata, aa)
        active_font = load_label_font(metadata["cell_size"] * aa)
        header_y = 8 * aa
    else:
        frame = base_image.copy()
        active_metadata = metadata
        active_font = font
        header_y = 8
    draw = ImageDraw.Draw(frame)
    header = (
        f"{row['DayTrial']}  tile {tile_index + 1}/{total_tiles}  "
        f"frame_id={int(row['frame_id'])}"
    )
    draw.text((active_metadata["origin_x"], header_y), header, fill=DISPLAY_LABEL_COLOR, font=active_font)

    item_color = (178, 45, 45)
    # 当前棋盘图的格子比旧游戏帧更稀疏，旧 renderer 的普通豆子半径会几乎不可见。
    # 因此这里保留 energizer 接近半格的视觉比例，并让普通豆子半径固定为
    # energizer 的一半，使两类食物既清楚可见又保持大小层级。
    energizer_radius = max(3, int(0.47 * active_metadata["cell_size"]))
    bean_radius = max(2, int(round(energizer_radius * 0.5)))
    if "beans" in row.index:
        draw_items(
            draw,
            parse_position_list(row["beans"]),
            active_metadata,
            radius=bean_radius,
            fill=item_color,
            outline=item_color,
            kind="bean",
        )
    if "energizers" in row.index:
        draw_items(
            draw,
            parse_position_list(row["energizers"]),
            active_metadata,
            radius=energizer_radius,
            fill=(198, 58, 52),
            outline=(198, 58, 52),
            kind="energizer",
        )

    actors = [
        ("p1_pos", "p1_alive", "p1_mode", "p1_action_dir", (236, 174, 0), (122, 88, 0)),
        ("p2_pos", "p2_alive", "p2_mode", "p2_action_dir", (70, 202, 62), (32, 126, 26)),
    ]
    for column, alive_column, mode_column, action_column, fill, outline in actors:
        if column not in row.index or is_missing_position(row[column]):
            continue
        # alive 是 02 阶段结合 mode 和位置生成的有效存活状态；它能覆盖
        # mode 已恢复但坐标仍停在死亡位置的边界帧。旧数据缺 alive 时才用 mode 兜底。
        if alive_column in row.index:
            actor_alive = bool(row[alive_column])
        else:
            actor_alive = mode_column not in row.index or int(row[mode_column]) == 1
        if not actor_alive:
            fill = (145, 145, 145)
            outline = (80, 80, 80)
        draw_player_sprite(
            draw,
            parse_grid_position(row[column]),
            active_metadata,
            fill=fill,
            outline=outline,
            direction=row[action_column] if action_column in row.index else None,
            mouth_fill=DISPLAY_WALKABLE_COLOR,
        )

    ghost_specs = [
        ("ghost1Pos", "ifscared1", (255, 26, 26)),
        ("ghost2Pos", "ifscared2", (224, 126, 0)),
    ]
    for position_column, scared_column, normal_color in ghost_specs:
        if position_column not in row.index or is_missing_position(row[position_column]):
            continue

        # 旧脚本中 ifscared 小于 3 表示普通鬼，大于等于 4 表示 scared/flashing，
        # 其它非负状态使用 eye 图标表示死亡/回家状态。
        scared_value = int(row[scared_column]) if scared_column in row.index and not pd.isna(row[scared_column]) else 0
        if 0 <= scared_value < 3:
            fill = normal_color
            scared = False
            eye_only = False
        elif scared_value >= 4:
            fill = normal_color
            scared = True
            eye_only = False
        else:
            fill = (255, 255, 255)
            scared = False
            eye_only = True

        draw_ghost_sprite(
            draw,
            parse_grid_position(row[position_column]),
            active_metadata,
            fill=fill,
            outline=fill,
            scared=scared,
            eye_only=eye_only,
        )

    if aa > 1:
        frame = frame.resize(base_image.size, Image.Resampling.LANCZOS)
    return np.asarray(frame)


def collect_tile_files(tile_root: Path, task: str) -> list[Path]:
    """收集指定任务下可用于视频绘制的 04 tile 文件。

    输入语义：tile_root 是 ``data/04_corrected_tile_data``，task 通常是 comp 或 coop。
    输出语义：返回排序后的 pkl 路径列表。
    关键约束：脚本只处理当前嵌套目录结构，不兼容旧扁平目录。
    """

    task_dir = tile_root / task
    if not task_dir.is_dir():
        raise FileNotFoundError(f"找不到任务目录：{task_dir}")
    files = sorted(task_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"任务目录下没有 pkl 文件：{task_dir}")
    return files


def choose_tile_file(tile_root: Path, task: str, session: str | None) -> Path:
    """选择要绘制的 04 corrected tile 文件。

    输入语义：session 可以是文件名或不带后缀的 session stem；为空时使用任务下第一个文件。
    输出语义：返回一个存在的 pkl 路径。
    关键约束：默认选择是为了快速生成一个示例 game 视频。
    """

    files = collect_tile_files(tile_root, task)
    if session is None:
        return files[0]

    candidates = {session, f"{session}.pkl"}
    for path in files:
        if path.name in candidates or path.stem in candidates:
            return path
    raise FileNotFoundError(f"在 {tile_root / task} 中找不到 session：{session}")


def load_game_rows(tile_path: Path, trial: str | None, max_tiles: int | None) -> pd.DataFrame:
    """读取一个 04 corrected tile 文件中的某个 game。

    输入语义：tile_path 指向 04 corrected tile pkl；trial 是可选 DayTrial。
    输出语义：返回按原顺序排列的单个 game 行。
    关键约束：如果 trial 为空，默认选择文件中第一个 DayTrial，方便快速检查。
    """

    data = pd.read_pickle(tile_path)
    required_position_columns = ("p1_pos", "p1_alive", "ghost1Pos", "ghost2Pos")
    required_columns = (*required_position_columns, *ITEM_COLUMNS)
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise TileVideoRenderError(f"{tile_path} 缺少位置字段：{missing}")
    if "DayTrial" not in data.columns or "frame_id" not in data.columns:
        raise TileVideoRenderError(f"{tile_path} 缺少 DayTrial 或 frame_id 字段。")

    if data.empty:
        raise TileVideoRenderError(f"tile 数据为空：{tile_path}")
    selected_trial = trial or str(data["DayTrial"].iloc[0])
    game_rows = data[data["DayTrial"].astype(str) == selected_trial].copy()
    if game_rows.empty:
        available = data["DayTrial"].drop_duplicates().astype(str).head(10).tolist()
        raise TileVideoRenderError(f"找不到 DayTrial={selected_trial!r}。可选示例：{available}")
    if max_tiles is not None:
        if max_tiles <= 0:
            raise TileVideoRenderError("--max-tiles 必须大于 0。")
        game_rows = game_rows.head(max_tiles).copy()
    game_rows.reset_index(drop=True, inplace=True)
    return game_rows


def default_output_path(output_dir: Path, tile_path: Path, trial: str) -> Path:
    """根据输入文件和 game 名称生成默认视频路径。

    输入语义：output_dir 是视频输出目录，tile_path/trial 标识当前绘制对象。
    输出语义：返回一个 mp4 文件路径。
    关键约束：替换 DayTrial 中的路径敏感字符，避免文件名包含斜杠或空格。
    """

    safe_trial = trial.replace("/", "_").replace(" ", "_")
    return output_dir / tile_path.parent.name / f"{tile_path.stem}__{safe_trial}.mp4"


def write_video(frames: list[np.ndarray], output_path: Path, fps: float) -> None:
    """把渲染帧写成 MP4 视频。

    输入语义：frames 是 RGB numpy 帧列表，output_path 是目标 mp4 路径。
    输出语义：在磁盘写出视频文件。
    关键约束：fps 必须为正；视频编码使用 imageio-ffmpeg，当前 LDS 环境已经可用。
    """

    if fps <= 0:
        raise TileVideoRenderError("--fps 必须大于 0。")
    if not frames:
        raise TileVideoRenderError("没有可写入的视频帧。")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)


def default_frame_output_dir(frame_output_root: Path, tile_path: Path, trial: str) -> Path:
    """根据输入文件和 game 名称生成默认图片帧目录。

    输入语义：frame_output_root 是图片帧根目录，tile_path/trial 标识当前 game。
    输出语义：返回用于保存 PNG 帧的目录。
    关键约束：目录结构包含 task、session 和 DayTrial，方便和 MP4 输出一一对应。
    """

    safe_trial = trial.replace("/", "_").replace(" ", "_")
    return frame_output_root / tile_path.parent.name / tile_path.stem / safe_trial


def save_frame_images(frames: list[np.ndarray], output_dir: Path) -> None:
    """把视频帧逐张保存为 PNG 图片。

    输入语义：frames 是已经渲染好的 RGB numpy 帧列表；output_dir 是目标目录。
    输出语义：写出 ``000000.png`` 形式的连续图片帧。
    关键约束：保存前先清理同目录下旧 PNG，避免上一版较长视频残留多余帧。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_png in output_dir.glob("*.png"):
        old_png.unlink()
    for index, frame in enumerate(frames):
        imageio.imwrite(output_dir / f"{index:06d}.png", frame)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输出语义：返回包含输入数据、地图常量、输出视频和绘图参数的命名空间。
    关键约束：默认路径全部位于当前 LoPS 仓库 ``data`` 目录下。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-root", type=Path, default=DEFAULT_TILE_ROOT)
    parser.add_argument("--map-constants", type=Path, default=DEFAULT_MAP_CONSTANTS)
    parser.add_argument("--icon-font", type=Path, default=DEFAULT_ICON_FONT, help="兼容保留参数；当前角色使用内置几何 sprite。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-path", type=Path, default=None, help="可选：显式指定 mp4 输出路径。")
    parser.add_argument("--save-frames", action="store_true", help="同时把视频帧保存为 PNG 图片。")
    parser.add_argument("--frame-output-dir", type=Path, default=DEFAULT_FRAME_OUTPUT_DIR, help="图片帧输出根目录。")
    parser.add_argument("--task", default=DEFAULT_TASK, help="任务目录名，例如 comp 或 coop。")
    parser.add_argument("--session", default=None, help="04 corrected tile 文件名或不带 .pkl 的 stem。")
    parser.add_argument("--trial", default=None, help="要绘制的 DayTrial；默认绘制文件中的第一个。")
    parser.add_argument("--max-tiles", type=int, default=None, help="可选：只绘制前 N 条 tile 行。")
    parser.add_argument("--cell-size", type=int, default=DEFAULT_CELL_SIZE)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--aa", type=int, default=DEFAULT_AA, help="动态元素抗锯齿超采样倍数；1 表示关闭。")
    return parser.parse_args()


def main() -> None:
    """命令行入口：读取 04 corrected tile 数据并生成一个简易 game 视频。"""

    args = parse_args()
    tile_path = choose_tile_file(args.tile_root, args.task, args.session)
    game_rows = load_game_rows(tile_path, args.trial, args.max_tiles)
    walkable_positions = load_walkable_positions(args.map_constants)
    base_image, metadata, font = build_base_map_image(walkable_positions, cell_size=args.cell_size)
    icon_fonts = None

    frames = [
        render_tile_frame(
            base_image,
            metadata,
            font,
            icon_fonts,
            row,
            tile_index=index,
            total_tiles=len(game_rows),
            aa=args.aa,
        )
        for index, row in game_rows.iterrows()
    ]

    selected_trial = str(game_rows["DayTrial"].iloc[0])
    output_path = args.output_path or default_output_path(args.output_dir, tile_path, selected_trial)
    write_video(frames, output_path, args.fps)
    frame_output_dir = default_frame_output_dir(args.frame_output_dir, tile_path, selected_trial)
    if args.save_frames:
        save_frame_images(frames, frame_output_dir)

    print("tile 视频生成完成")
    print(f"输入文件：{tile_path.resolve()}")
    print(f"DayTrial：{selected_trial}")
    print(f"tile 帧数：{len(frames)}")
    print(f"fps：{args.fps}")
    print(f"aa：{args.aa}")
    print(f"地图范围：x={metadata['x_min']}..{metadata['x_max']}, y={metadata['y_min']}..{metadata['y_max']}")
    print("角色图形：旧 renderer 静态 sprite")
    print(f"输出视频：{output_path.resolve()}")
    if args.save_frames:
        print(f"输出图片帧：{frame_output_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except TileVideoRenderError as exc:
        raise SystemExit(str(exc)) from exc
