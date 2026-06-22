#!/usr/bin/env python3
"""不依赖 MATLAB/Psychtoolbox 的 Pacman 帧渲染器。

脚本读取 ``prepare_render_data.py`` 生成的 merged frame PKL，使用 Pillow
直接绘制地图、豆子、Pacman、ghost、Actual/Model 方向箭头和底部信息 bar。
底部信息 bar 可通过 ``--bar-type`` 选择 grammar、strategy 或不画。当前默认
render table 已经只保留两个鬼 trial；如果输入仍包含四鬼 trial，
本渲染器也只绘制 g1/g2，因为视觉样式是按旧两鬼结果图复现的。
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# 内部方向枚举沿用 MATLAB/旧渲染逻辑，方便复用墙体追踪和动画方向计算。
DIR_UP = 0
DIR_LEFT = 1
DIR_DOWN = 2
DIR_RIGHT = 3

# 原始 Map 是 29 x 36 的 tile 字符串；输出画布按旧结果图比例固定。
NUM_COLS = 29
NUM_ROWS = 36
BASE_TILE = 25
BASE_WIDTH = 812
BASE_HEIGHT = 1170
BASE_BOARD_X_OFFSET = 43
BASE_MAP_SIDE_MARGIN = 17
BASE_MAP_BOTTOM = 870
BASE_LEFT_TUNNEL_CLIP_X = 43
BASE_LEFT_TUNNEL_CLIP_RANGES = ((398, 411), (464, 476))
# grammar bar 的位置和大小使用输出图像坐标；绘制时会乘以 aa 超采样倍数。
GRAM_BAR_Y = 986
GRAM_BAR_HEIGHT = 54
GRAM_BAR_MAX_WIDTH = 620
GRAM_BAR_MIN_SEGMENT_WIDTH = 118
GRAM_BAR_GAP = 3
# gram 字符串中的单字符编码到显示标签的映射。
GRAM_TOKEN_TO_LABEL = {
    "G": "global",
    "L": "local",
    "1": "evade",
    "2": "evade",
    "E": "energizer",
    "A": "approach",
    "S": "stay",
    "V": "vague",
    "N": "no energizer",
}
GRAM_LABEL_COLORS = {
    "global": (69, 180, 61),
    "local": (215, 25, 28),
    "evade": (254, 175, 97),
    "energizer": (131, 106, 183),
    "approach": (128, 179, 255),
    "stay": (200, 200, 200),
    "vague": (120, 120, 120),
    "no energizer": (203, 203, 60),
}
BAR_TYPE_CHOICES = {"grammar", "strategy", "none"}
@dataclass
class Point:
    """墙体轮廓追踪过程中的控制点。

    ``x/y`` 是当前轮廓点；``cx/cy`` 在需要圆角转弯时保存二次贝塞尔控制点。
    """

    x: float
    y: float
    cx: float = -1.0
    cy: float = -1.0


def direction_enum(value) -> int:
    """把英文方向字符串转成内部方向枚举；未知值返回 -1。"""

    if not isinstance(value, str):
        return -1
    value = value.strip().lower()
    if value == "up":
        return DIR_UP
    if value == "left":
        return DIR_LEFT
    if value == "down":
        return DIR_DOWN
    if value == "right":
        return DIR_RIGHT
    return -1


def direction_from_code(value) -> str | None:
    """把 render table 中的 1-4 方向编码转成英文方向。

    编码来自旧分析表：1=up，2=down，3=left，4=right。
    """

    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return {1: "up", 2: "down", 3: "left", 4: "right"}.get(value)


def clean_label(value) -> str:
    """清理策略标签；当前保留给调试和兼容旧调用。"""

    if not isinstance(value, str) or not value.strip():
        return "unknown"
    return value.strip()


def sanitize_name(value: str) -> str:
    """把 subject/trial 名变成安全的文件夹名。"""

    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return value.strip("-") or "trial"


def quadratic_path(
    p0x: float,
    p0y: float,
    p1x: float,
    p1y: float,
    p2x: float,
    p2y: float,
    steps: int = 8,
) -> list[tuple[float, float]]:
    """生成二次贝塞尔曲线采样点，用于墙体圆角和局部曲线。"""

    pts: list[tuple[float, float]] = []
    for i in range(steps + 1):
        t = i / steps
        px = (1 - t) * (1 - t) * p0x + 2 * (1 - t) * t * p1x + t * t * p2x
        py = (1 - t) * (1 - t) * p0y + 2 * (1 - t) * t * p1y + t * t * p2y
        pts.append((px, py))
    return pts


class StaticMap:
    """根据 Map 字符串重建静态地图轮廓。

    旧 MATLAB 代码不是逐 tile 画矩形墙，而是沿着墙体边缘追踪轮廓并绘制闭合
    多边形。这里复现这个思路，使输出更接近最终参考图：墙体线条连续、转角平滑，
    隧道边界也能保持旧布局。
    """

    def __init__(self, tiles: str, tile_size: int, scale: float) -> None:
        """初始化静态地图模型并预解析墙体轮廓。

        输入语义：tiles 是 29x36 展开的 Map 字符串，tile_size 和 scale 是渲染坐标比例。
        输出语义：实例会缓存隧道、墙体边缘、墙体多边形和 ghost house 位置。
        关键约束：Map 长度必须固定为 1044，否则墙体追踪无法与逐帧数据对齐。
        """

        if len(tiles) != NUM_COLS * NUM_ROWS:
            raise ValueError(f"Map length is {len(tiles)}, expected {NUM_COLS * NUM_ROWS}.")
        self.tiles = tiles
        self.tile = tile_size
        self.scale = scale
        self.tunnel_rows: dict[int, tuple[int, int]] = {}
        self.edges: dict[tuple[int, int], bool] = {}
        self.walls: list[list[tuple[float, float]]] = []
        self.ghost_house_tile: tuple[int, int] | None = None
        self._parse_tunnels()
        self._parse_edges()
        self._parse_wall_paths()
        self._find_ghost_house()

    def pos_to_index(self, x: int, y: int) -> int:
        """把 1-based tile 坐标转换成 Map 字符串下标。"""

        return (x - 1) + (y - 1) * NUM_COLS

    def to_index(self, x: int, y: int) -> int:
        """把包含左右隧道扩展区的坐标转换成旧算法使用的线性索引。"""

        if x > -2 and x < NUM_COLS + 3 and 0 < y <= NUM_ROWS:
            return (x + 2) + (y - 1) * (NUM_COLS + 4)
        return 0

    def tile_at(self, x: int, y: int, *, with_tunnels: bool = True) -> str:
        """读取 tile 字符，并在需要时合成左右隧道的虚拟 tile。"""

        if 0 < x <= NUM_COLS and 0 < y <= NUM_ROWS:
            return self.tiles[self.pos_to_index(x, y)]
        if with_tunnels and (x < 1 or x > NUM_COLS):
            if self.is_tunnel_tile(x, y - 1) or self.is_tunnel_tile(x, y + 1):
                return "|"
        if with_tunnels and self.is_tunnel_tile(x, y):
            return " "
        return "\0"

    def is_tunnel_tile(self, x: int, y: int) -> bool:
        """判断坐标是否位于左右隧道的画布外延伸区。"""

        row = self.tunnel_rows.get(y)
        if row is None:
            return False
        left, right = row
        return left != -1 and (x < left or x > right)

    def is_floor_tile(self, x: int, y: int, *, with_tunnels: bool = True) -> bool:
        """判断 tile 是否可通行；水果、豆子和空格都属于 floor。"""

        return self.tile_at(x, y, with_tunnels=with_tunnels) in {" ", ".", "o", "C", "S", "O", "A", "M"}

    def _get_tunnel_entrance(self, x: int, y: int, dx: int) -> int:
        """从边缘向内寻找隧道入口，确定需要延伸到画布外的行。"""

        while (
            not self.is_floor_tile(x, y - 1, with_tunnels=False)
            and not self.is_floor_tile(x, y + 1, with_tunnels=False)
            and self.is_floor_tile(x, y, with_tunnels=False)
        ):
            x += dx
        return x

    def _parse_tunnels(self) -> None:
        """预解析每一行的左右隧道入口。"""

        for y in range(1, NUM_ROWS + 1):
            if self.is_floor_tile(1, y, with_tunnels=False) and self.is_floor_tile(
                NUM_COLS, y, with_tunnels=False
            ):
                left = self._get_tunnel_entrance(1, y, 1)
                right = self._get_tunnel_entrance(NUM_COLS, y, -1)
                self.tunnel_rows[y] = (left, right)
            else:
                self.tunnel_rows[y] = (-1, -1)

    def _parse_edges(self) -> None:
        """标记墙体边缘 tile，为后续轮廓追踪做准备。"""

        for y in range(1, NUM_ROWS + 1):
            for x in range(-1, NUM_COLS + 3):
                is_edge = self.tile_at(x, y) == "|" and any(
                    self.tile_at(nx, ny) != "|"
                    for nx, ny in (
                        (x - 1, y),
                        (x + 1, y),
                        (x, y - 1),
                        (x, y + 1),
                        (x - 1, y - 1),
                        (x - 1, y + 1),
                        (x + 1, y - 1),
                        (x + 1, y + 1),
                    )
                )
                self.edges[(x, y)] = is_edge

    @staticmethod
    def _dir_vec(dir_enum: int) -> tuple[int, int]:
        """把内部方向枚举转换为 tile 坐标增量。"""

        if dir_enum == DIR_UP:
            return (0, -1)
        if dir_enum == DIR_LEFT:
            return (-1, 0)
        if dir_enum == DIR_DOWN:
            return (0, 1)
        if dir_enum == DIR_RIGHT:
            return (1, 0)
        return (0, 0)

    @staticmethod
    def _rotate_left(dir_enum: int) -> int:
        """返回当前方向左转后的方向枚举。"""

        return (dir_enum + 1) % 4

    @staticmethod
    def _rotate_right(dir_enum: int) -> int:
        """返回当前方向右转后的方向枚举。"""

        return (dir_enum + 3) % 4

    @staticmethod
    def _rotate_about_face(dir_enum: int) -> int:
        """返回当前方向掉头后的方向枚举。"""

        return (dir_enum + 2) % 4

    def _start_point(self, tx: int, ty: int, dir_enum: int, pad: float) -> tuple[Point, float]:
        """计算当前边缘 tile 在像素空间中的轮廓起点。"""

        dx, dy = self._dir_vec(dir_enum)
        left_idx = self.to_index(tx + dy, ty - dx)
        if left_idx != 0 and not self.edges.get((tx + dy, ty - dx), False):
            if self.is_floor_tile(tx + dy, ty - dx):
                pad = math.floor(5 * self.scale)
            else:
                pad = 0

        px = -self.tile / 2 + pad
        py = self.tile / 2
        c = math.cos(-dir_enum * math.pi / 2)
        s = math.sin(-dir_enum * math.pi / 2)
        x = (px * c - py * s) + (tx - 0.5) * self.tile + 1
        y = (px * s + py * c) + (ty - 0.5) * self.tile
        return Point(x, y), pad

    def _parse_wall_paths(self) -> None:
        """沿墙体边缘追踪闭合路径。

        ``excluded`` 和 ``included`` 是旧 MATLAB 绘图中特殊处理的墙体索引，
        用于避免 ghost house 附近的小边界被重复或漏画。
        """

        visited: set[tuple[int, int]] = set()
        excluded = {560, 561, 626, 627}
        included = {557, 558, 559, 623, 624, 625}

        for y in range(1, NUM_ROWS + 1):
            for x in range(-1, NUM_COLS + 3):
                idx = self.to_index(x, y)
                if not self.edges.get((x, y), False):
                    continue
                if not (((x, y) not in visited and idx not in excluded) or idx in included):
                    continue

                visited.add((x, y))
                tx, ty = x, y
                if self.edges.get((tx + 1, ty), False):
                    dir_enum = DIR_RIGHT
                elif self.edges.get((tx, ty + 1), False):
                    dir_enum = DIR_DOWN
                else:
                    continue

                dx, dy = self._dir_vec(dir_enum)
                tx += dx
                ty += dy
                init_tx, init_ty, init_dir = tx, ty, dir_enum
                turn = False
                pad = 0.0
                path: list[Point] = []

                for _ in range(10000):
                    visited.add((tx, ty))
                    point, pad = self._start_point(tx, ty, dir_enum, pad)

                    if turn and path:
                        last = path[-1]
                        if dx == 0:
                            point.cx = point.x
                            point.cy = last.y
                        else:
                            point.cx = last.x
                            point.cy = point.y

                    turn = False
                    turn_around = False
                    if self.edges.get((tx + dy, ty - dx), False):
                        dir_enum = self._rotate_left(dir_enum)
                        turn = True
                    elif self.edges.get((tx + dx, ty + dy), False):
                        pass
                    elif self.edges.get((tx - dy, ty + dx), False):
                        dir_enum = self._rotate_right(dir_enum)
                        turn = True
                    else:
                        dir_enum = self._rotate_about_face(dir_enum)
                        turn_around = True

                    new_dx, new_dy = self._dir_vec(dir_enum)
                    path.append(point)

                    if turn_around:
                        extra, pad = self._start_point(tx - new_dx, ty - new_dy, self._rotate_about_face(dir_enum), pad)
                        path.append(extra)
                        extra, pad = self._start_point(tx, ty, dir_enum, pad)
                        path.append(extra)

                    tx += new_dx
                    ty += new_dy
                    dx, dy = new_dx, new_dy

                    if tx == init_tx and ty == init_ty and dir_enum == init_dir:
                        if path:
                            self.walls.append(self._wall_from_path(path))
                        break

    def _wall_from_path(self, path: list[Point]) -> list[tuple[float, float]]:
        """把追踪到的控制点转换成可传给 Pillow 的多边形点集。"""

        wall: list[tuple[float, float]] = [(path[0].x, path[0].y)]
        for point in path[1:]:
            if point.cx >= 0:
                wall.extend(quadratic_path(wall[-1][0], wall[-1][1], point.cx, point.cy, point.x, point.y))
            else:
                wall.append((point.x, point.y))
        wall.extend(quadratic_path(wall[-1][0], wall[-1][1], path[-1].x, path[0].y, path[0].x, path[0].y))
        return wall

    def _find_ghost_house(self) -> None:
        """定位 ghost house 门口的 '-' tile，用于绘制粉色门。"""

        for y in range(1, NUM_ROWS + 1):
            for x in range(1, NUM_COLS):
                i = self.pos_to_index(x, y)
                if self.tiles[i] == "-" and self.tiles[i + 1] == "-":
                    self.ghost_house_tile = (x, y)
                    return


class PacmanRenderer:
    """单帧渲染器。

    一个 renderer 会缓存首帧 Map 对应的静态墙体轮廓和字体。实际逐帧绘制时，
    动态对象来自每一行数据：Pacman/ghost 坐标、Map 中剩余豆子、方向和信息 bar。
    """

    def __init__(self, first_map: str, aa: int = 3, bar_type: str = "grammar") -> None:
        """初始化单帧渲染器。

        输入语义：first_map 用于缓存静态地图轮廓，aa 控制超采样抗锯齿倍数，bar_type 控制底部信息栏。
        输出语义：实例保存画布尺寸、字体、地图模型和绘制参数，可重复渲染同一地图布局的多帧。
        关键约束：当前渲染器只绘制 g1/g2，调用方应在 render table 阶段过滤或明确接受该边界。
        """

        self.aa = aa
        self.bar_type = self._normalize_bar_type(bar_type)
        self.width = BASE_WIDTH * aa
        self.height = BASE_HEIGHT * aa
        self.tile = BASE_TILE * aa
        self.scale = (BASE_TILE / 8.0) * aa
        self.mid = math.ceil(BASE_TILE / 2) * aa
        self.board_x_offset = BASE_BOARD_X_OFFSET * aa
        self.static_map = StaticMap(first_map, self.tile, self.scale)

        self.font_regular = self._font("DejaVuSans.ttf", 38 * aa)
        self.font_bold = self._font("DejaVuSans-Bold.ttf", 38 * aa)
        self.font_label = self._font("DejaVuSans.ttf", 24 * aa)
        self.font_small = self._font("DejaVuSans.ttf", 18 * aa)

    @staticmethod
    def _normalize_bar_type(value: str) -> str:
        """规范化底部信息 bar 类型。"""

        normalized = str(value).strip().lower()
        if normalized not in BAR_TYPE_CHOICES:
            raise ValueError(f"bar_type 只支持 {sorted(BAR_TYPE_CHOICES)}，当前值：{value!r}")
        return normalized

    @staticmethod
    def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
        """按优先级加载字体；缺失时回退到 Pillow 默认字体。"""

        candidates = [
            f"/usr/share/fonts/truetype/dejavu/{name}",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _xy(self, x: float, y: float) -> tuple[int, int]:
        """把基础画布坐标转换为超采样画布坐标。"""

        return (int(round(x * self.aa)), int(round(y * self.aa)))

    def render(self, row: pd.Series) -> Image.Image:
        """把 render table 的一行数据绘制成一张 RGB 图片。"""

        image = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        draw = ImageDraw.Draw(image, "RGBA")

        self._draw_map(draw, str(row["Map"]))
        self._mask_map_side_margins(draw)
        self._draw_ghost(
            draw,
            row["g1pX"] * self.aa + self.board_x_offset,
            row["g1pY"] * self.aa,
            int(row["g1Frame"]),
            direction_enum(row["g1Dir"]),
            bool(row["g1Scared"]),
            int(row["g1ModeR"]) in {2, 3},
            (255, 26, 26, 230),
        )
        self._draw_ghost(
            draw,
            row["g2pX"] * self.aa + self.board_x_offset,
            row["g2pY"] * self.aa,
            int(row["g2Frame"]),
            direction_enum(row["g2Dir"]),
            bool(row["g2Scared"]),
            int(row["g2ModeR"]) in {2, 3},
            (255, 188, 91, 230),
        )
        self._draw_pacman(
            draw,
            row["ppX"] * self.aa + self.board_x_offset,
            row["ppY"] * self.aa,
            direction_enum(row["pDir"]),
            int(row["pFrame"]),
        )
        self._draw_hud(draw, row)
        self._draw_selected_bar(draw, row)

        # 先在高分辨率画布上绘制，再下采样到目标尺寸，边缘会更平滑。
        if self.aa > 1:
            image = image.resize((BASE_WIDTH, BASE_HEIGHT), Image.Resampling.LANCZOS)
        return image

    def _mask_map_side_margins(self, draw: ImageDraw.ImageDraw) -> None:
        """遮掉地图左右边缘和左侧隧道多余线段，匹配后期处理后的视觉边距。"""

        margin = BASE_MAP_SIDE_MARGIN * self.aa
        bottom = BASE_MAP_BOTTOM * self.aa
        draw.rectangle([0, 0, margin, bottom], fill=(0, 0, 0, 255))
        draw.rectangle([self.width - margin, 0, self.width, bottom], fill=(0, 0, 0, 255))
        clip_x = (BASE_LEFT_TUNNEL_CLIP_X - 1) * self.aa
        for top, bottom in BASE_LEFT_TUNNEL_CLIP_RANGES:
            draw.rectangle([0, top * self.aa, clip_x, bottom * self.aa], fill=(0, 0, 0, 255))

    def _draw_map(self, draw: ImageDraw.ImageDraw, map_tiles: str) -> None:
        """绘制静态墙体、ghost house 门、豆子和能量豆。"""

        # 先画一层较粗的淡蓝色光晕，再画黑色墙体面和亮色轮廓。
        ox = self.board_x_offset
        for wall in self.static_map.walls:
            pts = [(int(round(x + ox)), int(round(y))) for x, y in wall]
            if len(pts) < 2:
                continue
            draw.line(pts + [pts[0]], fill=(48, 72, 132, 150), width=max(2, int(3.0 * self.aa)), joint="curve")
        for wall in self.static_map.walls:
            pts = [(int(round(x + ox)), int(round(y))) for x, y in wall]
            if len(pts) < 3:
                continue
            draw.polygon(pts, fill=(0, 0, 0, 255))
            draw.line(pts + [pts[0]], fill=(210, 214, 255, 245), width=max(1, int(1.15 * self.aa)), joint="curve")

        if self.static_map.ghost_house_tile:
            x, y = self.static_map.ghost_house_tile
            rect = [
                (x - 1) * self.tile + ox,
                y * self.tile - 2 * self.scale,
                (x + 1) * self.tile + ox,
                y * self.tile,
            ]
            draw.rounded_rectangle(rect, radius=int(1.2 * self.aa), fill=(255, 184, 222, 255))

        pellet_color = (255, 190, 180, 255)
        energizer_color = (255, 160, 152, 255)
        for y in range(1, NUM_ROWS + 1):
            for x in range(1, NUM_COLS + 1):
                tile = map_tiles[(x - 1) + (y - 1) * NUM_COLS]
                cx = (x - 1) * self.tile + self.mid + ox
                cy = (y - 1) * self.tile + self.mid
                if tile == ".":
                    r = max(2, int(math.floor(1.05 * self.scale)))
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=pellet_color)
                elif tile == "o":
                    r = int(0.47 * self.tile)
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=energizer_color)

        # 复现旧 MATLAB 隧道 mask，隐藏右侧隧道处的一小段边缘伪影。
        draw.rectangle(
            [
                725 * self.aa + ox + 1,
                900 * self.aa / 2 - 51 * self.aa,
                725 * self.aa + ox + 52 * self.aa,
                900 * self.aa / 2 + 26 * self.aa,
            ],
            fill=(0, 0, 0, 255),
        )

    def _draw_pacman(self, draw: ImageDraw.ImageDraw, x: float, y: float, dir_enum: int, frame: int) -> None:
        """绘制 Pacman，嘴巴开合由 pFrame 控制。"""

        radius = self.tile * 0.9
        cx = x
        cy = y + self.scale
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(255, 238, 16, 245),
            outline=(255, 255, 120, 190),
            width=max(1, int(1.2 * self.aa)),
        )
        if dir_enum < 0:
            return

        body_angle = 330 - (math.floor(frame / 4) % 2) * 30
        mouth = math.radians(360 - body_angle)
        direction_angle = {
            DIR_RIGHT: 0.0,
            DIR_DOWN: math.pi / 2,
            DIR_LEFT: math.pi,
            DIR_UP: -math.pi / 2,
        }[dir_enum]
        p1 = (cx + radius * 1.15 * math.cos(direction_angle - mouth / 2), cy + radius * 1.15 * math.sin(direction_angle - mouth / 2))
        p2 = (cx + radius * 1.15 * math.cos(direction_angle + mouth / 2), cy + radius * 1.15 * math.sin(direction_angle + mouth / 2))
        draw.polygon([(cx, cy), p1, p2], fill=(0, 0, 0, 255))

    def _draw_ghost(
        self,
        draw: ImageDraw.ImageDraw,
        x: float,
        y: float,
        frame: int,
        dir_enum: int,
        scared: bool,
        eyes_only: bool,
        color: tuple[int, int, int, int],
    ) -> None:
        """绘制单个 ghost。

        ``eyes_only`` 对应旧游戏中的 eaten/返回房间状态；``scared`` 控制蓝色惊吓
        状态。当前主渲染流程只调用 g1/g2，因为默认 render table 已过滤为两鬼 trial。
        """

        s = self.scale
        x -= 6 * s
        y -= 6.5 * s
        flash = False
        body_color = (33, 33, 255, 245) if scared and not flash else color

        if not eyes_only:
            path = quadratic_path(x + 0.5 * s, y + 6 * s, x + 2 * s, y, x + 7 * s, y)
            path += quadratic_path(x + 7 * s, y, x + 12 * s, y, x + 13.5 * s, y + 6 * s)
            if (frame // 8) % 2 == 0:
                coords = [13, 13, 11, 11, 9, 13, 8, 13, 8, 11, 5, 11, 5, 13, 4, 13, 2, 11, 0, 13]
            else:
                coords = [13, 12, 12, 13, 11, 13, 9, 11, 7, 13, 6, 13, 4, 11, 2, 13, 1, 13, 0, 12]
            for i in range(0, len(coords), 2):
                path.append((x + (0.5 + coords[i]) * s, y + (0.5 + coords[i + 1]) * s))
            draw.polygon(path, fill=body_color)

        if scared:
            face = (255, 255, 0, 255)
            draw.rectangle([x + 4 * s, y + 5 * s, x + 6 * s, y + 7 * s], fill=face)
            draw.rectangle([x + 8 * s, y + 5 * s, x + 10 * s, y + 7 * s], fill=face)
            coords = [(1, 10), (2, 9), (3, 9), (4, 10), (5, 10), (6, 9), (7, 9), (8, 10), (9, 10), (10, 9), (11, 9), (12, 10)]
            pts = [(x + (0.5 + a) * s, y + (0.5 + b) * s) for a, b in coords]
            draw.line(pts, fill=face, width=max(1, int(s)))
            return

        eye_coords = [(0, 1), (1, 0), (2, 0), (3, 1), (3, 3), (2, 4), (1, 4), (0, 3)]
        if dir_enum == DIR_LEFT:
            xoff, yoff = -1, 0
            pxoff, pyoff = 5, 1
        elif dir_enum == DIR_RIGHT:
            xoff, yoff = 1, 0
            pxoff, pyoff = 9, 1
        elif dir_enum == DIR_UP:
            xoff, yoff = 0, -1
            pxoff, pyoff = 7, -1
        else:
            xoff, yoff = 0, 1
            pxoff, pyoff = 7, 4

        for base in (2.5, 8.5):
            pts = [(x + (xoff + base + a) * s, y + (yoff + 3.5 + b) * s) for a, b in eye_coords]
            draw.polygon(pts, fill=(255, 255, 255, 255))

        blue = (0, 55, 255, 255)
        draw.rectangle([x + (2 + pxoff) * s, y + (3 + pyoff) * s, x + (4 + pxoff) * s, y + (5 + pyoff) * s], fill=blue)
        draw.rectangle([x + (-4 + pxoff) * s, y + (3 + pyoff) * s, x + (-2 + pxoff) * s, y + (5 + pyoff) * s], fill=blue)

    def _draw_hud(self, draw: ImageDraw.ImageDraw, row: pd.Series) -> None:
        """绘制底部 Actual/Model 方向行。

        如果某一帧缺少 actual_dir 或 multi_dir，只跳过缺失的那一组；如果两者都缺失，
        整个方向行不画，但游戏帧仍然正常输出。
        """

        groups: list[tuple[str, str]] = []
        actual_direction = direction_from_code(row.get("actual_dir"))
        model_direction = direction_from_code(row.get("multi_dir"))
        if actual_direction is not None:
            groups.append(("Actual:", actual_direction))
        if model_direction is not None:
            groups.append(("Model:", model_direction))
        if not groups:
            return

        text_arrow_gap = 18 * self.aa
        group_gap = 110 * self.aa if len(groups) > 1 else 0
        arrow_size = 48 * self.aa
        cy = 920 * self.aa

        group_widths = [
            self._text_size(draw, text, self.font_regular)[0] + text_arrow_gap + arrow_size
            for text, _direction in groups
        ]
        total_width = sum(group_widths) + group_gap * (len(groups) - 1)
        x = (self.width - total_width) / 2

        for (text, direction), group_width in zip(groups, group_widths):
            text_width = self._text_size(draw, text, self.font_regular)[0]
            arrow_x = x + text_width + text_arrow_gap + arrow_size / 2
            self._draw_centered_y_text(draw, text, x, cy, self.font_regular, (250, 250, 250, 255))
            self._draw_arrow(draw, direction, arrow_x, cy, arrow_size)
            x += group_width + group_gap

    def _draw_selected_bar(self, draw: ImageDraw.ImageDraw, row: pd.Series) -> None:
        """按参数选择绘制 grammar bar、strategy bar 或跳过 bar。

        字段缺失或当前帧字段为空时直接返回，不影响游戏帧和方向箭头渲染。
        """

        if self.bar_type == "none":
            return
        if self.bar_type == "strategy":
            self._draw_strategy_bar(draw, row)
            return
        self._draw_grammar_bar(draw, row)

    def _draw_grammar_bar(self, draw: ImageDraw.ImageDraw, row: pd.Series) -> None:
        """绘制 grammar bar。

        ``gram`` 是由单字符组成的 grammar 序列，``gram_num`` 表示当前激活片段。
        bar 会根据 token 数量计算总宽度，并始终水平居中。
        """

        gram_value = row.get("gram")
        if gram_value is None or pd.isna(gram_value):
            return
        tokens = [token for token in str(gram_value).strip() if token in GRAM_TOKEN_TO_LABEL]
        if not tokens:
            return

        try:
            active_index = int(float(row.get("gram_num", 0)))
        except (TypeError, ValueError):
            active_index = 0
        active_index = max(0, min(active_index, len(tokens) - 1))

        segment_width = min(
            GRAM_BAR_MIN_SEGMENT_WIDTH * self.aa,
            (GRAM_BAR_MAX_WIDTH * self.aa - GRAM_BAR_GAP * self.aa * (len(tokens) - 1)) / len(tokens),
        )
        total_width = segment_width * len(tokens) + GRAM_BAR_GAP * self.aa * (len(tokens) - 1)
        left = (self.width - total_width) / 2
        top = GRAM_BAR_Y * self.aa
        height = GRAM_BAR_HEIGHT * self.aa

        for idx, token in enumerate(tokens):
            label = GRAM_TOKEN_TO_LABEL[token]
            color = GRAM_LABEL_COLORS[label]
            x0 = left + idx * (segment_width + GRAM_BAR_GAP * self.aa)
            x1 = x0 + segment_width
            rect = [x0, top, x1, top + height]
            draw.rounded_rectangle(rect, radius=4 * self.aa, fill=color + (238,))
            if idx == active_index:
                draw.rounded_rectangle(
                    rect,
                    radius=4 * self.aa,
                    outline=(255, 255, 255, 255),
                    width=max(2, int(2.0 * self.aa)),
                )
            self._draw_text_fit_centered(draw, label, rect, self.font_label, (255, 255, 255, 245))

    def _draw_strategy_bar(self, draw: ImageDraw.ImageDraw, row: pd.Series) -> None:
        """绘制 strategy bar。

        当前 render table 中 strategy 通常保存为 ``fitted_label``。为了兼容旧数据，
        如果存在 ``strategy`` 字段也可以读取；两者都不存在或当前值为空时跳过。
        """

        label_value = row.get("fitted_label")
        if label_value is None or pd.isna(label_value):
            label_value = row.get("strategy")
        if label_value is None or pd.isna(label_value):
            return

        label = clean_label(label_value)
        if label == "unknown":
            return

        color = self._label_color(label)
        top = GRAM_BAR_Y * self.aa
        height = GRAM_BAR_HEIGHT * self.aa
        padding = 30 * self.aa
        text_width = self._text_size(draw, label, self.font_label)[0]
        width = min(GRAM_BAR_MAX_WIDTH * self.aa, max(180 * self.aa, text_width + padding * 2))
        left = (self.width - width) / 2
        rect = [left, top, left + width, top + height]
        draw.rounded_rectangle(rect, radius=4 * self.aa, fill=color + (238,))
        draw.rounded_rectangle(
            rect,
            radius=4 * self.aa,
            outline=(255, 255, 255, 255),
            width=max(2, int(2.0 * self.aa)),
        )
        self._draw_text_fit_centered(draw, label, rect, self.font_label, (255, 255, 255, 245))

    def _draw_arrow(self, draw: ImageDraw.ImageDraw, direction: str | None, cx: float, cy: float, size: float) -> None:
        """绘制标准对称方向箭头；方向缺失时不画任何占位符。"""

        if direction is None:
            return
        base = np.array(
            [
                [-0.50, -0.18],
                [0.08, -0.18],
                [0.08, -0.36],
                [0.50, 0.00],
                [0.08, 0.36],
                [0.08, 0.18],
                [-0.50, 0.18],
            ]
        )
        angle = {"right": 0.0, "down": math.pi / 2, "left": math.pi, "up": -math.pi / 2}[direction]
        rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        pts = base @ rot.T
        pts = [(cx + x * size, cy + y * size) for x, y in pts]
        draw.polygon(pts, fill=(255, 255, 255, 255))

    @staticmethod
    def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
        """返回文字绘制后的宽高。"""

        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]

    @staticmethod
    def _draw_centered_y_text(
        draw: ImageDraw.ImageDraw,
        text: str,
        x: float,
        cy: float,
        font: ImageFont.FreeTypeFont,
        fill: tuple[int, int, int, int],
    ) -> None:
        """在给定 y 中线上绘制文字，x 使用左对齐。"""

        box = draw.textbbox((0, 0), text, font=font)
        y = cy - (box[3] - box[1]) / 2 - box[1]
        draw.text((x, y), text, fill=fill, font=font)

    def _centered_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        rect: Sequence[float],
        font: ImageFont.FreeTypeFont,
        fill: tuple[int, int, int, int],
    ) -> None:
        """在矩形中水平垂直居中文字。"""

        box = draw.textbbox((0, 0), text, font=font)
        tw = box[2] - box[0]
        th = box[3] - box[1]
        x = rect[0] + (rect[2] - rect[0] - tw) / 2
        y = rect[1] + (rect[3] - rect[1] - th) / 2 - 2 * self.aa
        draw.text((x, y), text, fill=fill, font=font)

    def _draw_text_fit_centered(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        rect: Sequence[float],
        font: ImageFont.FreeTypeFont,
        fill: tuple[int, int, int, int],
    ) -> None:
        """在矩形中居中文字；文字过长时缩小字体避免溢出。"""

        box = draw.textbbox((0, 0), text, font=font)
        tw = box[2] - box[0]
        th = box[3] - box[1]
        max_width = rect[2] - rect[0] - 10 * self.aa
        if tw > max_width:
            scale = max_width / max(tw, 1)
            size = max(10 * self.aa, int(font.size * scale))
            font = self._font("DejaVuSans.ttf", size)
            box = draw.textbbox((0, 0), text, font=font)
            tw = box[2] - box[0]
            th = box[3] - box[1]
        x = rect[0] + (rect[2] - rect[0] - tw) / 2
        y = rect[1] + (rect[3] - rect[1] - th) / 2 - box[1]
        draw.text((x, y), text, fill=fill, font=font)

    @staticmethod
    def _label_color(label: str) -> tuple[int, int, int]:
        """策略标签到颜色的旧兼容映射，当前主要保留给后续扩展。"""

        normalized = label.strip().lower()
        return {
            "approach": (128, 179, 255),
            "planned_hunting": (131, 106, 183),
            "energizer": (131, 106, 183),
            "global": (69, 180, 61),
            "pessimistic": (196, 81, 92),
            "evade(blinky)": (196, 81, 92),
            "evade(clyde)": (255, 208, 125),
            "local": (215, 25, 28),
            "vague": (173, 173, 173),
            "stay": (25, 25, 25),
            "no energizer": (214, 217, 49),
            "evade": (254, 175, 97),
        }.get(normalized, (95, 130, 190))


def load_render_rows(
    *,
    render_table_dir: Path,
    subject: str,
    trial: str | None = None,
    start: int = 0,
    max_frames: int | None = None,
) -> pd.DataFrame:
    """读取渲染输入表，并按 trial/start/max_frames 做可选筛选。

    注意：这里不会因为 grammar 或方向缺失而丢帧。缺失标注只会让对应视觉层
    不显示，游戏本身仍然会渲染。
    """

    data_path = resolve_render_table_path(render_table_dir, subject)
    merged = pd.read_pickle(data_path)
    # 这些是画出游戏帧的最低字段；grammar/方向不是必需列，因为可能有些帧无标注。
    required_columns = {
        "DayTrial",
        "Map",
        "ppX",
        "ppY",
        "pDir",
        "pFrame",
        "g1pX",
        "g1pY",
        "g1Dir",
        "g1ModeR",
        "g1Scared",
        "g1Frame",
        "g2pX",
        "g2pY",
        "g2Dir",
        "g2ModeR",
        "g2Scared",
        "g2Frame",
    }
    missing_columns = sorted(required_columns - set(merged.columns))
    if missing_columns:
        raise ValueError(f"{data_path} 缺少渲染所需列：{missing_columns}")
    if trial:
        merged = merged[merged["DayTrial"].astype(str) == trial]
        if merged.empty:
            raise ValueError(f"No rows matched --trial {trial!r}.")
    if start > 0:
        merged = merged.iloc[start:]
    if max_frames:
        merged = merged.iloc[:max_frames]
    return merged.reset_index(drop=True)


def resolve_render_table_path(render_table_dir: Path, subject: str) -> Path:
    """根据 subject 参数定位 render table pickle。

    输入语义：subject 可以是完整 ``{subject/session}``，也可以是旧视频脚本常用的
    ``041122-403`` 短前缀。
    输出语义：返回实际存在的 render table 路径。
    关键约束：正式文件名不再带 ``_merged_frame_data`` 后缀；旧后缀只作为读取兼容兜底。
    """

    exact_path = render_table_dir / f"{subject}.pkl"
    if exact_path.exists():
        return exact_path

    candidates = sorted(render_table_dir.glob(f"{subject}-*.pkl"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(f"{render_table_dir} 中存在多个匹配 {subject!r} 的 render table：{candidates}")

    legacy_path = render_table_dir / f"{subject}_merged_frame_data.pkl"
    if legacy_path.exists():
        return legacy_path

    raise FileNotFoundError(f"在 {render_table_dir} 中找不到 {subject!r} 对应的 render table pickle。")


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    """从命令行参数读取渲染输入表。

    输入语义：args 必须包含 render_table_dir、subject、trial、start 和 max_frames。
    输出语义：返回已经完成可选筛选的逐帧渲染行。
    关键约束：路径由外部显式传入，核心模块不内置项目数据目录。
    """

    return load_render_rows(
        render_table_dir=args.render_table_dir,
        subject=args.subject,
        trial=args.trial,
        start=args.start,
        max_frames=args.max_frames,
    )


def iter_output_paths(rows: pd.DataFrame, output_dir: Path, subject: str | None = None) -> Iterable[tuple[int, pd.Series, Path]]:
    """为每一帧生成输出路径。

    输出目录结构固定为 ``{output_dir}/{subject}/{DayTrial}/00001.jpg``，并且每个
    trial 单独从 00001 开始编号。
    """

    counters: dict[str, int] = {}
    subject_dir = output_dir / sanitize_name(subject) if subject else output_dir
    for i, row in rows.iterrows():
        trial = sanitize_name(str(row["DayTrial"]))
        counters[trial] = counters.get(trial, 0) + 1
        path = subject_dir / trial / f"{counters[trial]:05d}.jpg"
        yield i, row, path


def parse_args() -> argparse.Namespace:
    """解析渲染脚本命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "subject",
        help="被试名前缀，例如 041122-403。",
    )
    parser.add_argument("render_table_dir", type=Path, help="render table 输入目录。")
    parser.add_argument("output_dir", type=Path, help="JPG 图片帧输出根目录。")
    parser.add_argument("--trial", default=None, help="可选：只渲染指定 DayTrial/game。")
    parser.add_argument("--start", type=int, default=0, help="渲染前跳过 merged 表中的前 N 行。")
    parser.add_argument("--max-frames", type=int, default=None, help="最多渲染多少帧，预览时建议设置。")
    parser.add_argument("--aa", type=int, default=3, help="抗锯齿超采样倍数。")
    parser.add_argument(
        "--bar-type",
        choices=sorted(BAR_TYPE_CHOICES),
        default="grammar",
        help="底部信息 bar 类型：grammar=读取 gram/gram_num，strategy=读取 fitted_label，none=不画 bar。默认 grammar。",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口：读取 render table 并逐帧保存 JPG。"""

    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_data(args)
    if rows.empty:
        raise ValueError("No frames to render.")

    renderer = PacmanRenderer(str(rows.iloc[0]["Map"]), aa=max(1, args.aa), bar_type=args.bar_type)
    total = len(rows)
    for i, row, path in iter_output_paths(rows, output_dir, subject=args.subject):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = renderer.render(row)
        frame.save(path, quality=95, subsampling=0)
        if (i + 1) % 50 == 0 or i == 0 or i + 1 == total:
            print(f"rendered {i + 1}/{total}: {path}")


if __name__ == "__main__":
    main()
