#!/usr/bin/env python3
"""用 07 阶段修正策略结果数据绘制简易 Pacman 游戏视频。

本脚本直接读取 ``data/07_revised_strategy_data/{task}/{session}.pkl``，选择其中
一个 ``DayTrial`` 作为一个 game，把每条 tile 行渲染成一帧视频。地图背景来自
``data/constant_data/map_constants.pkl`` 的可走点集合：黑色表示墙，白色表示
可以走的位置；每帧额外绘制当前剩余的豆子、energizer、两个 Pacman、两个 ghost，
并在地图上方左右分开显示两个玩家当前拟合出的单个策略。

这个脚本是检查 07 修正策略结果和行为轨迹的轻量可视化工具，不依赖 render table、
grammar 或逐 frame 渲染链路。
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

DIR_UP = 0
DIR_LEFT = 1
DIR_DOWN = 2
DIR_RIGHT = 3


def direction_enum(value: Any) -> int:
    """把英文方向名称转换为 tile sprite 使用的稳定方向编号。

    输入语义：value 通常是 ``up/left/down/right`` 字符串，也可能是缺失值。
    输出语义：返回对应的整数编号；无法识别时返回 -1。
    关键约束：编号只服务于角色朝向绘制，不参与行为数据中的动作推断。
    """

    if not isinstance(value, str):
        return -1
    return {
        "up": DIR_UP,
        "left": DIR_LEFT,
        "down": DIR_DOWN,
        "right": DIR_RIGHT,
    }.get(value.strip().lower(), -1)


def quadratic_path(
    p0x: float,
    p0y: float,
    p1x: float,
    p1y: float,
    p2x: float,
    p2y: float,
    steps: int = 8,
) -> list[tuple[float, float]]:
    """生成角色局部曲线使用的二次贝塞尔采样点。

    输入语义：三组坐标分别是起点、控制点和终点，steps 控制采样密度。
    输出语义：返回包含首尾点的二维坐标列表。
    关键约束：steps 必须为正数；该函数只构造几何路径，不修改画布或数据状态。
    """

    if steps <= 0:
        raise ValueError(f"steps 必须为正数，实际为 {steps}")
    points: list[tuple[float, float]] = []
    for index in range(steps + 1):
        t = index / steps
        x = (1 - t) ** 2 * p0x + 2 * (1 - t) * t * p1x + t**2 * p2x
        y = (1 - t) ** 2 * p0y + 2 * (1 - t) * t * p1y + t**2 * p2y
        points.append((x, y))
    return points


DEFAULT_TILE_ROOT = PROJECT_ROOT / "data/07_revised_strategy_data"
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
STRATEGY_AGENTS = (
    "global",
    "local",
    "evade_blinky",
    "evade_clyde",
    "approach",
    "energizer",
    "no_energizer",
)
STRATEGY_DISPLAY_NAMES = {
    "global": "global",
    "local": "local",
    "evade_blinky": "evade",
    "evade_clyde": "evade",
    "approach": "approach",
    "energizer": "energizer",
    "no_energizer": "no energizer",
    "stay": "stay",
    "vague": "vague",
}
STRATEGY_NUMBER_TO_NAME = {
    0: "global",
    1: "local",
    2: "evade_blinky",
    3: "evade_clyde",
    6: "approach",
    7: "energizer",
    8: "no_energizer",
    9: "vague",
    10: "stay",
}
STRATEGY_COLORS = {
    # 颜色沿用旧 GenerateSimpleVideo_fMRI_gram.py 中每个单测量策略块的配色；
    # 新增的 no_energizer 使用旧脚本预留的深蓝色。
    "global": (69, 180, 61),
    "local": (215, 25, 28),
    "evade_blinky": (254, 175, 97),
    "evade_clyde": (254, 175, 97),
    "approach": (131, 106, 183),
    "energizer": (128, 179, 255),
    "no_energizer": (5, 21, 161),
    "stay": (45, 27, 17),
    "vague": (138, 134, 138),
}
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


def parse_numeric_vector(value: Any) -> np.ndarray:
    """解析 06 拟合结果中的权重向量。

    输入语义：value 通常是 ``p1_normalized_weight`` 或 ``p2_normalized_weight``，
    可以是 list/tuple/numpy 数组，也可能是字符串形式的列表。
    输出语义：返回一维 float 数组；空值或无法形成有效向量时返回空数组。
    关键约束：该函数只做安全字面量解析，不使用 eval；向量长度不足时由上层
    策略判断函数按可用长度处理。
    """

    if value is None:
        return np.asarray([], dtype=float)
    if isinstance(value, float) and pd.isna(value):
        return np.asarray([], dtype=float)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "nan", "none"}:
            return np.asarray([], dtype=float)
        value = ast.literal_eval(stripped)
    if isinstance(value, np.ndarray):
        vector = value.astype(float).reshape(-1)
    elif isinstance(value, (list, tuple)):
        vector = np.asarray(value, dtype=float).reshape(-1)
    else:
        return np.asarray([], dtype=float)
    return vector[~np.isnan(vector)]


def bool_from_row_value(value: Any) -> bool:
    """把 DataFrame 行中的布尔字段转换为普通 bool。

    输入语义：value 可能是 Python bool、numpy bool、0/1、字符串或 NaN。
    输出语义：返回普通 Python bool。
    关键约束：字符串 ``"False"`` 不能直接用 ``bool("False")``，需要显式解析。
    """

    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, float) and pd.isna(value):
        return False
    return bool(value)


def strategy_name_from_saved_value(value: Any) -> str | None:
    """把正式 07 或兼容输入中的策略字段转换成视频显示使用的策略名。

    输入语义：value 通常来自 ``p1_strategy`` 或 ``p2_strategy``，可以是旧编号、
    numpy 数值、字符串策略名或空值。
    输出语义：返回 ``STRATEGY_DISPLAY_NAMES`` 支持的策略名；无法识别时返回 None，
    让上层回退到按权重估计。
    关键约束：07 已经在 context 层处理了 scared-majority approach 优先级，
    视频端应优先使用该字段，避免渲染时用单行权重重复推断出不同标签。
    """

    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "nan", "none"}:
            return None
        if stripped in STRATEGY_DISPLAY_NAMES:
            return stripped
        try:
            number = int(float(stripped))
        except ValueError:
            return None
        return STRATEGY_NUMBER_TO_NAME.get(number)
    try:
        return STRATEGY_NUMBER_TO_NAME.get(int(value))
    except (TypeError, ValueError):
        return None


def estimate_player_strategy(row: pd.Series, player: str) -> str:
    """根据单行正式 07 结果判断某个玩家当前显示策略。

    输入语义：row 是视频当前帧对应的 07 revised strategy 行，player 是 ``p1`` 或 ``p2``。
    输出语义：返回 ``STRATEGY_AGENTS`` 中的策略名，或 ``stay``/``vague``。
    关键约束：优先使用正式 07 的 ``revised_strategy``，其次使用 06 保留的
    ``strategy``；只有这些标签均不存在时，才回退到历史 weight 字段的单行估计。
    """

    # 正式 07 不覆盖 06 原始策略，而是另存 revised_strategy；视频优先展示修正结果，
    # 同时保留 strategy 和历史 weight 输入的只读兼容。
    for strategy_column in (f"{player}_revised_strategy", f"{player}_strategy"):
        if strategy_column in row.index:
            saved_strategy = strategy_name_from_saved_value(row[strategy_column])
            if saved_strategy is not None:
                return saved_strategy

    stay_column = f"{player}_is_stay"
    vague_column = f"{player}_revised_is_vague" if f"{player}_revised_is_vague" in row.index else f"{player}_is_vague"
    weight_column = (
        f"{player}_revised_normalized_weight"
        if f"{player}_revised_normalized_weight" in row.index
        else f"{player}_normalized_weight"
    )

    if stay_column in row.index and bool_from_row_value(row[stay_column]):
        return "stay"
    if vague_column in row.index and bool_from_row_value(row[vague_column]):
        return "vague"
    if weight_column not in row.index:
        return "vague"

    weights = parse_numeric_vector(row[weight_column])
    if weights.size == 0 or np.sum(np.abs(weights)) == 0:
        return "vague"

    # 按当前可用长度截断，避免未来策略数量变化时因为旧视频脚本直接崩溃。
    agents = STRATEGY_AGENTS[: weights.size]
    max_value = np.max(weights)
    max_indices = np.where(weights == max_value)[0]
    if len(max_indices) > 1:
        # 没有 07 strategy 字段时才走这套兜底规则：并列时优先给出旧脚本中
        # 更稳定可解释的 local/global；若只在两个 evade 之间并列，则仍显示
        # 合并后的 evade。其它复杂并列显示 vague。
        if 1 in max_indices:
            return "local"
        if 0 in max_indices:
            return "global"
        tied_agents = {agents[index] for index in max_indices}
        if tied_agents and tied_agents <= {"evade_blinky", "evade_clyde"}:
            return agents[int(max_indices[0])]
        return "vague"
    return agents[int(max_indices[0])]


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


def load_strategy_font(cell_size: int) -> ImageFont.ImageFont:
    """加载策略条字体。

    输入语义：cell_size 是当前绘制画布上的格子尺寸；超采样绘制时会传入放大后的尺寸。
    输出语义：返回 PIL 字体对象。
    关键约束：策略条最终要经过下采样，不能复用带最大字号上限的坐标标签字体，
    否则文字会变得过小、难以阅读。
    """

    font_size = max(16, int(cell_size * 0.62))
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        candidate = Path(font_path)
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), font_size)
    return ImageFont.load_default()


def load_header_font(cell_size: int) -> ImageFont.ImageFont:
    """加载顶部帧信息和 context 信息字体。

    输入语义：cell_size 是当前绘制画布上的格子尺寸；超采样绘制时会传入放大后的尺寸。
    输出语义：返回比坐标标签更大的字体对象。
    关键约束：顶部文字用于快速检查当前视频帧、原始帧号和 context，字号必须明显
    大于行列号，但不能大到压住策略条或地图。
    """

    font_size = max(18, int(cell_size * 0.72))
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


def parse_context_range(value: Any) -> tuple[int, int] | None:
    """解析单个玩家的 trial context 区间。

    输入语义：value 通常来自 ``p1_trial_context`` 或 ``p2_trial_context``，可以是
    tuple/list，也可能是字符串形式的 ``(start, end)``。
    输出语义：返回 ``(start_row_id, end_row_id)``，其中 end 是右开边界；缺失值返回 None。
    关键约束：context 在 06 阶段使用 row_id 表达，视频显示前必须转成当前 trial 内的
    0-based 视频帧序号，不能把数据行号误读成原始游戏帧号。
    """

    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
        return None
    return int(parsed[0]), int(parsed[1])


def build_context_video_frame_lookup(game_rows: pd.DataFrame) -> dict[int, int]:
    """构造 row_id 到当前视频帧序号的查表。

    输入语义：game_rows 是同一个 DayTrial 的连续 tile 行，必须保留 04/05/06 阶段
    写入的 ``row_id``；当前视频帧序号按绘制顺序从 0 开始。
    输出语义：返回 ``{row_id: video_frame_index}``。
    关键约束：PNG 文件名、视频顶部编号和 context 区间统一使用 0-based 编号，
    避免同一张图片在文件名与画面内出现相差 1 的编号。
    """

    if "row_id" not in game_rows.columns:
        return {}
    return {
        int(row["row_id"]): int(video_frame_index)
        for video_frame_index, (_, row) in enumerate(game_rows.iterrows())
        if not pd.isna(row["row_id"])
    }


def format_player_context_video_frames(
    row: pd.Series,
    player: str,
    context_video_frame_lookup: dict[int, int],
) -> str:
    """把某个玩家当前 context 格式化为当前视频帧闭区间。

    输入语义：row 是当前视频帧对应的 tile 行，player 是 ``p1`` 或 ``p2``，
    context_video_frame_lookup 用于把 context row_id 转成当前视频帧序号。
    输出语义：返回类似 ``P1 ctx [37,42]`` 的短文本，左右端都是当前视频中的
    0-based 帧号，且是闭区间。
    关键约束：06 阶段的 context 本身是 row_id 右开区间 ``[start, end)``；
    视频显示时需要转换为闭区间 ``[start_video_frame, last_video_frame]``，
    即右端显示为 context 内最后一条实际渲染行，而不是右开边界对应的下一行。
    """

    label = player.upper()
    context_column = f"{player}_trial_context"
    if context_column not in row.index:
        return f"{label} ctx n/a"

    context_range = parse_context_range(row[context_column])
    if context_range is None:
        return f"{label} ctx n/a"

    start_row_id, end_row_id = context_range
    included_video_frames = [
        video_frame
        for row_id, video_frame in context_video_frame_lookup.items()
        if start_row_id <= row_id < end_row_id
    ]
    if not included_video_frames:
        return f"{label} ctx rows [{start_row_id},{end_row_id})"
    return f"{label} ctx [{min(included_video_frames)},{max(included_video_frames)}]"


def format_video_frame_counter(tile_index: int, total_tiles: int) -> str:
    """生成统一的 0-based 当前帧/末帧显示文本。

    输入语义：tile_index 是当前 0-based 帧号，total_tiles 是视频总帧数。
    输出语义：返回类似 ``video frame 29/222`` 的文本，分母是最后一帧编号。
    关键约束：当前帧必须落在 ``[0, total_tiles)``；总帧数不是显示编号，需减 1
    后再作为末帧编号，确保第一帧和最后一帧分别显示为 ``0`` 和 ``N-1``。
    """

    if total_tiles <= 0:
        raise TileVideoRenderError("视频总帧数必须大于 0。")
    if tile_index < 0 or tile_index >= total_tiles:
        raise TileVideoRenderError(
            f"当前视频帧超出 0-based 范围：tile_index={tile_index}, total_tiles={total_tiles}"
        )
    return f"video frame {tile_index}/{total_tiles - 1}"


def draw_strategy_bar(
    draw: ImageDraw.ImageDraw,
    row: pd.Series,
    metadata: dict[str, int],
    font: ImageFont.ImageFont,
) -> None:
    """在地图上方绘制两个玩家当前拟合策略。

    输入语义：row 是当前 07 策略结果行，metadata 是当前画布坐标元数据，font 是文本字体。
    输出语义：在当前帧顶部绘制左右分离的 P1/P2 策略色块。
    关键约束：旧 gram 图中一个人两个连续测量会挨着画；这里是两个玩家各一个
    单测量，因此 P1 放左侧、P2 放右侧，中间留出明显空白，避免被误读为序列。
    """

    cell_size = metadata["cell_size"]
    origin_x = metadata["origin_x"]
    map_width = (metadata["x_max"] - metadata["x_min"] + 1) * cell_size
    bar_width = max(int(map_width * 0.34), cell_size * 8)
    bar_width = min(bar_width, int(map_width * 0.42))
    bar_height = max(int(cell_size * 1.25), 32)
    # 顶部现在有两行较大的帧/context 文本，策略条需要下移，避免文字和色块互相遮挡。
    bar_y = max(int(cell_size * 2.55), 72)
    bar_specs = (
        ("p1", "P1", origin_x),
        ("p2", "P2", origin_x + map_width - bar_width),
    )

    for player, player_label, bar_x in bar_specs:
        strategy = estimate_player_strategy(row, player)
        display_name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy)
        fill = STRATEGY_COLORS.get(strategy, STRATEGY_COLORS["vague"])
        rectangle = [bar_x, bar_y, bar_x + bar_width, bar_y + bar_height]

        draw.rectangle(rectangle, fill=fill, outline=(0, 0, 0), width=max(2, int(cell_size * 0.06)))
        label = f"{player_label}  {display_name}"
        text_width, text_height = text_size(draw, label, font)
        draw.text(
            (bar_x + (bar_width - text_width) / 2, bar_y + (bar_height - text_height) / 2),
            label,
            fill=(255, 255, 255),
            font=font,
        )


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
    # 顶部空间同时容纳帧信息和左右分开的 P1/P2 策略条。这里不把策略条叠在
    # 棋盘上，避免遮挡地图第 1 行。
    header_height = max(128, int(cell_size * 4.8))
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
    context_video_frame_lookup: dict[int, int],
    aa: int = DEFAULT_AA,
) -> np.ndarray:
    """把一条 07 策略结果行渲染为视频帧。

    输入语义：row 是某个 ``DayTrial`` 的一条 07 修正结果记录，包含位置和策略字段。
    输出语义：返回 imageio 可写入的视频帧 numpy 数组。
    关键约束：本函数不改变底图；当 aa>1 时先在高分辨率副本上绘制，再缩回
    底图尺寸，以减少 Pacman、ghost 和豆子边缘锯齿。context_video_frame_lookup
    用于把 06 保留下来的 row_id context 显示成当前视频帧闭区间。
    """

    if aa <= 0:
        raise TileVideoRenderError("--aa 必须大于 0。")
    if aa > 1:
        frame = base_image.resize((base_image.width * aa, base_image.height * aa), Image.Resampling.NEAREST)
        active_metadata = scale_render_metadata(metadata, aa)
        active_font = load_header_font(metadata["cell_size"] * aa)
        active_strategy_font = load_strategy_font(metadata["cell_size"] * aa)
        header_y = 10 * aa
        header_line_gap = max(24 * aa, int(active_metadata["cell_size"] * 0.82))
    else:
        frame = base_image.copy()
        active_metadata = metadata
        active_font = load_header_font(metadata["cell_size"])
        active_strategy_font = load_strategy_font(metadata["cell_size"])
        header_y = 10
        header_line_gap = max(24, int(active_metadata["cell_size"] * 0.82))
    draw = ImageDraw.Draw(frame)
    header_line_1 = (
        f"{format_video_frame_counter(tile_index, total_tiles)}  "
        f"raw frame_id {int(row['frame_id'])}  {row['DayTrial']}"
    )
    header_line_2 = "  |  ".join(
        [
            format_player_context_video_frames(row, "p1", context_video_frame_lookup),
            format_player_context_video_frames(row, "p2", context_video_frame_lookup),
        ]
    )
    draw.text((active_metadata["origin_x"], header_y), header_line_1, fill=DISPLAY_LABEL_COLOR, font=active_font)
    draw.text(
        (active_metadata["origin_x"], header_y + header_line_gap),
        header_line_2,
        fill=DISPLAY_LABEL_COLOR,
        font=active_font,
    )
    draw_strategy_bar(draw, row, active_metadata, active_strategy_font)

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
    """收集指定任务下可用于视频绘制的 07 revised strategy 文件。

    输入语义：tile_root 默认是 ``data/07_revised_strategy_data``，task 通常是 comp 或 coop。
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
    """选择要绘制的 07 revised strategy 文件。

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
    """读取一个正式 07 策略文件中的某个 game。

    输入语义：tile_path 指向 ``07_revised_strategy_data`` 输出 pkl；trial 是可选 DayTrial。
    输出语义：返回按原顺序排列的单个 game 行。
    关键约束：如果 trial 为空，默认选择文件中第一个 DayTrial，方便快速检查。
    """

    data = pd.read_pickle(tile_path)
    required_position_columns = ("p1_pos", "p1_alive", "ghost1Pos", "ghost2Pos")
    required_columns = (*required_position_columns, *ITEM_COLUMNS)
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise TileVideoRenderError(f"{tile_path} 缺少 06 视频绘制所需字段：{missing}")
    for player in ("p1", "p2"):
        strategy_sources = {
            f"{player}_revised_strategy",
            f"{player}_strategy",
            f"{player}_revised_normalized_weight",
            f"{player}_normalized_weight",
        }
        if strategy_sources.isdisjoint(data.columns):
            raise TileVideoRenderError(f"{tile_path} 缺少 {player} 可用策略字段。")
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
    输出语义：按 0-based 编号写出 ``000000.png`` 形式的连续图片帧。
    关键约束：保存前先清理同目录下旧 PNG，避免上一版较长视频残留多余帧；
    文件名中的整数必须与图片顶部的 ``video frame`` 整数完全一致。
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
    parser.add_argument("--session", default=None, help="07 revised strategy 文件名或不带 .pkl 的 stem。")
    parser.add_argument("--trial", default=None, help="要绘制的 DayTrial；默认绘制文件中的第一个。")
    parser.add_argument("--max-tiles", type=int, default=None, help="可选：只绘制前 N 条 tile 行。")
    parser.add_argument("--cell-size", type=int, default=DEFAULT_CELL_SIZE)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--aa", type=int, default=DEFAULT_AA, help="动态元素抗锯齿超采样倍数；1 表示关闭。")
    return parser.parse_args()


def main() -> None:
    """命令行入口：读取 07 revised strategy 数据并生成一个简易 game 视频。"""

    args = parse_args()
    tile_path = choose_tile_file(args.tile_root, args.task, args.session)
    game_rows = load_game_rows(tile_path, args.trial, args.max_tiles)
    context_video_frame_lookup = build_context_video_frame_lookup(game_rows)
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
            context_video_frame_lookup=context_video_frame_lookup,
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
    print("策略显示：优先使用 07 revised_strategy，其次使用 strategy，P1/P2 左右分离显示")
    print(f"输出视频：{output_path.resolve()}")
    if args.save_frames:
        print(f"输出图片帧：{frame_output_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except TileVideoRenderError as exc:
        raise SystemExit(str(exc)) from exc
