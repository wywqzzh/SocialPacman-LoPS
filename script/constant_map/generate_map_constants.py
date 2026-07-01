#!/usr/bin/env python
"""直接从 Pacman 字符地图生成地图常量文件。

这个脚本把旧流程中的 MATLAB 地图展开、`map_info.csv` 中间文件、
邻接表转换和最短路径计算合并成一个清晰的 Python 流程。输入是脚本内
固定的 28*36 地图，输出是一个 `map_constants.pkl`，其中包含：

1. `adjacent_map`：每个可走点的四方向邻接表。
2. `dij_distance_map`：任意两个可走点之间的最短距离和路径表。

坐标使用从 1 开始的 `(x, y)`。地图内空格 `' '`、豆子 `'.'` 和能量豆
`'o'` 都是可走路径；鬼屋中 ghost 真实出现过的几个非空格位置会额外加入可走点。左右 tunnel 的正式
分析坐标只保留 `(0, 18)` 和 `(29, 18)`，不再保留过渡帧坐标
`(-1, 18)` 和 `(30, 18)`。
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


MAP_WIDTH = 28
MAP_HEIGHT = 36
DIRECTIONS = ("left", "right", "up", "down")
WALKABLE_CHARS = {" ", ".", "o"}
DIRECTION_DELTAS = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
}

MAP_ROWS = (
    "____________________________",
    "____________________________",
    "____________________________",
    "||||||||||||||||||||||||||||",
    "|            ||            |",
    "| |||| ||||| || ||||| |||| |",
    "| |||| ||||| || ||||| |||| |",
    "| |||| ||||| || ||||| |||| |",
    "|                          |",
    "| |||| || |||||||| || |||| |",
    "| |||| || |||||||| || |||| |",
    "|      ||    ||    ||      |",
    "|||||| ||||| || ||||| ||||||",
    "_____| ||||| || ||||| |_____",
    "_____| ||          || |_____",
    "_____| || |||--||| || |_____",
    "|||||| || |______| || ||||||",
    "          |______|          ",
    "|||||| || |______| || ||||||",
    "_____| || |||||||| || |_____",
    "_____| ||          || |_____",
    "_____| || |||||||| || |_____",
    "|||||| || |||||||| || ||||||",
    "|            ||            |",
    "| |||| ||||| || ||||| |||| |",
    "| |||| ||||| || ||||| |||| |",
    "| ||||                |||| |",
    "| |||| || |||||||| || |||| |",
    "| |||| || |||||||| || |||| |",
    "|      ||    ||    ||      |",
    "| |||||||||| || |||||||||| |",
    "| |||||||||| || |||||||||| |",
    "|                          |",
    "||||||||||||||||||||||||||||",
    "____________________________",
    "____________________________",
)

GHOST_HOUSE_WALKABLE_POSITIONS = {
    (14, 16),
    (15, 16),
    (14, 17),
    (15, 17),
    (14, 18),
    (15, 18),
    (14, 19),
    (15, 19),
}

TUNNEL_LINKS = {
    (0, 18): {"left": (29, 18), "right": (1, 18)},
    (29, 18): {"left": (28, 18), "right": (0, 18)},
}

_GRAPH: nx.Graph | None = None
_POSITIONS: tuple[tuple[int, int], ...] | None = None


def validate_map() -> None:
    """校验内置地图尺寸。

    输入语义：读取模块内的 `MAP_ROWS`。
    输出语义：尺寸正确时不返回值；尺寸错误时抛出 ValueError。
    关键约束：地图必须严格是 28*36，否则坐标和 tunnel 规则都会错位。
    """

    if len(MAP_ROWS) != MAP_HEIGHT:
        raise ValueError(f"地图高度应为 {MAP_HEIGHT}，实际为 {len(MAP_ROWS)}")
    wrong_rows = [index + 1 for index, row in enumerate(MAP_ROWS) if len(row) != MAP_WIDTH]
    if wrong_rows:
        raise ValueError(f"以下地图行宽度不是 {MAP_WIDTH}：{wrong_rows}")


def map_walkable_positions() -> list[tuple[int, int]]:
    """按旧坐标顺序列出地图内所有可走位置。

    输入语义：使用内置 28*36 字符地图。
    输出语义：返回所有可走字符位置和鬼屋额外可走位置，顺序为先 x 后 y。
    关键约束：鬼屋额外位置来自当前数据中 ghost 真实出现过、且经 02 阶段
    坐标修正后的合法位置；`(14, 20)` 和 `(15, 20)` 不再作为正式节点。
    """

    validate_map()
    return [
        (x, y)
        for x in range(1, MAP_WIDTH + 1)
        for y in range(1, MAP_HEIGHT + 1)
        if MAP_ROWS[y - 1][x - 1] in WALKABLE_CHARS or (x, y) in GHOST_HOUSE_WALKABLE_POSITIONS
    ]


def build_adjacent_map() -> pd.DataFrame:
    """生成四方向邻接表。

    输入语义：使用内置地图和固定 tunnel 连接规则。
    输出语义：返回 `pos/left/right/up/down` 五列 DataFrame。
    关键约束：地图内位置按旧顺序排列，2 个 tunnel 边界点追加在末尾。
    """

    positions = map_walkable_positions()
    position_set = set(positions) | {(0, 18), (29, 18)}
    ordered_positions = positions + [(0, 18), (29, 18)]
    rows: list[dict[str, Any]] = []

    for position in ordered_positions:
        neighbors = {direction: np.nan for direction in DIRECTIONS}
        if position in TUNNEL_LINKS:
            neighbors.update(TUNNEL_LINKS[position])
        else:
            x, y = position
            for direction, (dx, dy) in DIRECTION_DELTAS.items():
                candidate = (x + dx, y + dy)
                if candidate in position_set:
                    neighbors[direction] = candidate
        rows.append({"pos": position, **neighbors})

    return pd.DataFrame(rows, columns=["pos", "left", "right", "up", "down"])


def is_missing(value: Any) -> bool:
    """判断一个邻接值是否表示不可走方向。

    输入语义：value 来自邻接表四方向字段。
    输出语义：返回 True 表示该方向没有邻居。
    关键约束：邻接表中缺失方向统一使用 NaN。
    """

    return isinstance(value, float) and pd.isna(value)


def build_graph(adjacent_map: pd.DataFrame) -> nx.Graph:
    """把邻接表转换成无向图。

    输入语义：adjacent_map 包含每个可走点的四方向邻居。
    输出语义：返回 NetworkX 无向图，节点是坐标 tuple。
    关键约束：墙体不会进入图，tunnel 通过额外节点和边表达。
    """

    graph = nx.Graph()
    graph.add_nodes_from(adjacent_map["pos"])
    for _, row in adjacent_map.iterrows():
        for direction in DIRECTIONS:
            if not is_missing(row[direction]):
                graph.add_edge(row["pos"], row[direction])
    return graph


def relative_dir(source: tuple[int, int], target: tuple[int, int]) -> list[str]:
    """计算终点相对起点的大致几何方向。

    输入语义：source/target 是地图坐标。
    输出语义：返回 `left/right/up/down` 的组合；例如右上方是
    `["right", "up"]`，正上方是 `["up"]`。
    关键约束：这个字段不是第一步动作，而是终点相对起点的整体方向。
    """

    direction: list[str] = []
    if target[0] > source[0]:
        direction.append("right")
    elif target[0] < source[0]:
        direction.append("left")
    if target[1] > source[1]:
        direction.append("down")
    elif target[1] < source[1]:
        direction.append("up")
    return direction


def init_worker(graph: nx.Graph, positions: tuple[tuple[int, int], ...]) -> None:
    """初始化并行进程共享的只读地图数据。

    输入语义：graph 是邻接图，positions 是所有可走点的固定顺序。
    输出语义：没有返回值；把数据写入 worker 进程内全局变量。
    关键约束：worker 只读这些对象，保证并行结果稳定。
    """

    global _GRAPH, _POSITIONS
    _GRAPH = graph
    _POSITIONS = positions


def shortest_rows_for_source(source: tuple[int, int]) -> list[dict[str, Any]]:
    """计算一个起点到所有其它点的最短路径记录。

    输入语义：source 是当前起点；图和位置列表来自 `init_worker`。
    输出语义：返回多行 `dij_distance_map` 记录。
    关键约束：target 顺序固定，且跳过 source 自身。
    """

    if _GRAPH is None or _POSITIONS is None:
        raise RuntimeError("并行 worker 尚未初始化")

    rows: list[dict[str, Any]] = []
    for target in _POSITIONS:
        if source == target:
            continue
        rows.append(
            {
                "pos1": source,
                "pos2": target,
                "dis": nx.shortest_path_length(_GRAPH, source, target),
                "path": list(nx.all_shortest_paths(_GRAPH, source, target)),
                "relative_dir": relative_dir(source, target),
            }
        )
    return rows


def choose_workers(requested: int | None, task_count: int) -> int:
    """选择实际并行进程数。

    输入语义：requested 是命令行指定值，task_count 是起点数量。
    输出语义：返回最终 worker 数。
    关键约束：默认最多 128 个进程，避免在大机器上产生过高调度开销。
    """

    if requested is not None:
        if requested < 1:
            raise ValueError("--workers 必须大于等于 1")
        return min(requested, task_count)
    return min(os.cpu_count() or 1, 128, task_count)


def build_distance_map(adjacent_map: pd.DataFrame, workers: int | None) -> pd.DataFrame:
    """生成任意两个可走点之间的最短路径表。

    输入语义：adjacent_map 是四方向邻接表，workers 是并行进程数。
    输出语义：返回 `pos1/pos2/dis/path/relative_dir` 五列 DataFrame。
    关键约束：并行计算按 source 顺序合并结果，因此输出顺序稳定。
    """

    graph = build_graph(adjacent_map)
    positions = tuple(adjacent_map["pos"].tolist())
    worker_count = choose_workers(workers, len(positions))
    print(f"计算最短路径：{len(positions)} 个位置，使用 {worker_count} 个进程")

    rows: list[dict[str, Any]] = []
    with Pool(worker_count, initializer=init_worker, initargs=(graph, positions)) as pool:
        for index, source_rows in enumerate(pool.imap(shortest_rows_for_source, positions, chunksize=1), start=1):
            if index == 1 or index % 20 == 0 or index == len(positions):
                print(f"  source {index}/{len(positions)}")
            rows.extend(source_rows)

    return pd.DataFrame(rows, columns=["pos1", "pos2", "dis", "path", "relative_dir"])


def save_map_constants(adjacent_map: pd.DataFrame, distance_map: pd.DataFrame, output_dir: Path) -> Path:
    """把两张地图常量表保存到一个 pickle 文件。

    输入语义：adjacent_map 和 distance_map 是最终结果表。
    输出语义：在 output_dir 写出 `map_constants.pkl`，并返回文件路径。
    关键约束：pickle 内容是一个 dict，键名固定为 `adjacent_map` 和
    `dij_distance_map`，后续读取时可以直接按键获取对应 DataFrame。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "map_constants.pkl"
    pd.to_pickle(
        {
            "adjacent_map": adjacent_map,
            "dij_distance_map": distance_map,
        },
        output_path,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输入语义：读取当前进程的命令行。
    输出语义：返回包含输出目录和 worker 数的参数对象。
    关键约束：默认输出到当前仓库 ``data/constant_data``，让主流程直接读取。
    """

    project_root = Path(__file__).resolve().parents[2]
    default_output_dir = project_root / "data" / "constant_data"
    parser = argparse.ArgumentParser(description="生成 Pacman 地图常量 pickle 文件。")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="输出目录，脚本会写出 map_constants.pkl。")
    parser.add_argument("--workers", type=int, default=None, help="并行进程数，默认最多 128。")
    return parser.parse_args()


def main() -> int:
    """执行地图常量生成流程。

    输入语义：使用内置地图和命令行参数。
    输出语义：写出一个 `map_constants.pkl` 文件，并返回进程退出码。
    关键约束：不再生成或依赖 `map_info.csv` 中间文件。
    """

    args = parse_args()
    adjacent_map = build_adjacent_map()
    distance_map = build_distance_map(adjacent_map, args.workers)
    output_path = save_map_constants(adjacent_map, distance_map, args.output_dir)
    print(f"已写出地图常量到：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
