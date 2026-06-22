"""fMRI hierarchical utility 的 DataFrame 与文件级预计算入口。"""

from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from .model import (
    CompiledMapData,
    MapData,
    UtilityConfig,
    compile_frame_state,
    compile_map_data,
    load_map_data,
    parse_frame_state,
)
from .strategies import estimate_all_q_values


Q_COLUMNS: tuple[str, ...] = (
    "global_Q",
    "local_Q",
    "evade_blinky_Q",
    "evade_clyde_Q",
    "approach_Q",
    "energizer_Q",
    "no_energizer_Q",
)
_CHUNK_COMPILED_MAP: CompiledMapData | None = None
_CHUNK_CONFIG: UtilityConfig | None = None


def estimate_utility_for_dataframe(
    frame_data: pd.DataFrame,
    map_data: MapData,
    config: UtilityConfig | None = None,
) -> pd.DataFrame:
    """为一个 corrected tile DataFrame 追加 hierarchical utility Q 列。

    输入语义：frame_data 是单个被试的 corrected tile 数据；map_data 是地图常量。
    输出语义：返回 reset index 后追加 7 个 Q 列的新 DataFrame。
    关键约束：保留输入原有列与行顺序，只在末尾追加策略 Q 列。
    """

    config = UtilityConfig() if config is None else config
    compiled_map = compile_map_data(map_data)
    return _estimate_utility_with_compiled_map(frame_data, compiled_map, config)


def _estimate_utility_with_compiled_map(
    frame_data: pd.DataFrame,
    compiled_map: CompiledMapData,
    config: UtilityConfig,
) -> pd.DataFrame:
    """使用已编译地图为 DataFrame 追加 utility Q 列。

    输入语义：frame_data 是任意连续行块，compiled_map 已在文件级或进程级准备好。
    输出语义：返回 reset index 后追加 7 个 Q 列的新 DataFrame。
    关键约束：该函数不重新编译地图，便于行块级并行任务复用同一份地图快表。
    """

    result = frame_data.reset_index(drop=True).copy()
    q_values: dict[str, list[Any]] = {column: [] for column in Q_COLUMNS}
    columns = result.columns

    for _, row in result.iterrows():
        frame_state = parse_frame_state(row, columns)
        compiled_frame = compile_frame_state(frame_state, compiled_map)
        # 每帧只调用一次共享估计器，内部一次路径遍历同时结算全部路径型策略。
        frame_q_values = estimate_all_q_values(compiled_map, compiled_frame, config)
        for column in Q_COLUMNS:
            q_values[column].append(frame_q_values[column])

    for column in Q_COLUMNS:
        result[column] = q_values[column]
    return result


def process_utility_file(
    input_path: str | Path,
    output_path: str | Path,
    map_data: MapData,
    config: UtilityConfig | None = None,
) -> dict[str, Any]:
    """处理单个 corrected tile 文件并保存 utility 输出。

    输入语义：input_path 是单个 `.pkl`，output_path 是目标输出文件路径。
    输出语义：保存追加 Q 列后的 DataFrame，并返回文件摘要。
    关键约束：保存使用 `pickle.dump`，贴近旧脚本序列化方式。
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    with input_path.open("rb") as file:
        frame_data = pickle.load(file)
    utility_data = estimate_utility_for_dataframe(frame_data, map_data, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(utility_data, file)
    return {
        "input_file": input_path.name,
        "output_file": output_path.name,
        "row_count": int(utility_data.shape[0]),
        "column_count": int(utility_data.shape[1]),
    }


def process_utility_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    map_data: MapData,
    config: UtilityConfig | None = None,
    workers: int = 1,
    row_chunk_size: int | None = None,
) -> list[dict[str, Any]]:
    """批量处理一个 corrected tile 目录。

    输入语义：input_dir 包含被试 `.pkl` 文件，output_dir 是 utility 输出目录。
    输出语义：返回每个被试的处理摘要。
    关键约束：默认按文件并行；传入 row_chunk_size 时可按行块并行并按原顺序拼回。
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    config = UtilityConfig() if config is None else config
    output_dir.mkdir(parents=True, exist_ok=True)
    if row_chunk_size is not None and row_chunk_size > 0 and workers > 1:
        return _process_directory_by_row_chunks(input_dir, output_dir, map_data, config, workers, row_chunk_size)

    tasks = [
        (input_path, output_dir / input_path.name, map_data, config)
        for input_path in sorted(input_dir.glob("*.pkl"))
    ]

    if workers <= 1:
        return [_process_utility_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_process_utility_task, tasks))


def load_map_data_from_directory(constant_dir: str | Path) -> MapData:
    """从常量目录读取 fMRI utility 需要的地图数据。

    输入语义：constant_dir 必须包含 adjacent 和 distance 两个 fMRI csv 文件。
    输出语义：返回 `MapData`。
    关键约束：该函数只拼接当前功能的文件名，不引入任何旧项目路径。
    """

    constant_dir = Path(constant_dir)
    return load_map_data(
        constant_dir / "adjacent_map_fmri.csv",
        constant_dir / "dij_distance_map_fmri.csv",
    )


def _process_utility_task(task: tuple[Path, Path, MapData, UtilityConfig]) -> dict[str, Any]:
    """执行目录批处理中的单个任务。

    输入语义：task 包含输入路径、输出路径、地图数据和配置。
    输出语义：返回 `process_utility_file` 的摘要。
    关键约束：保持为顶层函数，便于 multiprocessing pickle。
    """

    input_path, output_path, map_data, config = task
    return process_utility_file(input_path, output_path, map_data, config)


def _process_directory_by_row_chunks(
    input_dir: Path,
    output_dir: Path,
    map_data: MapData,
    config: UtilityConfig,
    workers: int,
    row_chunk_size: int,
) -> list[dict[str, Any]]:
    """按行块并行处理整个目录。

    输入语义：input_dir/output_dir 是数据目录，row_chunk_size 控制每个并行任务包含的行数。
    输出语义：每个被试仍保存为一个完整同名 pickle 文件，并返回摘要列表。
    关键约束：行块只改变调度方式；拼接时按 file_index 和 chunk_index 恢复原始行顺序。
    """

    compiled_map = compile_map_data(map_data)
    file_infos: list[tuple[Path, Path, int]] = []
    chunk_tasks: list[tuple[int, int, pd.DataFrame]] = []
    for file_index, input_path in enumerate(sorted(input_dir.glob("*.pkl"))):
        output_path = output_dir / input_path.name
        with input_path.open("rb") as file:
            frame_data = pickle.load(file).reset_index(drop=True)
        file_infos.append((input_path, output_path, int(frame_data.shape[0])))
        for chunk_index, start in enumerate(range(0, frame_data.shape[0], row_chunk_size)):
            stop = min(start + row_chunk_size, frame_data.shape[0])
            # 每个任务携带一个连续行块；最终 concat(ignore_index=True) 恢复文件内行序。
            chunk_tasks.append((file_index, chunk_index, frame_data.iloc[start:stop].copy()))

    chunk_results: dict[int, list[tuple[int, pd.DataFrame]]] = {index: [] for index in range(len(file_infos))}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_utility_chunk_worker,
        initargs=(compiled_map, config),
    ) as executor:
        for file_index, chunk_index, chunk_result in executor.map(_process_utility_chunk_task, chunk_tasks):
            chunk_results[file_index].append((chunk_index, chunk_result))

    summaries: list[dict[str, Any]] = []
    for file_index, (input_path, output_path, row_count) in enumerate(file_infos):
        ordered_chunks = [chunk for _, chunk in sorted(chunk_results[file_index], key=lambda item: item[0])]
        utility_data = pd.concat(ordered_chunks, ignore_index=True) if ordered_chunks else pd.DataFrame()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(utility_data, file)
        summaries.append(
            {
                "input_file": input_path.name,
                "output_file": output_path.name,
                "row_count": row_count,
                "column_count": int(utility_data.shape[1]),
            }
        )
    return summaries


def _init_utility_chunk_worker(compiled_map: CompiledMapData, config: UtilityConfig) -> None:
    """初始化行块并行 worker 的只读共享配置。

    输入语义：compiled_map 和 config 由主进程在创建进程池时传入。
    输出语义：写入进程内全局变量，后续行块任务不再重复携带这些对象。
    关键约束：这些对象在 worker 内只读使用，避免跨任务状态污染。
    """

    global _CHUNK_COMPILED_MAP, _CHUNK_CONFIG
    _CHUNK_COMPILED_MAP = compiled_map
    _CHUNK_CONFIG = config


def _process_utility_chunk_task(task: tuple[int, int, pd.DataFrame]) -> tuple[int, int, pd.DataFrame]:
    """执行行块级并行任务。

    输入语义：task 包含文件序号、行块序号和行块 DataFrame。
    输出语义：返回文件序号、行块序号和追加 Q 列后的行块。
    关键约束：编译地图和配置来自 worker 初始化阶段，避免在每个 task 中重复序列化。
    """

    if _CHUNK_COMPILED_MAP is None or _CHUNK_CONFIG is None:
        raise RuntimeError("行块 worker 尚未初始化地图和配置。")
    file_index, chunk_index, frame_chunk = task
    return file_index, chunk_index, _estimate_utility_with_compiled_map(
        frame_chunk,
        _CHUNK_COMPILED_MAP,
        _CHUNK_CONFIG,
    )
