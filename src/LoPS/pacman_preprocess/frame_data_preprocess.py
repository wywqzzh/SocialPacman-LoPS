"""frame_data 标准分析字段预处理。

本模块位于 raw frame_data 之后、tile 抽样之前。它只负责把逐帧数据收敛到
后续分析流程需要的标准字段集合：统一 id 命名、生成 game_id、规范两个
Pacman 与 ghost 坐标和 ghost 状态类型，并删除视频或原始采集专用字段。
"""

from __future__ import annotations

import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


BASE_STANDARD_FRAME_COLUMNS: tuple[str, ...] = (
    "frame_id",
    "DayTrial",
    "game_id",
    "Step",
    "p1_pos",
    "p1_mode",
    "p1_alive",
)

OPTIONAL_P2_FRAME_COLUMNS: tuple[str, ...] = (
    "p2_pos",
    "p2_mode",
    "p2_alive",
)

TAIL_STANDARD_FRAME_COLUMNS: tuple[str, ...] = (
    "ghost1Pos",
    "ghost2Pos",
    "ifscared1",
    "ifscared2",
    "beans",
    "energizers",
)


class FrameDataPreprocessError(RuntimeError):
    """frame_data 标准字段预处理无法继续时抛出的明确异常。"""


def parse_literal_if_needed(value: Any) -> Any:
    """解析可能以字符串保存的 Python 字面量。

    输入语义：value 可以是字符串形式的位置、列表，也可以已经是 Python 对象。
    输出语义：字符串会用 ast.literal_eval 转成对象，其它值原样返回。
    关键约束：不使用 eval，避免把数据解析和代码执行混在一起。
    """

    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def parse_position(value: Any) -> tuple[int, int]:
    """把单个坐标字段规范为整数 tuple。

    输入语义：value 必须表示长度为 2 的坐标。
    输出语义：返回 ``(x, y)``，每个分量都是 Python int。
    关键约束：当前主流程只保留 two-ghost 数据，ghost1/ghost2 必须存在真实坐标。
    """

    parsed = parse_literal_if_needed(value)
    if not isinstance(parsed, (tuple, list, np.ndarray)) or len(parsed) != 2:
        raise FrameDataPreprocessError(f"无法解析坐标字段：{value!r}")
    return int(parsed[0]), int(parsed[1])


def parse_position_list(value: Any) -> list[tuple[int, int]]:
    """把 beans/energizers 字段规范为坐标列表。

    输入语义：value 是空列表、坐标列表，或这些对象的字符串表示。
    输出语义：返回 ``list[tuple[int, int]]``。
    关键约束：缺失值按空列表处理；非空元素必须都是长度为 2 的坐标。
    """

    parsed = parse_literal_if_needed(value)
    if parsed is None:
        return []
    if isinstance(parsed, float) and pd.isna(parsed):
        return []
    if isinstance(parsed, np.ndarray):
        parsed = parsed.tolist()
    if not isinstance(parsed, list):
        raise FrameDataPreprocessError(f"坐标列表字段必须是 list：{value!r}")
    return [parse_position(item) for item in parsed]


def build_game_id(day_trial: Any) -> str:
    """从 DayTrial 提取 game_id。

    输入语义：day_trial 通常形如 ``1-2-031222-401-03-Dec-2022``。
    输出语义：去掉第二段 trial round 后返回 game 标识，例如
    ``1-031222-401-03-Dec-2022``。
    关键约束：至少需要两个连字符分段，否则无法区分 round 编号。
    """

    parts = str(day_trial).split("-")
    if len(parts) < 2:
        raise FrameDataPreprocessError(f"DayTrial 无法提取 game_id：{day_trial!r}")
    return "-".join([parts[0]] + parts[2:])


def normalize_frame_id(data: pd.DataFrame) -> pd.Series:
    """读取并规范 frame_id 字段。

    输入语义：data 是 raw frame_data；新数据应包含 frame_id，验证旧数据时可能包含
    ``Unnamed: 0``。
    输出语义：返回 int64 的 frame id 序列。
    关键约束：正式输出只使用 frame_id；对 ``Unnamed: 0`` 的兼容只用于黄金数据适配。
    """

    if "frame_id" in data.columns:
        source = data["frame_id"]
    elif "Unnamed: 0" in data.columns:
        source = data["Unnamed: 0"]
    else:
        raise FrameDataPreprocessError("frame_data 缺少 frame_id 字段。")
    return pd.to_numeric(source, errors="raise").astype("int64")


def preprocess_frame_data(data: pd.DataFrame) -> pd.DataFrame:
    """把单个 raw frame_data DataFrame 转换为标准分析字段。

    输入语义：data 来自 raw_subject_data_to_frame_data 阶段，仍可能携带视频或原始采集列；
    双人数据包含 ``p2_pos``，单人数据不保存该列。
    输出语义：返回只包含标准分析字段的 DataFrame；单人数据不会生成空的 ``p2_pos``。
    关键约束：本阶段不计算 action_dir/available_dir，它们会在 tile/corrected tile
    行序稳定后由 human_tile_data_preprocess 生成。
    """

    has_second_player = "p2_pos" in data.columns
    required_columns = {
        "DayTrial",
        "Step",
        "p1_pos",
        "p1_mode",
        "p1_alive",
        "ghost1Pos",
        "ghost2Pos",
        "ifscared1",
        "ifscared2",
        "beans",
        "energizers",
    }
    missing = sorted(required_columns - set(data.columns))
    if missing:
        raise FrameDataPreprocessError(f"frame_data 缺少标准化所需字段：{missing}")
    if has_second_player:
        p2_required = {"p2_mode", "p2_alive"}
        p2_missing = sorted(p2_required - set(data.columns))
        if p2_missing:
            raise FrameDataPreprocessError(f"双人 frame_data 缺少 p2 状态字段：{p2_missing}")

    result = pd.DataFrame()
    result["frame_id"] = normalize_frame_id(data)
    result["DayTrial"] = data["DayTrial"].astype(str)
    result["game_id"] = result["DayTrial"].map(build_game_id)
    result["Step"] = pd.to_numeric(data["Step"], errors="raise").astype("int64")
    result["p1_pos"] = data["p1_pos"].map(parse_position)
    result["p1_mode"] = pd.to_numeric(data["p1_mode"], errors="raise").astype("int8")
    result["p1_alive"] = data["p1_alive"].astype(bool)
    if has_second_player:
        result["p2_pos"] = data["p2_pos"].map(parse_position)
        result["p2_mode"] = pd.to_numeric(data["p2_mode"], errors="raise").astype("int8")
        result["p2_alive"] = data["p2_alive"].astype(bool)
    result["ghost1Pos"] = data["ghost1Pos"].map(parse_position)
    result["ghost2Pos"] = data["ghost2Pos"].map(parse_position)
    # ghost 状态码在 two-ghost 数据中必须存在；缺失值使用 -1 后压缩为 int8。
    result["ifscared1"] = pd.to_numeric(data["ifscared1"], errors="coerce").fillna(-1).astype("int8")
    result["ifscared2"] = pd.to_numeric(data["ifscared2"], errors="coerce").fillna(-1).astype("int8")
    result["beans"] = data["beans"].map(parse_position_list)
    result["energizers"] = data["energizers"].map(parse_position_list)
    return result.loc[:, _standard_frame_columns(has_second_player)]


def _standard_frame_columns(has_second_player: bool) -> tuple[str, ...]:
    """返回当前玩家结构对应的标准 frame_data 列顺序。

    输入语义：has_second_player 表示输入表是否包含第二个 Pacman。
    输出语义：返回最终输出列顺序。
    关键约束：单人数据不补 ``p2_pos``，双人数据把 ``p2_pos`` 放在 ``p1_pos`` 后。
    """

    if has_second_player:
        return BASE_STANDARD_FRAME_COLUMNS + OPTIONAL_P2_FRAME_COLUMNS + TAIL_STANDARD_FRAME_COLUMNS
    return BASE_STANDARD_FRAME_COLUMNS + TAIL_STANDARD_FRAME_COLUMNS


def preprocess_frame_data_file(input_path: Path | str, output_path: Path | str) -> dict[str, Any]:
    """处理单个 frame_data pickle 文件并保存标准化输出。

    输入语义：input_path 指向 raw frame_data pickle，output_path 是标准化输出路径。
    输出语义：写出 DataFrame，并返回文件名、行数和列名摘要。
    关键约束：输出文件名由调用方决定，通常与输入文件同名。
    """

    source_path = Path(input_path)
    target_path = Path(output_path)
    data = pd.read_pickle(source_path)
    result = preprocess_frame_data(data)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_pickle(target_path)
    return {
        "input_file": source_path.name,
        "output_file": target_path.name,
        "input_path": str(source_path),
        "output_path": str(target_path),
        "rows": int(len(result)),
        "columns": list(result.columns),
    }


def _preprocess_worker(task: tuple[str, str]) -> dict[str, Any]:
    """多进程 worker：按字符串路径处理一个文件。

    输入语义：task 包含输入路径和输出路径字符串，便于跨进程 pickle。
    输出语义：返回 preprocess_frame_data_file 的摘要字典。
    关键约束：worker 内部不读取全局路径配置，确保并行任务可复用。
    """

    return preprocess_frame_data_file(Path(task[0]), Path(task[1]))


def preprocess_frame_data_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    files: Iterable[str] | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """批量预处理一个 frame_data 目录。

    输入语义：input_dir 必须是 ``task/session.pkl`` 嵌套目录；files 可限制
    session 文件名或 ``task/session``。
    输出语义：output_dir 下生成同名 pickle，并保留输入的 task 层级。
    关键约束：文件排序只影响日志顺序，不改变单个文件内部数据。
    """

    source_dir = Path(input_dir)
    target_dir = Path(output_dir)
    if not source_dir.is_dir():
        raise FrameDataPreprocessError(f"frame_data 输入目录不存在：{source_dir}")

    input_entries = _collect_frame_data_inputs(source_dir, files)
    if not input_entries:
        raise FrameDataPreprocessError(f"{source_dir} 下没有 frame_data pkl 文件。")
    if workers < 1:
        raise FrameDataPreprocessError("workers 必须大于等于 1。")

    target_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(path), str(target_dir / task_name / path.name if task_name else target_dir / path.name))
        for task_name, path in input_entries
    ]
    if workers == 1:
        return [_preprocess_worker(task) for task in tasks]

    summaries: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_preprocess_worker, task): Path(task[0]).name for task in tasks}
        for future in as_completed(futures):
            file_name = futures[future]
            try:
                summaries.append(future.result())
            except Exception as exc:
                raise FrameDataPreprocessError(f"{file_name} 预处理失败：{exc}") from exc
    summaries.sort(key=lambda item: str(item["input_file"]))
    return summaries


def _collect_frame_data_inputs(source_dir: Path, files: Iterable[str] | None) -> list[tuple[str | None, Path]]:
    """收集 03 阶段需要处理的 frame_data 文件。

    输入语义：source_dir 必须使用 task/session 两层结构；files 可写文件名、
    session stem 或 ``task/session``。
    输出语义：返回 ``(task_name, pkl_path)``。
    关键约束：不再兼容扁平目录，避免旧结构数据混入当前流程。
    """

    entries: list[tuple[str | None, Path]] = []
    for task_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        for path in sorted(task_dir.glob("*.pkl")):
            entries.append((task_dir.name, path))

    selected = set(files or [])
    if not selected:
        return entries

    matched: set[str] = set()
    filtered: list[tuple[str | None, Path]] = []
    for task_name, path in entries:
        stem = path.stem
        keys = {path.name, stem}
        if task_name:
            keys.add(f"{task_name}/{path.name}")
            keys.add(f"{task_name}/{stem}")
        if keys & selected:
            matched.update(keys & selected)
            filtered.append((task_name, path))

    missing = selected - matched
    if missing:
        raise FrameDataPreprocessError(f"找不到指定 frame_data 文件：{sorted(missing)}")
    return filtered
