#!/usr/bin/env python3
"""把原始 fMRI trial .mat 转换成 subject 级原始逐帧数据。

这个脚本复现 MATLAB 版 translateData + BEVdata.readData 的核心逻辑：
读取每个 session 下的 trial .mat，
抽取逐帧游戏状态字段。raw_mat_data 可以是旧版扁平 session 目录，也可以是
新版 task/session 嵌套目录；每个 subject/session 的多个 trial 会合并保存成
一个 raw_subject_data PKL。
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "Step",
    "DayTrial",
    "Map",
    "p1_pacMan_1",
    "p1_pacMan_2",
    "p1_mode",
    "p2_pacMan_1",
    "p2_pacMan_2",
    "p2_mode",
    "ghost1_1",
    "ghost1_2",
    "ghost1_3",
    "ghost2_1",
    "ghost2_2",
    "ghost2_3",
    "ghost3_1",
    "ghost3_2",
    "ghost3_3",
    "ghost4_1",
    "ghost4_2",
    "ghost4_3",
    "p1_JoyStick",
    "p2_JoyStick",
    "p1_ppX",
    "p1_ppY",
    "p1_pDir",
    "p1_pFrame",
    "p2_ppX",
    "p2_ppY",
    "p2_pDir",
    "p2_pFrame",
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
    "g3pX",
    "g3pY",
    "g3Dir",
    "g3ModeR",
    "g3Scared",
    "g3Frame",
    "g4pX",
    "g4pY",
    "g4Dir",
    "g4ModeR",
    "g4Scared",
    "g4Frame",
    "p1_waterTS",
    "p1_waterStatus",
    "p1_waterDelay",
    "p2_waterTS",
    "p2_waterStatus",
    "p2_waterDelay",
]

P2_OUTPUT_COLUMNS = {
    "p2_pacMan_1",
    "p2_pacMan_2",
    "p2_mode",
    "p2_JoyStick",
    "p2_ppX",
    "p2_ppY",
    "p2_pDir",
    "p2_pFrame",
    "p2_waterTS",
    "p2_waterStatus",
    "p2_waterDelay",
}


# 旧 fmriFrameData 里有几组 2022-11-13/14 session 的 DayTrial subject code
# 与原始 fmri 文件夹名不一致。这里仅在 DayTrial 字段层面复现旧数据命名，
# 输出文件名仍保留原始 session 文件夹名，便于回溯原始数据来源。
DAY_TRIAL_SESSION_BASE_ALIASES = {
    "131122-401-13-Nov-2022": "131222-401-13-Nov-2022",
    "131122-402-13-Nov-2022": "131222-402-13-Nov-2022",
    "141122-401-14-Nov-2022": "141222-401-14-Nov-2022",
    "141122-402-14-Nov-2022": "141222-402-14-Nov-2022",
}
class RawFmriError(RuntimeError):
    """原始 fMRI 行为数据无法转换时抛出的明确异常。"""


def parse_args() -> argparse.Namespace:
    """解析原始 trial `.mat` 抽取脚本的命令行参数。

    这个步骤通常不是每天都运行：只有当 ``raw_subject_data`` 缺失、
    或者原始 trial `.mat` 更新后，才需要重新抽取。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path, help="raw_mat_data 根目录。")
    parser.add_argument("output_dir", type=Path, help="raw_subject_data 输出目录。")
    parser.add_argument("sessions", nargs="*", help="可选：只处理这些 subject/session 文件夹名；不传则处理全部。")
    parser.add_argument("--tasks", nargs="*", default=None, help="可选：只处理这些任务目录，例如 comp coop。")
    parser.add_argument("--workers", type=int, default=34, help="并行进程数，默认 34；实际不会超过待处理 session 数。")
    return parser.parse_args()


def main() -> None:
    """命令行入口：把一个或多个 session 文件夹转换成逐帧原始 PKL。"""

    args = parse_args()

    results = convert_mat_root_to_raw_subject_data(
        args.raw_root,
        output_dir=args.output_dir,
        selected_subjects=args.sessions or None,
        selected_tasks=args.tasks,
        workers=args.workers,
    )

    print("raw_subject_data 生成完成")
    print(f"subject/session 数量：{len(results)}")
    print(f"trial 数量：{sum(item['trials'] for item in results)}")
    print(f"逐帧行数：{sum(item['rows'] for item in results)}")
    print(f"输出目录：{args.output_dir.resolve()}")


def convert_mat_root_to_raw_subject_data(
    raw_root: Path,
    *,
    output_dir: Path,
    selected_subjects: Iterable[str] | None = None,
    selected_tasks: Iterable[str] | None = None,
    add_key: bool = False,
    workers: int | None = None,
) -> list[dict[str, object]]:
    """读取 ``raw_root`` 下所有 subject/session，并单独保存为 raw_subject_data PKL。

    输入语义：raw_root 可直接包含 session 目录，也可包含 task/session 两层目录。
    输出语义：旧扁平输入写为 `{session}.pkl`；新版嵌套输入写为 `{task}/{session}.pkl`。
    关键约束：输出文件仍保留 session 原名，任务类型只由目录表达，不写入数据列。
    """

    if not raw_root.exists():
        raise RawFmriError(f"找不到原始数据目录：{raw_root}")

    # 同时兼容旧版扁平目录和新版 task/session 嵌套目录。selected_subjects
    # 可以写 session 名，也可以写 task/session，便于只跑某个任务下的同名 session。
    session_entries = _discover_session_entries(
        raw_root,
        selected_subjects=selected_subjects,
        selected_tasks=selected_tasks,
    )

    if not session_entries:
        raise RawFmriError(f"{raw_root} 下没有可处理的 subject/session 文件夹。")

    output_dir.mkdir(parents=True, exist_ok=True)
    # 原始 .mat 读取较重，默认最多开 8 个进程；需要更高并行度时显式传 --workers。
    if workers is None:
        workers = min(8, os.cpu_count() or 1, len(session_entries))
    if workers < 1:
        raise RawFmriError("--workers 必须大于等于 1。")

    print(f"开始转换 {len(session_entries)} 个 subject/session；并行进程数：{workers}")
    tasks = [(str(session_dir), str(output_dir), add_key, task_name) for task_name, session_dir in session_entries]
    if workers == 1:
        results = []
        for task in tasks:
            result = _convert_subject_worker(task)
            results.append(result)
            print(_format_result(result))
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_convert_subject_worker, task): f"{task[3]}/{Path(task[0]).name}" if task[3] else Path(task[0]).name
                for task in tasks
            }
            for future in as_completed(futures):
                subject = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RawFmriError(f"{subject} 转换失败：{exc}") from exc
                results.append(result)
                print(_format_result(result))

    results.sort(key=lambda item: str(item["subject"]))
    if not any(item["rows"] for item in results):
        raise RawFmriError("所有 subject/session 都为空；请检查 raw 数据。")
    return results


def _discover_session_entries(
    raw_root: Path,
    *,
    selected_subjects: Iterable[str] | None,
    selected_tasks: Iterable[str] | None,
) -> list[tuple[str | None, Path]]:
    """发现需要处理的 session 目录，并记录其所属任务目录。

    输入语义：raw_root 是 00_raw_mat_data；selected_subjects 可包含 session 名或
    `task/session`，selected_tasks 可限制 comp/coop 等顶层任务目录。
    输出语义：返回 `(task_name, session_dir)` 列表；旧扁平目录的 task_name 为 None。
    关键约束：任务信息只用于输出目录，不写入 raw DataFrame 字段。
    """

    selected = set(selected_subjects or [])
    task_filter = set(selected_tasks or [])
    entries: list[tuple[str | None, Path]] = []

    # 旧版数据直接把 session 放在 raw_root 下；只要目录名带 '-' 且含 mat 文件就视作 session。
    for path in sorted(raw_root.iterdir()):
        if path.is_dir() and "-" in path.name and any(path.glob("*.mat")) and not task_filter:
            entries.append((None, path))

    # 新版数据使用 task/session 两层目录，例如 comp/{session} 和 coop/{session}。
    for task_dir in sorted(path for path in raw_root.iterdir() if path.is_dir() and "-" not in path.name):
        if task_filter and task_dir.name not in task_filter:
            continue
        for session_dir in sorted(path for path in task_dir.iterdir() if path.is_dir() and "-" in path.name):
            if any(session_dir.glob("*.mat")):
                entries.append((task_dir.name, session_dir))

    if not selected:
        return entries

    matched: set[str] = set()
    filtered: list[tuple[str | None, Path]] = []
    for task_name, session_dir in entries:
        keys = {session_dir.name}
        if task_name is not None:
            keys.add(f"{task_name}/{session_dir.name}")
        if keys & selected:
            matched.update(keys & selected)
            filtered.append((task_name, session_dir))

    missing = selected - matched
    if missing:
        raise RawFmriError(f"找不到指定 subject/session：{sorted(missing)}")
    return filtered


def _format_result(item: dict[str, object]) -> str:
    """把一个 session 的抽取结果格式化成进度日志。"""

    return f"{item['subject']}: rows={item['rows']}, trials={item['trials']}, skipped={item['skipped']}, output={item['output']}"


def _convert_subject_worker(task: tuple[str, str, bool, str | None]) -> dict[str, object]:
    """多进程 worker：转换一个 subject/session 并写入独立 PKL。"""

    session_dir = Path(task[0])
    output_dir = Path(task[1])
    add_key = task[2]
    task_name = task[3]
    session_data, session_skipped = convert_mat_session_to_raw_subject_data(session_dir)
    # 固定已有列的顺序，但不为单人数据补出不存在的 p2 列，避免保存全 NaN 字段。
    session_data = _ordered_existing_output_columns(session_data)
    if add_key:
        session_data["Key"] = session_data["DayTrial"].astype(str) + "-" + session_data["Step"].astype(str)

    output_parent = output_dir / task_name if task_name else output_dir
    output_parent.mkdir(parents=True, exist_ok=True)
    output_path = output_parent / f"{session_dir.name}.pkl"
    session_data.to_pickle(output_path)
    subject_label = f"{task_name}/{session_dir.name}" if task_name else session_dir.name
    return {
        "subject": subject_label,
        "rows": len(session_data),
        "trials": session_data["DayTrial"].nunique() if not session_data.empty else 0,
        "skipped": session_skipped,
        "output": str(output_path.resolve()),
    }


def convert_mat_session_to_raw_subject_data(session_dir: Path) -> tuple[pd.DataFrame, int]:
    """转换一个 session 文件夹下的全部 trial `.mat`。

    输入语义：session_dir 是 raw_mat_data 下一个 subject/session 目录。
    输出语义：返回合并后的 raw_subject_data 表，以及跳过的 trial 数。
    关键约束：trial 按文件名前两段数字排序，保证合并行顺序稳定。
    """

    trial_paths = sorted(session_dir.glob("*.mat"), key=_trial_sort_key)
    parts: list[pd.DataFrame] = []
    skipped = 0
    for trial_path in trial_paths:
        trial_data = convert_mat_trial_to_frame_rows(trial_path, session_dir.name)
        if trial_data is None:
            skipped += 1
            continue
        parts.append(trial_data)

    if not parts:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), skipped
    _validate_session_player_schema(parts, session_dir)
    return pd.concat(parts, ignore_index=True), skipped


def convert_mat_trial_to_frame_rows(trial_path: Path, session_name: str | None = None) -> pd.DataFrame | None:
    """转换单个 trial `.mat` 为逐帧 DataFrame。

    输出仍然接近旧 MATLAB ``translateData`` 的字段命名：ghost 字段继续使用
    ``ghost1_1/g1pX`` 等历史字段；Pacman 玩家字段使用 ``p1_``/``p2_``
    前缀。若原始 trial 只有一个玩家，则只生成 ``p1_`` 字段，不额外保存
    全 NaN 的 ``p2_`` 列。
    """

    with h5py.File(trial_path, "r") as mat:
        if "data" not in mat:
            raise RawFmriError(f"{trial_path} 缺少 data 变量。")

        # 原始 `.mat` 中大部分变量是一帧一个值。先确定帧数，后续所有数组都
        # 会整理成长度为 n_frames 的列，避免 MATLAB 行/列方向差异造成错位。
        n_frames = _frame_count(mat)
        player_count = _player_count(mat, n_frames)
        scared = _ghost_matrix(mat, "data/ghosts/scared", n_frames)

        ghost_dir_enum = _ghost_dir_enum(mat, n_frames)
        mode = _mode_transfer(mat, n_frames)

        p1_pacman_tile_x = _player_col(mat, "data/pacMan/tile_x", n_frames, 0)
        p1_pacman_tile_y = _player_col(mat, "data/pacMan/tile_y", n_frames, 0)
        p1_pacman_mode = _player_col(mat, "data/pacMan/mode", n_frames, 0)
        ghost_tile_x = _ghost_matrix(mat, "data/ghosts/tile_x", n_frames)
        ghost_tile_y = _ghost_matrix(mat, "data/ghosts/tile_y", n_frames)

        ghost_mode_raw = _ghost_matrix(mat, "data/ghosts/mode", n_frames)
        ghost_frames = _ghost_matrix(mat, "data/ghosts/frames", n_frames)
        ghost_pixel_x = _ghost_matrix(mat, "data/ghosts/pixel_x", n_frames)
        ghost_pixel_y = _ghost_matrix(mat, "data/ghosts/pixel_y", n_frames)

        p1_water_ts, p1_water_status, p1_water_delay = _set_do_time_delay(mat, n_frames, 0)
        map_values = _map_strings(mat, n_frames)

        # 这里集中构造输出表，确保每一列都是逐帧长度。第三、第四个 ghost
        # 即使在 two-ghost trial 中不存在，也会由 _ghost_matrix 补成 inf，
        # 这样下游可以统一判断 trial 类型。玩家 2 相关字段只在原始数据确实
        # 有第二列时加入，满足单人数据不保存 p2 空列的约束。
        frame_columns: dict[str, object] = {
            "Step": np.arange(1, n_frames + 1, dtype=np.int64),
            "DayTrial": _canonical_day_trial(trial_path, session_name),
            "Map": map_values,
            "p1_pacMan_1": p1_pacman_tile_x,
            "p1_pacMan_2": p1_pacman_tile_y,
            "p1_mode": p1_pacman_mode,
            "ghost1_1": ghost_tile_x[:, 0],
            "ghost1_2": ghost_tile_y[:, 0],
            "ghost1_3": mode[:, 0],
            "ghost2_1": ghost_tile_x[:, 1],
            "ghost2_2": ghost_tile_y[:, 1],
            "ghost2_3": mode[:, 1],
            "ghost3_1": ghost_tile_x[:, 2],
            "ghost3_2": ghost_tile_y[:, 2],
            "ghost3_3": mode[:, 2],
            "ghost4_1": ghost_tile_x[:, 3],
            "ghost4_2": ghost_tile_y[:, 3],
            "ghost4_3": mode[:, 3],
            "p1_JoyStick": _joystick(mat, n_frames, 0),
            "p1_ppX": _player_col(mat, "data/pacMan/pixel_x", n_frames, 0),
            "p1_ppY": _player_col(mat, "data/pacMan/pixel_y", n_frames, 0),
            "p1_pDir": _dir_enum_to_text(_player_col(mat, "data/pacMan/dirEnum", n_frames, 0)),
            "p1_pFrame": _player_col(mat, "data/pacMan/frames", n_frames, 0),
            "g1pX": ghost_pixel_x[:, 0],
            "g1pY": ghost_pixel_y[:, 0],
            "g1Dir": _dir_enum_to_text(ghost_dir_enum[:, 0]),
            "g1ModeR": ghost_mode_raw[:, 0],
            "g1Scared": scared[:, 0],
            "g1Frame": ghost_frames[:, 0],
            "g2pX": ghost_pixel_x[:, 1],
            "g2pY": ghost_pixel_y[:, 1],
            "g2Dir": _dir_enum_to_text(ghost_dir_enum[:, 1]),
            "g2ModeR": ghost_mode_raw[:, 1],
            "g2Scared": scared[:, 1],
            "g2Frame": ghost_frames[:, 1],
            "g3pX": ghost_pixel_x[:, 2],
            "g3pY": ghost_pixel_y[:, 2],
            "g3Dir": _dir_enum_to_text(ghost_dir_enum[:, 2]),
            "g3ModeR": ghost_mode_raw[:, 2],
            "g3Scared": scared[:, 2],
            "g3Frame": ghost_frames[:, 2],
            "g4pX": ghost_pixel_x[:, 3],
            "g4pY": ghost_pixel_y[:, 3],
            "g4Dir": _dir_enum_to_text(ghost_dir_enum[:, 3]),
            "g4ModeR": ghost_mode_raw[:, 3],
            "g4Scared": scared[:, 3],
            "g4Frame": ghost_frames[:, 3],
            "p1_waterTS": p1_water_ts,
            "p1_waterStatus": p1_water_status,
            "p1_waterDelay": p1_water_delay,
        }
        if player_count >= 2:
            p2_water_ts, p2_water_status, p2_water_delay = _set_do_time_delay(mat, n_frames, 1)
            frame_columns.update(
                {
                    "p2_pacMan_1": _player_col(mat, "data/pacMan/tile_x", n_frames, 1),
                    "p2_pacMan_2": _player_col(mat, "data/pacMan/tile_y", n_frames, 1),
                    "p2_mode": _player_col(mat, "data/pacMan/mode", n_frames, 1),
                    "p2_JoyStick": _joystick(mat, n_frames, 1),
                    "p2_ppX": _player_col(mat, "data/pacMan/pixel_x", n_frames, 1),
                    "p2_ppY": _player_col(mat, "data/pacMan/pixel_y", n_frames, 1),
                    "p2_pDir": _dir_enum_to_text(_player_col(mat, "data/pacMan/dirEnum", n_frames, 1)),
                    "p2_pFrame": _player_col(mat, "data/pacMan/frames", n_frames, 1),
                    "p2_waterTS": p2_water_ts,
                    "p2_waterStatus": p2_water_status,
                    "p2_waterDelay": p2_water_delay,
                }
            )
        frame = pd.DataFrame(frame_columns)
    return frame


def _ordered_existing_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """按标准顺序排列当前 DataFrame 已有的输出列。

    输入语义：frame 是单人或双人 raw_subject_data；单人数据不含 ``p2_`` 列。
    输出语义：返回只包含已有列、且顺序符合 OUTPUT_COLUMNS 的 DataFrame。
    关键约束：该函数不会为缺失字段补列，避免单人数据出现全 NaN 的 p2 字段。
    """

    ordered_columns = [column for column in OUTPUT_COLUMNS if column in frame.columns]
    extra_columns = [column for column in frame.columns if column not in OUTPUT_COLUMNS]
    return frame.loc[:, ordered_columns + extra_columns]


def _validate_session_player_schema(parts: list[pd.DataFrame], session_dir: Path) -> None:
    """检查同一个 session 内的 trial 是否拥有一致的玩家字段结构。

    输入语义：parts 是同一 session 下各 trial 转换后的逐帧表。
    输出语义：无返回；发现单人/双人字段混杂时抛出 RawFmriError。
    关键约束：同一 session 不应混合单人和双人任务，否则后续 trial 合并会产生
    半空列，难以判断是数据缺失还是任务设计差异。
    """

    schema_flags = {tuple(column for column in part.columns if column in P2_OUTPUT_COLUMNS) for part in parts}
    if len(schema_flags) > 1:
        raise RawFmriError(f"{session_dir} 内同时存在单人和双人 trial，请先拆分后再转换。")


def _player_count(mat: h5py.File, n_frames: int) -> int:
    """根据 Pacman tile_x 字段判断 trial 中真实存在的玩家数量。

    输入语义：mat 是一个原始 trial 文件，n_frames 是该 trial 的帧数。
    输出语义：返回玩家列数，目前只允许 1 或 2 个玩家。
    关键约束：玩家数量以坐标主字段为准；其它玩家字段必须至少能提供对应列。
    """

    count = _player_matrix(mat, "data/pacMan/tile_x", n_frames).shape[1]
    if count not in (1, 2):
        raise RawFmriError(f"Pacman 玩家列数为 {count}，当前流程只支持单人或双人数据。")
    return count


def _frame_count(mat: h5py.File) -> int:
    """从方向键数组推断 trial 总帧数。"""

    return int(mat["data/direction/up"].shape[0])


def _col(mat: h5py.File, path: str, n_frames: int) -> np.ndarray:
    """读取一维逐帧变量，并兼容 MATLAB 保存的行向量/列向量。"""

    arr = np.asarray(mat[path])
    if arr.ndim == 2:
        if arr.shape[0] == n_frames:
            arr = arr[:, 0]
        elif arr.shape[1] == n_frames:
            arr = arr[0, :]
    return np.asarray(arr).reshape(-1)


def _player_col(
    mat: h5py.File,
    path: str,
    n_frames: int,
    player_index: int,
    *,
    missing_value: float = np.nan,
) -> np.ndarray:
    """读取玩家相关逐帧字段中的指定玩家列。

    输入语义：path 指向 Pacman、按键或 reward 等玩家字段；player_index 使用
    0/1 对应 p1/p2。
    输出语义：返回长度为 n_frames 的数组；单人数据缺少 p2 时返回 missing_value。
    关键约束：这里只用于玩家字段，ghost 矩阵仍由 _ghost_matrix 按 ghost 维度读取。
    """

    matrix = _player_matrix(mat, path, n_frames)
    if matrix.shape[1] <= player_index:
        return np.full(n_frames, missing_value)
    return matrix[:, player_index]


def _player_column_exists(mat: h5py.File, path: str, n_frames: int, player_index: int) -> bool:
    """判断玩家字段是否包含指定玩家列。"""

    return _player_matrix(mat, path, n_frames).shape[1] > player_index


def _player_matrix(mat: h5py.File, path: str, n_frames: int) -> np.ndarray:
    """把玩家字段统一整理成 ``n_frames x player_count`` 矩阵。

    输入语义：新双人数据通常是 ``n_frames x 2``，旧单人数据可能是一维或
    ``n_frames x 1``。
    输出语义：返回二维矩阵，列表示玩家。
    关键约束：该函数不补列；是否补 p2 由 _player_col 根据调用场景决定。
    """

    arr = np.asarray(mat[path])
    if arr.ndim == 1:
        if arr.shape[0] != n_frames:
            raise RawFmriError(f"{path} 的长度 {arr.shape[0]} 无法和帧数 {n_frames} 对齐。")
        return arr.reshape(n_frames, 1)
    if arr.ndim != 2:
        raise RawFmriError(f"{path} 不是一维或二维玩家字段。")
    if arr.shape[0] == n_frames:
        return arr
    if arr.shape[1] == n_frames:
        return arr.T
    raise RawFmriError(f"{path} 的形状 {arr.shape} 无法和帧数 {n_frames} 对齐。")


def _matrix(mat: h5py.File, path: str, n_frames: int) -> np.ndarray:
    """读取二维逐帧矩阵，并统一成 ``n_frames x n_columns``。"""

    arr = np.asarray(mat[path])
    if arr.ndim != 2:
        raise RawFmriError(f"{path} 不是二维矩阵。")
    if arr.shape[0] == n_frames:
        return arr
    if arr.shape[1] == n_frames:
        return arr.T
    raise RawFmriError(f"{path} 的形状 {arr.shape} 无法和帧数 {n_frames} 对齐。")


def _ghost_matrix(mat: h5py.File, path: str, n_frames: int, min_ghosts: int = 4) -> np.ndarray:
    """读取 ghost 矩阵，并补齐到 min_ghosts 列。

    两鬼 trial 的第三、第四个 ghost 在原始数据中通常是 inf；如果某些旧数据只保存
    两列，这里也用 inf 补齐，方便下游统一生成 ghost3/4 的 []/-1。
    """

    arr = _matrix(mat, path, n_frames)
    if arr.shape[1] >= min_ghosts:
        return arr

    padding = np.full((n_frames, min_ghosts - arr.shape[1]), np.inf)
    return np.column_stack([arr, padding])


def _has_dataset(mat: h5py.File, path: str) -> bool:
    """安全判断 HDF5/MAT 文件中是否存在某个数据集。"""

    try:
        mat[path]
        return True
    except KeyError:
        return False


def _ghost_dir_enum(mat: h5py.File, n_frames: int) -> np.ndarray:
    """读取 ghost 方向枚举；旧数据缺少 dirEnum 时由 dir_x/dir_y 反推。"""

    if _has_dataset(mat, "data/ghosts/dirEnum"):
        return _ghost_matrix(mat, "data/ghosts/dirEnum", n_frames)

    dir_x = _ghost_matrix(mat, "data/ghosts/dir_x", n_frames)
    dir_y = _ghost_matrix(mat, "data/ghosts/dir_y", n_frames)
    enum = np.full(dir_x.shape, 4.0)
    enum[(dir_x == 0) & (dir_y == -1)] = 0
    enum[(dir_x == 1) & (dir_y == 0)] = 3
    enum[(dir_x == 0) & (dir_y == 1)] = 2
    enum[(dir_x == -1) & (dir_y == 0)] = 1
    return enum


def _dir_enum_to_text(values: np.ndarray) -> np.ndarray:
    """把原始方向枚举转成渲染器直接使用的英文方向字符串。"""

    result = np.full(len(values), "", dtype=object)
    result[values == 0] = "up"
    result[values == 2] = "down"
    result[values == 1] = "left"
    result[values == 3] = "right"
    return result


def _joystick(mat: h5py.File, n_frames: int, player_index: int) -> np.ndarray:
    """复现 MATLAB ``TransDir``：把指定玩家的四个按键通道合成为方向字符串。"""

    if not _player_column_exists(mat, "data/direction/up", n_frames, player_index):
        return np.full(n_frames, "", dtype=object)

    up = _player_col(mat, "data/direction/up", n_frames, player_index)
    down = _player_col(mat, "data/direction/down", n_frames, player_index)
    left = _player_col(mat, "data/direction/left", n_frames, player_index)
    right = _player_col(mat, "data/direction/right", n_frames, player_index)
    direction = np.zeros(n_frames)

    # MATLAB TransDir 的赋值顺序不能改：down, right, up, left。
    direction[down == 1] = 2
    direction[right == 1] = 4
    direction[up == 1] = 1
    direction[left == 1] = 3

    result = np.full(n_frames, "", dtype=object)
    result[direction == 1] = "up"
    result[direction == 2] = "down"
    result[direction == 3] = "left"
    result[direction == 4] = "right"
    return result


def _mode_transfer(mat: h5py.File, n_frames: int) -> np.ndarray:
    """复现 MATLAB ``TransMode``：把原始 ghost mode 转成分析用状态。

    旧逻辑会根据 ghost 是否 scared、是否 dead、energizer 闪烁阶段，以及 Clyde
    与 Pacman 的距离重新编码 mode。这个函数保留这些规则，使 Python 产物能和
    旧 ``fmriFrameData`` 对齐。
    """

    mode = _ghost_matrix(mat, "data/ghosts/mode", n_frames)
    scared = _ghost_matrix(mat, "data/ghosts/scared", n_frames)
    tile_x = _ghost_matrix(mat, "data/ghosts/tile_x", n_frames)
    dir_x = _ghost_matrix(mat, "data/ghosts/dir_x", n_frames)
    dir_y = _ghost_matrix(mat, "data/ghosts/dir_y", n_frames)

    count = _col(mat, "data/energizer/count", n_frames)
    duration = _col(mat, "data/energizer/duration", n_frames)
    flash_interval = _col(mat, "data/energizer/flashInterval", n_frames)
    flashes = _col(mat, "data/energizer/flashes", n_frames)
    with np.errstate(divide="ignore", invalid="ignore"):
        flash_index = np.floor((duration - count) / flash_interval)

    # ghost mode 是公共字段；Clyde 的距离分支沿用第一个玩家作为参考，
    # 避免在第 01 步引入尚未定义的双人联合 mode 语义。
    pacman_tile_x = _player_col(mat, "data/pacMan/tile_x", n_frames, 0)
    pacman_tile_y = _player_col(mat, "data/pacMan/tile_y", n_frames, 0)
    dx = pacman_tile_x - (tile_x[:, 1] + dir_x[:, 1])
    tile_y = _ghost_matrix(mat, "data/ghosts/tile_y", n_frames)
    dy = pacman_tile_y - (tile_y[:, 1] + dir_y[:, 1])
    dist = dx * dx + dy * dy

    transferred = mode.copy()
    for ghost_index in range(transferred.shape[1]):
        raw = mode[:, ghost_index]
        new = raw.copy()
        missing = np.isinf(raw) | np.isinf(scared[:, ghost_index])
        outside = (raw == 0) | (raw == 4) | (raw == 5)
        dead = (raw == 1) | (raw == 2) | (raw == 3)

        # Clyde 是第二个 ghost；旧逻辑对它有距离阈值 64 的特殊处理。
        if ghost_index == 1:
            new[(dist >= 64) & (scared[:, ghost_index] == 0) & outside] = 1
            new[(dist < 64) & (scared[:, ghost_index] == 0) & outside] = 2
        else:
            new[(scared[:, ghost_index] == 0) & outside] = 1

        new[dead] = 3
        new[(flash_index > 2 * flashes - 1) & (scared[:, ghost_index] == 1)] = 4
        new[(flash_index <= 2 * flashes - 1) & (scared[:, ghost_index] == 1)] = 5
        new[missing] = -1
        transferred[:, ghost_index] = new

    return transferred


def _set_do_time_delay(mat: h5py.File, n_frames: int, player_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """复现指定玩家奖励给水事件的时间窗和延迟字段。

    输出三列：
    - ``waterTS``：给水阀打开到关闭期间为 1；
    - ``waterStatus``：打开帧为 1，关闭帧为 2；
    - ``waterDelay``：打开/关闭事件对应的保存、摇杆检测和奖励成本之和。
    """

    reward_path = "data/reward" if _has_dataset(mat, "data/reward") else "data/rewd/reward"
    if not _player_column_exists(mat, reward_path, n_frames, player_index):
        missing = np.full(n_frames, np.nan)
        return missing, missing.copy(), missing.copy()

    reward = _player_col(mat, reward_path, n_frames, player_index)
    reward_diff = reward[1:] - reward[:-1]

    water_ts = np.zeros(n_frames)
    water_status = np.zeros(n_frames)
    water_delay = np.zeros(n_frames)
    reward_change = np.isfinite(reward_diff) & (reward_diff != 0)
    if not np.any(reward_change):
        return water_ts, water_status, water_delay

    open_ts = np.where(reward_change)[0] + 1
    diff_at_open = reward_diff[open_ts - 1]
    close_ts = open_ts + diff_at_open.astype(int) - 1
    close_ts[close_ts > n_frames - 1] = n_frames - 1

    data_saving_cost = _col(mat, "data/time/datasavingCost", n_frames)
    js_check_cost = _col(mat, "data/time/JSCheckCost", n_frames)
    reward_cost = _col(mat, "data/time/rewardCost", n_frames)

    open_delay = data_saving_cost[open_ts] + js_check_cost[open_ts] + reward_cost[open_ts]
    close_delay = data_saving_cost[close_ts] + js_check_cost[close_ts] + reward_cost[close_ts]

    ts_gap = np.concatenate([np.diff(open_ts), np.array([close_ts[-1] + 1])])
    bug_ts = diff_at_open > ts_gap
    if np.any(bug_ts):
        bug_indices = np.where(bug_ts)[0]
        if len(bug_indices) != 1:
            raise RawFmriError("two reward are very close")
        idx = bug_indices[0]
        if idx + 1 >= len(close_ts):
            raise RawFmriError("reward close timestamp cannot be repaired")
        close_ts[idx + 1] = open_ts[idx] + int(diff_at_open[idx]) + int(diff_at_open[idx + 1]) - 1
        close_delay = data_saving_cost[close_ts] + js_check_cost[close_ts] + reward_cost[close_ts]

    for start, end in zip(open_ts, close_ts):
        water_ts[start : end + 1] = 1
    water_status[open_ts] = 1
    water_status[close_ts] = 2
    water_delay[open_ts] = open_delay
    water_delay[close_ts] = close_delay
    return water_ts, water_status, water_delay


def _map_strings(mat: h5py.File, n_frames: int) -> list[str]:
    """把原始 gameMap/currentTiles 的 ASCII 数值矩阵转成 Map 字符串。"""

    maps = _matrix(mat, "data/gameMap/currentTiles", n_frames)
    return ["".join(chr(int(value)) for value in row) for row in maps]


def _trial_sort_key(path: Path) -> tuple[int, int, str]:
    """按 trial 文件名前两段数字排序，保证输出顺序稳定。"""

    parts = path.stem.split("-")
    try:
        return int(parts[0]), int(parts[1]), path.name
    except (IndexError, ValueError):
        return 10**9, 10**9, path.name


def _canonical_day_trial(trial_path: Path, session_name: str | None) -> str:
    """用 subject/session 文件夹名规范 DayTrial 的 subject 部分。

    个别原始 trial 文件名中的 subject ID 和外层 subject 文件夹不一致，例如
    031222-401 文件夹内的 trial 文件名写成了 031222-501。旧分析数据使用的是
    文件夹对应的 subject ID，因此这里保留 trial 的 round/used 编号，同时用
    session 文件夹去掉最后一段 session 序号后的名字作为 subject/date 部分。
    """

    if not session_name:
        return trial_path.stem

    trial_parts = trial_path.stem.split("-")
    session_parts = session_name.split("-")
    if len(trial_parts) < 2 or len(session_parts) < 2:
        return trial_path.stem

    session_base = "-".join(session_parts[:-1]) if session_parts[-1].isdigit() else session_name
    session_base = DAY_TRIAL_SESSION_BASE_ALIASES.get(session_base, session_base)
    return f"{trial_parts[0]}-{trial_parts[1]}-{session_base}"


if __name__ == "__main__":
    main()
