#!/usr/bin/env python3
"""把原始 fMRI trial .mat 转换成 subject 级原始逐帧数据。

这个脚本复现 MATLAB 版 translateData + BEVdata.readData 的核心逻辑：
读取每个 session 下的 trial .mat，
抽取逐帧游戏状态字段。raw_mat_data 下每个文件夹视为一个 subject/session，
每个 subject/session 的多个 trial 会合并保存成一个 raw_subject_data PKL。
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
    "pacMan_1",
    "pacMan_2",
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
    "JoyStick",
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
    "waterTS",
    "waterStatus",
    "waterDelay",
]


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
    parser.add_argument("--workers", type=int, default=34, help="并行进程数。默认使用 CPU 数、8 和 subject 数中的较小值。")
    return parser.parse_args()


def main() -> None:
    """命令行入口：把一个或多个 session 文件夹转换成逐帧原始 PKL。"""

    args = parse_args()

    results = convert_mat_root_to_raw_subject_data(
        args.raw_root,
        output_dir=args.output_dir,
        selected_subjects=args.sessions or None,
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
    add_key: bool = False,
    workers: int | None = None,
) -> list[dict[str, object]]:
    """读取 ``raw_root`` 下所有 subject/session，并单独保存为 raw_subject_data PKL。

    输入语义：raw_root 下每个 session 目录包含多个 trial `.mat` 文件。
    输出语义：每个 session 目录对应一个 `{session}.pkl`。
    关键约束：输出文件仍保留 session 原名，便于回溯到 raw_mat_data。
    """

    if not raw_root.exists():
        raise RawFmriError(f"找不到原始数据目录：{raw_root}")

    # 默认处理所有带 '-' 的 session 文件夹；如果传入位置参数 sessions，则只处理白名单。
    selected = set(selected_subjects or [])
    session_dirs = sorted(path for path in raw_root.iterdir() if path.is_dir() and "-" in path.name)
    if selected:
        session_dirs = [path for path in session_dirs if path.name in selected]
        missing = selected - {path.name for path in session_dirs}
        if missing:
            raise RawFmriError(f"找不到指定 subject/session：{sorted(missing)}")

    if not session_dirs:
        raise RawFmriError(f"{raw_root} 下没有可处理的 subject/session 文件夹。")

    output_dir.mkdir(parents=True, exist_ok=True)
    # 原始 .mat 读取较重，默认最多开 8 个进程；需要更高并行度时显式传 --workers。
    if workers is None:
        workers = min(8, os.cpu_count() or 1, len(session_dirs))
    if workers < 1:
        raise RawFmriError("--workers 必须大于等于 1。")

    print(f"开始转换 {len(session_dirs)} 个 subject/session；并行进程数：{workers}")
    tasks = [(str(session_dir), str(output_dir), add_key) for session_dir in session_dirs]
    if workers == 1:
        results = []
        for task in tasks:
            result = _convert_subject_worker(task)
            results.append(result)
            print(_format_result(result))
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_convert_subject_worker, task): Path(task[0]).name for task in tasks}
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


def _format_result(item: dict[str, object]) -> str:
    """把一个 session 的抽取结果格式化成进度日志。"""

    return f"{item['subject']}: rows={item['rows']}, trials={item['trials']}, skipped={item['skipped']}, output={item['output']}"


def _convert_subject_worker(task: tuple[str, str, bool]) -> dict[str, object]:
    """多进程 worker：转换一个 subject/session 并写入独立 PKL。"""

    session_dir = Path(task[0])
    output_dir = Path(task[1])
    add_key = task[2]
    session_data, session_skipped = convert_mat_session_to_raw_subject_data(session_dir)
    # 固定列顺序可以让后续 frame table 构建和旧数据对比更稳定。
    session_data = session_data[OUTPUT_COLUMNS]
    if add_key:
        session_data["Key"] = session_data["DayTrial"].astype(str) + "-" + session_data["Step"].astype(str)

    output_path = output_dir / f"{session_dir.name}.pkl"
    session_data.to_pickle(output_path)
    return {
        "subject": session_dir.name,
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
    return pd.concat(parts, ignore_index=True), skipped


def convert_mat_trial_to_frame_rows(trial_path: Path, session_name: str | None = None) -> pd.DataFrame | None:
    """转换单个 trial `.mat` 为逐帧 DataFrame。

    输出仍然接近旧 MATLAB ``translateData`` 的字段命名：tile 坐标使用
    ``pacMan_1/ghost1_1`` 这类历史字段；渲染需要的像素坐标、朝向和动画帧
    则保存在 ``ppX/g1pX/g1Dir`` 等列中。
    """

    with h5py.File(trial_path, "r") as mat:
        if "data" not in mat:
            raise RawFmriError(f"{trial_path} 缺少 data 变量。")

        # 原始 `.mat` 中大部分变量是一帧一个值。先确定帧数，后续所有数组都
        # 会整理成长度为 n_frames 的列，避免 MATLAB 行/列方向差异造成错位。
        n_frames = _frame_count(mat)
        scared = _ghost_matrix(mat, "data/ghosts/scared", n_frames)

        ghost_dir_enum = _ghost_dir_enum(mat, n_frames)
        mode = _mode_transfer(mat, n_frames)

        pacman_tile_x = _col(mat, "data/pacMan/tile_x", n_frames)
        pacman_tile_y = _col(mat, "data/pacMan/tile_y", n_frames)
        ghost_tile_x = _ghost_matrix(mat, "data/ghosts/tile_x", n_frames)
        ghost_tile_y = _ghost_matrix(mat, "data/ghosts/tile_y", n_frames)

        ghost_mode_raw = _ghost_matrix(mat, "data/ghosts/mode", n_frames)
        ghost_frames = _ghost_matrix(mat, "data/ghosts/frames", n_frames)
        ghost_pixel_x = _ghost_matrix(mat, "data/ghosts/pixel_x", n_frames)
        ghost_pixel_y = _ghost_matrix(mat, "data/ghosts/pixel_y", n_frames)

        water_ts, water_status, water_delay = _set_do_time_delay(mat, n_frames)
        map_values = _map_strings(mat, n_frames)

        # 这里集中构造输出表，确保每一列都是逐帧长度。第三、第四个 ghost
        # 即使在 two-ghost trial 中不存在，也会由 _ghost_matrix 补成 inf，
        # 这样下游可以统一判断 trial 类型。
        frame = pd.DataFrame(
            {
                "Step": np.arange(1, n_frames + 1, dtype=np.int64),
                "DayTrial": _canonical_day_trial(trial_path, session_name),
                "Map": map_values,
                "pacMan_1": pacman_tile_x,
                "pacMan_2": pacman_tile_y,
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
                "JoyStick": _joystick(mat, n_frames),
                "ppX": _col(mat, "data/pacMan/pixel_x", n_frames),
                "ppY": _col(mat, "data/pacMan/pixel_y", n_frames),
                "pDir": _dir_enum_to_text(_col(mat, "data/pacMan/dirEnum", n_frames)),
                "pFrame": _col(mat, "data/pacMan/frames", n_frames),
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
                "waterTS": water_ts,
                "waterStatus": water_status,
                "waterDelay": water_delay,
            }
        )
    return frame


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


def _joystick(mat: h5py.File, n_frames: int) -> np.ndarray:
    """复现 MATLAB ``TransDir``：把四个按键通道合成为一个方向字符串。"""

    up = _col(mat, "data/direction/up", n_frames)
    down = _col(mat, "data/direction/down", n_frames)
    left = _col(mat, "data/direction/left", n_frames)
    right = _col(mat, "data/direction/right", n_frames)
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

    pacman_tile_x = _col(mat, "data/pacMan/tile_x", n_frames)
    pacman_tile_y = _col(mat, "data/pacMan/tile_y", n_frames)
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


def _set_do_time_delay(mat: h5py.File, n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """复现奖励给水事件的时间窗和延迟字段。

    输出三列：
    - ``waterTS``：给水阀打开到关闭期间为 1；
    - ``waterStatus``：打开帧为 1，关闭帧为 2；
    - ``waterDelay``：打开/关闭事件对应的保存、摇杆检测和奖励成本之和。
    """

    reward_path = "data/reward" if _has_dataset(mat, "data/reward") else "data/rewd/reward"
    reward = _col(mat, reward_path, n_frames)
    reward_diff = reward[1:] - reward[:-1]

    water_ts = np.zeros(n_frames)
    water_status = np.zeros(n_frames)
    water_delay = np.zeros(n_frames)
    if not np.any(reward_diff):
        return water_ts, water_status, water_delay

    open_ts = np.where(reward_diff != 0)[0] + 1
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
