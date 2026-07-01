#!/usr/bin/env python3
"""把 raw_subject_data 转成逐帧 clean frame_data。

本模块位于 ``01_mat_to_raw_subject_data`` 之后，负责把接近原始 MATLAB
导出的逐帧表整理成后续分析和视频渲染都能复用的逐帧表。当前项目的新数据
以双人 Pacman 为主，但未来也会纳入单人数据；因此本阶段按输入列动态输出
``p1_`` 和可选 ``p2_`` 字段，不再生成旧单人字段 ``pacmanPos``。

地图信息只来自 raw_subject_data 的 ``Map`` 列；本模块不会读取
``data/constant_data``。坐标修正只作用于分析用的坐标列，原始 ``Map`` 字符串
保持不变，便于追溯原始数据。
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# 当前新数据的 Map 是 28 列 x 36 行，按行展开成长度 1008 的字符串。
# beans/energizers 坐标必须使用这个宽度还原，否则奖励坐标会整体错位。
MAP_WIDTH = 28
MAP_HEIGHT = 36
MAP_LENGTH = MAP_WIDTH * MAP_HEIGHT

# 逐帧坐标修正规则。Pacman 的两个 tunnel 外过渡点不作为正式 tile 保存；
# ghost house 底部两个墙格是 ghost 坐标记录规则造成的边界点，分析时并入上方格。
PACMAN_POSITION_FIXES = {
    (-1, 18): (0, 18),
    (30, 18): (29, 18),
}
GHOST_POSITION_FIXES = {
    (14, 20): (14, 19),
    (15, 20): (15, 19),
}

# 渲染器和人工排查需要的原始像素坐标、方向和动画帧。它们不参与坐标修正，
# 但必须随 frame_data 保留下来，否则后续无法只依赖 frame 表渲染视频。
P1_RENDER_SOURCE_COLUMNS = [
    "p1_ppX",
    "p1_ppY",
    "p1_pDir",
    "p1_pFrame",
]

P2_RENDER_SOURCE_COLUMNS = [
    "p2_ppX",
    "p2_ppY",
    "p2_pDir",
    "p2_pFrame",
]

GHOST_RENDER_SOURCE_COLUMNS = [
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
]

RENDER_SOURCE_COLUMNS = P1_RENDER_SOURCE_COLUMNS + P2_RENDER_SOURCE_COLUMNS + GHOST_RENDER_SOURCE_COLUMNS

P1_TAIL_COLUMNS = [
    "p1_waterTS",
    "p1_waterStatus",
    "p1_waterDelay",
]

P2_TAIL_COLUMNS = [
    "p2_waterTS",
    "p2_waterStatus",
    "p2_waterDelay",
]

TAIL_COLUMNS = [
    *P1_TAIL_COLUMNS,
    *P2_TAIL_COLUMNS,
    "Key",
]


class FrameDataError(RuntimeError):
    """raw_subject_data 转换为 frame_data 失败时抛出的明确异常。"""


def _mode_position_to_alive(
    *,
    mode_values: pd.Series,
    positions: list[tuple[int, int]],
    day_trials: pd.Series,
) -> pd.Series:
    """结合原始 mode 和 Pacman 坐标生成分析用有效存活状态。

    输入语义：mode_values 来自原始 ``data/pacMan/mode``；positions 是已经完成
    tunnel 坐标修正的 Pacman tile 坐标；day_trials 用于在 trial 边界重置状态机。
    输出语义：返回 bool 序列，True 表示 Pacman 已经完成刷新并处于可行动的存活状态。
    关键约束：原始数据中存在 ``mode`` 先于坐标恢复的边界帧；这类帧虽然
    ``mode == 1``，但 Pacman 仍停在死亡位置，分析和视频中仍应视为未复活。
    """

    numeric_mode = pd.to_numeric(mode_values, errors="raise").astype("int8")
    if len(numeric_mode) != len(positions) or len(numeric_mode) != len(day_trials):
        raise FrameDataError("mode、Pacman 坐标和 DayTrial 长度不一致，无法生成 alive 字段。")

    alive_values: list[bool] = []
    current_day_trial: object | None = None
    waiting_for_respawn_position = False
    death_position: tuple[int, int] | None = None

    for day_trial, mode, position in zip(day_trials, numeric_mode, positions):
        # DayTrial 之间不能延续死亡状态；每个 trial 都从干净状态重新判断。
        if day_trial != current_day_trial:
            current_day_trial = day_trial
            waiting_for_respawn_position = False
            death_position = None

        if mode != 1:
            # mode=2 表示死亡停留，mode=0 表示刷新/重置过渡。两者都不是有效存活。
            # 第一次进入非正常状态时记录死亡位置；之后等坐标离开该位置才算真正复活。
            if not waiting_for_respawn_position:
                death_position = position
                waiting_for_respawn_position = True
            alive_values.append(False)
            continue

        if waiting_for_respawn_position:
            # 关键边界：有些帧 mode 已经回到 1，但坐标仍停在死亡位置。
            # 只有坐标离开死亡位置，才认为 Pacman 已完成重置并重新存活。
            if position == death_position:
                alive_values.append(False)
                continue
            waiting_for_respawn_position = False
            death_position = None

        alive_values.append(True)

    return pd.Series(alive_values, index=mode_values.index, dtype=bool)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输入语义：input_dir 是 ``01_raw_subject_data``，可以是当前新数据的
    ``task/session.pkl`` 嵌套目录，也兼容扁平 pkl 目录用于临时排查。
    输出语义：返回输入目录、输出目录、CSV 输出目录、session 白名单和并行数。
    关键约束：CSV 只用于人工检查，默认不写，避免生成过大的中间文件。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="raw_subject_data 输入目录。")
    parser.add_argument("output_dir", type=Path, help="frame_data 输出目录。")
    parser.add_argument("sessions", nargs="*", help="可选：只处理这些 session，支持 session 或 task/session。")
    parser.add_argument("--csv-output-dir", type=Path, default=None, help="CSV 输出目录；仅在 --write-csv 时使用。")
    parser.add_argument("--workers", type=int, default=34, help="并行进程数。")
    parser.add_argument("--write-csv", action="store_true", help="同时写出 CSV。")
    return parser.parse_args()


def main() -> None:
    """命令行入口：读取参数、执行批量转换并打印摘要。"""

    args = parse_args()
    csv_output_dir = args.csv_output_dir if args.csv_output_dir is not None else args.output_dir.parent / "frame_data_csv"
    results = convert_raw_subject_data_to_frame_data_dir(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        csv_output_dir=csv_output_dir,
        subjects=args.sessions or None,
        workers=args.workers,
        write_csv=args.write_csv,
    )
    print("raw_subject_data -> frame_data 转换完成")
    print(f"subject/session 数量：{len(results)}")
    print(f"总行数：{sum(item['rows'] for item in results)}")
    print(f"输出目录：{args.output_dir.resolve()}")


def convert_raw_subject_data_to_frame_data_dir(
    *,
    input_dir: Path,
    output_dir: Path,
    csv_output_dir: Path,
    subjects: Iterable[str] | None = None,
    workers: int | None = None,
    write_csv: bool = False,
) -> list[dict[str, object]]:
    """并行转换 ``input_dir`` 下的 raw_subject_data PKL。

    输入语义：正式新数据使用 ``input_dir/task/session.pkl``；若 input_dir 下仍有
    扁平 pkl，也会被识别为无 task 的输入。
    输出语义：写出同名 pkl，并保留 task 嵌套结构，例如
    ``output_dir/comp/session.pkl``。
    关键约束：session 白名单既可写 ``session``，也可写 ``task/session``。
    """

    if not input_dir.exists():
        raise FrameDataError(f"找不到输入目录：input_dir={input_dir}")

    input_entries = _collect_input_entries(input_dir, subjects)
    if not input_entries:
        raise FrameDataError(f"{input_dir} 下没有可转换的 pkl 文件。")

    output_dir.mkdir(parents=True, exist_ok=True)
    if write_csv:
        csv_output_dir.mkdir(parents=True, exist_ok=True)

    if workers is None:
        workers = min(8, os.cpu_count() or 1, len(input_entries))
    if workers < 1:
        raise FrameDataError("--workers 必须大于等于 1。")

    print(f"开始转换 {len(input_entries)} 个 subject/session；并行进程数：{workers}")
    tasks = [
        (str(path), str(output_dir), str(csv_output_dir), write_csv, task_name)
        for task_name, path in input_entries
    ]
    if workers == 1:
        results = []
        for task in tasks:
            result = _convert_one_worker(task)
            results.append(result)
            print(_format_result(result))
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_convert_one_worker, task): _subject_label(task[4], Path(task[0])) for task in tasks}
            for future in as_completed(futures):
                subject = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise FrameDataError(f"{subject} 转换失败：{exc}") from exc
                results.append(result)
                print(_format_result(result))

    results.sort(key=lambda item: str(item["subject"]))
    return results


def _collect_input_entries(input_dir: Path, subjects: Iterable[str] | None) -> list[tuple[str | None, Path]]:
    """收集待转换 pkl，并按可选白名单过滤。

    输入语义：input_dir 可以同时包含扁平 pkl 和 task 子目录。
    输出语义：返回 ``(task_name, pkl_path)`` 列表；扁平文件的 task_name 为 None。
    关键约束：不递归读取超过 task/session 两层的任意 pkl，避免误扫其它产物。
    """

    entries: list[tuple[str | None, Path]] = []
    for path in sorted(input_dir.glob("*.pkl")):
        entries.append((None, path))
    for task_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        for path in sorted(task_dir.glob("*.pkl")):
            entries.append((task_dir.name, path))

    selected = set(subjects or [])
    if not selected:
        return entries

    matched: set[str] = set()
    filtered: list[tuple[str | None, Path]] = []
    for task_name, path in entries:
        keys = {_session_from_input(path)}
        if task_name:
            keys.add(f"{task_name}/{_session_from_input(path)}")
        if keys & selected:
            matched.update(keys & selected)
            filtered.append((task_name, path))

    missing = selected - matched
    if missing:
        raise FrameDataError(f"找不到指定 subject/session：{sorted(missing)}")
    return filtered


def _convert_one_worker(task: tuple[str, str, str, bool, str | None]) -> dict[str, object]:
    """单个多进程任务：读取一个 raw_subject_data PKL 并写出 frame_data。

    输入语义：task 使用字符串路径和简单值，便于 ``ProcessPoolExecutor`` pickle。
    输出语义：返回本文件转换摘要。
    关键约束：输出目录必须保留 task 层级；CSV 输出也遵守同样结构。
    """

    input_path = Path(task[0])
    output_dir = Path(task[1])
    csv_output_dir = Path(task[2])
    write_csv = task[3]
    task_name = task[4]

    subject = _subject_label(task_name, input_path)
    raw_subject_data = pd.read_pickle(input_path)
    frame_data = convert_raw_subject_data_to_frame_data(raw_subject_data)

    output_parent = output_dir / task_name if task_name else output_dir
    output_parent.mkdir(parents=True, exist_ok=True)
    output_path = output_parent / f"{_session_from_input(input_path)}.pkl"
    frame_data.to_pickle(output_path)

    csv_path = None
    if write_csv:
        csv_parent = csv_output_dir / task_name if task_name else csv_output_dir
        csv_parent.mkdir(parents=True, exist_ok=True)
        csv_path = csv_parent / f"{_session_from_input(input_path)}.csv"
        frame_data.to_csv(csv_path)

    return {
        "subject": subject,
        "rows": len(frame_data),
        "columns": list(frame_data.columns),
        "output": str(output_path.resolve()),
        "csv": str(csv_path.resolve()) if csv_path else None,
    }


def convert_raw_subject_data_to_frame_data(df: pd.DataFrame) -> pd.DataFrame:
    """把一个 session 的 raw_subject_data 转成 clean frame_data。

    输入语义：df 是 01 阶段输出的逐帧表，至少包含 p1 和 ghost 原始字段；
    双人数据额外包含 ``p2_`` 字段，单人数据不会保存这些列。
    输出语义：返回逐帧 clean frame_data，坐标字段已经完成分析用修正，Step 转为
    0-based，frame_id 为排序后的稳定行号；单人数据不生成任何 ``p2_`` 输出列。
    关键约束：Map 原样保留；beans/energizers 只从 Map 字符串解析，不读取外部地图。
    """

    has_second_player = _has_second_player_columns(df)
    required = [
        "Step",
        "DayTrial",
        "Map",
        "p1_pacMan_1",
        "p1_pacMan_2",
        "p1_mode",
        "ghost1_1",
        "ghost1_2",
        "ghost1_3",
        "ghost2_1",
        "ghost2_2",
        "ghost2_3",
        "p1_JoyStick",
        "p1_pDir",
    ]
    if has_second_player:
        # 双人数据中 p2 的坐标、方向和按键都是正式分析字段，缺任意一个都应
        # 立即报错，避免后续把半残缺数据误当成单人任务。
        required.extend(["p2_pacMan_1", "p2_pacMan_2", "p2_mode", "p2_JoyStick", "p2_pDir"])
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise FrameDataError(f"输入数据缺少必要字段：{missing}")

    keys = ["DayTrial", "Step"]
    grouped_first = df.groupby(keys, sort=False, as_index=False).first()
    grouped_first = _filter_two_ghost_trials(grouped_first)
    grouped_first = _sort_grouped_frame_by_daytrial_step(grouped_first)
    _validate_map_strings(grouped_first["Map"])

    # 先计算 Pacman 坐标，再结合原始 mode 生成有效 alive 状态。这里不能只看
    # mode == 1，因为死亡后会出现 mode 先恢复、坐标下一帧才刷新的边界帧。
    p1_positions = _position_list(grouped_first, "p1_pacMan_1", "p1_pacMan_2", PACMAN_POSITION_FIXES)
    if has_second_player:
        p2_positions = _position_list(grouped_first, "p2_pacMan_1", "p2_pacMan_2", PACMAN_POSITION_FIXES)
    else:
        p2_positions = None

    # 先构造单人和双人都必需的公共字段；p2 相关列按原双人输出顺序插入，
    # 但只有在输入表确实存在第二个玩家时才加入。
    output_columns: dict[str, object] = {
        "DayTrial": grouped_first["DayTrial"],
        "Step": grouped_first["Step"],
        "p1_pos": p1_positions,
        "p1_mode": pd.to_numeric(grouped_first["p1_mode"], errors="raise").astype("int8"),
        "p1_alive": _mode_position_to_alive(
            mode_values=grouped_first["p1_mode"],
            positions=p1_positions,
            day_trials=grouped_first["DayTrial"],
        ),
    }
    if has_second_player:
        output_columns.update(
            {
                "p2_pos": p2_positions,
                "p2_mode": pd.to_numeric(grouped_first["p2_mode"], errors="raise").astype("int8"),
                "p2_alive": _mode_position_to_alive(
                    mode_values=grouped_first["p2_mode"],
                    positions=p2_positions,
                    day_trials=grouped_first["DayTrial"],
                ),
            }
        )
    output_columns.update(
        {
            "ghost1Pos": _position_list(grouped_first, "ghost1_1", "ghost1_2", GHOST_POSITION_FIXES),
            "ghost2Pos": _position_list(grouped_first, "ghost2_1", "ghost2_2", GHOST_POSITION_FIXES),
            "ghost3Pos": _ghost_position(grouped_first, "ghost3", GHOST_POSITION_FIXES),
            "ghost4Pos": _ghost_position(grouped_first, "ghost4", GHOST_POSITION_FIXES),
            "ifscared1": grouped_first["ghost1_3"],
            "ifscared2": grouped_first["ghost2_3"],
            "ifscared3": _ghost_mode(grouped_first, "ghost3"),
            "ifscared4": _ghost_mode(grouped_first, "ghost4"),
            "p1_dir": grouped_first["p1_pDir"],
        }
    )
    if has_second_player:
        output_columns["p2_dir"] = grouped_first["p2_pDir"]
    output_columns["p1_JoyStick"] = grouped_first["p1_JoyStick"]
    if has_second_player:
        output_columns["p2_JoyStick"] = grouped_first["p2_JoyStick"]
    output_columns["Map"] = grouped_first["Map"]
    data_frame = pd.DataFrame(output_columns)

    for column in RENDER_SOURCE_COLUMNS:
        if column in grouped_first.columns and column not in data_frame.columns:
            data_frame[column] = grouped_first[column]

    reward_frame = _extract_reward_lists(grouped_first)
    data_frame = pd.merge(data_frame, reward_frame, on=keys, how="left")

    tail_columns = ["DayTrial", "Step"] + [column for column in TAIL_COLUMNS if column in df.columns]
    if len(tail_columns) > 2:
        tail_frame = df.loc[:, tail_columns].groupby(keys, sort=False, as_index=False).first()
        data_frame = pd.merge(data_frame, tail_frame, on=keys, how="left")

    # 01 阶段保留 MATLAB/Data 表的 1-based Step；frame_data 延续旧流程使用 0-based Step。
    data_frame["Step"] = data_frame["Step"] - 1
    data_frame.insert(0, "frame_id", np.arange(len(data_frame), dtype=np.int64))
    return data_frame


def _has_second_player_columns(frame: pd.DataFrame) -> bool:
    """判断输入 raw_subject_data 是否包含第二个玩家。

    输入语义：frame 是 01 阶段产出的 raw 表；单人数据不应保存任何 ``p2_`` 列。
    输出语义：若存在 p2 坐标列则返回 True。
    关键约束：只出现部分 p2 列时视为结构错误，避免误判为可处理的单人数据。
    """

    p2_columns = [column for column in frame.columns if column.startswith("p2_")]
    if not p2_columns:
        return False
    required_position_columns = {"p2_pacMan_1", "p2_pacMan_2"}
    if not required_position_columns.issubset(frame.columns):
        raise FrameDataError(f"输入数据存在部分 p2 字段但缺少坐标列：{sorted(p2_columns)}")
    return True


def _filter_two_ghost_trials(frame: pd.DataFrame) -> pd.DataFrame:
    """只保留 two-ghost trial，丢弃 four-ghost trial。

    输入语义：frame 是按 DayTrial-Step 去重后的逐帧原始表。
    输出语义：返回第三、第四个 ghost 全程缺失的 trial。
    关键约束：过滤粒度是完整 DayTrial，不能只删除个别帧，否则轨迹会被截断。
    """

    if "DayTrial" not in frame.columns:
        raise FrameDataError("输入数据缺少 DayTrial，无法按 trial 过滤 two-ghost 数据。")

    ghost_presence_columns = [
        column
        for column in ("ghost3_1", "ghost3_2", "ghost4_1", "ghost4_2")
        if column in frame.columns
    ]
    if not ghost_presence_columns:
        return frame.reset_index(drop=True)

    has_extra_ghost = pd.Series(False, index=frame.index)
    for column in ghost_presence_columns:
        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        has_extra_ghost = has_extra_ghost | (numeric_values.notna() & ~np.isinf(numeric_values))

    four_ghost_day_trials = set(frame.loc[has_extra_ghost, "DayTrial"])
    if not four_ghost_day_trials:
        return frame.reset_index(drop=True)

    filtered = frame.loc[~frame["DayTrial"].isin(four_ghost_day_trials)].copy()
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def _sort_grouped_frame_by_daytrial_step(frame: pd.DataFrame) -> pd.DataFrame:
    """按 DayTrial 的数字前缀和 Step 数值稳定排序。

    输入语义：frame 是已按 DayTrial-Step 去重后的逐帧表。
    输出语义：返回排序并 reset index 的 DataFrame。
    关键约束：DayTrial 前两个分段必须按整数比较，避免字符串排序错位。
    """

    sort_keys = frame["DayTrial"].map(_day_trial_numeric_sort_key)
    sortable = frame.assign(
        _trial_major=[key[0] for key in sort_keys],
        _trial_minor=[key[1] for key in sort_keys],
        _trial_rest=[key[2] for key in sort_keys],
        _step_numeric=pd.to_numeric(frame["Step"], errors="raise"),
    )
    sortable = sortable.sort_values(
        by=["_trial_major", "_trial_minor", "_trial_rest", "_step_numeric"],
        kind="mergesort",
    )
    return sortable.drop(
        columns=["_trial_major", "_trial_minor", "_trial_rest", "_step_numeric"]
    ).reset_index(drop=True)


def _day_trial_numeric_sort_key(value: object) -> tuple[int, int, str]:
    """提取 DayTrial 的数字排序键。

    输入语义：value 通常形如 ``"1-2-session"``。
    输出语义：返回 ``(第一数字段, 第二数字段, 剩余文本)``。
    关键约束：前两个字段不是整数时直接报错，避免静默产生错误排序。
    """

    parts = str(value).split("-")
    if len(parts) < 2:
        raise FrameDataError(f"DayTrial 缺少前两个数字段，无法排序：{value!r}")
    try:
        major = int(parts[0])
        minor = int(parts[1])
    except ValueError as exc:
        raise FrameDataError(f"DayTrial 前两个字段必须是数字，无法排序：{value!r}") from exc
    return major, minor, "-".join(parts[2:])


def _validate_map_strings(values: pd.Series) -> None:
    """检查 Map 字符串长度是否符合 28x36。

    输入语义：values 是逐帧 Map 列。
    输出语义：无返回；发现异常时抛出 FrameDataError。
    关键约束：本阶段不修正 Map，只验证其长度，防止奖励坐标解析错位。
    """

    lengths = values.astype(str).str.len()
    bad_lengths = sorted(set(lengths[lengths != MAP_LENGTH]))
    if bad_lengths:
        raise FrameDataError(f"Map 长度不是 {MAP_LENGTH}，异常长度：{bad_lengths}")


def _position_list(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    fixes: dict[tuple[int, int], tuple[int, int]],
) -> list[tuple[int, int]]:
    """生成已修正的必选坐标列。

    输入语义：x_col/y_col 必须存在且每一行都是有限坐标。
    输出语义：返回 ``(x, y)`` 整数 tuple 列表。
    关键约束：p1/p2 和 ghost1/ghost2 都是必选实体，缺失或 inf 坐标视为数据错误。
    """

    positions: list[tuple[int, int]] = []
    for x_value, y_value in zip(frame[x_col], frame[y_col]):
        if _is_missing_position_value(x_value) or _is_missing_position_value(y_value):
            raise FrameDataError(f"{x_col}/{y_col} 包含缺失或无穷坐标。")
        position = (int(x_value), int(y_value))
        positions.append(fixes.get(position, position))
    return positions


def _ghost_position(
    frame: pd.DataFrame,
    ghost_prefix: str,
    fixes: dict[tuple[int, int], tuple[int, int]],
) -> list[object]:
    """生成 ghost3/ghost4 的可选位置列。

    输入语义：ghost3/ghost4 在 two-ghost trial 中通常为 NaN/inf。
    输出语义：不存在时返回空列表 ``[]``，存在时返回修正后的坐标 tuple。
    关键约束：保留空列表表示，便于后续明确区分“不存在的 ghost”和缺失错误。
    """

    x_col = f"{ghost_prefix}_1"
    y_col = f"{ghost_prefix}_2"
    if x_col not in frame.columns or y_col not in frame.columns:
        return [[] for _ in range(len(frame))]

    positions: list[object] = []
    for x_value, y_value in zip(frame[x_col], frame[y_col]):
        if _is_missing_position_value(x_value) or _is_missing_position_value(y_value):
            positions.append([])
        else:
            position = (int(x_value), int(y_value))
            positions.append(fixes.get(position, position))
    return positions


def _is_missing_position_value(value: object) -> bool:
    """判断坐标分量是否为缺失或无穷值。"""

    if pd.isna(value):
        return True
    return bool(np.isinf(float(value)))


def _ghost_mode(frame: pd.DataFrame, ghost_prefix: str) -> pd.Series:
    """生成 ghost3/ghost4 的模式列。

    输入语义：不存在的 ghost 模式通常是 NaN/inf。
    输出语义：不存在时使用 ``-1``。
    关键约束：该列仍沿用旧命名 ``ifscared3/4``，但值来自 01 阶段的 mode 字段。
    """

    mode_col = f"{ghost_prefix}_3"
    if mode_col not in frame.columns:
        return pd.Series(np.full(len(frame), -1), index=frame.index)

    values = pd.to_numeric(frame[mode_col], errors="coerce")
    values = values.mask(values.isna() | np.isinf(values), -1)
    return values


def _extract_reward_lists(frame: pd.DataFrame) -> pd.DataFrame:
    """从 Map 解析每个 DayTrial-Step 的 beans/energizers 列。

    输入语义：frame 必须包含 DayTrial、Step 和 Map。
    输出语义：返回与 frame 行一一对应的 beans/energizers 列表。
    关键约束：最后一帧若 Map 中无豆子/能量豆，则保留空列表，复现旧流程语义。
    """

    reward_enabled = _reward_enabled_mask(frame)
    cache: dict[str, tuple[list[tuple[int, int]], list[tuple[int, int]]]] = {}
    beans_values: list[object] = []
    energizer_values: list[object] = []

    for enabled, map_value in zip(reward_enabled, frame["Map"]):
        if not enabled:
            beans_values.append([])
            energizer_values.append([])
            continue
        map_text = str(map_value)
        if map_text not in cache:
            cache[map_text] = _parse_map_rewards(map_text)
        beans, energizers = cache[map_text]
        beans_values.append(list(beans))
        energizer_values.append(list(energizers))

    return pd.DataFrame(
        {
            "DayTrial": frame["DayTrial"].to_numpy(),
            "Step": frame["Step"].to_numpy(),
            "beans": beans_values,
            "energizers": energizer_values,
        }
    )


def _reward_enabled_mask(frame: pd.DataFrame) -> np.ndarray:
    """判断每一帧是否应当保留 reward 列表。

    输入语义：frame 是排序后的逐帧表。
    输出语义：布尔数组，False 表示该帧 reward 列应为空列表。
    关键约束：每个 trial 最后一帧若已经没有豆子/能量豆，则不再解析 reward。
    """

    enabled = np.ones(len(frame), dtype=bool)
    for _, indices in frame.groupby("DayTrial", sort=False).groups.items():
        group_positions = np.asarray(indices)
        last_position = group_positions[-1]
        last_map = str(frame.loc[last_position, "Map"])
        if "." not in last_map and "o" not in last_map:
            enabled[last_position] = False
    return enabled


def _parse_map_rewards(map_text: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """从 28x36 的 Map 字符串中解析豆子和能量豆坐标。

    输入语义：map_text 是按行展开的一维字符串。
    输出语义：返回 beans 和 energizers 的 1-based tile 坐标列表。
    关键约束：只识别 ``.`` 和 ``o``；空格也可走但不是 reward。
    """

    if len(map_text) != MAP_LENGTH:
        raise FrameDataError(f"Map 长度是 {len(map_text)}，预期 {MAP_LENGTH}。")

    beans: list[tuple[int, int]] = []
    energizers: list[tuple[int, int]] = []
    for index, char in enumerate(map_text):
        if char != "." and char != "o":
            continue
        position = (index % MAP_WIDTH + 1, index // MAP_WIDTH + 1)
        if char == ".":
            beans.append(position)
        else:
            energizers.append(position)
    return beans, energizers


def _session_from_input(path: Path) -> str:
    """从 raw_subject_data 文件名还原 session 名。"""

    return path.stem.removesuffix("_raw_subject_data")


def _subject_label(task_name: str | None, path: Path) -> str:
    """生成日志中使用的 subject/session 标签。"""

    session = _session_from_input(path)
    return f"{task_name}/{session}" if task_name else session


def _format_result(item: dict[str, object]) -> str:
    """把转换结果整理成一行日志，便于观察并行进度。"""

    return f"{item['subject']}: rows={item['rows']}, columns={len(item['columns'])}, output={item['output']}"


if __name__ == "__main__":
    main()
