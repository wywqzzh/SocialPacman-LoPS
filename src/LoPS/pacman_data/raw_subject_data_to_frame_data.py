#!/usr/bin/env python3
"""把 raw_subject_data 转成渲染和分析共用的 frame_data。

旧流程是：
1. MATLAB Raw2Mat2CSV 把 Data 表写成 Raw CSV Data/fmri/{subject}.csv；
2. 同时根据 Map 写出 {subject}-R.csv，里面记录每一帧 beans/energizers 的位置；
3. csvFormatTransform/toPkl.py 读取这两个 CSV，并调用 ppRaw.transData() 生成 frameData pkl。

当前脚本直接读取 raw_subject_data 中已经生成的 PKL，不再依赖
MATLAB table、txt 或 Raw CSV。
其中 beans/energizers 会从每一帧的 Map 字符串重新解析出来，从而复现 transData()
实际使用到的 dfR Reward==1 和 Reward==2 合并逻辑。Map 会从输入表保留；
ghost3/ghost4 相关字段在输入存在时使用真实值，不存在时按 two-ghost trial 填
ghostPos=[]、ifscared=-1。
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# Map 在 PKL/CSV 中保存为逐行展开后的 29 x 36 字符串。这里必须用
# 序列化后的行宽 29 反推 reward 坐标；否则 beans/energizers 会整体错位。
MAP_WIDTH = 29
# 渲染器需要的原始像素坐标、帧编号和方向列。frame table 不只服务分析，
# 也作为后续图片渲染的唯一逐帧数据来源，因此这些列不能在转换时丢弃。
RENDER_SOURCE_COLUMNS = [
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
]
class FrameDataError(RuntimeError):
    """raw_subject_data 转换为 frame_data 失败时抛出的明确异常。"""


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    这个脚本既可以独立运行，也会被 ``run_pacman_pipeline.py`` 调用。
    默认只写 PKL，因为 frame table 很大；CSV 仅用于人工排查或和旧流程对照。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="raw_subject_data 输入目录。")
    parser.add_argument("output_dir", type=Path, help="frame_data 输出目录。")
    parser.add_argument("sessions", nargs="*", help="可选：只处理这些 subject/session；不传则处理全部。")
    parser.add_argument("--csv-output-dir", type=Path, default=None, help="CSV 输出目录；仅在 --write-csv 时使用。")
    parser.add_argument("--workers", type=int, default=34, help="并行进程数。默认使用 CPU 数、8 和 subject 数中的较小值。")
    parser.add_argument("--write-csv", action="store_true", help="同时写出 CSV；默认只保存 PKL，避免生成过大的中间文件。")
    return parser.parse_args()


def main() -> None:
    """命令行入口：读取参数、启动转换，并打印可读摘要。"""

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

    输入文件名约定为 ``{subject/session}_raw_subject_data.pkl``，输出文件名为
    ``{subject/session}_frame_data.pkl``。这样下游可以按被试名前缀自动查找
    frame_data，同时仍能保留 session 日期信息。
    """

    if not input_dir.exists():
        raise FrameDataError(f"找不到输入目录：{input_dir}")

    # ``subjects`` 是可选白名单；不传时处理目录下所有已有 raw frame PKL。
    selected = set(subjects or [])
    input_paths = sorted(input_dir.glob("*_raw_subject_data.pkl"))
    if selected:
        input_paths = [path for path in input_paths if _subject_from_input(path) in selected]
        missing = selected - {_subject_from_input(path) for path in input_paths}
        if missing:
            raise FrameDataError(f"找不到指定 subject/session：{sorted(missing)}")
    if not input_paths:
        raise FrameDataError(f"{input_dir} 下没有 *_raw_subject_data.pkl 文件。")

    output_dir.mkdir(parents=True, exist_ok=True)
    if write_csv:
        csv_output_dir.mkdir(parents=True, exist_ok=True)

    # 默认并行数限制在 8，避免一次性读写太多大型 PKL 导致内存和磁盘压力过大。
    if workers is None:
        workers = min(8, os.cpu_count() or 1, len(input_paths))
    if workers < 1:
        raise FrameDataError("--workers 必须大于等于 1。")

    print(f"开始转换 {len(input_paths)} 个 subject/session；并行进程数：{workers}")
    tasks = [(str(path), str(output_dir), str(csv_output_dir), write_csv) for path in input_paths]
    if workers == 1:
        results = []
        for task in tasks:
            result = _convert_one_worker(task)
            results.append(result)
            print(_format_result(result))
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_convert_one_worker, task): _subject_from_input(Path(task[0])) for task in tasks}
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


def _convert_one_worker(task: tuple[str, str, str, bool]) -> dict[str, object]:
    """单个多进程任务：读取一个 raw frame PKL 并写出 frame table。

    ``ProcessPoolExecutor`` 要求任务参数可 pickle，因此这里使用字符串路径组成
    tuple，而不是直接闭包捕获 Path 对象和布尔值。
    """

    input_path = Path(task[0])
    output_dir = Path(task[1])
    csv_output_dir = Path(task[2])
    write_csv = task[3]

    subject = _subject_from_input(input_path)
    raw_subject_data = pd.read_pickle(input_path)
    frame_data = convert_raw_subject_data_to_frame_data(raw_subject_data)

    output_path = output_dir / f"{subject}_frame_data.pkl"
    frame_data.to_pickle(output_path)
    csv_path = None
    if write_csv:
        csv_path = csv_output_dir / f"{subject}.csv"
        # 旧 toPkl.py 使用 dataFrame.to_csv(path)，默认保留 index；
        # 这里也保持相同写法，便于人工排查 CSV 时看到导出行号。
        frame_data.to_csv(csv_path)

    return {
        "subject": subject,
        "rows": len(frame_data),
        "columns": list(frame_data.columns),
        "output": str(output_path.resolve()),
        "csv": str(csv_path.resolve()) if csv_path else None,
    }


def convert_raw_subject_data_to_frame_data(df: pd.DataFrame) -> pd.DataFrame:
    """复现 ppRaw.transData(df, dfR) 中实际用到的字段转换。

    输入语义：df 是一个 session 级 raw_subject_data 表，保留接近原始导出的逐帧字段。
    输出语义：返回整理后的 frame_data，包含分析和视频模块共用的逐帧字段。
    关键约束：beans/energizers 直接从 Map 解析，结果与旧 dfR merge 逻辑等价。
    """

    required = [
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
        "JoyStick",
        "pDir",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise FrameDataError(f"输入数据缺少必要字段：{missing}")

    # 旧 DataFrame 可能一帧有多行重复记录；旧流程实际按 DayTrial-Step 取首行。
    # 这里显式 groupby(first)，保证后续每个输出行都是唯一的一帧。
    keys = ["DayTrial", "Step"]
    grouped_first = df.groupby(keys, sort=False, as_index=False).first()
    # 当前完整流程只保留 two-ghost trial。four-ghost trial 会在后续 fMRI utility
    # 中进入地图常量未覆盖的位置，因此在 frame_data 生成阶段按整局直接过滤。
    grouped_first = _filter_two_ghost_trials(grouped_first)
    # frame data 是后续 tile 抽样和行级回指的基础表，必须先按 trial 数字编号
    # 和帧号稳定排序，再生成 frame_id，避免字符串排序把 10-1 排在 2-1 前面。
    grouped_first = _sort_grouped_frame_by_daytrial_step(grouped_first)

    # 这里构造的是旧 ``ppRaw.transData`` 的核心字段：
    # 位置字段使用 tuple，状态字段保持数值，Map/JoyStick 原样带入。
    data_frame = pd.DataFrame(
        {
            "DayTrial": grouped_first["DayTrial"],
            "Step": grouped_first["Step"],
            "pacmanPos": list(zip(grouped_first["pacMan_1"], grouped_first["pacMan_2"])),
            "ghost1Pos": list(zip(grouped_first["ghost1_1"], grouped_first["ghost1_2"])),
            "ghost2Pos": list(zip(grouped_first["ghost2_1"], grouped_first["ghost2_2"])),
            "ghost3Pos": _ghost_position(grouped_first, "ghost3"),
            "ghost4Pos": _ghost_position(grouped_first, "ghost4"),
            "ifscared1": grouped_first["ghost1_3"],
            "ifscared2": grouped_first["ghost2_3"],
            "ifscared3": _ghost_mode(grouped_first, "ghost3"),
            "ifscared4": _ghost_mode(grouped_first, "ghost4"),
            "pacman_dir": grouped_first["pDir"],
            "JoyStick": grouped_first["JoyStick"],
            "Map": grouped_first["Map"],
        }
    )
    for column in RENDER_SOURCE_COLUMNS:
        if column in grouped_first.columns and column not in data_frame.columns:
            # 渲染器直接使用像素坐标、朝向和动画帧编号；这些列不是旧分析字段，
            # 但必须保留，否则后续无法只依赖 frame table 完成画图。
            data_frame[column] = grouped_first[column]

    reward_frame = _extract_reward_lists(grouped_first)
    data_frame = pd.merge(data_frame, reward_frame, on=keys, how="left")

    # 原 ppRaw.transData 会继续合并尾部字段。这里显式保留当前需要的尾部字段，
    # 避免未来输入表新增 ghost3/ghost4 原始列时被重复带入输出。
    tail_columns = ["DayTrial", "Step"] + [
        col for col in ["waterTS", "waterStatus", "waterDelay", "Key"] if col in df.columns
    ]
    if len(tail_columns) > 2:
        tail_frame = df.loc[:, tail_columns].groupby(keys, sort=False, as_index=False).first()
        data_frame = pd.merge(data_frame, tail_frame, on=keys, how="left")
    # 旧 fmriFrameData 使用 0-based Step；逐帧原始 PKL 保留 MATLAB/Data 表的
    # 1-based Step。frame table 层转换时统一改成 0-based，便于和旧分析结果对齐。
    data_frame["Step"] = data_frame["Step"] - 1
    # frame_id 表示排序后逐帧表的稳定行号；后续 tile/corrected tile 阶段会直接
    # 使用它回到原始 frame 区间补中间格，不再生成 Unnamed: 0 或 frameIndex。
    data_frame.insert(0, "frame_id", np.arange(len(data_frame), dtype=np.int64))
    return data_frame


def _filter_two_ghost_trials(frame: pd.DataFrame) -> pd.DataFrame:
    """只保留 two-ghost trial，丢弃 four-ghost trial。

    输入语义：frame 是按 ``DayTrial`` 和 ``Step`` 去重后的逐帧原始表，可能包含
    ``ghost3_1/ghost3_2/ghost4_1/ghost4_2`` 列。
    输出语义：返回只包含第三、第四个 ghost 全程为空的 trial。
    关键约束：过滤粒度是完整 ``DayTrial``，不能只删除四鬼帧，否则同一局内部
    的轨迹和 reward 状态会被截断，后续 tile 抽样也会失去语义。
    """

    if "DayTrial" not in frame.columns:
        raise FrameDataError("输入数据缺少 DayTrial，无法按 trial 过滤 two-ghost 数据。")

    ghost_presence_columns = [
        column
        for column in ("ghost3_1", "ghost3_2", "ghost4_1", "ghost4_2")
        if column in frame.columns
    ]
    if not ghost_presence_columns:
        # 输入完全没有第三、第四个 ghost 字段时，按 two-ghost 数据处理。
        return frame.reset_index(drop=True)

    has_extra_ghost = pd.Series(False, index=frame.index)
    for column in ghost_presence_columns:
        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        # NaN 或 inf 表示该 ghost 不存在；有限数值表示 four-ghost trial。
        has_extra_ghost = has_extra_ghost | (numeric_values.notna() & ~np.isinf(numeric_values))

    four_ghost_day_trials = set(frame.loc[has_extra_ghost, "DayTrial"])
    if not four_ghost_day_trials:
        return frame.reset_index(drop=True)

    filtered = frame.loc[~frame["DayTrial"].isin(four_ghost_day_trials)].copy()
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def _sort_grouped_frame_by_daytrial_step(frame: pd.DataFrame) -> pd.DataFrame:
    """按 DayTrial 的数字前缀和 Step 数值排序 frame 行。

    输入语义：frame 是已按 DayTrial-Step 去重后的逐帧表，DayTrial 通常形如
    ``"1-2-031222-401-03-Dec-2022"``。
    输出语义：返回重排并 reset index 的 DataFrame，排序键为 DayTrial 前两个数字段、
    DayTrial 剩余文本和 Step 数值。
    关键约束：DayTrial 前两个分段必须按整数比较，不能按字符串比较，否则 ``10-1``
    会错误排在 ``2-1`` 前面。
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

    输入语义：value 是 DayTrial 字段值，至少需要包含两个以连字符分隔的数字段。
    输出语义：返回 ``(第一数字段, 第二数字段, 剩余文本)``，供 frame 行排序使用。
    关键约束：如果前两个字段不是整数，直接抛出 FrameDataError，避免静默退回字符串排序。
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


def _ghost_position(frame: pd.DataFrame, ghost_prefix: str) -> list[object]:
    """生成 ghost3/ghost4 的位置列。

    two-ghost trial 中第三、第四个 ghost 通常以 ``inf`` 填充。旧 pkl 使用空列表
    ``[]`` 表示该 ghost 不存在，因此这里把 ``NaN/inf`` 都转换成空列表。
    """

    x_col = f"{ghost_prefix}_1"
    y_col = f"{ghost_prefix}_2"
    if x_col not in frame.columns or y_col not in frame.columns:
        return [[] for _ in range(len(frame))]

    positions: list[object] = []
    for x_value, y_value in zip(frame[x_col], frame[y_col]):
        if pd.isna(x_value) or pd.isna(y_value) or np.isinf(float(x_value)) or np.isinf(float(y_value)):
            positions.append([])
        else:
            positions.append((x_value, y_value))
    return positions


def _ghost_mode(frame: pd.DataFrame, ghost_prefix: str) -> pd.Series:
    """生成 ghost3/ghost4 的模式列。

    旧数据中不存在的 ghost 使用 ``-1``，而不是缺失值。这个表示会被后续
    two-/four-ghost trial 判断和历史数据对比逻辑依赖。
    """

    mode_col = f"{ghost_prefix}_3"
    if mode_col not in frame.columns:
        return pd.Series(np.full(len(frame), -1), index=frame.index)

    values = pd.to_numeric(frame[mode_col], errors="coerce")
    values = values.mask(values.isna() | np.isinf(values), -1)
    return values


def _extract_reward_lists(frame: pd.DataFrame) -> pd.DataFrame:
    """从 Map 解析每个 DayTrial-Step 的 beans/energizers 列。

    旧 fmriFrameData 在某一帧没有对应 reward 时保存为空列表 []，而不是 NaN；
    这里保持相同表示，便于严格对齐旧 pkl。
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

    旧脚本对 trial 最后一帧有一个特殊行为：如果最后一帧 Map 中已经没有豆子
    和能量豆，则对应 reward 列保存空列表。这里复现这个行为，避免最后一帧
    因 Map 缓存解析而错误带入上一帧的 reward。
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
    """从 29x36 的 Map 字符串中解析豆子和能量豆坐标。

    Map 是按行展开的一维字符串，因此 index 需要通过 ``index % 29`` 和
    ``index // 29`` 还原为 1-based tile 坐标。
    """

    beans: list[tuple[int, int]] = []
    energizers: list[tuple[int, int]] = []
    for index, char in enumerate(map_text):
        if char != "." and char != "o":
            continue
        position = (index % MAP_WIDTH + 1, index // MAP_WIDTH + 1)
        if char == ".":
            beans.append(position)
        elif char == "o":
            energizers.append(position)
    return beans, energizers


def _subject_from_input(path: Path) -> str:
    """从 ``*_frame_data.pkl`` 文件名还原 subject/session 名。"""

    return path.name.removesuffix("_raw_subject_data.pkl")


def _format_result(item: dict[str, object]) -> str:
    """把转换结果整理成一行日志，便于并行执行时观察进度。"""

    return f"{item['subject']}: rows={item['rows']}, columns={len(item['columns'])}, output={item['output']}"


if __name__ == "__main__":
    main()
