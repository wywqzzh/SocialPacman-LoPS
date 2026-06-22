#!/usr/bin/env python3
"""Pacman 视频渲染前的数据准备流程。

这个模块替代旧的 ``GenerateVideos_.py`` 中和画图强相关的数据准备逻辑。
核心目标是把 tile-level 的模型/策略结果对齐到 frame-level 的逐帧游戏数据，
并生成渲染脚本可以直接读取的规范 PKL 文件。当前默认读取
``data/02_frame_data`` 中的 frame_data，不再依赖历史导出的
``data/*.mat`` 或 ``data/*.txt``。
"""

from __future__ import annotations

import math
import re
import filecmp
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


AGENT_NAMES = [
    "global",
    "local",
    "evade_blinky",
    "evade_clyde",
    "evade_ghost3",
    "evade_ghost4",
    "approach",
    "energizer",
    "no_energizer",
]

# 旧数据中 strategy 是数字编码；这里统一转回可读标签。
STRATEGY_NUMBER_REVERSE = {
    0: "global",
    1: "local",
    2: "evade",
    3: "evade",
    4: "evade",
    5: "evade",
    6: "approach",
    7: "energizer",
    8: "no energizer",
    9: "vague",
    10: "stay",
}

# 渲染脚本使用的方向编码。这个编码需要和旧 MATLAB/Python 输出保持一致。
DIRECTION_TO_CODE = {
    "up": 1,
    "down": 2,
    "left": 3,
    "right": 4,
}
CODE_TO_DIRECTION = {value: key for key, value in DIRECTION_TO_CODE.items()}

# 模型 Q 矩阵的列顺序来自旧脚本 ``_getDir``：left/right/up/down。
MODEL_DIRECTION_ORDER = ["left", "right", "up", "down"]


class DataProcessingError(RuntimeError):
    """数据处理失败时抛出的明确异常。"""


@dataclass(frozen=True)
class SubjectPaths:
    """一个被试数据处理所需的输入/输出路径。"""

    subject: str
    frame_data: str
    gram_pkl: str
    output_dir: str
    merged_frame_pkl: str


@dataclass(frozen=True)
class ProcessingSummary:
    """数据处理完成后的摘要。

    这个对象只保存轻量统计和输出路径，不保存完整 DataFrame。命令行入口用它
    打印处理结果；统一 pipeline 用它决定下一步要读取哪个 render table。
    """

    subject: str
    ghost_count_filter: str
    input_frame_rows: int
    input_trial_count: int
    dropped_trial_count: int
    frame_rows: int
    tile_rows: int
    trial_count: int
    missing_tile_trials: list[str]
    fitted_label_missing: int
    model_dir_missing: int
    actual_dir_missing: int
    outputs: dict[str, str]


def find_subject_paths(
    subject: str,
    *,
    project_root: Path,
    output_root: Path,
    frame_data: Path | None = None,
    frame_data_dir: Path | None = None,
    gram_pkl: Path | None = None,
) -> SubjectPaths:
    """根据被试名自动定位输入文件，并构造统一输出路径。

    参数
    ----
    subject:
        被试/数据前缀，例如 ``041122-403``。
    project_root:
        当前保留为调用方项目根目录标识，不再用于推断默认输入路径。
    output_root:
        新数据文件的统一输出根目录。
    frame_data / frame_data_dir / gram_pkl:
        frame table 和 grammar/model 数据来源必须由调用方显式提供，避免正式模块
        隐式依赖某个项目目录布局。
    """

    project_root = project_root.resolve()
    output_root = output_root.resolve()
    if gram_pkl is None:
        raise DataProcessingError("必须显式传入 grammar/model pkl 路径。")
    else:
        gram_pkl = gram_pkl.resolve()

    if frame_data is None:
        if frame_data_dir is None:
            raise DataProcessingError("必须显式传入 frame_data 或 frame_data_dir。")
        search_dir = frame_data_dir.resolve()
        frame_data = _find_frame_data_for_subject(subject, frame_data_dir=search_dir, gram_pkl=gram_pkl)
    else:
        frame_data = frame_data.resolve()

    output_dir = output_root
    output_stem = frame_data.stem
    return SubjectPaths(
        subject=subject,
        frame_data=str(frame_data),
        gram_pkl=str(gram_pkl),
        output_dir=str(output_dir),
        merged_frame_pkl=str(output_dir / f"{output_stem}.pkl"),
    )


def _find_frame_data_for_subject(subject: str, *, frame_data_dir: Path, gram_pkl: Path) -> Path:
    """自动选择和 gram pkl trial 最匹配的 frame table。

    当前主流程只把 frame_data 视为逐帧游戏数据来源；
    不再读取 ``data/*.mat`` 或 ``data/*.txt``。若同一被试有多份 frame table，
    读取 gram pkl 中的 trial 名并用 ``DayTrial`` 交集选择最匹配的候选。
    """

    # 一个 subject 可能跨多个日期/session，因此候选 frame table 可能不止一份。
    # 这里先用文件名前缀粗筛，再用 gram pkl 中的 trial 名做交集评分。
    candidates = sorted(frame_data_dir.glob(f"{subject}-*.pkl")) if frame_data_dir.exists() else []
    candidates = [path.resolve() for path in candidates]

    if not candidates:
        raise DataProcessingError(f"在 {frame_data_dir} 中找不到 {subject} 的 frame table pkl。")
    if len(candidates) == 1:
        return candidates[0]

    try:
        tile_df = pd.read_pickle(gram_pkl)
        trial_column = "file" if "file" in tile_df.columns else "DayTrial"
        tile_trials = set(tile_df[trial_column].dropna().astype(str).unique())
    except Exception as exc:
        raise DataProcessingError(f"无法读取 {gram_pkl} 来判断 frame table 匹配度：{exc}") from exc

    scored: list[tuple[int, Path]] = []
    for path in candidates:
        try:
            frame_trials = set(pd.read_pickle(path)["DayTrial"].dropna().astype(str).unique())
        except Exception:
            frame_trials = set()
        overlap = len(tile_trials & frame_trials)
        scored.append((overlap, path))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_overlap, best_path = scored[0]
    tied = [item for item in scored if item[0] == best_overlap]
    if len(tied) > 1 and _all_files_identical([path for _, path in tied]):
        return best_path
    if best_overlap <= 0 or len(tied) > 1:
        details = "\n".join(f"  - overlap={overlap}, path={path}" for overlap, path in scored)
        raise DataProcessingError(f"{subject} 的 frame table 无法自动唯一匹配，请检查 frame table 与 grammar trial 是否对应：\n{details}")
    return best_path


def _all_files_identical(paths: list[Path]) -> bool:
    """候选 txt 完全相同时可安全任选其一，避免重复文件造成手动指定。"""

    if len(paths) < 2:
        return True
    first = paths[0]
    return all(filecmp.cmp(first, path, shallow=False) for path in paths[1:])


def process_subject(paths: SubjectPaths, *, ghost_count: str = "2") -> ProcessingSummary:
    """执行完整数据处理流程并写出结果文件。

    数据流：
    1. 读取逐帧 frame table；
    2. 读取 tile-level grammar/model pkl；
    3. 根据 ``--ghost-count`` 过滤 trial；
    4. 把 tile-level 的策略、model 方向和 actual 方向扩展到逐帧；
    5. 保存一个扁平的 ``{subject/session}.pkl``。
    """

    frame_df = read_frame_table(Path(paths.frame_data))
    tile_df = pd.read_pickle(paths.gram_pkl)

    _validate_frame_table(frame_df, paths.frame_data)
    frame_df = _ensure_frame_key(frame_df)
    input_frame_rows = int(len(frame_df))
    input_trial_count = int(frame_df["DayTrial"].nunique())

    # tile 数据可能来自不同阶段：有些已经包含 fitted_label/multi_dir/move_dir，
    # 有些只有 strategy/weight/Q_norm。这里统一补齐到标准列。
    tile_df = prepare_tile_table(tile_df, paths.gram_pkl)

    # 默认只保留 two-ghost trial，因为当前渲染器只绘制 g1/g2；
    # 如需完整保留四鬼 trial，可在 CLI 中设置 --ghost-count all 或 4。
    frame_df, kept_trials, dropped_trials, normalized_ghost_count = filter_trials_by_ghost_count(frame_df, ghost_count)
    tile_df = tile_df[tile_df["file"].astype(str).isin(kept_trials)].copy()

    aligned_df, missing_trials = align_tile_data_to_frames(frame_df, tile_df)

    output_dir = Path(paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _remove_obsolete_outputs(output_dir, paths.subject)
    aligned_df.to_pickle(paths.merged_frame_pkl)

    summary = ProcessingSummary(
        subject=paths.subject,
        ghost_count_filter=normalized_ghost_count,
        input_frame_rows=input_frame_rows,
        input_trial_count=input_trial_count,
        dropped_trial_count=len(dropped_trials),
        frame_rows=int(len(frame_df)),
        tile_rows=int(len(tile_df)),
        trial_count=int(frame_df["DayTrial"].nunique()),
        missing_tile_trials=missing_trials,
        fitted_label_missing=int(aligned_df["fitted_label"].isna().sum()),
        model_dir_missing=int(aligned_df["multi_dir"].isna().sum()),
        actual_dir_missing=int(aligned_df["actual_dir"].isna().sum()),
        outputs={
            "merged_frame_data": paths.merged_frame_pkl,
        },
    )

    return summary


def filter_trials_by_ghost_count(
    frame_df: pd.DataFrame,
    ghost_count: str | int,
) -> tuple[pd.DataFrame, list[str], list[str], str]:
    """按 two-/four-ghost trial 过滤逐帧表。

    ``ghost_count`` 支持：
    - ``"2"``：只保留第三、第四个鬼不存在的 trial；
    - ``"4"``：只保留第三、第四个鬼存在真实坐标的 trial；
    - ``"all"``：不过滤。

    返回过滤后的逐帧表、保留 trial、丢弃 trial、规范化后的参数值。
    """

    normalized = str(ghost_count).strip().lower()
    if normalized in {"two", "2-ghost", "two-ghost", "2ghost"}:
        normalized = "2"
    elif normalized in {"four", "4-ghost", "four-ghost", "4ghost"}:
        normalized = "4"

    if normalized not in {"2", "4", "all"}:
        raise DataProcessingError(f"--ghost-count 只支持 2、4 或 all，当前值：{ghost_count!r}")

    ordered_trials = [str(value) for value in pd.unique(frame_df["DayTrial"].astype(str))]
    if normalized == "all":
        return frame_df.copy(), ordered_trials, [], normalized

    target_count = int(normalized)
    trial_counts = infer_trial_ghost_counts(frame_df)
    kept_trials = [trial for trial in ordered_trials if trial_counts.get(trial) == target_count]
    kept_trial_set = set(kept_trials)
    dropped_trials = [trial for trial in ordered_trials if trial not in kept_trial_set]

    if not kept_trials:
        raise DataProcessingError(f"按 --ghost-count {normalized} 过滤后没有可保存的 trial。")

    filtered = frame_df[frame_df["DayTrial"].astype(str).isin(kept_trials)].copy()
    return filtered, kept_trials, dropped_trials, normalized


def infer_trial_ghost_counts(frame_df: pd.DataFrame) -> dict[str, int]:
    """根据第三、第四个鬼是否有真实坐标推断 trial 是两鬼还是四鬼。"""

    counts: dict[str, int] = {}
    for trial_name, group in frame_df.groupby("DayTrial", sort=False):
        counts[str(trial_name)] = 4 if _has_extra_ghost_coordinates(group) else 2
    return counts


def _has_extra_ghost_coordinates(group: pd.DataFrame) -> bool:
    """判断一个 trial 中 ghost3/ghost4 是否存在真实坐标。

    新数据里 two-ghost trial 的 ``g3/g4`` 像素坐标会被补成 ``inf``，
    ``ghost3Pos/ghost4Pos`` 会是 ``[]``；four-ghost trial 则有有限坐标。
    """

    pixel_columns = [column for column in ["g3pX", "g3pY", "g4pX", "g4pY"] if column in group.columns]
    if pixel_columns:
        values = group[pixel_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return bool(np.isfinite(values).any())

    position_columns = [column for column in ["ghost3Pos", "ghost4Pos"] if column in group.columns]
    if position_columns:
        return any(_looks_like_real_position(value) for column in position_columns for value in group[column])

    # 兼容更老的只有两个鬼的 frame table：没有 ghost3/4 字段时视为 two-ghost。
    return False


def _looks_like_real_position(value: object) -> bool:
    """判断位置对象是否代表真实 ghost 坐标。"""

    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return False
        if len(value) >= 2:
            try:
                return bool(np.isfinite([float(value[0]), float(value[1])]).all())
            except (TypeError, ValueError):
                return False
    text = str(value).strip()
    if text in {"", "[]", "nan", "NaN", "None", "<NA>"}:
        return False
    return bool(parse_position(text))


def read_frame_table(path: Path) -> pd.DataFrame:
    """读取逐帧游戏状态表。

    主流程使用 ``data/02_frame_data/*.pkl``。保留 CSV 读取能力只是为了
    人工排查历史文件，不作为默认数据依赖。
    """

    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise DataProcessingError(f"不支持的逐帧数据格式：{path}")


def _remove_obsolete_outputs(output_dir: Path, subject: str) -> None:
    """删除旧流程留下的 CSV/JSON 和二级目录，当前只保留扁平 PKL。"""

    for suffix in ["frame_data", "fitted_label", "model_dir", "actual_dir", "merged_frame_data"]:
        path = output_dir / f"{subject}_{suffix}.csv"
        if path.exists():
            path.unlink()
        legacy_pickle = output_dir / f"{subject}_{suffix}.pkl"
        if legacy_pickle.exists():
            legacy_pickle.unlink()
    for path in output_dir.glob(f"{subject}_*.json"):
        path.unlink()

    legacy_subject_dir = output_dir / subject
    if legacy_subject_dir.exists() and legacy_subject_dir.is_dir():
        shutil.rmtree(legacy_subject_dir)


def prepare_tile_table(tile_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """把 tile-level 表整理成 frame 对齐需要的标准列。

    输出至少包含：
    ``file``、``Step``、``fitted_label``、``multi_dir``、``actual_dir``。
    如果输入 pkl 包含 grammar 字段，也会保留 ``gram/gram_num/gramStart/gramLen``。
    """

    tile_df = tile_df.copy()
    # 旧文件有时使用 DayTrial，有时使用 file；后续统一按 file 分组。
    if "file" not in tile_df.columns:
        if "DayTrial" in tile_df.columns:
            tile_df["file"] = tile_df["DayTrial"]
        else:
            raise DataProcessingError(f"{source_name} 缺少 file/DayTrial，无法按 trial 对齐。")

    _require_columns(tile_df, ["file", "Step"], source_name)

    # fitted_label 是每个 tile/关键帧对应的策略标签。若输入只有数字 strategy，
    # 就使用旧编码表转回字符串标签。
    if "fitted_label" not in tile_df.columns:
        _require_columns(tile_df, ["strategy"], source_name)
        tile_df["fitted_label"] = tile_df["strategy"].map(_strategy_to_label)
    else:
        tile_df["fitted_label"] = tile_df["fitted_label"].map(_clean_label)

    # multi_dir 是模型预测方向。若输入已带方向，直接规范成 1-4 编码；
    # 否则根据各 agent 的 Q 值和权重重新计算。
    if "multi_dir" in tile_df.columns:
        tile_df["multi_dir"] = tile_df["multi_dir"].map(direction_to_code)
    else:
        tile_df["multi_dir"] = compute_model_direction_codes(tile_df, source_name)

    # actual_dir 是 Pacman 的真实移动方向。优先使用已有 move_dir；
    # 没有时用相邻 pacmanPos 推断。
    if "move_dir" in tile_df.columns:
        tile_df["actual_dir"] = tile_df["move_dir"].map(direction_to_code)
    else:
        tile_df["actual_dir"] = compute_actual_direction_codes(tile_df, source_name)

    return tile_df


def align_tile_data_to_frames(frame_df: pd.DataFrame, tile_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """把 tile-level 的策略/方向扩展到每一帧。

    旧数据的 tile 表通常每 25 帧一行，``Step`` 是 0-based 帧索引；
    frame 表则是逐帧记录，``Step`` 往往是 1-based。这里不重新拼 Key，
    而是直接复用 frame 表已有的 ``Key``，避免空格 padding 或 off-by-one 问题。
    """

    aligned_parts: list[pd.DataFrame] = []
    missing_trials: list[str] = []

    # 先把 tile 表按 trial 缓存，避免逐个 frame_group 反复过滤整个 DataFrame。
    tile_by_trial = {str(name): group.sort_values("Step") for name, group in tile_df.groupby("file", sort=False)}

    for trial_name, frame_group in frame_df.groupby("DayTrial", sort=False):
        trial_name = str(trial_name)
        frame_group = frame_group.copy().reset_index(drop=True)
        tile_group = tile_by_trial.get(trial_name)
        if tile_group is None or tile_group.empty:
            missing_trials.append(trial_name)
            _assign_missing_aligned_columns(frame_group)
            aligned_parts.append(frame_group)
            continue

        frame_group = _fill_one_trial(frame_group, tile_group.reset_index(drop=True))
        aligned_parts.append(frame_group)

    aligned = pd.concat(aligned_parts, ignore_index=True)
    if len(aligned) != len(frame_df):
        raise DataProcessingError(f"对齐后行数 {len(aligned)} 与逐帧数据行数 {len(frame_df)} 不一致。")
    return aligned, missing_trials


def compute_model_direction_codes(tile_df: pd.DataFrame, source_name: str) -> pd.Series:
    """根据模型权重和各策略 Q 值计算预测方向编码。"""

    q_columns = [f"{name}_Q_norm" for name in AGENT_NAMES]
    _require_columns(tile_df, q_columns + ["weight"], source_name)

    def _row_to_code(row: pd.Series) -> float:
        """把单行 Q 值和权重转换为模型预测方向编码。"""

        # 每个 agent 提供一个四方向 Q 向量，weight 表示当前 grammar/context 下
        # 各 agent 的权重。加权求和后取最大方向，即旧脚本中的模型预测方向。
        try:
            weight = _as_numeric_array(row["weight"])
            q_matrix = np.vstack([_as_numeric_array(row[col]) for col in q_columns])
            if q_matrix.shape != (len(AGENT_NAMES), 4):
                return np.nan
            scores = weight @ np.nan_to_num(q_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            direction = MODEL_DIRECTION_ORDER[int(np.argmax(scores))]
            return DIRECTION_TO_CODE[direction]
        except Exception:
            return np.nan

    return tile_df.apply(_row_to_code, axis=1)


def compute_actual_direction_codes(tile_df: pd.DataFrame, source_name: str) -> pd.Series:
    """根据相邻 pacmanPos 计算真实移动方向编码。"""

    _require_columns(tile_df, ["file", "pacmanPos"], source_name)
    parts: list[pd.Series] = []
    for _, group in tile_df.groupby("file", sort=False):
        group = group.reset_index(drop=True)
        positions = group["pacmanPos"].map(parse_position).tolist()
        directions: list[str | float] = []
        for i in range(len(positions)):
            if i + 1 < len(positions):
                directions.append(_move_direction(positions[i], positions[i + 1]))
            else:
                directions.append(np.nan)
        series = pd.Series(directions).bfill().ffill().map(direction_to_code)
        parts.append(series)
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, ignore_index=True)


def direction_to_code(value) -> float:
    """把字符串/数字方向统一成 1-4 编码。未知方向返回 NaN。"""

    if value is None:
        return np.nan
    if isinstance(value, float) and math.isnan(value):
        return np.nan
    if isinstance(value, (int, np.integer)) and int(value) in CODE_TO_DIRECTION:
        return int(value)
    if isinstance(value, float) and value.is_integer() and int(value) in CODE_TO_DIRECTION:
        return int(value)
    text = str(value).strip().lower()
    if not text or text == "nan":
        return np.nan
    return DIRECTION_TO_CODE.get(text, np.nan)


def parse_position(value) -> tuple[int, int] | None:
    """解析 ``(x, y)`` 形式的位置。无法解析时返回 None。"""

    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, list) and len(value) == 2:
        return int(value[0]), int(value[1])
    match = re.search(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", str(value))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _fill_sparse_object_values(values: pd.Series) -> pd.Series:
    """对 object 列做前向/后向填充，避免 pandas 未来版本的隐式降类型警告。"""

    items = list(values.astype("object"))
    filled: list[object] = []

    last_value: object | None = None
    has_last_value = False
    for item in items:
        if _is_missing_object(item):
            filled.append(last_value if has_last_value else pd.NA)
            continue
        last_value = item
        has_last_value = True
        filled.append(item)

    next_value: object | None = None
    has_next_value = False
    for index in range(len(filled) - 1, -1, -1):
        if _is_missing_object(filled[index]):
            filled[index] = next_value if has_next_value else pd.NA
            continue
        next_value = filled[index]
        has_next_value = True

    return pd.Series(filled, index=values.index, dtype="object")


def _is_missing_object(value: object) -> bool:
    """判断标量 object 是否缺失；非标量对象按有效值处理。"""

    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _fill_one_trial(frame_group: pd.DataFrame, tile_group: pd.DataFrame) -> pd.DataFrame:
    """把一个 trial 的 tile 数据填充到逐帧表。"""

    n_frames = len(frame_group)
    # tile_group 的 Step 通常是关键帧下标（例如 0, 25, 50...），
    # 先映射到逐帧表中的行号，再在这些行上写入 annotation。
    tile_indices = _tile_steps_to_frame_indices(tile_group["Step"], n_frames)

    label = pd.Series([pd.NA] * n_frames, dtype="object")
    model = pd.Series([np.nan] * n_frames, dtype="float")
    actual = pd.Series([np.nan] * n_frames, dtype="float")
    grammar_columns = [column for column in ["gram", "gram_num", "gramStart", "gramLen"] if column in tile_group.columns]
    grammar_values = {column: pd.Series([pd.NA] * n_frames, dtype="object") for column in grammar_columns}

    for row_idx, frame_idx in enumerate(tile_indices):
        if frame_idx is None:
            continue
        label.iloc[frame_idx] = tile_group.iloc[row_idx]["fitted_label"]
        model.iloc[frame_idx] = tile_group.iloc[row_idx]["multi_dir"]
        actual.iloc[frame_idx] = tile_group.iloc[row_idx]["actual_dir"]
        for column in grammar_columns:
            grammar_values[column].iloc[frame_idx] = tile_group.iloc[row_idx][column]

    # tile 数据只在关键帧上有值，两个关键帧之间沿用上一个值；
    # 开头缺值则用第一个有效值回填，避免 trial 首帧没有方向/标签。
    frame_group["fitted_label"] = _fill_sparse_object_values(label)
    frame_group["multi_dir"] = model.ffill().bfill()
    frame_group["actual_dir"] = actual.ffill().bfill()
    for column, values in grammar_values.items():
        frame_group[column] = _fill_sparse_object_values(values)
    return frame_group


def _assign_missing_aligned_columns(frame_group: pd.DataFrame) -> None:
    """没有 tile 数据的 trial 仍然补齐输出列，保证下游列结构稳定。"""

    frame_group["fitted_label"] = np.nan
    frame_group["multi_dir"] = np.nan
    frame_group["actual_dir"] = np.nan
    for column in ["gram", "gram_num", "gramStart", "gramLen"]:
        frame_group[column] = np.nan


def _tile_steps_to_frame_indices(steps: pd.Series, n_frames: int) -> list[int | None]:
    """把 tile 表 Step 转换成 0-based frame 下标。"""

    numeric_steps = pd.to_numeric(steps, errors="coerce")
    valid_steps = numeric_steps.dropna()
    if valid_steps.empty:
        return [None] * len(steps)

    min_step = int(valid_steps.min())
    max_step = int(valid_steps.max())
    # 绝大多数 gram pkl 是 0-based：0, 25, 50...
    if min_step == 0 and max_step < n_frames:
        offset = 0
    # 兼容少数 1-based 数据。
    elif min_step >= 1 and max_step <= n_frames:
        offset = 1
    else:
        offset = 0

    indices: list[int | None] = []
    for value in numeric_steps:
        if pd.isna(value):
            indices.append(None)
            continue
        idx = int(value) - offset
        indices.append(idx if 0 <= idx < n_frames else None)
    return indices


def _move_direction(pos1: tuple[int, int] | None, pos2: tuple[int, int] | None) -> str | float:
    """根据相邻 Pacman 坐标推断移动方向。

    输入语义：pos1 和 pos2 是相邻 tile-level 位置，缺失时为 None。
    输出语义：返回 up/down/left/right 字符串；无法推断方向时返回 NaN。
    关键约束：左右隧道穿越 `(0, 18) <-> (30, 18)` 使用旧流程的特殊方向规则。
    """

    if pos1 is None or pos2 is None:
        return np.nan
    # tunnel 穿越的特殊规则，保留旧脚本行为。
    if pos1 == (0, 18) and pos2 == (30, 18):
        return "left"
    if pos2 == (0, 18) and pos1 == (30, 18):
        return "right"
    if pos1 == pos2:
        return np.nan
    if pos1[0] == pos2[0]:
        return "down" if pos1[1] < pos2[1] else "up"
    if pos1[1] == pos2[1]:
        return "right" if pos1[0] < pos2[0] else "left"
    return np.nan


def _ensure_frame_key(frame_df: pd.DataFrame) -> pd.DataFrame:
    """确保逐帧表有 Key 列。已有 Key 时绝不重建，避免破坏旧格式。"""

    frame_df = frame_df.copy()
    if "Key" in frame_df.columns:
        frame_df["Key"] = frame_df["Key"].astype(str)
        return frame_df
    width = len(str(int(frame_df["Step"].max())))
    step_text = frame_df["Step"].map(lambda value: str(int(value)).rjust(width))
    frame_df["Key"] = frame_df["DayTrial"].astype(str) + "-" + step_text
    return frame_df


def _strategy_to_label(value) -> str | float:
    """把旧数字 strategy 编码转换为可读策略标签。"""

    if value is None:
        return np.nan
    if isinstance(value, float) and math.isnan(value):
        return np.nan
    try:
        return STRATEGY_NUMBER_REVERSE[int(value)]
    except (KeyError, TypeError, ValueError):
        return str(value).strip()


def _clean_label(value) -> str | float:
    """清理输入中的策略标签，空值保持为 NaN。"""

    if value is None:
        return np.nan
    if isinstance(value, float) and math.isnan(value):
        return np.nan
    text = str(value).strip()
    return text if text else np.nan


def _as_numeric_array(value) -> np.ndarray:
    """把 list/ndarray/字符串形式的数值数组统一成 ndarray。"""

    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    else:
        # 某些 CSV/对象列可能把数组存成字符串；这里做保守解析。
        text = str(value).replace("[", " ").replace("]", " ").replace(",", " ")
        arr = np.fromstring(text, sep=" ")
    return arr.astype(float)


def _find_unique_file(search_dirs: Iterable[Path], pattern: str, *, description: str) -> Path:
    """按优先级目录查找唯一输入文件。

    这个函数用于 grammar pkl 自动定位。如果同一目录下匹配到多份文件，直接报错，
    避免悄悄选错模型结果。
    """

    all_matches: list[Path] = []
    for directory in search_dirs:
        if directory.exists():
            matches = sorted(directory.glob(pattern))
            all_matches.extend(matches)
            # 目录顺序代表优先级；调用方应显式传入 grammar_data 目录或文件。
            if len(matches) == 1:
                return matches[0].resolve()
            if len(matches) > 1:
                joined = "\n".join(f"  - {path}" for path in matches)
                raise DataProcessingError(f"{description} 在 {directory} 中匹配到多份文件，请用参数显式指定：\n{joined}")

    if not all_matches:
        raise DataProcessingError(f"找不到 {description}，搜索模式：{pattern}")
    raise DataProcessingError(f"找不到唯一的 {description}，搜索模式：{pattern}")


def _validate_frame_table(frame_df: pd.DataFrame, source_name: str) -> None:
    """检查 frame table 是否包含渲染准备阶段的最低要求。"""

    _require_columns(frame_df, ["DayTrial", "Step", "Map"], source_name)
    map_lengths = frame_df["Map"].astype(str).str.len()
    bad_count = int((map_lengths != 29 * 36).sum())
    if bad_count:
        raise DataProcessingError(f"{source_name} 中有 {bad_count} 行 Map 长度不是 1044。")


def _require_columns(df: pd.DataFrame, columns: Iterable[str], source_name: str) -> None:
    """统一的必需列检查，错误消息中带上来源文件名。"""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise DataProcessingError(f"{source_name} 缺少必要列：{missing}")
