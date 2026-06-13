#!/usr/bin/env python3
"""从人类 CorrectedWeightData 中提取连续特征和离散特征。"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INF_VALUE = 100
DIRECTION_NAMES = ["left", "right", "up", "down"]
DISCRETE_OUTPUT_COLUMNS = [
    "PG1",
    "PG2",
    "PE",
    "BW10",
    "BB10",
    "IS1",
    "IS2",
]
FEATURE_APPEND_COLUMNS = [
    "revised_normalized_weight",
    "normalized_weight",
    "weight",
    "row_id",
    "strategy",
    "DayTrial",
    "frame_id",
    "game_id",
    "action_dir",
]


def project_root() -> Path:
    """返回当前 LoPS 仓库根目录。

    输入语义：无输入，路径由脚本文件位置推导。
    输出语义：返回可用于构造默认数据目录的仓库根路径。
    关键约束：默认路径只出现在脚本层，正式模块不保存旧项目绝对路径。
    """

    return Path(__file__).resolve().parents[2]


def list_pickle_files(data_dir: Path) -> list[Path]:
    """按稳定顺序列出目录中的 pickle 文件。

    输入语义：data_dir 是待处理的扁平数据目录。
    输出语义：返回按文件名排序后的 `.pkl` 路径列表。
    关键约束：目录必须存在，且至少包含一个 pickle 文件。
    """

    if not data_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{data_dir}")
    file_paths = sorted(data_dir.glob("*.pkl"))
    if not file_paths:
        raise FileNotFoundError(f"输入目录中没有 pickle 文件：{data_dir}")
    return file_paths


def read_adjacent_map(filename: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """读取 fMRI 迷宫邻接表。

    输入语义：filename 指向旧流程使用的 `adjacent_map_fmri.csv`。
    输出语义：返回以地图坐标为键、四个方向邻接坐标为值的字典。
    关键约束：保留旧脚本对左右隧道 `(0, 18)` 和 `(30, 18)` 的手工补丁。
    """

    adjacent_data = pd.read_csv(filename)

    # CSV 中坐标以字符串形式保存；旧脚本用 eval 还原 tuple，这里保持同一语义。
    for column_name in ["pos", "left", "right", "up", "down"]:
        adjacent_data[column_name] = adjacent_data[column_name].apply(
            lambda value: eval(value) if not isinstance(value, float) else np.nan
        )

    dict_adjacent_data: dict[tuple[int, int], dict[str, Any]] = {}
    for row in adjacent_data.values:
        dict_adjacent_data[row[1]] = {
            "left": row[2] if not isinstance(row[2], float) else np.nan,
            "right": row[3] if not isinstance(row[3], float) else np.nan,
            "up": row[4] if not isinstance(row[4], float) else np.nan,
            "down": row[5] if not isinstance(row[5], float) else np.nan,
        }

    # 隧道口在原 CSV 中可能缺失或不完整；旧脚本固定补成环形相邻关系。
    if (0, 18) not in dict_adjacent_data:
        dict_adjacent_data[(0, 18)] = {}
    if (30, 18) not in dict_adjacent_data:
        dict_adjacent_data[(30, 18)] = {}
    dict_adjacent_data[(0, 18)]["left"] = (30, 18)
    dict_adjacent_data[(0, 18)]["right"] = (1, 18)
    dict_adjacent_data[(0, 18)]["up"] = np.nan
    dict_adjacent_data[(0, 18)]["down"] = np.nan
    dict_adjacent_data[(30, 18)]["left"] = (29, 18)
    dict_adjacent_data[(30, 18)]["right"] = (0, 18)
    dict_adjacent_data[(30, 18)]["up"] = np.nan
    dict_adjacent_data[(30, 18)]["down"] = np.nan
    return dict_adjacent_data


def read_location_distance(filename: Path) -> dict[tuple[int, int], dict[tuple[int, int], int]]:
    """读取 fMRI 迷宫任意两点之间的最短路径距离。

    输入语义：filename 指向旧流程使用的 `dij_distance_map_fmri.csv`。
    输出语义：返回两层字典，第一层是起点坐标，第二层是终点坐标到距离。
    关键约束：保留旧脚本对左右隧道距离的手工补丁。
    """

    locs_df = pd.read_csv(filename)[["pos1", "pos2", "dis"]]
    locs_df.pos1 = locs_df.pos1.apply(eval)
    locs_df.pos2 = locs_df.pos2.apply(eval)

    dict_locs_df: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
    for row in locs_df.values:
        if row[0] not in dict_locs_df:
            dict_locs_df[row[0]] = {}
        dict_locs_df[row[0]][row[1]] = row[2]

    # 隧道两端视作相邻格；这里与旧脚本保持完全一致。
    dict_locs_df[(0, 18)][(30, 18)] = 1
    dict_locs_df[(0, 18)][(1, 18)] = 1
    dict_locs_df[(30, 18)][(0, 18)] = 1
    dict_locs_df[(30, 18)][(29, 18)] = 1
    return dict_locs_df


def normalize_empty_lists(data: pd.DataFrame) -> pd.DataFrame:
    """把旧数据中的空列表统一替换为浮点 0。

    输入语义：data 是旧流程任一阶段保存的 DataFrame。
    输出语义：返回就地修改后的同一个 DataFrame。
    关键约束：只替换空列表；非空列表代表坐标集合或权重向量，必须原样保留。
    """

    for column_name in data.columns:
        data[column_name] = data[column_name].apply(
            lambda value: float(0) if isinstance(value, list) and len(value) == 0 else value
        )
    return data


def adjacent_distance(
    pacman_pos: tuple[int, int],
    target_pos: tuple[int, int] | list[tuple[int, int]],
    direction: str,
    adjacent_data: dict[tuple[int, int], dict[str, Any]],
    locs_df: dict[tuple[int, int], dict[tuple[int, int], int]],
) -> int:
    """计算 Pacman 朝某方向移动一步后到目标位置的最短距离。

    输入语义：pacman_pos 是当前位置，target_pos 是 ghost 或 energizer 坐标，
    direction 是四个方向之一，adjacent_data 和 locs_df 是地图常量。
    输出语义：返回相邻位置到目标的距离；若该方向不可走，返回 INF_VALUE。
    关键约束：ghost 位于家门口特殊坐标时按旧脚本移动到门外坐标。
    """

    if isinstance(target_pos, list):
        target_pos = target_pos[0]

    # 旧脚本先检查该方向是否为墙；不可走时直接给最大距离编码。
    if isinstance(adjacent_data[pacman_pos][direction], float):
        return INF_VALUE

    if direction == "left":
        adjacent = (pacman_pos[0] - 1, pacman_pos[1])
    elif direction == "right":
        adjacent = (pacman_pos[0] + 1, pacman_pos[1])
    elif direction == "up":
        adjacent = (pacman_pos[0], pacman_pos[1] - 1)
    elif direction == "down":
        adjacent = (pacman_pos[0], pacman_pos[1] + 1)
    else:
        raise ValueError(f"未知方向：{direction}")

    # ghost 在 home 门口时，旧流程把它投影到可寻路位置。
    if target_pos == (14, 20):
        target_pos = (14, 19)
    if target_pos == (15, 20):
        target_pos = (15, 19)
    if target_pos == (16, 20):
        target_pos = (16, 19)

    return 0 if adjacent == target_pos else locs_df[adjacent][target_pos]


def compute_target_distance(
    trial: pd.DataFrame,
    pacman_column: str,
    target_column: str,
    adjacent_data: dict[tuple[int, int], dict[str, Any]],
    locs_df: dict[tuple[int, int], dict[tuple[int, int], int]],
) -> np.ndarray:
    """计算 Pacman 到某类目标的四方向最短距离特征。

    输入语义：trial 是单个被试数据，pacman_column 通常是 `pacmanPos`，
    target_column 可以是 ghost 位置列或 `energizers`。
    输出语义：返回每一行上四个可选方向中距离目标最近的距离。
    关键约束：energizer 是坐标列表，需要先对列表内所有 energizer 取最小距离。
    """

    directional_distances = []
    for direction in DIRECTION_NAMES:
        if target_column == "energizers":
            values = trial[[pacman_column, target_column]].apply(
                lambda row: INF_VALUE
                if isinstance(row[target_column], float)
                else np.min(
                    [
                        adjacent_distance(row[pacman_column], target, direction, adjacent_data, locs_df)
                        for target in row[target_column]
                    ]
                ),
                axis=1,
            )
        else:
            values = trial[[pacman_column, target_column]].apply(
                lambda row: INF_VALUE
                if isinstance(row[target_column], float)
                else adjacent_distance(row[pacman_column], row[target_column], direction, adjacent_data, locs_df),
                axis=1,
            )
        directional_distances.append(values)

    return np.min(np.array(directional_distances), axis=0)


def compute_eat_energizer_flags(trial: pd.DataFrame) -> list[bool]:
    """根据相邻时间点 energizer 集合变化判断是否吃到 energizer。

    输入语义：trial 必须包含旧数据中的 `energizers` 列。
    输出语义：返回与 trial 等长的布尔列表，最后一帧固定为 False。
    关键约束：旧脚本只通过列表变短或列表变为 float 判断，不使用其它事件字段。
    """

    eat_energizer: list[bool] = []
    for index in range(len(trial) - 1):
        next_value = trial["energizers"][index + 1]
        current_value = trial["energizers"][index]
        if isinstance(next_value, float) and isinstance(current_value, list):
            eat_energizer.append(True)
        elif isinstance(next_value, list) and isinstance(current_value, list) and len(current_value) > len(next_value):
            eat_energizer.append(True)
        else:
            eat_energizer.append(False)
    eat_energizer.append(False)
    return eat_energizer


def count_beans_by_distance(
    trial: pd.DataFrame,
    locs_df: dict[tuple[int, int], dict[tuple[int, int], int]],
    *,
    beyond_threshold: bool,
) -> pd.Series:
    """统计当前位置附近或远处的 beans 数量。

    输入语义：trial 包含 `pacmanPos` 和 `beans`；beyond_threshold 为 False
    时统计距离小于 10 的 beans，为 True 时统计距离大于 10 的 beans。
    输出语义：返回每一行对应的 beans 数量 Series。
    关键约束：旧变量名叫 `beans_within_5`，但实际阈值是 10；新输出改名为
    `beans_within_10`，让字段名和含义一致。
    """

    def count_one(row: pd.Series) -> int:
        """统计单行中满足距离条件的 beans 数量。"""

        if isinstance(row.beans, float):
            return 0

        distances = np.array(
            [0 if row.pacmanPos == bean_pos else locs_df[row.pacmanPos][bean_pos] for bean_pos in row.beans]
        )
        if beyond_threshold:
            return len(np.where(distances > 10)[0])
        return len(np.where(distances < 10)[0])

    return trial[["pacmanPos", "beans"]].apply(count_one, axis=1)


def extract_feature(
    trial: pd.DataFrame,
    adjacent_data: dict[tuple[int, int], dict[str, Any]],
    locs_df: dict[tuple[int, int], dict[tuple[int, int], int]],
) -> pd.DataFrame:
    """提取旧流程中的连续人类行为特征。

    输入语义：trial 是一个被试的 CorrectedWeightData 表，地图常量由调用方传入。
    输出语义：返回 two-ghost 基础特征列组成的 DataFrame，后续会补充权重和索引字段。
    关键约束：所有空列表、距离阈值和 `EE` 判定都按旧脚本执行。
    """

    trial = trial.reset_index(drop=True)
    eat_energizer = compute_eat_energizer_flags(trial)
    trial = normalize_empty_lists(trial)

    pg1 = compute_target_distance(trial, "pacmanPos", "ghost1Pos", adjacent_data, locs_df)
    pg2 = compute_target_distance(trial, "pacmanPos", "ghost2Pos", adjacent_data, locs_df)
    pe = compute_target_distance(trial, "pacmanPos", "energizers", adjacent_data, locs_df)

    beans_10step = count_beans_by_distance(trial, locs_df, beyond_threshold=False)
    beans_over_10step = count_beans_by_distance(trial, locs_df, beyond_threshold=True)

    return pd.DataFrame(
        data={
            "ifscared1": trial.ifscared1,
            "ifscared2": trial.ifscared2,
            "PG1": pg1,
            "PG2": pg2,
            "PE": pe,
            "beans_within_10": beans_10step,
            "beans_beyond_10": beans_over_10step,
            "EE": eat_energizer,
            "weight": trial["weight"],
        }
    )


def combine_evade(predictors: pd.DataFrame, feature_data: pd.DataFrame) -> pd.DataFrame:
    """把最近 ghost 对应的 PG/IS 编码合并为单列。

    输入语义：predictors 包含每个 ghost 的离散 PG/IS 编码，feature_data 包含原始距离。
    输出语义：返回增加 `PG` 和 `IS` 两列后的 predictors。
    关键约束：最近 ghost 由原始 PG1/PG2 的最小值决定，平局时沿用 numpy.argmin 的首个最小值。
    """

    combined_pg = []
    combined_is = []
    for index in range(len(feature_data)):
        nearest_ghost_index = np.argmin(np.array(feature_data[["PG1", "PG2"]].iloc[index]))
        if nearest_ghost_index == 0:
            combined_pg.append(predictors["PG1"].iloc[index])
            combined_is.append(predictors["IS1"].iloc[index])
        elif nearest_ghost_index == 1:
            combined_pg.append(predictors["PG2"].iloc[index])
            combined_is.append(predictors["IS2"].iloc[index])

    predictors["PG"] = np.array(combined_pg, dtype=int)
    predictors["IS"] = np.array(combined_is, dtype=int)
    return predictors


def predictor_for_prediction(feature_data: pd.DataFrame) -> pd.DataFrame:
    """把连续特征离散化为 grammar 上游使用的 predictor 表。

    输入语义：feature_data 是 extract_feature 输出并已补齐旧权重字段的数据。
    输出语义：返回 two-ghost `DiscreteFeatureData` 的离散特征列。
    关键约束：ghost 死亡态、距离分箱和 beans 二值化都按旧脚本逐步执行。
    """

    df = feature_data.copy()

    # 死亡 ghost 的距离被固定为 100，再参与 0/1 距离分箱。
    df.loc[df.ifscared1 == 3, "PG1"] = 100
    df.loc[df.ifscared2 == 3, "PG2"] = 100

    for ghost_index in [1, 2]:
        scared_column = f"ifscared{ghost_index}"
        df[f"if_exist{ghost_index}"] = (df[scared_column] != -1).astype(int)
        df[f"if_normal{ghost_index}"] = (df[scared_column] <= 2).astype(int)
        df[f"if_dead{ghost_index}"] = (df[scared_column] == 3).astype(int)
        df[f"if_scared{ghost_index}"] = (df[scared_column] >= 4).astype(int)

    is_encode = pd.DataFrame()
    for ghost_index in [1, 2]:
        status_columns = [f"if_{status}{ghost_index}" for status in ["normal", "dead", "scared"]]
        is_encode[f"IS_EXIST{ghost_index}"] = df[f"if_exist{ghost_index}"]
        is_encode[f"IS{ghost_index}"] = np.argmax(df[status_columns].values, 1)

    numerical_columns = ["PG1", "PG2", "PE"]
    distance_bins = [0, 11, 101]
    numerical_encode1 = pd.concat(
        [pd.cut(df[column], distance_bins, right=False, labels=[0, 1]) for column in numerical_columns],
        axis=1,
    )
    numerical_encode1.columns = numerical_columns

    numerical_encode2 = pd.DataFrame()
    # 旧脚本先做归一化，再只检查是否为 0；这个步骤不影响二值结果，但保留以保证流程一致。
    df["beans_within_10"] = np.array(df["beans_within_10"]) / np.max(df["beans_within_10"])
    df["beans_beyond_10"] = np.array(df["beans_beyond_10"]) / np.max(df["beans_beyond_10"])
    numerical_encode2["BW10"] = 1 - np.array(df["beans_within_10"] == 0, dtype=int)
    numerical_encode2["BB10"] = 1 - np.array(df["beans_beyond_10"] == 0, dtype=int)

    predictors = pd.concat([numerical_encode1, numerical_encode2, is_encode], axis=1)
    predictors = combine_evade(predictors, df)
    predictors = predictors[DISCRETE_OUTPUT_COLUMNS]
    for column_name in predictors.columns:
        predictors[column_name] = predictors[column_name].astype(int)
    return predictors


def output_file_name(input_path: Path) -> str:
    """根据旧脚本规则生成输出文件名。

    输入语义：input_path 是 CorrectedWeightData 中带长后缀的文件路径。
    输出语义：返回由前两个 `-` 分段组成的短文件名，例如 `031222-401.pkl`。
    关键约束：该规则直接影响下游被试文件名，不能随意改动。
    """

    return "-".join(input_path.name.split("-")[:2]) + ".pkl"


def append_legacy_columns(features: pd.DataFrame, source_data: pd.DataFrame) -> pd.DataFrame:
    """向连续特征表补齐后续流程需要的标准分析字段。

    输入语义：features 是基础连续特征表，source_data 是原始 CorrectedWeightData。
    输出语义：返回带权重、策略、行号、trial id 和动作字段的 DataFrame。
    关键约束：`weight` 是已存在列，重新赋值不会改变列位置。
    """

    for column_name in FEATURE_APPEND_COLUMNS:
        features[column_name] = np.array(source_data[column_name])
    features.reset_index(drop=True, inplace=True)
    return features


def append_discrete_legacy_columns(
    predictors: pd.DataFrame,
    features: pd.DataFrame,
    source_data: pd.DataFrame,
) -> pd.DataFrame:
    """向离散 predictor 表补齐后续 fMRI 流程需要的标准字段。

    输入语义：predictors 是离散状态特征，source_data 提供权重、策略和 trial id。
    输出语义：返回可直接进入 human_fmri_data_preprocess 的 DataFrame。
    关键约束：`EE`、合并 PG/IS 和 ghost 存在标记不再进入离散主链路。
    """

    for column_name in FEATURE_APPEND_COLUMNS:
        predictors[column_name] = np.array(source_data[column_name])
    return predictors


def process_one_file(
    input_path: Path,
    constant_dir: Path,
    feature_output_dir: Path,
    discrete_output_dir: Path,
) -> dict[str, Any]:
    """处理单个被试文件并写出连续特征和离散特征。

    输入语义：input_path 指向单个 CorrectedWeightData pickle，constant_dir 提供地图常量。
    输出语义：写出两个 pickle 文件，并返回包含文件名和行数的摘要。
    关键约束：每个文件独立处理，因此并行执行不会改变结果内容。
    """

    locs_df = read_location_distance(constant_dir / "dij_distance_map_fmri.csv")
    adjacent_data = read_adjacent_map(constant_dir / "adjacent_map_fmri.csv")

    source_data = pd.read_pickle(input_path)
    source_data = normalize_empty_lists(source_data)

    features = extract_feature(source_data, adjacent_data, locs_df)
    features = append_legacy_columns(features, source_data)

    predictors = predictor_for_prediction(features)
    predictors = append_discrete_legacy_columns(predictors, features, source_data)

    file_name = output_file_name(input_path)
    feature_path = feature_output_dir / file_name
    discrete_path = discrete_output_dir / file_name
    features.to_pickle(feature_path)
    predictors.to_pickle(discrete_path)

    return {
        "input_file": input_path.name,
        "output_file": file_name,
        "rows": int(len(source_data)),
        "feature_columns": int(len(features.columns)),
        "discrete_columns": int(len(predictors.columns)),
    }


def process_extract_features_human(
    input_dir: Path,
    constant_dir: Path,
    feature_output_dir: Path,
    discrete_output_dir: Path,
    *,
    processes: int,
) -> list[dict[str, Any]]:
    """批量执行人类特征提取流程。

    输入语义：所有目录均为 LoPS 仓库内扁平目录；processes 控制并行进程数。
    输出语义：返回每个输入文件的处理摘要，同时写出 feature 和 discrete feature。
    关键约束：输出目录会自动创建，已存在的同名输出会被覆盖。
    """

    input_paths = list_pickle_files(input_dir)
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    discrete_output_dir.mkdir(parents=True, exist_ok=True)

    worker = partial(
        process_one_file,
        constant_dir=constant_dir,
        feature_output_dir=feature_output_dir,
        discrete_output_dir=discrete_output_dir,
    )
    if processes <= 1:
        return [worker(input_path) for input_path in input_paths]

    process_count = min(processes, len(input_paths))
    with multiprocessing.Pool(processes=process_count) as pool:
        return pool.map(worker, input_paths)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输入语义：允许调用方覆盖输入、常量、输出目录和并行度。
    输出语义：返回可直接传给批处理函数的 argparse 命名空间。
    关键约束：默认目录全部位于 LoPS 仓库内，不依赖旧项目路径。
    """

    data_root = project_root() / "pipeline_data"
    default_processes = min(34, os.cpu_count() or 1)
    parser = argparse.ArgumentParser(description="提取人类 CorrectedWeightData 的连续特征和离散特征。")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=data_root / "revise_human_weight" / "corrected_weight_data",
        help="扁平 CorrectedWeightData 输入目录。",
    )
    parser.add_argument(
        "--constant-dir",
        type=Path,
        default=data_root / "constant_data",
        help="包含 fMRI 邻接表和距离表的常量目录。",
    )
    parser.add_argument(
        "--feature-output-dir",
        type=Path,
        default=data_root / "extract_features_human" / "feature_data",
        help="连续特征输出目录。",
    )
    parser.add_argument(
        "--discrete-output-dir",
        type=Path,
        default=data_root / "extract_features_human" / "discrete_feature_data",
        help="离散特征输出目录。",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=default_processes,
        help="并行进程数；设为 1 时串行运行。",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口：批量提取并打印摘要。"""

    args = parse_args()
    summaries = process_extract_features_human(
        input_dir=args.input_dir,
        constant_dir=args.constant_dir,
        feature_output_dir=args.feature_output_dir,
        discrete_output_dir=args.discrete_output_dir,
        processes=args.processes,
    )
    print(
        "extract_features_human 完成 "
        f"input_files={len(summaries)} "
        f"rows={sum(item['rows'] for item in summaries)} "
        f"feature_dir={args.feature_output_dir} "
        f"discrete_dir={args.discrete_output_dir}"
    )


if __name__ == "__main__":
    main()
