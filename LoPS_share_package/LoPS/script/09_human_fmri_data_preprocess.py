"""完整执行人类 fMRI ghost2 数据预处理流程。"""

from __future__ import annotations

import argparse
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


ALL_STRATEGY_NAMES = [
    "global",
    "local",
    "evade_blinky",
    "evade_clyde",
    "approach",
    "energizer",
    "no_energizer",
    "vague",
    "stay",
]
GHOST2_SEGMENT_STRATEGY_NAMES = [
    "global",
    "local",
    "evade_blinky",
    "evade_clyde",
    "approach",
    "energizer",
    "no_energizer",
    "vague",
    "stay",
]
NEIGHBOR_FEATURE_NAMES = [
    "global",
    "local",
    "evade_blinky",
    "evade_clyde",
    "approach",
    "energizer",
    "no_energizer",
    "stay",
]
STATE_OUTPUT_NAMES = ["IS1", "IS2", "PG1", "PG2", "PE", "BW10", "BB10"]
STRATEGY_OUTPUT_NAMES = [
    "global",
    "local",
    "evade_blinky",
    "evade_clyde",
    "approach",
    "energizer",
    "no_energizer",
    "stay",
    "vague",
]
STRATEGY_SYMBOLS = ["G", "L", "1", "2", "A", "E", "N", "S", "V"]
STRATEGY_ID_TO_LABEL = {
    0: "G",
    1: "L",
    2: "1",
    3: "2",
    6: "A",
    7: "E",
    8: "N",
    9: "V",
    10: "S",
}
STRATEGY_ID_TO_COLUMN = {
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


def project_root() -> Path:
    """返回当前 LoPS 仓库根目录，用于构造默认数据路径。"""

    return Path(__file__).resolve().parents[1]


def list_pickle_files(data_dir: Path) -> list[str]:
    """列出目录下的 pickle 文件名，并使用稳定顺序返回。"""

    return sorted(path.name for path in data_dir.glob("*.pkl"))


def save_pickle(data: Any, path: Path) -> None:
    """将对象保存为 pickle 文件，并确保父目录存在。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(data, file)


def build_game_id(data: pd.DataFrame) -> pd.Series:
    """根据 `DayTrial` 字段生成 game_id，用于按单局游戏分组。

    输入语义：data 必须包含 DayTrial，通常已经由上游携带 game_id。
    输出语义：返回从 DayTrial 去掉 trial round 后得到的 game_id。
    关键约束：该函数只作为缺失 game_id 时的兜底，不重新引入旧 file 字段。
    """

    return data["DayTrial"].astype(str).str.split("-").apply(lambda parts: "-".join([parts[0]] + parts[2:]))


def split_fmri_discrete_feature_data(raw_dir: Path, ghost2_dir: Path) -> dict[str, int]:
    """把离散特征数据整理为 two-ghost fMRI 输入。

    输入语义：raw_dir 是 extract_features_human 输出的离散特征目录。
    输出语义：所有文件都会写入 ghost2_dir，并保证存在 game_id 字段。
    关键约束：当前主流程已经只保留 two-ghost 数据，不再做二鬼/四鬼分流。
    """

    raw_files = list_pickle_files(raw_dir)
    ghost2_dir.mkdir(parents=True, exist_ok=True)

    written_ghost2 = 0
    for file_name in raw_files:
        data = pd.read_pickle(raw_dir / file_name)

        if "game_id" not in data.columns:
            data["game_id"] = build_game_id(data)
        data.to_pickle(ghost2_dir / file_name)
        written_ghost2 += 1

    return {
        "raw_files": len(raw_files),
        "written_ghost2_files": written_ghost2,
    }


def encode_strategy_to_onehot(strategy_id: int, strategy_names: list[str]) -> np.ndarray:
    """将单个旧策略编号转为 1/2 编码的 one-hot 向量。

    输入语义：strategy_id 保留旧流程的非连续编号，strategy_names 是当前输出列顺序。
    输出语义：返回与 strategy_names 对齐的 1/2 编码向量。
    关键约束：3/4 鬼策略列已删除，不能再把旧编号直接当作列下标。
    """

    strategy_id = int(strategy_id)
    if strategy_id not in STRATEGY_ID_TO_COLUMN:
        raise ValueError(f"未知策略编号：{strategy_id}")
    encoded = np.ones(len(strategy_names), dtype=np.int64)
    encoded[strategy_names.index(STRATEGY_ID_TO_COLUMN[strategy_id])] = 2
    return encoded


def form_strategy_onehot(data: pd.DataFrame) -> pd.DataFrame:
    """把 `strategy` 编号列展开为策略 one-hot 列。

    输入语义：data 是 two-ghost 离散特征表，包含 strategy 编号。
    输出语义：返回追加策略 one-hot 列后的 DataFrame。
    关键约束：当前数据流不再生成合并的 evade 字段，避免后续状态表携带未使用冗余列。
    """

    data = data.copy()
    data.reset_index(drop=True, inplace=True)

    # 旧数据使用 1/2 表示布尔状态：1 表示未启用，2 表示启用。
    strategies = np.stack(data["strategy"].apply(lambda value: encode_strategy_to_onehot(value, ALL_STRATEGY_NAMES)))
    encoded_strategies = (
        pd.DataFrame(strategies == strategies.max(axis=1, keepdims=True), columns=ALL_STRATEGY_NAMES).astype(int) + 1
    )
    data[ALL_STRATEGY_NAMES] = encoded_strategies

    return data


def keep_first_point_per_strategy_segment(data: pd.DataFrame, strategy_names: list[str]) -> pd.DataFrame:
    """在每个 game_id 内只保留连续相同策略段的第一个时间点。

    输入语义：data 是已经展开策略 one-hot 的单被试数据。
    输出语义：返回每个连续策略段首行组成的 formed 数据。
    关键约束：game_id 从上游持续保留；本阶段不创建或删除旧 game 字段。
    """

    data = data.copy()
    if "game_id" not in data.columns:
        data["game_id"] = build_game_id(data)
    selected_groups: list[pd.DataFrame] = []
    for _, group in data.groupby("game_id"):
        # 只要任一策略列与上一行不同，就认为进入了新的策略段，需要保留该行。
        first_point_mask = (group[strategy_names] == group[strategy_names].shift(1)).sum(axis=1) < len(strategy_names)
        selected_groups.append(deepcopy(group.loc[first_point_mask]))

    formed_data = pd.concat(selected_groups, axis=0)
    formed_data = formed_data.sort_index()
    formed_data.reset_index(drop=True, inplace=True)
    return formed_data


def form_single_subject_data(input_path: Path, output_path: Path) -> pd.DataFrame:
    """将单个 ghost2 被试离散特征数据转换为 formed 数据并保存。"""

    data = pd.read_pickle(input_path)
    data = form_strategy_onehot(data)
    data = keep_first_point_per_strategy_segment(data, GHOST2_SEGMENT_STRATEGY_NAMES)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_pickle(output_path)
    return data


def form_ghost2_data(ghost2_dir: Path, formed_dir: Path) -> dict[str, int]:
    """批量生成所有 ghost2 被试的 formed 数据。"""

    file_names = list_pickle_files(ghost2_dir)
    if not file_names:
        raise FileNotFoundError(f"没有在 {ghost2_dir} 中找到 ghost2 离散特征数据。")

    formed_dir.mkdir(parents=True, exist_ok=True)
    for file_name in file_names:
        form_single_subject_data(ghost2_dir / file_name, formed_dir / file_name)
    return {"input_ghost2_files": len(file_names), "formed_ghost2_files": len(file_names)}


def load_subject_tables(data_dir: Path, file_names: list[str]) -> dict[str, pd.DataFrame]:
    """按文件名读取被试数据表，返回以文件名为键的字典。"""

    return {file_name: pd.read_pickle(data_dir / file_name) for file_name in file_names}


def compute_subject_feature(data: pd.DataFrame) -> np.ndarray:
    """根据策略出现次数计算单个被试的近邻搜索特征。"""

    strategy_counts = (data[NEIGHBOR_FEATURE_NAMES] - 1).to_numpy().sum(axis=0)
    total_count = strategy_counts.sum()
    if total_count == 0:
        raise ValueError("策略出现次数总和为 0，无法计算近邻特征。")
    return strategy_counts / total_count


def build_normalized_feature_matrix(subject_data: dict[str, pd.DataFrame], file_names: list[str]) -> np.ndarray:
    """计算所有被试的归一化策略特征矩阵。"""

    features = np.array([compute_subject_feature(subject_data[file_name]) for file_name in file_names])
    feature_min = features.min(axis=0)
    feature_max = features.max(axis=0)
    feature_range = feature_max - feature_min
    if np.any(feature_range == 0):
        raise ValueError("至少一个近邻特征在所有被试上没有变化，无法执行 min-max 归一化。")

    features = (features - feature_min) / feature_range

    # 旧流程对 approach、energizer、stay 三个维度加权，以改变近邻距离。
    features[:, [4, 5, 7]] = 2 * features[:, [4, 5, 7]]
    return features


def find_nearest_subject_indices(
    features: np.ndarray,
    min_neighbor_count: int = 5,
    threshold_start: float = 1.0,
    threshold_step: float = 0.1,
) -> list[list[int]]:
    """根据归一化特征寻找每个被试的邻近被试索引。"""

    nearest_model = NearestNeighbors(n_neighbors=len(features))
    nearest_model.fit(features)
    distances, indices = nearest_model.kneighbors(features)

    neighbors: list[list[int]] = []
    for subject_index in range(len(distances)):
        threshold = threshold_start

        # 逐步放宽阈值，直到每个邻域至少包含指定数量的被试。
        while True:
            selected_positions = np.where(distances[subject_index].reshape(-1, 1) <= threshold)[0]
            if len(selected_positions) >= min_neighbor_count:
                neighbors.append(list(indices[subject_index][selected_positions]))
                break
            threshold += threshold_step
    return neighbors


def get_neighbor_file_names(file_names: list[str], neighbor_indices: list[int]) -> list[np.str_]:
    """将近邻索引转换为文件名列表，并保留旧结果中的 NumPy 字符串类型。"""

    return list(np.array(file_names)[neighbor_indices])


def build_strategy_sequence(subject_data: dict[str, pd.DataFrame], neighbor_file_names: list[np.str_]) -> dict[str, Any]:
    """合并一个近邻组的 formed 数据，并生成最终策略序列结构。"""

    combined_data = pd.concat([subject_data[str(file_name)] for file_name in neighbor_file_names], axis=0)
    state = combined_data[STATE_OUTPUT_NAMES]
    strategy = combined_data[STRATEGY_OUTPUT_NAMES]
    strategy_label = combined_data["strategy"].apply(lambda value: STRATEGY_ID_TO_LABEL[int(value)])

    # 序列符号使用策略输出列的 argmax 位置生成，列顺序决定了符号映射。
    strategy_indices = np.argmax(strategy.to_numpy(), axis=1)
    sequence = "".join(STRATEGY_SYMBOLS[index] for index in strategy_indices)

    return {
        "seq": sequence,
        "S": STRATEGY_SYMBOLS,
        "state": state,
        "strategy": strategy,
        "strategyLabel": strategy_label,
        "fileNames": neighbor_file_names,
    }


def consolidate_strategy_sequences(formed_dir: Path, output_dir: Path) -> dict[str, int]:
    """根据 formed 数据计算近邻组，并保存每个被试对应的最终策略序列。"""

    file_names = list_pickle_files(formed_dir)
    if not file_names:
        raise FileNotFoundError(f"没有在 {formed_dir} 中找到 formed 数据。")

    subject_data = load_subject_tables(formed_dir, file_names)
    features = build_normalized_feature_matrix(subject_data, file_names)
    neighbors = find_nearest_subject_indices(features)

    output_dir.mkdir(parents=True, exist_ok=True)
    for neighbor_indices in neighbors:
        neighbor_file_names = get_neighbor_file_names(file_names, neighbor_indices)
        result = build_strategy_sequence(subject_data, neighbor_file_names)
        save_pickle(result, output_dir / str(neighbor_file_names[0]))

    return {"formed_input_files": len(file_names), "strategy_sequence_files": len(neighbors)}


def process_human_fmri_data(
    raw_discrete_dir: Path,
    ghost2_discrete_dir: Path,
    formed_ghost2_dir: Path,
    strategy_sequence_dir: Path,
) -> dict[str, dict[str, int]]:
    """完整执行当前 human fMRI 数据预处理流程。"""

    split_summary = split_fmri_discrete_feature_data(raw_discrete_dir, ghost2_discrete_dir)

    # 当前旧项目的 raw 目录为空，旧脚本会继续使用已经存在的 ghost2 目录；这里显式检查该输入。
    if not list_pickle_files(ghost2_discrete_dir):
        raise FileNotFoundError(
            f"{raw_discrete_dir} 中没有 raw 数据，且 {ghost2_discrete_dir} 中没有可继续处理的 ghost2 数据。"
        )

    formed_summary = form_ghost2_data(ghost2_discrete_dir, formed_ghost2_dir)
    sequence_summary = consolidate_strategy_sequences(formed_ghost2_dir, strategy_sequence_dir)
    return {
        "split": split_summary,
        "formed": formed_summary,
        "strategy_sequence": sequence_summary,
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数，允许外部指定各阶段输入输出目录。"""

    data_root = project_root() / "data"
    parser = argparse.ArgumentParser(description="完整执行 human fMRI ghost2 数据预处理。")
    parser.add_argument(
        "--raw-discrete-dir",
        type=Path,
        default=data_root / "08_discrete_feature_data",
        help="原始 fMRI 离散特征数据目录；当前旧数据状态下该目录为空。",
    )
    parser.add_argument(
        "--ghost2-discrete-dir",
        type=Path,
        default=data_root / "09_fmri_discrete_feature_data_ghost2",
        help="ghost2 离散特征数据目录；既是分流输出，也是 formed 阶段输入。",
    )
    parser.add_argument(
        "--formed-ghost2-dir",
        type=Path,
        default=data_root / "09_fmri_formed_data_ghost2",
        help="ghost2 formed 数据输出目录。",
    )
    parser.add_argument(
        "--strategy-sequence-dir",
        type=Path,
        default=data_root / "09_strategy_sequence",
        help="最终 StrategySequence 输出目录。",
    )
    return parser.parse_args()


def main() -> None:
    """运行完整 human fMRI 数据预处理流程并打印阶段摘要。"""

    args = parse_args()
    summary = process_human_fmri_data(
        raw_discrete_dir=args.raw_discrete_dir,
        ghost2_discrete_dir=args.ghost2_discrete_dir,
        formed_ghost2_dir=args.formed_ghost2_dir,
        strategy_sequence_dir=args.strategy_sequence_dir,
    )
    print(
        "processed "
        f"raw={summary['split']['raw_files']} "
        f"ghost2={summary['formed']['input_ghost2_files']} "
        f"formed={summary['formed']['formed_ghost2_files']} "
        f"strategy_sequence={summary['strategy_sequence']['strategy_sequence_files']}"
    )


if __name__ == "__main__":
    main()
