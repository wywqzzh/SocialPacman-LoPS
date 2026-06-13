#!/usr/bin/env python3
"""运行旧版 human fMRI 数据整理逻辑的临时验证入口。

该脚本只服务于深度验证流程：输入为旧链路特征提取后的离散特征数据，输出到
``data_temp`` 下的本轮验证目录。旧项目没有独立的 ghost2/ghost4 分流入口，旧流程
默认使用已经整理好的 ``HumanData/DiscreteFeatureData/session2``。因此这里先按
当前重构后显式化的 ``IS_EXIST3`` 规则拆分数据，再调用旧版 ``DataFormedHuman``
中的 formed 处理函数，并用旧版近邻合并算法生成策略序列。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


ALL_NEIGHBOR_FEATURE_NAMES = [
    "global",
    "local",
    "evade_blinky",
    "evade_clyde",
    "approach",
    "energizer",
    "no_energizer",
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
STATE_OUTPUT_NAMES = ["IS1", "IS2", "PG1", "PG2", "PE", "BN5", "BN10"]
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
    4: "3",
    5: "4",
    6: "A",
    7: "E",
    8: "N",
    9: "V",
    10: "S",
}


def list_pickle_files(data_dir: Path) -> list[str]:
    """列出目录中的 pkl 文件名，并返回稳定排序后的列表。"""

    return sorted(path.name for path in data_dir.glob("*.pkl"))


def build_game_id(data: pd.DataFrame) -> pd.Series:
    """根据旧版 ``file`` 字段生成单局游戏标识。"""

    return data["file"].str.split("-").apply(lambda parts: "-".join([parts[0]] + parts[2:]))


def load_legacy_data_formed_module(legacy_root: Path) -> Any:
    """导入旧项目 ``GrammarInduction/DataFormedHuman.py`` 模块。

    输入语义：``legacy_root`` 是旧项目根目录。
    输出语义：返回导入后的旧版模块对象，可调用其中 formed 数据处理函数。
    关键约束：旧模块使用包路径 ``GrammarInduction`` 与 ``PGM``，因此需要把旧项目根
    目录插入 ``sys.path``。
    """

    legacy_root = legacy_root.resolve()
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))

    module_path = legacy_root / "GrammarInduction" / "DataFormedHuman.py"
    spec = importlib.util.spec_from_file_location("legacy_data_formed_human", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入旧版 DataFormedHuman：{module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_discrete_feature_data(raw_dir: Path, ghost2_dir: Path, ghost4_dir: Path) -> dict[str, int]:
    """按旧数据语义将离散特征分成 ghost2 与 ghost4 数据。

    旧流程没有单独脚本执行该步骤，但下游 formed 逻辑只处理 ghost2 数据。这里使用
    ``IS_EXIST3`` 判断每个 game 是否属于四鬼条件：整局该字段都为 1 时归为 ghost4，
    否则归为 ghost2。
    """

    ghost2_dir.mkdir(parents=True, exist_ok=True)
    ghost4_dir.mkdir(parents=True, exist_ok=True)
    written_ghost2 = 0
    written_ghost4 = 0
    raw_files = list_pickle_files(raw_dir)

    for file_name in raw_files:
        data = pd.read_pickle(raw_dir / file_name)

        # ``game`` 是分流阶段保留下来的旧式辅助列，formed 阶段会重新计算并删除。
        data["game"] = build_game_id(data)
        ghost2_groups: list[pd.DataFrame] = []
        ghost4_groups: list[pd.DataFrame] = []
        for _, group in data.groupby("game"):
            if np.sum(group["IS_EXIST3"]) / len(group) == 1:
                ghost4_groups.append(deepcopy(group))
            else:
                ghost2_groups.append(deepcopy(group))

        if ghost2_groups:
            pd.concat(ghost2_groups, axis=0).to_pickle(ghost2_dir / file_name)
            written_ghost2 += 1
        if ghost4_groups:
            pd.concat(ghost4_groups, axis=0).to_pickle(ghost4_dir / file_name)
            written_ghost4 += 1

    return {
        "raw_files": len(raw_files),
        "written_ghost2_files": written_ghost2,
        "written_ghost4_files": written_ghost4,
    }


def form_one_file(module: Any, input_path: Path, output_path: Path) -> int:
    """使用旧版函数把单个 ghost2 离散特征文件转换为 formed 数据。

    输入语义：``input_path`` 是 ghost2 离散特征表。
    输出语义：写出旧版 one-hot 与连续策略段首点筛选后的 formed 表，并返回行数。
    关键约束：调用旧模块的 ``formStrategyToOnehot`` 与 ``keepFirstPoint``，只接管文件
    输入输出位置。
    """

    data = pd.read_pickle(input_path)
    data = module.formStrategyToOnehot(data)
    data = module.keepFirstPoint(data, GHOST2_SEGMENT_STRATEGY_NAMES)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_pickle(output_path)
    return int(len(data))


def form_ghost2_data(module: Any, ghost2_dir: Path, formed_dir: Path) -> dict[str, int]:
    """批量运行旧版 formed 数据转换。"""

    file_names = list_pickle_files(ghost2_dir)
    if not file_names:
        raise FileNotFoundError(f"没有在旧版 ghost2 分流目录中找到 pkl：{ghost2_dir}")

    formed_dir.mkdir(parents=True, exist_ok=True)
    row_count = 0
    for file_name in file_names:
        row_count += form_one_file(module, ghost2_dir / file_name, formed_dir / file_name)
    return {"input_ghost2_files": len(file_names), "formed_ghost2_files": len(file_names), "rows": row_count}


def compute_legacy_neighbor_features(subject_data: dict[str, pd.DataFrame], file_names: list[str]) -> np.ndarray:
    """按旧版 ``nearestNeighbors`` 的规则计算被试级策略频数特征。"""

    features = []
    for file_name in file_names:
        data = subject_data[file_name]
        feature = (data[ALL_NEIGHBOR_FEATURE_NAMES] - 1).to_numpy()
        feature = np.sum(feature, axis=0)
        feature = feature / np.sum(feature)
        features.append(deepcopy(feature))

    features = np.array(features)
    feature_min = features.min(axis=0)
    feature_max = features.max(axis=0)
    features = (features - feature_min) / (feature_max - feature_min)

    # 旧版算法对 approach、energizer、stay 三个维度做距离加权。
    features[:, [4, 5, 7]] = 2 * features[:, [4, 5, 7]]
    return features


def find_legacy_neighbors(features: np.ndarray) -> list[list[int]]:
    """按旧版阈值扩张策略寻找每个被试的邻近被试索引。"""

    nearest_model = NearestNeighbors(n_neighbors=len(features))
    nearest_model.fit(features)
    distances, indices = nearest_model.kneighbors(features)

    neighbors: list[list[int]] = []
    for subject_index in range(len(distances)):
        distance_column = distances[subject_index, :].reshape(-1, 1)
        threshold = 1.0
        while True:
            selected = np.where(distance_column <= threshold)[0]
            if len(selected) < 5:
                threshold += 0.1
                continue
            neighbors.append(list(indices[subject_index][selected]))
            break
    return neighbors


def build_legacy_strategy_sequence(subject_data: dict[str, pd.DataFrame], file_names: list[np.str_]) -> dict[str, Any]:
    """按旧版 ``consolidateDataATNearestNb`` 的字段结构生成一个近邻组序列。"""

    data_frames = [deepcopy(subject_data[str(file_name)]) for file_name in file_names]
    data = pd.concat(data_frames, axis=0)

    states = data[STATE_OUTPUT_NAMES]
    strategy = data[STRATEGY_OUTPUT_NAMES]
    strategy_label = data["strategy"].apply(lambda value: STRATEGY_ID_TO_LABEL[int(value)])

    sequence = ""
    for row_index in range(len(data)):
        current_strategy = data[STRATEGY_OUTPUT_NAMES].iloc[row_index]
        strategy_index = int(np.argmax(current_strategy))
        sequence += STRATEGY_SYMBOLS[strategy_index]

    return {
        "seq": sequence,
        "S": STRATEGY_SYMBOLS,
        "state": states,
        "strategy": strategy,
        "strategyLabel": strategy_label,
        "fileNames": file_names,
    }


def consolidate_strategy_sequences(formed_dir: Path, sequence_dir: Path) -> dict[str, int]:
    """按旧版近邻合并算法生成策略序列文件。"""

    file_names = list_pickle_files(formed_dir)
    if not file_names:
        raise FileNotFoundError(f"没有在旧版 formed 目录中找到 pkl：{formed_dir}")

    subject_data = {file_name: pd.read_pickle(formed_dir / file_name) for file_name in file_names}
    features = compute_legacy_neighbor_features(subject_data, file_names)
    neighbors = find_legacy_neighbors(features)

    sequence_dir.mkdir(parents=True, exist_ok=True)
    for neighbor_indices in neighbors:
        neighbor_file_names = list(np.array(file_names)[neighbor_indices])
        result = build_legacy_strategy_sequence(subject_data, neighbor_file_names)
        with (sequence_dir / str(neighbor_file_names[0])).open("wb") as file:
            pickle.dump(result, file)
    return {"formed_input_files": len(file_names), "strategy_sequence_files": len(neighbors)}


def parse_args() -> argparse.Namespace:
    """解析旧版 human fMRI 数据整理临时入口参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--raw-discrete-dir", type=Path, required=True)
    parser.add_argument("--ghost2-discrete-dir", type=Path, required=True)
    parser.add_argument("--ghost4-discrete-dir", type=Path, required=True)
    parser.add_argument("--formed-ghost2-dir", type=Path, required=True)
    parser.add_argument("--strategy-sequence-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """执行旧版 human fMRI 数据整理流程并写出摘要报告。"""

    args = parse_args()
    module = load_legacy_data_formed_module(args.legacy_root)

    split_summary = split_discrete_feature_data(
        raw_dir=args.raw_discrete_dir,
        ghost2_dir=args.ghost2_discrete_dir,
        ghost4_dir=args.ghost4_discrete_dir,
    )
    formed_summary = form_ghost2_data(module, args.ghost2_discrete_dir, args.formed_ghost2_dir)
    sequence_summary = consolidate_strategy_sequences(args.formed_ghost2_dir, args.strategy_sequence_dir)

    report = {
        "legacy_root": str(args.legacy_root),
        "raw_discrete_dir": str(args.raw_discrete_dir),
        "split": split_summary,
        "formed": formed_summary,
        "strategy_sequence": sequence_summary,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
