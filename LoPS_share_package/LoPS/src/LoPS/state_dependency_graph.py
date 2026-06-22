"""Pacman StrategySequence 到状态依赖图的业务边界。

本模块只负责 Pacman 数据字段到结构学习输入矩阵的转换、结果保存和目录批处理。
PC skeleton、条件独立检验和离散结构学习算法集中在 ``LoPS.structure_learning`` 中。
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from LoPS.structure_learning import (
    StructureLearningError,
    learn_pc_skeleton,
    learn_state_dependency_graph as learn_state_dependency_graph_from_matrix,
)


DEFAULT_STATE_NAMES = ("IS1", "IS2", "PG1", "PG2", "PE", "BW10")


class StateDependencyGraphError(StructureLearningError):
    """状态依赖图学习无法继续时抛出的明确异常。"""


def build_state_matrix(strategy_sequence: Mapping[str, Any], state_names: Sequence[str]) -> np.ndarray:
    """从一个被试的 strategy_sequence 中构造 PC 学习使用的状态矩阵。

    输入语义：strategy_sequence 必须包含 ``state`` 字段，且该字段是按帧排列的 DataFrame。
    输出语义：返回形状为 ``(状态变量数, 样本帧数)`` 的离散矩阵。
    关键约束：PC 计数过程要求离散取值从 1 开始，因此这里会对状态表原始取值整体加 1。
    """

    if "state" not in strategy_sequence:
        raise StateDependencyGraphError("strategy_sequence 缺少 'state' 字段。")

    states = strategy_sequence["state"]
    if not isinstance(states, pd.DataFrame):
        raise StateDependencyGraphError("strategy_sequence['state'] 必须是 pandas.DataFrame。")

    missing = [name for name in state_names if name not in states.columns]
    if missing:
        raise StateDependencyGraphError(f"状态表缺少列：{missing}")

    # 保留 DataFrame 原始 dtype；离散计数只要求数值正确，不需要额外转换类型。
    state_matrix = states[list(state_names)].values.T + 1
    if state_matrix.ndim != 2 or state_matrix.shape[1] == 0:
        raise StateDependencyGraphError("状态矩阵必须是二维矩阵，且至少包含一个样本帧。")
    return state_matrix


def learn_state_dependency_graph(
    strategy_sequence: Mapping[str, Any],
    state_names: Sequence[str] = DEFAULT_STATE_NAMES,
    alpha: float = 0.5,
    trace_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """学习单个被试的状态依赖图并返回清晰字段结构。

    输入语义：strategy_sequence 是一个被试的策略序列数据字典。
    输出语义：返回包含状态名、状态矩阵和邻接矩阵的字典。
    关键约束：本函数只做 Pacman 输入字段适配，实际结构学习由 structure_learning 完成。
    """

    state_matrix = build_state_matrix(strategy_sequence, state_names)
    return learn_state_dependency_graph_from_matrix(
        state_matrix,
        state_names=state_names,
        alpha=alpha,
        trace_callback=trace_callback,
    )


def save_state_dependency_graph(result: Mapping[str, Any], output_path: Path | str) -> None:
    """把一个状态依赖图结果保存到 pickle 文件。

    输入语义：result 是 ``learn_state_dependency_graph`` 返回的新结构字典。
    输出语义：在 output_path 写入 pickle 文件。
    关键约束：保存前会自动创建父目录，避免调用脚本重复处理目录初始化。
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(dict(result), file)


def convert_to_legacy_state_graph(result: Mapping[str, Any]) -> dict[str, Any]:
    """把新状态依赖图结构转换为旧字段结构。

    输入语义：result 使用 ``state_names``、``state_matrix`` 和 ``adjacency_matrix`` 字段。
    输出语义：返回 ``G``、``stateNames`` 和 ``data`` 字段，用于一致性验证或外部兼容适配。
    关键约束：这个转换不参与正式学习流程，只在边界处提供格式桥接。
    """

    return {
        "G": result["adjacency_matrix"],
        "stateNames": list(result["state_names"]),
        "data": result["state_matrix"],
    }


def process_state_dependency_graph_file(
    input_path: Path | str,
    output_path: Path | str,
    state_names: Sequence[str] = DEFAULT_STATE_NAMES,
    alpha: float = 0.5,
) -> dict[str, Any]:
    """处理一个被试的 strategy_sequence 文件并保存状态依赖图。

    输入语义：input_path 指向一个 strategy_sequence pickle 文件。
    输出语义：output_path 会写入新结构状态依赖图，返回本次处理的摘要信息。
    关键约束：函数只处理调用方显式传入的路径，不推断任何数据目录。
    """

    input_file = Path(input_path)
    output_file = Path(output_path)
    strategy_sequence = pd.read_pickle(input_file)
    result = learn_state_dependency_graph(strategy_sequence, state_names=state_names, alpha=alpha)
    save_state_dependency_graph(result, output_file)
    return {
        "input": str(input_file),
        "output": str(output_file),
        "sample_count": int(result["state_matrix"].shape[1]),
        "edge_count": int(np.sum(result["adjacency_matrix"]) // 2),
    }


def process_state_dependency_graph_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    state_names: Sequence[str] = DEFAULT_STATE_NAMES,
    alpha: float = 0.5,
) -> list[dict[str, Any]]:
    """批量处理目录下所有 strategy_sequence pickle 文件。

    输入语义：input_dir 下每个 ``.pkl`` 文件代表一个被试。
    输出语义：output_dir 下生成同名状态依赖图 pickle 文件，并返回每个文件的处理摘要。
    关键约束：排序只用于稳定运行日志，单个被试的学习结果不依赖文件处理顺序。
    """

    source_dir = Path(input_dir)
    target_dir = Path(output_dir)
    if not source_dir.exists():
        raise StateDependencyGraphError(f"找不到输入目录：{source_dir}")

    input_files = sorted(source_dir.glob("*.pkl"))
    if not input_files:
        raise StateDependencyGraphError(f"输入目录下没有 .pkl 文件：{source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for input_file in input_files:
        output_file = target_dir / input_file.name
        summaries.append(
            process_state_dependency_graph_file(
                input_file,
                output_file,
                state_names=state_names,
                alpha=alpha,
            )
        )
    return summaries
