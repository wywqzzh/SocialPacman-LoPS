"""generate_grammar 标准数据结构、输入读取和输出写入。

本模块集中管理 generate_grammar 正式运行所需的数据边界：读取 StrategySequence，
读取状态依赖图，并把结构化 grammar 结果写回 pickle。这里不实现 grammar 学习算法，
只负责把磁盘数据转换成核心学习流程可以直接消费的内存对象。
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from LoPS.structure_learning import StateDependencyGraph


@dataclass(frozen=True)
class StrategyStateData:
    """保存单个 StrategySequence 输入文件解析后的 token、状态和被试信息。

    输入语义：由 load_strategy_state_data() 从 pickle 中抽取字段后构造。
    输出语义：向 grammar 学习流程提供 token 序列、初始 token、状态特征和被试标识。
    关键约束：state_features 只包含调用方指定的状态列，participant_ids 去掉文件后缀。
    """

    # 一个 StrategySequence pickle 对应一个 StrategyStateData，便于后续按文件独立处理。
    input_file_name: str
    token_sequence: list[str]
    initial_tokens: list[str]
    state_features: pd.DataFrame
    participant_file_names: list[str]
    participant_ids: list[str]


def list_strategy_state_files(strategy_sequence_dir: Path) -> list[str]:
    """列出目录中可作为 StrategySequence 输入的 pickle 文件名。

    输入语义：strategy_sequence_dir 是已存在的数据目录。
    输出语义：返回按文件名排序的 .pkl 文件名列表，不包含目录前缀。
    关键约束：只筛选文件后缀，不读取文件内容；排序用于稳定日志和测试输出。
    """

    # 文件之间互不影响，排序只用于让运行日志和测试结果更稳定。
    return sorted(path.name for path in strategy_sequence_dir.iterdir() if path.suffix == ".pkl")


def load_strategy_state_data(path: Path, state_names: Sequence[str]) -> StrategyStateData:
    """读取一个 StrategySequence pickle 并转换为内部数据对象。

    输入语义：path 指向 pandas pickle，state_names 指定要保留的状态特征列及顺序。
    输出语义：返回 StrategyStateData，包含 token 序列、初始 token、状态特征和被试信息。
    关键约束：pickle 必须包含 seq、S、state、fileNames 字段，状态列缺失会由 pandas 抛错。
    """

    # StrategySequence 文件使用 pandas pickle 存储，读取后只抽取学习流程需要的字段。
    result = pd.read_pickle(path)

    # 源数据 fileNames 包含 .pkl 后缀，因此原始文件名和被试 ID 分开保存。
    participant_file_names = [str(name) for name in result["fileNames"]]
    participant_ids = [Path(name).stem for name in participant_file_names]
    return StrategyStateData(
        input_file_name=path.name,
        token_sequence=list(result["seq"]),
        initial_tokens=list(result["S"]),
        state_features=result["state"][list(state_names)].copy(),
        participant_file_names=participant_file_names,
        participant_ids=participant_ids,
    )


def load_state_dependency_graph(path: Path) -> StateDependencyGraph:
    """读取状态依赖图 pickle 并提取条件状态索引。

    输入语义：path 指向新版本状态依赖图 pickle，验证场景下也可读取旧字段结构。
    输出语义：返回 StateDependencyGraph，其中每行取值为 1 的列被转换为条件状态下标。
    关键约束：正式链路优先读取 adjacency_matrix；G 只作为旧结果验证或历史数据适配入口。
    """

    # 新版本状态依赖图使用清晰字段名，供正式 generate_grammar 链路直接消费。
    result = pd.read_pickle(path)
    if "adjacency_matrix" in result:
        graph = result["adjacency_matrix"]
    elif "G" in result:
        # 旧字段只保留为验证适配能力，避免旧格式反向污染正式输出结构。
        graph = result["G"]
    else:
        raise KeyError(f"{path} 中缺少 adjacency_matrix 或 G 字段，无法读取状态依赖图。")
    return StateDependencyGraph.from_adjacency_matrix(graph)


def write_generate_grammar_output(output: Mapping[str, Any], path: Path) -> None:
    """将结构化 grammar 输出写入 pickle 文件。

    输入语义：output 是可转为 dict 的结构化结果，path 是目标 pickle 文件路径。
    输出语义：成功时创建父目录并写入二进制 pickle，函数返回 None。
    关键约束：该函数不解释输出内容，只负责持久化调用方传入的数据结构。
    """

    # 输出统一由 LoPS pipeline 管理；这里仅确保目录存在并执行 pickle 写入。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(dict(output), file)
