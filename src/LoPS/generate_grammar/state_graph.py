from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StateDependencyGraph:
    # conditions_by_state[i] 表示旧状态图 G 的第 i 行中取值为 1 的依赖状态下标。
    # 这等价于旧 getConditionGraph() 的返回结构，供 learn_state_condition_links 使用。
    conditions_by_state: list[list[int]]


def load_state_dependency_graph(path: Path) -> StateDependencyGraph:
    # StateGraph pickle 中的 G 是旧脚本用于约束状态条件的邻接矩阵。
    result = pd.read_pickle(path)
    graph = result["G"]
    conditions = []
    for index in range(len(graph)):
        # 保留 np.where 的旧语义：矩阵行中值为 1 的列才作为该状态的条件。
        conditions.append(list(np.where(graph[index, :] == 1)[0]))
    return StateDependencyGraph(conditions_by_state=conditions)
