from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StateDependencyGraph:
    conditions_by_state: list[list[int]]


def load_state_dependency_graph(path: Path) -> StateDependencyGraph:
    result = pd.read_pickle(path)
    graph = result["G"]
    conditions = []
    for index in range(len(graph)):
        conditions.append(list(np.where(graph[index, :] == 1)[0]))
    return StateDependencyGraph(conditions_by_state=conditions)
