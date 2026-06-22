"""离散变量结构学习的通用算法。

本模块集中放置与具体业务模型无关的结构学习能力：离散状态计数、
Bayesian Dirichlet 打分、条件变量到效果变量的连线学习，以及 PC skeleton
状态依赖图学习。调用方负责提供已经整理好的内存矩阵，本模块不读取或写入数据文件。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from scipy.special import gammaln


class StructureLearningError(RuntimeError):
    """结构学习输入不满足算法约束时抛出的明确异常。"""


@dataclass(frozen=True)
class StateDependencyGraph:
    """保存每个状态对应的条件状态下标列表。

    输入语义：可由邻接矩阵转换得到，也可由调用方直接构造。
    输出语义：conditions_by_state[i] 表示第 i 个状态依赖的状态下标列表。
    关键约束：下标顺序来自邻接矩阵列顺序，后续评分逻辑依赖该顺序展开条件变量。
    """

    # conditions_by_state[i] 表示邻接矩阵第 i 行中取值为 1 的依赖状态下标。
    conditions_by_state: list[list[int]]

    @classmethod
    def from_adjacency_matrix(cls, adjacency_matrix: np.ndarray) -> "StateDependencyGraph":
        """从邻接矩阵构造状态依赖图条件索引。

        输入语义：adjacency_matrix 是二维邻接矩阵，值等于 1 的位置表示存在依赖关系。
        输出语义：返回 StateDependencyGraph，每一行转成一个条件状态下标列表。
        关键约束：只有精确等于 1 的位置会被视为依赖关系。
        """

        return cls(conditions_by_state=conditions_from_adjacency_matrix(adjacency_matrix))


def conditions_from_adjacency_matrix(adjacency_matrix: np.ndarray) -> list[list[int]]:
    """把邻接矩阵转换为每个节点的条件节点下标列表。

    输入语义：adjacency_matrix 是二维矩阵，行表示目标节点，列表示候选条件节点。
    输出语义：conditions[i] 是第 i 行中取值为 1 的列下标列表。
    关键约束：不解释权重大小，非 1 的位置不会被纳入条件列表。
    """

    graph = np.asarray(adjacency_matrix)
    if graph.ndim != 2:
        raise StructureLearningError("adjacency_matrix 必须是二维矩阵。")

    conditions = []
    for index in range(len(graph)):
        # np.where 返回的下标顺序与矩阵列顺序一致，保持条件变量展开顺序稳定。
        conditions.append(list(np.where(graph[index, :] == 1)[0]))
    return conditions


def count_state_combinations(data: np.ndarray, nstates: np.ndarray) -> np.ndarray:
    """统计多变量离散状态组合在样本中的出现次数。

    输入语义：data 的每一行是一类变量，状态值使用 1-based 编码；nstates 给出各变量状态数。
    输出语义：返回计数向量，第 k 位表示第 k 种 0-based 展平组合出现的次数。
    关键约束：空输入返回空列表；非空输入要求 data 与 nstates 的变量维度一致。
    """

    data = np.asarray(data)
    nstates = np.asarray(nstates)

    # 空输入没有可统计的状态组合，沿用调用方依赖的空列表边界返回值。
    if len(data) == 0 or data.size == 0:
        return []

    counts = np.zeros(np.prod(nstates))
    zero_based_data = data - 1
    combination_index = zero_based_data[0].copy()
    for index in range(1, len(zero_based_data)):
        # 第 index 个变量的偏移量是之前所有变量状态数的乘积，对应列主序展平组合编码。
        multiplier = np.prod(nstates[:index])
        combination_index += zero_based_data[index] * multiplier

    # np.unique 只返回实际出现过的组合；未出现组合保持 0。
    indices, index_counts = np.unique(combination_index, return_counts=True)
    for index in range(len(indices)):
        counts[indices[index]] = index_counts[index]
    return counts


def bd_score(
    data_v: np.ndarray,
    data_parents: np.ndarray,
    nstates_v: int,
    nstates_parents: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray]:
    """计算子变量在给定父变量条件下的 Bayesian Dirichlet 得分。

    输入语义：data_v 是子变量样本，data_parents 是父变量样本矩阵，nstates* 描述离散状态数。
    输出语义：返回总得分和后验参数矩阵，后验矩阵形状为 child 状态数乘 parent 组合数。
    关键约束：alpha 由调用方按变量组合数缩放；data_v 和父变量样本数必须对齐。
    """

    # alpha 在调用侧根据变量状态数完成缩放，这里只负责构造均匀 Dirichlet 先验。
    data_v = np.asarray(data_v)
    data_parents = np.asarray(data_parents)
    nstates_parents = np.asarray(nstates_parents)

    prior = alpha * np.ones((nstates_v, int(np.prod(nstates_parents))))
    if len(data_parents) != 0:
        # 有 parent 时，先把 child 和 parent 垂直拼接，再按所有变量状态数统计组合频次。
        counts = count_state_combinations(
            np.vstack((data_v, data_parents)),
            np.hstack((nstates_v, nstates_parents)),
        )
    else:
        # 无 parent 时需要把子变量整理为单变量样本矩阵，覆盖一维和二维输入形态。
        if len(data_v.shape) == 2:
            sample_count = max(data_v.shape[0], data_v.shape[1])
            counts = count_state_combinations(data_v.reshape(-1, sample_count), np.array([nstates_v]))
        else:
            counts = count_state_combinations(data_v.reshape(-1, len(data_v)), np.array([nstates_v]))

    child_given_parents = counts.reshape(
        np.prod(nstates_v),
        int(np.prod(nstates_parents)),
        order="F",
    )
    posterior = prior + child_given_parents
    # 使用 gammaln 写出 Dirichlet-multinomial 边际似然，逐列累加 parent 条件下的得分。
    score = np.sum(
        gammaln(np.sum(prior, axis=0))
        - gammaln(np.sum(posterior, axis=0))
        + np.sum(gammaln(posterior), axis=0)
        - np.sum(gammaln(prior), axis=0),
        axis=0,
    )
    return score, posterior


def learn_condition_effect_links(
    data: np.ndarray,
    nstates: np.ndarray,
    block_message: dict[int, list[int]],
    casual_num: int,
    block_num: int,
    effect_num: int,
    alpha: float,
    conditions: list[list[int]],
) -> tuple[np.ndarray, list, list, list]:
    """学习条件变量块到效果变量的候选依赖连线。

    输入语义：data 按变量行、样本列排列，conditions 描述每个条件块的前置条件变量下标。
    输出语义：返回学习到的邻接矩阵，以及参数、父节点、得分占位列表。
    关键约束：当前调用路径只消费邻接矩阵；其余三个列表保持形状稳定以便调用侧解包。
    """

    # best* 三个返回值当前仍为空列表，保留返回元组形状便于调用侧解包。
    var_num = data.shape[0]
    bestparents = [[] for _ in range(var_num)]
    bestparameters = [[] for _ in range(var_num)]
    bestscores = [[] for _ in range(var_num)]
    var_casual = list(range(block_num))
    var_effect = list(range(var_num - effect_num, var_num))
    learned_adjacency = np.zeros((block_num + effect_num, block_num + effect_num))

    for v in var_effect:
        for parent_block in var_casual:
            # conditions[parent_block] 给出当前 parent block 需要共同纳入评分的条件状态下标。
            # block_message 将块下标展开为变量下标，支持一个块包含多个状态变量的情况。
            condition = conditions[parent_block]
            condition = [block_message[index] for index in condition]
            condition = sum(condition, [])
            condition_alpha = alpha / (np.prod(nstates[v]) * np.prod(nstates[condition]))
            # bd1 表示只考虑条件状态，不加入候选 parent block 时的得分。
            bd1, _ = bd_score(
                data[v, :],
                data[condition, :],
                nstates[v],
                nstates[condition],
                condition_alpha,
            )

            parent_variables = condition + block_message[parent_block]
            parent_alpha = alpha / (np.prod(nstates[v]) * np.prod(nstates[parent_variables]))
            # bd2 表示条件状态加候选 parent block 后的得分，分数比值用于判定是否连线。
            bd2, _ = bd_score(
                data[v, :],
                data[parent_variables, :],
                nstates[v],
                nstates[parent_variables],
                parent_alpha,
            )
            if bd1 / bd2 > 1:
                # effect 变量位于 data 尾部，写入邻接矩阵时需要转换到输出矩阵中的效果列下标。
                learned_adjacency[parent_block, v - (casual_num - block_num)] = 1

    return learned_adjacency, bestparameters, bestparents, bestscores


def learn_state_condition_links(
    data: np.ndarray,
    nstates: np.ndarray,
    block_message: dict[int, list[int]],
    casual_num: int,
    block_num: int,
    effect_num: int,
    alpha: float,
    conditions: list[list[int]],
) -> tuple[np.ndarray, list, list, list]:
    """兼容旧命名的条件状态到效果变量连线学习接口。

    输入语义：参数与 learn_condition_effect_links 完全一致。
    输出语义：返回同样的邻接矩阵和占位列表。
    关键约束：该函数只做命名兼容，正式新代码应优先调用 learn_condition_effect_links。
    """

    return learn_condition_effect_links(
        data=data,
        nstates=nstates,
        block_message=block_message,
        casual_num=casual_num,
        block_num=block_num,
        effect_num=effect_num,
        alpha=alpha,
        conditions=conditions,
    )


def learn_pc_skeleton(
    state_matrix: np.ndarray,
    alpha: float = 0.5,
    trace_callback: Callable[[dict[str, Any]], None] | None = None,
) -> np.ndarray:
    """使用 PC skeleton 过程学习状态变量之间的无向依赖图。

    输入语义：state_matrix 的形状为 ``(变量数, 样本数)``，所有离散取值必须从 1 开始。
    输出语义：返回对称邻接矩阵，1 表示两个状态变量之间保留依赖边，0 表示无边。
    关键约束：算法只使用离散联合计数，因此样本列顺序不会影响学习结果。
    """

    data = np.asarray(state_matrix)
    if data.ndim != 2:
        raise StructureLearningError("state_matrix 必须是二维矩阵。")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise StructureLearningError("state_matrix 不能包含空的变量维度或样本维度。")
    if np.any(data < 1):
        raise StructureLearningError("state_matrix 的离散取值必须从 1 开始。")

    variable_count = data.shape[0]
    sample_count = data.shape[1]
    nstates = np.max(data, axis=1).T.astype(np.int64)

    adjacency_matrix = np.ones((variable_count, variable_count)) - np.diag([1] * variable_count)
    variables = list(range(variable_count))
    separation_sets: list[list[list[int]]] = [[[] for _ in range(variable_count)] for _ in range(variable_count)]

    _emit_trace(
        trace_callback,
        {
            "event": "start",
            "variable_count": variable_count,
            "sample_count": sample_count,
            "nstates": nstates.astype(int).tolist(),
        },
    )

    for condition_size in range(len(variables)):
        if (_neighbor_sizes(adjacency_matrix, variables) < condition_size).all():
            _emit_trace(trace_callback, {"event": "stop", "condition_size": condition_size})
            break

        _emit_trace(trace_callback, {"event": "level_start", "condition_size": condition_size})
        for x in variables:
            neighbors = _neighbors(adjacency_matrix, x)
            for y in neighbors:
                condition_sets = _choose_condition_sets(list(set(neighbors) - {y}), condition_size)
                if len(condition_sets) == 0:
                    priors = {
                        "Uxgz": alpha / np.prod([nstates[x]]),
                        "Uygz": alpha / np.prod([nstates[y]]),
                        "Uz": alpha / 1,
                        "Uxyz": alpha / (np.prod([nstates[x]]) * np.prod([nstates[y]])),
                    }
                    independent, logpindep, logpdep = _conditional_independence_test(
                        data[x, :].reshape(-1, sample_count),
                        data[y, :].reshape(-1, sample_count),
                        [],
                        [nstates[x]],
                        [nstates[y]],
                        [],
                        0,
                        priors,
                    )
                    _record_independence_test(
                        trace_callback,
                        condition_size,
                        x,
                        y,
                        [],
                        priors,
                        independent,
                        logpindep,
                        logpdep,
                    )
                    if independent:
                        adjacency_matrix[x, y] = adjacency_matrix[y, x] = 0
                        _emit_trace(trace_callback, {"event": "remove_edge", "x": x, "y": y, "z": []})
                else:
                    for z in condition_sets:
                        priors = {
                            "Uxgz": alpha / (np.prod(nstates[x]) * np.prod(nstates[z])),
                            "Uygz": alpha / (np.prod(nstates[y]) * np.prod(nstates[z])),
                            "Uz": alpha / np.prod(nstates[z]),
                            "Uxyz": alpha
                            / (np.prod(nstates[x]) * np.prod(nstates[y]) * np.prod(nstates[z])),
                        }
                        independent, logpindep, logpdep = _conditional_independence_test(
                            data[x, :].reshape(-1, sample_count),
                            data[y, :].reshape(-1, sample_count),
                            data[z, :].reshape(-1, sample_count),
                            nstates[x],
                            nstates[y],
                            nstates[z],
                            0,
                            priors,
                        )
                        _record_independence_test(
                            trace_callback,
                            condition_size,
                            x,
                            y,
                            z,
                            priors,
                            independent,
                            logpindep,
                            logpdep,
                        )
                        if independent:
                            adjacency_matrix[x, y] = adjacency_matrix[y, x] = 0
                            separation_sets[x][y] = list(set(separation_sets[x][y]) | set(z))
                            separation_sets[y][x] = separation_sets[x][y]
                            _emit_trace(trace_callback, {"event": "remove_edge", "x": x, "y": y, "z": list(z)})

    _emit_trace(
        trace_callback,
        {
            "event": "finish",
            "adjacency_matrix": adjacency_matrix.astype(float).tolist(),
            "separation_sets": separation_sets,
        },
    )
    return adjacency_matrix


def learn_state_dependency_graph(
    state_matrix: np.ndarray,
    state_names: Sequence[str] | None = None,
    alpha: float = 0.5,
    trace_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """从离散状态矩阵学习状态依赖图并返回结构化结果。

    输入语义：state_matrix 是 1-based 离散状态矩阵，state_names 可选提供矩阵行名。
    输出语义：返回包含状态名、状态矩阵和邻接矩阵的字典。
    关键约束：该函数只消费内存矩阵，不知道 StrategySequence、pickle 或数据目录。
    """

    adjacency_matrix = learn_pc_skeleton(state_matrix, alpha=alpha, trace_callback=trace_callback)
    return {
        "state_names": list(state_names or []),
        "state_matrix": state_matrix,
        "adjacency_matrix": adjacency_matrix,
    }


def _record_independence_test(
    trace_callback: Callable[[dict[str, Any]], None] | None,
    condition_size: int,
    x: int,
    y: int,
    z: Sequence[int],
    priors: Mapping[str, float],
    independent: bool,
    logpindep: float,
    logpdep: float,
) -> None:
    """记录一次条件独立检验的关键过程信息。

    输入语义：传入变量编号、条件集合、先验参数和检验结果。
    输出语义：如果 trace_callback 存在，则向外发送一个可序列化事件。
    关键约束：记录逻辑不能改变任何学习状态。
    """

    _emit_trace(
        trace_callback,
        {
            "event": "test",
            "condition_size": int(condition_size),
            "x": int(x),
            "y": int(y),
            "z": [int(item) for item in z],
            "priors": {key: float(value) for key, value in priors.items()},
            "independent": bool(independent),
            "logpindep": float(logpindep),
            "logpdep": float(logpdep),
        },
    )


def _conditional_independence_test(
    data_x: np.ndarray,
    data_y: np.ndarray,
    data_z: np.ndarray | list[Any],
    x_states: np.ndarray | Sequence[int] | int,
    y_states: np.ndarray | Sequence[int] | int,
    z_states: np.ndarray | Sequence[int] | int,
    threshold: float,
    priors: Mapping[str, float],
) -> tuple[bool, float, float]:
    """用 Dirichlet-multinomial 边际似然判断 X 和 Y 在 Z 下是否独立。

    输入语义：data_x、data_y 和 data_z 是 1-based 离散观测，states 参数给出各变量取值数。
    输出语义：返回是否独立、独立模型 log 概率和依赖模型 log 概率。
    关键约束：所有联合计数都使用列主序重排，保持离散组合索引的一致性。
    """

    x_states_array = np.asarray(x_states)
    y_states_array = np.asarray(y_states)
    z_states_array = np.asarray(z_states)
    has_z = not _is_empty_observation(data_z)

    if has_z:
        cxz = count_state_combinations(np.vstack((data_x, data_z)), np.hstack((x_states_array, z_states_array)))
    else:
        cxz = count_state_combinations(data_x, x_states_array)
    log_zux = _log_z_dirichlet(priors["Uxgz"] * np.ones(int(np.prod(x_states_array))))
    cx_given_z = np.asarray(cxz).reshape(
        int(np.prod(x_states_array)),
        int(np.prod(z_states_array)),
        order="F",
    )
    logpxgz = np.sum(_log_z_dirichlet(cx_given_z + priors["Uxgz"]) - log_zux)

    if has_z:
        cyz = count_state_combinations(np.vstack((data_y, data_z)), np.hstack((y_states_array, z_states_array)))
    else:
        cyz = count_state_combinations(data_y, y_states_array)
    log_zuy = _log_z_dirichlet(priors["Uygz"] * np.ones(int(np.prod(y_states_array))))
    cy_given_z = np.asarray(cyz).reshape(
        int(np.prod(y_states_array)),
        int(np.prod(z_states_array)),
        order="F",
    )
    logpygz = np.sum(_log_z_dirichlet(cy_given_z + priors["Uygz"]) - log_zuy)

    cz = count_state_combinations(data_z, z_states_array)
    log_zuz = _log_z_dirichlet(priors["Uz"] * np.ones(int(np.prod(z_states_array))))
    if len(cz) != 0:
        logpz = _log_z_dirichlet(np.asarray(cz) + priors["Uz"]) - log_zuz
    else:
        logpz = _log_z_dirichlet(np.array([])) - log_zuz

    if has_z:
        logpindep = logpxgz + logpygz + logpz
    else:
        logpindep = logpxgz + logpygz

    # 依赖模型直接对 X、Y 和条件变量的联合分布打分。
    if has_z:
        cxyz = count_state_combinations(
            np.vstack((data_x, data_y, data_z)),
            np.hstack((x_states_array, y_states_array, z_states_array)),
        )
        joint_state_count = int(np.prod(np.hstack((x_states_array, y_states_array, z_states_array))))
    else:
        cxyz = count_state_combinations(np.vstack((data_x, data_y)), np.hstack((x_states_array, y_states_array)))
        joint_state_count = int(np.prod(np.hstack((x_states_array, y_states_array))))

    log_zuxyz = _log_z_dirichlet(priors["Uxyz"] * np.ones(joint_state_count))
    logpdep = _log_z_dirichlet(np.asarray(cxyz) + priors["Uxyz"]) - log_zuxyz
    log_bayes_factor = logpindep - logpdep
    return bool(log_bayes_factor > threshold), float(logpindep), float(logpdep)


def _log_z_dirichlet(values: np.ndarray) -> np.ndarray | float:
    """计算 Dirichlet 分布归一化常数的 log 形式。

    输入语义：values 是一个或多个 Dirichlet 参数。
    输出语义：返回 ``sum(gammaln(values)) - gammaln(sum(values))``。
    关键约束：axis 固定为 0，使二维计数矩阵按每个条件配置分别计算。
    """

    return np.sum(gammaln(values), axis=0) - gammaln(np.sum(values, axis=0))


def _neighbor_sizes(adjacency_matrix: np.ndarray, variables: Sequence[int]) -> np.ndarray:
    """计算每个变量当前连接到候选变量集合中的邻居数量。

    输入语义：adjacency_matrix 是无向邻接矩阵，variables 是候选变量编号。
    输出语义：返回每个候选变量的当前邻居数。
    关键约束：这个判断决定 PC skeleton 是否还能构造指定大小的条件集合。
    """

    return np.sum(adjacency_matrix[:, variables], axis=0)


def _neighbors(adjacency_matrix: np.ndarray, variable: int) -> list[int]:
    """读取一个变量在当前无向图中的邻居编号。

    输入语义：adjacency_matrix 是当前邻接矩阵，variable 是目标变量编号。
    输出语义：返回与目标变量相连的其它变量编号列表。
    关键约束：同时检查行和列，确保即使矩阵更新过程中出现方向差异也能读取到无向邻居。
    """

    connected = adjacency_matrix[:, variable] + adjacency_matrix.T[:, variable]
    neighbors = np.where(connected > 0)[0].tolist()
    return list(set(neighbors) - {variable})


def _choose_condition_sets(candidates: Sequence[int], condition_size: int | None = None) -> list[list[int]]:
    """生成给定大小的条件变量组合。

    输入语义：candidates 是可作为条件变量的编号列表，condition_size 是组合大小。
    输出语义：返回由变量编号列表组成的组合列表。
    关键约束：当条件大小为 0 或候选数不足时返回空列表，由调用方进入无条件检验分支。
    """

    if condition_size is None or candidates is None or condition_size == 0 or len(candidates) < condition_size:
        return []
    return [list(item) for item in combinations(candidates, condition_size)]


def _is_empty_observation(data: np.ndarray | list[Any]) -> bool:
    """判断观测数据是否为空条件变量。

    输入语义：data 可以是空列表，也可以是 numpy 数组。
    输出语义：返回 True 表示没有条件变量观测。
    关键约束：空列表没有 size 属性，因此需要先处理列表形态。
    """

    if isinstance(data, list) and len(data) == 0:
        return True
    data_array = np.asarray(data)
    return len(data_array) == 0 or data_array.size == 0


def _emit_trace(trace_callback: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
    """向外部 trace 回调发送一个过程事件。

    输入语义：trace_callback 为 None 时表示不记录过程。
    输出语义：存在回调时调用它并传出事件字典。
    关键约束：trace 只用于验证和诊断，不能反向影响算法状态。
    """

    if trace_callback is not None:
        trace_callback(event)
