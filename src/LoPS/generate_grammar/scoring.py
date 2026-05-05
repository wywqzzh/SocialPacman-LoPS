from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def count_state_combinations(data: np.ndarray, nstates: np.ndarray) -> np.ndarray:
    # 该函数复刻旧 src.Utils.count：输入状态值使用 1-based 编码，组合索引用 0-based 计算。
    # 返回向量的第 k 位表示第 k 种状态组合在样本中出现的次数。
    data = np.asarray(data)
    nstates = np.asarray(nstates)

    # 旧函数对空输入返回 [] 而不是空 ndarray；这里保留该边界行为用于旧新精确对照。
    if len(data) == 0 or data.size == 0:
        return []

    counts = np.zeros(np.prod(nstates))
    zero_based_data = data - 1
    combination_index = zero_based_data[0].copy()
    for index in range(1, len(zero_based_data)):
        # 第 index 个变量的偏移量是之前所有变量状态数的乘积，和旧实现 np.prod(nstates[:i]) 一致。
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
    # 复刻旧 bayesianScore.BDscore：alpha 在调用侧已按旧逻辑决定，这里只负责构造 Dirichlet 先验。
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
        # 无 parent 分支保留旧代码对一维/二维 data_v 的 reshape 方式，避免微小形状差异影响结果。
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
    # gammaln 公式是旧 BDscore 的核心打分，必须保持逐项顺序一致以支持精确行为测试。
    score = np.sum(
        gammaln(np.sum(prior, axis=0))
        - gammaln(np.sum(posterior, axis=0))
        + np.sum(gammaln(posterior), axis=0)
        - np.sum(gammaln(prior), axis=0),
        axis=0,
    )
    return score, posterior


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
    # 该函数只迁移旧 learnBayesNetBlock 默认路径用到的状态条件连线学习。
    # best* 三个返回值旧函数虽然基本为空，但 legacy 行为测试依赖返回元组形状，因此保留。
    var_num = data.shape[0]
    bestparents = [[] for _ in range(var_num)]
    bestparameters = [[] for _ in range(var_num)]
    bestscores = [[] for _ in range(var_num)]
    var_casual = list(range(block_num))
    var_effect = list(range(var_num - effect_num, var_num))
    learned_adjacency = np.zeros((block_num + effect_num, block_num + effect_num))

    for v in var_effect:
        for parent_block in var_casual:
            # conditions[parent_block] 给出旧 StateGraph 约束的条件状态下标；
            # block_message 当前默认是 {i: [i]}，但仍按旧代码展开，保证结构兼容。
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
            # bd2 表示条件状态加候选 parent block 后的得分；旧代码用 bd1 / bd2 > 1 判定连线。
            bd2, _ = bd_score(
                data[v, :],
                data[parent_variables, :],
                nstates[v],
                nstates[parent_variables],
                parent_alpha,
            )
            if bd1 / bd2 > 1:
                # 输出矩阵下标保留旧 v - (casual_num - block_num) 的偏移方式。
                learned_adjacency[parent_block, v - (casual_num - block_num)] = 1

    return learned_adjacency, bestparameters, bestparents, bestscores
