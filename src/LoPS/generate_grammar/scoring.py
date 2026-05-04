from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def count_state_combinations(data: np.ndarray, nstates: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    nstates = np.asarray(nstates)

    if len(data) == 0 or data.size == 0:
        return []

    counts = np.zeros(np.prod(nstates))
    zero_based_data = data - 1
    combination_index = zero_based_data[0].copy()
    for index in range(1, len(zero_based_data)):
        multiplier = np.prod(nstates[:index])
        combination_index += zero_based_data[index] * multiplier

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
    data_v = np.asarray(data_v)
    data_parents = np.asarray(data_parents)
    nstates_parents = np.asarray(nstates_parents)

    prior = alpha * np.ones((nstates_v, int(np.prod(nstates_parents))))
    if len(data_parents) != 0:
        counts = count_state_combinations(
            np.vstack((data_v, data_parents)),
            np.hstack((nstates_v, nstates_parents)),
        )
    else:
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
    var_num = data.shape[0]
    bestparents = [[] for _ in range(var_num)]
    bestparameters = [[] for _ in range(var_num)]
    bestscores = [[] for _ in range(var_num)]
    var_casual = list(range(block_num))
    var_effect = list(range(var_num - effect_num, var_num))
    learned_adjacency = np.zeros((block_num + effect_num, block_num + effect_num))

    for v in var_effect:
        for parent_block in var_casual:
            condition = conditions[parent_block]
            condition = [block_message[index] for index in condition]
            condition = sum(condition, [])
            condition_alpha = alpha / (np.prod(nstates[v]) * np.prod(nstates[condition]))
            bd1, _ = bd_score(
                data[v, :],
                data[condition, :],
                nstates[v],
                nstates[condition],
                condition_alpha,
            )

            parent_variables = condition + block_message[parent_block]
            parent_alpha = alpha / (np.prod(nstates[v]) * np.prod(nstates[parent_variables]))
            bd2, _ = bd_score(
                data[v, :],
                data[parent_variables, :],
                nstates[v],
                nstates[parent_variables],
                parent_alpha,
            )
            if bd1 / bd2 > 1:
                learned_adjacency[parent_block, v - (casual_num - block_num)] = 1

    return learned_adjacency, bestparameters, bestparents, bestscores
