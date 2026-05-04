from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from LoPS.generate_grammar.config import GrammarLearningParams
from LoPS.generate_grammar.scoring import learn_state_condition_links
from LoPS.generate_grammar.state_graph import StateDependencyGraph
from LoPS.generate_grammar.token import split_token, token_length


@dataclass
class OrganizedGrammarData:
    data_child: pd.DataFrame
    data_parent: pd.DataFrame
    data_condition: pd.DataFrame
    condition_state: list[list[str]]


@dataclass
class GrammarLearningResult:
    grammar_tokens: list[str]
    probabilities: list[float]
    position_grammar: list[str]
    original_sequence: list[str]
    time_probabilities: np.ndarray
    frequencies: list[int]
    parsed_sequence: list[str]
    parsed_state_features: pd.DataFrame
    active_tokens: list[str]
    participant_file_names: list[str]
    participant_ids: list[str]
    components: list[list[str]]


@dataclass
class SkipGramResult:
    found: bool
    count: int | float


def static_probability(tokens: Sequence[str], active_tokens: Sequence[str]) -> list[float]:
    counts = {}
    for active_token in active_tokens:
        counts.update({active_token: 0})
    for token in tokens:
        counts[token] += 1
    total = np.sum(list(counts.values()))
    return list(np.array(list(counts.values())) / total)


def choose_candidate_chunks(
    ratios: list[float],
    chunks: list[str],
    components: list[list[str]],
    keep_ratio: float,
) -> tuple[list[str], list[float], list[list[str]]]:
    ordered_indices = sorted(range(len(ratios)), key=lambda index: ratios[index], reverse=True)
    if len(ordered_indices) == 0:
        return [], [], []

    ordered = [
        (chunks[index], ratios[index], components[index])
        for index in ordered_indices
        if ratios[index] > 1
    ]
    if len(ordered) == 0:
        return [], [], []

    best_ratio = ordered[0][1]
    selected = [ordered[0]]
    for candidate in ordered[1:]:
        if candidate[1] / best_ratio > keep_ratio:
            selected.append(candidate)
        else:
            break

    selected_chunks, selected_ratios, selected_components = zip(*selected)
    return list(selected_chunks), list(selected_ratios), list(selected_components)


def kl_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    value = 0
    for key in p.keys():
        probability = p[key]
        if key in q:
            reference_probability = q[key]
        else:
            reference_probability = 0.00001
        value += probability * math.log2(probability / reference_probability)
    return value


class GrammarLearner:
    def __init__(self, params: GrammarLearningParams):
        self.params = params

    def _parse_longest(
        self,
        tokens: list[str],
        grammar_tokens: list[str],
        state_features: pd.DataFrame | None = None,
    ) -> tuple[list[str], pd.DataFrame | None]:
        parsed_tokens = []
        parsed_state_rows = []
        pointer = 0
        while pointer < len(tokens):
            matched_index = 0
            matched_length = 0
            for index, grammar_token in enumerate(grammar_tokens):
                length = token_length(grammar_token)
                if tokens[pointer:pointer + length] == split_token(grammar_token) and length > matched_length:
                    matched_length = length
                    matched_index = index
            if matched_length == 0:
                raise ValueError(f"No grammar token matches sequence position {pointer}: {tokens[pointer:]}")

            parsed_tokens.append(grammar_tokens[matched_index])
            if state_features is not None:
                parsed_state_rows.append(list(state_features.iloc[pointer]))
            pointer += matched_length

        if state_features is None:
            return parsed_tokens, None
        parsed_state_features = pd.DataFrame(parsed_state_rows, columns=state_features.columns)
        return parsed_tokens, parsed_state_features

    def _parse_probabilities(
        self,
        tokens: list[str],
        grammar_tokens: list[str],
    ) -> tuple[list[str], list[float], list[str], list[int]]:
        cover_indices = []
        pointer = 0
        position_grammar = []
        last_grammar_length = token_length(grammar_tokens[-1])

        while pointer < len(tokens):
            matched_index = 0
            matched_length = 0
            for index, grammar_token in enumerate(grammar_tokens):
                length = token_length(grammar_token)
                if tokens[pointer:pointer + length] == split_token(grammar_token) and length > matched_length:
                    matched_length = length
                    matched_index = index
            if matched_length == 0:
                raise ValueError(f"No grammar token matches sequence position {pointer}: {tokens[pointer:]}")

            cover_indices.append(matched_index)
            pointer += matched_length
            position_grammar += [grammar_tokens[matched_index]] * last_grammar_length

        frequencies_by_token = {}
        for grammar_token in grammar_tokens:
            frequencies_by_token.update({grammar_token: 0})
        for index in cover_indices:
            frequencies_by_token[grammar_tokens[index]] += 1

        frequencies = np.array(list(frequencies_by_token.values()))
        probabilities = frequencies / np.sum(frequencies)
        return list(grammar_tokens), list(probabilities), position_grammar, list(frequencies)

    def _organize_discrete_data(
        self,
        tokens: list[str],
        active_tokens: list[str],
        state_features: pd.DataFrame,
        state_dependencies: StateDependencyGraph,
    ) -> OrganizedGrammarData:
        state_features = state_features.reset_index(drop=True)
        data_parent = {}
        data_child = {}
        for token in active_tokens:
            data_parent.update({token: np.ones(len(tokens) - 1)})
            data_child.update({token: np.ones(len(tokens) - 1)})

        data_condition = {}
        data_policy_condition = {}
        for state_name in state_features.columns:
            data_condition.update({state_name: np.ones(len(tokens) - 1)})
            data_policy_condition.update({state_name: np.ones(len(tokens) - 1)})

        for index in range(1, len(tokens)):
            data_parent[tokens[index - 1]][index - 1] = 2
            data_child[tokens[index]][index - 1] = 2
            for state_name in state_features.columns:
                data_condition[state_name][index - 1] = state_features[state_name].iloc[index] + 1
                data_policy_condition[state_name][index - 1] = state_features[state_name].iloc[index - 1] + 1

        data_parent_frame = pd.DataFrame(data_parent, dtype=int)
        data_child_frame = pd.DataFrame(data_child, dtype=int)
        data_condition_frame = pd.DataFrame(data_condition, dtype=int)
        data_policy_condition_frame = pd.DataFrame(data_policy_condition, dtype=int)

        data = pd.concat([data_policy_condition_frame, data_parent_frame], axis=1).values.T
        data = np.array(data, dtype=int)
        nstates = np.max(data, axis=1).T
        nstates = np.array(nstates, dtype=int)
        casual_num = data_policy_condition_frame.shape[1]
        effect_num = data_parent_frame.shape[1]
        block_message = {index: [index] for index in range(casual_num)}

        learned_adjacency, _, _, _ = learn_state_condition_links(
            data=data,
            nstates=nstates,
            block_message=block_message,
            casual_num=casual_num,
            block_num=len(block_message),
            effect_num=effect_num,
            alpha=self.params.condition_alpha,
            conditions=state_dependencies.conditions_by_state,
        )
        condition_state = []
        names = np.array(list(data_condition_frame.columns))
        for index in range(casual_num, casual_num + effect_num):
            condition_indices = np.where(learned_adjacency[:, index] == 1)[0]
            condition_state.append(list(names[condition_indices]))

        return OrganizedGrammarData(
            data_child=data_child_frame,
            data_parent=data_parent_frame,
            data_condition=data_condition_frame,
            condition_state=condition_state,
        )
