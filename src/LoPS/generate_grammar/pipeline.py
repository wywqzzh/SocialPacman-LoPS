from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from LoPS.generate_grammar.config import GenerateGrammarConfig
from LoPS.generate_grammar.data_io import (
    StrategyStateData,
    list_strategy_state_files,
    load_strategy_state_data,
    write_generate_grammar_output,
)
from LoPS.generate_grammar.grammar import GrammarLearner
from LoPS.generate_grammar.legacy import build_legacy_output
from LoPS.generate_grammar.state_graph import StateDependencyGraph, load_state_dependency_graph
from LoPS.generate_grammar.structured import build_structured_output


@dataclass
class PreparedStrategyStateData:
    input_file_name: str
    token_sequence: list[str]
    n_positions: np.ndarray
    initial_tokens: list[str]
    state_features: pd.DataFrame
    participant_file_names: list[str]
    participant_ids: list[str]
    state_dependencies: StateDependencyGraph


def prepare_strategy_state_data(
    data: StrategyStateData,
    state_dependencies: StateDependencyGraph,
    removed_token: str = "N",
) -> PreparedStrategyStateData:
    token_array = np.array(data.token_sequence)
    n_positions = np.where(token_array == removed_token)[0]
    token_sequence = [token for token in data.token_sequence if token != removed_token]
    state_features = data.state_features.reset_index(drop=True)
    state_features = state_features.drop(n_positions).reset_index(drop=True)
    return PreparedStrategyStateData(
        input_file_name=data.input_file_name,
        token_sequence=token_sequence,
        n_positions=n_positions,
        initial_tokens=list(data.initial_tokens),
        state_features=state_features,
        participant_file_names=list(data.participant_file_names),
        participant_ids=list(data.participant_ids),
        state_dependencies=state_dependencies,
    )


def process_strategy_state_file(input_file_name: str, config: GenerateGrammarConfig) -> dict[str, Any]:
    strategy_state_data = load_strategy_state_data(
        config.strategy_sequence_dir / input_file_name,
        config.learning.state_names,
    )
    state_dependencies = load_state_dependency_graph(config.state_graph_dir / input_file_name)
    prepared = prepare_strategy_state_data(
        strategy_state_data,
        state_dependencies,
        removed_token=config.learning.removed_token,
    )

    learner = GrammarLearner(config.learning)
    grammar_result = learner.learn(
        token_sequence=prepared.token_sequence,
        initial_tokens=prepared.initial_tokens,
        state_features=prepared.state_features,
        state_dependencies=prepared.state_dependencies,
        participant_file_names=prepared.participant_file_names,
        participant_ids=prepared.participant_ids,
    )
    skip_gram = learner.detect_skip_gram(grammar_result, prepared.n_positions)
    legacy_output = build_legacy_output(grammar_result, skip_gram)
    structured_output = build_structured_output(input_file_name, config.learning, grammar_result, skip_gram)

    return {
        "legacy": legacy_output,
        "structured": structured_output,
    }


def run_generate_grammar(config: GenerateGrammarConfig) -> list[Path]:
    config.validate()
    output_paths = []
    for input_file_name in list_strategy_state_files(config.strategy_sequence_dir):
        output = process_strategy_state_file(input_file_name, config)
        output_path = config.output_dir / input_file_name
        write_generate_grammar_output(output, output_path)
        output_paths.append(output_path)
    return output_paths
