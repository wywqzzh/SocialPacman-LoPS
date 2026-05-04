from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class StrategyStateData:
    input_file_name: str
    token_sequence: list[str]
    initial_tokens: list[str]
    state_features: pd.DataFrame
    participant_file_names: list[str]
    participant_ids: list[str]


def list_strategy_state_files(strategy_sequence_dir: Path) -> list[str]:
    return sorted(path.name for path in strategy_sequence_dir.iterdir() if path.suffix == ".pkl")


def load_strategy_state_data(path: Path, state_names: Sequence[str]) -> StrategyStateData:
    result = pd.read_pickle(path)
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


def write_generate_grammar_output(output: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(dict(output), file)
