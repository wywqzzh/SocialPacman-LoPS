from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class StrategyStateData:
    # 一个 StrategySequence pickle 对应一个 StrategyStateData。
    # participant_file_names 保留旧 fileNames 原值，用于 legacy 输出逐值一致；
    # participant_ids 则去掉 .pkl 后缀，供 structured 输出使用。
    input_file_name: str
    token_sequence: list[str]
    initial_tokens: list[str]
    state_features: pd.DataFrame
    participant_file_names: list[str]
    participant_ids: list[str]


def list_strategy_state_files(strategy_sequence_dir: Path) -> list[str]:
    # 旧脚本使用 os.listdir，文件之间互不影响；这里排序只让运行日志和测试更稳定。
    return sorted(path.name for path in strategy_sequence_dir.iterdir() if path.suffix == ".pkl")


def load_strategy_state_data(path: Path, state_names: Sequence[str]) -> StrategyStateData:
    # 旧 StrategySequence pickle 是 pandas pickle；读取后只抽取默认 ghost2 分支需要的字段。
    result = pd.read_pickle(path)

    # 旧输出 fileNames 包含 .pkl 后缀，legacy 对比必须保留这一点。
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
    # 输出统一由 LoPS pipeline 管理；调用方传入的 output 是 legacy/structured 双字典。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(dict(output), file)
