"""generate_grammar 文件级流水线。

本模块负责把单个或多个输入文件组织为核心学习所需的数据，并输出新版本结构化结果。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from LoPS.generate_grammar.config import GenerateGrammarConfig
from LoPS.generate_grammar.config import GrammarLearningParams
from LoPS.generate_grammar.data import (
    StateDependencyGraph,
    StrategyStateData,
    list_strategy_state_files,
    load_state_dependency_graph,
    load_strategy_state_data,
    write_generate_grammar_output,
)
from LoPS.generate_grammar.grammar import GrammarLearner, GrammarLearningResult, SkipGramResult
from LoPS.generate_grammar.token import split_token


ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass
class PreparedStrategyStateData:
    """保存进入 GrammarLearner 前的单文件数据。

    输入语义：由 StrategyStateData 和对应 StateDependencyGraph 预处理得到。
    输出语义：token_sequence 已删除配置指定的 removed_token；n_positions 保存被删除 token 的原始位置；
    state_features 已同步删除对应行并重新编号；其余字段保留输入文件和参与者元数据。
    关键约束：state_features 必须与 token_sequence 等长且逐行对齐，否则 grammar 学习的状态条件会错位。
    """

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
    """清理单个策略状态数据并构造学习器输入。

    输入语义：data 包含原始 token 序列、状态特征和参与者信息；state_dependencies 是同名状态依赖图；
    removed_token 指定需要从学习序列中临时移除的 token。
    输出语义：返回 PreparedStrategyStateData，其中 token 序列和状态特征已经按 removed_token 删除结果重新对齐。
    关键约束：被删除 token 的原始位置必须保留，用于学习结束后的 skip-gram 检测。
    """

    # 在 grammar 学习前删除所有 removed_token，并保存其原始位置供 skip_gram 使用。
    token_array = np.array(data.token_sequence)
    n_positions = np.where(token_array == removed_token)[0]
    token_sequence = [token for token in data.token_sequence if token != removed_token]
    # state_features 与原始 seq 等长；删除 N 后必须同步删除对应状态行，否则状态与 token 会错位。
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


def build_structured_output(
    input_file_name: str,
    params: GrammarLearningParams,
    result: GrammarLearningResult,
    skip_gram: SkipGramResult,
) -> dict[str, Any]:
    """组装单个输入文件对应的结构化 grammar 输出。

    输入语义：input_file_name 标记来源文件，params 是学习配置，result 与 skip_gram 是学习产物。
    输出语义：返回可 pickle 的 dict，包含来源、参数、grammar 条目、解析序列和 skip-gram 摘要。
    关键约束：grammar 列表按 result.grammar_tokens 顺序展开，各概率和频率字段必须同长度对齐。
    """

    # 输出目标是给后续科研分析提供清晰、去冗余的新结构。
    grammar_items = []
    for index, token in enumerate(result.grammar_tokens):
        grammar_items.append(
            {
                # token 保留新核心表示；base_tokens 明确展开基础动作，避免后续再解析字符串。
                "token": token,
                "base_tokens": split_token(token),
                "probability": result.probabilities[index],
                "frequency": result.frequencies[index],
                "time_probability": result.time_probabilities[index],
                "components": result.components[index],
            }
        )

    return {
        "source": {
            # source 记录文件来源和被试信息，同时保留原始文件名和去后缀后的被试 ID。
            "input_file_name": input_file_name,
            "participant_file_names": result.participant_file_names,
            "participant_ids": result.participant_ids,
        },
        # 参数完整展开，保证同一份 structured 输出可以追溯当时的学习阈值和状态列。
        "parameters": asdict(params),
        "grammar": grammar_items,
        "parsed": {
            # parsed 保存最终解析序列和对齐状态；旧格式 gram 字段由验证适配器从 sequence 重建。
            "original_sequence": result.original_sequence,
            "sequence": result.parsed_sequence,
            "state_features": result.parsed_state_features,
        },
        "skip_gram": {
            # skip_gram 字段使用新目标 token 名称，例如 "E-A"。
            "target": params.skip_gram_target,
            "found": skip_gram.found,
            "count": skip_gram.count,
        },
    }


def process_strategy_state_file(
    input_file_name: str,
    config: GenerateGrammarConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """处理单个 StrategySequence 文件并返回结构化输出对象。

    输入语义：input_file_name 是输入目录下的文件名；config 提供输入目录、状态图目录和学习参数。
    progress_callback 可选，用于把文件准备、学习迭代和 skip-gram 检测过程报告给运行脚本。
    输出语义：返回可直接写入磁盘的结构化字典，不在本函数内执行文件写出。
    关键约束：StrategySequence 文件和 StateGraph 文件必须同名，且状态列名由 config.learning.state_names 指定。
    """

    def emit_progress(event: str, payload: Mapping[str, object]) -> None:
        """给单文件过程事件补充文件名后转发给外部回调。"""

        if progress_callback is None:
            return
        enriched_payload = {"input_file_name": input_file_name}
        enriched_payload.update(dict(payload))
        progress_callback(event, enriched_payload)

    # 单文件处理函数只返回内存结果，不写文件；这样测试和验证脚本都可以复用同一流程。
    emit_progress("file_start", {})
    strategy_state_data = load_strategy_state_data(
        config.strategy_sequence_dir / input_file_name,
        config.learning.state_names,
    )
    emit_progress(
        "file_loaded",
        {
            "raw_token_count": len(strategy_state_data.token_sequence),
            "state_row_count": len(strategy_state_data.state_features),
            "participant_count": len(strategy_state_data.participant_file_names),
        },
    )
    # StateGraph 文件名与 StrategySequence 文件名一一对应。
    state_dependencies = load_state_dependency_graph(config.state_graph_dir / input_file_name)
    prepared = prepare_strategy_state_data(
        strategy_state_data,
        state_dependencies,
        removed_token=config.learning.removed_token,
    )
    emit_progress(
        "file_prepared",
        {
            "token_count": len(prepared.token_sequence),
            "removed_token_count": len(prepared.n_positions),
            "initial_token_count": len(prepared.initial_tokens),
            "state_row_count": len(prepared.state_features),
            "state_count": len(prepared.state_features.columns),
        },
    )

    # GrammarLearner 接收显式参数和内存数据，不知道输入输出目录。
    learner = GrammarLearner(config.learning)
    grammar_result = learner.learn(
        token_sequence=prepared.token_sequence,
        initial_tokens=prepared.initial_tokens,
        state_features=prepared.state_features,
        state_dependencies=prepared.state_dependencies,
        participant_file_names=prepared.participant_file_names,
        participant_ids=prepared.participant_ids,
        progress_callback=emit_progress,
    )
    # skip_gram 必须在 grammar 学习完成后执行，因为它依赖最终 parsed_sequence。
    skip_gram = learner.detect_skip_gram(grammar_result, prepared.n_positions, progress_callback=emit_progress)
    emit_progress(
        "file_finished",
        {
            "grammar_token_count": len(grammar_result.grammar_tokens),
            "parsed_length": len(grammar_result.parsed_sequence),
            "skip_gram_found": skip_gram.found,
        },
    )

    # 核心 pipeline 只返回当前模块定义的结构化结果；验证适配逻辑不进入正式流程。
    return build_structured_output(input_file_name, config.learning, grammar_result, skip_gram)


def run_generate_grammar(
    config: GenerateGrammarConfig,
    progress_callback: ProgressCallback | None = None,
) -> list[Path]:
    """批量运行 generate_grammar 流程并写出结果文件。

    输入语义：config 提供输入、状态图、输出目录和学习参数；progress_callback 可选，用于报告批量进度。
    输出语义：返回本轮写出的输出文件路径列表，顺序与排序后的输入文件一致。
    关键约束：运行前会校验配置路径；每个输入文件独立处理并写入 config.output_dir 下的同名文件。
    """

    def emit_progress(event: str, payload: Mapping[str, object]) -> None:
        """转发批量运行过程事件，未设置回调时保持静默。"""

        if progress_callback is None:
            return
        progress_callback(event, payload)

    # 全量运行入口：校验路径、排序枚举输入文件、逐个写入 LoPS 输出目录。
    config.validate()
    input_file_names = list_strategy_state_files(config.strategy_sequence_dir)
    emit_progress(
        "run_start",
        {
            "file_count": len(input_file_names),
            "output_dir": str(config.output_dir),
        },
    )
    output_paths = []
    for file_index, input_file_name in enumerate(input_file_names, start=1):
        def emit_file_progress(event: str, payload: Mapping[str, object]) -> None:
            """给单文件事件补充批量序号后转发。"""

            enriched_payload = {
                "file_index": file_index,
                "file_count": len(input_file_names),
            }
            enriched_payload.update(dict(payload))
            emit_progress(event, enriched_payload)

        output = process_strategy_state_file(input_file_name, config, progress_callback=emit_file_progress)
        output_path = config.output_dir / input_file_name
        write_generate_grammar_output(output, output_path)
        output_paths.append(output_path)
        emit_progress(
            "file_written",
            {
                "file_index": file_index,
                "file_count": len(input_file_names),
                "input_file_name": input_file_name,
                "output_path": str(output_path),
            },
        )
    emit_progress(
        "run_finished",
        {
            "file_count": len(output_paths),
            "output_dir": str(config.output_dir),
        },
    )
    return output_paths
