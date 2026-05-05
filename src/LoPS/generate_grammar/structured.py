from __future__ import annotations

from dataclasses import asdict
from typing import Any

from LoPS.generate_grammar.config import GrammarLearningParams
from LoPS.generate_grammar.grammar import GrammarLearningResult, SkipGramResult
from LoPS.generate_grammar.token import split_token


def build_structured_output(
    input_file_name: str,
    params: GrammarLearningParams,
    result: GrammarLearningResult,
    skip_gram: SkipGramResult,
) -> dict[str, Any]:
    # structured 输出不参与旧结果一致性判定，目标是给后续科研分析提供清晰、去冗余的结构。
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
            # source 记录文件来源和被试信息，participant_file_names 保留旧后缀，participant_ids 供新结构使用。
            "input_file_name": input_file_name,
            "participant_file_names": result.participant_file_names,
            "participant_ids": result.participant_ids,
        },
        # 参数完整展开，保证同一份 structured 输出可以追溯当时的学习阈值和状态列。
        "parameters": asdict(params),
        "grammar": grammar_items,
        "parsed": {
            # parsed 保存最终解析序列、对齐状态和每个基础位置对应的 grammar。
            "sequence": result.parsed_sequence,
            "state_features": result.parsed_state_features,
            "position_grammar": result.position_grammar,
        },
        "skip_gram": {
            # skip_gram 字段使用新目标 token 名称，例如旧 EA 在核心中表示为 E-A。
            "target": params.skip_gram_target,
            "found": skip_gram.found,
            "count": skip_gram.count,
        },
    }
