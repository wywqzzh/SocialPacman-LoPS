from __future__ import annotations

from typing import Any

from LoPS.generate_grammar.grammar import GrammarLearningResult, SkipGramResult
from LoPS.generate_grammar.token import token_length


LEGACY_FIELD_ORDER = (
    "sets",
    "pro",
    "gram",
    "sequence",
    "time_pro",
    "frequency",
    "seq",
    "state",
    "S",
    "fileNames",
    "components",
    "skipGram",
    "skipGramNum",
)


def _legacy_places() -> list[str]:
    place_set = [chr(i) for i in range(32, 126)]
    for token in ("e", "G", "L", "E", "A", "1", "2", "3", "4", "S", "V", "N"):
        place_set.remove(token)
    return place_set


def _legacy_token(token: str) -> str:
    return token.replace("-", "")


def _legacy_symbol_by_token(tokens: list[str]) -> dict[str, str]:
    places = _legacy_places()
    symbol_by_token = {}
    place_index = 0
    for token in tokens:
        if token_length(token) == 1:
            symbol_by_token[token] = token
        else:
            symbol_by_token[token] = places[place_index]
            place_index += 1
    return symbol_by_token


def build_legacy_output(result: GrammarLearningResult, skip_gram: SkipGramResult) -> dict[str, Any]:
    symbol_by_token = _legacy_symbol_by_token(result.grammar_tokens)
    legacy_sets = [_legacy_token(token) for token in result.grammar_tokens]
    legacy_s = [symbol_by_token[token] for token in result.grammar_tokens]
    legacy_seq = "".join(symbol_by_token[token] for token in result.parsed_sequence)
    legacy_components = [
        [_legacy_token(component[0]), _legacy_token(component[1])]
        for component in result.components
    ]
    legacy_output = {}
    legacy_output["sets"] = legacy_sets
    legacy_output["pro"] = result.probabilities
    legacy_output["gram"] = [_legacy_token(token) for token in result.position_grammar]
    legacy_output["sequence"] = "".join(result.original_sequence)
    legacy_output["time_pro"] = result.time_probabilities
    legacy_output["frequency"] = result.frequencies
    legacy_output["seq"] = legacy_seq
    legacy_output["state"] = result.parsed_state_features
    legacy_output["S"] = legacy_s
    legacy_output["fileNames"] = result.participant_file_names
    legacy_output["components"] = legacy_components
    legacy_output["skipGram"] = skip_gram.found
    legacy_output["skipGramNum"] = skip_gram.count
    return legacy_output
