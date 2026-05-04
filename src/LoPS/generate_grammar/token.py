from __future__ import annotations

from collections.abc import Sequence


TOKEN_SEPARATOR = "-"


def split_token(token: str) -> list[str]:
    if token == "":
        return []
    return token.split(TOKEN_SEPARATOR)


def format_token(base_tokens: Sequence[str]) -> str:
    return TOKEN_SEPARATOR.join(base_tokens)


def combine_tokens(parent_token: str, child_token: str) -> str:
    return format_token(split_token(parent_token) + split_token(child_token))


def token_length(token: str) -> int:
    return len(split_token(token))


def tokens_share_base_token(left: str, right: str) -> bool:
    return bool(set(split_token(left)) & set(split_token(right)))
