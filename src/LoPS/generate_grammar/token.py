from __future__ import annotations

from collections.abc import Sequence


TOKEN_SEPARATOR = "-"


def split_token(token: str) -> list[str]:
    # 新核心算法不再使用旧单字符占位符，复合 grammar 统一写成 "G-L-E-A"。
    # 空字符串只会出现在 legacy components 的占位字段中，拆分时保持为空列表。
    if token == "":
        return []
    return token.split(TOKEN_SEPARATOR)


def format_token(base_tokens: Sequence[str]) -> str:
    # 所有复合 token 都通过同一个分隔符生成，避免算法依赖字符串字符位置。
    return TOKEN_SEPARATOR.join(base_tokens)


def combine_tokens(parent_token: str, child_token: str) -> str:
    # 旧代码将 parent+child 直接拼成新 chunk；新实现先拆基础 token 再重新格式化。
    return format_token(split_token(parent_token) + split_token(child_token))


def token_length(token: str) -> int:
    # token 长度是基础动作数量，不是字符串长度；例如 "E-A" 的长度是 2。
    return len(split_token(token))


def tokens_share_base_token(left: str, right: str) -> bool:
    # 旧代码会跳过基础字符有交集的 parent/child，防止生成自重叠 chunk。
    return bool(set(split_token(left)) & set(split_token(right)))
