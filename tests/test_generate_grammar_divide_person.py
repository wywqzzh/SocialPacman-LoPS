"""generate_grammar DividePerson 后处理测试。"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from LoPS.generate_grammar.grammar_process import (
    DividePersonRecord,
    build_cluster_grammar_book,
    build_divide_person_features,
    load_divide_person_record,
)


def test_build_divide_person_features_matches_legacy_length_bins() -> None:
    """验证 DividePerson 三维特征按旧版 grammar 长度规则累计。"""

    records = [
        DividePersonRecord(
            input_file_name="a.pkl",
            participant_file_names=["a.pkl"],
            grammar_tokens=["G", "GL", "GEA"],
            components=[["G", ""], ["G", "L"], ["GE", "A"]],
            frequencies=[1, 2, 3],
            skip_gram_count=4,
        )
    ]

    features = build_divide_person_features(records)

    np.testing.assert_array_equal(features, np.array([[0.1, 0.2, 0.7]]))


def test_build_cluster_grammar_book_deduplicates_and_sorts_like_legacy() -> None:
    """验证 cluster grammar book 去重、追加 N 和排序规则与旧版一致。"""

    records = [
        DividePersonRecord(
            input_file_name="a.pkl",
            participant_file_names=["a.pkl"],
            grammar_tokens=["GL", "G", "EA"],
            components=[["G", "L"], ["G", ""], ["E", "A"]],
            frequencies=[2, 1, 3],
            skip_gram_count=0,
        ),
        DividePersonRecord(
            input_file_name="b.pkl",
            participant_file_names=["b.pkl"],
            grammar_tokens=["G", "A"],
            components=[["unexpected", ""], ["A", ""]],
            frequencies=[5, 1],
            skip_gram_count=0,
        ),
    ]

    grammar_book, component_book = build_cluster_grammar_book(records, [0, 1])

    assert grammar_book == ["A", "G", "N", "EA", "GL"]
    assert component_book == [["A", ""], ["G", ""], ["N", ""], ["E", "A"], ["G", "L"]]


def test_load_divide_person_record_accepts_only_structured_output(tmp_path: Path) -> None:
    """验证正式读取边界只接受当前结构化 grammar 输出。"""

    structured_path = tmp_path / "structured.pkl"
    with structured_path.open("wb") as file:
        pickle.dump(
            {
                "source": {"participant_file_names": ["a.pkl"]},
                "grammar": [
                    {"token": "G-L", "frequency": 3, "components": ["G", "L"]},
                    {"token": "E", "frequency": 2, "components": ["E", ""]},
                ],
                "skip_gram": {"count": 1},
            },
            file,
        )

    record = load_divide_person_record(structured_path)

    assert record.grammar_tokens == ["GL", "E"]
    assert record.components == [["G", "L"], ["E", ""]]
    assert record.frequencies == [3.0, 2.0]
    assert record.skip_gram_count == 1.0

    old_format_path = tmp_path / "old.pkl"
    with old_format_path.open("wb") as file:
        pickle.dump({"sets": ["G"], "frequency": [1], "skipGramNum": 0}, file)

    with pytest.raises(KeyError):
        load_divide_person_record(old_format_path)
