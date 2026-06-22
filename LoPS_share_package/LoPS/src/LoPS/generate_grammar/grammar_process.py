"""grammar 后处理中的人群划分逻辑。

本模块重构旧脚本 ``GrammarInduction/GrammarProcess.py`` 中的 ``DividePerson``。
正式入口只接收当前 ``generate_grammar`` 结构化输出，不再读取或保存旧版
``GrammarCluster`` 数据文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


@dataclass(frozen=True)
class DividePersonRecord:
    """保存单个 grammar 输出参与人群划分所需的最小字段。

    输入语义：由当前结构化 grammar pickle 读取后构造，字段已经转换为旧算法需要的
    连续 grammar 字符串和组件字符串。
    输出语义：作为 ``divide_person`` 的输入元素。
    关键约束：``grammar_tokens``、``components`` 和 ``frequencies`` 三者必须逐项对齐。
    """

    input_file_name: str
    participant_file_names: list[str]
    grammar_tokens: list[str]
    components: list[list[str]]
    frequencies: list[float]
    skip_gram_count: float


def list_grammar_files(grammar_dir: Path) -> list[str]:
    """列出当前 grammar 输出目录中的 pickle 文件名。

    输入语义：grammar_dir 是 ``11_generate_grammar.py`` 生成的结构化 grammar 目录。
    输出语义：返回排序后的 ``.pkl`` 文件名列表。
    关键约束：排序只用于保证输出摘要稳定，不改变单个文件的计算语义。
    """

    return sorted(path.name for path in grammar_dir.iterdir() if path.suffix == ".pkl")


def load_divide_person_record(path: Path) -> DividePersonRecord:
    """读取一个当前结构化 grammar pickle 并转换为人群划分记录。

    输入语义：path 指向当前 ``generate_grammar`` 的结构化输出，必须包含
    ``source``、``grammar`` 和 ``skip_gram`` 三个分区。
    输出语义：返回 ``DividePersonRecord``，其中复合 token 使用旧算法的无分隔符形式。
    关键约束：本函数不兼容旧版 ``sets/frequency/skipGramNum`` pickle；旧格式只应在独立
    验证脚本中转换，不能进入正式运行边界。
    """

    result = pd.read_pickle(path)
    missing_sections = [section for section in ("source", "grammar", "skip_gram") if section not in result]
    if missing_sections:
        raise KeyError(f"{path} 不是当前结构化 grammar 输出，缺少分区：{missing_sections}")

    grammar_items = list(result["grammar"])
    grammar_tokens = [_legacy_token(str(item["token"])) for item in grammar_items]
    components = [[_legacy_token(str(left)), _legacy_token(str(right))] for left, right in (item["components"] for item in grammar_items)]
    frequencies = [float(item["frequency"]) for item in grammar_items]
    return DividePersonRecord(
        input_file_name=path.name,
        participant_file_names=[str(name) for name in result["source"]["participant_file_names"]],
        grammar_tokens=grammar_tokens,
        components=components,
        frequencies=frequencies,
        skip_gram_count=float(result["skip_gram"]["count"]),
    )


def load_divide_person_records(grammar_dir: Path) -> list[DividePersonRecord]:
    """批量读取当前结构化 grammar 输出，构造人群划分输入记录。

    输入语义：grammar_dir 是当前 grammar 输出目录。
    输出语义：返回按文件名排序的 ``DividePersonRecord`` 列表。
    关键约束：目录必须存在且至少包含一个 ``.pkl`` 文件。
    """

    if not grammar_dir.is_dir():
        raise FileNotFoundError(f"grammar 输出目录不存在：{grammar_dir}")

    file_names = list_grammar_files(grammar_dir)
    if not file_names:
        raise FileNotFoundError(f"grammar 输出目录中没有 .pkl 文件：{grammar_dir}")
    return [load_divide_person_record(grammar_dir / file_name) for file_name in file_names]


def divide_person(records: list[DividePersonRecord], cluster_count: int = 2) -> dict[str, Any]:
    """按旧版 ``DividePerson`` 语义对被试 grammar 结果做人群划分。

    输入语义：records 是当前结构化 grammar 输出转换后的记录列表。
    输出语义：返回 JSON 友好的划分结果，包含特征矩阵、每个 cluster 的文件名、grammar book
    和 component book，以及逐文件归属。
    关键约束：本函数只返回内存结果，不保存 ``GrammarCluster`` 文件；聚类特征和 grammar book
    排序规则与旧脚本保持一致。
    """

    if not records:
        raise ValueError("records 不能为空。")

    features = build_divide_person_features(records)
    model = AgglomerativeClustering(n_clusters=cluster_count, metric="euclidean", linkage="ward")
    labels = model.fit_predict(features)

    clusters = []
    for label in sorted(set(int(value) for value in labels)):
        indices = np.where(labels == label)[0].tolist()
        grammar_book, component_book = build_cluster_grammar_book(records, indices)
        clusters.append(
            {
                "label": label,
                "indices": indices,
                "input_file_names": [records[index].input_file_name for index in indices],
                "participant_file_names": [records[index].participant_file_names[0] for index in indices],
                "grammar_book": grammar_book,
                "component_book": component_book,
            }
        )

    assignments = [
        {
            "input_file_name": record.input_file_name,
            "participant_file_name": record.participant_file_names[0] if record.participant_file_names else None,
            "label": int(labels[index]),
        }
        for index, record in enumerate(records)
    ]
    return {
        "input_file_names": [record.input_file_name for record in records],
        "features": features.tolist(),
        "labels": [int(label) for label in labels],
        "clusters": clusters,
        "assignments": assignments,
    }


def build_divide_person_features(records: list[DividePersonRecord]) -> np.ndarray:
    """构造旧版人群划分使用的三维 grammar 长度特征。

    输入语义：每个记录提供 grammar 频次和 skip-gram 次数。
    输出语义：返回形状为 ``(被试数, 3)`` 的浮点特征矩阵。
    关键约束：长度 1、长度 2、长度 3 及以上分别累加到三个维度；``skip`` 始终落入第三维。
    """

    features = np.zeros((len(records), 3), dtype=float)
    for record_index, record in enumerate(records):
        grammar = list(record.grammar_tokens)
        frequencies = np.array(record.frequencies, dtype=float)
        if record.skip_gram_count != 0:
            # 旧脚本把 skip-gram 作为额外的长度类别参与聚类特征，但不放入最终 grammar book。
            grammar.append("skip")
            frequencies = np.append(frequencies, record.skip_gram_count)

        total_frequency = float(np.sum(frequencies))
        if total_frequency == 0:
            raise ValueError(f"{record.input_file_name} 的 grammar 频次总和为 0，无法归一化。")
        probabilities = frequencies / total_frequency
        for grammar_token, probability in zip(grammar, probabilities, strict=True):
            feature_index = len(grammar_token) - 1 if len(grammar_token) <= 3 else 2
            features[record_index][feature_index] += probability
    return features


def build_cluster_grammar_book(records: list[DividePersonRecord], indices: list[int]) -> tuple[list[str], list[list[str]]]:
    """汇总一个 cluster 内所有被试的 grammar book 和 component book。

    输入语义：indices 是属于同一 cluster 的 records 下标。
    输出语义：返回去重并排序后的 grammar book 与逐项对齐的 component book。
    关键约束：每个被试都会额外加入旧流程固定的 ``N`` grammar；去重保留首次出现的组件。
    """

    grammar_book: list[str] = []
    component_book: list[list[str]] = []
    for index in indices:
        record = records[index]
        grammar_book.extend(record.grammar_tokens + ["N"])
        component_book.extend(record.components + [["N", ""]])

    unique_pairs = _unique_grammar_component_pairs(grammar_book, component_book)
    # 旧脚本先按字符串排序，再利用稳定排序按长度排序，长度相同的项保留字典序。
    unique_pairs = sorted(unique_pairs, key=lambda pair: pair[0])
    unique_pairs = sorted(unique_pairs, key=lambda pair: len(pair[0]))
    return [grammar for grammar, _ in unique_pairs], [component for _, component in unique_pairs]


def _unique_grammar_component_pairs(grammar_book: list[str], component_book: list[list[str]]) -> list[tuple[str, list[str]]]:
    """按旧脚本语义对 grammar/component 配对去重。

    输入语义：grammar_book 与 component_book 等长且逐项对应。
    输出语义：返回去重后的 ``(grammar, component)`` 列表。
    关键约束：同名 grammar 多次出现时保留第一次出现的 component。
    """

    if len(grammar_book) != len(component_book):
        raise ValueError("grammar_book 与 component_book 长度必须一致。")

    seen: set[str] = set()
    pairs: list[tuple[str, list[str]]] = []
    for grammar, component in zip(grammar_book, component_book, strict=True):
        if grammar in seen:
            continue
        seen.add(grammar)
        pairs.append((str(grammar), [str(item) for item in component]))
    return pairs


def _legacy_token(token: str) -> str:
    """把当前复合 token 表示转换为旧版连续字符串表示。"""

    return token.replace("-", "")
