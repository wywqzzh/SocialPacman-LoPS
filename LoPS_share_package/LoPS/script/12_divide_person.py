#!/usr/bin/env python3
"""运行 grammar DividePerson 人群划分后处理。

该入口读取当前 ``generate_grammar`` 结构化输出目录，按旧版 DividePerson 语义计算
两类人群及各自 grammar book，并把结果打印为 JSON。脚本不保存任何中间或最终数据文件。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.generate_grammar.grammar_process import divide_person, load_divide_person_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析 DividePerson 运行入口参数。

    输入语义：调用方提供当前结构化 grammar 输出目录和聚类数量。
    输出语义：返回 argparse 参数对象。
    关键约束：默认只面向当前 data 主流程的结构化 grammar 输出，不兼容旧版 grammar pickle。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grammar-dir",
        type=Path,
        default=PROJECT_ROOT / "data/11_grammar",
        help="当前 generate_grammar 结构化输出目录。",
    )
    parser.add_argument("--cluster-count", type=int, default=2, help="聚类数量，旧版 DividePerson 固定为 2。")
    parser.add_argument("--indent", type=int, default=2, help="JSON 输出缩进；设为 0 时输出单行 JSON。")
    return parser.parse_args()


def main() -> None:
    """读取当前 grammar 输出，执行人群划分，并把结果打印到标准输出。"""

    args = parse_args()
    records = load_divide_person_records(args.grammar_dir)
    result = divide_person(records, cluster_count=args.cluster_count)
    indent = None if args.indent == 0 else args.indent
    print(json.dumps(result, ensure_ascii=False, indent=indent))


if __name__ == "__main__":
    main()
