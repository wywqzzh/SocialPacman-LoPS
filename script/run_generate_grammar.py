from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from LoPS.generate_grammar.config import (
    GenerateGrammarConfig,
    GrammarLearningParams,
)
from LoPS.generate_grammar.pipeline import run_generate_grammar


def parse_args() -> argparse.Namespace:
    # 常用数据目录固定在 data/generate_grammar 下，因此参数提供字符串默认值。
    # 用户仍可通过命令行覆盖这些路径，便于比较其它数据集或临时输出目录。
    parser = argparse.ArgumentParser(description="Run LoPS generate_grammar refactor pipeline.")
    parser.add_argument("--strategy-sequence-dir", type=Path, default="data/generate_grammar/input/strategy_sequence")
    parser.add_argument("--state-graph-dir", type=Path, default="data/generate_grammar/input/state_graph")
    parser.add_argument("--output-dir", type=Path, default="data/generate_grammar/refactored-output/grammar")
    parser.add_argument("--baseline-grammar-dir", type=Path, default="data/generate_grammar/baseline/grammar")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-iterations", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 旧入口 main("ghost2", 0.5, False) 的 alpha 同时影响 chunk、condition 和 skip-gram。
    learning = replace(
        GrammarLearningParams(),
        chunk_alpha=args.alpha,
        condition_alpha=args.alpha,
        skip_gram_alpha=args.alpha,
        max_iterations=args.max_iterations,
    )
    # GenerateGrammarConfig 集中承载输入、输出和验证基准路径，run_generate_grammar 只消费这个对象。
    config = GenerateGrammarConfig(
        strategy_sequence_dir=args.strategy_sequence_dir,
        state_graph_dir=args.state_graph_dir,
        output_dir=args.output_dir,
        baseline_grammar_dir=args.baseline_grammar_dir,
        learning=learning,
    )
    output_paths = run_generate_grammar(config)
    # 输出简短运行结果，便于 shell 调用和验证报告记录。
    print(f"Generated {len(output_paths)} files in {config.output_dir}")


if __name__ == "__main__":
    main()
