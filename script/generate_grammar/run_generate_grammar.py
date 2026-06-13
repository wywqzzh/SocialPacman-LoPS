"""generate_grammar 命令行运行入口。

该脚本读取固定数据目录或用户传入的数据目录，运行新版本流水线并写出结构化结果。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

from LoPS.generate_grammar.config import (
    GenerateGrammarConfig,
    GrammarLearningParams,
)
from LoPS.generate_grammar.pipeline import run_generate_grammar


def build_progress_printer(progress_interval: int) -> Callable[[str, Mapping[str, object]], None]:
    """构造只打印每轮学习后 chunk 集合的过程信息函数。

    输入语义：progress_interval 控制学习迭代事件的打印间隔，最小按 1 处理。
    输出语义：返回一个接收事件名和事件 payload 的回调函数。
    关键约束：打印函数只响应 learn_iteration 事件，不展示候选数、KL、skip-gram 等其它过程指标。
    """

    interval = max(1, progress_interval)

    def print_progress(event: str, payload: Mapping[str, object]) -> None:
        """在每轮学习后打印当前 chunk 集合。"""

        if event != "learn_iteration":
            return

        file_index = payload.get("file_index")
        file_count = payload.get("file_count")
        file_name = payload.get("input_file_name", "-")
        file_prefix = f"[file {file_index}/{file_count}] {file_name}" if file_index else f"[file] {file_name}"
        iteration = int(payload.get("iteration") or 0)
        # 终止迭代无论是否命中间隔都打印，避免收敛或无候选时缺少最终 chunk 集合。
        if iteration % interval != 0 and payload.get("stop_reason") is None:
            return
        chunk_set = list(payload.get("active_tokens") or [])
        print(f"{file_prefix} iter={iteration} chunks={chunk_set}")

    return print_progress


def parse_args() -> argparse.Namespace:
    """解析 generate_grammar 运行入口的命令行参数。

    返回值包含输入目录、状态图目录、输出目录和学习参数；默认路径指向
    当前仓库内前置脚本生成的数据，调用方可以通过命令行覆盖。
    """
    # 默认输入接入当前仓库的正式链路：预处理脚本生成 strategy_sequence，
    # 状态依赖图脚本生成 state_dependency_graph，grammar 脚本只消费这些新结构数据。
    parser = argparse.ArgumentParser(description="Run LoPS generate_grammar refactor pipeline.")
    parser.add_argument(
        "--strategy-sequence-dir",
        type=Path,
        default="pipeline_data/human_fmri_data_preprocess/strategy_sequence",
    )
    parser.add_argument(
        "--state-graph-dir",
        type=Path,
        default="pipeline_data/state_dependency_graph/state_dependency_graph_data",
    )
    parser.add_argument("--output-dir", type=Path, default="pipeline_data/generate_grammar/grammar")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--progress-interval", type=int, default=1, help="每隔多少轮学习迭代打印一次过程信息。")
    parser.add_argument("--quiet", action="store_true", help="只打印最终摘要，不打印学习过程信息。")
    return parser.parse_args()


def main() -> None:
    """执行 generate_grammar 文件级处理流程并打印输出摘要。

    函数从命令行参数构造配置对象，运行 pipeline 写出结果文件；无返回值，
    运行失败时由底层异常直接暴露给命令行调用方。
    """
    args = parse_args()
    # 单个 alpha 参数同步应用到 chunk、condition 和 skip-gram 三类学习过程。
    learning = replace(
        GrammarLearningParams(),
        chunk_alpha=args.alpha,
        condition_alpha=args.alpha,
        skip_gram_alpha=args.alpha,
        max_iterations=args.max_iterations,
    )
    # GenerateGrammarConfig 集中承载输入、输出和学习参数，pipeline 只消费这个对象。
    config = GenerateGrammarConfig(
        strategy_sequence_dir=args.strategy_sequence_dir,
        state_graph_dir=args.state_graph_dir,
        output_dir=args.output_dir,
        learning=learning,
    )
    progress_callback = None if args.quiet else build_progress_printer(args.progress_interval)
    output_paths = run_generate_grammar(config, progress_callback=progress_callback)
    # 输出简短运行结果，便于 shell 调用和验证报告记录。
    print(f"Generated {len(output_paths)} files in {config.output_dir}")


if __name__ == "__main__":
    main()
