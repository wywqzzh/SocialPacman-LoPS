from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# 旧脚本 ghost2 默认最终使用 6 个状态列；BN10 等被旧代码覆盖掉，不属于本轮有效分支。
DEFAULT_STATE_NAMES = ("IS1", "IS2", "PG1", "PG2", "PE", "BN5")


@dataclass(frozen=True)
class GrammarLearningParams:
    # 状态列、alpha、阈值和排除 token 都从旧 generateGrammar.py 默认路径抽取而来。
    # 将它们集中到参数对象中，避免核心算法里散落魔法数字，也便于后续新脚本复用。
    state_names: tuple[str, ...] = DEFAULT_STATE_NAMES
    chunk_alpha: float = 0.5
    condition_alpha: float = 0.5
    skip_gram_alpha: float = 0.5
    max_iterations: int = 100000
    convergence_window: int = 5
    convergence_kl_threshold: float = 0.05
    candidate_ratio_min: float = 1.0
    candidate_ratio_keep: float = 0.85
    min_pair_frequency: float = 0.05
    removed_token: str = "N"
    skip_gram_target: str = "E-A"
    skip_gram_min_offset: int = 2
    skip_gram_max_offset: int = 5
    skip_gram_min_frequency: float = 0.025
    excluded_child_tokens: tuple[str, ...] = ("V", "1", "2", "N", "S", "e")
    excluded_parent_tokens: tuple[str, ...] = ("V", "N")
    reject_shared_base_tokens: bool = True


@dataclass(frozen=True)
class GenerateGrammarConfig:
    # 路径配置只负责描述输入、输出和验证基准；算法参数由 learning 单独承载。
    # 输入和输出路径必须由调用方显式传入，src 层不保存任何项目外部数据目录默认值。
    strategy_sequence_dir: Path
    state_graph_dir: Path
    output_dir: Path
    baseline_grammar_dir: Path | None = None
    learning: GrammarLearningParams = field(default_factory=GrammarLearningParams)

    def validate(self) -> None:
        # 输入目录必须已经存在；当前项目内的默认数据由脚本层显式传入。
        if not self.strategy_sequence_dir.is_dir():
            raise FileNotFoundError(f"strategy_sequence directory not found: {self.strategy_sequence_dir}")
        if not self.state_graph_dir.is_dir():
            raise FileNotFoundError(f"state_graph directory not found: {self.state_graph_dir}")
        if self.baseline_grammar_dir is not None and not self.baseline_grammar_dir.is_dir():
            raise FileNotFoundError(f"Baseline grammar directory not found: {self.baseline_grammar_dir}")

        output = self.output_dir
        # 防止误把新结果写回旧 grammar 基准目录；旧目录只能作为验证基准读取。
        if self.baseline_grammar_dir is not None and output.resolve() == self.baseline_grammar_dir.resolve():
            raise ValueError("output_dir must not be the original baseline grammar directory")
        output.mkdir(parents=True, exist_ok=True)
