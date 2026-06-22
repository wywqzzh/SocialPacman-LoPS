"""generate_grammar 模块的配置对象和默认学习参数。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# 当前 grammar 学习流程使用的状态特征列；列顺序会影响状态组合编码和得分结果。
DEFAULT_STATE_NAMES = ("IS1", "IS2", "PG1", "PG2", "PE", "BW10")


@dataclass(frozen=True)
class GrammarLearningParams:
    """保存 grammar 学习算法的数值阈值、状态列和 token 过滤规则。

    输入语义：字段均为不可变配置值，由运行脚本或调用方传入核心学习流程。
    输出语义：作为配置快照参与学习和结构化输出，不在算法过程中被修改。
    关键约束：状态列顺序、alpha 和候选筛选阈值会直接影响学习结果，调用方应显式记录。
    """

    # 将学习参数集中到单一对象中，避免核心算法散落魔法数字，也便于脚本复用。
    state_names: tuple[str, ...] = DEFAULT_STATE_NAMES
    chunk_alpha: float = 0.5
    condition_alpha: float = 0.5
    skip_gram_alpha: float = 0.5
    max_iterations: int = 100000
    convergence_window: int = 5
    convergence_kl_threshold: float = 0.05
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
    """保存一次 generate_grammar 运行所需的输入、状态图、输出路径和学习参数。

    输入语义：strategy_sequence_dir 与 state_graph_dir 必须指向已存在的数据目录。
    输出语义：output_dir 会作为结构化结果写入位置，由 validate() 确保存在。
    关键约束：核心配置不内置项目外部路径，所有数据来源必须由调用方显式传入。
    """

    # 路径配置只描述本次运行所需的输入和输出，验证基准等流程由脚本层单独管理。
    strategy_sequence_dir: Path
    state_graph_dir: Path
    output_dir: Path
    learning: GrammarLearningParams = field(default_factory=GrammarLearningParams)

    def validate(self) -> None:
        """检查输入目录并创建输出目录。

        输入语义：读取当前配置对象中的三个路径字段，不接收额外参数。
        输出语义：成功时返回 None，并保证 output_dir 已创建；路径无效时抛出 FileNotFoundError。
        关键约束：只创建输出目录，不自动创建或修正输入目录，避免误用缺失数据。
        """

        # 输入目录必须已经存在；当前项目内的默认数据由脚本层显式传入。
        if not self.strategy_sequence_dir.is_dir():
            raise FileNotFoundError(f"strategy_sequence directory not found: {self.strategy_sequence_dir}")
        if not self.state_graph_dir.is_dir():
            raise FileNotFoundError(f"state_graph directory not found: {self.state_graph_dir}")
        output = self.output_dir
        output.mkdir(parents=True, exist_ok=True)
