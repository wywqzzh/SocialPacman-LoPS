from __future__ import annotations

from pathlib import Path


# 测试只能读取 LoPS 仓库内已经迁移的数据，不能再依赖旧 Pacman 项目的数据目录。
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "generate_grammar"
STRATEGY_SEQUENCE_DIR = DATA_ROOT / "input" / "strategy_sequence"
STATE_GRAPH_DIR = DATA_ROOT / "input" / "state_graph"
BASELINE_GRAMMAR_DIR = DATA_ROOT / "baseline" / "grammar"
