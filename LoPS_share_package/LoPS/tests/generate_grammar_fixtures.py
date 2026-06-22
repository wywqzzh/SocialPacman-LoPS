"""generate_grammar 测试数据路径。

本模块集中定义测试使用的仓库内数据目录，避免测试文件重复拼接路径。
"""

from __future__ import annotations

from pathlib import Path


# 测试只读取 LoPS 仓库内 data 的当前主流程结果，保证测试环境不依赖仓库外部路径。
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
STRATEGY_SEQUENCE_DIR = DATA_ROOT / "09_strategy_sequence"
STATE_GRAPH_DIR = DATA_ROOT / "10_state_dependency_graph_data"
BASELINE_GRAMMAR_DIR = DATA_ROOT / "11_grammar"
