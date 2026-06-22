"""generate_grammar 包的公共入口。

该 package 只暴露常用配置对象；数据边界、grammar 学习和结构化输出逻辑
分别放在同级模块中。通用结构学习算法放在 LoPS.structure_learning 中。
"""

from .config import GenerateGrammarConfig, GrammarLearningParams

__all__ = ["GenerateGrammarConfig", "GrammarLearningParams"]
