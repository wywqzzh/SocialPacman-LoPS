"""generateGrammar 默认路径重构模块入口。

该 package 只暴露最常用的配置对象；具体的数据读取、状态图、scoring、
grammar 学习和输出适配逻辑分别放在同级模块中，避免入口文件承载业务细节。
"""

from .config import GenerateGrammarConfig, GrammarLearningParams

__all__ = ["GenerateGrammarConfig", "GrammarLearningParams"]
