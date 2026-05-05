# generateGrammar 数据来源记录

本轮 `generateGrammar` 重构所需数据已经迁移到 LoPS 仓库的 `data/generate_grammar` 下。新实现、运行脚本和测试都应读取本目录中的数据，不再依赖旧项目的数据目录。

## 输入目录

策略序列输入：

```text
data/generate_grammar/input/strategy_sequence/
```

状态图输入：

```text
data/generate_grammar/input/state_graph/
```

## 固定旧基准目录

旧实现 grammar 基准输出也已迁移到当前仓库：

```text
data/generate_grammar/baseline/grammar/
```

该基准在 Phase 2 分析阶段通过原始脚本 sandbox 全量重跑验证：34/34 个输出文件与既有基准 MD5 完全一致，因此作为本轮固定旧基准。后续验证只读取本仓库内的 `baseline/grammar`，不再读取旧项目目录。

## 新输出目录

新输出写入：

```text
data/generate_grammar/refactored-output/grammar/
```

脚本级 smoke test 或临时验证输出也应写在 `data/generate_grammar` 下，例如：

```text
data/generate_grammar/smoke-output/
```

所有验证输出和报告只写入 LoPS 仓库，不写入外部 Pacman 项目；其中数据文件写入 `data`，规划和结论文档写入 `.planning`。

## 代码路径规则

`src/LoPS` 中不得保存任何项目外部数据目录默认值。运行脚本可以为本目录下的固定数据位置设置默认参数；直接使用模块时，应在创建 `GenerateGrammarConfig` 时显式传入输入目录、输出目录和验证基准目录。
