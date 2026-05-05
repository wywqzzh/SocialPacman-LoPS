---
quick_id: 260505-cs4
slug: src-generategrammar-data
status: complete
completed: 2026-05-05T09:22:29+08:00
commit: uncommitted
---

# Quick 任务总结：移除 src 中旧项目数据目录依赖

## 完成内容

- 已将本轮 `generateGrammar` 实际使用的数据迁移到 LoPS 仓库：
  - `data/generate_grammar/input/strategy_sequence`：34 个输入文件。
  - `data/generate_grammar/input/state_graph`：34 个输入文件。
  - `data/generate_grammar/baseline/grammar`：34 个旧基准文件。
- 已修改 `src/LoPS/generate_grammar/config.py`：
  - 删除旧项目数据目录默认常量。
  - `GenerateGrammarConfig` 的 `strategy_sequence_dir`、`state_graph_dir`、`output_dir` 仍由调用方构造配置时显式提供。
  - `baseline_grammar_dir` 仅作为验证基准路径，可由验证脚本显式传入。
- 已修改 `script/run_generate_grammar.py` 和 `script/validate_generate_grammar.py`：
  - 输入路径、输出路径和验证基准路径提供 `data/generate_grammar` 下的字符串默认值。
  - 仍可通过命令行覆盖这些默认路径。
- 已修改测试：
  - 测试只读取 `data/generate_grammar` 下的迁移数据。
  - scoring 测试不再导入旧项目代码，而是使用旧实现行为快照作为固定期望值。
- 已更新 `AGENTS.md`、`README.md` 和 `data/generate_grammar/README.md`，记录 `src` 中不得保存外部项目路径；运行脚本可以为本仓库固定数据目录设置默认参数。

## 验证

已运行：

```bash
PYTHONPATH=src /home/zzh/anaconda3/envs/LoPS/bin/python -m unittest discover -s tests
PYTHONPATH=src /home/zzh/anaconda3/envs/LoPS/bin/python script/run_generate_grammar.py --strategy-sequence-dir data/generate_grammar/input/strategy_sequence --state-graph-dir data/generate_grammar/input/state_graph --output-dir data/generate_grammar/smoke-output
PYTHONPATH=src /home/zzh/anaconda3/envs/LoPS/bin/python script/validate_generate_grammar.py --strategy-sequence-dir data/generate_grammar/input/strategy_sequence --state-graph-dir data/generate_grammar/input/state_graph --baseline-grammar-dir data/generate_grammar/baseline/grammar --output-dir data/generate_grammar/refactored-output/grammar
rg -n "/home/zzh/project/Pacman|Pac-man|Monkey_Analysis|structre-learning|DEFAULT_STRATEGY|DEFAULT_STATE_GRAPH|DEFAULT_BASELINE|DEFAULT_OUTPUT_DIR" src script tests data README.md AGENTS.md --glob '!*.pkl' --glob '!poetry.lock'
```

结果：

- 单元测试：18 个测试通过。
- 脚本运行：在 `data/generate_grammar/smoke-output` 生成 34 个文件。
- 一致性验证：`Validation passed for 34 files.`。
- 路径扫描：`src`、`script`、`tests`、`data` 说明和根说明中没有旧项目路径或旧默认路径常量。

## 提交状态

本 quick 任务未创建 git commit。当前工作区在任务开始前已有多项未提交修改，且本任务继续修改了其中部分文件；为避免混入非本任务改动，本次只完成文件修改和验证。
