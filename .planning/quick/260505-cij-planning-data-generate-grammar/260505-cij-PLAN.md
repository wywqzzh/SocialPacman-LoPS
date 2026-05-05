---
quick_id: 260505-cij
slug: planning-data-generate-grammar
status: in_progress
created: 2026-05-05
---

# Quick 任务计划：generateGrammar 数据目录迁移

## 目标

当前 `generateGrammar` 重构相关脚本和测试使用的 LoPS 内部数据不能继续放在 `.planning` 中。`.planning` 只保留 GSD 规划、阶段记录和分析文档；脚本输入、脚本输出、验证输出等数据类文件应放在 `data/generate_grammar` 的合适目录下。

## 范围

- 将新实现默认输出目录从 `.planning/runs/2026-05-04-generateGrammar/refactored-output/grammar2` 改为 `data/generate_grammar/refactored-output/grammar`。
- 将已有的 34 个新实现输出 `pkl` 文件迁移到 `data/generate_grammar/refactored-output/grammar`。
- 调整 `.gitignore`，避免继续把 `.planning/runs` 当作运行数据输出目录。
- 更新 `data/generate_grammar/README.md`、`README.md` 和 `AGENTS.md` 中的数据目录规则。
- 运行现有测试，并用 `script/run_generate_grammar.py` 指定 `data/generate_grammar/smoke-output` 做一次脚本级验证。

## 非目标

- 不复制外部 Pacman 项目的只读原始数据和旧 `grammar2` 基准目录。
- 不修改原始 Pacman 脚本和原始数据目录。
- 不改动 `generateGrammar` 的业务算法。
