# Milestones

## v1.0 generateGrammar 重构（Shipped: 2026-05-05）

**Phases completed:** 3 phases, 13 plans, 38 tasks

**交付内容：**

- 建立 LoPS 科研脚本重构流程：目标脚本接收、原始行为分析、疑点讨论、方案确认、执行和验证。
- 完成 `generateGrammar.py` 第一轮完整重构，把默认运行分支及其调用模块重实现为 `src/LoPS/generate_grammar` 下的正式模块。
- 将运行、测试和验证需要的数据迁移到 `data/generate_grammar`，正式代码不依赖旧项目代码、旧项目数据目录或旧格式输出结构。
- 在 `script/generate_grammar` 下建立运行入口和验证入口；新旧比对通过独立验证适配器完成，不污染核心模块。
- 完成 `generate_grammar` 顶层算法优化，覆盖解析流程、概率统计、离散状态数据组织、候选评分、主循环和 skip-gram trace。
- 通过 26 个单元测试、34 被试全量逐 key/value 一致性验证，以及旧代码库 learn 过程级一致性验证。

**归档文件：**

- `.planning/milestones/v1.0-ROADMAP.md`
- `.planning/milestones/v1.0-REQUIREMENTS.md`
- `.planning/milestones/v1.0-MILESTONE-AUDIT.md`

**一致性结论：**

新实现使用当前仓库 `data/generate_grammar` 下的数据运行。34 个被试的新输出经 `script/generate_grammar/legacy_adapter.py` 映射后，与旧基准逐 key/value 完全一致；过程级验证也确认 `learn()` 关键迭代变量与旧实现一致。

**非阻塞观察：**

- 未生成独立的 `*-VALIDATION.md` Nyquist 标准文件；v1.0 已由 phase `VERIFICATION.md`、Phase 3 `REVIEW.md`、单元测试和全量验证覆盖。后续需要统一 GSD Nyquist 格式时可追加标准化验证文档。

---
