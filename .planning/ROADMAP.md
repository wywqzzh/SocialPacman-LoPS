# 路线图：LoPS

## 概览

v1 的目标是把 LoPS 建成一套可重复的科研脚本重构流程：Phase 1 建立任务接收和目录契约；从 Phase 2 开始，每个 phase 都对应一项具体脚本重构，并在同一个 phase 内完成深度分析、疑点讨论、方案设计、用户确认、实现和一致性验证。

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions marked with INSERTED

- [x] **Phase 1: 项目骨架与任务接收契约** - 定义每轮重构如何接收目标脚本、环境、数据和状态记录。 (completed 2026-05-03)
- [x] **Phase 2: 重构 generateGrammar 模块** - 作为第一项完整脚本重构，在一个 phase 内完成 generateGrammar.py 的深度分析、多轮讨论、方案确认、实现和一致性验证。 (completed 2026-05-04)
- [ ] **Phase 3: generateGrammar 顶层算法审计与优化** - 在保持科研结果可验证的前提下，从顶层学习流程开始重新审计算法设计，识别可删除、可合并、可缓存或可改写的逻辑，再决定底层函数的保留、删除或修改。

## Phase Details

### Phase 1: 项目骨架与任务接收契约
**Goal**: 明确一轮重构开始时必须收集的信息，并让仓库目录和记录方式支持后续执行。
**Depends on**: Nothing (first phase)
**Requirements**: [INTK-01, INTK-02, INTK-03, INTK-04]
**UI hint**: no
**Success Criteria** (what must be TRUE):
  1. 用户可以提供目标脚本路径、环境、运行命令和数据来源。
  2. 每轮重构都有统一记录位置保存输入信息和当前状态。
  3. `src/LoPS`、`script`、`data`、`docs`、`.planning` 的职责被明确记录。
  4. 后续阶段可以直接读取任务记录，不需要重新询问基础上下文。
**Plans**: 2 plans

Plans:
**Wave 1**
- [x] 01-01: 定义每轮重构的任务接收记录格式。

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 01-02: 补齐项目目录职责和初始化说明。

### Phase 2: 重构 generateGrammar 模块
**Goal**: 完整重构 generateGrammar.py 脚本：先深度分析原始行为（包括依赖模块、随机过程和数据使用），再围绕不清楚的功能和可舍弃范围与用户讨论，随后制定并确认重构方案，最后在 LoPS 中实现新模块及其调用模块，并验证新旧输出完全一致。
**Depends on**: Phase 1
**Requirements**: [ANLY-01, ANLY-02, ANLY-03, ANLY-04, ANLY-05, DSGN-01, DSGN-02, DSGN-03, DSGN-04, MOD-01, MOD-02, MOD-03, MOD-04, MOD-05, ARCH-01, ARCH-02, ARCH-03, ARCH-04, DATA-01, DATA-02, DATA-03, DATA-04, VERF-01, VERF-02, VERF-03, VERF-04, VERF-05, VERF-06, VERF-07, VERF-08, VERF-09, VERF-10]
**UI hint**: no
**Success Criteria** (what must be TRUE):
  1. 已产出脚本深度分析报告，覆盖当前功能、执行流程、调用模块、输入输出、数据来源、随机过程、副作用和工作目录假设。
  2. 目标脚本在本轮有效分支中调用到的其它本地模块已被纳入分析、重构范围和测试范围；除非用户明确确认某个模块或分支不参与。
  3. discuss 阶段已基于分析报告向用户追问不清楚的功能、边界、保留范围、舍弃范围和验证期望；如问题未收敛，允许多轮 discuss 后再进入 plan。
  4. plan 阶段已制定详细重构计划和实施计划，覆盖模块边界、接口设计、数据路径、运行入口、迁移步骤和验证方式。
  5. 重构方案经用户确认后，才修改正式实现代码。
  6. 重构方案明确说明这不是代码搬运，而是基于原始行为进行架构模块重设计、接口重设计和代码重实现。
  7. 新架构满足高内聚低耦合，且恪守 KISS 原则：优先直接、清晰、易维护，避免过度工程化、过早抽象和不必要的防御性设计。
  8. `src/LoPS` 中实现了边界清晰的新模块，通过显式参数接收路径、配置和必要随机种子。
  9. `script` 中存在可运行新模块的入口脚本。
  10. 必要数据被整理到 `data` 并记录来源。
  11. 使用相同输入和随机种子，新旧脚本级输出完全一致；若完全一致不现实，必须记录原因、容差和差异结论。
  12. 参与重构的调用模块已完成模块级行为测试，并在相同数据和相同随机参数下与原始模块结果一致。
  13. `src/LoPS/temp` 中的临时验证代码已清理。
  14. 完成记录说明运行方式、验证方式和一致性结论。
**Plans**: 5 plans

Plans:
**Wave 1**
- [x] 02-01: 建立 generate_grammar 基础模块和数据入口。

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 02-02: 重实现 scoring 模块并对比原始模块行为。

**Wave 3** *(blocked on Wave 2 completion)*
- [x] 02-03: 实现 token 化 grammar 学习核心。

**Wave 4** *(blocked on Wave 3 completion)*
- [x] 02-04: 实现 legacy/structured 输出、pipeline 和运行脚本。

**Wave 5** *(blocked on Wave 4 completion)*
- [x] 02-05: 建立验证脚本、数据来源记录和完成验证报告。

### Phase 3: generateGrammar 顶层算法审计与优化
**Goal**: 对 Phase 2 已完成的 `generate_grammar` 模块进行算法级优化。优化必须从顶层学习流程和数据流开始，而不是先局部微调底层函数；顶层方案需要决定哪些底层算法应保留、删除、合并、缓存、向量化或重写，并通过全量验证证明结果仍可追溯。
**Depends on**: Phase 2
**Requirements**: [OPT-01, OPT-02, OPT-03, OPT-04, OPT-05, OPT-06, OPT-07, OPT-08]
**UI hint**: no
**Success Criteria** (what must be TRUE):
  1. 已产出当前 `generate_grammar` 顶层算法分析报告，覆盖 pipeline、`GrammarLearner.learn`、候选生成、状态矩阵构建、BD score、收敛逻辑、skip-gram 检测的调用关系、数据形状和复杂度。
  2. discuss 阶段从顶层流程开始确认哪些计算是必要的，哪些可以延迟、缓存、合并、删除或换成更直接的表达。
  3. 每个优化点都说明影响范围：涉及的顶层步骤、底层函数、数据结构、输出字段和验证风险。
  4. plan 阶段先形成完整优化设计，经用户确认后才修改正式实现代码，避免零散底层改动先行。
  5. 优化设计不得重新引入旧版本代码、旧版本数据路径或旧格式兼容逻辑。
  6. 优化实现继续遵守 KISS 原则，优先选择清晰、直接、可验证的算法和数据结构。
  7. 每个算法优化都有对应等价性验证；默认要求 34 个被试的新输出经验证适配后与基准输出逐 key/value 一致。
  8. 完成记录说明优化前后的运行方式、验证方式、性能观察和一致性结论。
**Plans**: 6 plans

Plans:
**Wave 1**
- [x] 03-01: 建立过程一致性基线和核心类型契约。

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 03-02: 合并解析与概率统计。

**Wave 3** *(blocked on Wave 2 completion)*
- [x] 03-03: 数组化离散状态数据组织。

**Wave 4** *(blocked on Wave 3 completion)*
- [x] 03-04: 重整 GrammarLearner.learn 主循环。

**Wave 5** *(blocked on Wave 4 completion)*
- [ ] 03-05: 重构 skip-gram 和输出适配边界。

**Wave 6** *(blocked on Wave 5 completion)*
- [ ] 03-06: 全量回归验证和优化记录。

## 重构 Phase 模式

从 Phase 2 开始，每个新增重构 phase 都代表一项完整脚本重构，而不是把一次重构拆成多个 phase。若某个已完成重构模块需要算法级优化，可以新增独立优化 phase；优化 phase 必须先从顶层算法和数据流审计开始，再决定底层算法的保留、删除或修改。每个重构 phase 应按以下顺序推进：

1. 用户提供目标脚本、运行环境、运行命令、数据来源和预期输出。
2. 先对目标脚本及其在本轮有效分支中调用到的本地模块做深度分析，并把分析报告写入该 phase 文档。
3. 基于分析报告进入 discuss，向用户追问不清楚的功能、边界、弃用逻辑、数据语义和验证要求。
4. 如果讨论后仍不清楚，继续 discuss；不强行进入 plan。
5. plan 阶段制定详细重构计划和实施计划，并在用户审核通过后进入 execute。计划必须体现架构模块重设计、接口重设计和代码重实现，而不是把旧代码搬运到新目录；计划还必须覆盖被目标脚本调用且参与重构的本地模块。
6. execute 阶段按确认计划实现重构、整理数据和运行入口。
7. 最后验证新旧脚本级输出一致性，并对参与重构的调用模块做模块级行为测试，留下验证记录。

## 架构设计原则

所有重构 phase 都必须遵守以下原则：

1. 重构的目标是重新设计高内聚低耦合的模块和接口，不是机械迁移旧代码。
2. 先保护原始行为和科研结果，再在这个边界内重新实现更清晰的结构。
3. 恪守 KISS 原则，优先采用直接、清晰、易维护的实现。
4. 避免过度工程化、过早抽象和不必要的防御性设计。
5. 新增抽象必须能降低实际复杂度、隔离真实变化点，或消除有意义的重复。

## 调用模块重构与测试原则

所有重构 phase 都必须遵守以下规则：

1. 目标脚本在本轮有效分支中调用到的本地模块，默认属于同一轮重构范围。
2. 若某个调用模块或分支不参与重构，必须在 discuss 或 plan 中获得用户明确确认，并记录原因。
3. 被纳入范围的调用模块必须重构到 LoPS 的新模块结构中，不能只保留旧模块作为黑盒依赖。
4. 每个参与重构的调用模块都必须有模块级行为测试。
5. 模块级行为测试必须使用相同数据和相同随机参数，对比重构模块与原始模块结果一致。

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> ...

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. 项目骨架与任务接收契约 | 2/2 | Complete    | 2026-05-03 |
| 2. 重构 generateGrammar 模块 | 5/5 | Complete | 2026-05-04 |
| 3. generateGrammar 顶层算法审计与优化 | 0/6 | Ready to execute | - |
