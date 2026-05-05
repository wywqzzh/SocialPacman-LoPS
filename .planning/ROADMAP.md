# 路线图：LoPS

## 概览

LoPS 的长期目标是形成一套可重复执行的科研脚本重构流程：每一项脚本重构都先分析原始行为，再讨论不清楚的功能和边界，随后制定并确认重构方案，最后完成模块化实现、数据整理、运行入口和一致性验证。

v1.0 已完成并归档。当前没有正在执行的 phase；下一项重构需要先由用户提供目标脚本、运行环境、运行命令和数据来源。

## Milestones

- ✅ **v1.0 generateGrammar 重构** — Phase 1-3，已于 2026-05-05 归档。
- 📋 **v1.1 待规划** — 等待下一项目标脚本和重构范围。

## Phases

- [x] **Phase 1: 项目骨架与任务接收契约** - 定义每轮重构如何接收目标脚本、环境、数据和状态记录。 (completed 2026-05-03)
- [x] **Phase 2: 重构 generateGrammar 模块** - 作为第一项完整脚本重构，在一个 phase 内完成 generateGrammar.py 的深度分析、多轮讨论、方案确认、实现和一致性验证。 (completed 2026-05-04)
- [x] **Phase 3: generateGrammar 顶层算法审计与优化** - 在保持科研结果可验证的前提下，从顶层学习流程开始重新审计算法设计，识别可删除、可合并、可缓存或可改写的逻辑，再决定底层函数的保留、删除或修改。 (completed 2026-05-05)

<details>
<summary>✅ v1.0 generateGrammar 重构（Phase 1-3）— SHIPPED 2026-05-05</summary>

- [x] **Phase 1: 项目骨架与任务接收契约** — 2/2 plans，完成于 2026-05-03。
- [x] **Phase 2: 重构 generateGrammar 模块** — 5/5 plans，完成于 2026-05-04。
- [x] **Phase 3: generateGrammar 顶层算法审计与优化** — 6/6 plans，完成于 2026-05-05。

</details>

### 下一里程碑待规划

- [ ] 下一项重构 phase：等待用户提供目标脚本、环境、命令和数据来源。

## Phase Details

### Phase 1: 项目骨架与任务接收契约
**Goal**: 明确一轮重构开始时必须收集的信息，并让仓库目录和记录方式支持后续执行。  
**Depends on**: Nothing (first phase)  
**Requirements**: [INTK-01, INTK-02, INTK-03, INTK-04]  
**UI hint**: no  
**Plans**: 2 plans

Plans:
- [x] 01-01: 定义每轮重构的任务接收记录格式。
- [x] 01-02: 补齐项目目录职责和初始化说明。

### Phase 2: 重构 generateGrammar 模块
**Goal**: 完整重构 `generateGrammar.py` 脚本：先深度分析原始行为和调用模块，再讨论、设计、实现并验证新旧输出一致。  
**Depends on**: Phase 1  
**Requirements**: [ANLY-01, ANLY-02, ANLY-03, ANLY-04, ANLY-05, DSGN-01, DSGN-02, DSGN-03, DSGN-04, MOD-01, MOD-02, MOD-03, MOD-04, MOD-05, ARCH-01, ARCH-02, ARCH-03, ARCH-04, DATA-01, DATA-02, DATA-03, DATA-04, VERF-01, VERF-02, VERF-03, VERF-04, VERF-05, VERF-06, VERF-07, VERF-08, VERF-09, VERF-10]  
**UI hint**: no  
**Plans**: 5 plans

Plans:
- [x] 02-01: 建立 generate_grammar 基础模块和数据入口。
- [x] 02-02: 重实现 scoring 模块并对比原始模块行为。
- [x] 02-03: 实现 token 化 grammar 学习核心。
- [x] 02-04: 实现 structured 输出、pipeline 和运行脚本。
- [x] 02-05: 建立验证脚本、数据来源记录和完成验证报告。

### Phase 3: generateGrammar 顶层算法审计与优化
**Goal**: 对 Phase 2 已完成的 `generate_grammar` 模块进行算法级优化，从顶层学习流程和数据流开始审计，并通过过程和最终输出验证证明行为一致。  
**Depends on**: Phase 2  
**Requirements**: [OPT-01, OPT-02, OPT-03, OPT-04, OPT-05, OPT-06, OPT-07, OPT-08]  
**UI hint**: no  
**Plans**: 6 plans

Plans:
- [x] 03-01: 建立过程一致性基线和核心类型契约。
- [x] 03-02: 合并解析与概率统计。
- [x] 03-03: 数组化离散状态数据组织。
- [x] 03-04: 重整 GrammarLearner.learn 主循环。
- [x] 03-05: 重构 skip-gram 和输出适配边界。
- [x] 03-06: 全量回归验证和优化记录。

## 归档摘要

**目标：** 建立 LoPS 科研脚本重构流程，并用 `generateGrammar.py` 完成第一项端到端重构和算法优化。

**完成内容：**

1. 建立每轮重构的任务接收契约、目录职责和中文规划文档规则。
2. 对 `generateGrammar.py` 及其默认分支调用模块进行深度分析。
3. 将 `generateGrammar` 重构为 `src/LoPS/generate_grammar` 下的独立模块，正式代码不依赖旧项目代码或旧项目数据目录。
4. 将必要输入、旧基准和新输出整理到 `data/generate_grammar`。
5. 在 `script/generate_grammar` 下建立运行脚本和验证脚本，旧格式转换只保留在验证适配层。
6. 完成顶层算法优化，覆盖解析合并、离散数据数组化、候选评分边界整理、主循环梳理和 skip-gram trace 分离。
7. 通过单元测试、过程一致性测试、历史测试和 34 被试全量逐 key/value 一致性验证。

**归档文档：**

- `.planning/milestones/v1.0-ROADMAP.md`
- `.planning/milestones/v1.0-REQUIREMENTS.md`
- `.planning/milestones/v1.0-MILESTONE-AUDIT.md`
- `.planning/MILESTONES.md`

## 重构 Phase 模式

从 Phase 2 开始，每个新增重构 phase 都代表一项完整脚本重构，而不是把一次重构拆成多个 phase。若某个已完成重构模块需要算法级优化，可以新增独立优化 phase；优化 phase 必须先从顶层算法和数据流审计开始，再决定底层算法的保留、删除或修改。

每个重构 phase 应按以下顺序推进：

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

## 数据与验证原则

所有重构 phase 都必须遵守以下规则：

1. 新版本正式代码必须完全独立于旧版本代码和旧版本数据。
2. 运行、测试、验证需要使用的数据必须复制或整理到当前 LoPS 仓库的 `data/` 下。
3. 新版本设计阶段默认不考虑旧版本输出格式兼容；如果需要新旧结果比对，使用独立验证脚本或适配模块完成转换。
4. 目标脚本在本轮有效分支中调用到的本地模块默认属于同一轮重构范围。
5. 每个参与重构的调用模块都必须有模块级行为测试。
6. 模块级行为测试必须使用相同数据和相同随机参数，对比重构模块与原始模块结果一致。

## Progress（进度）

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. 项目骨架与任务接收契约 | v1.0 | 2/2 | Complete | 2026-05-03 |
| 2. 重构 generateGrammar 模块 | v1.0 | 5/5 | Complete | 2026-05-04 |
| 3. generateGrammar 顶层算法审计与优化 | v1.0 | 6/6 | Complete | 2026-05-05 |
| 4. 下一项重构 phase | v1.1 | 0/0 | Waiting for scope | - |
