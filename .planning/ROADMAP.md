# 路线图：LoPS

## 概览

v1 的目标是把 LoPS 建成一套可重复的科研脚本重构流程：先建立任务接收和目录契约，再分析原始行为，随后形成需要用户确认的重构方案，确认后实施模块化和数据脚本整理，最后用一致性验证证明结果未变。

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions marked with INSERTED

- [ ] **Phase 1: 项目骨架与任务接收契约** - 定义每轮重构如何接收目标脚本、环境、数据和状态记录。
- [ ] **Phase 2: 原始行为分析与证据采集** - 建立原脚本功能、输入输出、依赖和随机性的分析记录。
- [ ] **Phase 3: 重构方案与确认门** - 形成模块边界、接口、数据路径和验证方案，并等待用户确认。
- [ ] **Phase 4: 模块化实现与运行入口** - 按确认方案实现 `src/LoPS` 模块、`script` 运行脚本和 `data` 数据组织。
- [ ] **Phase 5: 一致性验证与收尾记录** - 对比旧新输出，处理随机过程，清理临时代码并记录结论。

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
- [ ] 01-01: 定义每轮重构的任务接收记录格式。

**Wave 2** *(blocked on Wave 1 completion)*
- [ ] 01-02: 补齐项目目录职责和初始化说明。

### Phase 2: 原始行为分析与证据采集
**Goal**: 在任何实现修改前，完整理解目标脚本的功能、执行流程、依赖、输入输出和随机过程。
**Depends on**: Phase 1
**Requirements**: [ANLY-01, ANLY-02, ANLY-03, ANLY-04]
**UI hint**: no
**Success Criteria** (what must be TRUE):
  1. 原始脚本的功能和执行流程被记录到分析文档。
  2. 输入、输出、关键中间结果和副作用都有明确说明。
  3. 本地依赖、外部模块、数据文件和工作目录假设被列出。
  4. 随机过程和 seed 状态被识别并记录。
**Plans**: 3 plans

Plans:
- [ ] 02-01: 阅读目标脚本及依赖文件，记录执行流程。
- [ ] 02-02: 采集原始输入输出和数据依赖证据。
- [ ] 02-03: 分析随机过程、工作目录假设和验证风险。

### Phase 3: 重构方案与确认门
**Goal**: 在实现前给出可审阅的重构方案，并将用户确认作为进入代码修改的硬门。
**Depends on**: Phase 2
**Requirements**: [DSGN-01, DSGN-02, DSGN-03, DSGN-04]
**UI hint**: no
**Success Criteria** (what must be TRUE):
  1. 每个候选功能都有模块化必要性判断。
  2. 重构方案覆盖功能拆分、文件拆分、模块边界、接口、数据路径和验证方式。
  3. 用户确认方案前，没有业务实现代码被修改。
  4. 不适合模块化的逻辑有明确原因和替代整理方式。
**Plans**: 2 plans

Plans:
- [ ] 03-01: 形成模块化必要性评估和目标结构。
- [ ] 03-02: 产出重构方案并执行用户确认门。

### Phase 4: 模块化实现与运行入口
**Goal**: 按确认方案把功能迁移到清晰的模块、脚本和数据结构中，同时保留原始算法语义。
**Depends on**: Phase 3
**Requirements**: [MOD-01, MOD-02, MOD-03, MOD-04, DATA-01, DATA-02, DATA-03, DATA-04]
**UI hint**: no
**Success Criteria** (what must be TRUE):
  1. `src/LoPS` 中存在边界清晰的新模块或模块更新。
  2. 新模块通过显式参数接收路径、配置和必要 seed。
  3. `script` 中存在可以运行新实现的入口脚本。
  4. 必要数据被整理到 `data`，并记录来源和用途。
  5. 新实现不依赖未记录的当前工作目录或隐藏全局状态。
**Plans**: 3 plans

Plans:
- [ ] 04-01: 实现或更新 `src/LoPS` 模块。
- [ ] 04-02: 整理数据路径并记录数据来源。
- [ ] 04-03: 编写并运行 `script` 入口。

### Phase 5: 一致性验证与收尾记录
**Goal**: 用相同输入和随机种子比较旧实现与新实现，确认重构没有改变科研结果。
**Depends on**: Phase 4
**Requirements**: [VERF-01, VERF-02, VERF-03, VERF-04, VERF-05, VERF-06, VERF-07, VERF-08]
**UI hint**: no
**Success Criteria** (what must be TRUE):
  1. 原始实现和新实现使用相同输入完成运行。
  2. 无随机过程时不额外设置 seed。
  3. 有未固定随机过程时，`src/LoPS/temp` 临时旧副本和新实现使用同一 seed。
  4. 输出比较结果被记录，默认完全一致；若使用容差，记录原因和容差。
  5. 验证通过后，`src/LoPS/temp` 中本轮临时代码被删除。
  6. 完成记录说明运行方式、验证方式和一致性结论。
**Plans**: 3 plans

Plans:
- [ ] 05-01: 构建旧新实现运行对比。
- [ ] 05-02: 处理随机过程和输出比较。
- [ ] 05-03: 清理临时代码并记录验证结论。

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. 项目骨架与任务接收契约 | 0/2 | Not started | - |
| 2. 原始行为分析与证据采集 | 0/3 | Not started | - |
| 3. 重构方案与确认门 | 0/2 | Not started | - |
| 4. 模块化实现与运行入口 | 0/3 | Not started | - |
| 5. 一致性验证与收尾记录 | 0/3 | Not started | - |
