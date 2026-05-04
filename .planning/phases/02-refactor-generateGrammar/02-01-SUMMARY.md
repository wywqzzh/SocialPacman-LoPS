---
phase: 02-refactor-generateGrammar
plan: 02-01
subsystem: foundation
tags: [python, dataclass, pandas, unittest, generate-grammar]
requires:
  - phase: 02-refactor-generateGrammar
    provides: "02-DESIGN.md 中确认的模块、类和接口设计"
provides:
  - "generate_grammar package 骨架"
  - "显式路径和学习参数配置"
  - "token 组合与拆分工具"
  - "StrategySequence 和 StateGraph 只读数据入口"
  - "foundation unittest"
affects: [generate_grammar, phase-2]
tech-stack:
  added: []
  patterns: ["dataclass 配置对象", "只读外部数据加载器", "标准库 unittest"]
key-files:
  created:
    - src/LoPS/generate_grammar/__init__.py
    - src/LoPS/generate_grammar/config.py
    - src/LoPS/generate_grammar/token.py
    - src/LoPS/generate_grammar/state_graph.py
    - src/LoPS/generate_grammar/data_io.py
    - tests/test_generate_grammar_foundation.py
  modified: []
key-decisions:
  - "保留旧 fileNames 原始值为 participant_file_names，同时提供去后缀 participant_ids 给 structured 输出使用。"
  - "默认输出目录固定在 LoPS 仓库内，避免写入原项目 grammar2。"
patterns-established:
  - "生产模块不导入原项目代码，只读取外部 pickle 数据。"
  - "复合 grammar token 使用 G-L 形式，并通过 token.py 辅助函数操作。"
requirements-completed: [ANLY-01, ANLY-02, ANLY-03, ANLY-04, ANLY-05, DSGN-01, DSGN-02, DSGN-03, DSGN-04, MOD-01, MOD-02, MOD-03, ARCH-01, ARCH-02, ARCH-03, ARCH-04]
duration: 8 min
completed: 2026-05-04
---

# Phase 2 Plan 02-01: 建立 generate_grammar 基础模块和数据入口 Summary

**generate_grammar 基础包、显式配置、token 工具和外部 StrategySequence/StateGraph 只读数据入口**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-04T12:11:00Z
- **Completed:** 2026-05-04T12:19:20Z
- **Tasks:** 4
- **Files modified:** 6

## Accomplishments

- 创建 `src/LoPS/generate_grammar/` package 和默认路径配置，默认输出写入 LoPS 仓库内。
- 实现 `"G-L"` 形式 token 的拆分、组合、长度和基础 token 重叠判断。
- 实现 StrategySequence 与 StateGraph 的只读加载器，同时保留旧 `fileNames` 和去后缀被试名。
- 添加 foundation unittest，并在 conda `fmri` 环境中通过。

## Task Commits

Each task was committed atomically:

1. **Task 1: 创建 package 骨架和配置数据类** - `9acb7b3` (feat)
2. **Task 2: 实现 token 工具函数** - `4a1fdf7` (feat)
3. **Task 3: 实现状态图和 StrategySequence 数据读取** - `50e1489` (feat)
4. **Task 4: 添加基础模块 unittest** - `2f4b40c` (test)

## Files Created/Modified

- `src/LoPS/generate_grammar/__init__.py` - 暴露基础配置入口。
- `src/LoPS/generate_grammar/config.py` - 定义默认只读输入路径、输出路径和学习参数。
- `src/LoPS/generate_grammar/token.py` - 提供复合 grammar token 操作。
- `src/LoPS/generate_grammar/state_graph.py` - 读取旧 StateGraph 并转换状态依赖列表。
- `src/LoPS/generate_grammar/data_io.py` - 读取旧 StrategySequence 数据并写新输出 pickle。
- `tests/test_generate_grammar_foundation.py` - 覆盖基础配置、token 和数据读取行为。

## Decisions Made

- 旧 `fileNames` 不去后缀保存为 `participant_file_names`，用于后续 `legacy["fileNames"]` 精确兼容。
- 去掉 `.pkl` 后的被试名保存为 `participant_ids`，用于后续清晰的 `structured` 输出。

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `gsd-sdk query config-set workflow._auto_chain_active false` 当前版本不支持该内部 key；未影响手动执行，后续未自动推进。

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

基础模块已经可供 `02-02` 的 scoring 重实现使用。下一步执行 `02-02-PLAN.md`。

---
*Phase: 02-refactor-generateGrammar*
*Completed: 2026-05-04*
