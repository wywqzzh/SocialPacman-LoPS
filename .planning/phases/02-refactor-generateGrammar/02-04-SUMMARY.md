---
phase: 02-refactor-generateGrammar
plan: 02-04
subsystem: pipeline-output
tags: [python, argparse, pickle, unittest, pipeline]
requires:
  - phase: 02-refactor-generateGrammar
    provides: "02-03 GrammarLearner 核心结果"
provides:
  - "legacy 输出兼容层"
  - "structured 输出构造"
  - "文件级 pipeline"
  - "命令行运行脚本"
  - "pipeline 单元测试"
affects: [generate_grammar, pipeline, legacy-output, structured-output, phase-2]
tech-stack:
  added: []
  patterns: ["legacy 适配层隔离旧占位符", "structured 清晰输出", "argparse 运行入口"]
key-files:
  created:
    - src/LoPS/generate_grammar/legacy.py
    - src/LoPS/generate_grammar/structured.py
    - src/LoPS/generate_grammar/pipeline.py
    - script/run_generate_grammar.py
    - tests/test_generate_grammar_pipeline.py
  modified: []
key-decisions:
  - "旧占位符生成只存在于 legacy.py，不回流到 grammar.py。"
  - "新 pickle 顶层固定为 legacy 和 structured 两个字典。"
  - "运行脚本默认输出到 LoPS 仓库内的 .planning/runs 路径。"
patterns-established:
  - "process_strategy_state_file() 负责单文件内存处理，不直接写文件。"
  - "run_generate_grammar() 负责枚举输入、写输出并返回路径列表。"
requirements-completed: [DATA-02, DATA-03]
duration: 11 min
completed: 2026-05-04
---

# Phase 2 Plan 02-04: 实现 legacy/structured 输出、pipeline 和运行脚本 Summary

**完成核心结果到可运行文件级流程的连接层，输出同时包含旧字段兼容结构和新结构。**

## Performance

- **Duration:** 11 min
- **Started:** 2026-05-04T12:39:20Z
- **Completed:** 2026-05-04T12:50:14Z
- **Tasks:** 4
- **Files modified:** 5

## Accomplishments

- 创建 `legacy.py`，按旧占位符顺序构造 `sets`、`seq`、`S`、`components` 等旧字段。
- 创建 `structured.py`，输出 `source`、`parameters`、`grammar`、`parsed` 和 `skip_gram`。
- 创建 `pipeline.py`，实现数据准备、单文件处理和全量运行。
- 创建 `script/run_generate_grammar.py`，支持显式传入输入路径、状态图路径、输出路径、基准路径、alpha 和最大迭代次数。
- 添加 pipeline 单元测试，验证代表性文件输出包含完整 `legacy` 与 `structured` 顶层结构。

## Task Commits

Each task was committed atomically:

1. **Task 1: 实现 legacy 兼容输出构造** - `5b0d8f1` (feat)
2. **Task 2: 实现 structured 输出构造** - `0373382` (feat)
3. **Task 3: 实现文件级 pipeline** - `aba4f21` (feat)
4. **Task 4: 添加运行脚本和 pipeline 测试** - `5fded25` (test)

## Files Created/Modified

- `src/LoPS/generate_grammar/legacy.py` - 旧字段兼容输出。
- `src/LoPS/generate_grammar/structured.py` - 新结构输出。
- `src/LoPS/generate_grammar/pipeline.py` - 单文件和全量 pipeline。
- `script/run_generate_grammar.py` - 命令行运行入口。
- `tests/test_generate_grammar_pipeline.py` - pipeline 行为测试。

## Verification

已通过：

```bash
PYTHONPATH=src conda run -n fmri python -m unittest tests.test_generate_grammar_pipeline
PYTHONPATH=src conda run -n fmri python script/run_generate_grammar.py --max-iterations 1 --output-dir .planning/runs/2026-05-04-generateGrammar/smoke-output
PYTHONPATH=src conda run -n fmri python -m unittest tests.test_generate_grammar_foundation tests.test_generate_grammar_scoring tests.test_generate_grammar_grammar tests.test_generate_grammar_pipeline
```

结果：

```text
Generated 34 files in .planning/runs/2026-05-04-generateGrammar/smoke-output
Ran 14 tests in 1.194s
OK
```

## Decisions Made

- `legacy` 字典字段插入顺序遵循 `LEGACY_FIELD_ORDER`。
- `structured` 不参与旧结果一致性判定，但由同一次核心结果稳定生成。
- `process_strategy_state_file()` 只返回内存结果，写文件由 `run_generate_grammar()` 统一处理。

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None.

## Next Phase Readiness

`02-05` 可以基于当前运行入口实现验证脚本、数据来源记录和全量一致性报告。

---
*Phase: 02-refactor-generateGrammar*
*Completed: 2026-05-04*
