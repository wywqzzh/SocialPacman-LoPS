---
phase: 03-optimize-generateGrammar
plan: 03-01
subsystem: testing
tags: [generate_grammar, process-baseline, tuple-token]
requires:
  - phase: 02-refactor-generateGrammar
    provides: generate_grammar 正式模块、验证适配器和 34 文件一致性基准
provides:
  - tuple token 最小辅助接口
  - ParsedSequence 解析结果契约
  - 解析、概率、离散矩阵和 pair posterior 过程基线测试
affects: [generate_grammar, phase-3-optimization]
tech-stack:
  added: []
  patterns: [tuple token 边界转换, process snapshot tests]
key-files:
  created:
    - tests/test_generate_grammar_process.py
  modified:
    - src/LoPS/generate_grammar/token.py
    - src/LoPS/generate_grammar/grammar.py
key-decisions:
  - "tuple token 作为核心内部契约，字符串继续作为边界展示格式。"
  - "ParsedSequence 只保存解析派生信息，不承载候选评分或 pair posterior。"
  - "过程测试先锁定当前行为，再允许后续算法重构。"
patterns-established:
  - "过程一致性测试使用内联快照值，避免依赖被测函数动态生成期望。"
  - "核心 tuple token 辅助函数与现有字符串辅助函数并存，便于后续逐步迁移。"
requirements-completed: [OPT-01, OPT-02, OPT-03, OPT-05, OPT-07]
duration: 16 min
completed: 2026-05-05
---

# Phase 03 Plan 03-01: 建立过程一致性基线和核心类型契约 Summary

**tuple token 边界接口、ParsedSequence 契约和 grammar 学习过程快照测试**

## Performance

- **Duration:** 16 min
- **Started:** 2026-05-05T04:09:00Z
- **Completed:** 2026-05-05T04:24:57Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- 新增 `GrammarToken`、`parse_token_string()`、`format_grammar_token()`、`combine_grammar_tokens()` 和 `grammar_tokens_share_base()`，为后续核心 tuple token 迁移建立最小接口。
- 在 `grammar.py` 中加入 `ParsedSequence` 数据类，但未接入学习主流程，保持当前行为不变。
- 新增 `tests/test_generate_grammar_process.py`，锁定 `_parse_longest()`、`_parse_probabilities()`、`_organize_discrete_data()` 和 `pair_posterior` 的关键过程指标。

## Task Commits

1. **Task 1: 加入 tuple token 的最小边界辅助函数** - `b96ce26`
2. **Task 2: 定义 ParsedSequence 但不改变学习主流程** - `68cbac2`
3. **Task 3: 建立关键过程一致性测试文件** - `1c5adf5`

## Files Created/Modified

- `src/LoPS/generate_grammar/token.py` - 新增核心 tuple token 边界转换和组合辅助函数。
- `src/LoPS/generate_grammar/grammar.py` - 新增 `ParsedSequence` 解析结果契约。
- `tests/test_generate_grammar_process.py` - 新增过程一致性快照测试。

## Decisions Made

- 按计划只建立类型契约和测试基线，不改变 `learn()`、解析、离散矩阵或 skip-gram 行为。
- pair posterior 测试明确区分 BD score 后验和平凡 raw count，防止后续误做纯频次预筛选。

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- 过程测试会触发当前 `learn_state_condition_links()` 在空状态依赖图上的 `RuntimeWarning`，但测试通过且该 warning 来自现有行为快照；后续计划会继续保留过程一致性约束。

## Verification

```bash
PYTHONPATH=.:src /home/zzh/anaconda3/envs/LoPS/bin/python -m unittest tests.test_generate_grammar_process
PYTHONPATH=.:src /home/zzh/anaconda3/envs/LoPS/bin/python -m unittest discover -s tests
PYTHONPATH=src /home/zzh/anaconda3/envs/LoPS/bin/python script/generate_grammar/validate_generate_grammar.py --quiet
```

结果：

- `tests.test_generate_grammar_process`: 4 tests OK。
- `unittest discover -s tests`: 22 tests OK。
- 全量验证：`Validation passed for 34 files.`

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

03-01 已提供过程基线，03-02 可以开始把 `_parse_longest()` 和 `_parse_probabilities()` 合并到一次 `ParsedSequence` 构建流程中。

---
*Phase: 03-optimize-generateGrammar*
*Completed: 2026-05-05*
