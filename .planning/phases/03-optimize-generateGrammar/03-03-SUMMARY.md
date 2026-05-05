---
phase: 03-optimize-generateGrammar
plan: 03-03
subsystem: algorithm
tags: [generate_grammar, ndarray, discrete-data, bd-score]
requires:
  - phase: 03-02
    provides: ParsedSequence 共享解析构建流程
provides:
  - DiscreteLearningData ndarray 离散矩阵结构
  - _organize_discrete_data 数组化实现
  - parent/child/condition 矩阵和状态依赖邻接矩阵过程测试
affects: [generate_grammar, scoring-inputs]
tech-stack:
  added: []
  patterns: [ndarray-core, named-row-accessors]
key-files:
  created: []
  modified:
    - src/LoPS/generate_grammar/grammar.py
    - tests/test_generate_grammar_process.py
key-decisions:
  - "状态矩阵在核心离散组织流程中转换为 ndarray，并通过 state_names 保持列语义。"
  - "DiscreteLearningData 使用 token_names/state_names 提供按名称取行接口，避免 DataFrame 高频索引。"
  - "每轮仍重新执行状态条件链接学习，没有引入缓存或降频重算。"
patterns-established:
  - "核心评分输入使用 ndarray，边界语义由 token_names 和 state_names 显式保存。"
  - "过程测试直接断言 ndarray 矩阵和 learned_state_adjacency。"
requirements-completed: [OPT-02, OPT-04, OPT-05, OPT-06, OPT-07]
duration: 6 min
completed: 2026-05-05
---

# Phase 03 Plan 03-03: 数组化离散状态数据组织 Summary

**ndarray 离散矩阵替代 DataFrame 高频索引，同时保持 BD score 输入过程一致**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-05T04:29:40Z
- **Completed:** 2026-05-05T04:35:22Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- 用 `DiscreteLearningData` 替代旧 `OrganizedGrammarData`，核心字段改为 `np.ndarray`。
- `_organize_discrete_data()` 直接从 `ParsedSequence`、`active_tokens` 和状态 ndarray 构建 parent/child/condition 矩阵。
- `GrammarLearner.learn()` 改为通过 `child_values()`、`parent_values()` 和 `condition_values()` 读取评分输入，候选评分顺序不变。
- 扩展过程测试，断言 parent、child、condition、condition_state 和 `learned_state_adjacency` 的快照。

## Task Commits

1. **Task 1-3: 定义 DiscreteLearningData 并数组化离散矩阵组织** - `e643dfd`
2. **Task 3: 增加数组化过程快照测试** - `b9b69eb`

## Files Created/Modified

- `src/LoPS/generate_grammar/grammar.py` - 新增 ndarray 离散学习数据结构，重写 `_organize_discrete_data()`。
- `tests/test_generate_grammar_process.py` - 更新过程测试以直接验证 ndarray 矩阵和状态依赖邻接矩阵。

## Decisions Made

- `data_parent` 和 `data_child` 使用 token 行、样本列布局，`data_condition` 使用 state 行、样本列布局。
- 保留 `token_names` 和 `state_names`，用轻量取行方法替代 DataFrame 列选择，避免位置语义不清。

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- 小型空依赖图过程测试仍会触发现有 `learn_state_condition_links()` 的 `RuntimeWarning`；这是 03-01 已记录的现有边界行为，未影响测试和全量验证。

## Verification

```bash
PYTHONPATH=.:src /home/zzh/anaconda3/envs/LoPS/bin/python -m unittest tests.test_generate_grammar_process tests.test_generate_grammar_scoring
PYTHONPATH=.:src /home/zzh/anaconda3/envs/LoPS/bin/python -m unittest discover -s tests
PYTHONPATH=src /home/zzh/anaconda3/envs/LoPS/bin/python script/generate_grammar/validate_generate_grammar.py --quiet
```

结果：

- `tests.test_generate_grammar_process tests.test_generate_grammar_scoring`: 8 tests OK。
- `unittest discover -s tests`: 23 tests OK。
- 全量验证：`Validation passed for 34 files.`

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

03-04 可以在 `ParsedSequence` 和 `DiscreteLearningData` 基础上重整 `GrammarLearner.learn()` 的候选评分和选择函数边界，并删除无效参数 `candidate_ratio_min`。

---
*Phase: 03-optimize-generateGrammar*
*Completed: 2026-05-05*
