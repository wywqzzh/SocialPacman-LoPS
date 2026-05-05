---
phase: 03-optimize-generateGrammar
plan: 03-04
subsystem: algorithm
tags: [generate_grammar, candidate-score, learn-loop]
requires:
  - phase: 03-03
    provides: ParsedSequence 和 DiscreteLearningData 数组化评分输入
provides:
  - CandidateScore 候选评分过程对象
  - _score_candidate_pair 候选评分函数
  - _select_next_chunk 候选选择函数
  - 删除 candidate_ratio_min 无效参数
affects: [generate_grammar, learning-loop]
tech-stack:
  added: []
  patterns: [candidate score row, explicit selection wrapper]
key-files:
  created: []
  modified:
    - src/LoPS/generate_grammar/config.py
    - src/LoPS/generate_grammar/grammar.py
    - tests/test_generate_grammar_process.py
key-decisions:
  - "candidate_ratio_min 未参与旧行为，直接删除而不是补齐新行为。"
  - "候选评分抽为 CandidateScore 行对象，但不改变 parent-child 遍历和过滤顺序。"
  - "pair_posterior 继续通过 bd_score(data_child, data_parent, 2, 2, 1) 获取。"
patterns-established:
  - "主循环保留顶层流程，候选评分细节收敛在 _score_candidate_pair。"
  - "候选选择包装函数继续委托 choose_candidate_chunks，避免重写旧筛选语义。"
requirements-completed: [OPT-02, OPT-04, OPT-05, OPT-06, OPT-07]
duration: 4 min
completed: 2026-05-05
---

# Phase 03 Plan 03-04: 重整 GrammarLearner.learn 主循环 Summary

**CandidateScore 候选评分边界和无效参数清理，保持候选评分语义不变**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-05T04:35:22Z
- **Completed:** 2026-05-05T04:39:19Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- 删除 `GrammarLearningParams.candidate_ratio_min`，确认 `src` 和 `tests` 中无引用残留。
- 新增 `CandidateScore`，显式记录候选的 parent、child、chunk、score、posterior、frequency 和 ratio。
- 抽出 `_score_candidate_pair()` 和 `_select_next_chunk()`，主循环更清晰地表达“组织离散数据 -> 评分候选 -> 选择 chunk -> 更新解析”流程。
- 增加候选评分过程测试，锁定单个候选的 score、posterior、frequency、ratio 和选择规则。

## Task Commits

1. **Task 1: 删除无效配置参数 candidate_ratio_min** - `f855e96`
2. **Task 2-3: 抽出候选评分和选择函数并重整主循环** - `b0e9042`
3. **Task 2-3: 增加候选评分过程测试** - `173875b`

## Files Created/Modified

- `src/LoPS/generate_grammar/config.py` - 删除无效参数 `candidate_ratio_min`。
- `src/LoPS/generate_grammar/grammar.py` - 新增候选评分数据类和私有函数，重整 `learn()` 候选评分段。
- `tests/test_generate_grammar_process.py` - 新增候选评分和候选选择过程测试。

## Decisions Made

- 不引入 `TransitionStats`，也不做 pair frequency 预筛选。
- `_select_next_chunk()` 保留现有 `choose_candidate_chunks()` 的 ratio 降序和 `candidate_ratio_keep` 规则。

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Verification

```bash
rg "candidate_ratio_min" src tests
PYTHONPATH=.:src /home/zzh/anaconda3/envs/LoPS/bin/python -m unittest tests.test_generate_grammar_process tests.test_generate_grammar_grammar tests.test_generate_grammar_foundation
PYTHONPATH=.:src /home/zzh/anaconda3/envs/LoPS/bin/python -m unittest discover -s tests
PYTHONPATH=src /home/zzh/anaconda3/envs/LoPS/bin/python script/generate_grammar/validate_generate_grammar.py --quiet
```

结果：

- `rg "candidate_ratio_min" src tests`: 无匹配。
- 局部测试：16 tests OK。
- `unittest discover -s tests`: 25 tests OK。
- 全量验证：`Validation passed for 34 files.`

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

03-05 可以在当前主循环稳定的基础上重构 `detect_skip_gram()`，重点验证 `N` 插入位置、posterior 和最终 skip-gram 结果。

---
*Phase: 03-optimize-generateGrammar*
*Completed: 2026-05-05*
