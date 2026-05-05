# Phase 3: generateGrammar 顶层算法审计与优化 - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-05T10:51:22+08:00
**Phase:** 03-generateGrammar 顶层算法审计与优化
**Areas discussed:** 核心内存数据模型、状态条件与候选评分策略、正式输出结构边界、优化范围和验收目标、参数清理与接口收敛

---

## 核心内存数据模型

| Option | Description | Selected |
|--------|-------------|----------|
| A. 改为结构化核心模型 | 核心 token 用 `tuple[str, ...]`，状态矩阵用 `np.ndarray + state_names`；输出层再格式化成 `"G-L"` 和可读结构。 | ✓ |
| B. 半步优化 | token 仍用 `"G-L"`，但缓存 `split/length/base_set`；状态矩阵改为 `np.ndarray`。 | |
| C. 保守优化 | 保留当前字符串 token 和 DataFrame，只局部减少 pandas 索引。 | |

**User's choice:** 用户先要求解释为什么 tuple 优于字符串；解释后选择 `1=A`。

**Notes:** 锁定为“内部 tuple，外部字符串”。`"G-L"` 仍作为输出格式保留；tuple 只用于核心算法内部，避免算法依赖展示分隔符并减少重复 `split_token()`。

---

## 状态条件与候选评分策略

| Option | Description | Selected |
|--------|-------------|----------|
| A. 保持每轮重算语义，只数组化实现 | 仍每轮学习状态条件，保证算法语义最接近当前结果；重点优化矩阵构建、token id、pair posterior 等热点。 | ✓ |
| B. 只做安全缓存 | 仅在解析序列完全相同的情况下复用条件结果；行为风险低，但收益可能有限。 | |
| C. 允许阶段化/降频重算 | 例如先学状态条件再多轮复用。可能更快，但更容易改变候选选择，需要额外实验论证。 | |

**User's choice:** `2=A`

**Notes:** 本阶段不改变状态条件学习语义，先把 DataFrame/pandas 热点改成数组化数据流。

---

## 正式输出结构边界

| Option | Description | Selected |
|--------|-------------|----------|
| A. 正式输出保留清晰最小结构 | 保留科研分析真正需要的 `source/parameters/grammar/parsed/skip_gram`，派生旧字段由验证适配器重建。 | ✓ |
| B. 保留当前所有结构化字段 | 包括 `position_grammar`、`state_features` 等，降低迁移风险，但输出仍有一部分历史痕迹。 | |
| C. 进一步压缩正式输出 | 只保存 grammar、parsed sequence、skip-gram 和必要元数据；其它都按需重算。 | |

**User's choice:** `3=A`

**Notes:** 正式输出不应因为旧字段验证而保留冗余派生结构；旧字段重建放在验证适配器。

---

## 优化范围和验收目标

| Option | Description | Selected |
|--------|-------------|----------|
| A. 先做单文件核心算法优化 | 不做并行化，先把 `GrammarLearner.learn()` 的数据流、矩阵构建、token 表示整理干净；全量验证必须完全一致。 | ✓ |
| B. 核心优化 + 批处理并行化 | 同时优化单文件和 34 文件全量运行 wall time。收益更明显，但计划复杂度更高。 | |
| C. 只设结构目标，不设性能目标 | 重点是代码更清晰，性能只观察不作为验收指标。 | |

**User's choice:** `4=A`

**Notes:** 全量并行化延后；本阶段先优化单文件核心算法，并记录性能观察。

---

## 参数清理与接口收敛

| Option | Description | Selected |
|--------|-------------|----------|
| A. 删除未生效或无明确用途参数 | 例如 `candidate_ratio_min` 如果不打算改变行为就删除，保持 KISS。 | ✓ |
| B. 补齐所有已暴露参数的实际行为 | 让 `candidate_ratio_min` 等真正参与逻辑。接口更完整，但会增加行为变化风险。 | |
| C. 先标记 deprecated，暂不删 | 保守过渡，后续阶段再清理。 | |

**User's choice:** `5=A`

**Notes:** 清理无效参数，避免为了兼容当前未使用接口而保留技术债。

---

## the agent's Discretion

- 具体文件拆分、函数命名、测试拆分和性能观察脚本由 planner 决定。
- planner 可以决定 tuple token 的类型别名位置，但不应引入复杂 token 类。
- planner 可以决定 pair posterior 的等价实现方式，但必须测试证明语义一致。

## Deferred Ideas

- 批处理并行化。
- 状态条件阶段化或降频重算。
- BD score 数学公式改写。
- `ghost4`、`needShuffle=True` 和默认路径外旧分支。
