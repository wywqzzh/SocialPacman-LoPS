# Phase 3: generateGrammar 顶层算法审计与优化 - Context

**Gathered:** 2026-05-05T10:51:22+08:00
**Status:** Ready for planning

<domain>
## Phase Boundary

本阶段对 Phase 2 已完成的 `src/LoPS/generate_grammar` 模块做算法级优化。优化必须从顶层学习流程、内存数据模型、数据流和输出约束开始，而不是先局部修改某个底层函数。

本阶段包括：

- 基于 `.planning/phases/03-optimize-generateGrammar/03-ALGORITHM-ANALYSIS.md` 的顶层算法优化设计。
- 重新设计核心内存 token 表示和状态矩阵表示。
- 重写 `GrammarLearner.learn()` 内部数据流，使候选评分、状态条件组织、最长匹配解析和输出组装更直接。
- 保持当前正式代码完全独立于旧版本代码、旧版本数据目录和旧格式兼容逻辑。
- 保持旧格式转换只存在于验证适配器。
- 使用 34 个被试数据做全量一致性验证，默认要求经适配器转换后逐 key/value 完全一致。
- 记录优化前后的运行方式、验证方式、性能观察和一致性结论。

本阶段不包括：

- 支持 `ghost4`、`needShuffle=True` 或 Phase 2 已明确排除的旧代码分支。
- 将旧输出字段重新引入正式核心输出。
- 批处理并行化；本阶段先优化单文件核心算法。
- 改变 BD score 数学公式、状态条件学习语义或候选选择语义来追求速度。

</domain>

<decisions>
## Implementation Decisions

### 核心内存数据模型

- **D-01:** 核心算法内部 token 表示改为结构化模型，推荐使用 `tuple[str, ...]` 表示基础 token 序列，例如 `("G", "L")`、`("E", "A")`。
- **D-02:** 对外输出和人工可读展示继续使用 `"G-L"`、`"E-A"`、`"G-L-E-A"` 这种字符串形式；字符串是输出格式，不是核心算法的判断基础。
- **D-03:** 核心算法不得依赖 `"-"` 分隔符来判断 token 长度、组成或基础 token 重叠；这些语义应直接来自 tuple。
- **D-04:** 若 planner 认为需要类型别名，应保持简单，例如 `GrammarToken = tuple[str, ...]`；不要引入复杂 token 类，除非能显著降低真实复杂度。
- **D-05:** 状态矩阵在核心算法内部改为 `np.ndarray + state_names`，避免在主循环中反复使用 pandas 行列索引。
- **D-06:** DataFrame 可以保留在输入读取边界或输出组装边界；核心学习、状态条件组织和候选评分应以数组为主。

### 状态条件与候选评分策略

- **D-07:** 保持当前每轮重新学习状态条件链接的算法语义。因为 parsed token 会随新 chunk 改变，不能为了速度直接改成固定条件或降频重算。
- **D-08:** Phase 3 的第一层优化是把每轮离散矩阵构建数组化，减少 DataFrame 构造、pandas 索引和重复 token 解析。
- **D-09:** 不在本阶段引入阶段化状态条件学习、跨轮复用状态条件或行为可能变化的条件链接缓存。
- **D-10:** 候选 pair frequency 可以从当前 `bd_score(data_child, data_parent, 2, 2, 1)` 的间接后验读取改为等价的直接后验单元计算，但必须保留 Dirichlet 先验语义，不能误写为纯 raw count。
- **D-11:** `bd_score()` 的数学公式本阶段默认保留，不把公式改写和数据流优化混在同一风险面里。

### 正式输出结构边界

- **D-12:** 正式输出继续保持新结构分区：`source`、`parameters`、`grammar`、`parsed`、`skip_gram`。
- **D-13:** 正式核心输出不为旧格式兼容保留字段；旧字段只允许由 `script/generate_grammar/legacy_adapter.py` 或同类验证适配层重建。
- **D-14:** 如果 `parsed.position_grammar` 仅用于还原旧字段 `gram`，则 planner 应优先设计为从更基础的新结构信息重建，而不是继续作为核心结果字段存在。
- **D-15:** 如果确实需要保留位置级映射给后续科研分析，必须使用清晰的新语义命名并说明含义，不能因为旧字段需要而保留含混结构。
- **D-16:** 验证适配器可以包含旧格式符号映射、旧字段顺序和旧字段重建逻辑；这些逻辑不得进入 `src/LoPS/generate_grammar` 的核心算法层。

### 优化范围与验收目标

- **D-17:** Phase 3 先做单文件核心算法优化，不做批处理并行化。
- **D-18:** 主要优化对象是 `GrammarLearner.learn()` 的顶层数据流，以及它调用的离散矩阵组织、最长匹配解析、候选评分准备和结果组装。
- **D-19:** 全量运行仍处理 34 个文件；文件之间无共享状态这一事实可以记录，但不在本阶段实现进程池或并行调度。
- **D-20:** 验收必须包括当前单元测试和全量验证：34 个被试的新输出经验证适配器转换后，与基准逐 key/value 完全一致。
- **D-21:** 性能作为观察和改进目标记录，基准为分析报告中的全量验证 `elapsed=0:32.46`；本阶段不为了达到某个固定提速比例牺牲清晰度或一致性。

### 参数清理与接口收敛

- **D-22:** 删除未生效、无明确用途或只是为了“看起来可配置”而暴露的参数，优先保持 KISS。
- **D-23:** `candidate_ratio_min` 当前已暴露但未真正参与逻辑；若计划不改变当前行为，应删除该参数，而不是补齐新行为。
- **D-24:** 不采用“先 deprecated 暂不删”的保守拖延方式；本阶段允许清理不必要接口。
- **D-25:** 保留的参数必须能说明其对当前算法结果、验证或运行入口有实际影响。

### 验证约束

- **D-26:** Phase 3 不重新依赖旧项目代码或旧项目数据路径；所有输入、输出和基准仍使用当前仓库 `data/generate_grammar` 下的数据。
- **D-27:** 优化后的新结构输出必须能通过统一验证适配器映射到旧格式基准，并逐 key/value 完全一致。
- **D-28:** 若某个优化会改变浮点计算顺序、候选并列顺序、字典顺序或 pickle 结构，plan 必须先识别风险并设计针对性验证。
- **D-29:** 若验证失败，不能降低验证标准；应定位差异来源，并回到设计或实现修正。

### the agent's Discretion

- planner 可以决定具体文件拆分、辅助函数命名、类型别名位置和测试文件组织，只要遵守核心 tuple token、数组状态矩阵、正式输出独立于旧格式、单文件核心优先的决策。
- planner 可以决定是否把 `GrammarLearner` 内部拆成更小的私有方法，但不得引入过度抽象或复杂类层级。
- planner 可以决定如何实现等价 pair posterior 计算，但必须在测试中证明与当前后验语义一致。
- planner 可以设计性能观察脚本或 benchmark 片段，但不得把性能脚本混入正式核心模块。

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 项目与流程约束

- `.planning/PROJECT.md` — 项目核心价值、确认门、独立于旧版本、KISS、目录边界和 Phase 3 当前状态。
- `.planning/REQUIREMENTS.md` — Phase 3 的 `OPT-01` 到 `OPT-08` 算法优化需求。
- `.planning/ROADMAP.md` — Phase 3 目标、成功标准、优化 phase 边界和验证要求。
- `.planning/STATE.md` — 当前状态、Phase 2 已完成结论、Phase 3 待办和已锁定规则。
- `AGENTS.md` — 中文交流、目录职责、强制流程、注释规则和质量要求。

### Phase 3 前置分析

- `.planning/phases/03-optimize-generateGrammar/03-ALGORITHM-ANALYSIS.md` — 当前顶层算法、数据规模、热点、验证结果和待讨论问题；Phase 3 plan 必须以此为主要分析依据。
- `.planning/phases/03-optimize-generateGrammar/03-CONTEXT.md` — 本文件，记录本轮 discuss 锁定的实现决策。

### Phase 2 行为基准

- `.planning/phases/02-refactor-generateGrammar/02-CONTEXT.md` — Phase 2 重构范围、验证基准、输出兼容和架构原则。
- `.planning/phases/02-refactor-generateGrammar/02-ANALYSIS.md` — 原始 `generateGrammar.py` 默认路径行为、数据结构、调用闭包和随机过程。
- `.planning/phases/02-refactor-generateGrammar/02-DESIGN.md` — Phase 2 新模块设计和 token 表示背景。
- `.planning/phases/02-refactor-generateGrammar/02-VERIFICATION.md` — Phase 2 验证方式和一致性结论。

### 当前实现入口

- `src/LoPS/generate_grammar/grammar.py` — 核心学习算法，Phase 3 的主要优化对象。
- `src/LoPS/generate_grammar/pipeline.py` — 文件级数据流入口，负责读取、预处理、学习、skip-gram 和结构化输出编排。
- `src/LoPS/generate_grammar/scoring.py` — BD score、离散状态组合计数和状态条件链接学习。
- `src/LoPS/generate_grammar/token.py` — 当前字符串 token 辅助函数，Phase 3 可能替换为 tuple 内部模型。
- `src/LoPS/generate_grammar/structured.py` — 正式结构化输出组装边界。
- `script/generate_grammar/legacy_adapter.py` — 唯一旧格式验证适配入口。
- `script/generate_grammar/validate_generate_grammar.py` — 全量一致性验证入口。
- `tests/` — 当前 18 个测试的测试结构和行为约束。

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `GenerateGrammarConfig` 和 `GrammarLearningParams` 已提供集中配置入口，但需要清理未生效参数。
- `GrammarLearner` 已集中承载学习流程，适合作为 Phase 3 的核心重写边界。
- `bd_score()` 和 `learn_state_condition_links()` 已有稳定行为测试，本阶段应尽量把它们作为受保护的数学基础。
- `validate_generate_grammar.py` 已能全量运行并逐字段比较 34 个文件，是优化后的主验收入口。
- `legacy_adapter.py` 已隔离旧格式转换，后续旧字段重建应继续放在脚本层。

### Established Patterns

- 正式模块不读写旧项目路径，运行脚本从 `data/generate_grammar` 默认目录读取。
- 正式核心输出只表达 LoPS 新结构，旧字段转换不进入 `src/LoPS/generate_grammar`。
- 文档和注释使用中文；代码标识符按技术语境使用英文。
- 每个函数和类需要中文 docstring，关键逻辑处保留解释性中文注释。

### Integration Points

- `process_strategy_state_file()` 是单文件数据流入口；tuple token 和 ndarray 状态矩阵的转换应优先在该边界或其下游清晰完成。
- `GrammarLearner.learn()` 是单文件核心算法入口；Phase 3 plan 应围绕该函数的数据流重新设计。
- `build_structured_output()` 是正式输出边界；需要配合核心结果对象变化。
- `convert_generate_grammar_output_to_legacy()` 是旧格式验证边界；若正式输出删除 `position_grammar` 等派生字段，适配器需要从新结构重建旧字段。
- `tests/test_generate_grammar_*.py` 需要同步调整，既覆盖新内部模型，又保护旧新验证一致性。

### Current Hotspots

- `_organize_discrete_data()` 中 DataFrame 构造和 pandas 索引是当前最大热点。
- `split_token()` 和 `token_length()` 调用次数很高，说明字符串 token 在核心循环中造成重复解析。
- 候选数量和迭代数都较小，Phase 3 不应优先设计复杂候选搜索框架。

</code_context>

<specifics>
## Specific Ideas

- 用户选择讨论全部灰区。
- 用户最终选择：`1=A 2=A 3=A 4=A 5=A`。
- 对于核心 token 表示，用户追问为什么不继续用字符串；已解释并锁定：字符串仍是输出展示格式，tuple 只作为核心内部算法表示。
- 用户关注点不是表面搬运，而是真正分析函数内部是否有更高效、更直接的处理方式。
- 用户要求重构和优化仍遵守 KISS，不要过度工程化。

</specifics>

<deferred>
## Deferred Ideas

- 批处理并行化：34 个文件之间无依赖，未来可以作为运行层优化单独处理；Phase 3 先不纳入。
- 阶段化或降频状态条件学习：可能提升速度，但会改变算法语义；本阶段先不纳入。
- 更激进的 BD score 公式改写：本阶段保留数学公式，避免与数据流优化混合风险。
- 支持 `ghost4`、`needShuffle=True` 或默认路径外旧代码分支：仍然延后。

</deferred>

---

*Phase: 3-generateGrammar 顶层算法审计与优化*
*Context gathered: 2026-05-05T10:51:22+08:00*
