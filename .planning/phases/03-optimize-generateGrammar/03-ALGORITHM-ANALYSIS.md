# Phase 3 算法分析报告：generateGrammar 顶层算法审计

## 分析目的

本报告是进入 Phase 3 discuss 前的前置分析。目标不是立即提出实现方案，而是先明确当前 `generate_grammar` 模块的功能、顶层流程、核心算法、数据规模、验证边界和性能热点，从而判断哪些问题值得在 discuss 阶段与用户确认。

本阶段分析对象是当前 LoPS 新实现，不再分析旧项目代码。旧格式输出只作为验证基准，通过 `script/generate_grammar/legacy_adapter.py` 进行转换比较。

## 当前代码结构

正式模块位于 `src/LoPS/generate_grammar/`：

| 文件 | 职责 | 优化相关性 |
|------|------|------------|
| `config.py` | 保存状态列、alpha、迭代阈值、候选过滤阈值和 skip-gram 参数 | 参数边界和未使用参数需要检查 |
| `data_io.py` | 读取 StrategySequence pickle，写出结构化结果 | 当前不是主要算法热点 |
| `state_graph.py` | 读取 StateGraph 中的 `G` 矩阵并转为条件状态索引 | 状态条件学习依赖该结构 |
| `token.py` | 提供 token 拆分、组合、长度、重叠判断 | 当前大量重复调用，且内部仍依赖字符串拆分 |
| `scoring.py` | 实现状态组合计数、BD score、状态条件连线学习 | BD score 本身不是当前最大耗时，但语义很关键 |
| `grammar.py` | 实现 grammar 学习主循环、最长匹配解析、概率统计、skip-gram 检测 | 核心优化对象 |
| `structured.py` | 构造新结构化输出 | 输出 schema 是否允许调整需要 discuss |
| `pipeline.py` | 单文件和全量运行编排 | 顶层流程优化入口 |

脚本位于 `script/generate_grammar/`：

| 文件 | 职责 |
|------|------|
| `run_generate_grammar.py` | 读取仓库内默认数据目录或命令行路径，运行新 pipeline |
| `validate_generate_grammar.py` | 运行新 pipeline，经适配器转换后与基准逐字段比较 |
| `legacy_adapter.py` | 只服务验证，将新结构映射到基准字段 |

测试位于 `tests/`，当前覆盖 token、配置、数据读取、状态图读取、scoring、核心学习、pipeline 和验证适配器。

## 当前功能边界

当前实现完成的功能是：对 34 个被试文件逐个读取策略序列和状态依赖图，删除 `N` 后学习 grammar chunk，再根据最终 chunk 序列检测 `N -> E-A` skip-gram，最后写出结构化 pickle。

当前默认运行条件：

- 输入策略序列目录：`data/generate_grammar/input/strategy_sequence`
- 输入状态图目录：`data/generate_grammar/input/state_graph`
- 输出目录：`data/generate_grammar/refactored-output/grammar`
- 状态列：`IS1, IS2, PG1, PG2, PE, BN5`
- `alpha = 0.5`
- 无随机过程
- 每个文件独立处理，全量运行只是按文件名排序后串行循环

当前正式输出顶层字段：

- `source`
- `parameters`
- `grammar`
- `parsed`
- `skip_gram`

旧字段 `sets/pro/gram/sequence/time_pro/frequency/seq/state/S/fileNames/components/skipGram/skipGramNum` 不在正式输出中直接生成，只由验证适配器转换得到。

## 数据规模

基于当前仓库内 `data/generate_grammar` 的 34 个输入文件统计：

| 指标 | 最小值 | 最大值 | 平均值 |
|------|--------|--------|--------|
| 原始序列长度 | 1897 | 7635 | 4137.82 |
| `N` 数量 | 100 | 542 | 300.47 |
| 删除 `N` 后长度 | 1797 | 7093 | 3837.35 |
| 初始 token 数 | 9 | 9 | 9.0 |
| 最终 grammar 数 | 12 | 14 | 13.12 |
| 最终解析序列长度 | 1103 | 4276 | 2313.68 |
| 最长 grammar 基础 token 数 | 2 | 4 | 2.91 |

`skip_gram.found=True` 的文件数为 17/34。

最大输入文件是 `141222-402.pkl`：

- 原始序列长度：7635
- 删除 `N` 后长度：7093
- 最终 grammar 数：13
- 最终解析序列长度：4276
- 最长 grammar 长度：3
- skip-gram：True

## 顶层 pipeline 逻辑

当前文件级入口是 `process_strategy_state_file()`：

1. 从 StrategySequence pickle 读取：
   - `seq`
   - `S`
   - `state[state_names]`
   - `fileNames`
2. 从同名 StateGraph pickle 读取 `G`，转换为每个状态对应的条件状态索引。
3. `prepare_strategy_state_data()` 删除序列中的 `N`，并同步删除状态表对应行。
4. 创建 `GrammarLearner`。
5. 调用 `GrammarLearner.learn()` 学习 grammar chunk。
6. 调用 `GrammarLearner.detect_skip_gram()` 检测 skip-gram。
7. 调用 `build_structured_output()` 组装结构化输出。

全量入口 `run_generate_grammar()` 只是排序枚举文件并逐文件执行。文件之间没有共享状态，所以全量流程天然可以串行、批处理或并行运行。是否进行并行化属于顶层设计问题，但它主要降低 wall time，不改变单文件算法复杂度。

## 核心学习主循环

`GrammarLearner.learn()` 的核心状态：

- `original_sequence`：删除 `N` 后的基础 token 序列。每轮新增 chunk 后都从它重新解析。
- `active_tokens`：当前可用 grammar token，初始为 9 个基础 token，后续追加新 chunk。
- `parsed_sequence`：当前 grammar 对 `original_sequence` 的最长匹配解析结果。
- `parsed_state_features`：与 `parsed_sequence` 对齐的状态特征，chunk 对齐到其覆盖片段的首个基础 token。
- `components`：每个 grammar token 的直接组成。
- `probabilities`：当前解析序列中各 active token 的出现概率。
- `kl_history`：用解析概率分布变化判断收敛。

每轮迭代执行：

1. `_organize_discrete_data()` 将当前解析序列转为 BD score 使用的离散矩阵：
   - `data_parent[token]`：上一个时间点是否为某 token，使用 1/2 编码。
   - `data_child[token]`：当前时间点是否为某 token，使用 1/2 编码。
   - `data_condition[state]`：当前时间点状态值 + 1。
   - `data_policy_condition[state]`：上一个时间点状态值 + 1。
   - 调用 `learn_state_condition_links()` 为每个 grammar token 学习应附加的状态条件。
2. 遍历 child token：
   - 排除 `V, 1, 2, N, S, e`。
   - 针对该 child 计算一次不加入 grammar parent 的 BD score。
3. 遍历 parent token：
   - 排除 parent 等于 child、排除 `V, N`。
   - 默认拒绝 parent 和 child 共享基础 token。
   - 计算加入 parent 后的 BD score。
   - 再调用一次 `bd_score(data_child, data_parent, 2, 2, 1)` 取得二值共现后验，用于频率过滤。
   - 通过独立概率乘积和最小频率阈值过滤弱候选。
   - 保存 `score_without_parent / score_with_parent` 作为候选 ratio。
4. `choose_candidate_chunks()` 选择 ratio 大于 1 且接近最佳 ratio 的候选。
5. 把被选中的 chunk 追加到 `active_tokens`。
6. `_parse_longest()` 从 `original_sequence` 重新最长匹配解析，更新 `parsed_sequence` 和状态对齐。
7. `_parse_probabilities()` 重新统计 grammar 概率，用 KL 均值判断是否收敛。

循环结束后再次调用 `_parse_probabilities()`，删除概率为 0 的 grammar，计算 `time_probabilities` 并返回 `GrammarLearningResult`。

## skip-gram 逻辑

`detect_skip_gram()` 在 grammar 学习完成后执行：

1. 根据最终 `parsed_sequence` 中每个 token 的基础长度，把删除前的 `N` 位置映射回解析序列。
2. 在解析序列中插回 `N`。
3. 对每个 `N`，检查后续第 2 到第 5 个非 `N` 位置是否为 `E-A`。
4. 构造 `N` 和 `E-A` 两个二值变量。
5. 比较无 parent 与加入 `N` parent 的 BD score。
6. 同时要求后验矩阵 `posterior[1, 1] / len(sequence_with_n) > 0.025`。

这里的 `posterior[1, 1]` 包含 Dirichlet 先验，不是纯粹原始计数。后续若把这个频率判断改成直接计数，必须显式补齐同样的先验语义，否则可能造成边界样本结果变化。

## 验证结果

本次分析前重新运行了当前测试和全量验证：

```bash
PYTHONPATH=.:src /home/zzh/anaconda3/envs/LoPS/bin/python -m unittest discover -s tests
```

结果：

- `Ran 18 tests in 2.127s`
- `OK`

```bash
PYTHONPATH=src /home/zzh/anaconda3/envs/LoPS/bin/python script/generate_grammar/validate_generate_grammar.py --quiet
```

结果：

- `Validation passed for 34 files.`

全量验证计时：

```bash
/usr/bin/time -f 'elapsed=%E user=%U sys=%S' env PYTHONPATH=src /home/zzh/anaconda3/envs/LoPS/bin/python script/generate_grammar/validate_generate_grammar.py --quiet
```

结果：

- `elapsed=0:32.46`
- `user=47.36`
- `sys=0.15`

## 性能和复杂度观察

对 3 个代表性文件做轻量插桩：

| 文件 | clean_len | elapsed | 迭代数 | 候选数累计 | 选中数累计 | BD score 调用 | `_organize_discrete_data` 时间 | `_parse_longest` 时间 | `_parse_probabilities` 时间 |
|------|-----------|---------|--------|------------|------------|---------------|-------------------------------|----------------------|-----------------------------|
| `031222-401.pkl` | 约 2000 级 | 0.696s | 4 | 8 | 4 | 346 | 0.531s | 0.086s | 0.035s |
| `141222-402.pkl` | 7093 | 1.676s | 3 | 10 | 5 | 248 | 1.334s | 0.193s | 0.098s |
| `041122-401.pkl` | 6508 | 1.535s | 3 | 11 | 6 | 259 | 1.213s | 0.177s | 0.093s |

对最大文件 `141222-402.pkl` 做 cProfile：

- 总函数调用：15,335,138 次。
- 总耗时：4.702s。
- `learn()` 累计耗时：4.689s。
- `_organize_discrete_data()` 调用 3 次，累计 3.775s。
- pandas `indexing.__getitem__` 调用 219,162 次，累计 2.518s。
- `_parse_longest()` 调用 2 次，累计 0.546s。
- `_parse_probabilities()` 调用 4 次，累计 0.305s。
- `scoring.bd_score()` 调用 644 次，累计 0.141s。
- `token_length()` 调用 368,482 次。
- `split_token()` 调用 733,020 次。

结论：

1. 当前最大热点不是 BD score 数学公式，而是 `_organize_discrete_data()` 中反复使用 pandas/DataFrame 行列访问并重建离散矩阵。
2. 最长匹配解析不是最大热点，但 token 拆分和长度计算调用次数非常高，说明当前字符串 token 表示对循环路径有持续开销。
3. 候选数量很小，3 个代表文件累计候选数只有 8 到 11 个；因此优化重点不应先放在复杂候选搜索结构上。
4. 单文件迭代数很低，通常 3 到 4 轮；`max_iterations=100000` 只是保护上限，不代表实际循环很深。
5. 文件之间没有依赖，并行批处理可以降低全量 wall time，但不会改善单文件算法，也会增加输出顺序和资源管理复杂度。

## 当前实现中的重要设计问题

### 1. 核心 token 仍是字符串表示

当前核心算法使用 `"G-L"` 这类字符串保存复合 token，并通过 `split_token()`、`token_length()`、`tokens_share_base_token()` 避免直接按字符处理。这个设计比旧版单字符占位符清晰，但本质上仍把复合 token 的内部结构编码在字符串中。

从算法优化角度，更直接的内部表示可能是 `tuple[str, ...]`，例如 `("G", "L")`、`("E", "A")`。正式输出时再格式化为 `"G-L"`。这样可以减少重复 split，并更符合“算法不能基于字符串结构判断 chunk”的要求。

需要 discuss：是否允许 Phase 3 把核心内部 token 表示改为 tuple，输出层继续保持 `"G-L"`。

### 2. 状态矩阵在核心算法中仍以 DataFrame 传递

当前 `state_features` 在核心学习中是 DataFrame。`_organize_discrete_data()` 每轮会通过 pandas 行列索引构造 parent、child、condition、policy_condition 多个 DataFrame。性能分析显示这是当前最大热点。

从算法角度，核心学习只需要：

- 状态矩阵值。
- 状态列名。
- 每一行与解析 token 的对齐关系。

因此可以考虑在 pipeline 边界把 DataFrame 转成 `np.ndarray` + `state_names`，核心算法只使用数组，输出层或验证适配层再恢复 DataFrame。

需要 discuss：正式核心结果是否仍必须直接保存 DataFrame，还是允许核心使用数组、输出层再组装可读结构。

### 3. 每轮重新学习状态条件链接

当前 `_organize_discrete_data()` 每一轮都会调用 `learn_state_condition_links()`。因为 parsed token 会随着新 chunk 改变，grammar parent 变量也会改变，所以这一点有算法意义，不能直接删掉。

但可以讨论三种层级：

1. 保留每轮重新学习，只把实现改成数组化。
2. 对相同解析状态下的条件结果做缓存。
3. 改为更激进的顶层策略，例如只在部分轮次重算，或将状态条件学习和 chunk 候选评分拆成两个阶段。

第三种可能改变算法行为，必须谨慎，并且需要更强验证。

### 4. pair frequency 当前通过 BD score 后验间接取得

候选过滤中调用 `bd_score(data_child, data_parent, 2, 2, 1)` 只是为了读取 `posterior[1, 1] / len(parsed_sequence)`。这个值包含先验，不是纯计数。如果替换为直接向量化共现计数，必须保持：

```text
posterior[1, 1] = raw_count + prior
```

其中 prior 由当前 `bd_score()` 的 alpha 和矩阵形状决定。

这是一个适合优化的点，但不能把它误写成普通频率。

### 5. `candidate_ratio_min` 参数当前未使用

`GrammarLearningParams.candidate_ratio_min` 默认值为 1.0，但当前筛选逻辑实际在 `choose_candidate_chunks()` 中硬编码 `ratio > 1`。这不是行为错误，因为默认值等于 1.0，但它说明参数暴露和实现之间不完全一致。

需要 discuss：Phase 3 是删除这个参数以保持 KISS，还是真正把它接入候选筛选接口。

### 6. `position_grammar` 的长度规则需要保留或重新定义

`_parse_probabilities()` 当前用最后一个 grammar token 的基础长度作为 `position_grammar` 的固定填充长度。这是为了保持当前验证输出一致，但从新结构化输出角度看并不直观。

需要 discuss：新版本正式输出是否还需要这个字段，或者是否把它改名为更明确的验证派生字段。若保留，必须解释其语义；若删除，验证适配器需要从更基础的信息重新构造旧字段。

### 7. 顶层批处理并行化不是第一优先级

34 个文件之间互不依赖，理论上可以并行处理。但当前单文件热点主要在 DataFrame 构造和解析循环，先优化单文件核心更能改善算法结构。并行化更像运行层优化，可能会引入进程池、日志顺序、异常聚合和资源占用问题。

需要 discuss：Phase 3 是否只做单文件算法优化，还是也纳入批处理并行化。

## 初步优化方向

以下只是讨论前的候选方向，不是最终计划。

1. 将核心 token 内部表示从字符串改为 tuple，输出层再格式化。
2. 将核心状态数据从 DataFrame 改为 `np.ndarray` + 状态列名，减少 pandas 索引。
3. 把 `_organize_discrete_data()` 拆成更明确的数组构建过程：
   - 解析 token id 序列。
   - parent id 为前一位 token id。
   - child id 为后一位 token id。
   - 状态条件矩阵直接来自 `state_matrix[1:] + 1`。
   - policy condition 矩阵直接来自 `state_matrix[:-1] + 1`。
4. 对 active token 建立稳定 id 映射，避免每轮构造 token 字符串列的 DataFrame。
5. 将 pair frequency 从 BD score 间接调用改为等价的直接后验单元计算。
6. 缓存 token 长度、基础 token 集合和 token 展开结果，或用 tuple 表示自然消除该缓存需求。
7. 先不改 `bd_score()` 公式，避免把数学验证风险和数据结构优化混在一起。
8. 保留验证适配器作为唯一旧格式转换入口，优化核心不引入旧格式字段。

## 进入 discuss 时应优先确认的问题

1. 是否允许核心内部 token 表示改为 `tuple[str, ...]`，而正式输出继续展示为 `"G-L"`？
2. 是否允许核心算法内部完全使用 `np.ndarray` 表示状态矩阵，输出层再恢复 DataFrame 或更清晰的结构？
3. 状态条件链接是否必须每轮重算，还是允许讨论缓存或阶段化学习？
4. Phase 3 的性能目标是什么：只要求结构更清晰，还是希望全量验证耗时有明确下降目标？
5. 新结构化输出中是否仍需要 `position_grammar`，如果保留，它的语义应如何命名和解释？
6. `candidate_ratio_min` 这类当前未真正生效的参数，是删除以保持简洁，还是接入实现以保留可配置性？
7. 批处理并行化是否纳入 Phase 3，还是先限定为单文件核心算法优化？
8. 验证标准是否仍为默认 34 个文件经适配器转换后逐 key/value 完全一致？

## 当前结论

当前实现已经实现了清晰的模块拆分和旧新一致性验证，但函数内部仍较多沿用旧算法的处理顺序：每轮重新构造完整 DataFrame、使用字符串 token 反复拆分、通过 BD score 间接获取二值共现后验。Phase 3 的优化应从 `GrammarLearner.learn()` 的顶层数据流开始，而不是先局部改 `bd_score()` 或某个小函数。

最有价值的设计方向是先重新定义核心内存数据模型：token 用结构化表示，状态用数组表示，输出格式放在边界层处理。在这个前提下，再决定 `_organize_discrete_data()`、最长匹配解析、候选频率计算和状态条件学习哪些保留、哪些改写、哪些删除。
