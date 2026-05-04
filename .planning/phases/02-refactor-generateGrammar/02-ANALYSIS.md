# Phase 2 深度分析报告：generateGrammar.py

## 分析范围

- 目标脚本: `/home/zzh/project/Pacman/2.Pac-man/structre-learning/scripts/fmriDataProcess/generateGrammar.py`
- 默认入口: `main("ghost2", 0.5, False)`
- 本轮只分析默认入口实际走到的分支。
- `Type != "ghost2"`、`needShuffle == True`、`organize_data_skip_gram()`、`learnBayesNet_noparallelize()` 等默认入口未使用路径，本轮暂不纳入重构。
- 原项目目录只读；所有运行探测都在 LoPS 当前仓库的 sandbox 中完成。

## 结论摘要

`generateGrammar.py` 的核心功能是：读取每个 cluster 的策略序列和状态图，把原始策略序列中的基础 token 逐步组合成更长 grammar chunk，并为每个 cluster 输出 grammar 统计结果。随后它会基于原始序列中被移除的 `N` 位置做一次 skip-gram 检测，判断是否存在 `N -> EA` 关系，并把该信息写回输出 pickle。

本轮默认运行实际涉及的本地模块不止目标脚本本身，还包括：

- `src.bayesianScore`
- `src.Utils`

`src.condindepEmp` 会被 `src.bayesianScore` 导入，但默认运行实际调用链没有调用其中的函数。重构时应优先把 `BDscore`、`learnBayesNetBlock` 和 `Utils.count` 纳入同轮重构与模块级行为测试；`condindepEmp` 是否需要迁移，应在 discuss 中确认。

## 默认运行路径

入口位于 `generateGrammar.py` 末尾：

```python
main("ghost2", 0.5, False)
```

因此实际参数为：

- `Type = "ghost2"`
- `alpha = 0.5`
- `needShuffle = False`
- `dataType = "human"`，但函数体内未使用。

默认分支使用的路径是相对当前工作目录的路径：

- 输入序列: `../../../Monkey_Analysis/fmri_data_process/StrategySequence/`
- 输入状态图: `../../../Monkey_Analysis/fmri_data_process/StateGraph/`
- 输出 grammar: `../../../Monkey_Analysis/fmri_data_process/grammar2/`

这意味着原脚本存在当前工作目录假设。直接重构时必须消除这个隐式假设，改为显式传入输入目录和输出目录。

## 数据分析

本轮默认数据位于：

- `/home/zzh/project/Pacman/2.Pac-man/Monkey_Analysis/fmri_data_process/StrategySequence/`
- `/home/zzh/project/Pacman/2.Pac-man/Monkey_Analysis/fmri_data_process/StateGraph/`
- `/home/zzh/project/Pacman/2.Pac-man/Monkey_Analysis/fmri_data_process/grammar2/`

探测结果：

| 目录 | 文件数 | 总大小 | 用途 |
|------|--------|--------|------|
| `StrategySequence` | 34 | 18,518,857 bytes | 每个 cluster 的策略序列、状态表和原始文件名 |
| `StateGraph` | 34 | 3,396,325 bytes | 每个 cluster 的状态依赖图 |
| `grammar2` | 34 | 4,531,606 bytes | 原脚本已有输出，可作为一致性基准 |

`StrategySequence` 单个文件结构示例：

- `seq`: 字符串策略序列，包含 `G/L/1/2/A/E/N/S/V` 等 token。
- `S`: 初始 token 集合。
- `state`: `DataFrame`，示例形状为 `(2186, 19)`。
- `strategy`: `DataFrame`。
- `strategyLabel`: `Series`。
- `fileNames`: 原始来源文件名列表。

`StateGraph` 单个文件结构示例：

- `G`: 状态依赖矩阵，默认分支使用 6 个状态时形状为 `(6, 6)`。
- `stateNames`: 状态名列表。
- `data`: 状态数据矩阵。

默认分支最终使用的状态列是：

```python
["IS1", "IS2", "PG1", "PG2", "PE", "BN5"]
```

`BN10` 在代码中被赋值过两次，但最终被覆盖，不参与默认运行。

## 执行流程

### 1. main 读取输入

`main()` 创建 `Chunk()`，读取 `StrategySequence` 下的文件名，然后逐文件处理：

1. 用 `pd.read_pickle()` 读取策略序列文件。
2. 取出 `seq`、`S`、`state[stateNames]`。
3. 因为 `needShuffle=False`，不会执行 `random.shuffle()`。
4. 找到序列中所有 `N` 的位置，保存为 `indexN`。
5. 从 `seq` 中删除 `N`。
6. 从 `states` 中删除对应 `N` 行。
7. 读取同名 `StateGraph` 文件，将 `G` 的每一行中值为 1 的列转换为 `condition`。
8. 调用 `Chunk.Chunking()` 生成 grammar 结果并先写一次 pickle。
9. 重新打开刚写出的 pickle，调用 `Chunk.skip_gram()` 检测 `N -> EA`。
10. 写入 `skipGram` 和 `skipGramNum` 后，再次覆盖同一个输出 pickle。

### 2. Chunking 生成 grammar

`Chunking()` 是主要算法入口。

核心状态：

- `gramLen`: 记录每个内部 token 对应 grammar 的长度。
- `sequence`: 保留删除 `N` 后的原始序列，后续每轮都基于原始序列重新 parse。
- `S`: 当前 token 集合，会随着新 chunk 发现而增长。
- `aggregate_dict`: 内部占位符到 chunk 的映射。
- `place_set`: 可用内部占位符，排除了 `e/G/L/E/A/1/2/3/4/S/V/N`。

每轮循环做以下事情：

1. `organize_data()` 将当前 `seq` 和 `state` 转成 parent/child/condition 三类离散数据表。
2. `learnBayesNetBlock()` 根据状态图为每个 grammar child 选择相关状态条件。
3. 遍历候选 child `cr` 和 parent `cl`。
4. 排除默认不参与组合的 token：child 跳过 `V/1/2/N/S/e`；parent 跳过 `V/N`；相同 token 或原始字符集合有交集的组合也跳过。
5. 用 `BDscore()` 比较 “child 无 parent” 与 “child 有 parent/condition” 的得分。
6. 用 `U[1,1] / len(seq)` 过滤频率不足的候选：必须同时满足不低于 `P[i] * P[j]` 和 `0.05`。
7. 计算 `ratio = score2 / score1`。
8. `choice_max_n()` 只保留 ratio 大于 1，且与最大 ratio 至少达到 85% 的候选。
9. 将候选 chunk 映射到新的内部占位符，加入 `S`。
10. 用新的 grammar 集合对原始序列做最长匹配 parse。
11. 计算 grammar 分布的 KL 变化；如果最近 5 次平均 KL 不超过 `0.05`，停止。
12. 若没有候选、没有可用占位符或达到其它停止条件，也停止。

输出 pickle 中包含：

- `sets`: 展开后的 grammar 列表。
- `pro`: grammar 在 parse 序列中的出现比例。
- `gram`: 每个原始位置对应的 grammar。
- `sequence`: 删除 `N` 后的原始序列。
- `time_pro`: 按 grammar 长度加权后的时间占比。
- `frequency`: grammar 出现次数。
- `seq`: 用内部占位符表示的新序列。
- `state`: parse 后对齐的新状态表。
- `S`: 内部 token 集合。
- `fileNames`: 来源文件列表。
- `components`: 每个 grammar 的组成来源。

### 3. skip_gram 检测

`skip_gram()` 使用 `Chunking()` 的结果和原始 `N` 位置：

1. 根据内部 token 对应的 grammar 长度，重建删除 `N` 前后的相对位置。
2. 在新序列中把 `N` 插回对应位置。
3. 对每个 `N`，检查后续第 2 到第 5 个 token 中是否出现 grammar `EA`。
4. 构造二值变量 `N` 和 `EA`。
5. 用 `BDscore()` 比较 `EA` 无 parent 与 `EA` 以 `N` 为 parent 的得分。
6. 若 `score2 / score1 > 1` 且 `U[1,1] / len(newSeq) > 0.025`，则 `skipGram=True`，否则为 `False`。

## 默认运行实际调用的模块闭包

### 必须纳入重构和模块级行为测试

1. `generateGrammar.py`
   - `Tools.static_pro`
   - `Tools.choice_max_n`
   - `Tools.KL`
   - `Chunk.parse`
   - `Chunk.parse_pro`
   - `Chunk.deep`
   - `Chunk.get_cover_set`
   - `Chunk.organize_data`
   - `Chunk.skip_gram`
   - `Chunk.Chunking`
   - `getConditionGraph`
   - `main`

2. `src.bayesianScore`
   - `BDscore`
   - `learnBayesNetBlock`

3. `src.Utils`
   - `count`

### 默认入口导入但实际未调用

- `learnBayesNet_Option`
- `learnBayesNet`
- `learnBayesNet_noparallelize`
- `data_balance`
- `learnBayesNet_f`
- `condindepEmp.condindepEmp`
- `condindepEmp.BDscore`
- `organize_data_skip_gram`

这些函数是否纳入本轮重构，需要 discuss 确认。按照当前规则，默认应只重构本轮有效分支实际调用到的部分。

## 随机过程分析

默认运行 `main("ghost2", 0.5, False)` 不使用随机过程。

证据：

- `random.shuffle()` 只在 `needShuffle == True` 分支执行，默认 `False`。
- `np.random.choice()` 只出现在注释代码或 `data_balance()`；默认调用链没有调用 `data_balance()`。
- 全量 sandbox 运行的 34 个输出与原项目已有 `grammar2` 输出字节级一致，进一步说明默认运行是确定性的。

因此本轮一致性验证默认不需要人为设置随机种子。若后续用户要求支持 `needShuffle=True`，则必须重新讨论随机种子接口。

## 副作用与风险

### 写入副作用

原脚本默认会写入：

```text
../../../Monkey_Analysis/fmri_data_process/grammar2/
```

每个文件会被写两次：

1. `Chunking()` 中先写入 grammar 结果。
2. `main()` 中读取该结果，补充 `skipGram` 和 `skipGramNum` 后再次写回。

本轮不允许写原项目目录，因此后续重构和验证必须显式传入 LoPS 内的输出目录，或在 LoPS 内建立只读输入、可写输出的运行环境。

### 路径与导入风险

原脚本存在两个隐式环境假设：

1. 输入输出路径依赖当前工作目录，而不是脚本文件位置。
2. `src.bayesianScore` 依赖 `PYTHONPATH` 或特定工作目录设置。

在 sandbox 中直接运行复制后的脚本时，未设置 `PYTHONPATH` 会失败：

```text
ModuleNotFoundError: No module named 'src'
```

设置 `PYTHONPATH` 后运行成功。重构时应去掉这两个隐式假设：

- 输入目录、状态图目录、输出目录应显式参数化。
- LoPS 模块应使用正常包内导入，而不是运行时修改 `sys.path`。

### 文件顺序风险

原脚本使用 `os.listdir(fileFolder)`，未排序。逐文件输出本身以文件名命名，没有跨文件累计状态，因此 grammar pickle 内容不受处理顺序影响；但控制台日志顺序会依赖文件系统返回顺序。重构时可以使用排序获得稳定日志，但若要求严格复刻日志顺序，需要 discuss 确认。

## 受控运行记录

为避免写入原项目，本次在 LoPS 下构建了 sandbox：

```text
.planning/runs/2026-05-04-generateGrammar/sandbox/
```

sandbox 只复制了原脚本和依赖代码，输入数据通过符号链接只读引用，输出写入 sandbox 内的 `grammar2/`。

### 运行 1：直接运行失败

命令：

```bash
conda run -n fmri python generateGrammar.py
```

结果：

```text
ModuleNotFoundError: No module named 'src'
```

结论：原脚本依赖额外 Python 路径或工作目录约定。

### 运行 2：单文件 smoke run 成功

命令：

```bash
PYTHONPATH=<sandbox>/structre-learning conda run -n fmri python generateGrammar.py
```

只链接 `031222-401.pkl` 后运行成功，输出：

```text
['1', '2', 'A', 'E', 'G', 'L', 'S', 'V', 'EA', 'GL', 'LE', 'LG'] False
```

sandbox 输出与原项目 `grammar2/031222-401.pkl` 的 MD5 完全一致：

```text
b1284fb30cf5b4a6ab3fe6173913ddbf
```

### 运行 3：全量 34 文件运行成功

链接 34 个 `StrategySequence` 和 34 个 `StateGraph` 文件后，全量运行成功。

对比结果：

- sandbox 输出文件数: 34
- 原项目 `grammar2` 输出文件数: 34
- 缺失文件: 0
- 额外文件: 0
- MD5 完全一致: 34/34
- MD5 不一致: 0

全量输出中 `skipGram=True` 的文件数为 17：

```text
041222-401.pkl
051122-402.pkl
051122-501.pkl
071122-401.pkl
091122-401.pkl
101122-401.pkl
111122-401.pkl
131122-402.pkl
141222-402.pkl
151122-401.pkl
161122-401.pkl
161122-402.pkl
161122-404.pkl
231122-402.pkl
241122-402.pkl
301122-402.pkl
311022-501.pkl
```

全量输出中出现过的 grammar token：

```text
1, 2, A, AL, E, EA, EAG, EAGL, EAL, G, GL, L, LE, LEA, LG, S, SEA, V
```

## 重构设计初步建议

正式 plan 前还需要 discuss，但从分析结果看，重构应至少拆出以下高内聚模块：

1. 数据读取与路径配置
   - 显式接收 strategy sequence 目录、state graph 目录、输出目录。
   - 提供 cluster 文件发现策略，默认可排序。

2. 状态条件解析
   - 从 `StateGraph` 的 `G` 生成 condition 列表。

3. 离散数据组织
   - 从 sequence、grammar set、state 构建 child、parent、condition 数据。

4. Bayesian scoring
   - 迁移 `Utils.count`、`BDscore`、`learnBayesNetBlock`。
   - 为这三个函数建立模块级行为测试。

5. Grammar chunk 学习
   - 执行候选生成、BDscore 比较、chunk 选择、最长匹配 parse、KL 收敛判断。

6. Skip-gram 检测
   - 单独实现 `N -> EA` 检测和 `skipGramNum` 计算。

7. 运行入口
   - 在 `script/` 下提供明确参数的运行脚本。

## discuss 前必须确认的问题

1. 是否确认本轮只保留 `main("ghost2", 0.5, False)` 路径，完全舍弃 `ghost4` 和 `needShuffle=True`？
2. `src.bayesianScore` 中默认运行未调用的函数是否全部不迁移？
3. `src.condindepEmp` 当前只因 `bayesianScore` import side effect 被加载，默认路径未调用；是否确认不迁移？
4. 输出一致性的目标是 “pickle 内容语义一致” 还是要求 “pickle 文件字节级一致”？当前原始脚本在 sandbox 中能做到字节级一致。
5. 是否允许重构后对文件处理顺序排序，以获得稳定日志？这不会影响每个输出 pickle 的内容，但会改变控制台输出顺序。
6. `grammar2` 现有 34 个输出是否可以作为后续验证基准，还是每次验证都必须先在 sandbox 中重新运行原实现？
7. 输出 pickle 的字段名和结构是否必须完全保留，还是可以提供更清晰的新结构并额外导出兼容旧结构？
