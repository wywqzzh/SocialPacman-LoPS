# Phase 2: 重构 generateGrammar 模块 - Context

**Gathered:** 2026-05-04
**Status:** Ready for planning

<domain>
## Phase Boundary

本阶段交付 generateGrammar.py 脚本的完整重构：深度分析原始行为（包括依赖模块 bayesianScore.py）、制定重构方案、在 LoPS 中实现新模块、验证新旧输出完全一致。

**范围包括**：
- 分析目标脚本的功能、执行流程、依赖关系、输入输出和随机过程
- 设计模块边界、接口和数据路径
- 在 `src/LoPS/grammar_chunking/` 实现新模块
- 在 `script/` 创建运行脚本
- 使用相同输入和随机种子验证新旧输出一致性
- 清理 `src/LoPS/temp/` 中的临时验证代码

**范围不包括**：
- 重构 bayesianScore.py（属于外部项目）
- 复制输入数据到 LoPS 项目（只记录外部路径）
- 优化算法性能或改进算法逻辑
- 添加新功能或扩展原有功能

</domain>

<decisions>
## Implementation Decisions

### 模块拆分策略
- **D-01:** 采用三层拆分架构：数据层（加载/保存 pickle）、算法层（Chunking 核心逻辑）、工具层（BDscore、KL 散度等计算）
- **D-02:** bayesianScore.py 不纳入重构范围，新模块直接导入外部的 bayesianScore.py
- **D-03:** 模块结构为 `src/LoPS/grammar_chunking/`，包含 `__init__.py`、`data_loader.py`、`chunking.py`、`tools.py`

### 随机种子处理
- **D-04:** 当前执行路径 `main("ghost2", 0.5, False)` 中**无随机源被触发**（已通过代码追踪确认）
- **D-05:** `needShuffle=False` 时 generateGrammar.py 不调用 `random.shuffle()`
- **D-06:** bayesianScore.py 中的 `data_balance()` 随机调用已被注释掉，不在调用链中
- **D-07:** 新模块接口仍保留可选的 `random_seed` 参数，以支持未来可能的随机场景
- **D-08:** 验证阶段**不需要**在 `src/LoPS/temp/` 中创建带种子注入的副本（原始代码已是确定性的）

### 数据路径处理
- **D-08:** 新模块接口接收绝对路径参数（input_dir, state_dir, output_dir）
- **D-09:** 不复制输入数据到 LoPS 项目（34 个文件约 18MB），只记录外部路径
- **D-10:** 在 `script/run_generateGrammar.py` 中传入外部项目的绝对路径

### 接口设计
- **D-11:** 主函数签名：`chunking_pipeline(input_dir, state_dir, output_dir, state_names, alpha=0.5, random_seed=None)`
- **D-12:** 所有路径参数使用绝对路径，避免工作目录依赖
- **D-13:** 随机种子参数可选，默认 None（保持原始行为），验证时传入固定值

### 验证策略
- **D-09:** 由于原始代码是确定性的（无随机源），验证应该**完全一致**（pickle 字节级比较）
- **D-10:** 如果字节级比较失败，逐字段比较，对浮点数组使用 `np.allclose(rtol=1e-9, atol=1e-12)`
- **D-11:** 先验证 1 个文件（快速反馈），通过后验证全部 34 个文件
- **D-12:** 记录每个文件的验证结果和任何使用的数值容差
- **D-13:** 不需要创建 `src/LoPS/temp/` 临时代码（原始代码已是确定性的）

### Claude's Discretion
- 具体的类和函数命名（只要清晰表达意图）
- 内部数据结构的组织方式
- 日志和调试信息的详细程度
- 验证脚本的输出格式

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 项目约束
- `.planning/PROJECT.md` - 项目目标、核心价值、确认门和中文文档要求
- `.planning/REQUIREMENTS.md` - Phase 2 对应的所有需求（ANLY-*, DSGN-*, MOD-*, DATA-*, VERF-*）
- `.planning/ROADMAP.md` - Phase 2 目标和成功标准
- `CLAUDE.md` - 中文交流、目录职责、强制流程和质量要求

### 原始脚本和分析
- `/home/zzh/project/Pacman/2.Pac-man/structre-learning/scripts/fmriDataProcess/generateGrammar.py` - 目标脚本（664 行）
- `/home/zzh/project/Pacman/2.Pac-man/structre-learning/src/bayesianScore.py` - 关键依赖模块（372 行）
- `.planning/runs/2026-05-04-generateGrammar/intake.md` - 任务接收记录
- `.planning/runs/2026-05-04-generateGrammar/analysis.md` - 深度分析报告（功能、流程、依赖、随机源）

### 数据路径
- 输入序列：`/home/zzh/project/Pacman/2.Pac-man/Monkey_Analysis/fmri_data_process/StrategySequence/` (34 个 .pkl 文件)
- 状态图：`/home/zzh/project/Pacman/2.Pac-man/Monkey_Analysis/fmri_data_process/StateGraph/` (34 个 .pkl 文件)
- 原始输出：`/home/zzh/project/Pacman/2.Pac-man/Monkey_Analysis/fmri_data_process/grammar2/`

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **无现有代码资产**：`src/LoPS/` 目前为空，这是第一个正式模块
- **外部依赖可用**：bayesianScore.py 提供 BDscore、learnBayesNetBlock 等函数，可直接导入

### Established Patterns
- **Python 环境**：conda 环境 `fmri`，Python 3.10.16，numpy 2.0.1，pandas 2.3.0
- **数据格式**：pickle 序列化，包含 seq（字符串）、S（列表）、state（DataFrame）、fileNames（列表）
- **目录约定**：`src/LoPS/` 存放可复用模块，`script/` 存放运行入口，`src/LoPS/temp/` 仅用于验证

### Integration Points
- **外部模块导入**：需要将 `/home/zzh/project/Pacman/2.Pac-man/structre-learning` 添加到 sys.path
- **数据访问**：运行脚本需要读取外部项目的数据文件
- **输出位置**：新实现输出到 LoPS 项目内的临时目录，用于与原始输出对比

</code_context>

<specifics>
## Specific Ideas

### 已识别的随机源（已完成追踪）
1. **generateGrammar.py 第 623 行**：`random.shuffle(shuffleIndex)` - 当前调用 `needShuffle=False`，**不会触发**
2. **bayesianScore.py 第 24-26 行**：`np.random.choice()` 和 `random.shuffle()` 在 `data_balance()` 函数中 - **已被注释掉**，不在调用链中
3. **调用链确认**：`learnBayesNetBlock` → `BDscore`，不调用 `data_balance()`
4. **Utils.py 和 condindepEmp.py**：无随机调用
5. **结论**：当前执行路径 `main("ghost2", 0.5, False)` **完全确定性**，无随机源被触发

### 核心算法
- **Chunking 算法**：迭代式语法分块，使用 BDscore 评估组合合理性
- **收敛标准**：最近 5 次迭代的平均 KL 散度 <= 0.05
- **Skip-gram 分析**：检测 "N" 与 "EA" 的跳跃关系

### 验证关键点
- **浮点数精度**：BDscore 和 KL 散度涉及大量浮点运算，可能需要数值容差
- **迭代收敛**：如果存在随机性，收敛轮次可能不同，需要对比趋势而非绝对值
- **文件遍历顺序**：`os.listdir()` 顺序不保证，需要排序文件名列表

</specifics>

<deferred>
## Deferred Ideas

### 未来优化（不在本阶段）
- 算法性能优化（并行化、缓存等）
- 添加进度条或详细日志
- 支持其他数据格式（非 pickle）
- 批量验证报告生成工具

### 其他脚本重构
- Phase 3 及后续阶段将重构其他科研脚本
- 每个脚本作为独立的 Phase

</deferred>

---

*Phase: 2-重构 generateGrammar 模块*
*Context gathered: 2026-05-04*
