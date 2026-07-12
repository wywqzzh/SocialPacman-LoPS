# Codebase Concerns

> 历史快照：本文记录 2026-06-22 的审计结论，行数、测试范围、数据数量和流程缺口
> 可能已经变化。当前运行接口请以 `README.md`、`data/README.md` 和
> `docs/data_flow.html` 为准；本文只用于追溯当时的技术债判断。

**Analysis Date:** 2026-06-22

## Tech Debt

**动态策略拟合模块过大且承担多种职责：**
- Issue: `src/LoPS/dynamic_strategy_fitting.py` 约 1215 行，同时包含数据解析、邻接表读取、切段、GA 拟合、随机预测诊断、并行调度和文件写入；`script/06_dynamic_strategy_fitting.py` 只做薄 CLI 包装。
- Files: `src/LoPS/dynamic_strategy_fitting.py`, `script/06_dynamic_strategy_fitting.py`
- Impact: 后续修改 GA 参数、切段逻辑或随机种子策略时，容易误改同一文件中的其它阶段；单元测试也难以隔离算法、I/O 和并行行为。
- Fix approach: 保持现有输出不变，按职责拆为 `data_io`、`segments`、`fitting`、`randomness`、`batch` 等内部模块；先补固定输入的回归测试，再移动代码。

**人工规则修正仍是脚本级业务实现：**
- Issue: `script/07_revise_human_weight.py` 约 684 行，包含正式权重修正规则、策略编号、上下文提取、预测结果计算、批处理和 CLI；未放入 `src/LoPS/` 可复用模块。
- Files: `script/07_revise_human_weight.py`
- Impact: 其它调用方无法复用该阶段核心规则；测试只能通过脚本或私有函数间接覆盖，重构时容易把一次性运行逻辑和正式业务逻辑继续混在一起。
- Fix approach: 把纯规则函数和数据模型迁移到 `src/LoPS/` 下独立模块，脚本只保留参数解析和目录批处理；迁移前用 `data/06_weight_data/*.pkl` 到 `data/07_corrected_weight_data/*.pkl` 的一致性验证锁定输出。

**旧格式兼容逻辑仍散落在正式模块：**
- Issue: 多个正式模块保留旧字段或旧文件名兜底，例如 `src/LoPS/generate_grammar/data.py:75` 可读取旧字段 `G`，`src/LoPS/pacman_video/frame_renderer.py:962` 允许读取 `{subject}_merged_frame_data.pkl`，`src/LoPS/pacman_video/render_table.py:371` 会删除旧流程 CSV/JSON/二级目录。
- Files: `src/LoPS/generate_grammar/data.py`, `src/LoPS/pacman_video/frame_renderer.py`, `src/LoPS/pacman_video/render_table.py`
- Impact: 正式边界被旧数据形态拉宽，调用方不容易判断当前数据契约；后续新增格式时可能继续扩大兼容分支。
- Fix approach: 把旧格式读取和清理逻辑移到验证/迁移脚本；正式模块只接受当前 README 记录的新 schema，必要时提供显式 `legacy_adapter`。

**数据流程文档存在仓库路径漂移：**
- Issue: `data/README.md:32` 的运行说明仍写 `cd /home/zzh/project/LoPS`，当前仓库路径是 `/home/zzh/project/SocialPacman-LoPS`。
- Files: `data/README.md`
- Impact: 新用户直接复制命令会进入不存在或错误仓库；对路径约束审计也会产生噪音。
- Fix approach: 改为 `cd /home/zzh/project/SocialPacman-LoPS` 或用 `<repo-root>` 占位；文档路径更新不影响业务代码。

## Known Bugs

**`one_hot_direction()` 的类型检查顺序会把非字符串报成未知方向：**
- Symptoms: 传入非字符串方向时，`src/LoPS/dynamic_strategy_fitting.py:179` 和 `script/07_revise_human_weight.py:86` 先执行 membership 检查，再执行 `isinstance(value, str)`；例如 `np.nan` 会触发 `ValueError("未知方向：nan")` 而不是类型错误。
- Files: `src/LoPS/dynamic_strategy_fitting.py`, `script/07_revise_human_weight.py`
- Trigger: 上游 `action_dir` 清洗失败，或测试/调用方直接传入非字符串方向值。
- Workaround: 当前主流程在进入有效方向行前通常过滤 float/NaN；但函数自身的错误语义不稳定。

**并列方向随机诊断依赖全局 NumPy 随机状态：**
- Symptoms: `src/LoPS/dynamic_strategy_fitting.py:167` 的 `choose_max_direction()` 使用 `np.random.choice()`；`calculate_correct_rate()` 重复 100 次随机诊断，`calculate_is_correct()` 也随机选择预测方向。
- Files: `src/LoPS/dynamic_strategy_fitting.py`, `script/06_dynamic_strategy_fitting.py`
- Trigger: 直接调用 `DynamicStrategyFittingConfig(random_seed=None)`，或在同一进程中其它代码提前消费 NumPy 随机数。
- Workaround: CLI 默认 `--seed 20260610`；并行段落时 `script/06_dynamic_strategy_fitting.py:57` 默认在 `segment_workers > 1` 时启用段级派生 seed。

**全零或等值权重归一化可能产生 NaN：**
- Symptoms: `src/LoPS/dynamic_strategy_fitting.py:1078` 对非 stay 且 `np.sum(internal_weight) != 0` 的权重执行 `(weight - min) / (max - min)`；如果所有内部权重相等且和非零，分母为 0。
- Files: `src/LoPS/dynamic_strategy_fitting.py`
- Trigger: GA 返回所有 agent 相同的非零权重，或自定义配置/测试构造该结果。
- Workaround: 当前是否发生取决于真实 GA 输出；建议增加保护分支和固定输入测试后再修改。

## Security Considerations

**可替换 CSV 输入使用 `eval()` 解析坐标：**
- Risk: `script/08_extract_features_human.py:78`、`script/08_extract_features_human.py:118`、`script/08_extract_features_human.py:119` 对 `--constant-dir` 中 CSV 字段执行 `eval()`；如果常量 CSV 被替换为恶意内容，会执行任意 Python 表达式。
- Files: `script/08_extract_features_human.py`, `data/constant_data/adjacent_map_fmri.csv`, `data/constant_data/dij_distance_map_fmri.csv`
- Current mitigation: 默认 `--constant-dir` 指向当前仓库 `data/constant_data`；其它模块已使用 `ast.literal_eval` 或受控解析，例如 `src/LoPS/hierarchical_utility/model.py:390`。
- Recommendations: 将 `eval()` 替换为 `ast.literal_eval()`，并在解析后校验 tuple/list 长度与数值类型；替换后用现有 `data/08_feature_data` 和 `data/08_discrete_feature_data` 做一致性验证。

**大量 pickle 读取默认信任本地数据目录：**
- Risk: 主流程多处使用 `pd.read_pickle()` 或 `pickle.load()`，例如 `src/LoPS/calculate_utility/processing.py:441`、`src/LoPS/dynamic_strategy_fitting.py:1123`、`src/LoPS/generate_grammar/data.py:60`、`src/LoPS/pacman_video/frame_renderer.py:923`。
- Files: `src/LoPS/calculate_utility/processing.py`, `src/LoPS/dynamic_strategy_fitting.py`, `src/LoPS/generate_grammar/data.py`, `src/LoPS/pacman_video/frame_renderer.py`, `script/04_human_tile_data_preprocess.py`, `script/07_revise_human_weight.py`, `script/08_extract_features_human.py`, `script/09_human_fmri_data_preprocess.py`
- Current mitigation: README 和脚本默认均指向当前仓库 `data/`；未检测到 `.env`、密钥或凭证类文件。
- Recommendations: 明确文档约束“只读取可信 pickle”；对外部数据导入优先使用 CSV/Parquet/JSON 或先在隔离脚本中转换；不要把用户上传 pickle 直接交给正式流程。

**视频输出清理函数会递归删除匹配 subject 的旧目录：**
- Risk: `src/LoPS/pacman_video/render_table.py:371` 的 `_remove_obsolete_outputs()` 会删除 `output_dir / subject` 二级目录；若调用方传入过宽的 `output_dir` 或异常 `subject`，可能删除非预期旧产物。
- Files: `src/LoPS/pacman_video/render_table.py`
- Current mitigation: 删除路径通过 `Path` 拼接，subject 来自文件名/参数，默认输出目录是 `data/pacman_video/render_data`。
- Recommendations: 限制 subject 只允许安全文件名字符；删除前确认目标目录位于预期 output root 内，并把清理旧格式做成显式参数。

**硬编码 ffmpeg 兜底路径依赖外部系统布局：**
- Risk: `src/LoPS/pacman_video/video_renderer.py:26` 使用 `/usr/local/fsl/bin/ffmpeg` 作为 PATH 查找失败后的兜底。
- Files: `src/LoPS/pacman_video/video_renderer.py`
- Current mitigation: 优先使用 `shutil.which("ffmpeg")`；兜底只在该路径存在时启用。
- Recommendations: 改为 CLI 参数或环境变量配置 ffmpeg 路径；文档只说明 PATH 要求，不在正式代码内固定用户机器路径。

## Performance Bottlenecks

**动态拟合计算量高且有嵌套并行风险：**
- Problem: `script/06_dynamic_strategy_fitting.py` 同时提供文件级 `--workers` 和段落级 `--segment-workers`；默认 GA 参数为 population 100、iterations 500。若用户同时开大两个并行层，会产生大量进程和内存复制。
- Files: `script/06_dynamic_strategy_fitting.py`, `src/LoPS/dynamic_strategy_fitting.py`
- Cause: `src/LoPS/dynamic_strategy_fitting.py:963` 用 `ProcessPoolExecutor` 复制文件级 DataFrame 到段落 worker；`src/LoPS/dynamic_strategy_fitting.py:1194` 还支持文件级进程池。
- Improvement path: 在 CLI 中限制 `workers * segment_workers` 的默认上限；为段落拟合记录耗时和内存；必要时改为按文件或按段二选一并行。

**tile 补点在循环中反复 concat：**
- Problem: `script/04_human_tile_data_preprocess.py:405` 在每个 trial 内按插入点循环执行 `pd.concat()`。
- Files: `script/04_human_tile_data_preprocess.py`
- Cause: 为复现旧 `DataFrame.append` 行为逐次拼接；当一个 trial 插入点较多时会反复复制 DataFrame。
- Improvement path: 保持行序语义不变，先收集原行和插入行的排序键，最后一次性构建 DataFrame；修改前用 `data/04_tile_data` 到 `data/04_corrected_tile_data` 做完整一致性对比。

**特征提取重复读取常量表：**
- Problem: `script/08_extract_features_human.py:464` 和 `script/08_extract_features_human.py:465` 在每个输入文件处理时读取并解析同一组常量 CSV。
- Files: `script/08_extract_features_human.py`
- Cause: `process_one_file()` 为并行安全把常量读取放在文件级 worker 内。
- Improvement path: 单进程模式缓存常量；多进程模式可用 initializer 预加载常量，避免 34 个文件重复解析。

**视频渲染逐行绘制和按文件生成帧，缺少批量进度/断点策略说明：**
- Problem: `src/LoPS/pacman_video/frame_renderer.py:1011` 使用 `iterrows()` 逐帧渲染；`src/LoPS/pacman_video/video_renderer.py:138` 每个 game 先列出所有帧。
- Files: `src/LoPS/pacman_video/frame_renderer.py`, `src/LoPS/pacman_video/video_renderer.py`, `script/pacman_video/run_frame_renderer.py`, `script/pacman_video/run_video_renderer.py`
- Cause: 渲染是图片级 I/O 密集流程，当前更关注复现视觉样式。
- Improvement path: 保留确定性输出前提下补充断点续跑、已存在帧校验和可选并行；视频合成时对超大帧目录避免一次性持有不必要元数据。

## Fragile Areas

**科研结果一致性强依赖旧流程细节：**
- Files: `src/LoPS/hierarchical_utility/model.py`, `src/LoPS/hierarchical_utility/strategies.py`, `src/LoPS/calculate_utility/processing.py`, `src/LoPS/dynamic_strategy_fitting.py`, `script/07_revise_human_weight.py`
- Why fragile: 多处显式复现旧脚本规则，包括 tunnel 补丁、ghost 墙内坐标修正、临时 `pacman_dir`、临时 float ghost 状态、旧 9 维优化路径占位和手工策略修正规则。
- Safe modification: 每次只改一个阶段；使用相同输入重跑旧输出和新输出，默认要求完全一致；若容忍数值误差，必须记录原因、列名和容差。
- Test coverage: 当前 `tests/` 主要覆盖 `generate_grammar`、`raw_subject_data_to_frame_data`、`04_human_tile_data_preprocess`，缺少 `src/LoPS/dynamic_strategy_fitting.py`、`src/LoPS/hierarchical_utility/*`、`src/LoPS/calculate_utility/processing.py`、`script/07_revise_human_weight.py`、`script/08_extract_features_human.py` 和视频模块的系统回归测试。

**数据目录保存了完整阶段产物但缺少 manifest/checksum：**
- Files: `data/01_raw_subject_data`, `data/02_frame_data`, `data/03_preprocessed_frame_data`, `data/04_tile_data`, `data/04_corrected_tile_data`, `data/05_utility_data`, `data/06_weight_data`, `data/07_corrected_weight_data`, `data/08_feature_data`, `data/08_discrete_feature_data`, `data/09_strategy_sequence`, `data/10_state_dependency_graph_data`, `data/11_grammar`
- Why fragile: 当前每个主要阶段目录有 34 个 `.pkl` 文件，但没有记录生成命令、输入哈希、代码版本、随机 seed 和完成时间；重新跑部分阶段后很难判断上下游是否同源。
- Safe modification: 为每个阶段输出 `data/<stage>/MANIFEST.json` 或摘要 CSV，包含输入目录、输出文件数、seed、关键参数、代码 commit、文件 hash；不要放在 `.planning`。
- Test coverage: README 只提供 `find data -maxdepth 2 -type f -name "*.pkl" | wc -l` 数量检查，不能证明内容一致。

**`src/LoPS/temp/` 为空但历史计划要求临时旧实现验证：**
- Files: `src/LoPS/temp`, `.planning/preestimation_fmri_refactor_analysis.md`
- Why fragile: 计划文档写明验证时会在 `src/LoPS/temp/` 创建临时旧实现副本并清理；当前目录为空是好状态，但后续重构若忘记清理，会违反 AGENTS.md。
- Safe modification: 验证脚本退出时使用 `try/finally` 清理临时旧代码；把验证输出写入 `data/` 下专门目录，不写入 `.planning`。
- Test coverage: 未发现自动检查 `src/LoPS/temp/` 必须为空的测试或脚本。

## Scaling Limits

**默认并行数和文档命令可能超出普通机器资源：**
- Current capacity: `data/README.md` 示例多处使用 `--workers 34`，`script/06_dynamic_strategy_fitting.py` 示例还可组合 `--workers 8 --segment-workers 32`。
- Limit: 在 8-16 核或内存较小机器上，pickle DataFrame 和 GA worker 会导致内存压力、上下文切换和磁盘 I/O 放大。
- Scaling path: 文档提供“保守默认”和“高性能机器”两组命令；脚本运行时打印实际 `os.cpu_count()`、输入文件数和有效 worker 上限。

**pickle 阶段链路不适合增量/跨版本 schema 演进：**
- Current capacity: 当前主链路是 34 个 subject/session 的离线批处理。
- Limit: `.pkl` 文件不自描述 schema 版本；字段语义变化时只能靠代码和文档同步，难以做部分重算和跨版本读取。
- Scaling path: 在不破坏现有验证的前提下，新增阶段级 schema 版本、列清单和最小统计摘要；长期可评估 Parquet 保存中间表。

## Dependencies at Risk

**`scikit-opt` 与 multiprocessing 有导入期兼容补丁：**
- Risk: `src/LoPS/dynamic_strategy_fitting.py:860` patch `multiprocessing.set_start_method` 以兼容 `sko`，说明依赖存在全局副作用。
- Impact: 该补丁影响当前进程内后续 multiprocessing 行为；升级 `scikit-opt`、改变进程启动方式或在其它工具内嵌运行时，可能出现难以定位的并行问题。
- Migration plan: 把 GA 调用封装在更小边界内，补充单进程和多进程回归测试；评估固定依赖版本或替换为无导入期副作用的优化器。

**Python 版本锁定较窄：**
- Risk: `pyproject.toml` 要求 `>=3.10,<3.11`。
- Impact: 新环境若默认 Python 3.11/3.12，需要单独安装 3.10；未来依赖升级可能受限。
- Migration plan: 先在 CI 或本地矩阵测试 Python 3.11，再逐步放宽版本；科研输出一致性优先于版本扩展。

## Missing Critical Features

**缺少端到端一致性验证入口：**
- Problem: README 说明完整流程，`.planning/preestimation_fmri_refactor_analysis.md` 描述了验证计划，但仓库当前未发现统一脚本用于一键比较旧实现/新实现或比较现有基准输出。
- Blocks: 后续重构高风险阶段时，无法快速证明“同输入完全一致”；只能依赖人工运行和局部测试。

**缺少阶段产物质量门禁：**
- Problem: 当前文档建议用文件数量检查输出，未看到对关键列、dtype、行数、缺失值比例、seed、hash 的自动门禁。
- Blocks: 某阶段局部重跑后，可能生成文件数量正确但内容错位的数据，直到下游算法报错或结果漂移才发现。

**缺少 CI/测试运行约定文件：**
- Problem: 仓库有 pytest 测试文件，但未检测到 `pytest.ini`、GitHub Actions workflow 或覆盖率门槛配置。
- Blocks: 其它 agent 或开发者不知道每轮修改必须跑哪些测试；高风险科研一致性回归不会自动执行。

## Test Coverage Gaps

**权重拟合未见直接测试：**
- What's not tested: GA 拟合、随机 seed 派生、段落并行、vague/stay 标记、旧 9 维兼容占位列删除。
- Files: `src/LoPS/dynamic_strategy_fitting.py`, `script/06_dynamic_strategy_fitting.py`
- Risk: 随机路径或并行顺序改变会造成 `data/06_weight_data` 漂移。
- Priority: High

**utility 计算和 hierarchical strategy 未见直接测试：**
- What's not tested: tunnel 距离补丁、ghost 位置修正、临时 `pacman_dir`、临时 float 状态、随机扰动系数非 0 时的行为。
- Files: `src/LoPS/calculate_utility/processing.py`, `src/LoPS/hierarchical_utility/model.py`, `src/LoPS/hierarchical_utility/strategies.py`, `src/LoPS/hierarchical_utility/estimation.py`
- Risk: Q 列是下游拟合和策略判断基础，轻微漂移会级联影响多个阶段。
- Priority: High

**人工权重修正规则未见独立测试：**
- What's not tested: `revise_vague()`、`revise_approach()`、`revise_wrong_energizer()`、`process_trial()` 的边界段落和并列权重策略。
- Files: `script/07_revise_human_weight.py`
- Risk: 手工规则很多，且部分规则依赖 DataFrame 标签和旧段落区间；重构时容易出现 off-by-one 或标签/位置混淆。
- Priority: High

**特征提取与 fMRI 数据整形未见测试：**
- What's not tested: `script/08_extract_features_human.py` 的连续/离散特征、`script/09_human_fmri_data_preprocess.py` 的 ghost2 数据、formed 数据和 strategy sequence 生成。
- Files: `script/08_extract_features_human.py`, `script/09_human_fmri_data_preprocess.py`
- Risk: 特征编码错误会影响状态依赖图和 grammar 学习，但下游可能仍能运行。
- Priority: Medium

**视频渲染未见自动测试：**
- What's not tested: render table 对齐、旧字段兼容、frame renderer 必需列、ffmpeg 命令生成和临时 concat 文件清理。
- Files: `src/LoPS/pacman_video/render_table.py`, `src/LoPS/pacman_video/frame_renderer.py`, `src/LoPS/pacman_video/video_renderer.py`, `script/pacman_video/run_render_table.py`, `script/pacman_video/run_frame_renderer.py`, `script/pacman_video/run_video_renderer.py`
- Risk: 视频流程独立于主分析链路，数据列变化后可能静默渲染缺失层或只在人工查看时发现。
- Priority: Medium

## 数据迁移和验证风险

**当前数据链路已生成多阶段产物，但缺少“同一轮生成”的证据：**
- Issue: `data/` 下主流程各阶段目录大多有 34 个 `.pkl`，但没有 manifest 证明它们来自同一次代码版本和参数组合。
- Files: `data/README.md`, `data/01_raw_subject_data`, `data/02_frame_data`, `data/03_preprocessed_frame_data`, `data/04_tile_data`, `data/04_corrected_tile_data`, `data/05_utility_data`, `data/06_weight_data`, `data/07_corrected_weight_data`, `data/08_feature_data`, `data/08_discrete_feature_data`, `data/09_strategy_sequence`, `data/10_state_dependency_graph_data`, `data/11_grammar`
- Impact: 数据迁移或局部重跑时，可能把新旧参数生成的阶段产物混用。
- Fix approach: 每次重跑阶段都写入同目录 manifest；下游脚本读取前检查上游 manifest 的 stage、schema version、file count 和关键参数。

**`.planning` 中记录旧项目路径和验证计划，但不是运行数据存放地：**
- Issue: `.planning/preestimation_fmri_refactor_analysis.md:5` 记录旧项目绝对路径，`.planning/preestimation_fmri_refactor_analysis.md:208` 记录临时旧实现验证方案；这些是分析信息，不应被正式代码引用。
- Files: `.planning/preestimation_fmri_refactor_analysis.md`
- Impact: 后续 agent 可能误把 `.planning` 中的旧路径当作可运行依赖。
- Fix approach: 规划文档只作为历史分析；正式运行脚本只用当前仓库 `data/`，验证输入和输出放在 `data/` 专用目录。

**部分模块刻意保留旧行为，不能在无验证情况下“清理”：**
- Issue: `src/LoPS/calculate_utility/processing.py:256` 临时生成 `pacman_dir`，`src/LoPS/calculate_utility/processing.py:272` 临时转换 ghost 状态为 float，`src/LoPS/dynamic_strategy_fitting.py:239` 通过临时 Q 列复现旧 9 维搜索路径。
- Files: `src/LoPS/calculate_utility/processing.py`, `src/LoPS/dynamic_strategy_fitting.py`
- Impact: 看似冗余的兼容逻辑可能直接影响科研输出一致性；普通代码清理会破坏历史结果。
- Fix approach: 在 CONVENTIONS 或模块注释中标出“不可无验证删除”的兼容块；任何删除前必须做旧新输出对比。

## 需要用户确认的开放问题

**是否接受把 `script/07_revise_human_weight.py` 拆成正式模块？**
- Files: `script/07_revise_human_weight.py`, `src/LoPS/`
- Why it matters: 这会改变代码组织但不应改变输出；需要确认是否作为独立重构阶段处理。

**旧格式兼容要保留到什么范围？**
- Files: `src/LoPS/generate_grammar/data.py`, `src/LoPS/pacman_video/frame_renderer.py`, `src/LoPS/pacman_video/render_table.py`
- Why it matters: 若正式代码只支持当前 schema，旧字段读取应迁移到验证适配器；若仍需日常读取旧产物，则需要明确兼容承诺和测试。

**是否建立 `data/` 阶段 manifest 和基准 hash？**
- Files: `data/README.md`, `data/`
- Why it matters: 这是后续重构一致性验证的基础设施，会新增非业务数据说明文件。

**是否允许替换 `script/08_extract_features_human.py` 中的 `eval()`？**
- Files: `script/08_extract_features_human.py`
- Why it matters: 安全上应替换为 `ast.literal_eval()`，但必须确认当前输出完全一致后落地。

---

*Concerns audit: 2026-06-22*
