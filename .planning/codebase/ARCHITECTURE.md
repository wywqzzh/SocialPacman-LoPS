<!-- refreshed: 2026-06-22 -->
# Architecture

**Analysis Date:** 2026-06-22

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                         脚本入口层                           │
├──────────────────────┬──────────────────────┬───────────────┤
│  主分析编号脚本       │  视频渲染脚本          │  文档/说明      │
│  `script/01...12.py` │  `script/pacman_video`│  `README.md`   │
└──────────┬───────────┴──────────┬───────────┴───────┬───────┘
           │                      │                   │
           ▼                      ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                         可复用模块层                         │
├──────────────────────┬──────────────────────┬───────────────┤
│ `src/LoPS/pacman_...`│ `src/LoPS/hier...`   │ `src/LoPS/...`│
│ 原始行为预处理        │ utility/strategy     │ grammar/graph │
└──────────┬───────────┴──────────┬───────────┴───────┬───────┘
           │                      │                   │
           ▼                      ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                         数据阶段层                           │
│      `data/00_raw_mat_data` → ... → `data/11_grammar`        │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                         展示辅助层                           │
│ `data/pacman_video/grammar_data` → render table → frames → mp4│
└─────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| raw mat 转换 | 读取 `data/00_raw_mat_data` 中 session 目录和 trial `.mat`，输出 subject 级逐帧原始表。 | `src/LoPS/pacman_preprocess/mat_to_raw_subject_data.py` |
| frame data 构建 | 从 raw subject 表生成统一 frame 表，生成 `frame_id`，过滤四鬼 trial，并保留视频渲染所需原始字段。 | `src/LoPS/pacman_preprocess/raw_subject_data_to_frame_data.py` |
| frame data 标准化 | 将 frame 表收敛为分析字段：`frame_id`、`DayTrial`、`game_id`、`Step`、位置、恐惧状态、beans、energizers。 | `src/LoPS/pacman_preprocess/frame_data_preprocess.py` |
| tile 数据预处理 | 从预处理 frame 表按 `sample_rate` 抽样 tile，修正隧道/ghost 位置，补齐 Pacman 缺失路径，追加 `action_dir` 和 `available_dir`。 | `script/04_human_tile_data_preprocess.py` |
| hierarchical utility 模型 | 读取地图常量，解析 frame/tile 状态，使用共享路径树估计 global/local/evade/approach/energizer/no_energizer Q。 | `src/LoPS/hierarchical_utility/model.py`, `src/LoPS/hierarchical_utility/strategies.py`, `src/LoPS/hierarchical_utility/estimation.py` |
| utility 计算流水线 | 在 corrected tile 表上计算 Q，修正不可用方向，归一化 Q，写出集中 utility 表。 | `src/LoPS/calculate_utility/processing.py` |
| 动态策略拟合 | 基于归一化 Q 和真实动作，把 trial 切分成 context 段，用遗传算法拟合策略权重。 | `src/LoPS/dynamic_strategy_fitting.py` |
| 权重规则修正 | 对拟合权重做业务规则修正，生成 `revised_normalized_weight`、`revised_prediction_correct`、`strategy`。 | `script/07_revise_human_weight.py` |
| 特征提取 | 从 corrected weight 表提取连续特征和离散状态特征，生成 grammar/状态图使用的状态列。 | `script/08_extract_features_human.py` |
| fMRI strategy sequence | 过滤 ghost2 数据，形成策略 one-hot，按近邻被试合并策略序列。 | `script/09_human_fmri_data_preprocess.py` |
| 状态依赖图 | 从 strategy sequence 的状态矩阵学习状态依赖图，并保存兼容输出。 | `src/LoPS/state_dependency_graph.py`, `src/LoPS/structure_learning.py` |
| grammar 学习 | 读取 strategy sequence 和状态依赖图，学习 grammar token/chunk、parsed sequence 和 skip-gram。 | `src/LoPS/generate_grammar/pipeline.py`, `src/LoPS/generate_grammar/grammar.py` |
| DividePerson 后处理 | 读取 grammar 输出，按 chunk 比例聚类被试并打印 JSON。 | `src/LoPS/generate_grammar/grammar_process.py`, `script/12_divide_person.py` |
| 视频 render table | 合并 raw frame data 与视频专用 grammar 标注，生成逐帧渲染表。 | `src/LoPS/pacman_video/render_table.py` |
| 视频帧渲染 | 用 Pillow 将 render table 行绘制为 Pacman 图片帧。 | `src/LoPS/pacman_video/frame_renderer.py` |
| 视频合成 | 用 `ffmpeg` 将图片帧目录合成为 `.mp4`。 | `src/LoPS/pacman_video/video_renderer.py` |

## Pattern Overview

**Overall:** 分阶段文件流水线 + 薄 CLI 入口 + 局部可复用科研模块。

**Key Characteristics:**
- 主分析流程按 `script/01_mat_to_raw_subject_data.py` 到 `script/12_divide_person.py` 的编号顺序组织，默认只读写 `data/` 下的阶段目录。
- 每个主数据阶段以同名 `{subject/session}.pkl` 在不同目录间传递，例如 `data/05_utility_data/031222-401-03-Dec-2022-1.pkl`。
- 可复用算法放在 `src/LoPS/`；仍保留在脚本中的业务逻辑集中在 `script/04_human_tile_data_preprocess.py`、`script/07_revise_human_weight.py`、`script/08_extract_features_human.py`、`script/09_human_fmri_data_preprocess.py`。
- 批处理入口优先处理目录级数据，内部按文件并行；输出摘要打印到命令行，数据产物写入 `data/`。
- 视频流程是独立展示链路，读取 `data/02_frame_data` 和 `data/pacman_video/grammar_data`，不参与 `data/11_grammar` 的生成。

## Layers

**CLI/脚本层:**
- Purpose: 暴露命令行参数、设置默认 `data/` 路径、将请求转给正式模块或执行脚本层流程。
- Location: `script/`
- Contains: `parse_args()`、`main()`、默认目录、并行 worker 参数、阶段摘要输出。
- Depends on: `src/LoPS/` 模块、`pandas`/`numpy`、少量 `sklearn`/`ffmpeg`。
- Used by: 人工运行、GSD 验证、数据重跑流程。

**预处理模块层:**
- Purpose: 将 MATLAB 原始行为数据整理为 Python 分析表。
- Location: `src/LoPS/pacman_preprocess/`
- Contains: `.mat` 读取、raw subject 拼接、frame 去重/排序/过滤、字段标准化。
- Depends on: `h5py`、`pandas`、`numpy`、`ProcessPoolExecutor`。
- Used by: `script/01_mat_to_raw_subject_data.py`、`script/02_raw_subject_data_to_frame_data.py`、`script/03_frame_data_preprocess.py`。

**策略 utility 模块层:**
- Purpose: 对 Pacman 状态估计多个策略的四方向 Q，并为策略拟合提供归一化输入。
- Location: `src/LoPS/hierarchical_utility/`, `src/LoPS/calculate_utility/`
- Contains: `MapData`、`FrameState`、`UtilityConfig`、共享路径搜索、Q 修正、Q 归一化、目录批处理。
- Depends on: `data/constant_data/adjacent_map_fmri.csv`、`data/constant_data/dij_distance_map_fmri.csv`。
- Used by: `script/05_calculate_utility.py`。

**策略拟合与状态特征层:**
- Purpose: 将 Q 和动作转成策略权重、修正策略编号、离散状态、strategy sequence。
- Location: `src/LoPS/dynamic_strategy_fitting.py`, `script/07_revise_human_weight.py`, `script/08_extract_features_human.py`, `script/09_human_fmri_data_preprocess.py`
- Contains: context 切分、遗传算法拟合、规则修正、特征提取、近邻被试合并。
- Depends on: `scikit-opt`、`sklearn.neighbors.NearestNeighbors`、地图常量表。
- Used by: `script/06_dynamic_strategy_fitting.py` 到 `script/09_human_fmri_data_preprocess.py`。

**结构学习与 grammar 层:**
- Purpose: 从离散状态和策略序列学习状态依赖图与 grammar chunk。
- Location: `src/LoPS/structure_learning.py`, `src/LoPS/state_dependency_graph.py`, `src/LoPS/generate_grammar/`
- Contains: BD score、PC skeleton、`StateDependencyGraph`、grammar token、chunk 候选评分、structured output、DividePerson 聚类输入。
- Depends on: `numpy`、`pandas`、`scipy.special.gammaln`。
- Used by: `script/10_state_dependency_graph.py`、`script/11_generate_grammar.py`、`script/12_divide_person.py`。

**视频展示层:**
- Purpose: 为 Pacman 行为和 grammar/strategy 标注生成可视化视频。
- Location: `src/LoPS/pacman_video/`, `script/pacman_video/`
- Contains: render table 对齐、Pillow 图片渲染、`ffmpeg` 视频合成。
- Depends on: `data/02_frame_data`、`data/pacman_video/grammar_data`、`Pillow`、`ffmpeg`。
- Used by: 视频调试与结果展示，不作为主分析输入。

**数据层:**
- Purpose: 保存原始输入、中间表、最终 grammar 和视频产物。
- Location: `data/`
- Contains: `00_raw_mat_data/` 到 `11_grammar/`、`constant_data/`、`pacman_video/`。
- Depends on: 文件系统；所有主流程产物为 `.pkl`，常量为 `.csv`。
- Used by: 所有脚本入口。

## Data Flow

### Primary Request Path

1. 用户在仓库根目录运行编号脚本，例如 `PYTHONPATH=src python script/05_calculate_utility.py`。
2. 脚本解析默认目录，例如 `script/05_calculate_utility.py` 默认读取 `data/04_corrected_tile_data` 并写入 `data/05_utility_data`。
3. 脚本构造配置对象或路径参数，并调用 `src/LoPS/` 中的目录级处理函数，例如 `process_calculate_utility_directory()`。
4. 目录级处理函数枚举 `.pkl` 文件，以文件为任务执行转换，必要时用进程池并行。
5. 单文件处理函数读取 `pandas.DataFrame` 或 `dict`，生成新对象后用 pickle 写回下游 `data/` 目录。
6. 下一个编号脚本读取上一阶段同名 `.pkl` 文件继续处理。

### 主分析流程

1. `script/01_mat_to_raw_subject_data.py` 调用 `convert_mat_root_to_raw_subject_data()`，把 `data/00_raw_mat_data/<session>/*.mat` 转成 `data/01_raw_subject_data/{session}.pkl`。样例输出是 49 列 raw subject DataFrame。
2. `script/02_raw_subject_data_to_frame_data.py` 调用 `convert_raw_subject_data_to_frame_data_dir()`，从 `data/01_raw_subject_data` 生成 `data/02_frame_data`，样例输出 48 列，包含 `frame_id`、`DayTrial`、`Step`、角色位置、方向、原始像素字段和奖励字段。
3. `script/03_frame_data_preprocess.py` 调用 `preprocess_frame_data_directory()`，将 `data/02_frame_data` 收敛为 `data/03_preprocessed_frame_data`，样例输出 11 列：`frame_id`、`DayTrial`、`game_id`、`Step`、`pacmanPos`、`ghost1Pos`、`ghost2Pos`、`ifscared1`、`ifscared2`、`beans`、`energizers`。
4. `script/04_human_tile_data_preprocess.py` 读取 `data/03_preprocessed_frame_data`，先写 `data/04_tile_data`，再结合 `data/constant_data/adjacent_map_fmri.csv` 写 `data/04_corrected_tile_data`；样例 corrected tile 输出 13 列，追加 `action_dir` 和 `available_dir`。
5. `script/05_calculate_utility.py` 读取 `data/04_corrected_tile_data` 和 `data/constant_data`，通过 `src/LoPS/calculate_utility/processing.py` 与 `src/LoPS/hierarchical_utility/` 写 `data/05_utility_data`；样例输出 28 列，追加 7 个 Q 列和 7 个 `*_Q_norm` 列。
6. `script/06_dynamic_strategy_fitting.py` 读取 `data/05_utility_data`，调用 `process_dynamic_strategy_directory()` 写 `data/06_weight_data`；样例输出 37 列，追加 `weight`、`normalized_weight`、`prediction_correct`、`predict_dir`、`trial_context`、`eat_energizer`、`eat_ghost`、`is_stay`、`is_vague`。
7. `script/07_revise_human_weight.py` 读取 `data/06_weight_data` 并写 `data/07_corrected_weight_data`；样例输出 40 列，追加 `revised_normalized_weight`、`revised_prediction_correct`、`strategy`。
8. `script/08_extract_features_human.py` 读取 `data/07_corrected_weight_data` 和 `data/constant_data`，写 `data/08_feature_data` 与 `data/08_discrete_feature_data`；样例连续特征 17 列，离散特征 16 列，核心状态列包括 `PG1`、`PG2`、`PE`、`BW10`、`BB10`、`IS1`、`IS2`。
9. `script/09_human_fmri_data_preprocess.py` 读取 `data/08_discrete_feature_data`，写 `data/09_fmri_discrete_feature_data_ghost2`、`data/09_fmri_formed_data_ghost2` 和 `data/09_strategy_sequence`；样例 strategy sequence 是 dict，包含 `seq`、`S`、`state`、`strategy`、`strategyLabel`、`fileNames`。
10. `script/10_state_dependency_graph.py` 读取 `data/09_strategy_sequence`，调用 `process_state_dependency_graph_directory()` 写 `data/10_state_dependency_graph_data`；样例输出 dict 包含 `state_names`、`state_matrix`、`adjacency_matrix`。
11. `script/11_generate_grammar.py` 读取 `data/09_strategy_sequence` 和 `data/10_state_dependency_graph_data`，调用 `run_generate_grammar()` 写 `data/11_grammar`；样例输出 dict 包含 `source`、`parameters`、`grammar`、`parsed`、`skip_gram`。
12. `script/12_divide_person.py` 读取 `data/11_grammar`，调用 `load_divide_person_records()` 和 `divide_person()`，只在命令行打印 JSON，不写入 `data/`。

**State Management:**
- 主流程没有服务端持久状态；状态由每个阶段的 `.pkl` 文件承载。
- `src/LoPS/hierarchical_utility/estimation.py` 在行级 chunk 并行时使用 worker 全局缓存保存编译后的地图与配置。
- `src/LoPS/dynamic_strategy_fitting.py` 为兼容 `sko` 对 `multiprocessing.set_start_method` 做安全补丁，影响当前 Python 进程。

### 视频流程

1. `script/pacman_video/run_render_table.py` 调用 `find_subject_paths()` 和 `process_subject()`，读取 `data/02_frame_data` 与 `data/pacman_video/grammar_data`，输出 `data/pacman_video/render_data/{subject-session}.pkl`。
2. `script/pacman_video/run_frame_renderer.py` 调用 `load_render_rows()` 和 `PacmanRenderer.render()`，读取 render table 并输出 `data/pacman_video/frame_images/<subject>/<game>/frame_*.jpg`。
3. `script/pacman_video/run_video_renderer.py` 调用 `find_ffmpeg()`、`find_game_dirs()` 和 `build_game_video()`，读取图片帧目录并写 `data/pacman_video/video_data/<subject>/*.mp4`。

## Key Abstractions

**MapData / CompiledMapData:**
- Purpose: 保存 `adjacent_map_fmri.csv` 和 `dij_distance_map_fmri.csv` 的解析结果，并将位置映射编译为适合路径搜索的数组。
- Examples: `src/LoPS/hierarchical_utility/model.py`
- Pattern: 不在策略函数中反复读 CSV；由入口加载一次，目录/行级处理复用。

**FrameState / CompiledFrameState:**
- Purpose: 将 DataFrame 行中的 Pacman、ghost、beans、energizers、方向等字段转成结构化状态。
- Examples: `src/LoPS/hierarchical_utility/model.py`
- Pattern: I/O 表格结构和搜索算法输入隔离，字段兼容与隧道位置修正在解析层完成。

**SharedPathUtilityEngine:**
- Purpose: 一次共享路径树遍历同时估计 local、evade、approach、energizer、no_energizer 等路径型策略。
- Examples: `src/LoPS/hierarchical_utility/strategies.py`
- Pattern: 几何展开共享，策略 reward/risk 独立更新。

**CalculateUtilityConfig / DynamicStrategyFittingConfig / GenerateGrammarConfig:**
- Purpose: 保存阶段参数，避免脚本层硬编码散落在算法内部。
- Examples: `src/LoPS/calculate_utility/processing.py`, `src/LoPS/dynamic_strategy_fitting.py`, `src/LoPS/generate_grammar/config.py`
- Pattern: CLI 参数构造 config，对 DataFrame/目录处理函数传入 config。

**StateDependencyGraph:**
- Purpose: 表示每个状态变量依赖哪些条件状态下标。
- Examples: `src/LoPS/structure_learning.py`, `src/LoPS/state_dependency_graph.py`
- Pattern: 通用结构学习算法和 Pacman strategy sequence 边界分离。

**GrammarLearner:**
- Purpose: 基于策略序列、离散状态和状态依赖图学习 grammar chunk。
- Examples: `src/LoPS/generate_grammar/grammar.py`, `src/LoPS/generate_grammar/pipeline.py`
- Pattern: `pipeline.py` 负责文件和结构化输出，`grammar.py` 负责学习过程。

**SubjectPaths / ProcessingSummary:**
- Purpose: 保存视频 render table 阶段的输入输出路径和处理摘要。
- Examples: `src/LoPS/pacman_video/render_table.py`
- Pattern: 视频处理的路径发现和数据转换分离。

## Entry Points

**主流程入口 01-03:**
- Location: `script/01_mat_to_raw_subject_data.py`, `script/02_raw_subject_data_to_frame_data.py`, `script/03_frame_data_preprocess.py`
- Triggers: 命令行运行。
- Responsibilities: 从 `.mat` 到 raw subject，再到 frame，再到分析用 frame。

**主流程入口 04:**
- Location: `script/04_human_tile_data_preprocess.py`
- Triggers: 命令行运行。
- Responsibilities: 从 preprocessed frame 抽样 tile，修正 tile 路径和动作字段。

**主流程入口 05-06:**
- Location: `script/05_calculate_utility.py`, `script/06_dynamic_strategy_fitting.py`
- Triggers: 命令行运行。
- Responsibilities: 计算策略 Q 并拟合动态策略权重。

**主流程入口 07-09:**
- Location: `script/07_revise_human_weight.py`, `script/08_extract_features_human.py`, `script/09_human_fmri_data_preprocess.py`
- Triggers: 命令行运行。
- Responsibilities: 修正策略权重，抽取连续/离散特征，形成 ghost2 strategy sequence。

**主流程入口 10-12:**
- Location: `script/10_state_dependency_graph.py`, `script/11_generate_grammar.py`, `script/12_divide_person.py`
- Triggers: 命令行运行。
- Responsibilities: 学习状态依赖图、生成 grammar、做人群划分后处理。

**视频入口:**
- Location: `script/pacman_video/run_render_table.py`, `script/pacman_video/run_frame_renderer.py`, `script/pacman_video/run_video_renderer.py`
- Triggers: 命令行运行。
- Responsibilities: 生成 render table、图片帧和 MP4。

## Architectural Constraints

- **Threading:** 无服务常驻线程；批处理使用 `ProcessPoolExecutor` 或 `multiprocessing.Pool`。相关文件包括 `src/LoPS/pacman_preprocess/mat_to_raw_subject_data.py`、`src/LoPS/pacman_preprocess/raw_subject_data_to_frame_data.py`、`src/LoPS/pacman_preprocess/frame_data_preprocess.py`、`src/LoPS/calculate_utility/processing.py`、`src/LoPS/dynamic_strategy_fitting.py`、`script/07_revise_human_weight.py`、`script/08_extract_features_human.py`。
- **Global state:** `src/LoPS/hierarchical_utility/estimation.py` 使用 worker 全局变量缓存 compiled map/config；`src/LoPS/dynamic_strategy_fitting.py` 会补丁 `multiprocessing.set_start_method`。
- **Circular imports:** 未检测到明确循环导入；主要依赖方向是 `script/` → `src/LoPS/*`，`generate_grammar` → `structure_learning`，`state_dependency_graph` → `structure_learning`。
- **路径约束:** 正式模块不保存旧项目绝对路径；脚本默认路径均指向本仓库 `data/`。新增正式模块保持这一边界，默认数据路径放在 `script/`。
- **数据格式:** 主流程以 pickle DataFrame/dict 为阶段边界；常量数据以 CSV 为边界；视频最终以 JPG/MP4 为边界。
- **样本数量:** 主流程多数阶段当前有 34 个 `.pkl` 文件，目录包括 `data/01_raw_subject_data` 到 `data/11_grammar`。

## Anti-Patterns

### 在脚本层继续堆积可复用业务逻辑

**What happens:** `script/04_human_tile_data_preprocess.py`、`script/07_revise_human_weight.py`、`script/08_extract_features_human.py`、`script/09_human_fmri_data_preprocess.py` 包含大量业务函数和数据转换逻辑。  
**Why it's wrong:** 后续测试、复用和阶段验证需要直接导入脚本文件，模块边界不如 `src/LoPS/` 清晰。  
**Do this instead:** 新增可复用逻辑放入 `src/LoPS/<domain>/`，脚本只保留 `parse_args()`、配置构造和 `main()`，参考 `script/05_calculate_utility.py` 调用 `src/LoPS/calculate_utility/processing.py` 的模式。

### 主流程和视频流程混用数据假设

**What happens:** 主流程使用 `data/03_preprocessed_frame_data` 后的精简字段；视频流程使用 `data/02_frame_data` 的原始渲染字段。  
**Why it's wrong:** 视频需要像素、方向帧和 HUD 字段，主分析需要两鬼分析字段；混用会导致字段缺失或把展示字段带入科研模型。  
**Do this instead:** 主分析新增阶段读取编号数据目录；视频新增能力放在 `src/LoPS/pacman_video/` 和 `script/pacman_video/`。

### 旧格式适配反向约束正式模型

**What happens:** `.planning/preestimation_fmri_refactor_analysis.md` 记录了旧脚本输出和旧字段兼容需求；当前正式模块已经将 `hierarchical_utility` 拆成模型、策略、估计三个文件。  
**Why it's wrong:** 为旧输出格式污染核心数据结构会降低新模块内聚性，并违反仓库协作说明。  
**Do this instead:** 核心模块使用当前清晰结构；新旧对比适配逻辑放在独立验证脚本或 `data/` 下验证产物，不能进入 `src/LoPS/` 核心路径。

## Error Handling

**Strategy:** 以明确异常类和 fail-fast 输入检查为主，批处理阶段返回文件级摘要。

**Patterns:**
- 原始预处理异常：`RawFmriError` 位于 `src/LoPS/pacman_preprocess/mat_to_raw_subject_data.py`。
- frame 转换异常：`FrameDataError` 位于 `src/LoPS/pacman_preprocess/raw_subject_data_to_frame_data.py`。
- frame 标准化异常：`FrameDataPreprocessError` 位于 `src/LoPS/pacman_preprocess/frame_data_preprocess.py`。
- 结构学习异常：`StructureLearningError` 位于 `src/LoPS/structure_learning.py`，`StateDependencyGraphError` 位于 `src/LoPS/state_dependency_graph.py`。
- 视频数据异常：`DataProcessingError` 位于 `src/LoPS/pacman_video/render_table.py`，`VideoBuildError` 位于 `src/LoPS/pacman_video/video_renderer.py`。
- 目录批处理函数通常返回包含 `input`、`output`、`rows`、`status` 等字段的 dict 摘要，脚本层负责打印。

## Cross-Cutting Concerns

**Logging:** 主要使用 `print()` 输出进度和摘要；`script/11_generate_grammar.py` 支持 progress callback 和 `--quiet`。  
**Validation:** 运行期以字段检查、路径检查和异常类为主；测试位于 `tests/`，覆盖 frame data、tile 预处理、grammar 和结构学习。  
**Authentication:** Not applicable；仓库未检测到外部服务认证流程。  
**Serialization:** 阶段数据使用 pickle；需要跨 Python/pandas 版本时优先用当前项目环境运行完整流水线。  
**Parallelism:** 并行 worker 数由 CLI 参数控制；新增阶段保持目录级并行和单文件函数可 pickle。  
**Reproducibility:** `script/06_dynamic_strategy_fitting.py` 默认 `--seed 20260610`，每个文件根据排序序号派生 seed。

---

*Architecture analysis: 2026-06-22*
