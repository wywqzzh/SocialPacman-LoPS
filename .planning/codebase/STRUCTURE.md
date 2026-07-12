# Codebase Structure

> 历史快照：本文的目录树、文件数量和主流程映射来自 2026-06-22，未包含后续分支。
> 当前代码结构和数据流请以实际目录、`README.md`、`data/README.md` 与
> `docs/data_flow.html` 为准。

**Analysis Date:** 2026-06-22

## Directory Layout

```text
SocialPacman-LoPS/
├── AGENTS.md                         # 仓库协作规范，要求中文文档、方案确认、数据只进 data/
├── README.md                         # 主分析流程与视频流程总览
├── pyproject.toml                    # Python 包配置，包根为 `src/LoPS`
├── poetry.lock                       # Poetry 锁文件
├── docs/
│   └── data_flow.html                # 人工可读的数据流程说明页面
├── script/
│   ├── 01_mat_to_raw_subject_data.py # 主流程第 01 阶段入口
│   ├── 02_raw_subject_data_to_frame_data.py
│   ├── 03_frame_data_preprocess.py
│   ├── 04_human_tile_data_preprocess.py
│   ├── 05_calculate_utility.py
│   ├── 06_dynamic_strategy_fitting.py
│   ├── 07_revise_human_weight.py
│   ├── 08_extract_features_human.py
│   ├── 09_human_fmri_data_preprocess.py
│   ├── 10_state_dependency_graph.py
│   ├── 11_generate_grammar.py
│   ├── 12_divide_person.py
│   └── pacman_video/                 # 视频渲染三个入口
├── src/
│   └── LoPS/
│       ├── pacman_preprocess/        # `.mat -> raw_subject -> frame -> preprocessed frame`
│       ├── hierarchical_utility/     # fMRI utility 核心模型、策略和估计
│       ├── calculate_utility/        # corrected tile 到 utility 表的阶段流水线
│       ├── generate_grammar/         # grammar 数据读取、学习、输出和人群划分
│       ├── pacman_video/             # render table、Pillow 渲染、ffmpeg 合成
│       ├── dynamic_strategy_fitting.py
│       ├── state_dependency_graph.py
│       ├── structure_learning.py
│       └── temp/                     # 验证阶段临时旧实现目录，当前无文件
├── data/
│   ├── 00_raw_mat_data/              # 原始 `.mat` session 目录
│   ├── 01_raw_subject_data/          # 主流程第 01 阶段输出
│   ├── ...                           # 第 02 到第 10 阶段输出
│   ├── 11_grammar/                   # 主流程最终 grammar 输出
│   ├── constant_data/                # fMRI 地图常量 CSV
│   └── pacman_video/                 # 视频专用输入和输出
├── tests/                            # pytest/unittest 测试
└── .planning/
    ├── preestimation_fmri_refactor_analysis.md
    └── codebase/
        ├── ARCHITECTURE.md
        └── STRUCTURE.md
```

## Directory Purposes

**`src/LoPS/`:**
- Purpose: 可复用正式 Python 包。
- Contains: 科研数据转换、模型估计、结构学习、grammar 学习、视频生成模块。
- Key files: `src/LoPS/dynamic_strategy_fitting.py`, `src/LoPS/state_dependency_graph.py`, `src/LoPS/structure_learning.py`。

**`src/LoPS/pacman_preprocess/`:**
- Purpose: 原始 Pacman fMRI 行为数据预处理。
- Contains: `.mat` 读取、raw subject DataFrame 构建、frame data 构建、分析字段标准化。
- Key files: `src/LoPS/pacman_preprocess/mat_to_raw_subject_data.py`, `src/LoPS/pacman_preprocess/raw_subject_data_to_frame_data.py`, `src/LoPS/pacman_preprocess/frame_data_preprocess.py`。

**`src/LoPS/hierarchical_utility/`:**
- Purpose: fMRI hierarchical utility 的正式核心模型。
- Contains: 地图常量读取与编译、行状态解析、共享路径树、策略 Q 估计、DataFrame/文件/目录处理。
- Key files: `src/LoPS/hierarchical_utility/model.py`, `src/LoPS/hierarchical_utility/strategies.py`, `src/LoPS/hierarchical_utility/estimation.py`, `src/LoPS/hierarchical_utility/__init__.py`。

**`src/LoPS/calculate_utility/`:**
- Purpose: 主流程第 05 阶段的集中 utility 计算边界。
- Contains: `CalculateUtilityConfig`、Q 修正、Q 归一化、文件级/目录级处理。
- Key files: `src/LoPS/calculate_utility/processing.py`, `src/LoPS/calculate_utility/__init__.py`。

**`src/LoPS/generate_grammar/`:**
- Purpose: 主流程第 11 和第 12 阶段的 grammar 学习与人群划分。
- Contains: 配置、数据读取、grammar token、学习器、pipeline、DividePerson 后处理。
- Key files: `src/LoPS/generate_grammar/config.py`, `src/LoPS/generate_grammar/data.py`, `src/LoPS/generate_grammar/token.py`, `src/LoPS/generate_grammar/grammar.py`, `src/LoPS/generate_grammar/pipeline.py`, `src/LoPS/generate_grammar/grammar_process.py`。

**`src/LoPS/pacman_video/`:**
- Purpose: 视频流程的可复用代码。
- Contains: grammar 标注和 frame data 对齐、渲染表生成、静态地图绘制、逐帧 JPG 渲染、MP4 合成。
- Key files: `src/LoPS/pacman_video/render_table.py`, `src/LoPS/pacman_video/frame_renderer.py`, `src/LoPS/pacman_video/video_renderer.py`。

**`script/`:**
- Purpose: 主分析流程命令行入口。
- Contains: `01` 到 `12` 的编号脚本，按执行顺序组织。
- Key files: `script/01_mat_to_raw_subject_data.py`, `script/05_calculate_utility.py`, `script/11_generate_grammar.py`。

**`script/pacman_video/`:**
- Purpose: 视频流程命令行入口。
- Contains: render table、frame renderer、video renderer 三个入口。
- Key files: `script/pacman_video/run_render_table.py`, `script/pacman_video/run_frame_renderer.py`, `script/pacman_video/run_video_renderer.py`。

**`data/`:**
- Purpose: 当前流程的全部输入、中间结果、最终输出和视频产物。
- Contains: 编号阶段目录、地图常量、视频数据目录。
- Key files: `data/README.md`, `data/constant_data/adjacent_map_fmri.csv`, `data/constant_data/dij_distance_map_fmri.csv`。

**`docs/`:**
- Purpose: 人工文档。
- Contains: 可浏览 HTML 数据流程说明。
- Key files: `docs/data_flow.html`。

**`.planning/`:**
- Purpose: GSD 规划、分析和代码库映射文档。
- Contains: 阶段性分析文档和 `.planning/codebase/` 输出。
- Key files: `.planning/preestimation_fmri_refactor_analysis.md`, `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`。

**`tests/`:**
- Purpose: 单元和集成测试。
- Contains: Pacman frame data、tile preprocess、grammar foundation/process/pipeline/scoring 测试。
- Key files: `tests/test_pacman_frame_data.py`, `tests/test_human_tile_data_preprocess.py`, `tests/test_generate_grammar_pipeline.py`。

## Key File Locations

**Entry Points:**
- `script/01_mat_to_raw_subject_data.py`: `data/00_raw_mat_data` 到 `data/01_raw_subject_data`。
- `script/02_raw_subject_data_to_frame_data.py`: `data/01_raw_subject_data` 到 `data/02_frame_data`，可选 CSV 到 `data/02_frame_data_csv`。
- `script/03_frame_data_preprocess.py`: `data/02_frame_data` 到 `data/03_preprocessed_frame_data`。
- `script/04_human_tile_data_preprocess.py`: `data/03_preprocessed_frame_data` 到 `data/04_tile_data` 和 `data/04_corrected_tile_data`。
- `script/05_calculate_utility.py`: `data/04_corrected_tile_data` 到 `data/05_utility_data`。
- `script/06_dynamic_strategy_fitting.py`: `data/05_utility_data` 到 `data/06_weight_data`。
- `script/07_revise_human_weight.py`: `data/06_weight_data` 到 `data/07_corrected_weight_data`。
- `script/08_extract_features_human.py`: `data/07_corrected_weight_data` 到 `data/08_feature_data` 和 `data/08_discrete_feature_data`。
- `script/09_human_fmri_data_preprocess.py`: `data/08_discrete_feature_data` 到 `data/09_fmri_discrete_feature_data_ghost2`、`data/09_fmri_formed_data_ghost2`、`data/09_strategy_sequence`。
- `script/10_state_dependency_graph.py`: `data/09_strategy_sequence` 到 `data/10_state_dependency_graph_data`。
- `script/11_generate_grammar.py`: `data/09_strategy_sequence` 与 `data/10_state_dependency_graph_data` 到 `data/11_grammar`。
- `script/12_divide_person.py`: 读取 `data/11_grammar` 并打印 JSON。
- `script/pacman_video/run_render_table.py`: `data/02_frame_data` 与 `data/pacman_video/grammar_data` 到 `data/pacman_video/render_data`。
- `script/pacman_video/run_frame_renderer.py`: `data/pacman_video/render_data` 到 `data/pacman_video/frame_images`。
- `script/pacman_video/run_video_renderer.py`: `data/pacman_video/frame_images` 到 `data/pacman_video/video_data`。

**Configuration:**
- `pyproject.toml`: 包元数据、Python 版本约束和依赖。
- `poetry.lock`: 依赖锁定。
- `AGENTS.md`: 仓库协作和注释/docstring/数据目录要求。
- `data/README.md`: 完整数据流程命令和目录说明。
- `README.md`: 主流程、视频流程和运行约束摘要。

**Core Logic:**
- `src/LoPS/pacman_preprocess/mat_to_raw_subject_data.py`: 原始 `.mat` 到 raw subject。
- `src/LoPS/pacman_preprocess/raw_subject_data_to_frame_data.py`: raw subject 到 frame data。
- `src/LoPS/pacman_preprocess/frame_data_preprocess.py`: frame data 到精简分析表。
- `src/LoPS/hierarchical_utility/model.py`: 地图与状态解析。
- `src/LoPS/hierarchical_utility/strategies.py`: Q 策略估计。
- `src/LoPS/hierarchical_utility/estimation.py`: DataFrame/文件/目录估计接口。
- `src/LoPS/calculate_utility/processing.py`: Q 修正和归一化。
- `src/LoPS/dynamic_strategy_fitting.py`: context 切分与权重拟合。
- `src/LoPS/structure_learning.py`: 通用离散结构学习。
- `src/LoPS/state_dependency_graph.py`: strategy sequence 到状态依赖图。
- `src/LoPS/generate_grammar/grammar.py`: grammar 学习核心。
- `src/LoPS/generate_grammar/pipeline.py`: grammar 文件级流水线。
- `src/LoPS/generate_grammar/grammar_process.py`: DividePerson 后处理。
- `src/LoPS/pacman_video/render_table.py`: 视频渲染表生成。
- `src/LoPS/pacman_video/frame_renderer.py`: JPG 帧渲染。
- `src/LoPS/pacman_video/video_renderer.py`: MP4 合成。

**Testing:**
- `tests/test_pacman_frame_data.py`: raw subject 到 frame data 行为测试。
- `tests/test_human_tile_data_preprocess.py`: tile 数据预处理脚本测试。
- `tests/test_generate_grammar_foundation.py`: grammar 基础数据和 token 测试。
- `tests/test_generate_grammar_grammar.py`: grammar 核心学习测试。
- `tests/test_generate_grammar_pipeline.py`: grammar pipeline 测试。
- `tests/test_generate_grammar_process.py`: grammar process 测试。
- `tests/test_generate_grammar_scoring.py`: 结构学习评分测试。
- `tests/test_generate_grammar_divide_person.py`: DividePerson 测试。
- `tests/generate_grammar_fixtures.py`: 测试数据路径常量。

## Data Directory Map

| Directory | Role | File Shape |
|-----------|------|------------|
| `data/00_raw_mat_data/` | 原始 MATLAB trial 数据，按 session 目录组织。 | `<session>/<trial>.mat` |
| `data/01_raw_subject_data/` | raw subject 阶段输出。 | 34 个 `{session}.pkl`，样例 DataFrame 247257x49 |
| `data/02_frame_data/` | frame data 阶段输出，视频流程也读取。 | 34 个 `{session}.pkl`，样例 DataFrame 148954x48 |
| `data/02_frame_data_csv/` | 可选 CSV 输出目录。 | 当前目录存在，文件数量由 `--write-csv` 控制 |
| `data/03_preprocessed_frame_data/` | 标准分析 frame 表。 | 34 个 `{session}.pkl`，样例 DataFrame 148954x11 |
| `data/04_tile_data/` | 抽样 tile 表。 | 34 个 `{session}.pkl`，样例 DataFrame 5998x11 |
| `data/04_corrected_tile_data/` | 修正并补齐路径后的 tile 表。 | 34 个 `{session}.pkl`，样例 DataFrame 6057x13 |
| `data/05_utility_data/` | Q 与归一化 Q。 | 34 个 `{session}.pkl`，样例 DataFrame 6057x28 |
| `data/06_weight_data/` | 动态策略权重拟合结果。 | 34 个 `{session}.pkl`，样例 DataFrame 6057x37 |
| `data/07_corrected_weight_data/` | 规则修正后的权重与策略编号。 | 34 个 `{session}.pkl`，样例 DataFrame 6057x40 |
| `data/08_feature_data/` | 连续特征。 | 34 个 `{session}.pkl`，样例 DataFrame 6057x17 |
| `data/08_discrete_feature_data/` | 离散特征。 | 34 个 `{session}.pkl`，样例 DataFrame 6057x16 |
| `data/09_fmri_discrete_feature_data_ghost2/` | ghost2 离散特征。 | 34 个 `{session}.pkl`，样例 DataFrame 6057x16 |
| `data/09_fmri_formed_data_ghost2/` | ghost2 formed strategy one-hot 表。 | 34 个 `{session}.pkl`，样例 DataFrame 404x25 |
| `data/09_strategy_sequence/` | grammar 和状态图输入。 | 34 个 `{session}.pkl`，dict: `seq`, `S`, `state`, `strategy`, `strategyLabel`, `fileNames` |
| `data/10_state_dependency_graph_data/` | 状态依赖图结果。 | 34 个 `{session}.pkl`，dict: `state_names`, `state_matrix`, `adjacency_matrix` |
| `data/11_grammar/` | grammar 最终输出。 | 34 个 `{session}.pkl`，dict: `source`, `parameters`, `grammar`, `parsed`, `skip_gram` |
| `data/constant_data/` | 地图常量。 | `adjacent_map_fmri.csv`, `dij_distance_map_fmri.csv` |
| `data/pacman_video/grammar_data/` | 视频流程标注输入。 | `*-gram.pkl` |
| `data/pacman_video/render_data/` | 视频渲染表。 | `{subject-session}.pkl` |
| `data/pacman_video/frame_images/` | JPG 图片帧。 | `<subject>/<game>/*.jpg` |
| `data/pacman_video/video_data/` | MP4 输出。 | `<subject>/*.mp4` |

## Naming Conventions

**Files:**
- 主流程入口使用两位编号前缀：`script/01_mat_to_raw_subject_data.py`。
- 主流程阶段目录使用两位编号前缀：`data/05_utility_data/`。
- 主流程 `.pkl` 文件在不同阶段保持同一 `{subject/session}.pkl` 名称：`031222-401-03-Dec-2022-1.pkl`。
- 视频脚本使用动作型 `run_*.py`：`script/pacman_video/run_frame_renderer.py`。
- 包内模块使用 snake_case：`src/LoPS/generate_grammar/grammar_process.py`。
- 测试文件使用 `test_*.py`：`tests/test_generate_grammar_pipeline.py`。

**Directories:**
- `src/LoPS/<domain>/` 用领域名组织可复用模块，例如 `src/LoPS/hierarchical_utility/`。
- `script/pacman_video/` 与 `src/LoPS/pacman_video/` 一一对应：脚本入口在 `script/`，实现逻辑在 `src/LoPS/`。
- `data/<NN>_<stage>/` 表示主分析链路阶段；阶段信息只在目录名，不写进文件名。
- `.planning/codebase/` 只保存代码库映射文档。

**Python Symbols:**
- 配置对象使用 dataclass 并以 `Config` 结尾：`UtilityConfig`、`CalculateUtilityConfig`、`DynamicStrategyFittingConfig`、`GenerateGrammarConfig`。
- 目录级处理函数使用 `process_*_directory`：`process_calculate_utility_directory()`、`process_dynamic_strategy_directory()`。
- 单文件处理函数使用 `process_*_file` 或 `process_one_file`：`process_state_dependency_graph_file()`、`script/08_extract_features_human.py` 的 `process_one_file()`。
- 数据读取/转换函数使用动词短语：`load_map_data()`、`compile_map_data()`、`prepare_strategy_state_data()`。

## Where to Add New Code

**New main-flow feature:**
- Primary code: 放入 `src/LoPS/<domain>/`；若是现有阶段扩展，优先放到对应模块，例如 utility 逻辑放 `src/LoPS/calculate_utility/` 或 `src/LoPS/hierarchical_utility/`。
- Entry script: 新增或修改 `script/<NN>_<name>.py`，保持编号顺序和默认 `data/` 路径。
- Data: 新增输入、输出、验证产物放入 `data/<NN>_<stage>/` 或明确的 `data/<domain>/` 子目录。
- Tests: 放入 `tests/test_<domain>.py`，需要 fixture 路径时参考 `tests/generate_grammar_fixtures.py`。

**New preprocessing module:**
- Implementation: `src/LoPS/pacman_preprocess/`
- Entry: `script/01_*` 到 `script/04_*` 范围内的编号入口，或新增相邻编号脚本。
- Data: 上游/下游目录必须在 `data/README.md` 和 `README.md` 中保持一致。

**New utility or strategy model:**
- Model/data parsing: `src/LoPS/hierarchical_utility/model.py`
- Search/reward/risk logic: `src/LoPS/hierarchical_utility/strategies.py`
- DataFrame/file/directory wrapper: `src/LoPS/hierarchical_utility/estimation.py` 或 `src/LoPS/calculate_utility/processing.py`
- Entry: `script/05_calculate_utility.py` 或新的脚本入口。

**New strategy fitting logic:**
- Implementation: `src/LoPS/dynamic_strategy_fitting.py`
- Entry: `script/06_dynamic_strategy_fitting.py`
- Downstream compatibility: 保持 `data/06_weight_data` 中 `weight`、`normalized_weight`、`predict_dir`、context 相关列可被 `script/07_revise_human_weight.py` 读取。

**New grammar/state learning logic:**
- Generic algorithm: `src/LoPS/structure_learning.py`
- Pacman state graph boundary: `src/LoPS/state_dependency_graph.py`
- Grammar token/learning/output: `src/LoPS/generate_grammar/`
- Entry: `script/10_state_dependency_graph.py`, `script/11_generate_grammar.py`, `script/12_divide_person.py`

**New video capability:**
- Render-table preparation: `src/LoPS/pacman_video/render_table.py`
- Frame drawing: `src/LoPS/pacman_video/frame_renderer.py`
- Video assembly: `src/LoPS/pacman_video/video_renderer.py`
- Entry: `script/pacman_video/run_*.py`
- Data: `data/pacman_video/<subdir>/`

**Utilities:**
- Shared research algorithm utilities: 放入最接近的 `src/LoPS/<domain>/`，不要新建宽泛 `utils.py`，除非多个领域确实共享。
- Validation-only adapters: 放在独立验证脚本或 `data/` 验证产物路径，不进入正式核心模块。

## Special Directories

**`src/LoPS/temp/`:**
- Purpose: 验证阶段临时旧实现副本。
- Generated: Yes，按轮次临时生成。
- Committed: No，当前目录无文件；每轮验证结束必须清理。

**`data/`:**
- Purpose: 脚本输入、输出、测试数据和验证产物。
- Generated: Mixed；`00_raw_mat_data/` 和 `constant_data/` 是输入/常量，其它编号目录多为脚本输出。
- Committed: 当前仓库包含阶段数据文件和 `.gitignore`；新增数据不要放入 `.planning/`。

**`data/02_frame_data_csv/`:**
- Purpose: `script/02_raw_subject_data_to_frame_data.py --write-csv` 的可选 CSV 输出。
- Generated: Yes。
- Committed: 目录存在；具体 CSV 由运行参数决定。

**`data/pacman_video/`:**
- Purpose: 视频渲染相关输入、图片帧和 MP4。
- Generated: Mixed；`grammar_data/` 是视频标注输入，`render_data/`、`frame_images/`、`video_data/` 是下游输出。
- Committed: 当前包含 grammar/render/video 示例产物。

**`.planning/codebase/`:**
- Purpose: GSD codebase mapper 输出。
- Generated: Yes。
- Committed: 由 orchestrator 决定；内容只记录分析文档，不保存数据。

**`.venv/`, `.pytest_cache/`, `__pycache__/`:**
- Purpose: 本地环境、pytest 缓存、Python 字节码缓存。
- Generated: Yes。
- Committed: No；不要作为源码或数据依赖。

## Module Boundary Rules

- 新正式代码放在 `src/LoPS/`，运行入口放在 `script/`，数据放在 `data/`，分析文档放在 `.planning/` 或 `docs/`。
- `src/LoPS/` 不能保存旧项目绝对路径；脚本默认路径只能指向当前仓库 `data/`。
- 主流程模型不要读取 `data/pacman_video/`；视频流程不要依赖 `data/03_preprocessed_frame_data` 以后的主分析字段，除非明确是展示用派生数据。
- 旧输出格式比较和验证适配放在独立验证脚本，不反向约束 `src/LoPS/` 的核心数据结构。
- 脚本层保留 CLI、默认路径和摘要打印；新增复杂可测逻辑应进入 `src/LoPS/<domain>/`。

---

*Structure analysis: 2026-06-22*
