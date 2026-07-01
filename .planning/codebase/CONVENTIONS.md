# Coding Conventions

**Analysis Date:** 2026-06-22

## Naming Patterns

**Files:**
- 正式模块使用小写加下划线的 Python 文件名，按业务阶段或领域分组，例如 `src/LoPS/generate_grammar/pipeline.py`、`src/LoPS/calculate_utility/processing.py`、`src/LoPS/pacman_preprocess/raw_subject_data_to_frame_data.py`。
- 运行入口放在 `script/`，主流程入口以两位序号加阶段名命名，例如 `script/01_mat_to_raw_subject_data.py`、`script/11_generate_grammar.py`；视频入口保持在 `script/pacman_video/run_frame_renderer.py` 这类 `run_*.py` 文件中。
- 测试文件使用 `test_*.py`，与被测阶段或模块同名，例如 `tests/test_generate_grammar_pipeline.py`、`tests/test_pacman_frame_data.py`。

**Functions:**
- 函数使用 `snake_case`，公开流程函数使用动词短语表达输入输出边界，例如 `run_generate_grammar`、`process_calculate_utility_directory`、`convert_raw_subject_data_to_frame_data`。
- 私有辅助函数使用前导下划线，例如 `src/LoPS/pacman_preprocess/raw_subject_data_to_frame_data.py` 中的 `_convert_one_worker`、`_subject_from_input`，以及 `src/LoPS/generate_grammar/grammar.py` 中的内部学习辅助方法。
- 脚本入口统一包含 `parse_args()` 和 `main()`，必要时增加 `build_config()`，例如 `script/05_calculate_utility.py`、`script/06_dynamic_strategy_fitting.py`。

**Variables:**
- 常量使用全大写，例如 `DIRECTION_NAMES`、`DEFAULT_STATE_NAMES`、`Q_NORM_COLUMNS`，位置在模块顶部。
- 路径变量使用 `Path` 对象命名，常见名称是 `PROJECT_ROOT`、`SRC_ROOT`、`data_root`、`input_dir`、`output_dir`，例如 `script/05_calculate_utility.py`。
- 表格字段保持科研数据原字段大小写，例如 `DayTrial`、`pacmanPos`、`ghost1Pos`、`available_dir`；不要为了 Python 风格重命名外部数据列，除非阶段输出规范明确要求。

**Types:**
- 配置和结构化中间结果使用 `@dataclass`，配置类通常 `frozen=True`，例如 `GenerateGrammarConfig`、`GrammarLearningParams`、`CalculateUtilityConfig`、`DynamicStrategyFittingConfig`。
- 结果对象和算法过程快照使用语义化类名，例如 `GrammarLearningResult`、`SkipGramCandidateTrace`、`PreparedStrategyStateData`。
- 自定义异常使用业务名加 `Error`，例如 `FrameDataError`、`StateDependencyGraphError`、`VideoBuildError`。

## Code Style

**Formatting:**
- 未检测到 `black`、`ruff`、`isort`、`flake8`、`mypy` 或 `pyright` 配置；`pyproject.toml` 只声明包元数据、依赖和 Python 版本。
- 代码风格以 PEP 8 为主：`from __future__ import annotations` 位于模块 docstring 后，标准库导入、第三方导入、项目导入分组，函数签名使用类型标注。
- 长函数在大文件中存在，例如 `src/LoPS/dynamic_strategy_fitting.py`、`src/LoPS/pacman_video/frame_renderer.py`、`script/07_revise_human_weight.py`；新增代码应优先提取清晰的纯函数或配置对象，避免继续扩大单文件复杂度。

**Linting:**
- 未检测到仓库级 lint 命令或 CI 检查。
- 脚本因运行时插入 `sys.path` 会在导入后使用 `# noqa: E402`，例如 `script/05_calculate_utility.py`。新增脚本如需同样模式，应只在入口层使用，不要放入正式模块。

## Import Organization

**Order:**
1. `from __future__ import annotations`
2. 标准库导入，例如 `argparse`、`pickle`、`pathlib.Path`、`dataclasses`
3. 第三方库导入，例如 `numpy`、`pandas`、`scipy`、`PIL`
4. 项目内导入，例如 `from LoPS.generate_grammar.config import GenerateGrammarConfig`

**Path Aliases:**
- 包源码位于 `src/LoPS`，正常测试和脚本运行依赖 `PYTHONPATH=src`，文档命令见 `data/README.md`。
- 多数脚本入口会通过 `PROJECT_ROOT / "src"` 插入 `sys.path`，例如 `script/03_frame_data_preprocess.py`、`script/06_dynamic_strategy_fitting.py`、`script/pacman_video/run_render_table.py`。
- 测试中应优先直接导入 `LoPS.*`。`tests/test_human_tile_data_preprocess.py` 通过 `importlib.util.spec_from_file_location` 加载 `script/04_human_tile_data_preprocess.py`，这是一次性脚本中仍承载可测试逻辑的边界信号；新增可复用逻辑应迁入 `src/LoPS/`。

## Error Handling

**Patterns:**
- 输入目录不存在、输入文件为空、关键列缺失时直接抛出明确异常，例如 `FileNotFoundError`、`ValueError` 或业务异常；示例见 `src/LoPS/generate_grammar/config.py` 的 `GenerateGrammarConfig.validate()`、`src/LoPS/calculate_utility/processing.py` 的 `correct_unavailable_q_values()`。
- 批处理目录函数通常先校验输入目录和 `.pkl` 文件集合，再创建输出目录，例如 `process_calculate_utility_directory()`、`process_state_dependency_graph_directory()`、`convert_raw_subject_data_to_frame_data_dir()`。
- 多进程任务包装 worker 异常并补充被试或文件名上下文，例如 `src/LoPS/pacman_preprocess/raw_subject_data_to_frame_data.py` 中的 `FrameDataError(f"{subject} 转换失败：{exc}")`。
- 数据解析使用 `ast.literal_eval` 或显式类型判断，不使用 `eval`，例如 `src/LoPS/calculate_utility/processing.py`、`src/LoPS/dynamic_strategy_fitting.py`。

## Logging

**Framework:** `console`

**Patterns:**
- 没有集中日志框架；脚本通过 `print()` 或 `json.dumps(..., ensure_ascii=False, indent=2)` 输出摘要，例如 `script/05_calculate_utility.py`、`script/06_dynamic_strategy_fitting.py`。
- 长流程提供简短进度回调或逐文件输出，例如 `script/11_generate_grammar.py` 的 `build_progress_printer()` 和 `src/LoPS/generate_grammar/pipeline.py` 的 `progress_callback`。
- 正式模块中只在批处理或旧行为复现边界处打印；新增核心算法应通过返回摘要或回调暴露状态，避免在纯函数中直接输出。

## Comments

**When to Comment:**
- 注释使用中文，重点解释科研数据语义、旧行为复现原因、边界条件和数据形态转换；示例见 `src/LoPS/pacman_preprocess/raw_subject_data_to_frame_data.py` 对 `Map`、tunnel、ghost3/ghost4 的说明。
- 不为显而易见的赋值写注释；注释应说明“为什么这样做”或“这个字段约束什么结果”。
- AGENTS.md 要求所有新增或修改代码加入详细中文注释，并在重点逻辑、难点逻辑、数据形态转换和边界条件处添加中间注释。

**JSDoc/TSDoc:**
- 不适用；本仓库为 Python 项目。
- Python docstring 覆盖状态良好：静态扫描显示 `src/LoPS/` 的 24 个 Python 文件、35 个类、311 个函数均有 docstring；`script/` 的 17 个 Python 文件、1 个类、108 个函数均有 docstring；`tests/` 的 10 个 Python 文件、7 个类、32 个函数均有 docstring。
- docstring 采用中文三段语义：功能说明、输入语义、输出语义、关键约束。典型示例见 `src/LoPS/generate_grammar/config.py`、`src/LoPS/generate_grammar/pipeline.py`、`src/LoPS/state_dependency_graph.py`。

## Function Design

**Size:** 新增代码应把算法、目录批处理、单文件 I/O 和 CLI 拆开；参考 `src/LoPS/generate_grammar/pipeline.py` 中 `prepare_strategy_state_data()`、`process_strategy_state_file()`、`run_generate_grammar()` 的层次。

**Parameters:** 正式模块函数接收显式路径、DataFrame、配置对象或地图对象，不从仓库根推断数据目录；脚本层才提供默认路径。示例：`GenerateGrammarConfig` 接收 `strategy_sequence_dir`、`state_graph_dir`、`output_dir`，`script/11_generate_grammar.py` 负责给出默认值。

**Return Values:** 核心函数返回结构化对象、DataFrame、dict 摘要或输出路径列表；脚本入口负责打印摘要。示例见 `src/LoPS/generate_grammar/pipeline.py` 的 `run_generate_grammar()` 返回 `list[Path]`，`src/LoPS/state_dependency_graph.py` 返回处理摘要 dict。

## Module Design

**Exports:** 包级 `__init__.py` 暴露稳定 API，例如 `src/LoPS/calculate_utility/__init__.py`、`src/LoPS/hierarchical_utility/__init__.py`。新增正式 API 应通过包级入口集中导出，避免脚本深层引用内部辅助函数。

**Barrel Files:** 有限使用；`src/LoPS/generate_grammar/__init__.py` 仅作为包标记，`src/LoPS/hierarchical_utility/__init__.py` 和 `src/LoPS/calculate_utility/__init__.py` 承担导出清单角色。

## Configuration And Data Paths

**Configuration Passing:**
- 算法参数集中在 dataclass 中，例如 `UtilityConfig`、`GrammarLearningParams`、`DynamicStrategyFittingConfig`。
- CLI 参数在脚本层解析后转换为配置对象，例如 `script/05_calculate_utility.py` 的 `build_config()`。
- 随机性必须显式配置和记录。`script/06_dynamic_strategy_fitting.py` 默认 `--seed 20260610`，并在段落并行时使用 `random_seed + file_index + segment_index` 语义。

**Data Path Handling:**
- 正式模块不得保存旧项目或仓库外绝对路径。`src/LoPS/generate_grammar/config.py`、`src/LoPS/calculate_utility/processing.py`、`src/LoPS/dynamic_strategy_fitting.py` 的 docstring 均明确要求调用方显式传入路径。
- 脚本层默认路径指向当前仓库 `data/`，例如 `script/04_human_tile_data_preprocess.py` 的 `DEFAULT_DATA_ROOT`、`script/05_calculate_utility.py` 的 `data_root = PROJECT_ROOT / "data"`。
- 运行数据、输出、验证产物必须放在 `data/`；`.planning/` 只保存计划、讨论、分析和结论文档。`README.md` 和 `data/README.md` 已按 01-12 阶段描述目录。

## Formal Modules Vs One-Off Scripts

**正式模块边界:**
- 可复用算法、数据结构、文件级/目录级处理函数放入 `src/LoPS/`，例如 `src/LoPS/generate_grammar/grammar.py`、`src/LoPS/hierarchical_utility/model.py`、`src/LoPS/state_dependency_graph.py`。
- 正式模块可以提供格式转换函数，但旧格式适配不应反向约束核心结构。示例：`src/LoPS/state_dependency_graph.py` 的 `convert_to_legacy_state_graph()` 明确只用于一致性验证或外部兼容边界。

**一次性脚本边界:**
- `script/` 负责 CLI、默认数据路径、运行摘要和批处理参数，不应承载新增核心算法。
- 当前存在脚本级可测试逻辑，例如 `script/04_human_tile_data_preprocess.py` 被 `tests/test_human_tile_data_preprocess.py` 直接加载；新增阶段应优先把可复用逻辑放入 `src/LoPS/`，脚本只调用正式模块。

**AGENTS.md 强制流程关联:**
- 修改业务实现前必须先阅读目标脚本及依赖、记录原始功能和输入输出，并提出重构方案等待确认；对应分析资料可放入 `.planning/*.md`，例如 `.planning/preestimation_fmri_refactor_analysis.md`。
- 若需要新旧比对，验证脚本或适配模块应与正式核心模块隔离；验证输出放入 `data/`，临时旧代码只允许放入 `src/LoPS/temp/` 并在验证通过后清理。
- 扫描显示 `src/LoPS/temp` 目录存在但无文件；新增工作不得在其中留下临时代码。

---

*Convention analysis: 2026-06-22*
