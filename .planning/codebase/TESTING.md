# Testing Patterns

**Analysis Date:** 2026-06-22

## Test Framework

**Runner:**
- `pytest` 发现机制可运行现有测试；仓库没有 `pytest.ini`、`tox.ini`、`.coveragerc` 或 `[tool.pytest]` 配置。
- `pyproject.toml` 声明 Python `>=3.10,<3.11` 和运行依赖，但未声明测试依赖分组；测试使用 `unittest`、`pytest`、`numpy.testing`、`pandas`。
- 测试文件位于 `tests/`，共 10 个 Python 文件、约 1075 行。

**Assertion Library:**
- `unittest.TestCase` 的 `self.assertEqual`、`self.assertTrue`、`self.assertAlmostEqual` 是主要风格，见 `tests/test_generate_grammar_foundation.py`、`tests/test_generate_grammar_process.py`。
- pytest 裸 `assert` 和 `pytest.raises` 用于函数式测试，见 `tests/test_generate_grammar_divide_person.py`。
- 数组一致性使用 `np.testing.assert_array_equal`，见 `tests/test_generate_grammar_scoring.py`、`tests/test_generate_grammar_process.py`。

**Run Commands:**
```bash
PYTHONPATH=src pytest tests
PYTHONPATH=src pytest tests/test_generate_grammar_pipeline.py
PYTHONPATH=src python -m unittest discover tests
```

## Test File Organization

**Location:**
- 测试集中在 `tests/` 根目录，没有按包目录镜像拆分。
- 共享真实数据路径集中在 `tests/generate_grammar_fixtures.py`，指向 `data/09_strategy_sequence`、`data/10_state_dependency_graph_data`、`data/11_grammar`。

**Naming:**
- 文件使用 `test_<area>.py`，例如 `tests/test_generate_grammar_grammar.py`、`tests/test_generate_grammar_scoring.py`。
- `generate_grammar` 相关测试按基础设施、核心 grammar、过程快照、pipeline、scoring、DividePerson 拆分。

**Structure:**
```text
tests/
├── generate_grammar_fixtures.py
├── test_generate_grammar_foundation.py
├── test_generate_grammar_grammar.py
├── test_generate_grammar_process.py
├── test_generate_grammar_pipeline.py
├── test_generate_grammar_scoring.py
├── test_generate_grammar_divide_person.py
├── test_pacman_frame_data.py
└── test_human_tile_data_preprocess.py
```

## Test Structure

**Suite Organization:**
```python
class GenerateGrammarPipelineTest(unittest.TestCase):
    """覆盖 generate_grammar 的文件级 pipeline 编排行为。"""

    def test_prepare_strategy_state_data_removes_n_and_aligns_state_features(self) -> None:
        """验证预处理会删除 N token 并同步对齐状态特征行。"""
        record = load_strategy_state_data(...)
        state_dependencies = load_state_dependency_graph(...)
        prepared = prepare_strategy_state_data(record, state_dependencies)
        self.assertNotIn("N", prepared.token_sequence)
        self.assertEqual(len(prepared.token_sequence), len(prepared.state_features))
```

**Patterns:**
- 每个测试函数都有中文 docstring，说明验证目标；测试内部用中文注释解释数据构造的关键边界。
- 小型单元测试直接构造 DataFrame 或数组，例如 `tests/test_pacman_frame_data.py` 验证 DayTrial 数字排序、four-ghost trial 删除、`frame_id` 生成。
- 代表性真实数据测试读取仓库内 `data/`，例如 `tests/test_generate_grammar_foundation.py` 读取 `031222-401-03-Dec-2022-1.pkl`。
- 临时输出使用 `tempfile.TemporaryDirectory()` 或 pytest `tmp_path`，不写入 `.planning/`；示例见 `tests/test_generate_grammar_pipeline.py` 和 `tests/test_generate_grammar_divide_person.py`。

## Mocking

**Framework:** Not detected

**Patterns:**
```python
progress_events = []

def capture_progress(event: str, payload: dict[str, object]) -> None:
    """收集单文件处理过程事件，便于验证运行脚本可打印学习进度。"""
    progress_events.append((event, dict(payload)))
```

**What to Mock:**
- 当前测试不使用 mock；新增代码优先使用小型 DataFrame、临时目录、显式配置对象和进度回调捕获来隔离外部 I/O。
- 对 CLI 入口中的纯解析或路径逻辑，可把可复用部分提取到 `src/LoPS/` 后直接测试函数。

**What NOT to Mock:**
- 不要 mock 当前仓库 `data/` 中作为科研一致性基准的真实 pickle 输入；`generate_grammar` 代表性测试直接读取真实阶段数据。
- 不要 mock 旧格式适配结果来替代真实新旧一致性验证；旧格式转换应通过独立验证脚本或适配函数与真实输出比较。

## Fixtures and Factories

**Test Data:**
```python
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
STRATEGY_SEQUENCE_DIR = DATA_ROOT / "09_strategy_sequence"
STATE_GRAPH_DIR = DATA_ROOT / "10_state_dependency_graph_data"
BASELINE_GRAMMAR_DIR = DATA_ROOT / "11_grammar"
```

**Location:**
- `tests/generate_grammar_fixtures.py` 是唯一共享 fixture 路径模块。
- 其它测试在函数内直接构造最小 DataFrame 或 dict，例如 `tests/test_human_tile_data_preprocess.py`、`tests/test_generate_grammar_divide_person.py`。
- 仓库 `data/` 包含主流程 01-11 各 34 个 `.pkl` 输出文件，以及 `data/constant_data` 的 2 个常量文件；这支持真实数据回归测试。

## Coverage

**Requirements:** None enforced

**View Coverage:**
```bash
PYTHONPATH=src pytest tests
```

- 未检测到覆盖率配置或最小覆盖率阈值。
- 测试覆盖重点是 `src/LoPS/generate_grammar/`、`src/LoPS/pacman_preprocess/raw_subject_data_to_frame_data.py` 的关键转换，以及 `script/04_human_tile_data_preprocess.py` 的 ghost 位置修正。
- 多个主流程阶段缺少自动测试：`script/01_mat_to_raw_subject_data.py`、`script/02_raw_subject_data_to_frame_data.py` 的目录级 I/O、`script/05_calculate_utility.py`、`script/06_dynamic_strategy_fitting.py`、`script/07_revise_human_weight.py`、`script/08_extract_features_human.py`、`script/09_human_fmri_data_preprocess.py`、`script/10_state_dependency_graph.py`、视频流程 `script/pacman_video/`。

## Test Types

**Unit Tests:**
- `tests/test_generate_grammar_process.py` 对最长匹配、概率解析、候选评分、skip-gram 过程使用快照式断言，强调与旧过程语义一致。
- `tests/test_generate_grammar_scoring.py` 验证状态组合计数、BD score、条件链接学习。
- `tests/test_pacman_frame_data.py` 验证 frame data 转换的局部表结构约束。

**Integration Tests:**
- `tests/test_generate_grammar_pipeline.py` 调用真实 `data/09_strategy_sequence` 和 `data/10_state_dependency_graph_data`，验证单文件 pipeline 输出结构和 progress 事件。
- `tests/test_generate_grammar_grammar.py` 使用代表性真实文件验证 `GrammarLearner.learn()` 可返回完整结果。
- `tests/test_generate_grammar_foundation.py` 验证配置对象、策略状态数据读取和状态依赖图读取。

**E2E Tests:**
- Not used as automated tests。
- 端到端运行方式由 `data/README.md` 记录，使用 `PYTHONPATH=src python script/01_...py` 到 `script/12_divide_person.py` 的顺序命令。
- `README.md` 和 `docs/data_flow.html` 描述完整数据流，但没有自动化脚本验证 01-12 全链路。

## Validation Scripts And Run Modes

**主流程运行:**
- `data/README.md` 给出 12 步命令，均在仓库根目录执行，使用 `PYTHONPATH=src`，输出写入 `data/01_raw_subject_data` 到 `data/11_grammar`。
- 脚本默认路径全部指向当前仓库 `data/`，例如 `script/05_calculate_utility.py` 默认读取 `data/04_corrected_tile_data` 并写入 `data/05_utility_data`。
- `script/12_divide_person.py` 只打印结果，不保存文件；这点在 `README.md` 和 `data/README.md` 中一致。

**视频流程运行:**
- `README.md` 描述视频入口顺序：`script/pacman_video/run_render_table.py`、`script/pacman_video/run_frame_renderer.py`、`script/pacman_video/run_video_renderer.py`。
- 视频输出默认写入 `data/pacman_video/render_data`、`data/pacman_video/frame_images`、`data/pacman_video/video_data`。

**验证脚本状态:**
- 未检测到独立命名的 `validate_*.py` 验证脚本。
- `.planning/preestimation_fmri_refactor_analysis.md` 规定过验证模式：旧实现临时副本放入 `src/LoPS/temp/`，相同输入运行旧新实现，严格比较输出，报告写入 `data/hierarchical_utility/validation/validation_report.json`，验证后清理 `src/LoPS/temp/`。
- 扫描显示 `src/LoPS/temp` 目录存在但无文件；后续验证完成后应保持该目录无临时代码或直接清理。

## Common Patterns

**Async Testing:**
```python
output = process_strategy_state_file(
    "031222-401-03-Dec-2022-1.pkl",
    config,
    progress_callback=capture_progress,
)
event_names = [event for event, _ in progress_events]
self.assertIn("learn_iteration", event_names)
```

**Error Testing:**
```python
with pytest.raises(KeyError):
    load_divide_person_record(old_format_path)
```

**DataFrame Testing:**
```python
frame_data = convert_raw_subject_data_to_frame_data(raw_data)
self.assertEqual(frame_data["frame_id"].tolist(), [0, 1])
self.assertEqual(frame_data["ghost3Pos"].tolist(), [[], []])
```

**Numerical Testing:**
```python
np.testing.assert_array_equal(actual_adjacency, expected_adjacency)
self.assertAlmostEqual(probabilities[0], 1 / 3)
```

## New/Old Consistency Verification

**Default expectation:**
- AGENTS.md 要求每轮重构使用相同输入运行原始实现和新实现，默认完全一致；如果使用数值容差，必须记录原因和容差。
- 随机过程必须显式记录 seed。若旧代码没有 seed，只能把临时旧代码复制到 `src/LoPS/temp` 并做最小 seed 注入，验证通过后删除临时代码。

**Current practice in tests:**
- `generate_grammar` 单元测试大量使用“matches legacy”“旧版一致”语义，使用固定数组和快照断言验证旧流程关键步骤，见 `tests/test_generate_grammar_process.py`、`tests/test_generate_grammar_scoring.py`。
- `src/LoPS/generate_grammar/grammar_process.py` 明确不兼容旧版 pickle，旧格式只应在独立验证适配层处理；测试 `tests/test_generate_grammar_divide_person.py` 验证旧格式会抛 `KeyError`。

**Recommended verification shape for new phases:**
- 验证输入、旧输出、新输出、报告都放入 `data/<phase>/validation/` 或阶段对应 `data/` 子目录。
- 比较项至少包括文件集合、行列数量、列名、输入原有列、关键输出列逐行逐元素一致。
- 对 DataFrame 使用 `pandas.testing.assert_frame_equal`；对数组列使用 `np.array_equal` 或 `np.testing.assert_array_equal`。如果必须使用容差，报告中写明字段、原因和容差。
- 验证适配逻辑不得进入 `src/LoPS/` 核心模块；可放在独立验证脚本或隔离适配模块中。

## Quality Assurance Mode

**Before implementation:**
- 按 AGENTS.md 先阅读目标脚本及依赖，记录原始功能、输入输出、执行流程、依赖文件和数据关系。
- 判断是否需要在 `src/LoPS/` 创建独立模块；不要在方案确认前修改业务实现代码。

**During implementation:**
- 正式模块不写旧项目路径，不依赖旧代码或旧数据；运行脚本可以为当前仓库 `data/` 设置默认参数。
- 新设计不为了旧输出格式污染正式核心结构；旧格式转换只在验证边界出现。
- 所有新增或修改函数/类必须有中文 docstring，重点逻辑必须有中文注释。

**After implementation:**
- 运行相关单元测试，例如 `PYTHONPATH=src pytest tests/test_generate_grammar_pipeline.py`。
- 对重构阶段运行旧新一致性验证，报告运行方式、验证方式和一致性结论。
- 清理 `src/LoPS/temp` 中本轮临时代码，确保 `.planning/` 不保存输入、输出或验证产物。

---

*Testing analysis: 2026-06-22*
