# Phase 2 重构设计：generateGrammar 模块

**日期:** 2026-05-04  
**状态:** 设计已收敛，准备进入 `PLAN.md` 任务拆分  
**目标脚本:** `/home/zzh/project/Pacman/2.Pac-man/structre-learning/scripts/fmriDataProcess/generateGrammar.py`  
**目标入口:** `main("ghost2", 0.5, False)`

## 设计边界

本阶段只重构旧脚本默认有效路径：

```python
main("ghost2", 0.5, False)
```

本阶段不迁移：

- `ghost4` 分支。
- `needShuffle=True` 分支。
- 旧脚本中默认路径未调用的函数和模块。
- `src.condindepEmp`。
- `bayesianScore.py` 中默认路径未调用的其它函数。

本阶段必须迁移并验证：

- `generateGrammar.py` 默认路径实际调用的 grammar 学习逻辑。
- `src.Utils.count`。
- `src.bayesianScore.BDscore`。
- `src.bayesianScore.learnBayesNetBlock`。

新实现必须写入当前 LoPS 仓库，不能修改原始脚本、原始脚本所在目录或原始数据目录。

## 核心设计原则

重构不是把旧代码搬到 `src/LoPS`，而是在保护原始科研结果一致性的前提下，重新设计模块边界、接口和实现。

设计原则：

1. 核心算法使用清晰的 token 序列，不再使用旧版单字符占位符表示 chunk。
2. 旧版输出兼容逻辑必须隔离在 `legacy` 适配层中。
3. 所有路径、状态列、学习阈值和算法参数都通过配置或函数参数暴露，不写死在核心算法中。
4. 模块职责高内聚低耦合：数据读取、状态图解析、离散计数、BD score、grammar 学习、输出兼容、运行编排和验证分开实现。
5. 遵守 KISS 原则，优先直接、清晰、易维护，避免过度工程化和不必要的通用框架。

## Token 表示设计

旧代码在学习到 chunk 后，会用新的单字符占位符代替 chunk，例如用某个字符表示 `"GL"`。新核心算法不采用这种设计。

新核心算法统一使用 `list[str]` 表示序列：

```python
["G", "L", "E", "A"]
["G-L", "E-A"]
["G-L-E-A"]
```

原始输入 token 长度为 1，例如：

```python
"G"
"L"
"E"
"A"
```

学习到的新 chunk 直接使用 `-` 连接基础 token：

```python
combine_tokens("G", "L") -> "G-L"
combine_tokens("G-L", "E-A") -> "G-L-E-A"
```

算法不能依赖字符串字符位置判断 chunk。所有 token 操作必须通过辅助函数完成：

```python
def split_token(token: str) -> list[str]
def combine_tokens(parent_token: str, child_token: str) -> str
def token_length(token: str) -> int
def tokens_share_base_token(left: str, right: str) -> bool
def format_token(base_tokens: Sequence[str]) -> str
```

示例：

```python
split_token("G-L-E-A") -> ["G", "L", "E", "A"]
token_length("G-L-E-A") -> 4
tokens_share_base_token("G-L", "L-E") -> True
```

## 模块设计

计划新增模块目录：

```text
src/LoPS/generate_grammar/
  __init__.py
  config.py
  data_io.py
  state_graph.py
  token.py
  scoring.py
  grammar.py
  legacy.py
  structured.py
  pipeline.py
```

### `config.py`

职责：

- 定义运行配置和算法参数。
- 提供默认 `ghost2` 状态列和旧默认参数。
- 校验输入目录存在、输出目录可创建。

不包含算法逻辑。

### `data_io.py`

职责：

- 读取 `StrategySequence/*.pkl`。
- 写出新 pickle。
- 枚举输入文件。
- 处理旧字段 `fileNames` 到被试名 `participant_ids` 的转换。

### `state_graph.py`

职责：

- 读取 `StateGraph/*.pkl`。
- 将旧字段 `G` 转换为状态条件列表。
- 表达状态依赖关系。

### `token.py`

职责：

- 提供 token 拆分、组合、长度、重叠判断。
- 提供新 token 到旧 legacy 占位符的转换辅助。

核心算法只能通过该模块理解 `"G-L"` 这类复合 token。

### `scoring.py`

职责：

- 重实现 `count`。
- 重实现 `BDscore`。
- 重实现默认路径实际调用的 `learnBayesNetBlock`。

该模块不依赖 grammar 学习类，也不读写文件。

### `grammar.py`

职责：

- 实现 grammar chunk 学习。
- 实现最长匹配解析。
- 实现 grammar 概率、频数、时间占比计算。
- 实现 skip-gram 检测。

该模块只接收内存中的 token、状态特征和状态依赖图，不读写文件。

### `legacy.py`

职责：

- 构造旧版兼容输出字段。
- 将新核心 token 结果转换为旧字段所需的 `sets`、`seq`、`S`、`components` 等结构。
- 临时生成旧版占位符映射，用于保证 `legacy` 字典和旧 pickle 字段值一致。

旧版占位符映射只允许存在于该兼容层，不进入核心算法。

### `structured.py`

职责：

- 构造新的清晰输出结构。
- 删除旧输出中的冗余表达。
- 用更明确的字段名组织结果，便于后续科研分析和维护。

### `pipeline.py`

职责：

- 串联数据读取、预处理、grammar 学习、skip-gram 检测、输出构造和写文件。
- 提供单文件和全量运行入口。

## 类设计

### `GrammarLearningParams`

```python
@dataclass
class GrammarLearningParams:
    state_names: tuple[str, ...] = ("IS1", "IS2", "PG1", "PG2", "PE", "BN5")
    chunk_alpha: float = 0.5
    condition_alpha: float = 0.5
    skip_gram_alpha: float = 0.5
    max_iterations: int = 100000
    convergence_window: int = 5
    convergence_kl_threshold: float = 0.05
    candidate_ratio_min: float = 1.0
    candidate_ratio_keep: float = 0.85
    min_pair_frequency: float = 0.05
    removed_token: str = "N"
    skip_gram_target: str = "E-A"
    skip_gram_min_offset: int = 2
    skip_gram_max_offset: int = 5
    skip_gram_min_frequency: float = 0.025
    excluded_child_tokens: tuple[str, ...] = ("V", "1", "2", "N", "S", "e")
    excluded_parent_tokens: tuple[str, ...] = ("V", "N")
    reject_shared_base_tokens: bool = True
```

职责：

- 暴露旧代码中影响结果的关键参数。
- 默认值复刻 `main("ghost2", 0.5, False)` 的实际行为。
- 允许后续测试和运行脚本显式传入参数。

### `GenerateGrammarConfig`

```python
@dataclass
class GenerateGrammarConfig:
    strategy_sequence_dir: Path
    state_graph_dir: Path
    output_dir: Path
    baseline_grammar_dir: Path | None = None
    learning: GrammarLearningParams = field(default_factory=GrammarLearningParams)
```

职责：

- 保存输入、输出和验证基准路径。
- 消除旧脚本对当前工作目录的隐式依赖。
- 为运行入口和验证入口提供统一配置。

### `StrategyStateData`

```python
@dataclass
class StrategyStateData:
    input_file_name: str
    token_sequence: list[str]
    initial_tokens: list[str]
    state_features: pd.DataFrame
    participant_file_names: list[str]
    participant_ids: list[str]
```

每个 `StrategySequence/*.pkl` 文件对应一个 `StrategyStateData`。

字段含义：

- `input_file_name`: 当前读取的 pkl 文件名，例如 `031222-401.pkl`。
- `token_sequence`: 旧字段 `seq` 转成的 token 序列。
- `initial_tokens`: 旧字段 `S`，初始基础 token。
- `state_features`: 旧字段 `state[state_names]`，和 token 序列按时间对齐的状态特征。
- `participant_file_names`: 旧字段 `fileNames` 的原始值，例如 `031222-401.pkl`，用于 `legacy["fileNames"]` 精确兼容。
- `participant_ids`: 旧字段 `fileNames` 去掉 `.pkl` 后得到的被试名列表，用于 `structured` 输出。

### `PreparedStrategyStateData`

```python
@dataclass
class PreparedStrategyStateData:
    input_file_name: str
    token_sequence: list[str]
    n_positions: np.ndarray
    initial_tokens: list[str]
    state_features: pd.DataFrame
    participant_file_names: list[str]
    participant_ids: list[str]
    state_dependencies: StateDependencyGraph
```

职责：

- 表示删除 `N` 后、进入 grammar 学习前的数据。
- `n_positions` 记录 `N` 在原始序列中的位置，用于后续 skip-gram 检测。
- `state_features` 必须同步删除 `N` 对应行。

### `StateDependencyGraph`

```python
@dataclass
class StateDependencyGraph:
    conditions_by_state: list[list[int]]
```

职责：

- 表示 `StateGraph/*.pkl` 中 `G` 矩阵转换后的状态依赖条件。
- 等价于旧函数 `getConditionGraph()` 的返回值。

### `GrammarLearningResult`

```python
@dataclass
class GrammarLearningResult:
    grammar_tokens: list[str]
    probabilities: list[float]
    position_grammar: list[str]
    original_sequence: list[str]
    time_probabilities: np.ndarray
    frequencies: list[int]
    parsed_sequence: list[str]
    parsed_state_features: pd.DataFrame
    active_tokens: list[str]
    participant_file_names: list[str]
    participant_ids: list[str]
    components: list[list[str]]
```

职责：

- 表示核心 grammar 学习结果。
- 使用新 token 表示，例如 `"G-L"`、`"E-A"`。
- 不包含旧版占位符字段。

### `SkipGramResult`

```python
@dataclass
class SkipGramResult:
    found: bool
    count: int | float
```

职责：

- 表示 skip-gram 检测结果。
- 对应旧字段 `skipGram` 和 `skipGramNum`。

### `GenerateGrammarOutput`

```python
@dataclass
class GenerateGrammarOutput:
    legacy: dict[str, Any]
    structured: dict[str, Any]
```

职责：

- `legacy` 保存旧字段兼容结果，用于和旧 pickle 精确比较。
- `structured` 保存重新组织后的清晰结果，用于后续维护和分析。

## 函数接口设计

### `data_io.py`

```python
def list_strategy_state_files(strategy_sequence_dir: Path) -> list[str]

def load_strategy_state_data(
    path: Path,
    state_names: Sequence[str],
) -> StrategyStateData

def write_generate_grammar_output(
    output: GenerateGrammarOutput,
    path: Path,
) -> None
```

`list_strategy_state_files()` 默认返回排序后的文件名，保证日志和验证顺序稳定。每个文件独立处理，排序不改变输出内容。

`load_strategy_state_data()` 必须同时保留旧 `fileNames` 原始值作为 `participant_file_names`，并把去掉 `.pkl` 后的值作为 `participant_ids`。

### `state_graph.py`

```python
def load_state_dependency_graph(path: Path) -> StateDependencyGraph
```

该函数读取旧 `StateGraph` pickle，将 `G` 中每行值为 1 的列转换为 `conditions_by_state`。

### `token.py`

```python
def split_token(token: str) -> list[str]

def combine_tokens(parent_token: str, child_token: str) -> str

def token_length(token: str) -> int

def tokens_share_base_token(left: str, right: str) -> bool

def format_token(base_tokens: Sequence[str]) -> str
```

### `scoring.py`

```python
def count_state_combinations(
    data: np.ndarray,
    nstates: np.ndarray,
) -> np.ndarray

def bd_score(
    data_v: np.ndarray,
    data_parents: np.ndarray | Sequence,
    nstates_v: int,
    nstates_parents: np.ndarray | Sequence,
    alpha: float,
) -> tuple[float, np.ndarray]

def learn_state_condition_links(
    data: np.ndarray,
    nstates: np.ndarray,
    block_message: dict[int, list[int]],
    casual_num: int,
    block_num: int,
    effect_num: int,
    alpha: float,
    conditions: list[list[int]],
) -> tuple[np.ndarray, list, list, list]
```

实现要求：

- `count_state_combinations()` 与旧 `src.Utils.count` 行为一致。
- `bd_score()` 与旧 `src.bayesianScore.BDscore` 行为一致。
- `learn_state_condition_links()` 与旧 `src.bayesianScore.learnBayesNetBlock` 默认路径行为一致。
- 保留旧算法的 1/2 编码、Fortran reshape、`gammaln` 计算和 `bd1 / bd2 > 1` 判断语义。

### `grammar.py`

```python
class GrammarLearner:
    def __init__(self, params: GrammarLearningParams): ...

    def learn(
        self,
        token_sequence: list[str],
        initial_tokens: list[str],
        state_features: pd.DataFrame,
        state_dependencies: StateDependencyGraph,
        participant_ids: list[str],
    ) -> GrammarLearningResult: ...

    def detect_skip_gram(
        self,
        result: GrammarLearningResult,
        n_positions: np.ndarray,
    ) -> SkipGramResult: ...
```

内部辅助方法：

```python
def _parse_longest(
    tokens: list[str],
    grammar_tokens: list[str],
    state_features: pd.DataFrame | None = None,
) -> tuple[list[str], pd.DataFrame | None]

def _parse_probabilities(
    tokens: list[str],
    grammar_tokens: list[str],
) -> tuple[list[str], list[float], list[str], list[int]]

def _organize_discrete_data(
    tokens: list[str],
    active_tokens: list[str],
    state_features: pd.DataFrame,
    state_dependencies: StateDependencyGraph,
) -> OrganizedGrammarData

def _select_candidate_chunks(...)

def _has_converged(kl_history: list[float]) -> bool
```

`GrammarLearner` 不读文件、不写文件、不知道目录结构。

### `pipeline.py`

```python
def prepare_strategy_state_data(
    data: StrategyStateData,
    state_dependencies: StateDependencyGraph,
    removed_token: str = "N",
) -> PreparedStrategyStateData

def process_strategy_state_file(
    input_file_name: str,
    config: GenerateGrammarConfig,
) -> GenerateGrammarOutput

def run_generate_grammar(
    config: GenerateGrammarConfig,
) -> list[Path]
```

## Pipeline 运行流程

`process_strategy_state_file()` 的处理顺序必须对应旧默认路径：

1. 从 `strategy_sequence_dir / input_file_name` 读取 `StrategyStateData`。
2. 从 `state_graph_dir / input_file_name` 读取 `StateDependencyGraph`。
3. 从原始 `token_sequence` 找到 `removed_token`，默认是 `"N"`。
4. 删除 token 序列中的 `"N"`，记录原始位置为 `n_positions`。
5. 同步删除 `state_features` 中与 `"N"` 对应的行。
6. 调用 `GrammarLearner.learn()` 学习 grammar。
7. 调用 `GrammarLearner.detect_skip_gram()` 检测 `N -> E-A`。
8. 调用 `legacy.py` 构造旧字段兼容字典。
9. 调用 `structured.py` 构造新结构字典。
10. 写出新 pickle：

```python
{
    "legacy": {...},
    "structured": {...},
}
```

`run_generate_grammar()` 的处理顺序：

1. 校验配置。
2. 枚举并排序 `strategy_sequence_dir` 下的 pkl 文件。
3. 对每个文件调用 `process_strategy_state_file()`。
4. 将输出写入 `output_dir / input_file_name`。
5. 返回所有输出文件路径。

## Legacy 输出设计

新输出顶层包含两个字典：

```python
{
    "legacy": {...},
    "structured": {...},
}
```

`legacy` 必须完整保留旧字段和值：

```python
legacy = {
    "sets": ...,
    "pro": ...,
    "gram": ...,
    "sequence": ...,
    "time_pro": ...,
    "frequency": ...,
    "seq": ...,
    "state": ...,
    "S": ...,
    "fileNames": ...,
    "components": ...,
    "skipGram": ...,
    "skipGramNum": ...,
}
```

字段要求：

- 旧输出中存在的 key，新输出 `legacy` 中必须存在。
- 新输出 `legacy` 中对应 value 必须与旧输出一致。
- 集合、概率、频数、序列、状态表、skip-gram 结果都必须纳入比较。
- 若 pickle 字节级 MD5 无法一致，必须逐 key/value 证明一致。

兼容策略：

- 新核心算法只使用 `"G-L"` 形式的 token。
- `legacy.py` 在导出旧字段时，临时生成旧版占位符映射。
- 该映射用于构造旧字段 `seq`、`S`、`sets`、`components` 等。
- 占位符映射不进入 `GrammarLearningResult`，也不影响核心算法。

## Structured 输出设计

`structured` 删除旧字段冗余，并使用更清晰的命名：

```python
structured = {
    "source": {
        "input_file_name": "...",
        "participant_file_names": [...],
        "participant_ids": [...],
    },
    "parameters": {
        "state_names": [...],
        "chunk_alpha": 0.5,
        "condition_alpha": 0.5,
        "skip_gram_alpha": 0.5,
        "max_iterations": 100000,
        "convergence_window": 5,
        "convergence_kl_threshold": 0.05,
        "candidate_ratio_min": 1.0,
        "candidate_ratio_keep": 0.85,
        "min_pair_frequency": 0.05,
        "removed_token": "N",
        "skip_gram_target": "E-A",
        "skip_gram_min_offset": 2,
        "skip_gram_max_offset": 5,
        "skip_gram_min_frequency": 0.025,
    },
    "grammar": [
        {
            "token": "G-L",
            "base_tokens": ["G", "L"],
            "probability": 0.0,
            "frequency": 0,
            "time_probability": 0.0,
            "components": ["G", "L"],
        }
    ],
    "parsed": {
        "sequence": ["G-L", "E-A"],
        "state_features": "...",
        "position_grammar": ["G-L", "G-L"],
    },
    "skip_gram": {
        "target": "E-A",
        "found": False,
        "count": 0,
    },
}
```

`structured` 不参与旧结果一致性判定，但必须由同一次核心结果稳定生成。

## 运行脚本设计

新增运行入口：

```text
script/run_generate_grammar.py
```

默认运行：

```bash
PYTHONPATH=src conda run -n fmri python script/run_generate_grammar.py
```

默认读取：

```text
/home/zzh/project/Pacman/2.Pac-man/Monkey_Analysis/fmri_data_process/StrategySequence/
/home/zzh/project/Pacman/2.Pac-man/Monkey_Analysis/fmri_data_process/StateGraph/
```

默认写入 LoPS 仓库内：

```text
data/generate_grammar/refactored-output/grammar2/
```

运行脚本必须允许通过命令行参数覆盖输入目录、状态图目录、输出目录和关键学习参数。

## 验证设计

新增验证入口：

```text
script/validate_generate_grammar.py
```

默认命令：

```bash
PYTHONPATH=src conda run -n fmri python script/validate_generate_grammar.py
```

验证分两层。

### 模块级行为测试

新增测试：

```text
tests/test_generate_grammar_scoring.py
tests/test_generate_grammar_pipeline.py
```

当前 `fmri` 环境没有安装 `pytest`，因此测试使用标准库 `unittest`：

```bash
PYTHONPATH=src conda run -n fmri python -m unittest discover -s tests
```

必须验证：

- `count_state_combinations()` 与旧 `src.Utils.count` 一致。
- `bd_score()` 与旧 `src.bayesianScore.BDscore` 一致。
- `learn_state_condition_links()` 与旧 `src.bayesianScore.learnBayesNetBlock` 一致。
- pipeline 在代表性文件上能生成包含 `legacy` 和 `structured` 的输出。

### 脚本级一致性验证

固定基准：

```text
/home/zzh/project/Pacman/2.Pac-man/Monkey_Analysis/fmri_data_process/grammar2/
```

已知该基准可信：原始脚本在 LoPS sandbox 中全量重跑后，34/34 输出与该目录既有 pickle 文件 MD5 完全一致。

验证规则：

1. 新实现生成 34 个输出文件。
2. 对每个文件，读取新输出的 `legacy` 字典。
3. 先尝试比较新旧 pickle 相关内容的 MD5。
4. 若 MD5 不一致，逐文件逐 key 比较。
5. 旧 pickle 中存在的每个 key，新输出 `legacy` 中必须存在。
6. 每个旧 key 对应的 value 必须精确一致。
7. `np.ndarray` 使用精确数组比较。
8. `pd.DataFrame` 使用精确 DataFrame 比较。
9. list、str、bool、int、float 使用精确比较。
10. 不默认使用数值容差；若未来需要容差，必须先记录原因并经过确认。

验证通过后，`src/LoPS/temp` 不能保留本轮临时旧代码。

## 风险和约束

### Legacy 占位符兼容风险

新核心算法不使用旧占位符，但旧输出中的 `seq` 和 `S` 依赖旧占位符表示。为保证旧字段 value 一致，`legacy.py` 必须按旧算法的顺序生成兼容占位符映射。

该逻辑是兼容层，不是核心算法设计。

### token 表示和旧输出顺序风险

新核心 token 使用 `"G-L"` 形式，旧输出使用无分隔符或占位符形式。实现时必须确保：

- 旧字段 `sets` 使用旧格式。
- 旧字段 `components` 使用旧格式。
- 旧字段 `sequence` 与旧输出一致。
- 旧字段 `seq` 与旧占位符序列一致。
- 旧字段顺序和类型尽量保持旧输出习惯。

### 文件顺序风险

旧脚本使用 `os.listdir()`，新实现计划排序输入文件。由于每个文件独立输出，排序只影响日志顺序，不影响单文件输出内容。

### 数据写入边界

输入数据和旧基准只读。新输出、验证记录和临时产物只能写入 LoPS 仓库。

## 非目标

本阶段不做：

- 通用 grammar 学习框架。
- 支持任意 token 分隔符。
- 支持 `ghost4`。
- 支持 shuffle 数据。
- 自动扫描外部项目。
- GUI 或交互式工作台。
- 对未调用旧函数的迁移。

## 计划阶段落地要求

后续 `PLAN.md` 必须按本设计拆分任务，至少覆盖：

1. 建立模块骨架和配置对象。
2. 实现 token 工具。
3. 实现 state graph 和数据读取。
4. 实现 scoring 模块并做模块级行为测试。
5. 实现 grammar 学习核心。
6. 实现 legacy 兼容输出。
7. 实现 structured 输出。
8. 实现 pipeline 和运行脚本。
9. 实现验证脚本和 unittest 行为测试。
10. 全量运行并验证 `legacy` 与旧 `grammar2/` 逐 key/value 一致。

---

*本设计基于 Phase 2 深度分析报告、Phase 2 discuss 决策，以及用户对命名、token 表示、参数暴露和输出结构的修正意见。*
