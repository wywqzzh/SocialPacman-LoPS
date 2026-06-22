# PreEstimation_fmri 重构分析与方案

## 目标脚本

- 原始脚本：`/home/zzh/project/Pacman/Language-of-Problem-Solving/Behavior_Analysis/HierarchicalModel/PreEstimation_fmri.py`
- 运行环境：conda `fmri` 环境。
- 本轮目标：重构 fMRI corrected tile 数据的策略 utility 预计算流程。
- 约束：正式新代码不能依赖旧项目代码、旧项目数据目录或绝对路径；修改业务实现前必须先确认方案。

## 原始功能概述

原始脚本的核心功能是读取一个或多个 corrected tile 数据文件，为每一行游戏状态计算多个策略的 Q 向量，并把这些 Q 向量追加为新列后保存。

旧脚本当前主入口只处理一个文件：

- 输入目录：`../../fmri_data_process/fmriCorrectedTileData/`
- 主入口指定文件：`180522-502-21-Nov-2022-1.pkl`
- 输出目录：`../../fmri_data_process/fmriUtilityData/`
- 输出文件名：`{subject}-with_Q.pkl`

旧代码中已有批处理函数 `preEstimation(filename_list, save_base)`，可以处理多个文件。此前已经用相同输入执行过 Language-of-Problem-Solving 与 Monkey_Analysis 两份 `PreEstimation_fmri.py`，34 个被试输出字节级、DataFrame 级和 Q 列级均完全一致。

## 输入数据

每个输入 `.pkl` 是一个 `pandas.DataFrame`，代表一个被试的 corrected tile 数据。当前 LoPS 仓库中对应数据已经存在于：

- `data/human_tile_data_preprocess/corrected_tile_data/`

样例字段包括：

- 基础索引与帧定位：`Unnamed: 0`, `DayTrial`, `Step`, `frameIndex`
- 游戏状态：`pacmanPos`, `ghost1Pos`, `ghost2Pos`, `ghost3Pos`, `ghost4Pos`
- 鬼状态：`ifscared1`, `ifscared2`, `ifscared3`, `ifscared4`
- 行为字段：`pacman_dir`, `JoyStick`
- 地图与奖励对象：`Map`, `beans`, `energizers`, `fruitPos`, `fruitType`

旧脚本也兼容少数字段变体：

- 如果存在 `ghost1_status` / `ghost2_status`，只读取两个鬼状态；否则读取 `ifscared1..4`。
- 如果存在 `fruitType`，使用 `fruitType`；否则使用 `Reward`。
- 如果存在 `fruit_pos`，使用 `fruit_pos`；否则使用 `fruitPos`。

## 辅助数据

旧脚本通过 `Utils.FileUtils_fmri` 读取地图常量：

- `../../Data/constant/dij_distance_map_fmri.csv`
- `../../Data/constant/adjacent_map_fmri.csv`

读取后得到：

- `adjacent_data`：每个位置四个方向的相邻位置。
- `locs_df`：任意两个位置的 Dijkstra 距离字典。
- `adjacent_path`：位置对之间的路径表；在目标流程中被读取，但实际没有参与 Q 计算。
- `reward_amount`：固定奖励字典。

`FileUtils_fmri.py` 在导入时会执行 `os.chdir(os.path.dirname(os.path.abspath(__file__)))`，这是旧实现的重要副作用。新实现不能保留这种导入期改变工作目录的行为。

## 输出数据

输出仍是输入 DataFrame，追加 Q 列：

- `global_Q`
- `local_Q`
- `evade_blinky_Q`
- `evade_clyde_Q`
- `evade_ghost3_Q`
- `evade_ghost4_Q`
- `approach_Q`
- `energizer_Q`
- `no_energizer_Q`

每个 Q 值是四方向向量，方向顺序固定为：

1. `left`
2. `right`
3. `up`
4. `down`

## 执行流程

1. 读取 corrected tile DataFrame，并 `reset_index(drop=True)`。
2. 读取地图相邻关系、距离表和奖励字典。
3. 对每一行数据：
   - 解析 Pacman 位置、豆子、能量豆、四个鬼位置、鬼状态、水果位置和水果类型。
   - 对 fMRI 地图隧道位置做修正：
     - `(0, 18)` / `(-1, 18)` 转为 `(1, 18)`
     - `(30, 18)` / `(31, 18)` 转为 `(29, 18)`
   - 对鬼出生墙内位置做修正：
     - `(14, 20)` 转为 `(14, 19)`
     - `(15, 20)` 转为 `(15, 19)`
     - `(16, 20)` 转为 `(16, 19)`
   - 分别构造 9 个策略对象并计算 Q 向量。
4. 将 9 个 Q 列追加到 DataFrame。
5. 用 pickle 保存为 `{subject}-with_Q.pkl`。

## 策略功能拆解

### Global

`GlobalAgent_beyond10.SimpleGlobal` 不使用树搜索。它把 Pacman 周围地图划成四个方向区域，统计每个方向区域内、距离当前 Pacman 位置大于 `ignore_depth=10` 的 bean 数量，作为四方向 Q 值。

默认参数：

- `depth=15`
- `ignore_depth=10`
- `reward_coeff=1.0`
- `risk_coeff=0.0`

### Local

`LocalAgent.PathTree` 使用深度优先/广度展开的路径树搜索，关注吃豆、能量豆、水果等即时奖励，同时保留鬼风险逻辑。目标脚本中 `reward_coeff=1.0`、`risk_coeff=0.0`，所以最终 Q 只受奖励项影响。

### Evade

`EvadeAgent_fmri.EvadeTree` 与 Local 使用相同的树展开和相同的奖励逻辑，但初始化时只保留指定的单个鬼：

- `blinky`
- `clyde`
- `ghost3`
- `ghost4`

目标脚本中 `reward_coeff=0.0`、`risk_coeff=1.0`，所以最终 Q 只受该单个鬼的风险项影响。

### Approach

`ApproachAgent.ApproachTree` 使用相同的树展开，但奖励逻辑改为接近或吃鬼。目标脚本中 `reward_coeff=1.0`、`risk_coeff=0.0`。

### Energizer

`EnergizerAgent.EnergizerTree` 使用相同的树展开，但奖励逻辑只关注能量豆。目标脚本中 `reward_coeff=1.0`、`risk_coeff=0.0`。

### NoEnergizer

`NoEnergizerAgent.NoEnerTree` 使用相同的树展开，但把吃能量豆视为风险惩罚，并不把能量豆当作奖励。目标脚本中 `reward_coeff=0.0`、`risk_coeff=1.0`。

## 冗余与脆弱点

1. 5 个树搜索 Agent 的 `_construct()` 完全同构，路径展开逻辑高度重复。
2. Local 与 Evade 的 `_computeReward()` 完全同构，Evade 只是在初始化时过滤出单个鬼。
3. 多个 Agent 的 `_attachNode()`、`_descendantUtility()` 和 `nextDir()` 基本同构。
4. 旧代码大量使用 `eval()` 解析位置和列表，数据边界不清晰；新实现应使用安全解析。
5. 旧代码用 `np.array([...])` 包装四个鬼位置；在当前 NumPy 版本下，如果 `ghost3Pos` / `ghost4Pos` 是空列表字符串，可能触发 ragged array 错误。新实现应显式解析为列表/元组结构。
6. 旧代码导入辅助模块时改变当前工作目录，容易污染其它脚本。
7. `preEstimation_parallelize()` 少传了 `filename` 参数，按当前代码直接调用会报错；主流程没有使用它。
8. 多个参数写死在 `_individualEstimation()` 内部，不利于验证和复用。
9. `adjacent_path` 被读取但没有实际使用，应在新设计中不作为核心计算依赖。

## 随机性分析

Agent 的 `nextDir(return_Q=True)` 中会构造随机扰动：

- `np.random.uniform(...)`
- `makeChoice()` 中使用 `np.random.choice(...)`

但目标脚本把：

- `randomness_coeff = 0.0`
- `laziness_coeff = 0.0`

因此随机扰动不会改变保存的 Q 向量。`makeChoice()` 返回的方向选择在目标脚本中也没有保存，目标输出只使用 `result[1]`，即 Q 向量。

新实现可以不把随机选择作为默认输出的一部分；如果保留可选方向选择接口，需要显式支持 seed，但本轮验证重点是 Q 列完全一致。

## 是否需要 src 模块

需要。

原因是该脚本虽然入口只是数据预计算，但其核心不是简单 I/O，而是一组复杂且重复的策略 utility 模型。若只在 `script/` 中重写，会继续把可复用模型逻辑与批处理流程绑在一起，不利于后续策略验证、模型拟合和单元级一致性测试。

但文件拆分不应过细。建议只创建一个正式模块文件，集中承载策略 utility 的核心模型。

## 建议文件设计

### 正式模块

- `src/LoPS/hierarchical_utility.py`

该文件包含：

- 地图常量读取。
- corrected tile 行数据解析与标准化。
- 策略参数配置。
- 共享路径树搜索引擎。
- 各策略的奖励/风险计算。
- 单个 DataFrame 的 Q 计算。
- 单文件和目录级处理函数。

不新增 `pipeline.py`、`io.py`、`utils.py` 等辅助文件，避免结构过碎。

### 运行脚本

- `script/hierarchical_utility/run_preestimate_fmri_utility.py`

职责：

- 传入 input/output/constant 路径。
- 传入策略参数和并行 worker 数。
- 默认读取当前仓库 `data/` 下路径。
- 运行 34 个被试或指定单文件。

### 验证脚本

- `script/hierarchical_utility/validate_preestimate_fmri_utility.py`

职责：

- 在 `src/LoPS/temp/` 中创建临时旧实现副本或临时运行包装。
- 用当前仓库内相同输入数据和相同常量运行旧实现与新实现。
- 输出严格对比报告到 `data/hierarchical_utility/validation/`。
- 验证结束后清理 `src/LoPS/temp/` 中的临时代码。

验证逻辑不放入正式模块，避免旧格式和旧路径污染核心设计。

## 数据目录设计

建议为本轮功能创建独立数据目录：

- 输入 corrected tile 数据：`data/hierarchical_utility/corrected_tile_data/`
- 地图常量：`data/hierarchical_utility/constant_data/`
- 新版本输出：`data/hierarchical_utility/utility_data/`
- 验证输出：`data/hierarchical_utility/validation/`

可以从当前仓库已有目录复制数据：

- corrected tile 数据来自 `data/human_tile_data_preprocess/corrected_tile_data/`
- 地图常量可从当前仓库已有 constant 数据目录复制或从旧项目复制后纳入当前仓库

正式模块不写默认路径，默认路径只写在运行脚本中。

## 详细设计

### 数据结构

建议使用少量 `dataclass` 表达核心概念：

- `MapData`
  - `adjacent_by_position`
  - `distance_by_position`
  - `reward_amount`

- `UtilityConfig`
  - `randomness_coeff`
  - `laziness_coeff`
  - `global_depth`
  - `global_ignore_depth`
  - `local_depth`
  - `evade_depth`
  - `approach_depth`
  - `energizer_depth`
  - `no_energizer_depth`
  - 各策略阈值与 reward/risk 系数

- `FrameState`
  - 单行数据解析后的规范状态：
    - Pacman 位置
    - 四个鬼位置
    - 鬼状态
    - beans
    - energizers
    - fruit 类型和位置
    - last direction

- `SearchNode`
  - 树搜索内部节点，保存当前路径位置、累计 reward/risk/utility、剩余对象集合、鬼状态和父子关系。

### 公共函数

- `load_map_data(adjacent_map_path, distance_map_path)`
  - 读取地图相邻表、距离表和 reward 字典。

- `load_corrected_tile_data(input_path)`
  - 读取单个 corrected tile `.pkl`。

- `parse_frame_state(row, columns)`
  - 将一行 DataFrame 解析为 `FrameState`。
  - 显式处理字符串、列表、空值、隧道修正和鬼出生点修正。

- `estimate_utility_for_dataframe(frame_data, map_data, config)`
  - 给一个被试 DataFrame 追加 9 个 Q 列。

- `process_utility_file(input_path, output_path, map_data, config)`
  - 处理单个文件。

- `process_utility_directory(input_dir, output_dir, map_data, config, workers)`
  - 批处理目录。

### 策略类

保留一个共享树搜索基类或引擎：

- `PathTreeUtilityStrategy`
  - 负责路径展开、避免立即回头、叶节点 utility 聚合和四方向 Q 计算。
  - 不直接决定奖励/风险含义。

不同策略只实现奖励/风险规则：

- `LocalUtilityStrategy`
- `EvadeUtilityStrategy`
- `ApproachUtilityStrategy`
- `EnergizerUtilityStrategy`
- `NoEnergizerUtilityStrategy`
- `GlobalUtilityStrategy`

`GlobalUtilityStrategy` 不继承树搜索策略，因为它本来不是树模型。

### 关键一致性要求

新实现必须保留：

- 四方向顺序：`left`, `right`, `up`, `down`
- 子节点展开顺序：`left`, `right`, `up`, `down`
- 不允许立即反向走回上一格的规则
- fMRI 隧道位置修正
- 鬼出生墙内位置修正
- 旧策略中已经实际生效的 reward/risk 公式
- 默认参数值和各策略 reward/risk 系数
- Q 列名和 Q 向量形态

新实现可以改变：

- 内部函数拆分。
- 是否使用 `anytree`；建议改为轻量 `SearchNode`，但必须验证结果完全一致。
- 是否计算未保存的方向 choice；默认只保存 Q。
- 是否读取 `adjacent_path`；目标流程未使用，可不作为核心依赖。

## 实施计划

1. 创建 `data/hierarchical_utility/` 数据目录，复制当前仓库内 corrected tile 数据和地图常量。
2. 创建 `src/LoPS/hierarchical_utility.py`。
3. 先实现地图读取、行解析和配置对象。
4. 实现共享路径树搜索引擎。
5. 实现 5 个树搜索策略和 1 个 global 策略。
6. 实现 DataFrame、单文件、目录级处理接口。
7. 创建 `script/hierarchical_utility/run_preestimate_fmri_utility.py`。
8. 创建 `script/hierarchical_utility/validate_preestimate_fmri_utility.py`。
9. 在 conda `fmri` 或 `LoPS` 环境中运行新脚本生成 34 个被试输出。
10. 运行验证脚本，对比新旧输出。
11. 清理 `src/LoPS/temp/` 中的临时旧代码。

## 验证计划

使用当前仓库数据作为唯一正式输入：

- `data/hierarchical_utility/corrected_tile_data/`
- `data/hierarchical_utility/constant_data/`

验证内容：

1. 文件集合完全一致。
2. 每个输出 DataFrame 的行数、列数、列名完全一致。
3. 输入原有列完全一致。
4. 9 个 Q 列逐行逐元素完全一致，使用 `np.array_equal`。
5. 若 pickle 字节级一致，也记录；但核心判定以 DataFrame 与 Q 值完全一致为准。
6. 输出整体报告到 `data/hierarchical_utility/validation/validation_report.json`。

如果新旧 Q 输出不一致，优先记录：

- 被试文件名。
- 行号。
- 策略名。
- 新旧 Q 值。
- 该行解析后的 `FrameState`。

## 需要确认的问题

1. 正式输出是否继续采用“输入 corrected tile 表追加 9 个 Q 列”的表结构？我建议保留，因为这就是该脚本的自然数据产物，也便于下游模型拟合使用。
2. 是否接受只创建一个正式模块文件 `src/LoPS/hierarchical_utility.py`？我建议这样做，避免文件拆分过细。
3. 是否接受额外创建一个独立验证脚本 `script/hierarchical_utility/validate_preestimate_fmri_utility.py`？我建议创建，避免验证逻辑污染运行脚本和正式模块。
4. 是否将本轮数据目录命名为 `data/hierarchical_utility/`？如果你更希望突出 fMRI 预估过程，也可以改为 `data/preestimate_fmri_utility/`。
