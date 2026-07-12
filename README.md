# LoPS

LoPS 是 Social Pacman 行为数据分析流程的重构仓库。当前代码以 `01–04` 作为共享
预处理上游；`05` 之后同时保留历史 GA 流程、range utility 实验流程、cluster-global
事件 context 流程，以及当前的 context 策略后验流程。因此，仓库已经不是一条简单的
`01 → 12` 直线流水线。

当前 01–07 的 joint-state 数据按 `data/<stage>/<task>/<session>.pkl` 组织，例如
`data/04_corrected_tile_data/comp/<session>.pkl`。阶段信息位于目录名中，文件名保持
session 名称。

## 主要目录

- `src/LoPS/`：可复用正式模块。
- `script/`：分析入口；数字后缀 `b/c` 表示同阶段的独立实验分支。
- `script/pacman_video/`：逐 frame 渲染和 tile 结果检查入口。
- `data/`：输入、中间结果、验证结果和视频产物；本地已有文件不等于已完成当前版本重跑。
- `docs/`：算法说明和审计记录。
- `.planning/`：历史规划与代码库快照，不作为当前运行接口的唯一依据。

## 当前数据流

### 共享上游

| 阶段 | 脚本 | 默认输入 | 默认输出 |
|---:|---|---|---|
| 01 | `script/01_mat_to_raw_subject_data.py` | `data/00_raw_mat_data` | `data/01_raw_subject_data` |
| 02 | `script/02_raw_subject_data_to_frame_data.py` | `data/01_raw_subject_data` | `data/02_frame_data` |
| 03 | `script/03_frame_data_preprocess.py` | `data/02_frame_data` | `data/03_preprocessed_frame_data` |
| 04 | `script/04_human_tile_data_preprocess.py` | `data/03_preprocessed_frame_data` | `data/04_tile_data`, `data/04_corrected_tile_data` |

04 不做固定间隔抽帧，也不插入人工路径点。它先按 P1/P2 的位置与 mode、两只 ghost
的位置保留联合状态变化候选帧，再在默认 13 帧窗口内压缩多对象异步换 tile 产生的
冗余候选帧，并重新计算玩家动作与合法性。

### 05–07 分支

| 分支 | 脚本与默认目录 | 当前语义 |
|---|---|---|
| Cluster-global utility | `05_calculate_utility.py`: `04_corrected_tile_data → 05_cluster_global_utility_data` | 当前 05 默认入口；保存七策略 Q，并为 Global 保存多个资源团候选。 |
| Range utility | `05b_calculate_range_utility.py`: `04_corrected_tile_data → 05_range_utility_data` | 基于地图距离、半径和衰减的独立实验 utility。 |
| 历史 GA | `06_dynamic_strategy_fitting.py`: `05_utility_data → 06_weight_data`；`07_revise_human_weight.py`: `06_weight_data → 07_corrected_weight_data` | 保留的 GA 权重拟合与人工修正接口。 |
| 事件 context GA | `06b_fit_dynamic_strategy_event_context.py`: `05_cluster_global_utility_data → 06_cluster_global_event_context_weight_data` | 使用玩家私有硬边界、掉头/队友事件软边界和 context 内 best Global；仍拟合 GA 权重。其输出可显式传给 `07_revise_human_weight.py`。 |
| Context 后验 | `06c_fit_context_strategy_posterior.py`: `05_cluster_global_utility_data → 06c_context_strategy_posterior_data`；`07c_revise_context_strategy_posterior.py`: `06c_context_strategy_posterior_data → 07c_context_strategy_posterior_corrected_data` | 当前概率模型分支；按信息覆盖率筛选策略，拟合文件级 beta，保存 posterior；07c 另存人工规则标签，不覆盖原 posterior。 |

`05_cluster_global_utility_data`、`06c_context_strategy_posterior_data` 和
`07c_context_strategy_posterior_corrected_data` 中现有本地文件是在 2026-07-11 的
05 数值修复之前生成的。重新执行 05→06c→07c 前，不应把这些文件视为修复后的最终结果。

### 08–12 历史下游

`08_extract_features_human.py` 到 `12_divide_person.py` 仍保留原编号和默认目录，但 08
仍扫描扁平 `*.pkl` 并读取历史无玩家前缀的单人字段，尚未为 06b/06c 的嵌套双人前缀
字段和 07c posterior/revised 字段建立正式适配。因此它们不是当前 06c→07c 分支的
已贯通下游。

## 视频流程

仓库提供两种互不依赖的渲染方式：

- 逐 frame 展示：`run_render_table.py → run_frame_renderer.py → run_video_renderer.py`，
  默认读取 `data/02_frame_data` 与 `data/pacman_video/grammar_data`。
- tile 结果检查：`run_tile_video_renderer.py`，直接读取 07 类结果和
  `data/constant_data/map_constants.pkl`。默认仍指向历史
  `data/07_corrected_weight_data`；检查 07c 时必须显式传入
  `--tile-root data/07c_context_strategy_posterior_corrected_data`。

## 环境与运行约束

- Python 版本为 `>=3.10,<3.11`；正式依赖以 `pyproject.toml` 为准。
- `scikit-learn`（09、grammar）、`networkx`（地图常量生成）、`imageio`（tile 视频）和
  `pytest`（部分测试）被代码直接使用，但当前未写入 `pyproject.toml`，运行对应入口前需另行安装。
- 所有正式默认路径都位于当前仓库 `data/`；正式模块不得读取旧项目绝对路径。
- 不要并行覆盖同一输出目录。调试 05/06b/06c/07c 时优先使用 `--single-file`。
- 当前准确运行命令、分支选择和下游限制见 `data/README.md`；详细算法见
  `docs/data_flow.html` 和 `docs/context_strategy_posterior_inference.md`。
