# LoPS

LoPS 是 Pacman 行为数据分析流程的重构仓库。当前主分析流程已经整理为
`script/` 根目录下按执行顺序编号的入口脚本，所有默认输入和输出都写入
`data/` 下按阶段编号组织的目录。

主分析数据文件统一使用 `{subject/session}.pkl` 命名，例如
`031222-401-03-Dec-2022-1.pkl`。阶段信息只体现在目录名中，不再写入文件名。

## 主要目录

- `src/LoPS/`：可复用正式模块。
- `src/LoPS/pacman_preprocess/`：Pacman 原始行为数据预处理模块，负责 `.mat -> raw_subject_data -> frame_data -> preprocessed frame_data`。
- `script/`：主分析流程入口，`01` 到 `12` 对应完整非视频数据处理流程。
- `script/pacman_video/`：视频渲染相关入口。
- `data/`：当前流程的原始输入、中间结果、最终输出和视频数据。
- `docs/`：人工说明和阶段性审计文档。
- `docs/data_flow.html`：带左侧目录导航的数据流程说明页面。

## 主分析流程

| 顺序 | 脚本 | 默认输入 | 默认输出 |
|---:|---|---|---|
| 1 | `script/01_mat_to_raw_subject_data.py` | `data/00_raw_mat_data` | `data/01_raw_subject_data` |
| 2 | `script/02_raw_subject_data_to_frame_data.py` | `data/01_raw_subject_data` | `data/02_frame_data` |
| 3 | `script/03_frame_data_preprocess.py` | `data/02_frame_data` | `data/03_preprocessed_frame_data` |
| 4 | `script/04_human_tile_data_preprocess.py` | `data/03_preprocessed_frame_data` | `data/04_tile_data`, `data/04_corrected_tile_data` |
| 5 | `script/05_calculate_utility.py` | `data/04_corrected_tile_data` | `data/05_utility_data` |
| 6 | `script/06_dynamic_strategy_fitting.py` | `data/05_utility_data` | `data/06_weight_data` |
| 7 | `script/07_revise_human_weight.py` | `data/06_weight_data` | `data/07_corrected_weight_data` |
| 8 | `script/08_extract_features_human.py` | `data/07_corrected_weight_data` | `data/08_feature_data`, `data/08_discrete_feature_data` |
| 9 | `script/09_human_fmri_data_preprocess.py` | `data/08_discrete_feature_data` | `data/09_fmri_discrete_feature_data_ghost2`, `data/09_fmri_formed_data_ghost2`, `data/09_strategy_sequence` |
| 10 | `script/10_state_dependency_graph.py` | `data/09_strategy_sequence` | `data/10_state_dependency_graph_data` |
| 11 | `script/11_generate_grammar.py` | `data/09_strategy_sequence`, `data/10_state_dependency_graph_data` | `data/11_grammar` |
| 12 | `script/12_divide_person.py` | `data/11_grammar` | 只打印结果，不保存文件 |

完整命令见 `data/README.md`。

## 视频流程

视频流程独立于主分析链路，默认读取 `data/02_frame_data` 和
`data/pacman_video` 下的数据：

| 顺序 | 脚本 | 默认输入 | 默认输出 |
|---:|---|---|---|
| 1 | `script/pacman_video/run_render_table.py` | `data/02_frame_data`, `data/pacman_video/grammar_data` | `data/pacman_video/render_data` |
| 2 | `script/pacman_video/run_frame_renderer.py` | `data/pacman_video/render_data` | `data/pacman_video/frame_images` |
| 3 | `script/pacman_video/run_video_renderer.py` | `data/pacman_video/frame_images` | `data/pacman_video/video_data` |

## 运行约束

- 正式模块不得依赖旧项目路径。
- 默认运行数据只来自当前仓库的 `data/`。
- 若需要和历史结果对比，应使用独立验证脚本或一次性验证代码，不把旧格式适配逻辑写入正式模块。
