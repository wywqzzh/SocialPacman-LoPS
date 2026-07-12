# 数据目录与当前运行方式

`data/` 是本仓库所有原始输入、中间结果、验证结果和视频产物的统一根目录。当前代码
在 `01–04` 后分支，不能再把所有编号目录理解成一条已经贯通的流水线。

## 目录语义

### 共享上游

- `00_raw_mat_data/`：原始 `.mat` session。
- `01_raw_subject_data/`：按 task/session 组织的 raw subject pickle。
- `02_frame_data/`：逐 frame joint-state 数据；`02_frame_data_csv/` 仅在 `--write-csv` 时生成。
- `03_preprocessed_frame_data/`：字段标准化并过滤当前流程不支持 trial 后的 frame 数据。
- `04_tile_data/`：P1/P2 位置与 mode、两只 ghost 位置组成的联合采样状态发生变化时
  保留的候选帧。
- `04_corrected_tile_data/`：在默认 13 帧窗口内压缩异步换 tile 候选、重算玩家动作后的
  tile 数据；不插入人工路径点。
- `constant_data/map_constants.pkl`：04、05 和 tile 视频使用的当前地图常量；历史 08
  仍直接读取同目录中的 `adjacent_map_fmri.csv` 与 `dij_distance_map_fmri.csv`。

### Utility、拟合与修正分支

- `05_cluster_global_utility_data/`：`05_calculate_utility.py` 的当前默认输出；包含七策略
  raw/normalized Q 和多个 Global 资源团候选。
- `05_range_utility_data/`：`05b_calculate_range_utility.py` 的实验输出。
- `05_utility_data/`：历史 05 输出，仍是 legacy 06 的默认输入。
- `06_weight_data/`：历史 GA 动态权重。
- `06_cluster_global_event_context_weight_data/`：06b 事件 context GA 权重。
- `06c_context_strategy_posterior_data/`：06c context 策略后验。
- `07_corrected_weight_data/`：历史 07 人工规则修正输出。
- `07c_context_strategy_posterior_corrected_data/`：07c 在保留 06c posterior 的同时新增的
 规则修正分数和标签。

其它 `06_*`、`07_*` 目录是此前实验参数或输入组合的产物，目录名不代表仍有同名专用
入口。复现实验时必须显式记录脚本、参数和输入目录。

### 历史下游和视频

- `08_feature_data/` 至 `11_grammar/`：历史 07 数据结构的下游目录；当前尚未接入 07c。
- `pacman_video/`：render table、图片帧、tile 检查视频和视频资源。
- `validation/`：旧新实现对比或修复验证；验证产物不得写入 `.planning/`。

> 状态提醒：现有 `05_cluster_global_utility_data`、06c 和 07c 本地结果生成于
> 2026-07-11 的 05 ghost 状态/raw Q 修复之前。重新运行之前，它们不是修复后的最终数据。

## 运行准备

以下命令都在仓库根目录执行。多数入口会自行加入 `src/`，仍统一写出
`PYTHONPATH=src` 以便脚本和模块调用方式一致。

项目要求 Python `>=3.10,<3.11`。`pyproject.toml` 当前没有声明所有可选依赖；运行
09/grammar、地图生成、tile 视频或测试前，还需分别确认 `scikit-learn`、`networkx`、
`imageio`、`pytest` 已安装。

## 01–04 共享上游

```bash
PYTHONPATH=src python script/01_mat_to_raw_subject_data.py
PYTHONPATH=src python script/02_raw_subject_data_to_frame_data.py
PYTHONPATH=src python script/03_frame_data_preprocess.py
PYTHONPATH=src python script/04_human_tile_data_preprocess.py
```

当前关键默认值：01 的 `--workers` 为 34；02 为 8；03 为 `min(34, CPU 数)`；04 为 8，
异步切换窗口 `--async-interval-frames` 为 13。内存不足时应降低 worker 数，而不是
照抄较大的并行参数。

只处理指定 session 时，01 接受 session 位置参数，02/04 接受 `session` 或
`task/session`，03 接受文件相对路径。具体格式以各脚本 `--help` 为准。

## 当前 05→06c→07c 后验分支

先用单文件验证。下面的文件名必须替换为实际存在的 `task/session.pkl`：

```bash
PYTHONPATH=src python script/05_calculate_utility.py \
  --single-file comp/<session>.pkl \
  --workers 1

PYTHONPATH=src python script/06c_fit_context_strategy_posterior.py \
  --single-file comp/<session>.pkl \
  --workers 1

PYTHONPATH=src python script/07c_revise_context_strategy_posterior.py \
  --single-file comp/<session>.pkl \
  --processes 1
```

单文件检查通过后，去掉 `--single-file` 即按嵌套的 `comp/*.pkl`、`coop/*.pkl` 批量运行。
默认参数包括：05 的 Evade/Approach 深度为 `6/20`；06c 的 beta 搜索范围为
`[0.05, 20]`、网格 81、最多 5 折、posterior 阈值 0.70、策略信息覆盖率阈值 0.50；
07c 的 scared 窗口为 34。

06c 输出的 `strategy_posterior` 只在 coverage 合格的行为策略间归一化。所有策略均不
合格时 posterior 为全零、candidate 为 `none`、最终标签为 `vague`；Null 均匀动作模型
仅保存为诊断，不参与 beta 或 posterior。

## 可选 06b 事件 context GA 分支

```bash
PYTHONPATH=src python script/06b_fit_dynamic_strategy_event_context.py \
  --single-file comp/<session>.pkl \
  --workers 1 \
  --segment-workers 1

PYTHONPATH=src python script/07_revise_human_weight.py \
  --input-dir data/06_cluster_global_event_context_weight_data \
  --output-dir data/07_cluster_global_corrected_weight_data \
  --processes 1
```

06b 默认 GA 种群/迭代数为 `100/500`、文件 seed 为 `20260610`。文件级并行和段落级
并行不要同时设得过大。06b 不再读取 `adjacent_map_fmri.csv`；best Global 直接使用
05 已保存的候选 utility。

## 可选 range utility 分支

```bash
PYTHONPATH=src python script/05b_calculate_range_utility.py
```

05b 只生成独立 utility。若继续使用 06/06b，必须显式指定相应输入、输出目录，并把
该组合视为实验配置，而不是脚本默认链路。

## 历史 GA 与 08–12

以下入口仍存在：

```text
06_dynamic_strategy_fitting.py -> 07_revise_human_weight.py
08_extract_features_human.py -> 09_human_fmri_data_preprocess.py
10_state_dependency_graph.py -> 11_generate_grammar.py -> 12_divide_person.py
```

06/07 默认读取 `05_utility_data → 06_weight_data → 07_corrected_weight_data`。08 仍使用
`input_dir.glob("*.pkl")` 扫描扁平目录，并按历史无前缀的单人权重/策略字段工作；当前
01–07 则使用 `task/*.pkl` 嵌套结构和双人前缀字段。尚未提供正式转换，因此不要把 07c
目录直接传给 08，也不要声称当前后验分支已完成 08–12 一致性验证。

## Tile 结果视频检查

07c 结果可直接交给 tile 渲染器：

```bash
PYTHONPATH=src python script/pacman_video/run_tile_video_renderer.py \
  --tile-root data/07c_context_strategy_posterior_corrected_data \
  --task comp \
  --session <session>.pkl \
  --trial <DayTrial> \
  --save-frames
```

不传 `--tile-root` 时，该脚本默认读取历史 `data/07_corrected_weight_data`。逐 frame 的
render-table 流程仍使用 `run_render_table.py`、`run_frame_renderer.py` 和
`run_video_renderer.py`。

## 运行后检查

```bash
find data/05_cluster_global_utility_data -type f -name "*.pkl" | wc -l
find data/06c_context_strategy_posterior_data -type f -name "*.pkl" | wc -l
find data/07c_context_strategy_posterior_corrected_data -type f -name "*.pkl" | wc -l
```

文件数量只能说明任务写出了结果，不能替代字段、attrs、数值和旧新一致性检查。重跑某
阶段时使用新的输出目录或明确清理该阶段及其下游；不要并行写入同一目录。
