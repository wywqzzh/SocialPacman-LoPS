# 数据目录与运行方式

正式数据按 `data/<stage>/<task>/<session>.pkl` 组织。`task` 当前包括 `comp` 和
`coop`；文件内同时保存公共状态以及实际存在的 P1/P2 字段。

## 目录

- `00_raw_mat_data/`：原始 MAT session。
- `01_raw_subject_data/`：MAT 字段整理后的逐帧 joint-state pickle。
- `02_frame_data/`：坐标修正、字段标准化，并整 trial 过滤 Ghost3/Ghost4 后的 frame 数据。
- `03_preprocessed_frame_data/`：保留正式分析字段并统一字段类型后的 frame 数据；本阶段
  不再执行 trial 过滤。
- `04_tile_data/`：P1/P2 位置与 mode、Ghost1/Ghost2 位置的联合状态变化候选帧。
- `04_corrected_tile_data/`：压缩异步 tile 切换并重算动作后的正式 tile 数据。
- `05_utility_data/`：七策略 Q 以及 Global、Energizer、Approach 候选 utility。
- `06_strategy_posterior_data/`：玩家事件 context、best target、beta 和策略 posterior。
- `07_revised_strategy_data/`：保留 posterior 并追加规则修正策略的最终结果。
- `constant_data/map_constants.pkl`：04、05 和 tile 视频共用的地图常量。
- `pacman_video/tile_video/`：当前策略视频。
- `pacman_video/tile_frame_images/`：与视频逐帧对应的 PNG。

## 地图常量

`map_constants.pkl` 不纳入版本控制。首次运行依赖地图的阶段前执行：

```bash
PYTHONPATH=src python script/constant_map/generate_map_constants.py --workers 8
```

生成器将内置的 28×36 地图、Ghost House 可走区域和左右通道连通性统一写入 pickle；
04、05 和视频只读取该文件，不再自行补边或修正地图。

## 主流程

```bash
PYTHONPATH=src python script/01_mat_to_raw_subject_data.py
PYTHONPATH=src python script/02_raw_subject_data_to_frame_data.py
PYTHONPATH=src python script/03_frame_data_preprocess.py
PYTHONPATH=src python script/04_human_tile_data_preprocess.py
PYTHONPATH=src python script/05_calculate_utility.py
PYTHONPATH=src python script/06_fit_context_strategy_posterior.py
PYTHONPATH=src python script/07_revise_context_strategy.py
```

05–07 单文件验证示例：

```bash
PYTHONPATH=src python script/05_calculate_utility.py \
  --single-file comp/<session>.pkl

PYTHONPATH=src python script/06_fit_context_strategy_posterior.py \
  --single-file comp/<session>.pkl --workers 1

PYTHONPATH=src python script/07_revise_context_strategy.py \
  --single-file comp/<session>.pkl --processes 1
```

默认关键参数包括：05 的 Local/Evade/Approach 深度为 `10/6/20`；Cluster Global
使用地图最短路聚类，聚类阈值为 `2`、参与当前方向距离计算的最小资源距离为 `2`、
最大目标距离为 `60`。05 仍暴露 `--global-depth` 和 `--global-ignore-depth`，但当前
正式 Global 实现没有读取这两个配置：正式 Global 的近距离排除固定为 `10`，修改这
两个参数不会改变输出。06 的 beta 搜索
范围为 `[0.05, 20]`、最多 5 折、posterior 阈值为 `0.70`、策略信息覆盖率阈值为
`0.50`；07 的 scared 窗口为 34 tile。

## Tile 视频

```bash
PYTHONPATH=src python script/pacman_video/run_tile_video_renderer.py \
  --task comp \
  --session <session>.pkl \
  --trial <DayTrial> \
  --save-frames
```

视频帧号、图片文件名和图中 context 均使用 0-based 编号；context 显示为闭区间。
当前 renderer 会同时验证 P1/P2 策略字段，因此仅能直接绘制双人 07 输出。它还会把
固定 Ghost House 内部显示区域补成白色空位；这些显示用格子不写回
`map_constants.pkl`，也不参与 utility 或连通性计算。

## 后续阶段

`08_extract_features_human.py` 至 `12_divide_person.py` 暂时保留，但尚未形成从当前
`07_revised_strategy_data` 开始的贯通流程。08 仍依赖历史字段、扁平目录以及
`dij_distance_map_fmri.csv`、`adjacent_map_fmri.csv`；09–12 已有部分当前结构接口，
但仍依赖 08 产物。正式接入时需要先改造 08 的双人字段、嵌套目录和地图读取接口。
