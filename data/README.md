# 完整数据处理流程数据目录

`data/` 是完整数据处理流程的统一数据根目录。原始 `.mat` 数据、各阶段
中间结果、最终 grammar 输出和视频辅助数据都保存在本目录下。

## 目录结构

- `00_raw_mat_data/`：原始 `.mat` 数据，作为完整流程初始输入。
- `01_raw_subject_data/`：由 raw mat 转换得到的单被试逐 trial 数据。
- `02_frame_data/`：由 raw subject 数据转换得到的 frame data。
- `03_preprocessed_frame_data/`：标准化后的分析用 frame data。
- `04_tile_data/`：从预处理 frame data 抽样得到的 tile data。
- `04_corrected_tile_data/`：插入缺失路径点并修正位置后的 tile data。
- `05_utility_data/`：集中计算、修正并归一化后的 utility 数据。
- `06_weight_data/`：动态策略拟合得到的权重数据。
- `07_corrected_weight_data/`：规则修正后的权重数据。
- `08_feature_data/`：连续特征数据。
- `08_discrete_feature_data/`：离散特征数据。
- `09_fmri_discrete_feature_data_ghost2/`：ghost2 离散特征数据。
- `09_fmri_formed_data_ghost2/`：ghost2 formed 数据。
- `09_strategy_sequence/`：grammar 和状态图使用的策略序列。
- `10_state_dependency_graph_data/`：状态依赖图结果。
- `11_grammar/`：最终 grammar 输出。
- `constant_data/`：fMRI 迷宫常量表。
- `pacman_video/`：视频渲染相关输入、图片帧和视频输出。

## 运行命令

以下命令都在仓库根目录执行：

```bash
cd /home/zzh/project/LoPS
```

1. raw mat 到 raw subject data：

```bash
PYTHONPATH=src python script/01_mat_to_raw_subject_data.py \
  --raw-root data/00_raw_mat_data \
  --output-dir data/01_raw_subject_data \
  --workers 34
```

2. raw subject data 到 frame data：

```bash
PYTHONPATH=src python script/02_raw_subject_data_to_frame_data.py \
  --input-dir data/01_raw_subject_data \
  --output-dir data/02_frame_data \
  --workers 34
```

3. frame data 标准字段预处理：

```bash
PYTHONPATH=src python script/03_frame_data_preprocess.py \
  --input-dir data/02_frame_data \
  --output-dir data/03_preprocessed_frame_data \
  --workers 34
```

4. preprocessed frame data 到 tile data 和 corrected tile data：

```bash
PYTHONPATH=src python script/04_human_tile_data_preprocess.py \
  --frame-dir data/03_preprocessed_frame_data \
  --tile-dir data/04_tile_data \
  --corrected-dir data/04_corrected_tile_data
```

5. corrected tile data 到集中 utility：

```bash
PYTHONPATH=src python script/05_calculate_utility.py \
  --input-dir data/04_corrected_tile_data \
  --output-dir data/05_utility_data \
  --constant-dir data/constant_data \
  --workers 34
```

6. 动态策略权重拟合：

```bash
PYTHONPATH=src python script/06_dynamic_strategy_fitting.py \
  --input-dir data/05_utility_data \
  --output-dir data/06_weight_data \
  --adjacent-map data/constant_data/adjacent_map_fmri.csv \
  --workers 8 \
  --segment-workers 32
```

7. 修正人类策略权重：

```bash
PYTHONPATH=src python script/07_revise_human_weight.py \
  --input-dir data/06_weight_data \
  --output-dir data/07_corrected_weight_data
```

8. 提取连续特征和离散特征：

```bash
PYTHONPATH=src python script/08_extract_features_human.py \
  --input-dir data/07_corrected_weight_data \
  --constant-dir data/constant_data \
  --feature-output-dir data/08_feature_data \
  --discrete-output-dir data/08_discrete_feature_data
```

9. 生成 human fMRI 预处理数据和 strategy sequence：

```bash
PYTHONPATH=src python script/09_human_fmri_data_preprocess.py \
  --raw-discrete-dir data/08_discrete_feature_data \
  --ghost2-discrete-dir data/09_fmri_discrete_feature_data_ghost2 \
  --formed-ghost2-dir data/09_fmri_formed_data_ghost2 \
  --strategy-sequence-dir data/09_strategy_sequence
```

10. 生成状态依赖图：

```bash
PYTHONPATH=src python script/10_state_dependency_graph.py \
  --input-dir data/09_strategy_sequence \
  --output-dir data/10_state_dependency_graph_data
```

11. 生成 grammar：

```bash
PYTHONPATH=src python script/11_generate_grammar.py \
  --strategy-sequence-dir data/09_strategy_sequence \
  --state-graph-dir data/10_state_dependency_graph_data \
  --output-dir data/11_grammar \
  --quiet
```

12. 可选：DividePerson 人群划分后处理，只打印结果，不保存数据：

```bash
PYTHONPATH=src python script/12_divide_person.py \
  --grammar-dir data/11_grammar
```

## 运行后检查

每一步跑完后，可以用下面的命令快速检查输出文件数量：

```bash
find data -maxdepth 2 -type f -name "*.pkl" | wc -l
find data/11_grammar -type f | wc -l
```

如果需要重新跑某一步，建议只清理该步骤及其下游输出目录，保留上游结果。
