# 数据流字段审计表

本文审计当前已经跑通的非视频主流程字段使用情况。统计口径是：字段必须被用于计算、
筛选、分组、索引、排序、特征生成、模型输入或结果组装；仅仅因为上游 DataFrame
透传而出现的字段不算“被使用”。视频渲染字段不纳入主流程审计。

## 流程范围

当前主链路：

1. `raw_mat_data -> raw_subject_data`
2. `raw_subject_data -> frame_data`
3. `frame_data -> tile_data -> corrected_tile_data`
4. `corrected_tile_data -> utility_data`
5. `utility_data -> corrected_utility_data`
6. `corrected_utility_data -> weight_data`
7. `weight_data -> corrected_weight_data`
8. `corrected_weight_data -> feature_data / discrete_feature_data`
9. `discrete_feature_data -> fmri_formed_data_ghost2 -> strategy_sequence`
10. `strategy_sequence -> state_dependency_graph`
11. `strategy_sequence + state_dependency_graph -> grammar`
12. `grammar -> divide_person_result`

## 主字段审计表

| 字段或字段组 | 首次出现/生成阶段 | 当前格式 | 被使用的位置 | 推荐标准格式 | 最早规范化阶段 | 保留建议 |
|---|---|---|---|---|---|---|
| `DayTrial` | `raw_subject_data` | 字符串，形如 `1-2-...` | raw 去重；frame 数字排序；tile/fitting/revise 分 trial；feature 传给离散数据；fMRI 构造 `game` | 字符串 + 独立派生 `trial_id` / `game_id` | `frame_data` | 主链路保留，但后续应使用标准 `trial_id/game_id` |
| `Step` | `raw_subject_data`，frame 阶段转 0-based | int，corrected tile 后可能因插入行变 object | raw 去重、排序；frame 行序生成 | `int64`，0-based | `frame_data` | frame/tile 阶段保留；utility 后若不再索引可丢弃 |
| `Unnamed: 0` | `frame_data` | int 行号，corrected 后 object | tile 修正时生成 `frameIndex`；feature/discrete 作为追踪列透传 | 改名为 `frame_id: int64` | `frame_data` | 保留但应改名，不建议继续叫 `Unnamed: 0` |
| `frameIndex` | `corrected_tile_data` | object/int | tile 修正时回指 frame 区间插入中间点 | `frame_id: int64` | `corrected_tile_data` | tile 修正后若无下游计算依赖，可不进入 utility |
| `pacmanPos` | `frame_data` | tuple，部分脚本仍兼容字符串 | tile 连续性和方向；utility 状态；correct Q；dynamic 可走方向；feature 距离 | `tuple[int, int]` | `frame_data` | 主链路核心字段，必须保留到 feature 阶段 |
| `ghost1Pos`, `ghost2Pos`, `ghost3Pos`, `ghost4Pos` | `frame_data` | tuple 或空 list `[]` | tile ghost 坐标修正；utility 状态；feature 计算 PG1-PG4 | 存在 ghost 用 `tuple[int,int]`，不存在 ghost 用 `None` | `frame_data` | 主链路保留到 feature；two-ghost 后 `ghost3/4=None` |
| `ifscared1`, `ifscared2`, `ifscared3`, `ifscared4` | `frame_data` | float/object；不存在 ghost 为 `-1` | utility ghost 状态；dynamic 吃鬼事件；revise 规则；feature 生成 IS/IS_EXIST | `int8` 状态码；不存在为 `-1` | `frame_data` | 主链路核心字段，保留到 discrete feature |
| `pacman_dir` | `frame_data`，corrected tile 重算 | str 或 NaN | dynamic 生成 `next_pacman_dir_fill`；feature 输出 `true_dir` | categorical/enum：`left/right/up/down` 或缺失 | `corrected_tile_data` | 保留到 feature；建议生成 `next_direction` 后不再反复 shift |
| `beans` | `frame_data` 从 `Map` 解析 | list[tuple] 或 float/空标记 | utility reward；feature 统计 BN5/BN10 | `tuple[tuple[int,int], ...]`，无 beans 为空 tuple | `frame_data` | 保留到 feature，后续用 BN5/BN10 替代 |
| `energizers` | `frame_data` 从 `Map` 解析 | list[tuple] 或 float/空标记 | utility reward；dynamic 吃 energizer 事件；feature 计算 PE/EE | `tuple[tuple[int,int], ...]`，无 energizer 为空 tuple | `frame_data` | 保留到 feature，后续用 `PE/EE` 替代 |
| `Map` | `raw_subject_data` | 地图字符串 | frame 阶段解析 `beans/energizers` | 不进入分析主链路；如需留存放原始层 | `frame_data` 生成前 | frame 之后分析链路可丢弃 |
| `pacMan_1`, `pacMan_2` | `raw_subject_data` | 数值列 | 生成 `pacmanPos` | 不进入 frame 后主链路 | `frame_data` | 生成后丢弃 |
| `ghost1_1/2/3`..`ghost4_1/2/3` | `raw_subject_data` | 数值列，缺失为 inf | 过滤 four-ghost trial；生成 ghostPos/ifscared | 不进入 frame 后主链路 | `frame_data` | 生成后丢弃 |
| `pDir` | `raw_subject_data` | str | 生成初始 `pacman_dir` | 不进入分析主链路 | `frame_data` | 生成后可丢弃 |
| `JoyStick` | `raw_subject_data/frame_data` | object | 当前非视频主流程未参与计算 | 若分析不用，留在原始/视频数据 | `frame_data` | 主分析链路建议丢弃 |
| `ppX`, `ppY`, `pFrame`, `g*pX`, `g*pY`, `g*Dir`, `g*ModeR`, `g*Scared`, `g*Frame` | `raw_subject_data/frame_data` | float/object | 视频渲染使用；当前非视频主流程未参与计算 | 与分析表分离为 render table | `frame_data` | 非视频主链路建议丢弃 |
| `waterTS`, `waterStatus`, `waterDelay` | `raw_subject_data/frame_data` | float | 当前主流程未参与计算 | 若需要行为分析，单独事件表 | `raw_subject_data` | 当前主链路建议丢弃 |
| `global_Q`, `local_Q`, `evade_blinky_Q`, `evade_clyde_Q`, `evade_ghost3_Q`, `evade_ghost4_Q`, `approach_Q`, `energizer_Q`, `no_energizer_Q` | `utility_data` | 每格为长度 4 的 ndarray/list | correct utility 修墙；dynamic 生成 `*_Q_norm`；revise 评分 | `np.ndarray(shape=(4,), dtype=float64)` | `utility_data` | 保留到 revise；之后可只保留离散 strategy/weight |
| `*_Q_norm` 同 9 个 agent | `weight_data` | 每格长度 4 ndarray/list | dynamic 拟合；revise 重新评分 | `np.ndarray(shape=(4,), dtype=float64)` | `dynamic_strategy_fitting` | revise 后若 feature 不再用，可丢弃 |
| `available_dir` | `weight_data` | bool | dynamic 中修正不可走方向；输出后下游不计算使用 | bool | `dynamic_strategy_fitting` | 诊断字段，主链路不应继续透传 |
| `file` | `weight_data`，由 `DayTrial` 复制 | 字符串 trial 名 | dynamic/revise 分 trial；feature/discrete/fMRI 构造 `game`；strategy sequence 文件聚合 | 标准 `trial_id` 字段 | `dynamic_strategy_fitting`，更好提前到 `frame_data` | 主链路保留到 strategy_sequence 前 |
| `game` | `weight_data`，fMRI preprocess 也重算 | 字符串，去掉 round 编号 | dynamic shift 边界；fMRI split/keep-first 按 game 分组 | 标准 `game_id` 字段 | `frame_data` 或 `dynamic` 前 | 主链路保留到 formed 前 |
| `next_pacman_dir_fill` | `weight_data` | str 或 NaN | dynamic 切段/拟合；revise 评分 | categorical/enum 或缺失 | `dynamic_strategy_fitting`，最好在 corrected tile 后 | 保留到 revise，之后可丢弃 |
| `level_0`, `index` | `weight_data` | int | dynamic/revise 用作全局行标签和旧索引映射 | 明确改为 `row_id`，不依赖 reset_index 副产物 | `dynamic_strategy_fitting` | `level_0` 当前被 feature/discrete 透传；建议改名后保留到 fMRI 前 |
| `weight` | `weight_data` | 9 维权重 list/array | feature 输出；discrete/fMRI 透传；revise 初始策略参考 | `np.ndarray(shape=(9,), dtype=float64)` | `dynamic_strategy_fitting` | 若后续只需 `strategy`，可进入 diagnostic 或 feature 附表 |
| `contribution` | `weight_data` | 9 维 list/array | revise 初始化 `revise_weight`；feature/discrete 透传 | `np.ndarray(shape=(9,), dtype=float64)` | `dynamic_strategy_fitting` | 保留到 revise/feature；后续可丢弃 |
| `is_correct` | `weight_data` | float/bool/NaN | revise 初始化 `revise_is_correct`；诊断 | nullable bool/float | `dynamic_strategy_fitting` | 诊断字段，主链路不必长期保留 |
| `predict_dir` | `weight_data` | float/int/NaN | revise 可能更新；诊断，不进入特征计算 | nullable int8 direction id | `dynamic_strategy_fitting` | 诊断字段 |
| `trial_context` | `weight_data` | `(start,end)` tuple | revise 规则分段 | `tuple[int,int]` 或 segment_id + segment table | `dynamic_strategy_fitting` | revise 后可丢弃 |
| `eat_energizer`, `eat_ghost` | `weight_data` | bool | revise 规则 | bool | `dynamic_strategy_fitting` | revise 后可丢弃 |
| `is_stay`, `is_vague` | `weight_data` | bool | revise 生成/修正 `strategy` | bool | `dynamic_strategy_fitting` | `is_vague` revise 内还会更新；最终可只保留 strategy |
| `revise_weight` | `corrected_weight_data` | 9 维 list/array | feature/discrete 透传；strategy 生成依据之一 | `np.ndarray(shape=(9,), dtype=float64)` | `revise_human_weight` | 若 grammar 只需 `strategy`，可转 diagnostic/feature 附表 |
| `revise_is_correct` | `corrected_weight_data` | float/bool/NaN | 当前 feature 不使用，只诊断 | nullable bool/float | `revise_human_weight` | 诊断字段，主链路可不透传 |
| `strategy` | `corrected_weight_data` | int 0-10 | fMRI one-hot；strategy sequence 符号；grammar token 序列 | `int8`，固定枚举 0-10 | `revise_human_weight` | grammar 主链路核心字段 |
| `PG1`, `PG2`, `PG3`, `PG4` | `feature_data` 连续；`discrete_feature_data` 离散 | 连续距离 int；离散 0/1 | 连续用于最近 ghost 和离散化；离散进入 state/formed/sequence | 连续和离散分表命名应区分，如 `dist_pg1` / `PG1_bin` | `extract_features_human` | 离散 PG1/PG2 保留到 state；PG3/4 当前 two-ghost 可丢弃或保留兼容 |
| `PG` | `discrete_feature_data` | int | 由最近 ghost PG 合并；当前后续未用于 state graph/grammar | int8 | `extract_features_human` | 当前主 grammar 不用，可评估丢弃 |
| `PE` | `feature/discrete` | 连续距离或离散 0/1 | fMRI state/grammar 条件 | 离散 `int8` | `extract_features_human` | 主链路保留 |
| `beans_within_5`, `beans_beyond_10` | `feature_data` | int 计数，命名与阈值不一致 | 生成 `BN5/BN10` | 建议改名 `beans_within_10_count` / `beans_beyond_10_count` | `extract_features_human` | 连续表保留；离散后可丢弃 |
| `BN5`, `BN10` | `discrete_feature_data` | int 0/1 | fMRI neighbor/state/grammar 条件；state graph 默认只用 BN5 | `int8` | `extract_features_human` | BN5 主链路保留；BN10 strategy_sequence 保存但 state graph 默认不用 |
| `IS_EXIST1..IS_EXIST4` | `discrete_feature_data` | int 0/1 | fMRI split 使用 `IS_EXIST3`；formed 透传 | `bool` 或 `int8` | `extract_features_human` | split 后 four-ghost 已丢；后续可丢弃或保留审计 |
| `IS1..IS4` | `discrete_feature_data` | int 状态分类 | fMRI state/grammar 条件主要用 IS1/IS2；IS3/4 two-ghost 不有效 | `int8` | `extract_features_human` | IS1/IS2 主链路保留；IS3/4 可丢弃 |
| `IS` | `discrete_feature_data` | int | 当前后续未用于 state graph/grammar | `int8` | `extract_features_human` | 可评估丢弃 |
| `EE` | `feature/discrete` | bool | 当前 fMRI preprocess 透传，后续 grammar/state 不用 | bool | `extract_features_human` | 若不做其它分析，主链路可丢弃 |
| `global`, `local`, `evade_blinky`, `evade_clyde`, `evade_3`, `evade_4`, `approach`, `energizer`, `no_energizer`, `vague`, `stay` | `fmri_formed_data_ghost2` | 1/2 one-hot | keep-first 策略段；近邻特征；strategy sequence 生成 | bool 或 `uint8` one-hot，建议 0/1 | `human_fmri_data_preprocess` | 保留到 strategy_sequence，之后以 `seq` 替代 |
| `evade` | `fmri_formed_data_ghost2` | 1/2 合并 one-hot | 当前主流程后续不使用 | bool/uint8 | `human_fmri_data_preprocess` | 可丢弃 |
| `seq` | `strategy_sequence` | 字符串 token 序列 | state graph 对齐；generate grammar 学习；skip-gram | `list[str]` 或字符串均可；推荐 list 便于索引 | `human_fmri_data_preprocess` | grammar 主链路核心 |
| `S` | `strategy_sequence` | token 列表 | grammar 初始 token 集合；sequence 构造 | `list[str]` | `human_fmri_data_preprocess` | 主链路保留 |
| `state` | `strategy_sequence` | DataFrame：`IS1`,`IS2`,`PG1`,`PG2`,`PE`,`BN5`,`BN10` | state graph；grammar 条件状态 | 独立状态矩阵/DataFrame，列 dtype `int8` | `human_fmri_data_preprocess` | 主链路核心；state graph 默认只用前 6 列 |
| `strategy` | `strategy_sequence` | DataFrame 9 个 one-hot 列 | 当前 grammar 不使用，主要供旧结果/诊断 | 若保留，用 0/1 bool matrix | `human_fmri_data_preprocess` | 可作为诊断，不是 grammar 必需 |
| `strategyLabel` | `strategy_sequence` | Series token label | 当前 grammar 不使用 | 可由 `seq` 派生 | `human_fmri_data_preprocess` | 可丢弃 |
| `fileNames` | `strategy_sequence` | list[str] | generate grammar source；DividePerson/结果归属 | `list[str]` | `human_fmri_data_preprocess` | 主链路保留 |
| `state_names` | `state_dependency_graph` | list[str] | 结果解释/验证 | `list[str]` | `state_dependency_graph` | 保留 |
| `state_matrix` | `state_dependency_graph` | ndarray `(n_state,n_sample)` | 验证/可追踪；grammar 实际只需 adjacency | `np.ndarray[int8/int64]` | `state_dependency_graph` | 可保留用于可复现审计 |
| `adjacency_matrix` | `state_dependency_graph` | ndarray `(n_state,n_state)` | generate grammar 条件依赖输入 | `np.ndarray[bool/int8]` | `state_dependency_graph` | 主链路核心 |
| grammar `source.input_file_name`, `participant_file_names`, `participant_ids` | `generate_grammar` | dict/list[str] | DividePerson 归属和输出追踪 | 结构化 dict | `generate_grammar` | 保留 |
| grammar `parameters` | `generate_grammar` | dict | 可复现配置，不参与后续计算 | 结构化 dict | `generate_grammar` | 保留为元数据 |
| grammar item：`token`, `base_tokens`, `probability`, `frequency`, `time_probability`, `components` | `generate_grammar` | list[dict] | DividePerson 使用 `token/frequency/components`；分析使用概率 | 结构化 list 或 DataFrame | `generate_grammar` | 主链路保留 |
| grammar `parsed.original_sequence`, `parsed.sequence`, `parsed.state_features` | `generate_grammar` | list/DataFrame | skip-gram 已完成后主要用于解释和后续分析 | 结构化 dict，state_features dtype `int8` | `generate_grammar` | 保留为可解释输出 |
| grammar `skip_gram.target`, `found`, `count` | `generate_grammar` | dict，count float | DividePerson 把 `count` 作为长度三类特征 | 结构化 dict，`found: bool`, `count: float` | `generate_grammar` | 主链路保留到 DividePerson |

## 当前明确不应进入非视频主链路的字段

这些字段当前在中间 DataFrame 中被透传，但本次审计没有发现其参与非视频主流程计算：

| 字段 | 当前来源 | 建议 |
|---|---|---|
| `JoyStick` | frame/tile/utility/weight | 若只做分析 pipeline，放到原始层或单独行为输入，不随 utility/weight 透传 |
| `ppX`, `ppY`, `pFrame`, `g*pX`, `g*pY`, `g*Dir`, `g*ModeR`, `g*Scared`, `g*Frame` | frame/tile/utility/weight | 视频/render 专用，应拆到 render table |
| `Map` | frame/tile/utility/weight | frame 阶段生成 `beans/energizers` 后即可停止透传 |
| `waterTS`, `waterStatus`, `waterDelay` | frame/tile/utility/weight | 当前主链路未使用；如需要，应建事件表 |
| `available_dir`, `predict_dir`, `is_correct`, `revise_is_correct` | weight/corrected_weight | 诊断字段，建议与主数据分离 |
| `strategyLabel`, formed 中的 `evade` | fMRI preprocess | 可由其它字段派生或当前后续不用，建议不进入主输出 |

## 优先治理建议

1. **位置字段最早标准化**：在 `frame_data` 统一 `pacmanPos/ghost*Pos/beans/energizers`，
   不再让 utility、dynamic、feature 各自重复解析字符串、空 list 和 float 缺失。
2. **事件和 ID 字段提前生成**：在 corrected tile 或 frame 阶段生成 `trial_id`、`game_id`、
   `frame_id`、`next_direction`，替代后续脚本重复用字符串 split、shift 和 reset_index。
3. **主数据与诊断数据分离**：`predict_dir/is_correct/revise_is_correct/available_dir` 这类字段不应
   一路进入 feature/discrete/fMRI 数据。
4. **Q 和权重格式固定**：9 个 agent 的 Q/weight/contribution/revise_weight 统一为定长
   `np.ndarray` 或拆成矩阵结构，禁止 list、ndarray、object 混用。
5. **离散状态表瘦身**：grammar 实际使用 `IS1/IS2/PG1/PG2/PE/BN5`，`BN10` 当前保存在
   strategy sequence 但 state graph 默认不用；`PG/IS/IS3/IS4/PG3/PG4/EE` 应按后续分析需要决定是否保留。
