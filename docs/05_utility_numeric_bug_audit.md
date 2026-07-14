# 05 Utility 数值与状态类型 Bug 审计

## 1. 审计目标

本文记录当前 05 hierarchical utility 计算中已经确认的两类重大问题：

1. ghost 状态在进入 utility 估计器前被错误转换为浮点数，随后又被风险函数当作缺失状态处理；
2. Evade 和 NoEnergizer 的归一化会原地修改 raw Q，导致保存字段的实际语义与字段名不一致。

这两个问题会继续影响 06 的 context 策略拟合、文件级参数估计、07 的策略修正及最终视频标签。2026-07-11 完成业务代码修复，2026-07-14 已从 05 开始重新生成当前正式数据及其下游结果。

## 2. 审计范围

本次静态检查覆盖：

- `src/LoPS/calculate_utility/processing.py`
- `src/LoPS/hierarchical_utility/model.py`
- `src/LoPS/hierarchical_utility/strategies.py`

全量数值差分以当前 05 目录中的以下 comp 文件为代表样例：

```text
data/05_utility_data/comp/10001-10022-2025-07-15-JJJ-1.pkl
```

当前 05 目录还包含 coop 文件
`data/05_utility_data/coop/10001-10005-2025-07-10-HHH-1.pkl`；“唯一文件”不再是
当前数据目录的事实。

对应的 04 输入为：

```text
data/04_corrected_tile_data/comp/10001-10022-2025-07-15-JJJ-1.pkl
```

检查规模：

| 玩家 | 可计算 player-tile 数量 |
|---|---:|
| P1 | 6407 |
| P2 | 6421 |
| 合计 | 12828 |

## 3. Bug 一：ghost 状态被错误解释

### 3.1 正确的数据语义

04 输入中的 `ifscared1` 和 `ifscared2` 已经是无缺失的 `int8` 字段。当前数据使用的状态值包括：

- `1/2`：需要回避的正常或危险 ghost；
- `3`：dead ghost；
- `4/5`：scared ghost 的相关状态。

因此，05 不需要为了兼容旧数据而把状态转换为浮点数。

### 3.2 错误的数据转换

`build_utility_estimation_input()` 在调用估计器前执行：

```python
result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
```

这会把状态转换为：

```text
1 -> 1.0
2 -> 2.0
3 -> 3.0
4 -> 4.0
5 -> 5.0
```

随后，风险函数使用 `_status_value()` 解析状态：

```python
return 0 if _is_float_marker(value) else value
```

而 `_is_float_marker()` 对任意 Python 或 NumPy 浮点数都返回 `True`。因此所有有效状态都会再次被转换成 `0`：

```text
1.0/2.0/3.0/4.0/5.0 -> 0
```

这里并没有判断浮点数是否为 `NaN`。有限浮点状态和真正的缺失值被错误地合并成了同一种情况。

### 3.3 同一状态在不同函数中的解释不一致

`_apply_approach_reward()` 直接比较原始状态：

```python
ghost_status[ghost_index] != 3
ghost_status[ghost_index] > 3
```

因此 `3.0` 和 `4.0` 在 Approach 奖励函数中仍能被正确识别。

但是随后的 `_two_ghost_risk()` 会通过 `_status_value()` 把同一个 `3.0/4.0` 转成 `0`。这意味着同一次路径扩展中：

- 奖励计算知道 ghost 已死亡或处于 scared；
- 风险计算却把它当成危险状态；
- 路径可能在错误的位置提前终止。

这是 0-based 第 35 帧 Approach 异常的直接原因。

### 3.4 0-based 第 35 帧复现样例

视频图片：

```text
data/pacman_video/tile_frame_images/comp/
10001-10022-2025-07-15-JJJ-1/
02-01-10001-10022-2025-07-15-JJJ/000035.png
```

`000035.png` 对应 0-based 视频第 35 帧。该帧状态为：

| 对象 | 坐标 | 状态 |
|---|---|---:|
| P2 | `(15, 15)` | alive |
| ghost1 | `(14, 15)` | `3`，dead |
| ghost2 | `(13, 15)` | `4`，scared |

正确路径应允许 P2 从 `(15,15)` 向左经过 dead ghost 所在位置，并继续接近 `(13,15)` 的 scared ghost。

当前错误流程中：

1. P2 向左到达 `(14,15)`；
2. Approach 奖励函数识别 `3.0 == 3`，不给 dead ghost 奖励；
3. 风险函数把 `(3.0,4.0)` 转换成 `(0,0)`；
4. `(14,15)` 的 dead ghost 被错误识别为危险 ghost；
5. 左方向路径立即终止，无法到达 `(13,15)`；
6. 最终 Approach Q 被错误计算为：

$$
Q_{\mathrm{approach}}
=
[0,\ 0,\ -\infty,\ 0].
$$

保持地图、资源、位置、搜索深度等条件不变，只保留整数 ghost 状态后，重新计算得到：

$$
Q_{\mathrm{approach}}^{\mathrm{raw}}
=
[1.2203,\ 0,\ -\infty,\ 1.1667].
$$

按照 06 的合法方向逐行 Min-Max 规则归一化后约为：

$$
\widetilde Q_{\mathrm{approach}}
=
[1,\ 0,\ -\infty,\ 0.956].
$$

修正后 Approach 会把左方向作为最大 utility 方向。

### 3.5 受影响的策略函数

该状态类型错误不是 Approach 的局部问题，而会进入所有 ghost 相关路径逻辑：

- Local：通过 `_two_ghost_risk()` 决定路径是否终止；
- EvadeBlinky/EvadeClyde：通过 `_single_ghost_evade_risk()` 判断 ghost 是否危险；
- Approach：奖励后通过 `_two_ghost_risk()` 决定是否终止；
- Energizer：通过 `_two_ghost_risk()` 决定路径是否终止；
- NoEnergizer：通过 `_two_ghost_termination_only()` 决定路径是否终止。

Global 和 Cluster Global 不读取 ghost 状态，不受此类型错误直接影响。

### 3.6 全量数值影响

下表比较了两种计算结果：

- 当前路径：把 ghost 状态转换为 float；
- 修正诊断路径：保持原始整数状态；

除状态类型外，其余输入、地图、搜索参数和计算逻辑完全相同。

| 策略 | Q 数值发生变化 | 最大预测方向集合发生变化 |
|---|---:|---:|
| Local | 1567（12.22%） | 444（3.46%） |
| Evade Blinky | 1036（8.08%） | 1036（8.08%） |
| Evade Clyde | 1078（8.40%） | 1078（8.40%） |
| Approach | 999（7.79%） | 79（0.62%） |
| Energizer | 212（1.65%） | 147（1.15%） |
| NoEnergizer | 140（1.09%） | 108（0.84%） |

其中 Evade 受到的影响最直接：

- 当前 P1 的 `evade_blinky_Q` 和 `evade_clyde_Q` 在全部 6407 个可计算 tile 上均为全零；
- 当前 P2 的这两个 Evade Q 在全部 6421 个可计算 tile 上也均为全零；
- 保持整数状态后，Blinky Evade 有 1036 行从无信息变成有信息；
- 保持整数状态后，Clyde Evade 有 1078 行从无信息变成有信息。

原因是 `_single_ghost_evade_risk()` 只接受状态 `1/2`，但当前所有 `1.0/2.0` 都先被转换成了 `0`。因此当前 Evade 实际没有提供任何有效的方向信息。

## 4. Bug 二：归一化原地修改 raw Q

### 4.1 错误位置

`make_evade_q_non_negative()` 对传入数组直接执行：

```python
q_values[available_indices] = q_values[available_indices] - offset
```

传入的 `q_values` 与 DataFrame 中保存的 raw Q 数组共享对象，因此为生成 `*_Q_norm` 所做的平移同时改写了 `*_Q`。

### 4.2 当前数据中的影响

将保存的 `no_energizer_Q` 与归一化前估计器直接输出比较：

| 玩家 | 被改写的 raw NoEnergizer Q 行数 |
|---|---:|
| P1 | 6232 |
| P2 | 6224 |
| 合计 | 12456/12828（97.10%） |

当前 Evade Q 因 Bug 一而全部为零，所以这次尚未表现出明显的 raw Q 改写。修复状态类型后，Evade 会产生负 utility；如果不同时修复原地修改问题，Evade raw Q 也会被归一化过程改写。

### 4.3 对当前下游的影响

NoEnergizer 当前采用的是对所有合法方向统一减去相同 offset。06 随后重新执行逐行 Min-Max，因此平移本身通常不会改变 06 的归一化结果。

但是该行为仍是数据结构错误：

- `*_Q` 不再表示估计器的 raw utility；
- raw Q 与 Q_norm 无法独立验证；
- 任何使用绝对 utility、跨行尺度或重新归一化的方法都会读取到被改写的数据；
- 修复 Evade 后，相同问题会扩展到 Evade raw Q。

因此该问题必须与状态类型 Bug 一起修复，不能因为当前 06 对平移不敏感而保留。

## 5. 对下游结果的影响

受影响的数据阶段包括：

1. 05 保存的 ghost 相关 Q；
2. 06 基于 raw Q 统一归一化后的 likelihood、posterior 和文件级 beta；
3. 07 使用单策略预测准确率进行的策略修正；
4. 使用 07 数据生成的策略视频。

即使某个 context 中 Global Q 本身正确，其他候选策略 Q 的错误仍会改变：

- context 内各策略 likelihood；
- posterior 的相对大小；
- `vague` 判定；
- 文件级共享或独立 beta 的估计与 BIC；
- 07 的修正候选和最终策略名称。

因此不能只修补 0-based 第 35 帧或 07 标签，必须从 05 开始重新计算。

## 6. 已排除的问题

本次对当前单文件同时检查了以下数据质量问题，均未发现异常：

- `ifscared1/ifscared2` 在 04 和 05 中均为无缺失整数；
- `p1_available_dir/p2_available_dir` 均为无缺失布尔值；
- 已保存 Q 的数组长度均为 4；
- 已保存 Q 中没有 NaN；
- 已保存 Q 中没有正无穷；
- Cluster Global 的候选构造不依赖 ghost 状态。

地图解析中的 `_is_float_marker()` 目前用于区分 tuple 坐标与 `NaN`，当前地图常量没有触发同类错误。不过“所有 float 都代表缺失”的接口仍然较脆弱，修复时应尽量将缺失判断收窄为显式的 `None/NaN`。

## 7. 建议修复方案

### 7.1 统一 ghost 状态类型

在进入 utility 核心计算前：

1. 使用 `pd.to_numeric(..., errors="raise")` 验证状态可解析；
2. 保持整数状态，不再执行 `astype(float)`；
3. 显式验证状态属于当前允许集合；
4. 在 `FrameState` 中将 ghost 状态规范成 Python `int` 或明确的缺失标记。

### 7.2 收窄缺失值判断

`_status_value()` 不应根据“是不是 float”判断缺失，而应只把真正的 `None/NaN` 视为缺失。有限浮点数如果仍被接口允许，应先验证其为整数值，再转换成 `int`。

### 7.3 禁止归一化修改 raw Q

`make_evade_q_non_negative()` 应先复制输入：

```python
normalized_input = np.asarray(q_values, dtype=float).copy()
```

所有平移和归一化只修改副本。保存后的 `*_Q` 必须与估计器原始输出在合法方向修正后完全一致。

### 7.4 增加回归测试

至少覆盖：

- 状态 `1/2` 的 ghost 能产生 Evade 风险；
- 状态 `3` 的 dead ghost 不产生 Evade 风险，也不阻断通往后方目标的路径；
- 状态 `4/5` 的 scared ghost 不产生 Evade 风险；
- Approach 能追踪 dead ghost 后方的 scared ghost；
- Local/Energizer 不会因 scared/dead ghost 错误提前终止；
- NoEnergizer 的碰撞终止读取正确状态；
- Python int、NumPy integer 以及允许的有限整值 float 具有一致语义，或者有限 float 被明确拒绝；
- 生成 Q_norm 前后的 raw Q 完全不变。

## 8. 修复后的执行与验收顺序

修复后应按以下顺序执行：

1. 运行新增状态语义与 raw Q 不变性单元测试；
2. 重新运行单文件 05；
3. 对同一批 12828 个 player-tile 重新执行差分审计；
4. 验证两套 Evade 不再全零；
5. 验证 0-based 第 35 帧 P2 Approach 左方向不再为 0；
6. 验证保存的 raw Q 与归一化前估计器输出一致；
7. 重新运行 06，检查 beta、BIC、posterior 和 context 写回；
8. 重新运行 07；
9. 重新生成视频并复查关键 context。

当前正式 06 已使用 context posterior 与文件级 beta，不再生成 GA 权重。任何由旧 05
数据派生的 posterior、07 修正结果和视频都必须从修复后的 05 重新运行，不能继续沿用。

## 9. 修复实现与验证结果

### 9.1 已完成的代码修改

2026-07-11 完成以下修改：

1. `build_utility_estimation_input()` 不再把 `ifscared1/ifscared2` 转成 float，而是验证其有限性和整数性后保存为整数；
2. `_status_value()` 只把真正的 `None/NaN` 映射为 0，有限整值 float 和整数保持相同状态语义；
3. 非整数、无穷和非数值状态会显式报错，避免静默污染搜索树；
4. `make_evade_q_non_negative()` 在显式数组副本上平移和归一化，不再修改 raw Q；
5. 新增 `tests/test_calculate_utility_status_and_normalization.py`，覆盖状态语义、非法输入、数组不变性及真实帧回归。

没有修改搜索深度、奖励参数、路径展开、地图、Global/Cluster Global、方向顺序或归一化公式。

### 9.2 自动测试

使用 LDS 环境执行：

```bash
PYTHONPATH=src /home/zzh/anaconda3/envs/LDS/bin/python -m pytest \
  tests/test_calculate_utility_status_and_normalization.py -q
```

该命令在修复当时得到 `7 passed`。截至 2026-07-14，当前 05 utility 专项测试集合已
扩展，重新执行结果为 `19 passed`；历史数字只用于说明最初修复的验收状态。

### 9.3 单文件正式入口验证

修复后的 05 通过 `script/05_calculate_utility.py` 运行到独立验证目录，没有覆盖正式输出。验证结果：

| 项目 | 结果 |
|---|---:|
| 输入/输出行数 | 6439/6439 |
| P1 计算/跳过 | 6407/32 |
| P2 计算/跳过 | 6421/18 |
| 当前输出列数 | 67 |

当前 67 列包含后续新增的双玩家 Energizer 与 Approach 候选字段。最初修复 ghost 状态
与 raw Q 污染问题时，输出行数、索引、当时已有列的顺序、DataFrame attrs 和所有非 Q
输入字段保持一致；新增候选字段属于后续正式方法演进，不应再用旧 55 列 schema 验收。

### 9.4 Bug 修复验证

修复后的正式入口输出满足：

- P1/P2 六种路径 raw Q 与整数状态估计器逐行重算结果完全一致，所有策略 mismatch 数均为 0；
- P1 Blinky/Clyde Evade 的有信息行数从 `0/0` 恢复为 `472/544`；
- P2 Blinky/Clyde Evade 的有信息行数从 `0/0` 恢复为 `564/534`；
- 0-based 第 35 帧 P2 Approach raw Q 恢复为：

$$
[1.2203389831,\ 0,\ -\infty,\ 1.1666666667],
$$

- 0-based 第 35 帧 P2 Approach 归一化 Q 为：

$$
[1,\ 0,\ -\infty,\ 0.9560185185],
$$

- 保存的 raw Q 不再被 Q_norm 生成过程修改；
- 所有保存 Q 均保持长度 4，且没有 NaN 或正无穷。

### 9.5 无关功能一致性

以下内容在修复前后逐行完全一致：

- P1/P2 `global_Q`；
- P1/P2 `global_Q_norm`；
- P1/P2 `global_utility_k`；
- P1/P2 `global_utility_k_norm`；
- P1/P2 `global_utility_k_meta`；
- 所有来自 04 的输入字段。

这说明修改只影响原先被错误状态解释污染的 ghost 相关路径，以及原先被数组别名改写的 raw Q，没有改变 Global、Cluster Global 或 joint-state 数据结构。

## 10. 当前结论

两个 Bug 已在业务代码中修复，并通过单元测试、真实关键帧和完整单文件 05 正式入口验证。修复后的结果满足预期，且未发现新的数值异常或无关字段变化。

当前 `data/05_utility_data`、`data/06_strategy_posterior_data` 和
`data/07_revised_strategy_data` 均来自修复后的正式流程。
