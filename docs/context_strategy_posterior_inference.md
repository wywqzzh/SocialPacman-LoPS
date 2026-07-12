# 基于 Context 的潜在策略后验推断

## 1. 目标

本方法用于从 Pacman 行为中推断每个玩家在每个 context 中采用的策略，进而生成 P1、P2 的策略序列，用于研究合作或竞争条件下两名玩家如何根据对方策略调整自身策略。

核心假设是：

> 一个 context 内的有效动作主要由同一个潜在策略生成。

因此，方法不再为每个 context 拟合七个可任意缩放的 utility 权重，而是计算整个 context 属于每个策略的后验概率：

$$
P(z_c=k\mid\mathbf a_c,\mathbf s_c).
$$

这里的后验表示“整个 context 由策略 $k$ 生成的可能性”，不表示该策略在 context 内所占的时间比例。

## 2. 已有数据

前面的处理阶段已经提供：

- 每个玩家独立划分的 context；
- 每个 tile 的真实动作和合法方向；
- 每个玩家的位置、存活状态；
- 七个策略在四个方向上的 Q 值。

七个候选策略为：

1. `global`
2. `local`
3. `evade_blinky`
4. `evade_clyde`
5. `approach`
6. `energizer`
7. `no_energizer`

策略数量记为：

$$
K=7.
$$

相关字段包括：

- `<player>_action_dir`
- `<player>_available_dir`
- `<player>_alive`
- `<player>_trial_context`
- `<player>_<strategy>_Q`
- `<player>_global_utility_k`
- `<player>_global_utility_k_norm`
- `<player>_global_utility_k_meta`

其中 `<player>` 为 `p1` 或 `p2`。

死亡、动作缺失、`available_dir=False` 或真实动作不合法的行不进入 likelihood。完全没有有效动作的长停留段直接标记为 `stay`。

### 2.1 普通豆边界与强事件窗口

06b 和 06c 使用同一套玩家私有 context 划分。该流程以 0-based **事件点**为基本单位：先检测事件发生行，再按规则删除事件点，最后才对排序后的相邻事件点生成 context。事件检测阶段不提前生成 context，也不把导致事件的前一动作行当作事件行。

资源事件列标在事件发生后的到达行。例如第 77 行动作使玩家在第 78 行吃到 energizer，则 `eat_energizer=True` 位于 78，context 强边界也只能是 78。第 77 行可在策略归因阶段被解释为导致事件的动作，但不能因此成为事件边界。

连续吃普通豆需要借助动作序列识别同一次连续采食过程，短 stay 可以按既定规则连接前后吃豆记录；识别完成后只输出第一个和最后一个 `eat_bean=True` 的实际事件行，不把内部动作范围写入边界集合。

trial 起止、Pacman 生死变化、吃 energizer、吃 ghost、长 stay 起止属于强事件点，必须始终保留。连续吃普通豆过程的首尾事件属于可抑制的弱硬边界。

设普通豆边界集合为 $B$，强事件边界集合为 $E$。默认窗口为 3 个 tile，仅保留满足下式的普通豆边界：

$$
B_{\mathrm{keep}}
=
\left\{
b\in B:
\min_{e\in E}|b-e|>3
\right\}.
$$

因此，强事件前后 3 个 tile 内的普通豆起止边界会被取消，但强事件自身永远不会被删除。距离使用同一 trial 内的时间 tile 下标，不使用地图空间距离。普通转向不切段，只有掉头作为可合并的软边界；玩家进入或离开普通豆 10 步范围不再属于 context 划分条件。P1/P2 分别使用自己的事件集合。

长 stay 还需要接受当前玩家私有吃 ghost 事件的二次过滤。设长 stay 区间为 $S=[s,e)$，当前玩家吃 ghost 的事件行为 $g$，事件行到 stay 最近实际行的距离为：

$$
d(g,S)
=
\begin{cases}
0, & s\le g<e,\\
s-g, & g<s,\\
g-(e-1), & g\ge e.
\end{cases}
$$

如果存在当前玩家自己的吃 ghost 事件满足：

$$
d(g,S)\le 5,
$$

则取消该 stay 的开始和结束硬边界。原始无动作行仍然保留，吃 ghost 边界也始终保留；取消的只是 stay 对 context 的额外切段作用。队友吃 ghost 不影响当前玩家的 stay。

### 2.2 Context 内选择 Best Global

05 为每一行保存多个资源团对应的 Global utility。06c 不把这些候选视为多个顶层策略，也不对候选 likelihood 做边际化。context 划分完成后，先沿用 06b 的规则计算每个 Global cluster 单独预测该 context 真实动作的概率准确率：

$$
k_c^\ast
=
\underset{k}{\arg\max}\;
\operatorname{Accuracy}_{c,k}.
$$

真实方向属于 $m$ 个正向并列最大方向时，该行贡献 $1/m$；候选全零或没有正向推进时贡献 $0$。准确率相同时，依次使用集合准确率、cluster size、起点距离和 cluster ID 破平。

选中的候选在整个 context 中作为唯一的 Global raw Q，随后与另外六种策略使用同一归一化和 likelihood 规则。输出同时保存 best cluster ID、准确率和资源团 meta，便于解释 Pacman 具体朝向哪一团资源。

## 3. 后验模型的 Q 值统一归一化

不同策略当前保存的 Q 值并不具有完全相同的数值尺度。例如，cluster-global 的有限值可能位于 $[-1,1]$，其他策略的有限值通常位于 $[0,1]$。如果直接使用共享的 temperature，数值跨度更大的策略会天然产生更集中的动作概率。因此，进入后验计算前必须使用同一规则重新归一化 Q。

归一化的最小单位是“一个玩家、一个 tile、一个策略”。不得跨 tile、跨策略、跨玩家或跨文件计算最大值和最小值。

设 tile $t$ 的合法方向集合为 $\mathcal A_t$。只使用合法方向上的有限 Q 值计算：

$$
Q^{\min}_{t,k}
=
\min_{d\in\mathcal A_t}Q_{t,k,d},
\qquad
Q^{\max}_{t,k}
=
\max_{d\in\mathcal A_t}Q_{t,k,d}.
$$

当 $Q^{\max}_{t,k}>Q^{\min}_{t,k}$ 时，对合法方向执行 Min-Max 归一化：

$$
\widetilde Q_{t,k,d}
=
\frac{
Q_{t,k,d}-Q^{\min}_{t,k}
}{
Q^{\max}_{t,k}-Q^{\min}_{t,k}
},
\qquad d\in\mathcal A_t.
$$

归一化后，所有合法方向的有限值均位于 $[0,1]$，同时保留原始 Q 在当前 tile 内的方向排序。

如果所有合法方向 Q 相同，即 $Q^{\max}_{t,k}=Q^{\min}_{t,k}$，则设置：

$$
\widetilde Q_{t,k,d}=0,
\qquad d\in\mathcal A_t.
$$

这表示该策略在当前 tile 对合法方向没有区分信息。后续 softmax 将自然得到合法方向上的均匀概率，不允许使用并列 `argmax` 将其计为有效预测。

非法方向在原始 Q 中通常表示为 $-\infty$。它们必须满足以下约束：

- 不参与最大值和最小值计算；
- 不执行 Min-Max 变换；
- 继续作为非法方向 mask；
- 不进入后续 softmax 的分母。

该统一归一化只服务于新的后验推断。原始 utility 和已有的 `Q_norm` 字段保持不变；后验拟合阶段必须从 raw Q 派生 $\widetilde Q$，其中 Global 使用当前 context 已选中的候选 raw Q。这样可以避免旧归一化规则造成信息丢失，同时保留新旧方法的对照能力。

## 4. 需要拟合的参数

每个数据文件对应一对固定被试。不同文件相互独立，不跨文件共同拟合 temperature。对于文件 $f$，需要比较两种参数结构：

- P1、P2 共享一个文件级参数 $\beta_f$；
- P1、P2 分别使用 $\beta_{f,p1}$ 和 $\beta_{f,p2}$。

所有 temperature 均满足：

$$
\beta_f>0
$$

或者：

$$
\beta_{f,p1}>0,
\qquad
\beta_{f,p2}>0.
$$

$\beta$ 控制 Q 差异转换成动作概率时的确定程度：

- $\beta$ 小时，合法方向概率较接近；
- $\beta$ 大时，模型更接近硬 `argmax`。

为自动满足正值约束，实际优化：

$$
\eta=\log\beta,
\qquad
\beta=\exp(\eta).
$$

策略先验第一版固定为均匀分布：

$$
\pi_k=P(z_c=k)=\frac{1}{K}.
$$

因此，每个 context 不再单独拟合七个权重；context 的七个后验由当前文件选定的 temperature 和该段行为直接计算。

## 5. 从 Q 计算动作概率

策略 $k$ 在 tile $t$ 选择合法方向 $d$ 的概率为：

$$
P_k(d\mid s_t,\beta)
=
\frac{
\exp\left(\beta\widetilde Q_{t,k,d}\right)
}{
\displaystyle\sum_{d'\in\mathcal A_t}
\exp\left(\beta\widetilde Q_{t,k,d'}\right)
},
\qquad d\in\mathcal A_t.
$$

非法方向概率为：

$$
P_k(d\mid s_t,\beta)=0,
\qquad d\notin\mathcal A_t.
$$

如果某个策略在所有合法方向上的 Q 都相同，则自然得到均匀概率：

$$
P_k(d\mid s_t,\beta)
=
\frac{1}{|\mathcal A_t|}.
$$

因此，全零 Q 会被视为无信息，而不会因为并列 `argmax` 得到伪预测准确率。

## 6. 计算 Context Likelihood

设 context $c$ 的有效 tile 集合为 $\mathcal T_c$，真实动作为 $a_t$。假设整个 context 使用策略 $k$，则：

$$
P(\mathbf a_c\mid\mathbf s_c,z_c=k,\beta)
=
\prod_{t\in\mathcal T_c}
P_k(a_t\mid s_t,\beta).
$$

实际计算使用对数似然：

$$
\ell_{c,k}(\beta)
=
\sum_{t\in\mathcal T_c}
\log P_k(a_t\mid s_t,\beta).
$$

不同长度 context 的描述性比较可以使用平均对数似然：

$$
\bar\ell_{c,k}
=
\frac{\ell_{c,k}}{|\mathcal T_c|}.
$$

正式后验和训练 loss 仍使用总对数似然 $\ell_{c,k}$，因为较长 context 应提供更多策略判断证据。

## 7. 计算策略后验

策略 $k$ 的未归一化对数后验分数为：

$$
r_{c,k}
=
\log\pi_k+\ell_{c,k}.
$$

使用 `logsumexp` 计算归一化常数：

$$
\log Z_c
=
\operatorname{logsumexp}_k(r_{c,k}).
$$

策略后验为：

$$
\gamma_{c,k}
=
P(z_c=k\mid\mathbf a_c,\mathbf s_c)
=
\exp\left(r_{c,k}-\log Z_c\right).
$$

满足：

$$
\sum_{k=1}^{K}\gamma_{c,k}=1.
$$

最终候选策略为：

$$
k_c^\ast
=
\underset{k}{\arg\max}\;\gamma_{c,k}.
$$

## 8. Beta 的 Loss

因为 context 的真实策略没有被直接观测，需要对七个策略求边际概率。单个 context 的负对数似然为：

$$
\mathcal L_c(\beta)
=
-\operatorname{logsumexp}_k
\left[
\log\pi_k+\ell_{c,k}(\beta)
\right].
$$

训练集总 loss 为：

$$
\mathcal L_{\mathrm{train}}(\beta)
=
\sum_{c\in\mathcal C_{\mathrm{train}}}
\mathcal L_c(\beta).
$$

最终在训练集上拟合：

$$
\beta^\ast
=
\underset{\beta>0}{\arg\min}\;
\mathcal L_{\mathrm{train}}(\beta).
$$

因为只有一个参数，可以对 $\eta=\log\beta$ 使用一维有界优化或网格搜索，不需要遗传算法。初始搜索范围可以设为：

$$
\beta\in[0.05,20].
$$

## 9. 文件内交叉验证与 BIC 模型选择

每个文件独立执行 grouped cross-validation，不使用其它文件的数据。不能随机拆分 tile；同一 `DayTrial` 中 P1、P2 的 contexts 必须进入同一 fold，同一 context 的所有 tile 也必须保持在同一 fold。

### 9.1 共享 Beta 模型

共享模型对文件 $f$ 的 P1、P2 使用同一个参数：

$$
\mathcal L_f^{\mathrm{shared}}(\beta_f)
=
\sum_{p\in\{p1,p2\}}
\sum_{c\in\mathcal C_{f,p}}
\mathcal L_{f,p,c}(\beta_f).
$$

需要拟合的参数数量为：

$$
m_{\mathrm{shared}}=1.
$$

### 9.2 玩家独立 Beta 模型

独立模型分别拟合两个玩家：

$$
\mathcal L_f^{\mathrm{separate}}
\left(\beta_{f,p1},\beta_{f,p2}\right)
=
\sum_{c\in\mathcal C_{f,p1}}
\mathcal L_{f,p1,c}(\beta_{f,p1})
+
\sum_{c\in\mathcal C_{f,p2}}
\mathcal L_{f,p2,c}(\beta_{f,p2}).
$$

需要拟合的参数数量为：

$$
m_{\mathrm{separate}}=2.
$$

### 9.3 使用 BIC 选择参数结构

设文件 $f$ 中可参与拟合的 context 总数为 $C_f$。由于 likelihood 按 context 分解，BIC 的样本数使用 $C_f$，而不是 tile 数。

$$
\operatorname{BIC}(M)
=
2\mathcal L_f^{\min}(M)
+
m_M\log C_f.
$$

分别计算：

$$
\operatorname{BIC}_{\mathrm{shared}}
$$

和：

$$
\operatorname{BIC}_{\mathrm{separate}}.
$$

选择 BIC 更小的模型：

$$
M_f^\ast
=
\underset{M\in\{\mathrm{shared},\mathrm{separate}\}}{\arg\min}
\operatorname{BIC}(M).
$$

### 9.4 文件内分组交叉验证

以 5 折为例，每一折都只使用当前文件中的完整 `DayTrial` 分组：

1. 在四个训练 folds 上分别拟合共享模型和玩家独立模型；
2. 记录训练数据上的 BIC 选择是否稳定；
3. 在验证 fold 上比较 held-out NLL；
4. 检查各 fold 的 temperature 是否稳定；
5. 完成交叉验证后，在当前文件全部 contexts 上重新拟合两个模型；
6. 使用全文件 BIC 确定最终采用一个还是两个 temperature。

如果共享模型胜出，保存 $\beta_f$；如果玩家独立模型胜出，保存 $\beta_{f,p1}$ 和 $\beta_{f,p2}$。该选择只对当前文件生效。

## 10. Vague 与 Stay

均匀随机动作模型的 context 对数似然为：

$$
\ell_{c,\mathrm{null}}
=
-\sum_{t\in\mathcal T_c}
\log|\mathcal A_t|.
$$

最佳策略相对随机模型的平均改善为：

$$
G_c
=
\frac{
\max_k\ell_{c,k}-\ell_{c,\mathrm{null}}
}{
|\mathcal T_c|
}.
$$

06c 第一版在以下条件成立时标记为 `vague`：

$$
\max_k\gamma_{c,k}<\tau_{\mathrm{posterior}},
$$

$\tau_{\mathrm{posterior}}$ 固定为 $0.70$。$G_c$ 作为诊断字段保存，但在第一版中不参与 `vague` 判定；后续若要增加 gain 阈值，应通过验证集、打乱动作或模拟 null 数据确定，不能直接使用经验常数。

完全没有有效动作的长停留 context 保持为 `stay`，不进行后验推断。

## 11. 双人处理与输出

P1 和 P2 分别使用自己的 context、真实动作、合法方向和 Q 值计算后验。两名玩家使用共享还是独立 temperature，由当前文件的 BIC 结果决定。

建议保存：

- `<player>_strategy_log_likelihood`
- `<player>_strategy_posterior`
- `<player>_strategy_posterior_max`
- `<player>_strategy_candidate`
- `<player>_null_log_likelihood`
- `<player>_log_likelihood_gain`
- `<player>_strategy`
- `<player>_strategy_name`
- `<player>_valid_action_count`
- `<player>_is_vague`
- `<player>_is_stay`
- `<player>_selected_global_Q`
- `<player>_best_global_cluster_id`
- `<player>_best_global_cluster_prob_accuracy`
- `<player>_best_global_cluster_set_accuracy`
- `<player>_best_global_cluster_meta`

P1、P2 的结果继续写回同一份 joint-state 表，保持两个玩家的时间对齐。

文件级模型元数据还需要保存：

- BIC 选择的模型类型；
- 共享 $\beta_f$，或者独立的 $\beta_{f,p1}$、$\beta_{f,p2}$；
- $\operatorname{BIC}_{\mathrm{shared}}$ 和 $\operatorname{BIC}_{\mathrm{separate}}$；
- grouped cross-validation 的 fold 划分和验证 NLL。

## 12. 完整流程

1. 读取 raw Q、动作、合法方向和 Global cluster 候选；
2. 使用 06b 的玩家私有事件规则划分 context；
3. 在每个 player-context 内按单独预测准确率选择 best Global；
4. 从选中的 Global raw Q 和另外六种 raw Q 统一计算 $\widetilde Q$；
5. 每个文件内部按完整 `DayTrial` 建立 grouped folds；
6. 分别拟合“P1/P2 共享一个 $\beta$”和“P1/P2 各一个 $\beta$”两个模型；
7. 使用交叉验证检查 temperature、BIC 选择和 held-out NLL 是否稳定；
8. 在当前文件全部 contexts 上重新拟合两个模型，并用 BIC 确定最终参数结构；
9. 使用选定的 temperature 计算每个 player-context 的七个策略 likelihood 和 posterior；
10. 根据最大 posterior 判断具体策略或 `vague`，并将 P1/P2 结果写回联合数据。

## 13. Global 选择的解释约束

06c 的 best Global cluster 根据当前 context 的真实动作选择，随后同一批动作还会用于计算选中 Global 的 likelihood。因此，Global 相比只有单一 utility 的其他策略拥有额外候选机会，其 likelihood 和 posterior 可能存在乐观偏差。

本版本接受该约束，因为研究目标明确要求先找出当前 context 中解释动作最好的资源团，再将该资源团作为唯一 Global 参与七策略比较。输出 metadata 必须保存：

```text
global_selection_uses_context_actions = True
```

所以 06c posterior 应解释为“在 best Global 预选择规则成立的条件下”的策略后验，而不是未经模型选择偏差修正的生成式后验。

## 14. 07c 人工规则修正

07c 保留 06c 的原始 `<player>_strategy_posterior`、`<player>_strategy` 和模型 metadata，不直接覆盖概率结果。进入现有事件修正规则前：

1. 将 context posterior 作为临时初始策略分数；
2. 从 raw Q 重新执行 06c 合法方向 Min-Max；
3. Global 使用 `<player>_selected_global_Q`；
4. 死亡、不可用和无动作行继续排除；
5. 按既有 07 顺序执行 vague、approach、energizer 和 scared-time 修正。

人工规则可能把分数改成 one-hot 或 multi-hot，因此修正结果保存为：

- `<player>_revised_strategy_score`
- `<player>_revised_strategy`
- `<player>_revised_strategy_name`
- `<player>_strategy_revised`

`revised_strategy_score` 不是 posterior，不要求总和为 $1$。分析时应明确区分 06c 的模型后验和 07c 的规则修正标签；视频优先显示 07c 标签，但两套结果都保留在同一个文件中。
