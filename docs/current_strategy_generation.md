# 当前策略生成方法

本文描述当前正式 `05 → 06 → 07` 流程。方向顺序固定为：

$$
\mathcal D=(\mathrm{left},\mathrm{right},\mathrm{up},\mathrm{down}).
$$

不可走方向的 utility 为 $-\infty$。地图距离 $D(x,y)$ 均指
`map_constants.pkl` 中的最短路距离，包含左右通道连接。
下文用 $\mathcal A_t$ 表示 tile $t$ 的合法方向集合，$a_t$ 表示真实动作，
$\mathcal T_c$ 表示 context $c$ 中的有效动作行。

## 1. 不同策略 Utility 的计算方式

### 1.1 Global

将当前剩余普通豆和 Energizer 合并为资源集合。若两个资源的地图距离不超过 2，则
连接为同一资源团；使用传递闭包得到多个 cluster，单点也可独立成团。

对位置 $p_t$ 和资源团 $c$，当前行先排除距离小于 2 的近处资源：

$$
R^G_{t,c}=\{r\in c: D(p_t,r)\ge 2\}.
$$

记 $D(p,R)=\min_{r\in R}D(p,r)$，则方向 $d$ 的候选 utility 为：

$$
Q^{\mathrm{global}}_{t,c,d}
=
|R^G_{t,c}|
\left[D(p_t,R^G_{t,c})-D(p_{t,d}',R^G_{t,c})\right].
$$

其中 $p_{t,d}'$ 是沿方向 $d$ 移动一格后的位置。只有
$D(p_t,R^G_{t,c})\le60$ 时该候选才提供信息；否则合法方向 utility 为 0。60 步限制
检查最近 Global 资源，不会逐个删除同团中更远的资源。普通豆和 Energizer 在 Global
中权重相同。

### 1.2 Local

从每个合法首方向展开深度 10 的路径树，禁止立即回到上一格。普通豆奖励为 2，
Energizer 奖励为 4；同一资源只在首次到达时计分。路径 $\pi$ 的 utility 为：

$$
U_{\mathrm{local}}(\pi)
=
\sum_{j=1}^{|\pi|}
0.9^{j-1}
\left[
2I^{\mathrm{bean}}_j+4I^{\mathrm{energizer}}_j
\right].
$$

每个首方向取该方向下最佳叶路径：

$$
Q^{\mathrm{local}}_{t,d}
=
\max_{\pi\in\Pi_d^{10}}U_{\mathrm{local}}(\pi).
$$

### 1.3 Evade Blinky / Evade Clyde

两只 Ghost 分别形成一个策略。每个策略展开深度 6 的路径树；只有目标 Ghost 状态为
1 或 2 且路径与其当前位置相撞时产生 $-8$，并终止该路径。每个首方向取叶路径均值：

$$
Q^{\mathrm{evade}(g)}_{t,d}
=
\frac{1}{|\Pi_d^6|}
\sum_{\pi\in\Pi_d^6}U_{\mathrm{evade}(g)}(\pi).
$$

### 1.4 Approach

每只非 dead Ghost 分别形成一个候选目标。固定首方向 $d$ 后，搜索 20 步内首次命中
目标 Ghost 的最短非立即折返路径，长度记为 $L_{t,g,d}$：

$$
Q^{\mathrm{approach}}_{t,g,d}
=
\begin{cases}
8\cdot0.95^{L_{t,g,d}-1}, & L_{t,g,d}\le 20,\\
0, & \text{目标在 20 步内不可达}.
\end{cases}
$$

正常、scared 和 flashing Ghost 都可以作为 Approach 目标；状态 3 的 dead Ghost 不参与。

### 1.5 Energizer

每颗剩余 Energizer 分别形成一个候选目标。对目标 $e$：

$$
Q^{\mathrm{energizer}}_{t,e,d}
=
D(p_t,e)-D(p_{t,d}',e).
$$

该 utility 不设距离上限；正值表示接近目标，负值表示远离目标。

### 1.6 No Energizer

展开深度 8 的路径树。首次经过 Energizer 产生 $-4$；Ghost 碰撞只终止路径，不额外
增加 utility。每个首方向取叶路径均值：

$$
Q^{\mathrm{no\_energizer}}_{t,d}
=
\frac{1}{|\Pi_d^8|}
\sum_{\pi\in\Pi_d^8}U_{\mathrm{no\_energizer}}(\pi).
$$

最终模型包含七个策略：Global、Local、Evade Blinky、Evade Clyde、Approach、
Energizer 和 No Energizer。05 还保留旧的混合目标 Global/Approach/Energizer Q 供
诊断，但 06 拟合实际使用上述 context 内选定目标的 Q。

## 2. Context 的划分方式

P1、P2 分别划分 context。所有事件都标在事件实际发生的到达行，context 使用半开区间
$[s,e)$。

### 2.1 硬边界

以下边界不可跨越：

1. trial 起点和终点；
2. 当前玩家死亡或复活；
3. 当前玩家吃 Energizer；
4. 任一玩家吃 Ghost；
5. 当前玩家连续吃普通豆过程的开始和结束；
6. 连续缺失动作长度至少为 4 的 long stay 起点和终点。

短于 4 tile 的 stay 可以连接前后连续吃豆过程。普通转向和掉头不产生边界。

### 2.2 普通豆边界抑制

默认抑制窗口为 3 tile。

- trial/long stay 边界只删除事件后 3 tile 内的吃豆开始边界，以及事件前 3 tile 内的
  吃豆结束边界；
- 死亡/复活、本人吃 Energizer、任一玩家吃 Ghost 会删除其前后 3 tile 内的所有吃豆
  开始或结束边界；
- 若 long stay 距任一玩家吃 Ghost 事件不超过 5 tile，则取消该 stay 的起止边界，
  但保留吃 Ghost 边界。

### 2.3 软边界与短段合并

队友吃 Energizer 是当前玩家的软边界。软边界产生的长度小于 4 的 context 会在同一
硬边界区间内与相邻段合并，且绝不跨越硬边界。Ghost 状态自然恢复不产生边界。

## 3. Context 内多 Utility 策略的选择

Global、Energizer、Approach 各有多个目标候选。候选只从 context 起点仍存在的目标中
选择，并在整个 context 中保持同一目标身份：

- Global：用起点资源坐标集合匹配后续行中重叠最多的资源团；
- Energizer：用 Energizer 坐标匹配；
- Approach：用 `ghost1` 或 `ghost2` 身份匹配。

对候选 $j$ 和有效动作行 $t$，令正向最大方向集合为：

$$
M_{t,j}
=
\arg\max_{d\in\mathcal A_t}Q_{t,j,d},
\qquad
\max_{d\in\mathcal A_t}Q_{t,j,d}>0.
$$

其单行动作得分为：

$$
s_{t,j}
=
\begin{cases}
1/|M_{t,j}|, & a_t\in M_{t,j},\\
0, & \text{其它情况}.
\end{cases}
$$

候选中途消失、无法跨行匹配或没有正向信息时记 0 分。候选准确率为：

$$
A_{c,j}=\frac{1}{|\mathcal T_c|}\sum_{t\in\mathcal T_c}s_{t,j}.
$$

选择 $A_{c,j}$ 最大的候选。并列时依次比较集合命中率和起点距离；Global 还优先选择
更大的资源团，最终用较小 cluster ID 破平；Energizer 用较小目标坐标破平；Approach
用较小 Ghost 编号破平。选定目标中途消失后，该策略在对应行的合法方向 utility 全为 0。

## 4. 基于 Context 的拟合

有效动作行必须同时满足：动作存在、玩家存活、`available_dir=True`，并且真实动作属于
Q 的合法方向。七种策略在同一行必须具有完全相同的非法方向 mask。

### 4.1 Q 归一化

对每个玩家、tile、策略，只在合法方向集合 $\mathcal A_t$ 内执行 Min-Max：

$$
\widetilde Q_{t,k,d}
=
\frac{Q_{t,k,d}-Q^{\min}_{t,k}}
{Q^{\max}_{t,k}-Q^{\min}_{t,k}}.
$$

若所有合法方向 Q 相等，则合法方向全部置 0；非法方向保持 $-\infty$。

### 4.2 信息覆盖率

若一个动作行的合法方向 Q 极差大于 $10^{-12}$，该策略在该行有方向信息。context
覆盖率为：

$$
\rho_{c,k}
=
\frac{1}{|\mathcal T_c|}
\sum_{t\in\mathcal T_c}
I\!\left(\max_d\widetilde Q_{t,k,d}-\min_d\widetilde Q_{t,k,d}>10^{-12}\right).
$$

只有 $\rho_{c,k}\ge0.50$ 的策略进入拟合和 posterior。

### 4.3 动作似然

有信息行采用 softmax：

$$
P(a_t\mid k,\beta)
=
\frac{\exp(\beta\widetilde Q_{t,k,a_t})}
{\sum_{d\in\mathcal A_t}\exp(\beta\widetilde Q_{t,k,d})}.
$$

若合法方向数为 $m_t>1$ 且该策略在该行无信息，则使用固定惩罚：

$$
\log P(a_t\mid k,\beta)=-\log m_t-2.
$$

该值不依赖 $\beta$。只有一个合法方向时，该动作 log-likelihood 为 0。context 的策略
log-likelihood 为：

$$
\ell_{c,k}(\beta)
=
\sum_{t\in\mathcal T_c}\log P(a_t\mid k,\beta).
$$

### 4.4 文件级 Beta

对 eligible 策略集合 $\mathcal K_c$ 使用均匀先验，单个 context 的边际 NLL 为：

$$
\mathcal L_c(\beta)
=
-\left[
\operatorname{logsumexp}_{k\in\mathcal K_c}\ell_{c,k}(\beta)
-\log|\mathcal K_c|
\right].
$$

每个文件独立在 $\beta\in[0.05,20]$ 内最小化所有有效 context 的总 NLL。双人文件同时
拟合共享一个 $\beta$ 和 P1/P2 各自一个 $\beta$ 两种模型，并使用：

$$
\operatorname{BIC}=2\mathcal L^{\min}+m\log C
$$

选择 BIC 较小者，其中 $m$ 是 beta 数量，$C$ 是有效 player-context 数量。按完整
`DayTrial` 的最多 5 折交叉验证只用于诊断；最终 beta 在全文件上重新拟合。

## 5. 根据拟合结果选择策略

对 eligible 策略计算 context posterior：

$$
\gamma_{c,k}
=
\frac{\exp(\ell_{c,k})}
{\sum_{j\in\mathcal K_c}\exp(\ell_{c,j})}.
$$

最终选择规则为：

1. context 没有有效动作：`stay`；
2. 没有 eligible 策略：`vague`；
3. 令 $k^*=\arg\max_k\gamma_{c,k}$；若 $\gamma_{c,k^*}<0.70$：`vague`；
4. 否则选择 $k^*$。完全并列时按七策略固定顺序取第一项。

Null 均匀动作模型只保存为诊断，不参与 beta、posterior 或策略选择。

## 6. 策略修正规则

07 保留原始 posterior，并把它复制为初始策略分数。修正只排除动作缺失、玩家死亡和
`available_dir=False` 的行；它不会再次用 raw Q mask 排除非法真实方向。各策略单独
预测动作的概率准确率为：有信息且真实方向属于 $m$ 个并列最大方向时记 $1/m$，无
信息、非法真实方向或预测错误记 0。

规则按以下顺序执行，每一步后重新确定策略标签。

### 6.1 Vague 修正

原始 `vague` context 只有同时满足以下条件时才修正：

$$
\frac{N_{\mathrm{valid}}}{N_{\mathrm{context}}}\ge0.50,
\qquad
\max_k A_{c,k}\ge0.70.
$$

所有并列最高策略写成 multi-hot 分数；没有有效动作绝对数量限制。

### 6.2 Energizer 结果修正

若 context 的结束边界记录当前玩家吃到 Energizer，则仅在

$$
A_{c,E}\ge0.70,
\qquad
\frac{A_{c,E}}{\max_k A_{c,k}}\ge0.80
$$

时将该 context 修正为 Energizer。

若结束边界没有吃到 Energizer，且 Energizer 与其它策略并列最高，则只从并列集合中
移除 Energizer；Energizer 唯一最高时不因“未吃到”而取消。

### 6.3 Energizer 后的 Approach 修正

吃 Energizer context 后的首个 context 先经过结构筛选：若该段到结束边界之间出现
至少一只 Ghost 的 `ifscared=3`，则保留；否则只有其后一行在当前修正状态下仍标为
Approach 时才保留。对保留的 context，若

$$
A_{c,A}>0.60,
\qquad
\frac{A_{c,A}}{\max_k A_{c,k}}\ge0.75,
$$

则修正为 Approach。

同一次 scared-time 内，若两个 Approach context 的起点相差不超过 34 tile，则把从
前一段起点到后一段终点的完整区间合并为 Approach。

### 6.4 并列策略的显示优先级

若 Approach 位于最大分数集合，并且 context 中超过一半行至少有一只 Ghost 满足
`ifscared >= 4`，则优先显示 Approach。否则：

1. Local 优先于其它并列策略；
2. 其次是 Global；
3. 仅 Evade 策略并列时按固定顺序选择；
4. 其它无法唯一解释的并列组合标记为 `vague`。

修正后的 one-hot/multi-hot 分数不是 posterior。当前不执行“玩家没有亲自吃到 Ghost
就否定 Approach”的规则，也不执行旧版错误 Energizer 回滚规则。
