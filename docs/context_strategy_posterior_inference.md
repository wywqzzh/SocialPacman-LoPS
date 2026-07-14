# 基于 Context 的潜在策略后验推断

> 当前实现版本：`strategy-posterior-v1`（代码状态核对日期：2026-07-14）。本版使用
> context 级策略信息覆盖率门控；Null 均匀动作模型只作为诊断基线，不参与 beta
> 拟合或 posterior。07 当前输出版本为 `strategy-revision-v1`。

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

05 utility 输入已经提供：

- 每个 tile 的真实动作和合法方向；
- 每个玩家的位置、存活状态；
- 七个策略在四个方向上的 Q 值。

06 使用正式玩家事件规则，为 P1/P2 分别划分 context，并在输出中生成
`<player>_trial_context`。

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

### 2.1 05 实际生成的正式 Q 与候选 utility

05 同时保存“七策略正式 Q”和“供 06 选择目标的候选矩阵”，二者不能混为同一种
算法：

| 策略 | 正式 raw Q | 额外候选 |
|---|---|---|
| Global | 按首方向对应的矩形区域统计距离严格大于 10 的普通豆数量；不计 Energizer | 普通豆与 Energizer 同权作为资源点，按地图最短路聚类；每团 Q 为该方向使最近目标距离减少的幅度乘以参与距离计算的资源数 |
| Local | 深度 10 的非立即折返路径树；普通豆奖励 2、Energizer 奖励 4，逐步乘 $0.9^{d-1}$，每个首方向取最佳叶路径 | 无 |
| Evade | Blinky/Clyde 分开计算；深度 6，仅状态 1/2 的 Ghost 碰撞产生 -8，按叶路径均值聚合 | 无 |
| Approach | 深度 20，对任意非 dead Ghost 命中奖励 8，乘 $0.95^{d-1}$，每个首方向取最佳叶路径 | 每只非 dead Ghost 单独作为目标，计算 20 步内最短非立即折返命中路径 |
| Energizer | 深度 10 的路径树，可累计多个 Energizer 奖励，按叶路径均值聚合 | 每颗 Energizer 单独作为目标，Q 为移动一步前后地图最短路距离差，不设搜索半径 |
| NoEnergizer | 深度 8 的路径树，踩到 Energizer 产生 -4，按叶路径均值聚合 | 无 |

Cluster Global 使用地图最短路而不是坐标欧氏距离。默认聚类阈值为 2，允许单点团，
并使用 union-find 传递闭包；参与某行 Global 方向距离计算的资源最小距离为 2，最大
目标距离为 60。每行的 `cluster_id` 会重新编号，跨行目标身份依赖 meta 中完整的
`resource_positions`，不能只比较整数 ID。

05 保存的旧 `Q_norm` 也不是统一 Min-Max：Global、Local、Approach、Energizer 逐行
除以有限最大值；Evade、NoEnergizer 先按玩家和策略减去文件内有限最小值，再逐行除
以最大值；候选矩阵在存在正值时除以最大正值，否则合法方向置 0。06 不直接使用这些
旧 norm，而是从 raw Q 重新执行第 3 节的合法方向 Min-Max。

05 CLI 仍保留 `--global-depth` 和 `--global-ignore-depth`，但当前正式 Global 没有读取
这两个配置，距离阈值固定为 10。`UtilityConfig` 中各策略的 `*_ghost_*_thr` 字段当前
也没有被路径搜索消费；它们只是保留字段，不应解释为本轮实验的有效参数。

相关字段包括：

- `<player>_action_dir`
- `<player>_available_dir`
- `<player>_alive`
- `<player>_<strategy>_Q`
- `<player>_global_utility_k`
- `<player>_global_utility_k_norm`
- `<player>_global_utility_k_meta`
- `<player>_energizer_utility_k`
- `<player>_energizer_utility_k_norm`
- `<player>_energizer_utility_k_meta`
- `<player>_approach_utility_k`
- `<player>_approach_utility_k_norm`
- `<player>_approach_utility_k_meta`

其中 `<player>` 为 `p1` 或 `p2`。

死亡、动作缺失、`available_dir=False` 或真实动作不合法的行不进入 likelihood。完全没有有效动作的长停留段直接标记为 `stay`。

### 2.2 普通豆边界与强事件窗口

06 使用同一套玩家事件与公共吃鬼 context 划分。该流程以 0-based **事件点**为基本单位：先检测事件发生行，再按规则删除事件点，最后才对排序后的相邻事件点生成 context。事件检测阶段不提前生成 context，也不把导致事件的前一动作行当作事件行。

资源事件列标在事件发生后的到达行。例如第 77 行动作使玩家在第 78 行吃到 energizer，则 `eat_energizer=True` 位于 78，context 强边界也只能是 78。第 77 行可在策略归因阶段被解释为导致事件的动作，但不能因此成为事件边界。

连续吃普通豆需要借助动作序列识别同一次连续采食过程，短 stay 可以按既定规则连接前后吃豆记录；识别完成后只输出第一个和最后一个 `eat_bean=True` 的实际事件行，不把内部动作范围写入边界集合。

trial 起止、当前 Pacman 生死变化、当前玩家吃 energizer、任一玩家吃 ghost、长 stay
起止属于强事件点，必须始终保留。连续吃普通豆过程的首尾事件属于可抑制的弱硬边界。

实现把强事件分成两类。trial 起止和最终保留的长 stay 起止属于方向性边界集合
$E_d$；Pacman 生死变化、当前玩家吃 energizer、任一玩家吃 ghost 属于对称行为强
事件集合 $E_s$。默认窗口为 3 个 tile。

方向性边界只取消事件后的吃豆开始点和事件前的吃豆结束点：

$$
B_{s,\mathrm{drop}}
=
\left\{
b\in B_s:
\exists e\in E_d,\ 0\le b-e\le 3
\right\}.
$$

$$
B_{e,\mathrm{drop}}
=
\left\{
b\in B_e:
\exists e\in E_d,\ 0\le e-b\le 3
\right\}.
$$

对称行为强事件会取消其前后窗口内的任意普通豆开始或结束边界：

$$
B_{\mathrm{sym,drop}}
=
\left\{
b\in B_s\cup B_e:
\exists e\in E_s,\ |b-e|\le 3
\right\}.
$$

最终删除集合是上述三部分的并集。强事件自身永远不会被删除；距离使用同一 trial
内的时间 tile 下标，不使用地图空间距离。该对称规则会把死亡、吃 Energizer 或公共
吃 Ghost 周围的普通豆碎片并入同一个行为转变，而 trial/stay 仍保留另一侧采食过程
的起止语义。

普通转向和掉头都不切段，而是作为 context 内动作参与策略 likelihood。另一名玩家
吃 energizer 仍作为当前玩家的公共环境软边界；软边界产生的小于 4 行的短段会在
同一硬边界区间内与相邻段合并，不会跨越硬边界。任一玩家吃 ghost 会立即移除共享
环境中的追逐目标，因此 P1/P2 都在同一真实事件行建立不可合并的公共强边界。玩家
进入或离开普通豆 10 步范围不再属于 context 划分条件。

长 stay 还需要接受公共吃 ghost 事件的二次过滤。设长 stay 区间为 $S=[s,e)$，任一玩家吃 ghost 的事件行为 $g$，事件行到 stay 最近实际行的距离为：

$$
d(g,S)
=
\begin{cases}
0, & s\le g<e,\\
s-g, & g<s,\\
g-(e-1), & g\ge e.
\end{cases}
$$

如果存在任一玩家的吃 ghost 事件满足：

$$
d(g,S)\le 5,
$$

则取消该 stay 的开始和结束硬边界。原始无动作行仍然保留，公共吃 ghost 边界也始终保留；取消的只是 ghost 交互或动画停顿对 context 的额外切段作用。

### 2.3 Context 内选择 Best Global、Energizer 与 Approach

05 为每一行保存多个资源团对应的 Global utility。06 不把这些候选视为多个顶层策略，也不对候选 likelihood 做边际化。context 划分完成后，先使用玩家事件规则计算每个 Global cluster 单独预测该 context 真实动作的概率准确率：

05 在计算每个候选时保留完整资源团用于跨行身份匹配，但距离小于 2 的单个资源不参与
本行 Global 距离计算。若同团仍有距离至少为 2 的资源，Global 使用这些远资源继续计算
方向；只有整团都落入距离 0/1 的 Local 范围时，该候选才成为无信息。

$$
k_c^\ast
=
\underset{k}{\arg\max}\;
\operatorname{Accuracy}_{c,k}.
$$

真实方向属于 $m$ 个正向并列最大方向时，该行贡献 $1/m$；候选全零或没有正向推进时贡献 $0$。准确率相同时，依次使用集合准确率、cluster size、起点距离和 cluster ID 破平。

选中的候选在整个 context 中作为唯一的 Global raw Q，随后与另外六种策略使用同一归一化和 likelihood 规则。输出同时保存 best cluster ID、准确率和资源团 meta，便于解释 Pacman 具体朝向哪一团资源。

Energizer 和 Approach 使用相同的“先选目标、再进入七策略比较”结构，但跨行身份键
不同：

- Energizer 以目标坐标为稳定键；context 起点的每颗 Energizer 都是候选，中途目标
  消失或该行不存在候选时，该行 selected Q 的合法方向全部置 0；
- Approach 以 `ghost1`/`ghost2` 身份为稳定键，不用会随时间变化的坐标匹配；context
  起点的每只非 dead Ghost 都是候选，目标死亡或消失后的行同样记为无信息；
- 两类候选都先比较概率准确率、再比较集合准确率和起点距离；最后分别用目标坐标或
  Ghost 编号破平；
- 目标选择只使用当前 context 的 utility 和真实动作，不读取最终是否吃到目标。事件
  结果留给 07 规则修正。

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

这表示该策略在当前 tile 对合法方向没有区分信息。后续 likelihood 会使用固定无信息
惩罚，不允许使用并列 `argmax` 将其计为有效预测。

### 3.1 Context 级策略信息覆盖率

strategy-posterior-v1 不再允许长期无方向信息的策略仅凭均匀动作概率进入 posterior。对 context $c$
中的每个策略 $k$，若某个有效动作行的合法方向归一化 Q 最大值与最小值之差大于
$\epsilon=10^{-12}$，则该行对该策略记为“有方向信息”。信息覆盖率定义为：

$$
\rho_{c,k}
=
\frac{
\#\{t\in\mathcal T_c:\max\widetilde Q_{t,k}-\min\widetilde Q_{t,k}>\epsilon\}
}{|\mathcal T_c|}.
$$

默认只有满足

$$
\rho_{c,k}\ge 0.50
$$

的策略才是该 context 的 eligible 行为策略。无信息行仍留在分母中，不会被静默删除。
该门控是 context 级候选筛选：策略在某一行无信息时，该行使用固定无信息惩罚；
只有整个 context 的覆盖率不足时，该策略才被排除出 beta loss 和 posterior。

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

令当前 context 中 coverage 合格的策略集合为
$\mathcal K_c=\{k:\rho_{c,k}\ge 0.50\}$。strategy-posterior-v1 只在该集合内部使用均匀先验：

$$
\pi_{c,k}=P(z_c=k)=\frac{1}{|\mathcal K_c|},
\qquad k\in\mathcal K_c.
$$

coverage 不合格的策略不占先验质量，其候选 log-likelihood 被门控为 $-\infty$。因此，
每个 context 不再单独拟合七个权重；长度仍为 7 的 posterior 由当前文件选定的
temperature 和该段行为计算，但不合格策略的位置固定为 0。

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

如果某个策略在所有合法方向上的 Q 都相同，说明该策略不能解释当前动作。strategy-posterior-v1
不再把该行作为普通均匀 softmax，而是在均匀动作 log probability 上增加固定惩罚：

$$
\log P_k(a_t\mid s_t)
=
-\log|\mathcal A_t|-\lambda_{\mathrm{noinfo}},
\qquad
\lambda_{\mathrm{noinfo}}=2.
$$

固定惩罚不依赖 $\beta$，因此无信息比例不会通过该项直接改变 temperature 的尺度。
只有一个合法方向时玩家没有行为选择，保持 log probability 为 $0$。全零 Q 同时仍会
降低 context 信息覆盖率，不会因为并列 `argmax` 得到伪预测准确率。

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
若 $k\notin\mathcal K_c$，后续计算使用门控值 $\ell'_{c,k}=-\infty$；未门控的
$\ell_{c,k}$ 仍保存在输出中供诊断。

## 7. 计算策略后验

对 coverage 合格策略，未归一化对数后验分数为：

$$
r_{c,k}=\ell_{c,k},\qquad k\in\mathcal K_c.
$$

使用 `logsumexp` 计算归一化常数：

$$
\log Z_c
=
\operatorname{logsumexp}_{k\in\mathcal K_c}(r_{c,k}).
$$

策略后验为：

$$
\gamma_{c,k}
=
P(z_c=k\mid\mathbf a_c,\mathbf s_c)
=
\exp\left(r_{c,k}-\log Z_c\right).
$$

对 $k\notin\mathcal K_c$，定义 $\gamma_{c,k}=0$。均匀先验在 eligible 集合内为相同
常数，归一化时抵消，但在 beta 的边际 NLL 中需要保留 $\log|\mathcal K_c|$。

满足：

$$
\sum_{k=1}^{K}\gamma_{c,k}=1,
\qquad |\mathcal K_c|>0.
$$

最终候选策略为：

$$
k_c^\ast
=
\underset{k\in\mathcal K_c}{\arg\max}\;\gamma_{c,k}.
$$

若 context 有有效动作但 $|\mathcal K_c|=0$，代码不强行选择七策略之一：posterior
保存为七个 0、candidate 保存为 `none`，最终策略标签为 `vague`。这类 context 也不
进入 beta 拟合、BIC 或交叉验证。若整个文件没有任何可拟合 player-context，06 会
直接报错。当前双人实现还会分别拟合 P1/P2 的独立 beta，因此只要任一玩家没有可拟合
context，独立模型拟合也会报错；双人文件当前要求两位玩家都至少有一个 eligible
player-context。

## 8. Beta 的 Loss

因为 context 的真实策略没有被直接观测，需要对 coverage 合格策略求边际概率。单个
有效 context 的负对数似然为：

$$
\mathcal L_c(\beta)
=
-\left[
\operatorname{logsumexp}_{k\in\mathcal K_c}\ell_{c,k}(\beta)
-\log|\mathcal K_c|
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

设文件 $f$ 中至少含一个 coverage 合格策略、因而可参与拟合的 player-context 总数为
$C_f$。stay、无有效动作以及 $|\mathcal K_c|=0$ 的 context 不计入 $C_f$。由于
likelihood 按 context 分解，BIC 的样本数使用 $C_f$，而不是 tile 数。

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

1. 在四个训练 folds 上分别拟合共享模型和玩家独立模型；若某折使任一玩家的训练集
   为空，则该折跳过，不构造无依据 beta；
2. 记录训练数据上的 BIC 选择是否稳定；
3. 在验证 fold 上比较 held-out NLL；
4. 检查各 fold 的 temperature 是否稳定；
5. 完成交叉验证后，在当前文件全部可拟合 contexts 上重新拟合两个模型；
6. 使用全文件 BIC 确定最终采用一个还是两个 temperature。

如果共享模型胜出，保存 $\beta_f$；如果玩家独立模型胜出，保存 $\beta_{f,p1}$ 和 $\beta_{f,p2}$。该选择只对当前文件生效。
单人文件不比较共享/独立结构，直接拟合一个 beta，并把模型类型记录为 `single`。
metadata 中 `fold_count` 当前记录已分配的 fold 数，而 `folds` 只记录实际完成拟合的
fold；极小数据触发上述跳过规则时，两者可能不相等。

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
\max_{k\in\mathcal K_c}\ell_{c,k}-\ell_{c,\mathrm{null}}
}{
|\mathcal T_c|
}.
$$

strategy-posterior-v1 在以下条件成立时标记为 `vague`：

$$
\max_k\gamma_{c,k}<\tau_{\mathrm{posterior}},
$$

$\tau_{\mathrm{posterior}}$ 默认固定为 $0.70$。$G_c$ 作为诊断字段保存，但不参与
`vague` 判定。Null 模型也不进入 beta loss、posterior 或候选策略集合；它不能否决
coverage 充分但偶有高置信度错误的行为策略。若未来增加 gain 阈值，应通过验证集、
打乱动作或模拟 null 数据确定，不能直接使用经验常数。

有有效动作但没有任何 coverage 合格策略的 context 直接标记 `vague`。完全没有有效
动作的 context 标记 `stay`，不进行 posterior 推断。


## 11. 双人处理与输出

P1 和 P2 分别使用自己的 context、真实动作、合法方向和 Q 值计算后验。两名玩家使用共享还是独立 temperature，由当前文件的 BIC 结果决定。

实际逐行保存：

- `<player>_strategy_log_likelihood`
- `<player>_strategy_information_coverage`
- `<player>_strategy_eligible`
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
- `<player>_selected_energizer_Q`
- `<player>_best_energizer_target_position`
- `<player>_best_energizer_target_prob_accuracy`
- `<player>_best_energizer_target_set_accuracy`
- `<player>_best_energizer_target_meta`
- `<player>_selected_approach_Q`
- `<player>_best_approach_target_id`
- `<player>_best_approach_target_position`
- `<player>_best_approach_target_prob_accuracy`
- `<player>_best_approach_target_set_accuracy`
- `<player>_best_approach_target_meta`

P1、P2 的结果继续写回同一份 joint-state 表，保持两个玩家的时间对齐。
06 还会写入 `<player>_eat_bean`、`<player>_eat_energizer` 和
`<player>_eat_ghost` 事件列；它们是 context 构造和 07 结果修正使用的玩家级事件，
不能与原始公共资源列表混为一列。

文件级模型元数据还需要保存：

- BIC 选择的模型类型；
- 共享 $\beta_f$，或者独立的 $\beta_{f,p1}$、$\beta_{f,p2}$；
- $\operatorname{BIC}_{\mathrm{shared}}$ 和 $\operatorname{BIC}_{\mathrm{separate}}$；
- grouped cross-validation 的 fold 划分和验证 NLL；
- `version=strategy-posterior-v1`、信息覆盖率阈值、固定无信息惩罚和浮点容差；
- Null 模型仅诊断、不会参与 beta 或 posterior 的显式标记。

## 12. 完整流程

1. 读取 raw Q、动作、合法方向和 Global cluster 候选；
2. 使用玩家私有事件规则划分 context；
3. 在每个 player-context 内分别选择 best Global cluster、best Energizer 目标和 best
   Approach Ghost；
4. 从三类 selected raw Q 和另外四种 raw Q 统一计算 $\widetilde Q$；
5. 计算每个 player-context、每个策略的信息覆盖率和 eligibility；
6. 排除 stay、无有效动作和没有 eligible 策略的 context，并按完整 `DayTrial` 建立 grouped folds；
7. 分别拟合“P1/P2 共享一个 $\beta$”和“P1/P2 各一个 $\beta$”两个模型；
8. 使用交叉验证检查 temperature、BIC 选择和 held-out NLL 是否稳定；
9. 在当前文件全部有效 contexts 上重新拟合两个模型，并用 BIC 确定最终参数结构；
10. 使用选定 temperature，在每个 context 的 eligible 策略内计算 posterior；
11. 根据 posterior 阈值或无 eligible 策略条件判断具体策略/`vague`，并将 P1/P2 结果写回联合数据。

## 13. Global 选择的解释约束

06 的 best Global cluster 根据当前 context 的真实动作选择，随后同一批动作还会用于计算选中 Global 的 likelihood。因此，Global 相比只有单一 utility 的其他策略拥有额外候选机会，其 likelihood 和 posterior 可能存在乐观偏差。

本版本接受该约束，因为研究目标明确要求先找出当前 context 中解释动作最好的资源团，再将该资源团作为唯一 Global 参与七策略比较。输出 metadata 必须保存：

```text
global_selection_uses_context_actions = True
```

所以 06 posterior 应解释为“在 best Global 预选择规则成立的条件下”的策略后验，而不是未经模型选择偏差修正的生成式后验。

## 14. 07 人工规则修正

07 保留 06 的原始 `<player>_strategy_posterior`、`<player>_strategy` 和模型 metadata，不直接覆盖概率结果。进入现有事件修正规则前：

1. 将 context posterior 作为临时初始策略分数；
2. 从 raw Q 重新执行 06 合法方向 Min-Max；
3. Global 使用 `<player>_selected_global_Q`；
4. Energizer 使用 `<player>_selected_energizer_Q`；
5. 死亡、`available_dir=False` 和无动作行继续排除。当前 07 不再次检查真实动作是否
   落在 raw Q 的合法方向 mask；若出现地图/Q mask 不一致，该行会进入准确率分母并记
   为未命中，因此其有效动作集合可能比 06 更宽；
6. 复用当前 07 的 vague、energizer 后 approach 和 scared-time 规则；“未亲自吃到
   ghost 就否定 Approach”的二次修正规则与旧错误 Energizer 回滚规则当前停用。

对于以玩家私有吃 Energizer 事件结束的 context，strategy-revision-v1 只有在 Energizer 的单策略
准确率同时满足

$$
A_E\ge 0.70,
\qquad
\frac{A_E}{A_{\max}}\ge 0.80
$$

时才把整段修正为 Energizer。这样允许 Energizer 略低于另一种解释能力最强的策略，
但不会让一次成功结果覆盖整段中不足的行为证据。没有吃到 Energizer 时，只有当
Energizer 与其它策略精确并列最优，才从并列集合中移除 Energizer；唯一最优结果
不会仅因最终没有吃到而被否定。

人工规则可能把分数改成 one-hot 或 multi-hot，因此修正结果保存为：

- `<player>_revised_strategy_score`
- `<player>_revised_prediction_correct`
- `<player>_revised_predict_dir`
- `<player>_revised_is_vague`
- `<player>_revised_strategy`
- `<player>_revised_strategy_name`
- `<player>_strategy_revised`

`revised_strategy_score` 不是 posterior，不要求总和为 $1$。分析时应明确区分 06 的模型后验和 07 的规则修正标签；视频优先显示 07 标签，但两套结果都保留在同一个文件中。

strategy-revision-v1 还要求输入含有 `<player>_strategy_information_coverage` 和
`<player>_strategy_eligible`，并在 attrs 中记录初始分数来自
`coverage_gated_strategy_posterior`。它不宣称完整复刻旧 07 的规则顺序；输出 metadata
明确保存 `historical_rule_order_reused=False` 和
`uneaten_ghost_approach_revision_enabled=False`。

当前规则中还包含以下会影响最终标签的阈值和并列选择：

- vague context 只有在有效动作比例至少为 `0.50`，且最佳单策略准确率至少为 `0.70`
  时才取消 vague；这里没有有效动作绝对数量门槛；
- Energizer 后续 Approach 修正要求 Approach 准确率大于 `0.60`，且相对最佳准确率
  至少为 `0.75`；
- 其它按主策略执行的规则默认要求主策略准确率大于 `0.60`，且相对最佳准确率严格
  大于 `0.80`；
- 多个策略准确率并列最高时，修正分数保留 multi-hot。若一个 context 超过一半行
  存在 `ifscared>=4` 的 Ghost，且 Approach 位于最大值集合，则 Approach 优先；否则
  Local 优先于 Global。剩余无法由既定并列规则唯一解释的组合标记为 vague；
- scared-time 的 Approach 段可在 34 tile 窗口内合并，具体仍受吃 Energizer 后续
  Approach 筛选条件约束。
