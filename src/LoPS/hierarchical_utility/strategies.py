"""fMRI hierarchical utility 的高效策略估计实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .model import CompiledFrameState, CompiledMapData, DIRECTIONS, GHOST_NAMES, UtilityConfig


GLOBAL_Q_COLUMN = "global_Q"
PATH_Q_COLUMNS: tuple[str, ...] = (
    "local_Q",
    "evade_blinky_Q",
    "evade_clyde_Q",
    "approach_Q",
    "energizer_Q",
    "no_energizer_Q",
)
LOCAL_INDEX = 0
EVADE_START_INDEX = 1
APPROACH_INDEX = 3
ENERGIZER_INDEX = 4
NO_ENERGIZER_INDEX = 5
STRATEGY_COUNT = len(PATH_Q_COLUMNS)


@dataclass(frozen=True, slots=True)
class StrategyState:
    """保存单个路径策略在一个节点上的局部状态。

    输入语义：状态来自根节点初始化，或父节点走一步后的状态转移。
    输出语义：共享路径引擎用它继续展开、记录叶节点 utility 或生成下一层状态。
    关键约束：对象集合用 bitmask 表示，ghost_status 使用 tuple 保持不可变，避免内层循环拷贝大对象。
    """

    utility: float
    bean_mask: int
    energizer_mask: int
    ghost_status: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class SharedSearchNode:
    """保存共享路径树中的一个几何节点。

    输入语义：节点表示 Pacman 沿某条路径走到当前位置后的共享几何信息。
    输出语义：节点携带每个仍在继续的策略状态，供下一层扩展。
    关键约束：不同策略可以在不同深度或不同事件处终止，因此 states 中允许使用 None 表示已记录叶节点。
    """

    position_id: int
    parent_id: int
    root_direction_id: int
    depth: int
    visited_mask: int
    states: tuple[StrategyState | None, ...]


class SharedPathUtilityEngine:
    """一次路径遍历同时估计所有路径型 utility 策略。

    输入语义：compiled_map 是整数化地图，frame_state 是整数化单帧状态，config 提供深度和系数。
    输出语义：`estimate()` 返回除 Global 外的 6 个 Q 向量，可选返回 leaf 统计 trace。
    关键约束：路径几何只展开一次；每个策略独立维护状态、终止条件和叶节点统计。
    """

    def __init__(self, compiled_map: CompiledMapData, frame_state: CompiledFrameState, config: UtilityConfig) -> None:
        """初始化共享路径搜索引擎。

        输入语义：compiled_map/frame_state/config 三者共同决定本帧所有路径型策略。
        输出语义：初始化策略深度、奖励常量、ghost 位置和空 leaf 统计。
        关键约束：路径方向顺序固定为 left/right/up/down，保证展开顺序稳定。
        """

        self.compiled_map = compiled_map
        self.frame_state = frame_state
        self.config = config
        self.reward_amount = compiled_map.reward_amount
        self.neighbor_ids = compiled_map.neighbor_ids
        self.strategy_depths = (
            config.local_depth,
            config.evade_depth,
            config.evade_depth,
            config.approach_depth,
            config.energizer_depth,
            config.no_energizer_depth,
        )
        self.strategy_laziness_coeffs = (
            config.laziness_coeff,
            config.laziness_coeff,
            config.laziness_coeff,
            0.0,
            config.laziness_coeff,
            config.laziness_coeff,
        )
        self.leaf_stats: list[list[list[float]]] = [
            [[0.0, 0.0] for _ in DIRECTIONS]
            for _ in PATH_Q_COLUMNS
        ]

    def estimate(self, *, return_trace: bool = False) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, Any]]:
        """估计本帧所有路径型策略的 Q 值。

        输入语义：return_trace 控制是否返回每个策略、每个初始方向的叶节点统计。
        输出语义：默认返回 Q 列名到四方向 Q 向量的字典；trace 模式额外返回 leaf 统计。
        关键约束：Q 聚合使用叶节点 utility 平均值，方向顺序固定为 left/right/up/down。
        """

        self._construct_shared_tree()
        q_values = self._build_q_values()
        if not return_trace:
            return q_values
        return q_values, self._build_trace(q_values)

    def _construct_shared_tree(self) -> None:
        """展开共享路径树并累积各策略叶节点统计。

        输入语义：根节点由当前 Pacman 位置和初始策略状态组成。
        输出语义：self.leaf_stats 被填充为 utility sum 与 leaf count。
        关键约束：几何路径继续展开到仍有至少一个策略 active，单个策略终止后不再参与后续节点。
        """

        root = SharedSearchNode(
            position_id=self.frame_state.pacman_id,
            parent_id=-1,
            root_direction_id=-1,
            depth=0,
            visited_mask=1 << self.frame_state.pacman_id,
            states=self._initial_strategy_states(),
        )
        frontier = [root]
        while frontier:
            next_frontier: list[SharedSearchNode] = []
            for node in frontier:
                self._expand_node(node, next_frontier)
            frontier = next_frontier

    def _expand_node(self, node: SharedSearchNode, next_frontier: list[SharedSearchNode]) -> None:
        """展开一个共享几何节点。

        输入语义：node 是当前层节点，next_frontier 收集下一层仍需继续搜索的节点。
        输出语义：叶节点直接写入 self.leaf_stats，未终止策略被写入下一层节点。
        关键约束：墙体方向跳过，非根节点禁止立即走回父节点位置。
        """

        created_child_count = 0
        child_depth = node.depth + 1
        for direction_id, next_position_id in enumerate(self.neighbor_ids[node.position_id]):
            if next_position_id == -1:
                continue
            if node.parent_id != -1 and next_position_id == node.parent_id:
                continue

            created_child_count += 1
            root_direction_id = direction_id if node.root_direction_id == -1 else node.root_direction_id
            child_states: list[StrategyState | None] = [None] * STRATEGY_COUNT
            has_active_strategy = False
            visited_before = bool(node.visited_mask & (1 << next_position_id))
            for strategy_index, state in enumerate(node.states):
                if state is None:
                    continue
                next_state, terminated = self._advance_strategy(
                    strategy_index,
                    state,
                    next_position_id,
                    visited_before,
                )
                if terminated or child_depth >= self.strategy_depths[strategy_index]:
                    self._add_leaf(strategy_index, root_direction_id, next_state.utility)
                else:
                    child_states[strategy_index] = next_state
                    has_active_strategy = True

            if has_active_strategy:
                next_frontier.append(
                    SharedSearchNode(
                        position_id=next_position_id,
                        parent_id=node.position_id,
                        root_direction_id=root_direction_id,
                        depth=child_depth,
                        visited_mask=node.visited_mask | (1 << next_position_id),
                        states=tuple(child_states),
                    )
                )

        if created_child_count == 0 and node.root_direction_id != -1:
            # 死胡同中的当前节点就是所有仍在继续策略的叶节点。
            for strategy_index, state in enumerate(node.states):
                if state is not None:
                    self._add_leaf(strategy_index, node.root_direction_id, state.utility)

    def _initial_strategy_states(self) -> tuple[StrategyState | None, ...]:
        """创建根节点上各路径策略的初始状态。

        输入语义：使用当前帧的 bean、energizer 和 ghost 状态。
        输出语义：返回长度固定为 6 的策略状态 tuple。
        关键约束：Evade 策略只携带被选择的单只 ghost 状态，其余策略携带完整 two-ghost 状态。
        """

        frame = self.frame_state
        base_state = StrategyState(0.0, frame.bean_mask, frame.energizer_mask, frame.ghost_status)
        states: list[StrategyState | None] = [base_state]
        for ghost_index in range(len(GHOST_NAMES)):
            states.append(
                StrategyState(
                    0.0,
                    frame.bean_mask,
                    frame.energizer_mask,
                    (frame.ghost_status[ghost_index],),
                )
            )
        states.extend(
            [
                base_state,
                base_state,
                base_state,
            ]
        )
        return tuple(states)

    def _advance_strategy(
        self,
        strategy_index: int,
        state: StrategyState,
        next_position_id: int,
        visited_before: bool,
    ) -> tuple[StrategyState, bool]:
        """计算某个策略走到下一位置后的状态和终止标记。

        输入语义：strategy_index 指定策略，state 是父节点状态，next_position_id 是下一位置。
        输出语义：返回子节点状态，以及该策略是否在子节点成为叶节点。
        关键约束：reward 更新和 risk 更新的先后关系按策略规则显式表达。
        """

        if strategy_index == LOCAL_INDEX:
            bean_mask, energizer_mask, ghost_status, exact_reward = self._apply_local_reward(
                state,
                next_position_id,
            )
            _, exact_risk, terminated = self._two_ghost_risk(state.ghost_status, next_position_id, visited_before)
            return (
                StrategyState(
                    state.utility + exact_reward + 0.0 * exact_risk,
                    bean_mask,
                    energizer_mask,
                    ghost_status,
                ),
                terminated,
            )

        if EVADE_START_INDEX <= strategy_index < APPROACH_INDEX:
            ghost_index = strategy_index - EVADE_START_INDEX
            bean_mask, energizer_mask, ghost_status, exact_reward = self._apply_local_reward(
                state,
                next_position_id,
            )
            exact_risk, terminated = self._single_ghost_evade_risk(
                state.ghost_status,
                self.frame_state.ghost_ids[ghost_index],
                next_position_id,
                visited_before,
            )
            return (
                StrategyState(
                    state.utility + 0.0 * exact_reward + exact_risk,
                    bean_mask,
                    energizer_mask,
                    ghost_status,
                ),
                terminated,
            )

        if strategy_index == APPROACH_INDEX:
            ghost_status, exact_reward, reward_terminated = self._apply_approach_reward(state, next_position_id)
            _, exact_risk, risk_terminated = self._two_ghost_risk(
                state.ghost_status,
                next_position_id,
                visited_before,
            )
            return (
                StrategyState(
                    state.utility + exact_reward + 0.0 * exact_risk,
                    state.bean_mask,
                    state.energizer_mask,
                    ghost_status,
                ),
                reward_terminated or risk_terminated,
            )

        if strategy_index == ENERGIZER_INDEX:
            energizer_mask, ghost_status, exact_reward = self._apply_energizer_reward(state, next_position_id)
            _, exact_risk, terminated = self._two_ghost_risk(state.ghost_status, next_position_id, visited_before)
            return (
                StrategyState(
                    state.utility + exact_reward + 0.0 * exact_risk,
                    state.bean_mask,
                    energizer_mask,
                    ghost_status,
                ),
                terminated,
            )

        bean_mask, exact_reward = self._apply_no_energizer_reward(state, next_position_id)
        energizer_mask, ghost_status, exact_risk, terminated = self._apply_no_energizer_risk(
            state,
            next_position_id,
            visited_before,
        )
        return (
            StrategyState(
                state.utility + 0.0 * exact_reward + exact_risk,
                bean_mask,
                energizer_mask,
                ghost_status,
            ),
            terminated,
        )

    def _apply_local_reward(self, state: StrategyState, next_position_id: int) -> tuple[int, int, tuple[Any, ...], float]:
        """计算 bean 和 energizer 的即时奖励状态转移。

        输入语义：state 是父节点策略状态，next_position_id 是下一步位置。
        输出语义：返回更新后的 bean/energizer/ghost 状态和即时奖励。
        关键约束：吃 energizer 后，非死亡 ghost 在下一节点进入 scared 状态。
        """

        exact_reward = 0.0
        bean_mask = state.bean_mask
        energizer_mask = state.energizer_mask
        ghost_status = state.ghost_status
        next_bit = 1 << next_position_id
        if bean_mask & next_bit:
            exact_reward += self.reward_amount[1]
            bean_mask &= ~next_bit
        if energizer_mask & next_bit:
            exact_reward += self.reward_amount[2]
            energizer_mask &= ~next_bit
            ghost_status = _scare_active_ghosts(ghost_status)
        return bean_mask, energizer_mask, ghost_status, exact_reward

    def _apply_approach_reward(self, state: StrategyState, next_position_id: int) -> tuple[tuple[Any, ...], float, bool]:
        """计算 Approach 策略的 ghost 接触奖励。

        输入语义：state 保存当前 ghost 状态，next_position_id 是下一步位置。
        输出语义：返回更新后的 ghost 状态、即时奖励和是否终止路径。
        关键约束：可接近 ghost 被吃后状态变为死亡；不可接近 ghost 会终止该策略路径。
        """

        exact_reward = 0.0
        terminated = False
        ghost_status = list(state.ghost_status)
        for ghost_index, ghost_id in enumerate(self.frame_state.ghost_ids):
            if ghost_status[ghost_index] != 3 and next_position_id == ghost_id:
                exact_reward += self.reward_amount[8]
                if ghost_status[ghost_index] > 3:
                    ghost_status[ghost_index] = 3
                else:
                    terminated = True
        return tuple(ghost_status), exact_reward, terminated

    def _apply_energizer_reward(self, state: StrategyState, next_position_id: int) -> tuple[int, tuple[Any, ...], float]:
        """计算 Energizer 策略的 energizer 即时奖励。

        输入语义：state 保存剩余 energizer 和 ghost 状态，next_position_id 是下一步位置。
        输出语义：返回更新后的 energizer bitmask、ghost 状态和即时奖励。
        关键约束：该策略不处理 bean 奖励。
        """

        exact_reward = 0.0
        energizer_mask = state.energizer_mask
        ghost_status = state.ghost_status
        next_bit = 1 << next_position_id
        if energizer_mask & next_bit:
            exact_reward += self.reward_amount[2]
            energizer_mask &= ~next_bit
            ghost_status = _scare_active_ghosts(ghost_status)
        return energizer_mask, ghost_status, exact_reward

    def _apply_no_energizer_reward(self, state: StrategyState, next_position_id: int) -> tuple[int, float]:
        """计算 NoEnergizer 策略中的 bean 状态转移。

        输入语义：state 是父节点状态，next_position_id 是下一步位置。
        输出语义：返回更新后的 bean bitmask 和即时奖励。
        关键约束：目标配置下该策略 reward 系数为 0，但对象移除状态仍会影响后续路径。
        """

        exact_reward = 0.0
        bean_mask = state.bean_mask
        next_bit = 1 << next_position_id
        if bean_mask & next_bit:
            exact_reward += self.reward_amount[1]
            bean_mask &= ~next_bit
        return bean_mask, exact_reward

    def _apply_no_energizer_risk(
        self,
        state: StrategyState,
        next_position_id: int,
        visited_before: bool,
    ) -> tuple[int, tuple[Any, ...], float, bool]:
        """计算 NoEnergizer 策略的 energizer 惩罚和 ghost 终止。

        输入语义：state 是父节点状态，next_position_id 是下一步位置，visited_before 表示路径是否到过该位置。
        输出语义：返回风险更新后的 energizer bitmask、ghost 状态、即时风险和终止标记。
        关键约束：重复访问路径中已有位置时不计算风险，也不更新风险侧状态。
        """

        if visited_before:
            return state.energizer_mask, state.ghost_status, 0.0, False

        exact_risk = 0.0
        energizer_mask = state.energizer_mask
        ghost_status = state.ghost_status
        termination_status = state.ghost_status
        next_bit = 1 << next_position_id
        if energizer_mask & next_bit:
            exact_risk -= self.reward_amount[2]
            energizer_mask &= ~next_bit
            ghost_status = _scare_active_ghosts(ghost_status)

        # 终止判断读取进入该位置前的 ghost 状态；吃 energizer 后的状态只传给后续节点。
        terminated = self._two_ghost_termination_only(termination_status, next_position_id)
        return energizer_mask, ghost_status, exact_risk, terminated

    def _two_ghost_risk(
        self,
        ghost_status: tuple[Any, ...],
        next_position_id: int,
        visited_before: bool,
    ) -> tuple[tuple[Any, ...], float, bool]:
        """计算前两只 ghost 的碰撞风险。

        输入语义：ghost_status 至少包含前两只 ghost 状态，next_position_id 是下一步位置。
        输出语义：返回原 ghost 状态、即时风险和是否终止路径。
        关键约束：重复访问路径中已有位置时跳过风险计算。
        """

        if visited_before:
            return ghost_status, 0.0, False

        ifscared1 = _status_value(ghost_status[0])
        ifscared2 = _status_value(ghost_status[1])
        ghost1_id = self.frame_state.ghost_ids[0]
        ghost2_id = self.frame_state.ghost_ids[1]
        if ifscared1 <= 2 or ifscared2 <= 2:
            if ifscared1 == 3:
                if next_position_id == ghost2_id:
                    return ghost_status, 0.0, True
            elif ifscared2 == 3:
                if next_position_id == ghost1_id:
                    return ghost_status, -self.reward_amount[9], True
            elif next_position_id == ghost1_id or next_position_id == ghost2_id:
                exact_risk = 0.0 if next_position_id == ghost2_id else -self.reward_amount[9]
                return ghost_status, exact_risk, True
        return ghost_status, 0.0, False

    def _single_ghost_evade_risk(
        self,
        ghost_status: tuple[Any, ...],
        ghost_id: int,
        next_position_id: int,
        visited_before: bool,
    ) -> tuple[float, bool]:
        """计算单只 ghost 的 Evade 风险。

        输入语义：ghost_status 只包含被选择 ghost 的状态，ghost_id 是其位置 id。
        输出语义：返回即时风险和是否终止路径。
        关键约束：只有状态为 1 或 2 的 ghost 与 Pacman 相遇时产生 evade 风险。
        """

        if visited_before:
            return 0.0, False
        ifscared = _status_value(ghost_status[0])
        if ifscared in (1, 2) and next_position_id == ghost_id:
            return -self.reward_amount[9], True
        return 0.0, False

    def _two_ghost_termination_only(self, ghost_status: tuple[Any, ...], next_position_id: int) -> bool:
        """判断 NoEnergizer 策略是否因前两只 ghost 碰撞而终止。

        输入语义：ghost_status 是风险更新后的 ghost 状态，next_position_id 是下一步位置。
        输出语义：需要终止时返回 True。
        关键约束：该策略的 ghost 碰撞只影响路径终止，不额外改变即时风险值。
        """

        ifscared1 = _status_value(ghost_status[0])
        ifscared2 = _status_value(ghost_status[1])
        ghost1_id = self.frame_state.ghost_ids[0]
        ghost2_id = self.frame_state.ghost_ids[1]
        if ifscared1 <= 2 or ifscared2 <= 2:
            if ifscared1 == 3:
                return next_position_id == ghost2_id
            if ifscared2 == 3:
                return next_position_id == ghost1_id
            return next_position_id == ghost1_id or next_position_id == ghost2_id
        return False

    def _add_leaf(self, strategy_index: int, root_direction_id: int, utility: float) -> None:
        """把一个策略叶节点计入对应初始方向统计。

        输入语义：strategy_index 指定策略，root_direction_id 指定第一步方向，utility 是路径累计值。
        输出语义：self.leaf_stats 中 utility sum 和 leaf count 增加。
        关键约束：root_direction_id 必须来自四方向之一，根节点本身不会被计为叶节点。
        """

        self.leaf_stats[strategy_index][root_direction_id][0] += utility
        self.leaf_stats[strategy_index][root_direction_id][1] += 1

    def _build_q_values(self) -> dict[str, np.ndarray]:
        """根据叶节点统计生成路径型策略 Q 向量。

        输入语义：self.leaf_stats 已经在共享路径搜索中填充。
        输出语义：返回 Q 列名到四方向 Q 向量的映射。
        关键约束：无叶节点的方向保持 0；路径型策略 Q 使用 float64。
        """

        q_values: dict[str, np.ndarray] = {}
        for strategy_index, column in enumerate(PATH_Q_COLUMNS):
            q_list = [0.0, 0.0, 0.0, 0.0]
            available_indices: list[int] = []
            for direction_id, (utility_sum, leaf_count) in enumerate(self.leaf_stats[strategy_index]):
                if leaf_count > 0:
                    q_list[direction_id] = utility_sum / leaf_count
                    available_indices.append(direction_id)
            q_array = np.array(q_list)
            q_values[column] = _apply_randomness_and_laziness(
                q_array,
                available_indices,
                self.frame_state.last_direction_id,
                self.config.randomness_coeff,
                self.strategy_laziness_coeffs[strategy_index],
            )
        return q_values

    def _build_trace(self, q_values: dict[str, np.ndarray]) -> dict[str, Any]:
        """生成过程一致性验证需要的 leaf 统计。

        输入语义：q_values 是已计算好的策略 Q 值。
        输出语义：返回每个路径型策略的 leaf count、utility sum 和 Q 值。
        关键约束：该结构只用于验证，不参与正式输出。
        """

        trace: dict[str, Any] = {}
        for strategy_index, column in enumerate(PATH_Q_COLUMNS):
            direction_stats = {}
            for direction_id, direction in enumerate(DIRECTIONS):
                utility_sum, leaf_count = self.leaf_stats[strategy_index][direction_id]
                if leaf_count > 0:
                    direction_stats[direction] = {
                        "utility_sum": float(utility_sum),
                        "leaf_count": int(leaf_count),
                    }
            trace[column] = {
                "leaf_stats": direction_stats,
                "q": np.asarray(q_values[column]).tolist(),
            }
        return trace


def estimate_all_q_values(
    compiled_map: CompiledMapData,
    frame_state: CompiledFrameState,
    config: UtilityConfig,
) -> dict[str, np.ndarray]:
    """估计单帧的全部 7 个 hierarchical utility Q 值。

    输入语义：compiled_map 是地图快表，frame_state 是单帧快表，config 是策略参数。
    输出语义：返回 Q 列名到四方向 Q 向量的字典。
    关键约束：Global 单独使用区域 bitmask，路径型策略由共享路径引擎一次完成。
    """

    q_values = {GLOBAL_Q_COLUMN: estimate_global_q(compiled_map, frame_state, config)}
    q_values.update(SharedPathUtilityEngine(compiled_map, frame_state, config).estimate())
    return q_values


def trace_all_q_values(
    compiled_map: CompiledMapData,
    frame_state: CompiledFrameState,
    config: UtilityConfig,
) -> dict[str, Any]:
    """估计单帧 Q 值并返回过程验证 trace。

    输入语义：compiled_map/frame_state/config 与正式估计相同。
    输出语义：返回 Global Q 和路径策略 leaf 统计。
    关键约束：trace 只记录关键过程指标，不改变正式估计逻辑。
    """

    global_q = estimate_global_q(compiled_map, frame_state, config)
    path_q_values, path_trace = SharedPathUtilityEngine(compiled_map, frame_state, config).estimate(return_trace=True)
    return {
        GLOBAL_Q_COLUMN: {"q": np.asarray(global_q).tolist()},
        **path_trace,
        "_q_values": {GLOBAL_Q_COLUMN: global_q, **path_q_values},
    }


def estimate_global_q(
    compiled_map: CompiledMapData,
    frame_state: CompiledFrameState,
    config: UtilityConfig,
) -> np.ndarray:
    """快速计算 Global 策略 Q 向量。

    输入语义：compiled_map 预先保存每个位置四方向区域 mask，frame_state 保存当前 bean bitmask。
    输出语义：返回 dtype 为 float32 的四方向 Q 向量。
    关键约束：只有当前位置可走方向会写入 bean 数量，其它方向保持 0。
    """

    neighbors = compiled_map.neighbor_ids[frame_state.pacman_id]
    available_indices = [direction_id for direction_id, next_id in enumerate(neighbors) if next_id != -1]
    if len(available_indices) == 0 or len(available_indices) == 1:
        position = compiled_map.id_to_position[frame_state.pacman_id]
        raise ValueError(f"位置 {position} 的可走方向数量为 {len(available_indices)}。")

    q_array = np.zeros(4, dtype=np.float32)
    region_masks = compiled_map.global_region_masks[frame_state.pacman_id]
    for direction_id in available_indices:
        q_array[direction_id] = float((frame_state.bean_mask & region_masks[direction_id]).bit_count())
    return _apply_randomness_and_laziness(
        q_array,
        available_indices,
        frame_state.last_direction_id,
        config.randomness_coeff,
        config.laziness_coeff,
    )


def scale_of_number(num: float) -> float | int:
    """计算随机扰动使用的数量级。

    输入语义：num 是 Q 向量最大绝对值。
    输出语义：返回与数值量级对应的 10 的幂。
    关键约束：默认随机系数为 0；该函数只在显式启用随机扰动时进入热点路径。
    """

    if num >= 1:
        order = len(str(num).split(".")[0])
        return 10 ** (order - 1)
    if num == 0:
        return 1
    order = str(num).split(".")[1]
    zero_count = 0
    for char in order:
        if char == "0":
            zero_count += 1
        else:
            break
    return 10 ** (-zero_count - 1)


def _scare_active_ghosts(ghost_status: tuple[Any, ...]) -> tuple[Any, ...]:
    """把非死亡 ghost 状态更新为 scared。

    输入语义：ghost_status 是当前策略携带的 ghost 状态 tuple。
    输出语义：返回更新后的状态 tuple。
    关键约束：状态值等于 3 表示死亡，保持为 3；其它状态统一更新为 4。
    """

    return tuple(4 if item != 3 else 3 for item in ghost_status)


def _status_value(value: Any) -> int:
    """把 ghost 状态规范为风险规则使用的整数。

    输入语义：value 通常是 Python/NumPy 整数，也允许有限的整值 float；旧输入中的
    ``None/NaN`` 仍视为缺失状态。
    输出语义：缺失状态返回 0，其余合法数值返回对应整数。
    关键约束：只有真正的缺失值可以映射为 0；不能再把所有 float 一概当作缺失。
    非整数、无穷和非数值类型直接报错，避免静默改变 ghost 风险语义。
    """

    if value is None:
        return 0
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"ghost 状态不能是布尔值：{value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if np.isnan(numeric_value):
            return 0
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"ghost 状态必须是有限整数：{value!r}")
        return int(numeric_value)
    raise TypeError(f"无法解析 ghost 状态：{value!r}")


def _apply_randomness_and_laziness(
    q_array: np.ndarray,
    available_indices: list[int],
    last_direction_id: int | None,
    randomness_coeff: float,
    laziness_coeff: float,
) -> np.ndarray:
    """按配置向 Q 向量加入随机扰动和上一方向惰性项。

    输入语义：q_array 是已聚合的 Q 值，available_indices 是当前位置可走方向。
    输出语义：返回可能被原地更新后的 Q 向量。
    关键约束：两个系数同时为 0 时直接返回，避免默认实验路径中的额外开销。
    """

    if randomness_coeff == 0 and laziness_coeff == 0:
        return q_array
    q_scale = scale_of_number(np.max(np.abs(q_array)))
    if available_indices and randomness_coeff != 0:
        randomness = np.random.uniform(low=0, high=0.1, size=len(available_indices)) * q_scale
        q_array[available_indices] += randomness_coeff * randomness
    if last_direction_id is not None and last_direction_id in available_indices and laziness_coeff != 0:
        q_array[last_direction_id] += laziness_coeff * q_scale
    return q_array
