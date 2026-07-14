"""基于事件 Context 推断 Social Pacman 潜在策略后验。

本模块在玩家级事件 context 内分别选择解释动作最好的 Global cluster、Energizer 目标
和 Approach ghost 目标，随后统一归一化七种策略的 raw Q，通过 softmax 得到动作概率。
context 内信息覆盖率不足的策略先被
排除，posterior 和文件级 temperature 只使用覆盖率合格的行为策略。策略在某行的
合法方向 Q 全相等时，使用独立于 beta 的固定无信息惩罚，避免无信息均匀预测比
明确错误获得过多优势。合法方向均匀分布的 Null 仍只作为诊断基线。

输入和输出均为 P1/P2 时间对齐的 joint-state DataFrame；本阶段直接保存策略 posterior，
不生成 GA weight 字段。
"""

from __future__ import annotations

import copy
import math
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from LoPS.context_segmentation import (
    ContextSegmentationConfig,
    append_player_event_columns,
    apply_best_global_candidates,
    build_event_context_segments,
    parse_global_cluster_matrix,
    parse_global_cluster_meta,
    parse_position_or_none,
    positive_prediction_indices,
    q_like_zero,
)


DIRECTION_NAMES: tuple[str, ...] = ("left", "right", "up", "down")
DIRECTION_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(DIRECTION_NAMES)}
PLAYER_PREFIXES: tuple[str, ...] = ("p1", "p2")
DEFAULT_AGENTS: tuple[str, ...] = (
    "global",
    "local",
    "evade_blinky",
    "evade_clyde",
    "approach",
    "energizer",
    "no_energizer",
)
STRATEGY_NUMBER: dict[str, int] = {
    "global": 0,
    "local": 1,
    "evade_blinky": 2,
    "evade_clyde": 3,
    "approach": 6,
    "energizer": 7,
    "no_energizer": 8,
    "vague": 9,
    "stay": 10,
}


@dataclass(frozen=True)
class ContextStrategyPosteriorConfig:
    """保存 06 context 后验拟合参数。

    输入语义：调用方可设置 context 边界参数、beta 搜索范围、交叉验证折数和文件级
    随机种子。
    输出语义：该不可变配置会贯穿 DataFrame、单文件和目录级处理。
    关键约束：``agents`` 顺序同时定义 likelihood/posterior 数组顺序，不可在处理中重排。
    """

    agents: tuple[str, ...] = DEFAULT_AGENTS
    stay_length: int = 4
    bean_event_suppression_window: int = 3
    ghost_stay_suppression_window: int = 5
    beta_min: float = 0.05
    beta_max: float = 20.0
    beta_grid_size: int = 81
    cv_folds: int = 5
    posterior_threshold: float = 0.70
    min_information_coverage: float = 0.50
    information_epsilon: float = 1e-12
    no_information_penalty: float = 2.0
    random_seed: int = 20260610


@dataclass(frozen=True)
class ContextObservation:
    """保存一个 player-context 参与概率拟合的数据。

    输入语义：``q_values`` 形状为 ``有效动作数 × 策略数 × 4``，已完成合法方向
    Min-Max；``action_indices`` 与第一维逐行对齐。
    输出语义：对象可以直接用于任意 beta 下的 context likelihood 计算。
    关键约束：context 使用全文件半开区间；无有效动作段的数组第一维为 0，并由
    ``is_stay`` 标记，不进入 beta loss。
    """

    player: str
    trial_name: str
    context: tuple[int, int]
    is_stay: bool
    row_indices: np.ndarray
    action_indices: np.ndarray
    q_values: np.ndarray
    null_log_likelihood: float
    strategy_information_coverage: np.ndarray | None = None
    strategy_eligible: np.ndarray | None = None

    @property
    def valid_action_count(self) -> int:
        """返回当前 context 中实际进入 likelihood 的动作数量。"""

        return int(self.action_indices.size)

    def resolved_information_coverage(self) -> np.ndarray:
        """返回当前 context 每个策略具有方向信息的有效动作比例。

        输入语义：正式流程会在构造 observation 时保存 coverage；测试或外部调用若未
        显式提供，则按归一化 Q 中合法方向是否存在差异即时计算。
        输出语义：返回长度等于策略数、范围位于 ``[0, 1]`` 的数组。
        关键约束：无有效动作时 coverage 全为 0；非法方向 ``-inf`` 不参与比较。
        """

        if self.strategy_information_coverage is not None:
            return np.asarray(self.strategy_information_coverage, dtype=float)
        return calculate_strategy_information_coverage(self.q_values)

    def resolved_strategy_eligible(self) -> np.ndarray:
        """返回当前 context 中允许进入 posterior 的策略布尔掩码。

        输入语义：正式流程按配置阈值预先保存 eligibility；未保存时使用默认 0.5
        coverage 阈值，保证手工构造 observation 仍遵循正式统计语义。
        输出语义：返回长度等于策略数的 bool 数组。
        关键约束：Null 模型不在该数组中，它始终作为独立候选参与比较。
        """

        if self.strategy_eligible is not None:
            return np.asarray(self.strategy_eligible, dtype=bool)
        return self.resolved_information_coverage() >= 0.50


@dataclass(frozen=True)
class BetaFitResult:
    """保存一次一维 beta 优化结果。

    输入语义：``beta`` 是 temperature，``loss`` 是指定 contexts 的最小边际 NLL。
    输出语义：用于完整文件模型、分玩家模型及交叉验证结果汇总。
    关键约束：beta 始终位于配置给出的闭区间内。
    """

    beta: float
    loss: float


@dataclass(frozen=True)
class ObservationBatch:
    """保存一组 contexts 的批量 likelihood 数组。

    输入语义：``q_values/action_indices`` 拼接所有有效动作，``context_starts`` 标记
    每个 context 在动作轴上的起点。
    输出语义：同一 beta 下可一次完成全部动作 softmax，再按 context 聚合。
    关键约束：批量化只改变计算组织方式，不改变先逐行动作 likelihood、再在 context
    内求和、最后对潜在策略边际化的统计公式。
    """

    q_values: np.ndarray
    action_indices: np.ndarray
    context_starts: np.ndarray
    context_count: int
    strategy_eligible: np.ndarray


@dataclass
class PreparedPlayerData:
    """保存单个玩家的临时视图和 context 观测。

    输入语义：``view`` 已完成死亡动作屏蔽与 best Global 选择；``observations`` 与
    玩家事件 context 一一对应。
    输出语义：供文件级 beta 拟合和最终逐行写回共同使用。
    关键约束：view 只在 06 内部使用，不会覆盖 05 输入 DataFrame 的原始字段。
    """

    player: str
    view: pd.DataFrame
    observations: list[ContextObservation]


def validate_config(config: ContextStrategyPosteriorConfig) -> None:
    """验证 06 配置参数。

    输入语义：config 是调用方构造的后验配置。
    输出语义：参数合法时无返回；不合法时抛出 ValueError。
    关键约束：beta 区间必须严格为正，posterior 阈值必须位于概率范围内。
    """

    if config.beta_min <= 0 or config.beta_max <= config.beta_min:
        raise ValueError("beta 搜索范围必须满足 0 < beta_min < beta_max。")
    if config.beta_grid_size < 3:
        raise ValueError("beta_grid_size 至少为 3。")
    if config.cv_folds < 2:
        raise ValueError("cv_folds 至少为 2；trial 不足时流程会自动跳过或减少折数。")
    if not 0 <= config.posterior_threshold <= 1:
        raise ValueError("posterior_threshold 必须位于 [0, 1]。")
    if not 0 <= config.min_information_coverage <= 1:
        raise ValueError("min_information_coverage 必须位于 [0, 1]。")
    if config.information_epsilon < 0:
        raise ValueError("information_epsilon 不能小于 0。")
    if config.no_information_penalty < 0:
        raise ValueError("no_information_penalty 不能小于 0。")
    if config.bean_event_suppression_window < 0:
        raise ValueError("bean_event_suppression_window 不能小于 0。")
    if config.ghost_stay_suppression_window < 0:
        raise ValueError("ghost_stay_suppression_window 不能小于 0。")
    missing_codes = sorted(set(config.agents) - set(STRATEGY_NUMBER))
    if missing_codes:
        raise ValueError(f"策略缺少数字编码：{missing_codes}")


def discover_posterior_players(data: pd.DataFrame, config: ContextStrategyPosteriorConfig) -> list[str]:
    """识别拥有完整 06 输入字段的玩家。

    输入语义：data 是 05 cluster-global utility 表。
    输出语义：返回 ``p1``、``p2`` 中字段完整的玩家列表；单人文件自然只返回 p1。
    关键约束：某玩家只要出现部分信号字段就必须完整，否则立即报错，避免静默跳过坏数据。
    """

    players: list[str] = []
    for player in PLAYER_PREFIXES:
        signal_columns = {f"{player}_action_dir", f"{player}_available_dir"}
        if signal_columns.isdisjoint(data.columns):
            continue
        required = {
            "row_id",
            "DayTrial",
            f"{player}_action_dir",
            f"{player}_available_dir",
            f"{player}_global_Q_norm",
            f"{player}_global_utility_k",
            f"{player}_global_utility_k_norm",
            f"{player}_global_utility_k_meta",
            f"{player}_energizer_utility_k",
            f"{player}_energizer_utility_k_norm",
            f"{player}_energizer_utility_k_meta",
            f"{player}_approach_utility_k",
            f"{player}_approach_utility_k_norm",
            f"{player}_approach_utility_k_meta",
        }
        required.update(f"{player}_{agent}_Q" for agent in config.agents)
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"{player} 06 输入字段不完整，缺少：{missing}")
        players.append(player)
    if not players:
        raise ValueError("没有找到可处理玩家，至少需要 p1_action_dir/p1_available_dir 和 raw Q 字段。")
    return players


def normalize_legal_q(values: Any) -> np.ndarray:
    """只在合法有限方向上执行逐行 Min-Max 归一化。

    输入语义：values 是长度为 4 的 raw Q；不可走方向应为 ``-inf``。
    输出语义：有限方向落在 ``[0, 1]``，不可走方向保持 ``-inf``。
    关键约束：所有合法方向相等时统一置 0，表示该策略在当前 tile 没有方向信息。
    """

    source = np.asarray(values, dtype=float)
    if source.ndim != 1 or source.shape[0] != len(DIRECTION_NAMES):
        raise ValueError(f"Q 必须是长度为 4 的一维数组，实际 shape={source.shape}")
    if np.any(np.isposinf(source)) or np.any(np.isnan(source)):
        raise ValueError(f"Q 中不允许出现 +inf 或 NaN：{source.tolist()}")

    result = source.copy()
    legal_mask = np.isfinite(source)
    if not np.any(legal_mask):
        raise ValueError("Q 的四个方向全部非法，无法计算动作概率。")
    legal_values = source[legal_mask]
    minimum = float(np.min(legal_values))
    maximum = float(np.max(legal_values))
    if maximum == minimum:
        result[legal_mask] = 0.0
    else:
        result[legal_mask] = (legal_values - minimum) / (maximum - minimum)
    return result


def calculate_strategy_information_coverage(
    q_values: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """计算一个 context 中各策略的方向信息覆盖率。

    输入语义：``q_values`` 形状为 ``有效动作数 × 策略数 × 4``，每个 Q 已完成合法
    方向 Min-Max；epsilon 用于忽略浮点噪声。
    输出语义：返回每个策略在多少比例的有效动作行上能区分至少两个合法方向。
    关键约束：合法方向全部相等表示策略在该行无信息；该行继续保留在 coverage 分母
    中，因此无信息等价于诊断准确率中的一次未命中，而不是被静默删除。
    """

    values = np.asarray(q_values, dtype=float)
    if values.ndim != 3 or values.shape[2] != len(DIRECTION_NAMES):
        raise ValueError(f"context Q 必须为 n×k×4，实际 shape={values.shape}")
    strategy_count = values.shape[1]
    if values.shape[0] == 0:
        return np.zeros(strategy_count, dtype=float)

    finite_mask = np.isfinite(values)
    finite_min = np.min(np.where(finite_mask, values, np.inf), axis=2)
    finite_max = np.max(np.where(finite_mask, values, -np.inf), axis=2)
    informative = (finite_max - finite_min) > float(epsilon)
    return np.mean(informative, axis=0, dtype=float)


def prepare_player_view(
    data: pd.DataFrame,
    player: str,
    config: ContextStrategyPosteriorConfig,
) -> pd.DataFrame:
    """构造 06 context 划分和 best Global 选择所需的单玩家视图。

    输入语义：data 已包含玩家私有事件列；player 指定当前玩家。
    输出语义：返回保留全部前缀字段并新增通用 ``action_dir/available_dir/global_Q`` 的副本。
    关键约束：死亡、不可用和缺失动作行会在目标评分前被屏蔽。
    """

    view = data.copy(deep=True).reset_index(drop=True)
    view["action_dir"] = view[f"{player}_action_dir"].apply(
        lambda value: value if isinstance(value, str) else np.nan
    )
    view["available_dir"] = view[f"{player}_available_dir"].astype(bool)
    alive_column = f"{player}_alive"
    if alive_column in view.columns:
        dead_mask = ~view[alive_column].astype(bool)
        view.loc[dead_mask, "action_dir"] = np.nan
        view.loc[dead_mask, "available_dir"] = False
    invalid_mask = ~view["available_dir"]
    view.loc[invalid_mask, "action_dir"] = np.nan

    # apply_best_global_candidates 同时依赖无前缀和玩家前缀 Global 字段；这里只构造
    # 临时别名，原始 05 DataFrame 不会被修改。
    view["global_Q"] = copy.deepcopy(view[f"{player}_global_Q"])
    view["global_Q_norm"] = copy.deepcopy(view[f"{player}_global_Q_norm"])
    return view


def energizer_target_position(meta: dict[str, Any]) -> tuple[int, int] | None:
    """从05 Energizer 候选 meta 中读取稳定目标坐标。

    输入语义：meta 对应候选矩阵中的一行。
    输出语义：返回 ``target_position``；字段缺失或非法时返回 None。
    关键约束：目标坐标是跨行稳定身份，不使用每行候选列表下标或 target_id 猜测。
    """

    return parse_position_or_none(meta.get("target_position"))


def match_energizer_target_index(
    row_meta: list[dict[str, Any]],
    target_position: tuple[int, int],
) -> int | None:
    """在某一行候选列表中匹配同一个 Energizer 目标。

    输入语义：row_meta 与该行候选矩阵逐行对齐；target_position 来自 context 起点。
    输出语义：找到目标时返回矩阵行号，否则返回 None。
    关键约束：目标可能被玩家或队友在 context 中途吃掉；消失后返回 None，让该行按
    无信息处理，而不是错误匹配到剩余列表中相同下标的另一个 energizer。
    """

    for index, meta in enumerate(row_meta):
        if energizer_target_position(meta) == target_position:
            return index
    return None


def score_context_energizer_candidate(
    data: pd.DataFrame,
    context: tuple[int, int],
    player: str,
    start_meta: dict[str, Any],
) -> dict[str, Any]:
    """计算一个明确 Energizer 目标对整个 context 动作的解释能力。

    输入语义：data 是单玩家临时视图；context 是半开区间；start_meta 描述 context
    起点仍存在的一个 energizer。
    输出语义：返回概率准确率、集合准确率、起点距离和稳定目标坐标。
    关键约束：所有真实动作行都进入分母；目标缺失或候选无正向推进时贡献0。并列最大
    方向包含真实动作时按 ``1/并列数`` 贡献概率准确率，与 Global 选择规则一致。
    """

    target_position = energizer_target_position(start_meta)
    if target_position is None:
        return {
            "target_position": None,
            "start_distance": float("inf"),
            "valid_actions": 0,
            "prob_accuracy": 0.0,
            "set_accuracy": 0.0,
            "meta": start_meta,
        }

    start, end = context
    valid_actions = 0
    probability_credit = 0.0
    set_hits = 0
    matrix_column = f"{player}_energizer_utility_k_norm"
    meta_column = f"{player}_energizer_utility_k_meta"
    for row_index in range(start, end):
        action = data.at[row_index, "action_dir"]
        if not isinstance(action, str):
            continue
        valid_actions += 1
        row_meta = parse_global_cluster_meta(data.at[row_index, meta_column])
        matched_index = match_energizer_target_index(row_meta, target_position)
        if matched_index is None:
            continue
        matrix = parse_global_cluster_matrix(data.at[row_index, matrix_column])
        if matched_index >= matrix.shape[0]:
            continue
        prediction_indices = positive_prediction_indices(matrix[matched_index])
        if not prediction_indices:
            continue
        true_index = DIRECTION_TO_INDEX[action]
        if true_index in prediction_indices:
            set_hits += 1
            probability_credit += 1.0 / len(prediction_indices)

    return {
        "target_position": target_position,
        "start_distance": float(start_meta.get("min_distance", np.inf)),
        "valid_actions": valid_actions,
        "prob_accuracy": probability_credit / valid_actions if valid_actions else 0.0,
        "set_accuracy": set_hits / valid_actions if valid_actions else 0.0,
        "meta": start_meta,
    }


def choose_best_energizer_candidate(
    data: pd.DataFrame,
    context: tuple[int, int],
    player: str,
) -> dict[str, Any] | None:
    """为一个 player-context 选择解释真实动作最好的 Energizer 目标。

    输入语义：data 已包含05候选字段；context 是半开区间；player 指定玩家。
    输出语义：返回最佳目标评分；context 起点没有 energizer 时返回 None。
    关键约束：破平依次使用概率准确率、集合准确率、较近起点距离和目标坐标。目标
    选择只依赖 utility 与动作，不读取最终是否吃到；事件结果留给 07 消除策略歧义。
    """

    start, _ = context
    meta_column = f"{player}_energizer_utility_k_meta"
    start_meta_values = parse_global_cluster_meta(data.at[start, meta_column])
    scored = [
        score_context_energizer_candidate(data, context, player, meta)
        for meta in start_meta_values
        if energizer_target_position(meta) is not None
    ]
    if not scored:
        return None
    return max(
        scored,
        key=lambda item: (
            item["prob_accuracy"],
            item["set_accuracy"],
            -item["start_distance"],
            -item["target_position"][0],
            -item["target_position"][1],
        ),
    )


def apply_best_energizer_candidates(
    data: pd.DataFrame,
    contexts: list[tuple[int, int]],
    player: str,
) -> pd.DataFrame:
    """在 context 级选择 best Energizer，并写入06临时正式 Q。

    输入语义：data 已完成 best Global 选择；contexts 是当前玩家的完整 context 列表。
    输出语义：返回新视图，其中 ``<player>_energizer_Q`` 已替换为每段最佳目标 Q，
    并新增 selected Q、目标坐标、准确率和 meta 解释列。
    关键约束：05原始 DataFrame 不被修改。没有目标或目标中途消失的行保留合法方向
    mask，但有限方向全部置0，明确表示目标导向 Energizer 在该行无信息。
    """

    result = data.copy(deep=True)
    selected_column = "selected_energizer_Q"
    best_columns = {
        "best_energizer_target_position": np.nan,
        "best_energizer_target_prob_accuracy": np.nan,
        "best_energizer_target_set_accuracy": np.nan,
        "best_energizer_target_meta": np.nan,
    }
    result[selected_column] = pd.Series([np.nan] * len(result), index=result.index, dtype=object)
    for column, default in best_columns.items():
        result[column] = pd.Series([default] * len(result), index=result.index, dtype=object)

    raw_column = f"{player}_energizer_utility_k"
    meta_column = f"{player}_energizer_utility_k_meta"
    formal_column = f"{player}_energizer_Q"
    for context in contexts:
        best = choose_best_energizer_candidate(result, context, player)
        start, end = context
        target_position = best["target_position"] if best is not None else None
        for row_index in range(start, end):
            raw_q = q_like_zero(result.at[row_index, formal_column])
            if target_position is not None:
                row_meta = parse_global_cluster_meta(result.at[row_index, meta_column])
                matched_index = match_energizer_target_index(row_meta, target_position)
                if matched_index is not None:
                    raw_matrix = parse_global_cluster_matrix(result.at[row_index, raw_column])
                    if matched_index < raw_matrix.shape[0]:
                        raw_q = raw_matrix[matched_index].tolist()

            result.at[row_index, formal_column] = raw_q
            result.at[row_index, selected_column] = raw_q
            if best is not None:
                result.at[row_index, "best_energizer_target_position"] = target_position
                result.at[row_index, "best_energizer_target_prob_accuracy"] = best["prob_accuracy"]
                result.at[row_index, "best_energizer_target_set_accuracy"] = best["set_accuracy"]
                result.at[row_index, "best_energizer_target_meta"] = best["meta"]
    return result


def approach_target_id(meta: dict[str, Any]) -> str | None:
    """从05 Approach 候选 meta 中读取稳定 ghost 身份。

    输入语义：meta 对应某行候选矩阵中的一行。
    输出语义：合法时返回 ``ghost1`` 或 ``ghost2``，否则返回 None。
    关键约束：位置会逐帧移动，不能拿 ``target_position`` 作为跨行匹配键；ghost 身份
    才是同一 context 中保持稳定的目标标识。
    """

    value = meta.get("target_id")
    return value if value in {"ghost1", "ghost2"} else None


def match_approach_target_index(
    row_meta: list[dict[str, Any]],
    target_id: str,
) -> int | None:
    """在某一行候选列表中匹配同一只目标 ghost。

    输入语义：row_meta 与候选矩阵逐行对齐，target_id 来自 context 起点。
    输出语义：目标仍是非死亡状态时返回矩阵行号，否则返回 None。
    关键约束：目标位置变化不影响匹配；若目标已死亡或缺失，05不会生成该候选，
    当前行应按无信息处理，不能改用同一下标的另一只 ghost。
    """

    for index, meta in enumerate(row_meta):
        if approach_target_id(meta) == target_id:
            return index
    return None


def score_context_approach_candidate(
    data: pd.DataFrame,
    context: tuple[int, int],
    player: str,
    start_meta: dict[str, Any],
) -> dict[str, Any]:
    """计算一只明确 ghost 目标对整个 context 动作的解释能力。

    输入语义：data 是单玩家临时视图；context 是半开区间；start_meta 描述 context
    起点的一只非死亡 ghost。
    输出语义：返回概率准确率、集合准确率、起点距离和稳定 ghost 身份。
    关键约束：所有真实动作行都进入分母；候选缺失或无正向预测时贡献0。目标选择只
    使用逐帧 utility 和动作，不读取 context 结束时是否真的吃到 ghost。
    """

    target_id = approach_target_id(start_meta)
    if target_id is None:
        return {
            "target_id": None,
            "target_position": None,
            "start_distance": float("inf"),
            "valid_actions": 0,
            "prob_accuracy": 0.0,
            "set_accuracy": 0.0,
            "meta": start_meta,
        }

    start, end = context
    valid_actions = 0
    probability_credit = 0.0
    set_hits = 0
    matrix_column = f"{player}_approach_utility_k_norm"
    meta_column = f"{player}_approach_utility_k_meta"
    for row_index in range(start, end):
        action = data.at[row_index, "action_dir"]
        if not isinstance(action, str):
            continue
        valid_actions += 1
        row_meta = parse_global_cluster_meta(data.at[row_index, meta_column])
        matched_index = match_approach_target_index(row_meta, target_id)
        if matched_index is None:
            continue
        matrix = parse_global_cluster_matrix(data.at[row_index, matrix_column])
        if matched_index >= matrix.shape[0]:
            continue
        prediction_indices = positive_prediction_indices(matrix[matched_index])
        if not prediction_indices:
            continue
        true_index = DIRECTION_TO_INDEX[action]
        if true_index in prediction_indices:
            set_hits += 1
            probability_credit += 1.0 / len(prediction_indices)

    return {
        "target_id": target_id,
        "target_position": parse_position_or_none(start_meta.get("target_position")),
        "start_distance": float(start_meta.get("min_distance", np.inf)),
        "valid_actions": valid_actions,
        "prob_accuracy": probability_credit / valid_actions if valid_actions else 0.0,
        "set_accuracy": set_hits / valid_actions if valid_actions else 0.0,
        "meta": start_meta,
    }


def choose_best_approach_candidate(
    data: pd.DataFrame,
    context: tuple[int, int],
    player: str,
) -> dict[str, Any] | None:
    """为一个 player-context 选择解释动作最好的 Approach 目标。

    输入语义：data 已包含05候选字段；context 是半开区间；player 指定玩家。
    输出语义：返回最佳 ghost 目标评分；context 起点没有非死亡 ghost 时返回 None。
    关键约束：破平依次使用概率准确率、集合准确率、较近起点距离和较小 ghost 编号；
    Ghost1/Ghost2 只在候选内部竞争，选定后仍作为一个顶层 Approach 策略参与 posterior。
    """

    start, _ = context
    meta_column = f"{player}_approach_utility_k_meta"
    start_meta_values = parse_global_cluster_meta(data.at[start, meta_column])
    scored = [
        score_context_approach_candidate(data, context, player, meta)
        for meta in start_meta_values
        if approach_target_id(meta) is not None
    ]
    if not scored:
        return None
    return max(
        scored,
        key=lambda item: (
            item["prob_accuracy"],
            item["set_accuracy"],
            -item["start_distance"],
            -int(item["target_id"].removeprefix("ghost")),
        ),
    )


def apply_best_approach_candidates(
    data: pd.DataFrame,
    contexts: list[tuple[int, int]],
    player: str,
) -> pd.DataFrame:
    """在 context 级选择 best Approach，并写入06临时正式 Q。

    输入语义：data 已完成 Global/Energizer 目标选择；contexts 是当前玩家的完整分段。
    输出语义：返回新视图，其中 ``<player>_approach_Q`` 已替换为每段最佳 ghost 目标
    Q，并新增 selected Q、目标身份、起点位置、准确率和 meta 解释列。
    关键约束：05原始候选和旧混合 Approach Q 均不被覆盖；目标中途缺失时保留当前
    tile 的合法方向 mask，但有限方向全置0，明确表示该目标当行没有信息。
    """

    result = data.copy(deep=True)
    selected_column = "selected_approach_Q"
    best_columns = {
        "best_approach_target_id": np.nan,
        "best_approach_target_position": np.nan,
        "best_approach_target_prob_accuracy": np.nan,
        "best_approach_target_set_accuracy": np.nan,
        "best_approach_target_meta": np.nan,
    }
    result[selected_column] = pd.Series([np.nan] * len(result), index=result.index, dtype=object)
    for column, default in best_columns.items():
        result[column] = pd.Series([default] * len(result), index=result.index, dtype=object)

    raw_column = f"{player}_approach_utility_k"
    meta_column = f"{player}_approach_utility_k_meta"
    formal_column = f"{player}_approach_Q"
    for context in contexts:
        best = choose_best_approach_candidate(result, context, player)
        start, end = context
        target_id = best["target_id"] if best is not None else None
        for row_index in range(start, end):
            raw_q = q_like_zero(result.at[row_index, formal_column])
            if target_id is not None:
                row_meta = parse_global_cluster_meta(result.at[row_index, meta_column])
                matched_index = match_approach_target_index(row_meta, target_id)
                if matched_index is not None:
                    raw_matrix = parse_global_cluster_matrix(result.at[row_index, raw_column])
                    if matched_index < raw_matrix.shape[0]:
                        raw_q = raw_matrix[matched_index].tolist()

            result.at[row_index, formal_column] = raw_q
            result.at[row_index, selected_column] = raw_q
            if best is not None:
                result.at[row_index, "best_approach_target_id"] = target_id
                result.at[row_index, "best_approach_target_position"] = best["target_position"]
                result.at[row_index, "best_approach_target_prob_accuracy"] = best["prob_accuracy"]
                result.at[row_index, "best_approach_target_set_accuracy"] = best["set_accuracy"]
                result.at[row_index, "best_approach_target_meta"] = best["meta"]
    return result


def build_context_observation(
    view: pd.DataFrame,
    player: str,
    context: tuple[int, int],
    is_stay: bool,
    config: ContextStrategyPosteriorConfig,
) -> ContextObservation:
    """把一个玩家事件 context 转换成后验模型可直接计算的观测对象。

    输入语义：view 已选好 best Global；context 是全文件半开区间。
    输出语义：返回有效动作、统一归一化 Q 和 null likelihood。
    关键约束：七个策略在同一行必须具有完全相同的非法方向 mask；真实动作非法时
    该行被排除，而不是强行赋予极小概率。
    """

    start, end = context
    valid_rows: list[int] = []
    actions: list[int] = []
    q_rows: list[np.ndarray] = []
    null_log_likelihood = 0.0

    for row_index in range(start, end):
        action = view.at[row_index, "action_dir"]
        if not isinstance(action, str) or action not in DIRECTION_TO_INDEX:
            continue

        normalized_by_agent: list[np.ndarray] = []
        legal_mask: np.ndarray | None = None
        for agent in config.agents:
            raw_q = view.at[row_index, "global_Q"] if agent == "global" else view.at[row_index, f"{player}_{agent}_Q"]
            normalized = normalize_legal_q(raw_q)
            current_mask = np.isfinite(normalized)
            if legal_mask is None:
                legal_mask = current_mask
            elif not np.array_equal(legal_mask, current_mask):
                raise ValueError(
                    f"{player} row={row_index} 七策略非法方向 mask 不一致：agent={agent}"
                )
            normalized_by_agent.append(normalized)

        action_index = DIRECTION_TO_INDEX[action]
        assert legal_mask is not None
        if not legal_mask[action_index]:
            # 上游偶尔可能保留一个与地图不一致的动作；按方法定义不让它进入 likelihood。
            continue
        legal_count = int(np.sum(legal_mask))
        valid_rows.append(row_index)
        actions.append(action_index)
        q_rows.append(np.stack(normalized_by_agent, axis=0))
        null_log_likelihood -= math.log(legal_count)

    if q_rows:
        q_values = np.stack(q_rows, axis=0)
    else:
        q_values = np.empty((0, len(config.agents), len(DIRECTION_NAMES)), dtype=float)
    information_coverage = calculate_strategy_information_coverage(
        q_values,
        epsilon=config.information_epsilon,
    )
    strategy_eligible = information_coverage >= config.min_information_coverage
    trial_name = str(view.at[start, "DayTrial"])
    return ContextObservation(
        player=player,
        trial_name=trial_name,
        context=context,
        is_stay=bool(is_stay or not actions),
        row_indices=np.asarray(valid_rows, dtype=int),
        action_indices=np.asarray(actions, dtype=int),
        q_values=q_values,
        null_log_likelihood=float(null_log_likelihood),
        strategy_information_coverage=information_coverage,
        strategy_eligible=strategy_eligible,
    )


def prepare_player_data(
    data: pd.DataFrame,
    player: str,
    config: ContextStrategyPosteriorConfig,
) -> PreparedPlayerData:
    """为一个玩家完成 context 划分、best Global 选择和概率观测预计算。

    输入语义：data 是已经追加私有事件列的 joint-state 表。
    输出语义：返回临时玩家视图和按顺序排列的 ContextObservation。
    关键约束：Global、Energizer、Approach 都先在完整
    context 上选定目标，再构造统一的七策略概率观测。
    """

    view = prepare_player_view(data, player, config)
    context_config = ContextSegmentationConfig(
        stay_length=config.stay_length,
        bean_event_suppression_window=config.bean_event_suppression_window,
        ghost_stay_suppression_window=config.ghost_stay_suppression_window,
    )
    contexts, is_stay = build_event_context_segments(
        view,
        player,
        context_config,
    )
    selected_view = apply_best_global_candidates(view, contexts, player)
    selected_view = apply_best_energizer_candidates(selected_view, contexts, player)
    selected_view = apply_best_approach_candidates(selected_view, contexts, player)
    observations = [
        build_context_observation(selected_view, player, context, stay, config)
        for context, stay in zip(contexts, is_stay)
    ]
    validate_context_coverage(observations, len(data), player)
    return PreparedPlayerData(player=player, view=selected_view, observations=observations)


def validate_context_coverage(
    observations: list[ContextObservation],
    row_count: int,
    player: str,
) -> None:
    """验证一个玩家的 contexts 无重叠且完整覆盖所有 joint 行。

    输入语义：observations 是按玩家事件 context 生成的结果，row_count 是文件行数。
    输出语义：覆盖正确时无返回；缺口、重叠或越界时抛出 RuntimeError。
    关键约束：逐行写回依赖每行恰好属于一个 context，因此不能容忍部分结果。
    """

    coverage = np.zeros(row_count, dtype=int)
    for observation in observations:
        start, end = observation.context
        if start < 0 or end > row_count or end <= start:
            raise RuntimeError(f"{player} context 越界或为空：{observation.context}")
        coverage[start:end] += 1
    invalid = np.where(coverage != 1)[0]
    if invalid.size:
        raise RuntimeError(f"{player} context 未完整覆盖数据，异常行示例：{invalid[:10].tolist()}")


def action_log_probability_with_no_information_penalty(
    q_values: np.ndarray,
    action_indices: np.ndarray,
    beta: float,
    no_information_penalty: float = 2.0,
    information_epsilon: float = 1e-12,
) -> np.ndarray:
    """计算动作 log probability，并对无信息策略行施加固定额外惩罚。

    输入语义：q_values 形状为 ``动作数×策略数×4``，已经按合法方向 Min-Max；
    action_indices 与动作轴对齐；beta 只控制有信息行的 softmax；固定惩罚和 epsilon
    分别控制无信息损失与浮点判定。
    输出语义：返回 ``动作数×策略数`` 的真实动作 log probability。
    关键约束：合法方向 Q 极差不超过 epsilon 且合法方向数大于 1 时，使用
    ``-log(合法方向数)-no_information_penalty``；该值不依赖 beta。只有一个合法方向
    时玩家没有选择空间，保留普通 softmax 的 0 损失。墙方向 ``-inf`` 不参与极差。
    """

    values = np.asarray(q_values, dtype=float)
    actions = np.asarray(action_indices, dtype=int)
    if values.ndim != 3 or values.shape[2] != len(DIRECTION_NAMES):
        raise ValueError(f"批量 Q 必须为 n×k×4，实际 shape={values.shape}")
    if actions.ndim != 1 or actions.shape[0] != values.shape[0]:
        raise ValueError(f"动作下标必须与 Q 动作轴对齐：{actions.shape} != {values.shape[0]}")
    if beta <= 0:
        raise ValueError("beta 必须大于 0。")
    if no_information_penalty < 0:
        raise ValueError("no_information_penalty 不能小于 0。")
    if information_epsilon < 0:
        raise ValueError("information_epsilon 不能小于 0。")

    scaled = beta * values
    log_denominator = logsumexp(scaled, axis=2)
    row_index = np.arange(actions.size)[:, None]
    agent_index = np.arange(values.shape[1])[None, :]
    action_index = np.broadcast_to(actions[:, None], (actions.size, values.shape[1]))
    true_scores = scaled[row_index, agent_index, action_index]
    action_log_probability = true_scores - log_denominator

    finite_mask = np.isfinite(values)
    legal_count = np.sum(finite_mask, axis=2)
    finite_min = np.min(np.where(finite_mask, values, np.inf), axis=2)
    finite_max = np.max(np.where(finite_mask, values, -np.inf), axis=2)
    no_information = (
        (finite_max - finite_min <= float(information_epsilon))
        & (legal_count > 1)
    )
    # 固定惩罚是在均匀动作损失上增加一个与 beta 无关的常数。它既把无信息视为
    # 解释失败，又保留有限 likelihood，避免单个无信息 tile 让整个 context 变成 -inf。
    fixed_log_probability = -np.log(legal_count) - float(no_information_penalty)
    action_log_probability[no_information] = fixed_log_probability[no_information]
    return action_log_probability


def context_strategy_log_likelihood(
    observation: ContextObservation,
    beta: float,
    no_information_penalty: float = 2.0,
    information_epsilon: float = 1e-12,
) -> np.ndarray:
    """计算一个 context 在给定 beta 下的七策略 log-likelihood。

    输入语义：observation 已包含统一 Q 和真实方向；beta 必须为正数；固定惩罚和
    epsilon 控制无信息行的损失语义。
    输出语义：返回长度为策略数的 log-likelihood 数组。
    关键约束：使用 logsumexp 计算 softmax 分母，非法方向 ``-inf`` 自动不参与分母。
    """

    if beta <= 0:
        raise ValueError("beta 必须大于 0。")
    if observation.valid_action_count == 0:
        return np.full(observation.q_values.shape[1], np.nan, dtype=float)

    action_log_probability = action_log_probability_with_no_information_penalty(
        observation.q_values,
        observation.action_indices,
        beta,
        no_information_penalty=no_information_penalty,
        information_epsilon=information_epsilon,
    )
    return np.sum(action_log_probability, axis=0)


def context_eligible_strategy_log_likelihood(
    observation: ContextObservation,
    beta: float,
    no_information_penalty: float = 2.0,
    information_epsilon: float = 1e-12,
) -> np.ndarray:
    """构造信息覆盖率门控后的行为策略 likelihood。

    输入语义：observation 至少包含一个有效动作，beta 为文件级 temperature。
    输出语义：返回长度等于策略数的数组；coverage 不足的策略被写为 ``-inf``。
    关键约束：Null 不进入该数组，也不参与 beta/posterior；这里的 ``-inf`` 是 context
    级候选排除，不是把某一无信息动作的概率强行设为 0。
    """

    strategy_log_likelihood = context_strategy_log_likelihood(
        observation,
        beta,
        no_information_penalty=no_information_penalty,
        information_epsilon=information_epsilon,
    )
    eligible = observation.resolved_strategy_eligible()
    if eligible.shape != strategy_log_likelihood.shape:
        raise ValueError(
            "strategy eligibility 与 likelihood 长度不一致："
            f"{eligible.shape} != {strategy_log_likelihood.shape}"
        )
    gated = strategy_log_likelihood.copy()
    gated[~eligible] = -np.inf
    return gated


def context_marginal_nll(
    observation: ContextObservation,
    beta: float,
    no_information_penalty: float = 2.0,
    information_epsilon: float = 1e-12,
) -> float:
    """计算覆盖率合格策略边际化后的单 context 负对数似然。

    输入语义：observation 必须至少包含一个有效动作。
    输出语义：返回当前合格行为策略均匀先验下的标量 NLL。
    关键约束：coverage 不足的策略不占候选先验质量；没有合格策略的 context 应在
    上层被排除出 beta 拟合并标记 vague，不能调用本函数强行构造 NLL。
    """

    candidate_log_likelihood = context_eligible_strategy_log_likelihood(
        observation,
        beta,
        no_information_penalty=no_information_penalty,
        information_epsilon=information_epsilon,
    )
    if np.any(np.isnan(candidate_log_likelihood)):
        raise ValueError("无有效动作的 context 不能进入 beta loss。")
    candidate_count = int(np.sum(np.isfinite(candidate_log_likelihood)))
    if candidate_count == 0:
        raise ValueError("没有覆盖率合格策略的 context 不能进入 beta loss。")
    return float(-(logsumexp(candidate_log_likelihood) - math.log(candidate_count)))


def total_context_nll(
    observations: Iterable[ContextObservation],
    beta: float,
    no_information_penalty: float = 2.0,
    information_epsilon: float = 1e-12,
) -> float:
    """汇总一组 contexts 的边际负对数似然。

    输入语义：observations 可以跨 trial，但应属于同一 beta 参数组。
    输出语义：返回所有非 stay contexts 的 NLL 总和。
    关键约束：无有效动作段直接跳过，不向 loss 添加任意常数。
    """

    effective = [
        item
        for item in observations
        if item.valid_action_count > 0 and bool(np.any(item.resolved_strategy_eligible()))
    ]
    if not effective:
        return 0.0
    return batch_total_context_nll(
        build_observation_batch(effective),
        beta,
        no_information_penalty=no_information_penalty,
        information_epsilon=information_epsilon,
    )


def build_observation_batch(observations: list[ContextObservation]) -> ObservationBatch:
    """把有效 contexts 合并为一次向量化计算所需的批量数组。

    输入语义：observations 中每个 context 都必须至少包含一个有效动作。
    输出语义：返回拼接 Q、动作下标和 context 起点的 ObservationBatch。
    关键约束：保持输入 context 顺序；后续 ``np.add.reduceat`` 依赖每段长度严格为正。
    """

    if not observations or any(item.valid_action_count <= 0 for item in observations):
        raise ValueError("ObservationBatch 只能由非空且均含有效动作的 contexts 构造。")
    lengths = np.asarray([item.valid_action_count for item in observations], dtype=int)
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(int)
    return ObservationBatch(
        q_values=np.concatenate([item.q_values for item in observations], axis=0),
        action_indices=np.concatenate([item.action_indices for item in observations], axis=0),
        context_starts=starts,
        context_count=len(observations),
        strategy_eligible=np.stack(
            [item.resolved_strategy_eligible() for item in observations],
            axis=0,
        ),
    )


def batch_total_context_nll(
    batch: ObservationBatch,
    beta: float,
    no_information_penalty: float = 2.0,
    information_epsilon: float = 1e-12,
) -> float:
    """向量化计算一组 contexts 的总边际 NLL。

    输入语义：batch 已把所有有效动作连续拼接，beta 为正数。
    输出语义：返回与逐 context 调用 ``context_marginal_nll`` 完全相同的 NLL 总和。
    关键约束：先按动作计算每个策略 log 概率，再在 context 边界内求和，不能直接
    在动作层面对策略做边际化，否则会改变“一段一个潜在策略”的模型假设。
    """

    if beta <= 0:
        raise ValueError("beta 必须大于 0。")
    action_log_probability = action_log_probability_with_no_information_penalty(
        batch.q_values,
        batch.action_indices,
        beta,
        no_information_penalty=no_information_penalty,
        information_epsilon=information_epsilon,
    )
    context_log_likelihood = np.add.reduceat(
        action_log_probability,
        batch.context_starts,
        axis=0,
    )
    if context_log_likelihood.shape[0] != batch.context_count:
        raise RuntimeError("批量 context 聚合数量不一致。")
    if batch.strategy_eligible.shape != context_log_likelihood.shape:
        raise RuntimeError(
            "批量 eligibility 与 context likelihood 形状不一致："
            f"{batch.strategy_eligible.shape} != {context_log_likelihood.shape}"
        )
    candidate_log_likelihood = context_log_likelihood.copy()
    candidate_log_likelihood[~batch.strategy_eligible] = -np.inf
    candidate_count = np.sum(np.isfinite(candidate_log_likelihood), axis=1)
    if np.any(candidate_count == 0):
        raise RuntimeError("ObservationBatch 包含没有覆盖率合格策略的 context。")
    context_nll = -(
        logsumexp(candidate_log_likelihood, axis=1)
        - np.log(candidate_count)
    )
    return float(np.sum(context_nll))


def fit_beta(
    observations: list[ContextObservation],
    config: ContextStrategyPosteriorConfig,
) -> BetaFitResult:
    """使用对数网格与局部有界优化拟合一个 beta。

    输入语义：observations 共享同一个 temperature，至少包含一个有效 context。
    输出语义：返回搜索区间内 loss 最低的 beta 和 NLL。
    关键约束：先做全区间网格可避免直接假定混合模型 loss 单峰，再只在最佳邻域精修。
    """

    effective = [
        item
        for item in observations
        if item.valid_action_count > 0 and bool(np.any(item.resolved_strategy_eligible()))
    ]
    if not effective:
        raise ValueError("没有有效 context，无法拟合 beta。")
    batch = build_observation_batch(effective)

    eta_grid = np.linspace(math.log(config.beta_min), math.log(config.beta_max), config.beta_grid_size)

    def objective(eta: float) -> float:
        """把无约束 log-beta 转换为 beta 后计算总 NLL。"""

        return batch_total_context_nll(
            batch,
            math.exp(float(eta)),
            no_information_penalty=config.no_information_penalty,
            information_epsilon=config.information_epsilon,
        )

    grid_losses = np.asarray([objective(eta) for eta in eta_grid], dtype=float)
    best_index = int(np.argmin(grid_losses))
    candidate_pairs: list[tuple[float, float]] = [(float(eta_grid[best_index]), float(grid_losses[best_index]))]

    left_index = max(0, best_index - 1)
    right_index = min(len(eta_grid) - 1, best_index + 1)
    if right_index > left_index:
        local_result = minimize_scalar(
            objective,
            bounds=(float(eta_grid[left_index]), float(eta_grid[right_index])),
            method="bounded",
            options={"xatol": 1e-8},
        )
        if local_result.success and np.isfinite(local_result.fun):
            candidate_pairs.append((float(local_result.x), float(local_result.fun)))

    best_eta, best_loss = min(candidate_pairs, key=lambda item: item[1])
    return BetaFitResult(beta=float(math.exp(best_eta)), loss=float(best_loss))


def calculate_bic(loss: float, parameter_count: int, context_count: int) -> float:
    """按有效 player-context 数计算 BIC。

    输入语义：loss 是最小 NLL，parameter_count 是 beta 数量，context_count 是有效段数。
    输出语义：返回 ``2*NLL + m*log(C)``。
    关键约束：context_count 必须大于 0；tile 不是本模型 likelihood 的独立分解单位。
    """

    if context_count <= 0:
        raise ValueError("context_count 必须大于 0。")
    return float(2.0 * loss + parameter_count * math.log(context_count))


def build_grouped_folds(
    observations: list[ContextObservation],
    fold_count: int,
    random_seed: int,
) -> dict[str, int]:
    """按完整 DayTrial 建立确定性的 grouped folds。

    输入语义：observations 可以同时包含 P1/P2；相同 trial 名会自然归到同一 fold。
    输出语义：返回 ``DayTrial -> fold_id`` 映射；trial 少于两组时返回空字典。
    关键约束：不随机拆分 context 或 tile，避免相邻行为泄漏到验证集。
    """

    trial_names = sorted(
        {
            item.trial_name
            for item in observations
            if item.valid_action_count > 0 and bool(np.any(item.resolved_strategy_eligible()))
        }
    )
    actual_folds = min(fold_count, len(trial_names))
    if actual_folds < 2:
        return {}
    rng = np.random.default_rng(random_seed)
    shuffled = list(rng.permutation(trial_names))
    return {str(trial): index % actual_folds for index, trial in enumerate(shuffled)}


def fit_full_beta_models(
    observations_by_player: dict[str, list[ContextObservation]],
    config: ContextStrategyPosteriorConfig,
) -> dict[str, Any]:
    """拟合文件级共享和玩家独立 beta，并使用 BIC 选择最终结构。

    输入语义：字典包含文件内实际存在玩家的全部 contexts。
    输出语义：返回完整模型拟合、BIC、最终模型名和每个玩家采用的 beta。
    关键约束：单人文件没有“共享/独立”之分，直接使用一个 beta 并标记为 single。
    """

    effective_by_player = {
        player: [
            item
            for item in observations
            if item.valid_action_count > 0 and bool(np.any(item.resolved_strategy_eligible()))
        ]
        for player, observations in observations_by_player.items()
    }
    all_effective = [item for observations in effective_by_player.values() for item in observations]
    if not all_effective:
        raise ValueError("文件中没有任何可拟合 player-context。")

    shared_fit = fit_beta(all_effective, config)
    context_count = len(all_effective)
    shared_bic = calculate_bic(shared_fit.loss, 1, context_count)

    if len(effective_by_player) == 1:
        only_player = next(iter(effective_by_player))
        return {
            "selected_model": "single",
            "beta_by_player": {only_player: shared_fit.beta},
            "shared_beta": shared_fit.beta,
            "shared_loss": shared_fit.loss,
            "shared_bic": shared_bic,
            "separate_beta": None,
            "separate_loss": None,
            "separate_bic": None,
            "effective_context_count": context_count,
        }

    separate_fits = {
        player: fit_beta(observations, config)
        for player, observations in effective_by_player.items()
    }
    separate_loss = float(sum(item.loss for item in separate_fits.values()))
    separate_bic = calculate_bic(separate_loss, len(separate_fits), context_count)
    if shared_bic <= separate_bic:
        selected_model = "shared"
        beta_by_player = {player: shared_fit.beta for player in effective_by_player}
    else:
        selected_model = "separate"
        beta_by_player = {player: item.beta for player, item in separate_fits.items()}

    return {
        "selected_model": selected_model,
        "beta_by_player": beta_by_player,
        "shared_beta": shared_fit.beta,
        "shared_loss": shared_fit.loss,
        "shared_bic": shared_bic,
        "separate_beta": {player: item.beta for player, item in separate_fits.items()},
        "separate_loss": separate_loss,
        "separate_bic": separate_bic,
        "effective_context_count": context_count,
    }


def run_grouped_cross_validation(
    observations_by_player: dict[str, list[ContextObservation]],
    config: ContextStrategyPosteriorConfig,
) -> dict[str, Any]:
    """在文件内按 DayTrial 评估共享和独立 beta 的稳定性。

    输入语义：observations_by_player 是完整文件数据；配置给出最大折数和 seed。
    输出语义：返回 trial-fold 映射和每折训练 beta、BIC 选择、held-out NLL。
    关键约束：交叉验证仅是诊断；最终参数仍由全文件拟合和全文件 BIC 决定。
    """

    all_observations = [item for values in observations_by_player.values() for item in values]
    trial_to_fold = build_grouped_folds(all_observations, config.cv_folds, config.random_seed)
    if not trial_to_fold:
        return {"fold_count": 0, "trial_to_fold": {}, "folds": []}

    fold_results: list[dict[str, Any]] = []
    for fold_id in sorted(set(trial_to_fold.values())):
        train_by_player = {
            player: [
                item
                for item in observations
                if item.valid_action_count > 0
                and bool(np.any(item.resolved_strategy_eligible()))
                and trial_to_fold[item.trial_name] != fold_id
            ]
            for player, observations in observations_by_player.items()
        }
        validation_by_player = {
            player: [
                item
                for item in observations
                if item.valid_action_count > 0
                and bool(np.any(item.resolved_strategy_eligible()))
                and trial_to_fold[item.trial_name] == fold_id
            ]
            for player, observations in observations_by_player.items()
        }
        # 若极小数据在某折造成玩家训练集为空，则跳过该折而不是构造无依据 beta。
        if any(not values for values in train_by_player.values()):
            continue

        train_model = fit_full_beta_models(train_by_player, config)
        shared_validation_nll = float(
            sum(
                total_context_nll(
                    values,
                    train_model["shared_beta"],
                    no_information_penalty=config.no_information_penalty,
                    information_epsilon=config.information_epsilon,
                )
                for values in validation_by_player.values()
            )
        )
        separate_validation_nll: float | None = None
        if train_model["separate_beta"] is not None:
            separate_validation_nll = float(
                sum(
                    total_context_nll(
                        validation_by_player[player],
                        beta,
                        no_information_penalty=config.no_information_penalty,
                        information_epsilon=config.information_epsilon,
                    )
                    for player, beta in train_model["separate_beta"].items()
                )
            )
        fold_results.append(
            {
                "fold_id": int(fold_id),
                "validation_trials": sorted(
                    trial for trial, assigned_fold in trial_to_fold.items() if assigned_fold == fold_id
                ),
                "train_context_count": int(train_model["effective_context_count"]),
                "validation_context_count": int(sum(len(values) for values in validation_by_player.values())),
                "shared_beta": float(train_model["shared_beta"]),
                "shared_bic": float(train_model["shared_bic"]),
                "shared_validation_nll": shared_validation_nll,
                "separate_beta": train_model["separate_beta"],
                "separate_bic": train_model["separate_bic"],
                "separate_validation_nll": separate_validation_nll,
                "training_bic_selected_model": train_model["selected_model"],
            }
        )
    return {
        "fold_count": len(set(trial_to_fold.values())),
        "trial_to_fold": trial_to_fold,
        "folds": fold_results,
    }


def posterior_from_log_likelihood(log_likelihood: np.ndarray) -> np.ndarray:
    """把均匀候选先验下的 log-likelihood 转换为 posterior。

    输入语义：log_likelihood 是一维数组；被 coverage 门控排除的候选允许为 ``-inf``。
    输出语义：返回总和为 1 的 posterior 数组。
    关键约束：至少一个候选必须有限，且不允许 NaN/+inf；使用 logsumexp 避免下溢。
    """

    values = np.asarray(log_likelihood, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or np.any(np.isnan(values))
        or np.any(np.isposinf(values))
        or not np.any(np.isfinite(values))
    ):
        raise ValueError(f"候选 log-likelihood 必须是一维且至少有一个有限值：{values}")
    return np.exp(values - logsumexp(values))


def initialize_player_output(index: pd.Index, player: str) -> pd.DataFrame:
    """创建 06 玩家级逐行输出空表。

    输入语义：index 与 joint-state 输入表一致。
    输出语义：返回所有 posterior 和 best Global 字段均为 object dtype 的 DataFrame。
    关键约束：复杂列表、tuple、字典不能让 pandas 自动展开为多列。
    """

    columns = (
        "selected_global_Q",
        "best_global_cluster_id",
        "best_global_cluster_prob_accuracy",
        "best_global_cluster_set_accuracy",
        "best_global_cluster_meta",
        "selected_energizer_Q",
        "best_energizer_target_position",
        "best_energizer_target_prob_accuracy",
        "best_energizer_target_set_accuracy",
        "best_energizer_target_meta",
        "selected_approach_Q",
        "best_approach_target_id",
        "best_approach_target_position",
        "best_approach_target_prob_accuracy",
        "best_approach_target_set_accuracy",
        "best_approach_target_meta",
        "trial_context",
        "strategy_log_likelihood",
        "strategy_information_coverage",
        "strategy_eligible",
        "strategy_posterior",
        "strategy_posterior_max",
        "strategy_candidate",
        "strategy",
        "strategy_name",
        "null_log_likelihood",
        "log_likelihood_gain",
        "valid_action_count",
        "is_stay",
        "is_vague",
    )
    output = pd.DataFrame(index=index)
    for column in columns:
        output[f"{player}_{column}"] = pd.Series([np.nan] * len(index), index=index, dtype=object)
    return output


def write_player_posterior(
    prepared: PreparedPlayerData,
    beta: float,
    config: ContextStrategyPosteriorConfig,
) -> pd.DataFrame:
    """使用最终 beta 计算并逐行写回一个玩家的 context 后验。

    输入语义：prepared 保存选好 Global 的视图；beta 已由文件级 BIC 模型确定。
    输出语义：返回与 joint-state 等长的玩家前缀结果列。
    关键约束：同一 context 的 likelihood/posterior 完全相同；selected_global_Q 仍逐行
    保存，因为资源变化后同一目标 cluster 的方向 utility 会随位置改变。
    """

    output = initialize_player_output(prepared.view.index, prepared.player)
    player = prepared.player
    for observation in prepared.observations:
        start, end = observation.context
        labels = list(range(start, end))
        selected_q = [copy.deepcopy(prepared.view.at[index, "global_Q"]) for index in labels]
        output.loc[labels, f"{player}_selected_global_Q"] = pd.Series(selected_q, index=labels, dtype=object)
        for source, target in (
            ("best_global_cluster_id", "best_global_cluster_id"),
            ("best_global_cluster_prob_accuracy", "best_global_cluster_prob_accuracy"),
            ("best_global_cluster_set_accuracy", "best_global_cluster_set_accuracy"),
            ("best_global_cluster_meta", "best_global_cluster_meta"),
        ):
            values = [copy.deepcopy(prepared.view.at[index, source]) for index in labels]
            output.loc[labels, f"{player}_{target}"] = pd.Series(values, index=labels, dtype=object)

        selected_approach_q = [
            copy.deepcopy(prepared.view.at[index, "selected_approach_Q"])
            for index in labels
        ]
        output.loc[labels, f"{player}_selected_approach_Q"] = pd.Series(
            selected_approach_q,
            index=labels,
            dtype=object,
        )
        for source, target in (
            ("best_approach_target_id", "best_approach_target_id"),
            ("best_approach_target_position", "best_approach_target_position"),
            ("best_approach_target_prob_accuracy", "best_approach_target_prob_accuracy"),
            ("best_approach_target_set_accuracy", "best_approach_target_set_accuracy"),
            ("best_approach_target_meta", "best_approach_target_meta"),
        ):
            values = [copy.deepcopy(prepared.view.at[index, source]) for index in labels]
            output.loc[labels, f"{player}_{target}"] = pd.Series(values, index=labels, dtype=object)

        selected_energizer_q = [
            copy.deepcopy(prepared.view.at[index, "selected_energizer_Q"])
            for index in labels
        ]
        output.loc[labels, f"{player}_selected_energizer_Q"] = pd.Series(
            selected_energizer_q,
            index=labels,
            dtype=object,
        )
        for source, target in (
            ("best_energizer_target_position", "best_energizer_target_position"),
            ("best_energizer_target_prob_accuracy", "best_energizer_target_prob_accuracy"),
            ("best_energizer_target_set_accuracy", "best_energizer_target_set_accuracy"),
            ("best_energizer_target_meta", "best_energizer_target_meta"),
        ):
            values = [copy.deepcopy(prepared.view.at[index, source]) for index in labels]
            output.loc[labels, f"{player}_{target}"] = pd.Series(values, index=labels, dtype=object)

        if observation.valid_action_count == 0:
            log_likelihood = [float("nan")] * len(config.agents)
            information_coverage = [float("nan")] * len(config.agents)
            strategy_eligible = [False] * len(config.agents)
            posterior = [float("nan")] * len(config.agents)
            posterior_max = float("nan")
            candidate = "stay"
            strategy_name = "stay"
            is_vague = False
            null_likelihood = float("nan")
            gain = float("nan")
        else:
            log_likelihood_array = context_strategy_log_likelihood(
                observation,
                beta,
                no_information_penalty=config.no_information_penalty,
                information_epsilon=config.information_epsilon,
            )
            candidate_log_likelihood = context_eligible_strategy_log_likelihood(
                observation,
                beta,
                no_information_penalty=config.no_information_penalty,
                information_epsilon=config.information_epsilon,
            )
            eligible = observation.resolved_strategy_eligible()
            if bool(np.any(eligible)):
                posterior_array = posterior_from_log_likelihood(candidate_log_likelihood)
                candidate_index = int(np.argmax(posterior_array))
                candidate = config.agents[candidate_index]
                posterior_max = float(posterior_array[candidate_index])
                is_vague = posterior_max < config.posterior_threshold
                strategy_name = "vague" if is_vague else candidate
            else:
                # Context 有动作但所有策略都缺乏足够信息时，不强行从七策略中选择。
                posterior_array = np.zeros(len(config.agents), dtype=float)
                candidate = "none"
                posterior_max = 0.0
                is_vague = True
                strategy_name = "vague"
            log_likelihood = log_likelihood_array.tolist()
            information_coverage = observation.resolved_information_coverage().tolist()
            strategy_eligible = eligible.tolist()
            posterior = posterior_array.tolist()
            null_likelihood = observation.null_log_likelihood
            eligible_log_likelihood = log_likelihood_array[observation.resolved_strategy_eligible()]
            gain = (
                float((float(np.max(eligible_log_likelihood)) - null_likelihood) / observation.valid_action_count)
                if eligible_log_likelihood.size
                else float("nan")
            )

        repeated_values: dict[str, Any] = {
            "trial_context": observation.context,
            "strategy_log_likelihood": log_likelihood,
            "strategy_information_coverage": information_coverage,
            "strategy_eligible": strategy_eligible,
            "strategy_posterior": posterior,
            "strategy_posterior_max": posterior_max,
            "strategy_candidate": candidate,
            "strategy": STRATEGY_NUMBER[strategy_name],
            "strategy_name": strategy_name,
            "null_log_likelihood": null_likelihood,
            "log_likelihood_gain": gain,
            "valid_action_count": observation.valid_action_count,
            "is_stay": observation.valid_action_count == 0,
            "is_vague": is_vague,
        }
        for name, value in repeated_values.items():
            output.loc[labels, f"{player}_{name}"] = pd.Series(
                [copy.deepcopy(value) for _ in labels],
                index=labels,
                dtype=object,
            )
    return output


def validate_row_id(data: pd.DataFrame) -> None:
    """验证 row_id 与重置后的 DataFrame 标签一一对应。

    输入语义：data 是 05 单文件表。
    输出语义：一致时无返回；否则抛错。
    关键约束：context 使用 row_id 作为全文件 offset，错位会导致跨 trial 写回错误。
    """

    if "row_id" not in data.columns:
        raise ValueError("06 输入缺少 row_id。")
    row_id = pd.to_numeric(data["row_id"], errors="raise").to_numpy(dtype=int)
    expected = np.arange(len(data), dtype=int)
    if not np.array_equal(row_id, expected):
        raise ValueError("06 要求 row_id 从 0 连续递增并与 DataFrame 行标签一致。")


def fit_context_strategy_posterior_dataframe(
    raw_data: pd.DataFrame,
    config: ContextStrategyPosteriorConfig | None = None,
) -> pd.DataFrame:
    """对一个 joint-state 文件执行完整 06 后验拟合。

    输入语义：raw_data 是 05 cluster-global utility 输出。
    输出语义：返回保留原字段并追加 P1/P2 posterior、strategy 和模型 attrs 的 DataFrame。
    关键约束：每个文件单独拟合 beta；P1/P2 是否共享由当前文件全数据 BIC 决定。
    """

    config = ContextStrategyPosteriorConfig() if config is None else config
    validate_config(config)
    base = raw_data.reset_index(drop=True).copy(deep=True)
    validate_row_id(base)
    players = discover_posterior_players(base, config)
    result = append_player_event_columns(base, players)

    prepared_by_player = {
        player: prepare_player_data(result, player, config)
        for player in players
    }
    observations_by_player = {
        player: prepared.observations
        for player, prepared in prepared_by_player.items()
    }
    full_model = fit_full_beta_models(observations_by_player, config)
    cross_validation = run_grouped_cross_validation(observations_by_player, config)

    for player, prepared in prepared_by_player.items():
        player_output = write_player_posterior(prepared, full_model["beta_by_player"][player], config)
        for column in player_output.columns:
            result[column] = player_output[column].to_numpy()

    # DataFrame.attrs 会随 pickle 保存；这里集中记录文件级参数，避免在每一行重复大段
    # CV 信息。global_selection_uses_context_actions 显式提醒后续分析其乐观偏差来源。
    result.attrs = copy.deepcopy(raw_data.attrs)
    result.attrs["context_strategy_posterior_model"] = {
        "version": "strategy-posterior-v1",
        "strategy_order": list(config.agents),
        "strategy_number": {name: STRATEGY_NUMBER[name] for name in config.agents},
        "normalization": "per_player_tile_strategy_legal_direction_minmax_from_raw_q",
        "bean_event_suppression_window": config.bean_event_suppression_window,
        "ghost_stay_suppression_window": config.ghost_stay_suppression_window,
        "strategy_prior": "uniform_over_coverage_eligible_behavior_strategies",
        "posterior_threshold": config.posterior_threshold,
        "min_information_coverage": config.min_information_coverage,
        "information_epsilon": config.information_epsilon,
        "no_information_likelihood": "uniform_log_probability_minus_fixed_penalty",
        "no_information_penalty": config.no_information_penalty,
        "null_model": "uniform_over_legal_actions_diagnostic_only",
        "null_participates_in_beta_or_posterior": False,
        "global_selection_rule": "context_probability_accuracy",
        "global_selection_uses_context_actions": True,
        "energizer_utility_rule": "target_shortest_path_distance_reduction_without_radius",
        "energizer_selection_rule": "context_probability_accuracy_by_target_position",
        "energizer_selection_uses_context_actions": True,
        "energizer_outcome_used_in_selection": False,
        "approach_utility_rule": "per_ghost_first_target_hit_discounted_reward",
        "approach_selection_rule": "context_probability_accuracy_by_stable_ghost_id",
        "approach_selection_uses_context_actions": True,
        "approach_outcome_used_in_selection": False,
        "selected_beta_model": full_model["selected_model"],
        "beta_by_player": full_model["beta_by_player"],
        "shared_beta": full_model["shared_beta"],
        "shared_loss": full_model["shared_loss"],
        "shared_bic": full_model["shared_bic"],
        "separate_beta": full_model["separate_beta"],
        "separate_loss": full_model["separate_loss"],
        "separate_bic": full_model["separate_bic"],
        "effective_context_count": full_model["effective_context_count"],
        "valid_action_count_by_player": {
            player: int(sum(item.valid_action_count for item in prepared.observations))
            for player, prepared in prepared_by_player.items()
        },
        "cross_validation": cross_validation,
    }
    return result


def process_context_strategy_posterior_file(
    input_path: str | Path,
    output_path: str | Path,
    config: ContextStrategyPosteriorConfig | None = None,
    file_index: int = 0,
) -> dict[str, Any]:
    """处理并保存一个 05 utility 文件。

    输入语义：input_path/output_path 是对应的嵌套 pickle 路径；file_index 派生 CV seed。
    输出语义：保存 06 DataFrame，并返回便于 CLI 汇总的轻量摘要。
    关键约束：输出目录必须与输入 utility 目录分离，避免覆盖上游数据。
    """

    base_config = ContextStrategyPosteriorConfig() if config is None else config
    file_config = ContextStrategyPosteriorConfig(
        agents=base_config.agents,
        stay_length=base_config.stay_length,
        bean_event_suppression_window=base_config.bean_event_suppression_window,
        ghost_stay_suppression_window=base_config.ghost_stay_suppression_window,
        beta_min=base_config.beta_min,
        beta_max=base_config.beta_max,
        beta_grid_size=base_config.beta_grid_size,
        cv_folds=base_config.cv_folds,
        posterior_threshold=base_config.posterior_threshold,
        min_information_coverage=base_config.min_information_coverage,
        information_epsilon=base_config.information_epsilon,
        no_information_penalty=base_config.no_information_penalty,
        random_seed=base_config.random_seed + file_index,
    )
    input_file = Path(input_path)
    output_file = Path(output_path)
    with input_file.open("rb") as file:
        raw_data = pickle.load(file)
    result = fit_context_strategy_posterior_dataframe(raw_data, file_config)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_pickle(output_file)
    model = result.attrs["context_strategy_posterior_model"]
    return {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "rows": int(len(result)),
        "players": sorted(model["beta_by_player"]),
        "selected_beta_model": model["selected_beta_model"],
        "beta_by_player": model["beta_by_player"],
        "shared_bic": model["shared_bic"],
        "separate_bic": model["separate_bic"],
    }


def process_context_strategy_posterior_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    config: ContextStrategyPosteriorConfig | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """按嵌套任务目录批量执行 06。

    输入语义：input_dir 包含 ``comp/*.pkl``、``coop/*.pkl``，workers 控制文件级进程数。
    输出语义：保持相对目录结构保存并返回所有文件摘要。
    关键约束：文件之间没有参数共享，因此文件级并行不改变统计结果。
    """

    config = ContextStrategyPosteriorConfig() if config is None else config
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    input_files = sorted(path for path in input_root.glob("*/*.pkl") if path.is_file())
    if not input_files:
        raise FileNotFoundError(f"输入目录中没有嵌套 pickle 文件：{input_root}")
    tasks = [
        (
            input_file,
            output_root / input_file.relative_to(input_root),
            config,
            file_index,
        )
        for file_index, input_file in enumerate(input_files)
    ]
    if workers <= 1:
        return [_process_posterior_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        return list(executor.map(_process_posterior_task, tasks))


def _process_posterior_task(
    task: tuple[Path, Path, ContextStrategyPosteriorConfig, int],
) -> dict[str, Any]:
    """执行目录级进程池中的单文件 06 任务。"""

    input_path, output_path, config, file_index = task
    return process_context_strategy_posterior_file(
        input_path,
        output_path,
        config,
        file_index=file_index,
    )
