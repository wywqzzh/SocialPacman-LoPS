#!/usr/bin/env python3
"""根据事件规则修正 Context 策略后验结果。

本阶段保留 06 的原始 posterior 和 strategy，只把 posterior 映射到修正规则使用的
临时策略分数，并从 raw Q 重建统一 Min-Max 方向分数。规则若触发 one-hot/multi-hot
覆盖，结果保存为 revised strategy score，避免把人工规则输出误称为概率后验。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.context_strategy_posterior import (  # noqa: E402
    DEFAULT_AGENTS,
    STRATEGY_NUMBER,
    normalize_legal_q,
)
from LoPS.context_strategy_revision import revise_player_view  # noqa: E402


PLAYER_PREFIXES: tuple[str, ...] = ("p1", "p2")
REVERSE_STRATEGY_NUMBER = {number: name for name, number in STRATEGY_NUMBER.items()}


def discover_posterior_players(data: pd.DataFrame) -> list[str]:
    """识别 06 输出中可执行 07 的玩家。

    输入语义：data 是 06 joint-state DataFrame。
    输出语义：返回字段完整的 p1/p2 列表，单人文件自动跳过缺失 p2。
    关键约束：posterior、context、私有事件和七策略 raw Q 必须同时存在，否则直接报错。
    """

    players: list[str] = []
    for player in PLAYER_PREFIXES:
        signal = {f"{player}_strategy_posterior", f"{player}_trial_context"}
        if signal.isdisjoint(data.columns):
            continue
        required = {
            "DayTrial",
            "row_id",
            "ifscared1",
            "ifscared2",
            f"{player}_action_dir",
            f"{player}_available_dir",
            f"{player}_strategy_posterior",
            f"{player}_strategy_candidate",
            f"{player}_strategy_information_coverage",
            f"{player}_strategy_eligible",
            f"{player}_trial_context",
            f"{player}_is_stay",
            f"{player}_is_vague",
            f"{player}_eat_energizer",
            f"{player}_eat_ghost",
            f"{player}_selected_global_Q",
            f"{player}_selected_energizer_Q",
            f"{player}_selected_approach_Q",
        }
        required.update(f"{player}_{agent}_Q" for agent in DEFAULT_AGENTS if agent != "global")
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"{player} 07 输入字段不完整，缺少：{missing}")
        players.append(player)
    if not players:
        raise ValueError("没有找到可修正的 06 玩家字段。")
    return players


def parse_posterior_score(value: Any, strategy_count: int) -> list[float]:
    """把 06 posterior 单元整理成修正规则可读取的策略分数。

    输入语义：value 通常是长度为 7 的 list；stay 行可能保存全 NaN。
    输出语义：有效 posterior 原样返回；全 NaN 返回全零，供 is_stay 优先级处理。
    关键约束：正式 06 的 posterior 已只在 coverage 合格的行为策略间归一化；部分
    NaN 或长度错误视为数据损坏，不能静默补齐。
    """

    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size != strategy_count:
        raise ValueError(f"strategy_posterior 长度应为 {strategy_count}，实际为 {values.size}")
    if np.all(np.isnan(values)):
        return [0.0] * strategy_count
    if np.any(~np.isfinite(values)):
        raise ValueError(f"strategy_posterior 包含部分非有限值：{values.tolist()}")
    return values.tolist()


def prepare_revision_view(data: pd.DataFrame, player: str) -> pd.DataFrame:
    """把 06 玩家字段映射为现有 07 规则所需的临时单人视图。

    输入语义：data 是 06 输出，player 指定当前玩家。
    输出语义：返回包含通用 posterior 分数、动作、事件、context 和统一 Q_norm 的副本。
    关键约束：该映射只存在于内存；不会在正式输出中伪造 normalized_weight 字段。
    """

    view = data.copy(deep=True)
    view["normalized_weight"] = view[f"{player}_strategy_posterior"].apply(
        lambda value: parse_posterior_score(value, len(DEFAULT_AGENTS))
    )
    view["action_dir"] = view[f"{player}_action_dir"].apply(
        lambda value: value if isinstance(value, str) else np.nan
    )
    view["trial_context"] = copy.deepcopy(view[f"{player}_trial_context"])
    view["is_stay"] = view[f"{player}_is_stay"].astype(bool)
    view["is_vague"] = view[f"{player}_is_vague"].astype(bool)
    view["eat_energizer"] = view[f"{player}_eat_energizer"].astype(bool)
    view["eat_ghost"] = view[f"{player}_eat_ghost"].astype(bool)

    # 07 明确排除死亡和 available_dir=False 行。与 06 不同，这里不再根据每行 raw Q
    # mask 预先排除“真实动作落在非法方向”的异常行；后续准确率函数会把它记为未命中。
    invalid_mask = ~view[f"{player}_available_dir"].astype(bool)
    alive_column = f"{player}_alive"
    if alive_column in view.columns:
        invalid_mask |= ~view[alive_column].astype(bool)
    view.loc[invalid_mask, "action_dir"] = np.nan

    # 修正规则只把 prediction_correct/predict_dir 当作诊断和后续写回容器；初始值不参与
    # context 的策略准确率重算，因此以 NaN 初始化最诚实。
    view["prediction_correct"] = pd.Series([np.nan] * len(view), index=view.index, dtype=object)
    view["predict_dir"] = pd.Series([np.nan] * len(view), index=view.index, dtype=object)

    for agent in DEFAULT_AGENTS:
        if agent == "global":
            raw_column = f"{player}_selected_global_Q"
        elif agent == "energizer":
            raw_column = f"{player}_selected_energizer_Q"
        elif agent == "approach":
            raw_column = f"{player}_selected_approach_Q"
        else:
            raw_column = f"{player}_{agent}_Q"
        normalized_values: list[list[float]] = []
        for row_index, value in view[raw_column].items():
            q_array = np.asarray(value, dtype=float)
            if q_array.shape != (4,):
                if bool(invalid_mask.loc[row_index]):
                    # 玩家死亡或该行没有有效动作时，上游会把玩家级 Q 保存为 NaN。
                    # 旧修正规则已排除这些行，因此用全零占位只用于维持 n×4 数据形态，
                    # 不会产生任何动作证据；有效动作行若缺 Q 则仍然立即报错。
                    normalized_values.append([0.0] * 4)
                    continue
                raise ValueError(
                    f"{player} row={row_index} agent={agent} 有有效动作但 raw Q 不是长度 4：{value!r}"
                )
            normalized_values.append(normalize_legal_q(q_array).tolist())
        view[f"{agent}_Q_norm"] = pd.Series(normalized_values, index=view.index, dtype=object)
    return view


def strategy_name(value: Any) -> str:
    """把修正规则输出的策略编号转换为稳定策略名。

    输入语义：value 应为整数或可转整数数值。
    输出语义：返回七策略、vague 或 stay 名称。
    关键约束：未知编号直接报错，避免视频静默显示错误颜色。
    """

    number = int(value)
    if number not in REVERSE_STRATEGY_NUMBER:
        raise ValueError(f"未知修正策略编号：{number}")
    return REVERSE_STRATEGY_NUMBER[number]


def write_player_revision(
    result: pd.DataFrame,
    player: str,
    revised_view: pd.DataFrame,
) -> dict[str, int]:
    """把 07 临时修正结果写回独立 revised 字段。

    输入语义：result 保留完整 06 数据，revised_view 来自现有规则处理。
    输出语义：追加 revised strategy score、诊断、编号、名称和是否改变，并返回计数。
    关键约束：不覆盖 06 的 ``strategy_posterior/strategy/strategy_name``。
    """

    if len(result) != len(revised_view):
        raise ValueError(f"{player} 07 修正后行数不一致：{len(result)} != {len(revised_view)}")
    revised_strategy = pd.to_numeric(revised_view["strategy"], errors="raise").astype(int)
    original_strategy = pd.to_numeric(result[f"{player}_strategy"], errors="raise").astype(int)
    changed = revised_strategy.to_numpy() != original_strategy.to_numpy()

    result[f"{player}_revised_strategy_score"] = revised_view["revised_normalized_weight"].to_numpy()
    result[f"{player}_revised_prediction_correct"] = revised_view["revised_prediction_correct"].to_numpy()
    result[f"{player}_revised_predict_dir"] = revised_view["predict_dir"].to_numpy()
    result[f"{player}_revised_is_vague"] = revised_view["is_vague"].astype(bool).to_numpy()
    result[f"{player}_revised_strategy"] = revised_strategy.to_numpy()
    result[f"{player}_revised_strategy_name"] = revised_strategy.map(strategy_name).to_numpy()
    result[f"{player}_strategy_revised"] = changed
    return {
        "changed_rows": int(np.sum(changed)),
        "changed_contexts": int(
            result.loc[changed, f"{player}_trial_context"].apply(tuple).nunique()
        ),
    }


def revise_context_strategy_dataframe(
    data: pd.DataFrame,
    input_path: Path,
    scared_time: int = 34,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """对一个策略后验 joint-state DataFrame 执行完整规则修正。

    输入语义：data 包含 06 posterior，input_path 用于规则错误定位信息。
    输出语义：返回保留原字段的修正结果和每个玩家的修改计数。
    关键约束：除当前停用的 Approach 二次修正外，其余规则按固定顺序执行；
    初始策略分数来自 coverage-gated posterior，方向证据来自统一 Q。
    """

    result = data.reset_index(drop=True).copy(deep=True)
    players = discover_posterior_players(result)
    summaries: dict[str, dict[str, int]] = {}
    for player in players:
        player_view = prepare_revision_view(result, player)
        revised_view = revise_player_view(
            player_view,
            input_path,
            player,
            scared_time,
        )
        summaries[player] = write_player_revision(result, player, revised_view)

    result.attrs = copy.deepcopy(data.attrs)
    result.attrs["context_strategy_posterior_revision"] = {
        "version": "strategy-revision-v1",
        "source": "strategy-posterior-v1",
        "initial_strategy_score": "coverage_gated_strategy_posterior",
        "q_normalization": "per_player_tile_strategy_legal_direction_minmax_from_raw_q",
        "global_q_source": "selected_global_Q",
        "energizer_q_source": "selected_energizer_Q",
        "approach_q_source": "selected_approach_Q",
        "energizer_outcome_rule": (
            "context_end_eat_event_and_energizer_accuracy_ge_0.70_"
            "and_relative_to_best_ge_0.80"
        ),
        # 显式记录 Approach 修正规则状态，保证仅查看输出文件 attrs 也能还原本次流程。
        "historical_rule_order_reused": False,
        "uneaten_ghost_approach_revision_enabled": False,
        "scared_time": int(scared_time),
        "summary": summaries,
    }
    return result, summaries


def process_one_file(input_path: Path, output_path: Path, scared_time: int = 34) -> dict[str, Any]:
    """读取、修正并保存一个策略后验 pickle。

    输入语义：input_path/output_path 是单文件输入输出，scared_time 沿用 tile 级窗口。
    输出语义：保存 07 文件并返回摘要。
    关键约束：输出目录与输入目录分离，不覆盖策略 posterior 数据。
    """

    data = pd.read_pickle(input_path)
    result, player_summary = revise_context_strategy_dataframe(data, input_path, scared_time)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_pickle(output_path)
    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "rows": int(len(result)),
        "players": player_summary,
    }


def list_nested_pickle_files(input_dir: Path) -> list[Path]:
    """列出策略后验嵌套目录中的全部 pickle。

    输入语义：input_dir 是包含 task 子目录的 06 根目录。
    输出语义：返回排序后的 ``*/*.pkl`` 文件列表。
    关键约束：不兼容扁平目录，保持当前项目统一嵌套结构。
    """

    files = sorted(path for path in input_dir.glob("*/*.pkl") if path.is_file())
    if not files:
        raise FileNotFoundError(f"输入目录中没有嵌套 pickle：{input_dir}")
    return files


def _process_task(task: tuple[Path, Path, int]) -> dict[str, Any]:
    """执行文件级进程池中的单个策略修正任务。"""

    return process_one_file(*task)


def process_directory(
    input_dir: Path,
    output_dir: Path,
    processes: int,
    scared_time: int,
) -> list[dict[str, Any]]:
    """按嵌套目录批量执行策略修正。

    输入语义：input_dir/output_dir 保持同一相对层级，processes 控制文件级并行。
    输出语义：返回全部文件摘要。
    关键约束：每个文件独立修正，进程并行不改变结果。
    """

    files = list_nested_pickle_files(input_dir)
    tasks = [(path, output_dir / path.relative_to(input_dir), scared_time) for path in files]
    if processes <= 1:
        return [_process_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=min(processes, len(tasks))) as executor:
        return list(executor.map(_process_task, tasks))


def resolve_single_file(input_dir: Path, value: Path) -> Path:
    """把单文件参数解析为存在的策略后验 pickle。

    输入语义：value 可以是绝对路径、当前目录相对路径或 input_dir 相对路径。
    输出语义：返回实际存在的文件。
    关键约束：不做模糊搜索，避免修正错误被试。
    """

    candidates = [value]
    if not value.is_absolute():
        candidates.append(input_dir / value)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"找不到 single-file：{value}")


def parse_args() -> argparse.Namespace:
    """解析策略修正输入输出、并行、单文件和 scared 窗口参数。"""

    data_root = PROJECT_ROOT / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=data_root / "06_strategy_posterior_data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_root / "07_revised_strategy_data",
    )
    parser.add_argument("--single-file", type=Path, default=None)
    parser.add_argument("--processes", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--scared-time", type=int, default=34)
    return parser.parse_args()


def main() -> None:
    """命令行入口：执行单文件或嵌套目录策略修正并打印 JSON 摘要。"""

    args = parse_args()
    if args.single_file is not None:
        input_file = resolve_single_file(args.input_dir, args.single_file)
        try:
            relative_path = input_file.relative_to(args.input_dir)
        except ValueError:
            relative_path = Path(input_file.name)
        summary = process_one_file(input_file, args.output_dir / relative_path, args.scared_time)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    summaries = process_directory(args.input_dir, args.output_dir, args.processes, args.scared_time)
    print(json.dumps({"processed_files": len(summaries), "files": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
