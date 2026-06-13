#!/usr/bin/env python3
"""运行旧版 ExtractFeaturesHuman 逻辑的临时流水线入口。

该脚本导入旧项目 ``FeatureExtractor/ExtractFeaturesHuman.py`` 中的特征计算函数，
但不调用其固定写入旧项目 ``HumanData`` 的 ``main``。本入口只负责把旧链路第 5 步
输出转换成连续特征和离散特征，并写入本轮深度验证的 ``data_temp`` 目录。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_APPEND_COLUMNS = [
    "revise_weight",
    "contribution",
    "weight",
    "file",
    "level_0",
    "strategy",
    "DayTrial",
    "Unnamed: 0",
]


def output_file_name(input_path: Path) -> str:
    """按旧脚本规则生成短输出文件名。"""

    return "-".join(input_path.name.split("-")[:2]) + ".pkl"


def load_legacy_module(legacy_root: Path) -> Any:
    """导入旧版 ExtractFeaturesHuman 模块。

    输入语义：legacy_root 是旧项目根目录。
    输出语义：返回已导入的模块对象。
    关键约束：旧函数内部使用 ``../ConstantData`` 相对路径，因此导入前切换到
    ``FeatureExtractor`` 目录。
    """

    feature_dir = legacy_root / "FeatureExtractor"
    os.chdir(feature_dir)
    module_path = feature_dir / "ExtractFeaturesHuman.py"
    spec = importlib.util.spec_from_file_location("legacy_extract_features_human", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法导入旧版 ExtractFeaturesHuman：{module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_one_file(input_path: Path, feature_output_dir: Path, discrete_output_dir: Path, legacy_root: Path) -> dict[str, Any]:
    """使用旧版函数处理单个 corrected weight 文件。

    输入语义：input_path 是旧链路第 5 步输出。
    输出语义：写出旧版连续特征和离散特征，并返回摘要。
    关键约束：保存路径由本入口控制，避免污染旧项目目录。
    """

    module = load_legacy_module(legacy_root)
    data = pd.read_pickle(input_path)
    for column_name, _ in data.items():
        data[column_name] = data[column_name].apply(
            lambda value: float(0) if isinstance(value, list) and len(value) == 0 else value
        )

    features = module.extractFeature(data)
    for column_name in FEATURE_APPEND_COLUMNS:
        features[column_name] = np.array(data[column_name])
    features.reset_index(drop=True, inplace=True)

    predictors = module.predictor4Prediction(features)
    predictors["EE"] = np.array(features["EE"])
    for column_name in FEATURE_APPEND_COLUMNS:
        predictors[column_name] = np.array(data[column_name])

    output_name = output_file_name(input_path)
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    discrete_output_dir.mkdir(parents=True, exist_ok=True)
    features.to_pickle(feature_output_dir / output_name)
    predictors.to_pickle(discrete_output_dir / output_name)
    return {"input_file": input_path.name, "output_file": output_name, "rows": int(len(data))}


def parse_args() -> argparse.Namespace:
    """解析旧版 extract features human 临时入口参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--feature-output-dir", type=Path, required=True)
    parser.add_argument("--discrete-output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """批量运行旧版特征提取并写出报告。"""

    args = parse_args()
    original_cwd = Path.cwd()
    args.legacy_root = args.legacy_root.resolve()
    args.input_dir = args.input_dir.resolve()
    args.feature_output_dir = args.feature_output_dir.resolve()
    args.discrete_output_dir = args.discrete_output_dir.resolve()
    args.report = args.report.resolve()

    input_paths = sorted(args.input_dir.glob("*.pkl"))
    if not input_paths:
        raise FileNotFoundError(f"旧版 extract features 输入目录中没有 pkl：{args.input_dir}")

    worker = partial(
        process_one_file,
        feature_output_dir=args.feature_output_dir,
        discrete_output_dir=args.discrete_output_dir,
        legacy_root=args.legacy_root,
    )
    process_count = max(1, min(args.processes, len(input_paths)))
    try:
        if process_count == 1:
            summaries = [worker(path) for path in input_paths]
        else:
            with mp.Pool(processes=process_count) as pool:
                summaries = pool.map(worker, input_paths)
    finally:
        os.chdir(original_cwd)

    report = {
        "input_dir": str(args.input_dir),
        "feature_output_dir": str(args.feature_output_dir),
        "discrete_output_dir": str(args.discrete_output_dir),
        "file_count": len(summaries),
        "total_rows": int(sum(item["rows"] for item in summaries)),
        "files": summaries,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("file_count", "total_rows")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
