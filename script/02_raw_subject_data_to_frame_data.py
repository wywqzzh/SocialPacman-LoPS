#!/usr/bin/env python3
"""运行 Pacman raw_subject_data 到 frame_data 的转换。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from LoPS.pacman_data.raw_subject_data_to_frame_data import convert_raw_subject_data_to_frame_data_dir


def parse_args() -> argparse.Namespace:
    """解析 frame_data 生成脚本参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data/01_raw_subject_data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/02_frame_data")
    parser.add_argument("--csv-output-dir", type=Path, default=PROJECT_ROOT / "data/02_frame_data_csv")
    parser.add_argument("--workers", type=int, default=34)
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("sessions", nargs="*", help="可选：只处理这些 session。")
    return parser.parse_args()


def main() -> None:
    """执行 frame_data 转换并打印摘要。"""

    args = parse_args()
    results = convert_raw_subject_data_to_frame_data_dir(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        csv_output_dir=args.csv_output_dir,
        subjects=args.sessions or None,
        workers=args.workers,
        write_csv=args.write_csv,
    )
    print("frame_data 生成完成")
    print(f"session 数量：{len(results)}")
    print(f"总行数：{sum(item['rows'] for item in results)}")
    print(f"输出目录：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
