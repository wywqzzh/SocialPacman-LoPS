# Technology Stack

**Analysis Date:** 2026-06-22

## Languages

**Primary:**
- Python 3.10 - 项目正式运行版本由 `pyproject.toml` 的 `requires-python = ">=3.10,<3.11"` 约束；`poetry env info` 显示当前 Poetry 虚拟环境为 CPython 3.10.20，路径为仓库内 `.venv`。

**Secondary:**
- HTML/CSS - 静态人工文档 `docs/data_flow.html`，用于说明 Pacman 数据流程。
- Markdown - 项目说明与流程说明位于 `README.md`、`data/README.md`、`.planning/preestimation_fmri_refactor_analysis.md`。
- Shell 命令 - 运行说明使用 `PYTHONPATH=src python ...` 形式，集中记录在 `data/README.md`。

## Runtime

**Environment:**
- Python 运行环境：Poetry 管理的本地虚拟环境 `.venv`，有效 Python 版本为 3.10.20。
- 本机当前 shell 的 `python --version` 为 3.12.11，不符合 `pyproject.toml` 的版本约束；运行项目命令应使用 Poetry 虚拟环境或显式使用 Python 3.10。
- `.vscode/settings.json` 将默认环境管理器设为 `ms-python.python:conda`，默认包管理器也设为 `conda`；仓库实际包元数据仍由 `pyproject.toml` 和 `poetry.lock` 管理。

**Package Manager:**
- Poetry 2.2.1 - 本机可用版本由 `poetry --version` 确认。
- Lockfile: present，`poetry.lock` 锁定了 `h5py` 3.16.0、`numpy` 2.2.6、`pandas` 2.3.3、`pillow` 11.3.0、`scikit-opt` 0.6.6、`scipy` 1.15.3 等依赖。
- 包安装配置：`pyproject.toml` 的 `[tool.poetry] packages = [{ include = "LoPS", from = "src" }]`，正式模块根目录是 `src/LoPS/`。

## Frameworks

**Core:**
- Python standard library `argparse` - 所有主入口脚本通过命令行参数运行，例如 `script/01_mat_to_raw_subject_data.py`、`script/06_dynamic_strategy_fitting.py`、`script/11_generate_grammar.py`。
- Python standard library `concurrent.futures.ProcessPoolExecutor` / `multiprocessing` - 数据转换、utility 计算和动态策略拟合使用进程级并行，相关文件包括 `src/LoPS/pacman_preprocess/mat_to_raw_subject_data.py`、`src/LoPS/hierarchical_utility/estimation.py`、`src/LoPS/dynamic_strategy_fitting.py`、`script/08_extract_features_human.py`。
- `dataclasses` - 核心配置和数据结构使用 dataclass，例如 `src/LoPS/hierarchical_utility/model.py` 的 `MapData`、`CompiledMapData`、`UtilityConfig`、`FrameState`，以及 `src/LoPS/dynamic_strategy_fitting.py` 的 `DynamicStrategyFittingConfig`。

**Testing:**
- `unittest` - 多数测试文件使用 Python 标准库 `unittest`，例如 `tests/test_generate_grammar_foundation.py`、`tests/test_generate_grammar_grammar.py`、`tests/test_pacman_frame_data.py`。
- `pytest` - `tests/test_generate_grammar_divide_person.py` 导入 `pytest`，但 `pyproject.toml` 和 `poetry.lock` 未声明或锁定 `pytest`。

**Build/Dev:**
- `poetry-core` - 构建后端配置在 `pyproject.toml` 的 `[build-system]`。
- `PYTHONPATH=src` - README 和 `data/README.md` 中的运行命令默认通过 `PYTHONPATH=src` 暴露 `LoPS` 包。
- `docs/data_flow.html` - 静态 HTML/CSS 文档，不需要前端构建工具。

## Key Dependencies

**Critical:**
- `numpy>=2.1,<3.0` - 数组计算、Q 向量、状态矩阵、随机选择和特征矩阵；用于 `src/LoPS/hierarchical_utility/strategies.py`、`src/LoPS/dynamic_strategy_fitting.py`、`src/LoPS/structure_learning.py`、`script/09_human_fmri_data_preprocess.py`。
- `pandas>=2.3,<3.0` - 主要表格数据层；所有阶段的 `.pkl` 读写、CSV 常量读取、DataFrame 处理都依赖 Pandas，例如 `src/LoPS/pacman_preprocess/raw_subject_data_to_frame_data.py`、`script/04_human_tile_data_preprocess.py`、`src/LoPS/generate_grammar/data.py`。
- `scipy>=1.15,<2.0` - 结构学习中的 Dirichlet-multinomial 边际似然使用 `scipy.special.gammaln`，实现位于 `src/LoPS/structure_learning.py`。
- `h5py>=3.12,<4.0` - 读取 MATLAB/HDF5 `.mat` 原始 fMRI 行为数据，核心入口是 `src/LoPS/pacman_preprocess/mat_to_raw_subject_data.py`。
- `scikit-opt>=0.6,<0.7` - 动态策略拟合的遗传算法依赖 `from sko.GA import GA`，使用点在 `src/LoPS/dynamic_strategy_fitting.py`。
- `pillow>=10.0,<12.0` - Pacman 视频流程中渲染 JPG 帧，使用点在 `src/LoPS/pacman_video/frame_renderer.py`。

**Infrastructure:**
- `scikit-learn` - 实际代码直接使用 `sklearn.neighbors.NearestNeighbors` 和 `sklearn.cluster.AgglomerativeClustering`，位于 `script/09_human_fmri_data_preprocess.py` 和 `src/LoPS/generate_grammar/grammar_process.py`；该依赖当前未在 `pyproject.toml` 或 `poetry.lock` 中声明。
- `ffmpeg` - 视频合成依赖系统可执行文件，定位与调用逻辑在 `src/LoPS/pacman_video/video_renderer.py`，包装入口在 `script/pacman_video/run_video_renderer.py`。
- `pickle` / `pandas.read_pickle` / `DataFrame.to_pickle` - 主分析数据交换格式；当前 `data/` 下主流程每个阶段通常有 34 个 `.pkl` 文件。
- `csv` via `pandas.read_csv` - 地图常量文件 `data/constant_data/adjacent_map_fmri.csv` 和 `data/constant_data/dij_distance_map_fmri.csv`。

## Configuration

**Environment:**
- Python 版本约束在 `pyproject.toml`，必须使用 Python `>=3.10,<3.11`。
- 运行命令通常需要 `PYTHONPATH=src`，见 `data/README.md` 中 12 步主流程命令。
- `.vscode/settings.json` 指向 Conda 环境管理，但不包含项目运行参数或密钥。
- 未检测到 `.env`、credential、secret、key、pem 等环境密钥文件。

**Build:**
- `pyproject.toml` - 项目元数据、Python 版本、依赖和包路径。
- `poetry.lock` - 锁定正式声明依赖及其传递依赖。
- 未检测到 `pytest.ini`、`tox.ini`、`noxfile.py`、`ruff.toml`、`eslint.config.*`、`prettier` 或其它格式化/静态检查配置。

## Command Entrypoints

**主分析流程:**
- `script/01_mat_to_raw_subject_data.py`：默认输入 `data/00_raw_mat_data`，输出 `data/01_raw_subject_data`；通过 `h5py` 读取每个 session 下的 `.mat` trial。
- `script/02_raw_subject_data_to_frame_data.py`：默认输入 `data/01_raw_subject_data`，输出 `data/02_frame_data`；可通过 `--write-csv` 额外输出到 `data/02_frame_data_csv`。
- `script/03_frame_data_preprocess.py`：默认输入 `data/02_frame_data`，输出 `data/03_preprocessed_frame_data`。
- `script/04_human_tile_data_preprocess.py`：默认输入 `data/03_preprocessed_frame_data` 和 `data/constant_data/adjacent_map_fmri.csv`，输出 `data/04_tile_data`、`data/04_corrected_tile_data`。
- `script/05_calculate_utility.py`：默认输入 `data/04_corrected_tile_data` 和 `data/constant_data`，输出 `data/05_utility_data`；调用 `src/LoPS/calculate_utility/` 与 `src/LoPS/hierarchical_utility/`。
- `script/06_dynamic_strategy_fitting.py`：默认输入 `data/05_utility_data` 和 `data/constant_data/adjacent_map_fmri.csv`，输出 `data/06_weight_data`；默认 seed 为 `20260610`。
- `script/07_revise_human_weight.py`：默认输入 `data/06_weight_data`，输出 `data/07_corrected_weight_data`。
- `script/08_extract_features_human.py`：默认输入 `data/07_corrected_weight_data` 和 `data/constant_data`，输出 `data/08_feature_data`、`data/08_discrete_feature_data`。
- `script/09_human_fmri_data_preprocess.py`：默认输入 `data/08_discrete_feature_data`，输出 `data/09_fmri_discrete_feature_data_ghost2`、`data/09_fmri_formed_data_ghost2`、`data/09_strategy_sequence`。
- `script/10_state_dependency_graph.py`：默认输入 `data/09_strategy_sequence`，输出 `data/10_state_dependency_graph_data`。
- `script/11_generate_grammar.py`：默认输入 `data/09_strategy_sequence` 和 `data/10_state_dependency_graph_data`，输出 `data/11_grammar`。
- `script/12_divide_person.py`：默认输入 `data/11_grammar`，输出到 stdout 的 JSON，不保存文件。

**视频流程:**
- `script/pacman_video/run_render_table.py`：默认输入 `data/02_frame_data` 和 `data/pacman_video/grammar_data`，输出 `data/pacman_video/render_data`。
- `script/pacman_video/run_frame_renderer.py`：默认输入 `data/pacman_video/render_data`，输出 `data/pacman_video/frame_images` 的 JPG 帧。
- `script/pacman_video/run_video_renderer.py`：默认输入 `data/pacman_video/frame_images`，输出 `data/pacman_video/video_data` 的 MP4 文件；依赖系统 `ffmpeg`。

## Data Dependencies

**仓库内数据结构:**
- `data/00_raw_mat_data/`：原始 fMRI Pacman `.mat` 数据；当前检测到 1648 个 `.mat` 文件，按 session 子目录组织。
- `data/01_raw_subject_data/` 到 `data/11_grammar/`：主流程阶段产物；当前多数阶段每个目录 34 个 `.pkl` 文件。
- `data/constant_data/`：fMRI 迷宫常量表，当前包含 `data/constant_data/adjacent_map_fmri.csv` 和 `data/constant_data/dij_distance_map_fmri.csv`。
- `data/pacman_video/`：视频流程输入与输出；当前检测到 `render_data` 等目录中共 35 个 `.pkl` 和 1 个 `.mp4` 文件。

**数据输入输出关系:**
- 原始 `.mat` trial 数据由 `src/LoPS/pacman_preprocess/mat_to_raw_subject_data.py` 转换为逐帧 subject `.pkl`。
- subject `.pkl` 由 `src/LoPS/pacman_preprocess/raw_subject_data_to_frame_data.py` 转换为 frame data `.pkl`，可选 CSV。
- frame data 经过预处理、tile 抽样、路径修正、utility 计算、动态策略拟合、权重修正、特征提取、fMRI formed 数据整理、状态依赖图和 grammar 学习，最终进入 `data/11_grammar/`。
- 视频流程从 `data/02_frame_data` 与 `data/pacman_video/grammar_data` 合并渲染表，再输出 JPG 帧和 MP4。

## Platform Requirements

**Development:**
- Python 3.10.x。
- Poetry 可用于创建和维护 `.venv`。
- `PYTHONPATH=src` 或已安装本地包，确保能导入 `LoPS`。
- 对完整数据流程，需有足够磁盘与 CPU 资源；多个入口默认使用 8 到 34 个进程。
- 若运行 `script/09_human_fmri_data_preprocess.py` 或 `script/12_divide_person.py`，需要安装当前未声明的 `scikit-learn`。
- 若运行测试，`tests/test_generate_grammar_divide_person.py` 需要 `pytest`，但当前未声明。

**Production:**
- Not applicable；当前仓库是本地科研数据处理与验证项目，没有 Web 服务、守护进程或部署目标。
- 视频导出需要系统 `ffmpeg` 支持 H.264/libx264 编码。

---

*Stack analysis: 2026-06-22*
