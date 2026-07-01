# External Integrations

**Analysis Date:** 2026-06-22

## APIs & External Services

**网络服务:**
- Not detected - 仓库代码未检测到 HTTP 客户端、云 SDK、远程数据库客户端、消息队列、Web API 回调或在线服务集成。
  - SDK/Client: Not applicable
  - Auth: Not applicable

**本地命令行工具:**
- `ffmpeg` - 用于把 Pacman JPG 帧合成为 MP4 视频。
  - SDK/Client: Python `subprocess.run`，实现位于 `src/LoPS/pacman_video/video_renderer.py`。
  - Auth: Not applicable
  - 入口：`script/pacman_video/run_video_renderer.py`。
  - 输入：`data/pacman_video/frame_images/{subject}/{game}/*.jpg`。
  - 输出：`data/pacman_video/video_data/{subject}/{game}.mp4`。
  - 编码参数：默认 `fps=30.0`、`crf=18`、`preset=medium`、`libx264`、`yuv420p`。

## Data Storage

**Databases:**
- Not detected - 没有 SQL/NoSQL 数据库、ORM、迁移工具或服务端持久化层。
  - Connection: Not applicable
  - Client: Not applicable

**File Storage:**
- Local filesystem only - 所有输入、输出、验证数据和静态文档均位于仓库本地目录。
  - 主数据根：`data/`。
  - 正式模块：`src/LoPS/`。
  - 运行入口：`script/`。
  - 文档：`README.md`、`data/README.md`、`docs/data_flow.html`、`.planning/preestimation_fmri_refactor_analysis.md`。

**Caching:**
- None - 未检测到 Redis、Memcached、磁盘缓存框架或持久缓存层。
- 运行时会在内存中复用编译后的地图结构，例如 `src/LoPS/hierarchical_utility/estimation.py` 的 `_CHUNK_COMPILED_MAP`，这是进程内计算优化，不是外部缓存集成。

## Scientific File Formats

**MATLAB / HDF5 `.mat`:**
- 原始输入位于 `data/00_raw_mat_data/`，当前检测到 1648 个 `.mat` 文件。
- 读取工具：`h5py.File`。
- 实现：`src/LoPS/pacman_preprocess/mat_to_raw_subject_data.py`。
- 数据语义：每个 session 目录包含多个 trial `.mat`，读取 HDF5 路径如 `data/pacMan/tile_x`、`data/ghosts/tile_x`、`data/direction/up`、`data/gameMap/currentTiles`，转换为逐帧 DataFrame。
- 输出关系：`data/00_raw_mat_data/{session}/*.mat` -> `data/01_raw_subject_data/{session}.pkl`。
- MATLAB 集成范围：只读取 MATLAB/HDF5 格式产物；仓库中未检测到 MATLAB Engine、`.m` 运行入口或调用 MATLAB 可执行文件。

**Pandas Pickle `.pkl`:**
- 主流程数据交换格式，使用 `pandas.read_pickle`、`DataFrame.to_pickle`、`pickle.load`、`pickle.dump`。
- 当前 `data/` 两层以内检测到 510 个 `.pkl` 文件，多数主流程阶段各有 34 个被试/session 文件。
- 关键读写模块：
  - `src/LoPS/pacman_preprocess/raw_subject_data_to_frame_data.py`
  - `src/LoPS/pacman_preprocess/frame_data_preprocess.py`
  - `script/04_human_tile_data_preprocess.py`
  - `src/LoPS/calculate_utility/processing.py`
  - `src/LoPS/hierarchical_utility/estimation.py`
  - `src/LoPS/dynamic_strategy_fitting.py`
  - `script/07_revise_human_weight.py`
  - `script/08_extract_features_human.py`
  - `script/09_human_fmri_data_preprocess.py`
  - `src/LoPS/state_dependency_graph.py`
  - `src/LoPS/generate_grammar/data.py`
- 命名约定：主分析数据文件统一使用 `{subject/session}.pkl`，说明见 `README.md` 和 `data/README.md`。

**CSV 常量与可选导出:**
- 地图常量：
  - `data/constant_data/adjacent_map_fmri.csv` - Pacman 迷宫四方向邻接表。
  - `data/constant_data/dij_distance_map_fmri.csv` - 位置间 Dijkstra 距离表。
- 读取位置：
  - `src/LoPS/hierarchical_utility/model.py`
  - `src/LoPS/dynamic_strategy_fitting.py`
  - `src/LoPS/calculate_utility/processing.py`
  - `script/04_human_tile_data_preprocess.py`
  - `script/08_extract_features_human.py`
- 可选 CSV 输出：`script/02_raw_subject_data_to_frame_data.py` 支持 `--write-csv`，默认输出目录为 `data/02_frame_data_csv`。

**Image / Video:**
- JPG/JPEG 帧由 `src/LoPS/pacman_video/frame_renderer.py` 使用 Pillow 生成，输出目录为 `data/pacman_video/frame_images`。
- MP4 视频由 `src/LoPS/pacman_video/video_renderer.py` 调用 `ffmpeg` 生成，输出目录为 `data/pacman_video/video_data`。
- 渲染表由 `src/LoPS/pacman_video/render_table.py` 从 `data/02_frame_data` 和 `data/pacman_video/grammar_data` 对齐生成，输出到 `data/pacman_video/render_data`。

**JSON:**
- `script/06_dynamic_strategy_fitting.py` 和 `script/12_divide_person.py` 使用 JSON 形式打印摘要或聚类结果；`script/12_divide_person.py` 默认只打印 JSON，不保存文件。
- 仓库中未检测到 JSON 数据文件作为主流程持久输入。

**HTML/Markdown:**
- `docs/data_flow.html` 是静态数据流程说明页，不依赖外部 JS/CSS CDN。
- `README.md` 和 `data/README.md` 记录主流程、视频流程和运行命令。

## Research Environment Integrations

**fMRI 行为数据:**
- 项目处理的是 Pacman fMRI 行为数据和派生特征。
- fMRI 相关常量集中在 `data/constant_data/adjacent_map_fmri.csv` 和 `data/constant_data/dij_distance_map_fmri.csv`。
- fMRI 处理链路：
  - `script/01_mat_to_raw_subject_data.py`：`.mat` -> raw subject `.pkl`。
  - `script/04_human_tile_data_preprocess.py`：生成 tile/corrected tile 数据并使用 fMRI 邻接表判断可行动作。
  - `script/05_calculate_utility.py`：根据 fMRI 地图常量计算 hierarchical utility。
  - `script/08_extract_features_human.py`：提取连续/离散特征。
  - `script/09_human_fmri_data_preprocess.py`：生成 ghost2 离散特征、formed 数据和 strategy sequence。

**NIfTI / neuroimaging volumes:**
- Not detected - 仓库未检测到 `.nii`、`.nii.gz`、`nibabel`、`nilearn`、`fslpy` 或体素/影像体数据处理代码。
- 当前 fMRI 范围是行为数据和迷宫/策略特征，不是脑影像体数据处理。

**FreeSurfer / FSL:**
- FreeSurfer integration: Not detected - 未检测到 `recon-all`、`SUBJECTS_DIR`、FreeSurfer Python SDK 或相关文件格式。
- FSL integration: Not detected as a data-processing dependency - 视频模块会检查系统 `ffmpeg`，代码中存在本地 FSL 风格的 `ffmpeg` 兜底路径常量，但没有调用 FSL 影像处理命令。

**MATLAB:**
- 输入数据来自 MATLAB `.mat` 产物；读取由 Python `h5py` 完成。
- 未检测到 MATLAB Runtime、MATLAB Engine、`.m` 文件或外部 MATLAB 命令调用。

## Authentication & Identity

**Auth Provider:**
- None - 本仓库没有用户认证、OAuth、API token、cookie、session 或身份服务逻辑。
  - Implementation: Not applicable

## Monitoring & Observability

**Error Tracking:**
- None - 未检测到 Sentry、OpenTelemetry、Datadog、Prometheus 或其它监控 SDK。

**Logs:**
- CLI stdout/stderr - 各脚本通过 `print` 输出阶段摘要、文件数量、进度和错误说明，例如 `script/01_mat_to_raw_subject_data.py`、`script/06_dynamic_strategy_fitting.py`、`src/LoPS/pacman_video/video_renderer.py`。
- 异常类型 - 部分模块定义领域错误以提供清晰失败信息，例如 `RawFmriError` 位于 `src/LoPS/pacman_preprocess/mat_to_raw_subject_data.py`，`VideoBuildError` 位于 `src/LoPS/pacman_video/video_renderer.py`。

## CI/CD & Deployment

**Hosting:**
- Not applicable - 本仓库是本地科研脚本重构与数据处理仓库，不包含 Web 服务或部署配置。

**CI Pipeline:**
- None detected - 未检测到 `.github/workflows/`、GitLab CI、CircleCI、Dockerfile 或 compose 部署配置。

## Environment Configuration

**Required env vars:**
- `PYTHONPATH=src` - README 中的运行命令使用该方式导入 `LoPS` 包；也可以通过 Poetry 安装本地包替代。
- Not detected - 没有必需的 API key、database URL、service account 或 secret 环境变量。

**Secrets location:**
- Not applicable - 未检测到 `.env`、credential、secret、key、pem 等密钥文件。
- 文档与代码中不应记录旧项目绝对数据路径细节；当前正式运行默认路径均指向仓库内 `data/`。

## Webhooks & Callbacks

**Incoming:**
- None - 没有 Web 服务入口、HTTP route、Webhook endpoint 或后台任务消费者。

**Outgoing:**
- None - 没有外发 HTTP 请求、Webhook 推送或远程事件上报。

## Data Flow Summary

**主分析链路:**
1. `data/00_raw_mat_data` -> `script/01_mat_to_raw_subject_data.py` -> `data/01_raw_subject_data`
2. `data/01_raw_subject_data` -> `script/02_raw_subject_data_to_frame_data.py` -> `data/02_frame_data`
3. `data/02_frame_data` -> `script/03_frame_data_preprocess.py` -> `data/03_preprocessed_frame_data`
4. `data/03_preprocessed_frame_data` + `data/constant_data/adjacent_map_fmri.csv` -> `script/04_human_tile_data_preprocess.py` -> `data/04_tile_data`、`data/04_corrected_tile_data`
5. `data/04_corrected_tile_data` + `data/constant_data` -> `script/05_calculate_utility.py` -> `data/05_utility_data`
6. `data/05_utility_data` + `data/constant_data/adjacent_map_fmri.csv` -> `script/06_dynamic_strategy_fitting.py` -> `data/06_weight_data`
7. `data/06_weight_data` -> `script/07_revise_human_weight.py` -> `data/07_corrected_weight_data`
8. `data/07_corrected_weight_data` + `data/constant_data` -> `script/08_extract_features_human.py` -> `data/08_feature_data`、`data/08_discrete_feature_data`
9. `data/08_discrete_feature_data` -> `script/09_human_fmri_data_preprocess.py` -> `data/09_fmri_discrete_feature_data_ghost2`、`data/09_fmri_formed_data_ghost2`、`data/09_strategy_sequence`
10. `data/09_strategy_sequence` -> `script/10_state_dependency_graph.py` -> `data/10_state_dependency_graph_data`
11. `data/09_strategy_sequence` + `data/10_state_dependency_graph_data` -> `script/11_generate_grammar.py` -> `data/11_grammar`
12. `data/11_grammar` -> `script/12_divide_person.py` -> stdout JSON

**视频链路:**
1. `data/02_frame_data` + `data/pacman_video/grammar_data` -> `script/pacman_video/run_render_table.py` -> `data/pacman_video/render_data`
2. `data/pacman_video/render_data` -> `script/pacman_video/run_frame_renderer.py` -> `data/pacman_video/frame_images`
3. `data/pacman_video/frame_images` + `ffmpeg` -> `script/pacman_video/run_video_renderer.py` -> `data/pacman_video/video_data`

---

*Integration audit: 2026-06-22*
