# LoPS

LoPS 是 Social Pacman 双人/单人行为数据分析流程。仓库当前只保留一条正式的
`01 → 07` 数据链路，并使用嵌套目录 `data/<stage>/<task>/<session>.pkl` 保存结果。

## 正式数据流

| 阶段 | 入口 | 默认输入 | 默认输出 |
|---:|---|---|---|
| 01 | `script/01_mat_to_raw_subject_data.py` | `data/00_raw_mat_data` | `data/01_raw_subject_data` |
| 02 | `script/02_raw_subject_data_to_frame_data.py` | `data/01_raw_subject_data` | `data/02_frame_data` |
| 03 | `script/03_frame_data_preprocess.py` | `data/02_frame_data` | `data/03_preprocessed_frame_data` |
| 04 | `script/04_human_tile_data_preprocess.py` | `data/03_preprocessed_frame_data` | `data/04_tile_data`、`data/04_corrected_tile_data` |
| 05 | `script/05_calculate_utility.py` | `data/04_corrected_tile_data` | `data/05_utility_data` |
| 06 | `script/06_fit_context_strategy_posterior.py` | `data/05_utility_data` | `data/06_strategy_posterior_data` |
| 07 | `script/07_revise_context_strategy.py` | `data/06_strategy_posterior_data` | `data/07_revised_strategy_data` |

05 为每位玩家计算七种正式策略的四方向 raw Q，同时额外保存 Cluster Global、
逐 Energizer 和逐 Ghost Approach 候选 utility。正式 Q 与候选 utility 是不同算法；
06 在玩家 context 内先从三类候选中分别选择解释当前动作最好的目标，再拟合文件级
beta 并计算七策略 posterior。07 保留模型 posterior，另行写入事件规则修正后的策略
分数和标签。

## 环境与地图常量

`pyproject.toml` 当前只声明核心数值依赖。地图生成和视频入口还直接需要
`networkx`、`imageio` 与 `imageio-ffmpeg`；运行测试需要 `pytest`。在 LDS 环境中应
先确认这些包已经安装。08–12 历史阶段还额外使用 `scikit-learn`。

`data/constant_data/map_constants.pkl` 被 `data/.gitignore` 排除，不随仓库分发。首次
运行 04、05 或 tile 视频前，需要用当前内置地图重新生成：

```bash
PYTHONPATH=src python script/constant_map/generate_map_constants.py --workers 8
```

## 视频检查

当前只保留 tile 策略视频入口：

```bash
PYTHONPATH=src python script/pacman_video/run_tile_video_renderer.py \
  --task comp \
  --session <session>.pkl \
  --trial <DayTrial> \
  --save-frames
```

默认读取 `data/07_revised_strategy_data`，视频和图片分别写入
`data/pacman_video/tile_video` 与 `data/pacman_video/tile_frame_images`。
当前视频入口要求输入同时包含 P1/P2 的策略来源字段，因此只能直接绘制双人结果；
主分析流程本身仍支持不补 P2 列的单人文件。视频为完整显示 Ghost House 内部，会在
分析地图常量之外额外把固定鬼屋内部格子画成白色背景，该显示补充不会修改分析图结构。

## 运行约束

- Python 版本为 `>=3.10,<3.11`，当前数据使用 LDS 环境生成。
- 正式 01–07 链路使用 `task/session.pkl` 嵌套结构；单人数据不补空的 P2 字段。01、02
  底层仍保留扁平目录兼容分支，但该输出不能直接交给只接受嵌套目录的 03，因此不属于
  正式连续流程。
- 地图信息统一读取 `data/constant_data/map_constants.pkl`，业务阶段不得再次修正地图。
- 调试 05–07 时优先使用 `--single-file task/session.pkl`，验证后再执行目录级并行。
- 08–12 暂时保留供后续改造，但尚未形成从当前 07 输出开始的贯通流程。主要入口 08
  仍读取历史字段、扁平目录和两个旧 CSV 地图文件；09–12 已有部分当前结构接口，但
  仍依赖 08 的输出，不能视为当前正式链路。
- 09–11 的部分集成测试依赖被 `data/.gitignore` 排除的本地 fixture；fixture 不存在时
  pytest 会跳过这些用例。因此“其余测试通过”不能单独证明 08–12 链路已经贯通。

更具体的运行命令见 `data/README.md`；当前策略生成的精简说明见
`docs/current_strategy_generation.md`，完整后验细节见
`docs/context_strategy_posterior_inference.md`。
