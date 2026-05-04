# generateGrammar 数据来源记录

本轮不复制原始 `StrategySequence`、`StateGraph`、`grammar2` 数据到 LoPS 仓库。

原因：

- 数据位于外部 Pacman 项目中，体积较大。
- 本轮重构只读使用这些数据，不需要在 LoPS 中维护副本。
- 复制数据会造成重复存储和版本混乱，也会增加后续验证时的数据来源歧义。

## 只读输入目录

`StrategySequence`：

```text
/home/zzh/project/Pacman/2.Pac-man/Monkey_Analysis/fmri_data_process/StrategySequence/
```

`StateGraph`：

```text
/home/zzh/project/Pacman/2.Pac-man/Monkey_Analysis/fmri_data_process/StateGraph/
```

## 固定旧基准目录

`grammar2`：

```text
/home/zzh/project/Pacman/2.Pac-man/Monkey_Analysis/fmri_data_process/grammar2/
```

该目录已在 Phase 2 分析阶段通过原始脚本 sandbox 全量重跑验证：34/34 个输出文件与既有基准 MD5 完全一致，因此作为本轮固定旧基准。

## 新输出目录

默认新输出写入 LoPS 仓库内：

```text
.planning/runs/2026-05-04-generateGrammar/refactored-output/grammar2/
```

所有验证输出和报告只写入 LoPS 仓库，不写入外部 Pacman 项目。
