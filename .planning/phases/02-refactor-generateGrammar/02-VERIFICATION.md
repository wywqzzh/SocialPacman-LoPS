# Phase 2 generateGrammar 验证报告

**日期:** 2026-05-04  
**conda 环境:** `fmri`  
**结论:** 通过

## 运行命令

模块级行为测试：

```bash
PYTHONPATH=src conda run -n fmri python -m unittest discover -s tests
```

脚本级旧新一致性验证：

```bash
PYTHONPATH=src conda run -n fmri python script/validate_generate_grammar.py
```

## 数据路径

只读 StrategySequence 输入目录：

```text
/home/zzh/project/Pacman/2.Pac-man/Monkey_Analysis/fmri_data_process/StrategySequence/
```

只读 StateGraph 输入目录：

```text
/home/zzh/project/Pacman/2.Pac-man/Monkey_Analysis/fmri_data_process/StateGraph/
```

固定旧基准目录：

```text
/home/zzh/project/Pacman/2.Pac-man/Monkey_Analysis/fmri_data_process/grammar2/
```

新输出目录：

```text
data/generate_grammar/refactored-output/grammar2/
```

## 模块级行为测试结论

`unittest discover` 结果：

```text
Ran 18 tests in 1.748s
OK
```

已覆盖：

- foundation：配置、token、StrategySequence 读取、StateGraph 读取。
- scoring：`Utils.count`、`BDscore`、`learnBayesNetBlock` 的旧新模块级行为对照。
- grammar：最长匹配解析、概率统计、候选筛选、skip-gram、真实文件 learn 调用。
- pipeline：数据准备、`legacy`/`structured` 输出结构。
- validation：精确比较工具和代表性文件 legacy key 覆盖。

## 脚本级一致性结论

验证脚本结果：

```text
Validation passed for 34 files.
```

验证方式：

- 使用新实现重新生成 34 个输出文件。
- 对每个新输出文件读取顶层 `legacy` 字典。
- 对旧 `grammar2/` 基准中存在的每个 key，检查新 `legacy` 中同名 key 存在。
- 对每个旧 key 的 value 做逐 key/value 精确比较：
  - `np.ndarray` 使用 `np.array_equal()`。
  - `pd.DataFrame` 使用 `pd.testing.assert_frame_equal(..., check_exact=True)`。
  - `list` / `tuple` 逐项递归比较。
  - `dict` 逐 key 递归比较。
  - `float` / `int` / `bool` / `str` 使用 `==` 精确比较。
- 未使用数值容差。

## MD5 说明

旧基准可信性已在分析阶段确认：原始脚本在 LoPS sandbox 中全量重跑后，34/34 输出与既有 `grammar2/` 文件 MD5 完全一致。

本次新输出 pickle 顶层结构为：

```python
{
    "legacy": {...},
    "structured": {...},
}
```

因此新 pickle 文件整体不以旧 pickle 文件级 MD5 一致作为通过条件。本轮通过标准是旧输出存在的全部字段在新输出 `legacy` 中逐 key/value 精确一致；该验证已通过。

## 随机过程

默认路径为 `main("ghost2", 0.5, False)`，本轮有效分支无随机过程。

本次验证没有人为设置随机种子；未来若支持 `needShuffle=True`，需要另行设计 seed 接口并重新讨论验证规则。

## 临时代码清理

`src/LoPS/temp` 检查结果：

```text
find src/LoPS/temp -mindepth 1 | wc -l
0
```

本轮没有保留临时旧实现副本，`src/LoPS/temp` 无本轮临时代码残留。
