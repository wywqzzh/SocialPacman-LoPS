# 重构轮次 Intake

## 基本信息

- Run ID: 2026-05-04-generateGrammar
- 状态: intake
- 对应 Phase: Phase 2 - 重构 generateGrammar 模块

## 必填信息

- 目标脚本路径: `/home/zzh/project/Pacman/2.Pac-man/structre-learning/scripts/fmriDataProcess/generateGrammar.py`
- 运行环境: conda 环境 `fmri`
- 数据来源: 由目标脚本自行读取；本轮只关注默认运行 `main("ghost2", 0.5, False)` 使用到的数据。

## 运行方式

- 激活环境: `conda activate fmri`
- 运行命令:

```bash
python /home/zzh/project/Pacman/2.Pac-man/structre-learning/scripts/fmriDataProcess/generateGrammar.py
```

## 本轮重构范围

- 只重构默认运行 `main("ghost2", 0.5, False)` 实际使用到的功能和分支。
- 原脚本中默认运行不会走到的分支，本轮不分析、不迁移、不实现。
- 本轮重构不是简单搬运旧代码；需要基于原始行为重新设计模块结构、接口和实现。
- 新实现应高内聚低耦合，恪守 KISS 原则，优先直接、清晰、易维护。
- 避免过度工程化、过早抽象和不必要的防御性设计。

## 写入边界

- 允许读取目标脚本及其默认运行所需的依赖代码和数据。
- 禁止修改原始脚本。
- 禁止修改原始脚本所在文件夹中的任何内容。
- 除当前 LoPS 仓库外，禁止写入其它目录。
- 本轮所有新增分析、方案、实现、脚本、数据记录和验证记录都应写在当前 LoPS 仓库内。

## 原始行为记录

- 功能说明: 待深度分析。
- 输入: 待深度分析；目前已知入口参数为 `ghost2`、`0.5`、`False`。
- 输出: 待深度分析。
- 调用模块: 待深度分析。
- 依赖文件: 待深度分析。
- 使用数据: 待深度分析。
- 随机过程: 待深度分析。
- 副作用: 待深度分析。
- 工作目录假设: 待深度分析。

## 后续阶段记录

- 分析文档: 待生成。
- discuss 问题清单: 待生成。
- 重构方案: 待确认。
- 实施计划: 待确认。
- 验证结论: 待验证。

## 安全提醒

不要记录 API keys、密码、token 或私有凭据。
