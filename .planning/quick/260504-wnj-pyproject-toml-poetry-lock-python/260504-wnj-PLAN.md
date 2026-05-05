---
quick_id: 260504-wnj
slug: pyproject-toml-poetry-lock-python
status: complete
created: 2026-05-04
completed: 2026-05-05
resolution: superseded
---

# Quick Task 260504-wnj: 修正 pyproject.toml 和 poetry.lock 模板内容

## 目标

用户已从其它项目复制 `pyproject.toml` 和 `poetry.lock`，当前文件仍包含旧项目元数据和依赖。需要将它们改成 LoPS 当前项目可用的最小配置，保持 Python 版本要求不变，依赖暂不声明。

## 任务

1. 修改 `pyproject.toml`：
   - 项目名改为 `lops`。
   - 描述改为当前 LoPS 项目语义。
   - Python 版本要求保持 `>=3.10,<3.11`。
   - `dependencies` 置为空列表。
   - package include 改为 `src/LoPS`。
   - 删除旧项目 `bayesianbrain` 相关内容和模板依赖组。

2. 重新生成 `poetry.lock`：
   - 使用当前 `pyproject.toml`。
   - 锁文件不保留旧模板项目依赖。

3. 验证：
   - `poetry check` 通过。
   - `poetry lock` 能基于当前配置生成锁文件。
   - `pyproject.toml` 和 `poetry.lock` 中不再包含 `bayesianbrain`。

## 关闭说明

该 quick 任务在执行过程中被后续依赖识别任务替代，用户明确说明“上面任务已经完成不需要继续”。本记录按已关闭处理，不再作为 v1 里程碑的开放工作项。
