# 技术栈研究

**领域：** 科研 Python 脚本重构与一致性验证  
**研究日期：** 2026-05-03  
**置信度：** MEDIUM

## 推荐技术栈

### 核心技术

| 技术 | 版本策略 | 用途 | 推荐理由 |
|------|----------|------|----------|
| Python | 跟随用户指定科研环境，项目代码建议支持 3.10+ | 模块化重构、脚本封装、验证工具 | 科研脚本生态通常以 Python 为核心，且本仓库已有 `src/LoPS` Python 包结构 |
| `pyproject.toml` | 采用 PyPA 当前标准 | 包元数据、构建系统、工具配置 | PyPA 将 Python Packaging User Guide 作为当前打包安装的权威资料，`pyproject.toml` 是现代项目配置中心 |
| pytest | 使用环境内可安装的稳定版本 | 单元测试、集成测试、验证夹具 | 官方 pytest 提供 `tmp_path`、`monkeypatch`、输出捕获等内建夹具，适合隔离运行旧脚本和新模块 |
| NumPy | 跟随目标脚本环境 | 数值数组、随机数、结果对比 | 多数科研脚本依赖 NumPy；官方随机 API 支持显式 seed 和独立 `Generator` |

### 支持库

| 库 | 版本策略 | 用途 | 何时使用 |
|----|----------|------|----------|
| pandas | 跟随目标脚本环境 | 表格数据读写和结果对比 | 原始脚本读写 CSV、Excel、DataFrame 时使用 |
| scipy | 跟随目标脚本环境 | 科学计算和统计函数 | 原始脚本已有 scipy 依赖时保持一致 |
| matplotlib | 跟随目标脚本环境 | 图形输出验证 | 原始脚本输出图片或科研图表时使用 |
| `pathlib` | Python 标准库 | 路径抽象 | 新代码内部优先使用，但执行外部命令时要注意 `Path("./cmd")` 归一化差异 |
| `subprocess.run` | Python 标准库 | 运行旧脚本和新脚本 | 官方建议在能覆盖的场景使用 `run()` 作为高层进程接口 |

### 开发工具

| 工具 | 用途 | 说明 |
|------|------|------|
| pytest fixtures | 构造临时目录、隔离环境变量、捕获输出 | `tmp_path` 适合每个测试独立的临时数据目录，`monkeypatch` 适合替换环境变量和路径 |
| git | 记录规划和重构提交 | 每个阶段提交对应产物，便于回滚和审阅 |
| 运行脚本 | 统一入口 | `script/` 中的脚本应能从整理后的 `data/` 读取输入并输出可比较结果 |

## 安装建议

LoPS 不应在项目初始化阶段强行锁定依赖版本。每一轮目标脚本可能来自不同科研环境，优先使用用户给出的 conda 或 virtualenv 环境；只有当某个模块被抽象成稳定包接口后，再补充项目级 `pyproject.toml` 和测试依赖。

```bash
# 后续需要时再添加
python -m pip install -e .
python -m pip install pytest
```

## 替代方案

| 推荐 | 替代 | 何时使用替代 |
|------|------|--------------|
| pytest | `unittest` | 目标环境无法安装第三方测试库时 |
| `subprocess.run` | 直接 `import` 旧脚本 | 旧脚本没有顶层副作用且函数边界清晰时 |
| `pathlib` | `os.path` | 需要逐字保留原脚本路径语义或处理 bytes path 时 |
| 跟随原环境依赖 | 统一升级依赖 | 只有在用户明确要求现代化依赖，并接受数值输出可能变化时 |

## 不建议使用

| 避免 | 原因 | 替代 |
|------|------|------|
| 先重写算法再补验证 | 容易失去旧实现基准，无法判断差异来自哪里 | 先采集旧输出，再小步重构 |
| 全局搜索替换路径字符串 | 数据路径含义通常和运行目录绑定，盲改会破坏脚本 | 设计显式路径参数和解析函数 |
| 无记录地删除临时旧代码 | 会丢失一致性验证证据 | 验证通过后删除，并在记录中说明对比结果 |
| 为所有脚本强制统一依赖版本 | 科研结果可能依赖旧库行为 | 每轮先跟随原环境，再评估是否升级 |

## 资料来源

- https://www.pypa.io/en/latest/index.html - PyPA 对 Python Packaging User Guide 的权威性说明
- https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml.html - `pyproject.toml` 构建系统说明
- https://docs.pytest.org/en/stable/reference/fixtures.html - pytest 内建 fixtures
- https://numpy.org/doc/stable/reference/random/generator.html - NumPy `Generator` 和 seed 行为
- https://docs.python.org/3/library/subprocess.html - `subprocess.run` 高层接口
- https://docs.python.org/3/library/pathlib.html - `pathlib` 与 `os.path` 差异

---
*Stack research for: LoPS 科研脚本重构*
*Researched: 2026-05-03*
