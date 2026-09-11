# TopoFlow 综合工具

拓扑流网络的一体化工作台：浏览器中可视化编辑图（In / Out / 分流器 S / 汇流器 C），
调用多种流量求解器、离散事件仿真、精确构造与物理布局。

## 功能

- **图形编辑**：放置/连接 In、Out、S、C 节点，支持 WG 文本导入导出、JSON 存档、WASD 平移画布。
- **流量求解**（全部基于唯一内嵌的 Z3 引擎，输出精确有理数）：
  - `MILP (Z3, exact)`：三态 MILP 多解求解（`solver.py`，经 `ilpbridge.py` 调用 Z3）。
  - `Rust - Z3 (rank-smt)`：基于 rank 的方法，SMT 辅助边状态搜索（`native/src/rank_smt.rs`）。
  - `Rust - Karzanov (stable)`：Rust 原生稳定分配求解器，聚合 LP 由内置确定性精确单纯形求解（`native/src/exact_lp.rs`）。
- **离散仿真**：逐帧回放传送带队列与节点占用（`simulator.py`，边容量 4）。
- **精确构造**：按目标分数 `p/q` 精确构造阻塞流拓扑，输出规约步骤与满秩证书（`constructor/`）。
- **模块生成**：标准分流模块（`p:q` 二分）与标准限流模块（`p/q ∈ (1/3, 1/2)`）。
- **物理布局**：内置布局求解器（Z3，经 `ilpbridge.py`）搜索最小可行网格并可视化（`layout/`）。

## 求解架构

全项目只依赖一个求解引擎：**Z3**（静态编译进 Rust 扩展 `topoflow_native`，全项目共享一份编译产物）。

- Rust 原生 `rank-smt` 直接使用 Z3；`stable` 的聚合 LP 由内置的确定性精确单纯形（`exact_lp.rs`）求解。
- Python 侧（精确 MILP、构造器排列 MILP、物理布局模型）统一经 `ilpbridge.py` 调用
  `topoflow_native.solve_ilp_exact`，已移除 HiGHS / OR-Tools。

## 快速开始

### 依赖

启动服务器前，需要安装 uv tool-chain（如已安装，请跳过）

**Linux 和 macOS**
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```
或
```sh
wget -qO- https://astral.sh/uv/install.sh | sh
```

**Windows**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 启动服务器

**脚本启动**
```powershell
./server.ps1
```

随后访问 [http://127.0.0.1:8081](http://127.0.0.1:8081) 即可

**手动启动**

```powershell
uv sync
uv run uvicorn server:app --port 8080 --host 127.0.0.1
```

> Rust 扩展通过 maturin 构建，需要 Rust 工具链与 LLVM/libclang。
> 若 LLVM 安装在 `C:\Program Files\LLVM`，`server.ps1` 会自动设置 `LIBCLANG_PATH`。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/config` | 仿真配置（最大帧数） |
| GET | `/api/solvers` | 可用求解器列表 |
| POST | `/api/solve` | MILP 多解求解 |
| POST | `/api/simulate` | 离散事件仿真（返回逐帧状态） |
| POST | `/api/solve-native` | Rust 原生求解（`engine=rank-smt` / `stable`，精确有理数输出） |
| POST | `/api/ratio-split` | 标准分流模块生成 |
| POST | `/api/limit-module` | 标准限流模块生成 |
| POST | `/api/construct` | 精确构造 `p/q` 阻塞流 |
| POST | `/api/topoflow-layout` | 物理布局（NDJSON 流式进度） |


## 测试

```powershell
uv run pytest
uv run mypy .
uvx pyright
```

## 各部分来源（不分先后）

| 部分 | 来源 |
|---|---|
| 网页编辑器、离散 simulator | @Fatal Error A1012 |
| MILP 流量求解器（`solver.py`） | @madSUNitist |
| Rust 原生 `rank_smt`、`stable_polynomial`（`native/`）流量求解器 | @恒星泰斗 |
| 精确 `p/q` 构造（`constructor/`） | @Orirock @madSUNitist 等 |
| 物理布局求解器（`layout/`） | @kokobird |
| 标准术语与工具综合 | @jnk |
