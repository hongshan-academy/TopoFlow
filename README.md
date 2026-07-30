# TopoFlow

基于遗传算法优化 DAG 拓扑流网络。给定目标吞吐率，自动搜索满足约束的最优图结构。

## 快速开始

```bash
# 安装依赖
uv sync

# 搜索最优拓扑（结果输出到 output/）
uv run search.py

# Web 可视化：手动编辑拓扑，实时求解/仿真
uv run uvicorn server:app --port 8080 --host 127.0.0.1

# 离线查看最优个体轨迹
./tools/ga_viewer.html

# 绘制搜索过程统计数据
uv run tools/plot.py
```

## 参数

编辑 `config.py` 中的 `DEFAULT_CONFIG`：

| 用途 | 字段 |
|---|---|
| 目标吞吐率 `p/q` | `target_pq: Tuple[int, int] = (p, q)` |
| 种群与代数 | `pop_size: int` / `generations: int` |
| 求解器模式 | `mode: Literal['MILP', 'simulation', 'mixed']` |
| 并行进程数 | `solver_workers: int` |
| 结果输出路径 | `output_path: str` |
