# TopoFlow

基于遗传算法优化 DAG 拓扑流网络。给定目标吞吐率，自动搜索满足约束的最优图结构。

## 快速开始

```bash
# 安装依赖
uv sync

# 搜索最优拓扑
uv run search.py

# 启动 Web 可视化
uv run uvicorn server:app --port 8080 --host 127.0.0.1
```


## 关键参数

```python
# config.py
pop_size = 100          # 种群大小
generations = 2000      # 代数
mutation_rate = 1.0     # 变异概率
eval_timeout = 10.0     # 每代求值超时（秒），慢个体跨代保留
tournament_size = 2     # 锦标赛规模
elitism_count = 2       # 精英保留数
immigration_rate = 0.05 # 每代注入新个体比例
solver_workers = 16     # 并行求解进程数
```

突变权重：边删 0.225 / 点删 0.225 / 边增 0.225 / 点增 0.225 / 子图替换 0.1。
