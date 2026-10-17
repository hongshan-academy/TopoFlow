# EnKA Beta 控制口核对

核对日期：2026-10-18。以下为网页脚本静态核对结果，尚未进行运行时仿真对比。

## 来源

相对于 `https://beta.enka.network/endfield/aic/_app/immutable/chunks/`：

- `CN3EVTOM.js`：`idToInt`、`log_conditioner`、`log_pipe_conditioner`、
  Port.direction、Building.serialize/deserialize、端口数组构造。
- `DCGiD871.js`：向量 `rotateDeg`。
- `BoupM2MD.js`：控制口速率滑块 `toDisplay`、`toBacking`、档位提示。

## 几何

| 设施 | 内部名称 | entity_id | 图层 | 占地 |
|---|---|---|---|---|
| 物品控制口 | log_conditioner | 37 | Root\|Ground = 3 | 1×1 |
| 管道控制口 | log_pipe_conditioner | 41 | Root\|Pipe = 5 | 1×1 |

两者都是一个输入、一个输出，都在实体所在格。输入和输出的数据 rotation.y
均为 180；这是物品运动方向，不能直接理解为端口向外的方向。
`direction = (0,1).rotateDeg(180).rotateDeg(building.rotation)`。
输入外侧格为 `cell - direction`，输出外侧格为 `cell + direction`。

| EnKA rotation | 输入外侧 | 输出外侧 | 流向 |
|---|---|---|---|
| 0 | 下 | 上 | 向上 |
| 90 | 左 | 右 | 向右 |
| 180 | 上 | 下 | 向下 |
| 270 | 右 | 左 | 向左 |

TopoFlow CU 模板 rotation=0 时上进下出，方向枚举为下、右、上、左，
因此 TopoFlow rot 0/1/2/3 对应 EnKA rotation 180/90/0/270：
`(180 - 90 * rot) % 360`。此表只针对控制口，不要套用到其他设施。

## 速率与物品

蓝图 `building.value` 为整数：非零时速率 = value / 2（件或流体单位/分钟）。
UI 的实际公式：

```text
toDisplay(value): value == 0 ? (pipe ? 120 : 30) : value / 2
toBacking(rate): rate == (pipe ? 120 : 30) ? 0 : floor(rate * 2)
```

- `value=0` 代表默认满速，不代表关闭。
- 物品控制口 UI 范围为 0.5–30/m，步长 0.5；标注游戏档位为 6/12/18/24/30。
- 管道控制口 UI 范围为 1–120/m，步长 1；标注游戏档位数组为 6..60、步长 6。
  UI 允许范围不能直接等同于游戏中可设置的范围。
- `counter` 为累计数量限制，UI 中 0 表示无限；保持 0 可避免引入数量截止。
- 没有选定输出物品时，UI 禁用速率编辑。导出必须提供适当的物品 ID，
  不能假设它能不分物品地透明限流。
- 原始端口侧 index 都为 0，但序列化的 `port_index` 是组合端口数组的下标。
  构造器先合并端口，再按输出优先排序。单入单出控制口的输出物品应写入
  `port_items: [{port_index: 0, item_id: <已核实的物品ID>}]`。

若把 TopoFlow 的归一化满速明确设为物品带 30/m，则平均速率映射为：

| TopoFlow | 目标速率 | value |
|---|---|---|
| CU1 | 6/m | 12 |
| CU2 | 12/m | 24 |
| CU3 | 18/m | 36 |
| CU4 | 24/m | 48 |

这只是平均速率映射，尚未证明 EnKA 的接收节拍、启动和阻塞行为等同于
TopoFlow 的“首件触发五 tick 窗口，每窗口接收最多 g 件”。两者 tick
不能直接等同。管道速率应另行定义归一化基准，不机械套用物品带的值。

## 后续导出要求

1. 使用以上独立旋转映射，保留一格对一格的坐标。
2. 明确运输类型、物品 ID、归一化速率基准；禁止猜测物品 ID。
3. 扩展现有编码器的建筑图层支持后，才能导出管道控制口。
4. 用持续满供给和下游畅通的小网络验证平均吞吐，再对比启动、阻塞恢复。
5. 在完成该验证前，标注为“几何和字段映射已核对，动态等价性未验证”。
