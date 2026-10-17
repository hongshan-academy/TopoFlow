# EnKA 蓝图导出

物理布局结果可通过网页中的「复制 EnKA 蓝图」「下载 EnKA 蓝图」导出。
主工作台和布局编辑器都提供入口。复制完整字符串，在 EnKA Beta 的空白可编辑
画布 Ctrl+V，然后点击放置。此字符串不是游戏内分享码。

## 输入和输出

输入必须是带 `topoflow` 元数据的布局结果，或主工作台下载的
`{"solution": ...}` 包装。逻辑图、预览坐标不能代替物理布局。

```http
POST /api/export/enkad
Content-Type: application/json

{
  "solution": {"n": 3, "m": 3, "machs": [], "belts": [], "topoflow": {}},
  "control_items": {},
  "full_rate_per_minute": "30",
  "offset": [8, 8]
}
```

上面的 solution 仅展示请求形状，需替换成实际求解结果。一个无需求解器的几何
输入示例在 `examples/enkad-cu2-layout.json`：左输入、CU2、右输出。
CU 需要配置 `control_items: {"CU2_0": <真实物品ID>}`。
`item_id` 可作为所有 CU 的共同默认物品；逐节点配置优先。
物品编号必须由调用者从当前 EnKA 目录确认，接口不能校验它代表什么。

响应字段：

- `blueprint`：标准 Base64 编码的 EnKA v4 原生蓝图。
- `layout`：编码前的原生实体结构，包含地面层或管道层。
- `manifest`：原节点到 UID 的映射、整体偏移、外部端点、原路线、分段 UID、直接连接。
- `warnings`：尚未验证的运行时行为与物品编号要求。

几何错误返回 HTTP 400，参数类型校验错误返回 HTTP 422。
外部端点不自动生成仓库或供给设施。粘贴平移后，对 manifest 坐标应用同一偏移。

## 命令行

从仓库根目录运行，无需 FastAPI、Z3、MCP 或相邻 enkad_mcp 仓库：

```text
python -m layout.enkad_export topoflow-layout.json -o blueprint.txt
python -m layout.enkad_export examples/enkad-cu2-layout.json -o blueprint.txt --item-id <已核实的物品ID>
```

第二条命令中的尖括号内容是占位符，应替换成整数。会同时写出
`blueprint.manifest.json`，包含原生布局、端点和验证提示。
不同 CU 使用不同物品时，使用 Python `export_layout(..., control_items=...)`
或 HTTP 接口。管道 CU 必须明确设置 `full_rate_per_minute`；CLI 用
`--full-rate-per-minute`。

## 转换约定

- `[row, column]` 转成 `(x, y)`，行列均保持一格对一格，只加整体偏移。
- 无序网格边从源端口重建有序路径，拒绝重复边、断链、端口不匹配和非法占格。
- S2/S3 映射为三出口分流器；C2/C3 映射为三入口合流器，未使用端口留空。
- S/CU 旋转为 `(180 - 90*rot) % 360`；C 旋转为 `(-90*rot) % 360`。
- 正交直线交叉生成显式桥，分割两条路线；相邻设施/桥直接连接，不插入空格。
- 外部 anchor 若存在，沿端口方向延伸至边界外一格；无 anchor 时只记录设施
  外侧一格的待接端点。若 anchor 不与端口对齐或线路被占用则拒绝导出。
- 按设施输入顺序、桥坐标顺序、路线顺序分配唯一 UID；不承诺等价于手动建设顺序。

目录编号核对于 2026-10-18 EnKA Beta 的 `CN3EVTOM.js`：

| 类别 | 分流器 | 合流器 | 控制口 | 桥 | 路径 |
|---|---:|---:|---:|---:|---:|
| 物品 | 35 | 34 | 37 | 36 | 33 |
| 管道 | 39 | 38 | 41 | 40 | 42 |

这不是完整目录，网站升级后需再次核实。当前不支持混合运输图或非 1×1 设施。

## CU 和验证边界

见 [控制口核对记录](enkad-control-port-notes.md)。物品带缺省满速基准为 30/m，
CU1..CU4 的 value 为 12/24/36/48；输出物品写入 port_index=0。
value=0 表示默认满速，counter 缺省 0 表示无累计数量截止。

目前已测试原生字段解码、旋转、图层、端口、路径重建及非法几何拒绝。
尚未验证浏览器实际粘贴、EnKA 仿真吞吐或 CU 的五 tick 窗口等价性。
网页几何预览成功和蓝图解码成功不能代替实际仿真。

```text
python -m pytest tests/test_enkad_export.py tests/test_layout_control_unit.py
```

HTTP 测试需要可导入服务器及其依赖；原生布局测试需要已构建的 Z3 扩展。
