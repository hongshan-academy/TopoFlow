/* Shared by the main workbench and the embedded physical layout editor. */
window.exportEnkad = async function (solution, copy = false) {
  if (!solution) throw new Error("请先生成物理布局结果");
  const cus = (solution.machs || []).filter(m => /^(?:CU|ICU)[1-4]$|^PCU[1-9]$/.test(m.topoflow_node_type || ""));
  const controlItems = {};
  for (const machine of cus) {
    const value = prompt(`${machine.topoflow_node_id}：请输入当前 EnKA 目录中的物品/流体整数 ID（不会自动核实物品）。`);
    if (value === null) return;
    if (!/^\d+$/.test(value.trim()) || Number(value) < 1 || Number(value) > 4294967295) {
      throw new Error("物品 ID 必须为 1–4294967295 的整数");
    }
    controlItems[machine.topoflow_node_id] = Number(value);
  }
  let baseline = null;
  if (cus.length) {
    baseline = prompt("准入口归一化满速基准（每分钟）。物品 ICU/CU 通常为 30；管道 PCU 通常为 120。仅映射平均流量，五 tick 行为尚未验证。",
                      solution.topoflow?.transport === "belt" ? "30" : "120");
    if (baseline === null) return;
    if (!baseline.trim()) throw new Error("请填写满速基准");
  }
  const response = await fetch("/api/export/enkad", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({solution, control_items: controlItems, full_rate_per_minute: baseline})
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "EnKA 导出失败");
  const download = () => {
    const url = URL.createObjectURL(new Blob([result.blueprint + "\n"], {type: "text/plain;charset=utf-8"}));
    const link = document.createElement("a");
    link.href = url; link.download = "topoflow.enkad.txt";
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  let message = "已下载 EnKA 蓝图";
  if (copy) {
    try {
      await navigator.clipboard.writeText(result.blueprint);
      message = "已复制 EnKA 蓝图，在可编辑画布中 Ctrl+V 放置";
    } catch {
      download();
      message = "剪贴板不可用，已下载 TXT，请复制文件内容";
    }
  } else download();
  alert([message, ...result.warnings].join("\n"));
};
