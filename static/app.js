/**
 * 前端应用 —— 通过 Fetch API 与 Python 后端通信
 * 仅保留连续求解器和图形编辑器功能
 */

(function () {
  "use strict";

  const API_BASE = "";
  const GRAPH_BOUNDS = { width: 2700, height: 1680, padding: 36 };

  // WASD/方向键平移：按键集合 + rAF 循环实现连续平滑移动
  const PAN_KEYS = {
    w: { dx: 0, dy: -1 }, s: { dx: 0, dy: 1 },
    a: { dx: -1, dy: 0 }, d: { dx: 1, dy: 0 },
    arrowup: { dx: 0, dy: -1 }, arrowdown: { dx: 0, dy: 1 },
    arrowleft: { dx: -1, dy: 0 }, arrowright: { dx: 1, dy: 0 },
  };
  let panKeys = new Set();
  let panRaf = null;
  let panVX = 0, panVY = 0; // 当前平移速度（视口宽/高的比例/帧），用于惯性
  const MAX_PAN_SPEED = 0.022; // 最高速：每帧移动视口宽/高的 2.2%（约 130% 视口/秒）
  const PAN_ACCEL = 0.2;       // 加速：每帧向目标速度逼近 20% 的速度差
  const PAN_DECAY = 0.86;      // 松键后的惯性衰减系数（每帧，越小滑得越短）
  const DEFAULT_VIEWPORT = { x: 0, y: 0, width: 900, height: 560 };
  const NODE_RADIUS = 28;
  const STORAGE_KEY = "conveyor-sim-graphs-v1";

  const MODE_LABELS = {
    select: "选择",
    "add-In": "放置输入",
    "add-Out": "放置输出",
    "add-S": "放置分流器",
    "add-C": "放置汇流器",
    edge: "连边",
  };
  const TYPE_LABELS = {
    In: "In", Out: "Out", S: "S", C: "C",
  };
  const NODE_COLORS = {
    In: "#2e8b57", Out: "#c84c3a", S: "#2f6fed", C: "#d98a1f",
  };
  const NODE_CAPACITY = {
    In: { maxIn: 0, maxOut: 1 },
    Out: { maxIn: 1, maxOut: 0 },
    S: { maxIn: 1, maxOut: 3 },
    C: { maxIn: 3, maxOut: 1 },
  };
  const INITIAL_GRAPH = {
    nodes: [
      { id: "In1", type: "In", x: 90, y: 280 },
      { id: "C1", type: "C", x: 270, y: 280 },
      { id: "S1", type: "S", x: 450, y: 210 },
      { id: "S2", type: "S", x: 620, y: 210 },
      { id: "C2", type: "C", x: 450, y: 440 },
      { id: "Out1", type: "Out", x: 780, y: 210 },
    ],
    edges: [
      { id: "edge1", from: "In1", to: "C1", toSlot: 0 },
      { id: "edge2", from: "C1", to: "S1" },
      { id: "edge3", from: "S1", to: "Out1", fromSlot: 0 },
      { id: "edge4", from: "S1", to: "C2", fromSlot: 1 },
      { id: "edge5", from: "S1", to: "S2", fromSlot: 2 },
      { id: "edge6", from: "S2", to: "C2" },
      { id: "edge7", from: "C2", to: "C1", toSlot: 1 },
      { id: "edge8", from: "S2", to: "C1", toSlot: 2 },
    ],
  };
  const BLANK_GRAPH = {
    nodes: [
      { id: "In1", type: "In", x: 180, y: 280 },
      { id: "Out1", type: "Out", x: 720, y: 280 },
    ],
    edges: [],
  };

  const state = {
    graph: cloneGraph(INITIAL_GRAPH),
    selected: { kind: "node", id: "C1" },
    mode: "select",
    draftEdgeFrom: null,
    pointerGraph: { x: 0, y: 0 },
    drag: null,
    viewport: { ...DEFAULT_VIEWPORT },
    pan: null,
    blankHold: false,
    flowGroups: [],
    currentFlowIdx: 0,
    currentStateIdx: 0,
    solveAttempted: false,
    pendingSolve: 0,  // 用于取消过期请求
    counters: buildCounters(INITIAL_GRAPH),
    saves: [],
    activeSaveId: null,
    message: "选择模式下可选中、拖动和删除节点。",
    backendBusy: false,
    topologyLayout: null,
    solverModel: "milp",
    // 离散仿真
    simFrames: [],
    simFrame: 0,
    simCycle: null,
    simPlaying: false,
    simFps: 8,
    simAutoFrames: 5000,
    simRaf: null,
    simCarry: 0,
  };

  const els = {
    svg: document.getElementById("graph"),
    clear: document.getElementById("clear-btn"),
    save: document.getElementById("save-btn"),
    load: document.getElementById("load-btn"),
    rename: document.getElementById("rename-btn"),
    removeSave: document.getElementById("delete-save-btn"),
    solveBtn: document.getElementById("solve-btn"),
    solverModel: document.getElementById("solver-model"),
    threeStateControls: document.getElementById("three-state-controls"),
    flowPrevBtn: document.getElementById("flow-prev-btn"),
    flowNextBtn: document.getElementById("flow-next-btn"),
    statePrevBtn: document.getElementById("state-prev-btn"),
    stateNextBtn: document.getElementById("state-next-btn"),
    flowCounter: document.getElementById("flow-counter"),
    stateCounter: document.getElementById("state-counter"),
    saveName: document.getElementById("save-name-input"),
    saveList: document.getElementById("save-list"),
    graphStats: document.getElementById("graph-stats"),
    toolButtons: [...document.querySelectorAll("[data-mode-button]")],
    importJsonInput: document.getElementById("import-json-input"),
    importBtn: document.getElementById("import-btn"),
    importStatus: document.getElementById("import-status"),
    layoutBtn: document.getElementById("layout-btn"),
    layoutStatus: document.getElementById("layout-status"),
    layoutExportBtn: document.getElementById("layout-export-btn"),
    layoutImportBtn: document.getElementById("layout-import-btn"),
    layoutImportFile: document.getElementById("layout-import-file"),
    layoutOverlay: document.getElementById("layout-overlay"),
    layoutFrame: document.getElementById("layout-frame"),
    layoutToggleBtn: document.getElementById("layout-toggle-btn"),
    wgInput: document.getElementById("wg-input"),
    wgExportBtn: document.getElementById("wg-export-btn"),
    wgImportBtn: document.getElementById("wg-import-btn"),
    wgMergeBtn: document.getElementById("wg-merge-btn"),
    wgStatus: document.getElementById("wg-status"),
    ratioP: document.getElementById("ratio-p"),
    ratioQ: document.getElementById("ratio-q"),
    ratioP: document.getElementById("ratio-p"),
    ratioQ: document.getElementById("ratio-q"),
    ratioBtn: document.getElementById("ratio-split-btn"),
    ratioStatus: document.getElementById("ratio-status"),
    simRunBtn: document.getElementById("sim-run-btn"),
    simPlayBtn: document.getElementById("sim-play-btn"),
    simPrevBtn: document.getElementById("sim-prev-btn"),
    simNextBtn: document.getElementById("sim-next-btn"),
    simFrameSlider: document.getElementById("sim-frame-slider"),
    simFrameValue: document.getElementById("sim-frame-value"),
    simFps: document.getElementById("sim-fps"),
    simAutoFrames: document.getElementById("sim-auto-frames"),
    simStatus: document.getElementById("sim-status"),
    limitP: document.getElementById("limit-p"),
    limitQ: document.getElementById("limit-q"),
    limitModuleBtn: document.getElementById("limit-module-btn"),
    limitDepth: document.getElementById("limit-depth"),
    limitOptimize: document.getElementById("limit-optimize"),
    limitCrosscheck: document.getElementById("limit-crosscheck"),
    limitStatus: document.getElementById("limit-status"),
    limitInfo: document.getElementById("limit-info"),
  };

  // ── MILP 连续求解（多解） ──────────────────────────────────────

  async function apiSolveMilp(nodes, edges) {
    const resp = await fetch(`${API_BASE}/api/solve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nodes, edges }),
    });
    if (!resp.ok) throw new Error(await resp.text());
    return resp.json();
  }

  // 把后端 MILP 多解包装成 flowGroups（每个解一个 flowGroup，单 state）
  function wrapMilpResult(data) {
    if (!data || !data.feasible || !Array.isArray(data.solutions)) return [];
    return data.solutions.map((sol) => {
      const edgeFlows = {};
      const edgeBlocked = {};
      for (const ef of sol.edgeFlows || []) {
        edgeFlows[ef.id] = {
          text: ef.flow.text,
          numerator: ef.flow.numerator,
          denominator: ef.flow.denominator,
        };
        edgeBlocked[ef.id] = ef.isBlocked ? "fb" : "nb";
      }
      const nodeFlows = {};
      for (const nf of sol.nodeFlows || []) {
        nodeFlows[nf.id] = {
          text: nf.flow.text,
          numerator: nf.flow.numerator,
          denominator: nf.flow.denominator,
        };
      }
      return { states: [{ edgeBlocked, edgeFlows, nodeFlows }] };
    });
  }

  async function triggerMilpSolve() {
    if (state.backendBusy) return;
    state.message = "MILP 求解中...";
    state.flowGroups = [];
    state.currentFlowIdx = 0;
    state.currentStateIdx = 0;
    state.solveAttempted = true;
    render();
    try {
      const result = await apiSolveMilp(state.graph.nodes, state.graph.edges);
      state.flowGroups = wrapMilpResult(result);
      state.currentFlowIdx = 0;
      state.currentStateIdx = 0;
      const fg = state.flowGroups;
      if (fg.length > 0) {
        state.message = `MILP 求解完成，共 ${fg.length} 种流量分配。`;
      } else {
        state.message = result.error || "MILP 求解未找到可行解。";
        showToast(state.message, "info");
      }
    } catch (error) {
      state.flowGroups = [];
      state.currentFlowIdx = 0;
      state.currentStateIdx = 0;
      state.message = `MILP 求解失败: ${error.message}`;
      showToast(`MILP 求解失败: ${error.message}`, "error");
    }
    render();
  }

  // ── 流量求解（Rust 原生：rank-smt / stable） ───────────────────

  async function apiSolveNative(model, nodes, edges) {
    const engine = model.replace("rust-", "");
    const resp = await fetch(`${API_BASE}/api/solve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nodes, edges, engine }),
    });
    if (!resp.ok) {
      let msg;
      try { msg = (await resp.json()).error; } catch { msg = await resp.text(); }
      throw new Error(msg || `后端返回 ${resp.status}`);
    }
    return resp.json();
  }

  // 把 Rust 逐边流量包装成单个“状态对象”，复用现有 edgeFlows 渲染
  function wrapNativeResult(res) {
    const edgeFlows = {};
    state.graph.edges.forEach((edge, i) => {
      const ef = res.edgeFlows && res.edgeFlows[i];
      if (!ef) return;
      edgeFlows[edge.id] = {
        text: ef.text || String(ef.flow),
        numerator: ef.numerator ?? ef.flow,
        denominator: ef.denominator ?? 1,
      };
    });
    return [{ states: [{ edgeBlocked: {}, edgeFlows, nodeFlows: {} }] }];
  }

  async function triggerNativeSolve() {
    if (state.backendBusy) return;
    state.message = "Rust 求解中...";
    state.flowGroups = [];
    state.currentFlowIdx = 0;
    state.currentStateIdx = 0;
    state.solveAttempted = true;
    render();
    try {
      const res = await apiSolveNative(state.solverModel, state.graph.nodes, state.graph.edges);
      state.flowGroups = wrapNativeResult(res);
      state.currentFlowIdx = 0;
      state.currentStateIdx = 0;
      state.message =
        `Rust 求解完成（${res.backend}，总流量 ${res.totalText || (+res.totalFlow).toFixed(4)}，${res.iterations} 次迭代）。`;
    } catch (error) {
      state.flowGroups = [];
      state.message = `Rust 求解失败: ${error.message}`;
      showToast(`Rust 求解失败: ${error.message}`, "error");
    }
    render();
  }

  // 切换模型：rust 模式只保留求解按钮（隐藏流量导航控件）
  function applySolverModel() {
    const opt = els.solverModel.selectedOptions[0];
    const kind = opt ? opt.dataset.kind : "milp";
    const isRust = kind === "rust";
    state.solverModel = els.solverModel.value;
    if (els.threeStateControls) els.threeStateControls.style.display = isRust ? "none" : "";
    if (isRust) {
      state.flowGroups = [];
      state.currentFlowIdx = 0;
      state.currentStateIdx = 0;
    }
    render();
  }

  // 从后端拉取求解器列表填充模型下拉（便于后续新增求解器）
  async function loadSolvers() {
    try {
      const resp = await fetch(`${API_BASE}/api/solvers`);
      if (!resp.ok) return;
      const { solvers } = await resp.json();
      if (!solvers || solvers.length === 0) return;
      els.solverModel.innerHTML = "";
      solvers.forEach((s) => {
        const o = document.createElement("option");
        o.value = s.id;
        o.textContent = s.label;
        o.dataset.kind = s.kind;
        els.solverModel.appendChild(o);
      });
      els.solverModel.value = state.solverModel;
    } catch (e) { /* 后端不可用时保留默认选项 */ }
  }

  // ── 物理布局（调用内置布局求解器） ─────────────────────────────

  async function apiTopoflowLayout(nodes, edges, opts = {}, onProgress) {
    const resp = await fetch(`${API_BASE}/api/topoflow-layout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        nodes, edges,
        requireBeltCell: !!opts.requireBeltCell,
        timeLimit: opts.timeLimit ?? 30.0,
        minGrid: opts.minGrid ?? 3,
      }),
    });
    if (!resp.ok) {
      let msg;
      try { msg = (await resp.json()).error; } catch { msg = await resp.text(); }
      throw new Error(msg || `后端返回 ${resp.status}`);
    }
    // 流式读取 NDJSON：实时处理 progress，末尾取出 result
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let result = null;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const item = JSON.parse(line);
        if (item.type === "progress" && onProgress) {
          onProgress(item.rows, item.cols);
        } else if (item.type === "result") {
          result = item.result;
        }
      }
    }
    return result;
  }

  async function triggerTopoflowLayout() {
    if (state.backendBusy) return;
    state.backendBusy = true;
    els.layoutStatus.textContent = "正在调用布局求解器计算物理布局…";
    try {
      const data = await apiTopoflowLayout(
        state.graph.nodes, state.graph.edges,
        { requireBeltCell: true, minGrid: 3 },
        (rows, cols) => {
          els.layoutStatus.textContent = `正在尝试 ${rows}×${cols}…`;
        });
      if (!data || data.status !== "ok") {
        els.layoutStatus.textContent = (data && data.error) || "未找到可行物理布局";
        showToast((data && data.error) || "物理布局失败", "error");
        return;
      }
      const [r, c] = data.grid;
      const machs = (data.solution && data.solution.machs) || [];
      state.topologyLayout = {
        grid: data.grid,
        routeLength: data.routeLength,
        solution: data.solution,
      };
      els.layoutStatus.textContent =
        `最小网格 ${r}×${c}，线路总长 ${data.routeLength}，设施 ${machs.length} 个（左上角返回）`;
      showLayoutOverlay(state.topologyLayout);
      showToast(`物理布局完成: ${r}×${c}`, "info");
    } catch (error) {
      els.layoutStatus.textContent = `物理布局失败: ${error.message}`;
      showToast(`物理布局失败: ${error.message}`, "error");
    } finally {
      state.backendBusy = false;
    }
  }

  // ── 标准分流计算（特定比例二分模块） ────────────────────────────
  async function apiRatioSplit(p, q) {
    const resp = await fetch("/api/ratio-split", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ p, q }),
    });
    if (!resp.ok) {
      const e = await resp.json().catch(() => ({}));
      throw new Error(e.error || `HTTP ${resp.status}`);
    }
    return resp.json();
  }

  async function triggerRatioSplit() {
    if (state.backendBusy) return;
    const p = parseInt(els.ratioP.value, 10);
    const q = parseInt(els.ratioQ.value, 10);
    if (!Number.isInteger(p) || !Number.isInteger(q) || p <= 0 || q <= 0 || p >= q) {
      els.ratioStatus.textContent = "需满足 0 < p < q 的正整数";
      return;
    }
    state.backendBusy = true;
    els.ratioStatus.textContent = `正在解算比例 ${p}:${q}…`;
    try {
      const data = await apiRatioSplit(p, q);
      if (!data.nodes || !data.nodes.length) {
        els.ratioStatus.textContent = data.error || "解算失败";
        return;
      }
      // 结算结果覆盖当前图
      window._importGraph({ nodes: data.nodes, edges: data.edges });
      const info = data.info || {};
      els.ratioStatus.textContent =
        `已生成 N=${info.N}（2^${info.a}·3^${info.b}），Out1:Out2=${info.ratio}，反馈 ${info.feedBack} 份`;
      showToast(`比例二分图已生成: ${info.ratio}`, "info");
    } catch (error) {
      els.ratioStatus.textContent = `解算失败: ${error.message}`;
      showToast(`解算失败: ${error.message}`, "error");
    } finally {
      state.backendBusy = false;
    }
  }

  // ── 物理布局展示（复用 layout-editor.html，iframe + postMessage） ──
  let layoutFrameLoaded = false;
  let layoutPendingMeta = null;

  /** 通过 postMessage 把物理解注入 iframe 内的 editor.html 渲染器。 */
  function postLayoutToFrame(meta) {
    els.layoutFrame.contentWindow.postMessage(
      { kind: "topoflow-layout", solution: meta.solution, fileName: "物理布局" },
      "*"
    );
  }

  /** 在中间画布上叠加显示物理布局（覆盖原图）。 */
  function showLayoutOverlay(meta) {
    if (!meta) return;
    if (layoutFrameLoaded) postLayoutToFrame(meta);
    else layoutPendingMeta = meta; // 等 iframe 加载完成后再注入
    els.layoutOverlay.style.display = "block";
    els.layoutToggleBtn.textContent = "查看原图";
    els.layoutToggleBtn.style.display = "block";
  }

  /** 关闭覆盖层，回归原图渲染。 */
  function closeLayoutOverlay() {
    els.layoutOverlay.style.display = "none";
    els.layoutToggleBtn.textContent = "查看物理结构";
  }

  /** 切换「原图」与「物理结构」视图。 */
  function toggleLayoutView() {
    if (els.layoutOverlay.style.display === "none") {
      showLayoutOverlay(state.topologyLayout);
    } else {
      closeLayoutOverlay();
    }
  }

  /** 导出物理布局为 JSON 文件（含 solution，可被本工具或布局编辑器读取）。 */
  function onLayoutExport() {
    const meta = state.topologyLayout;
    if (!meta) {
      showToast("尚无物理布局可导出，请先计算", "error");
      return;
    }
    const payload = {
      format: "zmd-topoflow-layout",
      grid: meta.grid,
      routeLength: meta.routeLength,
      solution: meta.solution,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)],
      { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "topoflow-layout.json";
    document.body.appendChild(anchor);
    anchor.click();
    setTimeout(() => {
      document.body.removeChild(anchor);
      URL.revokeObjectURL(url);
    }, 50);
    showToast("已导出物理布局", "info");
  }

  /** 打开文件选择，导入物理布局。 */
  function onLayoutImport() {
    els.layoutImportFile.value = "";
    els.layoutImportFile.click();
  }

  /** 解析导入的物理布局文件并渲染。 */
  function handleLayoutImportFile(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result);
        const solution = (data && data.solution) ? data.solution : data;
        if (!solution || !Array.isArray(solution.machs)
          || !Array.isArray(solution.belts)) {
          throw new Error("不是有效的物理布局文件");
        }
        const meta = {
          solution,
          grid: (data && data.grid) || [solution.n, solution.m],
          routeLength: (data && data.routeLength) || 0,
        };
        state.topologyLayout = meta;
        els.layoutStatus.textContent =
          `已导入布局 ${meta.grid[0]}×${meta.grid[1]}（左上角返回）`;
        showLayoutOverlay(meta);
        showToast("已导入物理布局", "info");
      } catch (error) {
        els.layoutStatus.textContent = `导入失败: ${error.message}`;
        showToast(`导入失败: ${error.message}`, "error");
      }
    };
    reader.readAsText(file);
  }

  function getCurrentSolution() {
    const fg = state.flowGroups;
    if (fg.length === 0) return null;
    const flowG = fg[state.currentFlowIdx];
    if (!flowG || !flowG.states || flowG.states.length === 0) return null;
    return flowG.states[state.currentStateIdx] || null;
  }

  // ── 离散仿真（帧回放） ─────────────────────────────────────────

  async function apiSimulate(nodes, edges, maxFrames) {
    const resp = await fetch(`${API_BASE}/api/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nodes, edges, options: { max_frames: maxFrames } }),
    });
    if (!resp.ok) {
      let msg;
      try { msg = (await resp.json()).error; } catch { msg = await resp.text(); }
      throw new Error(msg || `后端返回 ${resp.status}`);
    }
    return resp.json();
  }

  function stopSimPlay() {
    state.simPlaying = false;
    if (state.simRaf) { cancelAnimationFrame(state.simRaf); state.simRaf = null; }
    if (els.simPlayBtn) els.simPlayBtn.textContent = "播放";
  }

  async function triggerSimulate() {
    if (state.backendBusy) return;
    state.backendBusy = true;
    stopSimPlay();
    if (els.simStatus) els.simStatus.textContent = "仿真中…";
    try {
      const maxFrames = clampInt(els.simAutoFrames.value, 1, 10000000, 5000);
      state.simAutoFrames = maxFrames;
      const data = await apiSimulate(state.graph.nodes, state.graph.edges, maxFrames);
      state.simFrames = data.frames || [];
      state.simFrame = 0;
      state.simCycle = data.cycleInfo || null;
      if (els.simStatus) {
        const ci = state.simCycle;
        els.simStatus.textContent = ci
          ? `完成：周期 ${ci.period} 帧，预热 ${ci.warmupFrames} 帧，共 ${state.simFrames.length} 帧`
          : `完成：${state.simFrames.length} 帧`;
      }
      showToast(`仿真完成，共 ${state.simFrames.length} 帧`, "info");
    } catch (error) {
      state.simFrames = [];
      state.simFrame = 0;
      state.simCycle = null;
      if (els.simStatus) els.simStatus.textContent = `仿真失败: ${error.message}`;
      showToast(`仿真失败: ${error.message}`, "error");
    } finally {
      state.backendBusy = false;
    }
    render();
  }

  function stepSim(delta) {
    const n = state.simFrames.length;
    if (n === 0) return;
    state.simFrame = (state.simFrame + delta + n) % n;
    render();
  }

  function setSimFrame(idx) {
    const n = state.simFrames.length;
    if (n === 0) return;
    state.simFrame = clamp(idx, 0, n - 1);
    render();
  }

  function toggleSimPlay() {
    if (state.simFrames.length === 0) {
      showToast("请先运行仿真", "info");
      return;
    }
    if (state.simPlaying) { stopSimPlay(); render(); return; }
    state.simPlaying = true;
    if (els.simPlayBtn) els.simPlayBtn.textContent = "暂停";
    let last = performance.now();
    const loop = (now) => {
      if (!state.simPlaying) return;
      const dt = (now - last) / 1000;
      last = now;
      state.simCarry += dt * state.simFps;
      const steps = Math.floor(state.simCarry);
      if (steps > 0) {
        state.simCarry -= steps;
        const n = state.simFrames.length;
        if (n > 0) state.simFrame = (state.simFrame + steps) % n;
        render();
      }
      state.simRaf = requestAnimationFrame(loop);
    };
    state.simRaf = requestAnimationFrame(loop);
  }

  function getSimSnapshot() {
    if (!state.simFrames.length) return null;
    return state.simFrames[state.simFrame] || null;
  }

  // ── 标准限流计算 ─────────────────────────────────────

  async function apiLimitModule(p, q, opts) {
    const resp = await fetch(`${API_BASE}/api/limit-module`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        p, q,
        optimize: !!opts.optimize,
        reduction_depth: opts.reductionDepth ?? 3,
        cross_check: !!opts.crossCheck,
      }),
    });
    if (!resp.ok) {
      let msg;
      try { msg = (await resp.json()).error; } catch { msg = await resp.text(); }
      throw new Error(msg || `后端返回 ${resp.status}`);
    }
    return resp.json();
  }

  async function triggerLimitModule() {
    if (state.backendBusy) return;
    const p = parseInt(els.limitP.value, 10);
    const q = parseInt(els.limitQ.value, 10);
    if (!Number.isInteger(p) || !Number.isInteger(q) || p <= 0 || q <= 0 || p >= q) {
      els.limitStatus.textContent = "需满足 0 < p < q 的正整数";
      return;
    }
    state.backendBusy = true;
    els.limitStatus.textContent = `正在解算限流 ${p}/${q}…`;
    if (els.limitInfo) els.limitInfo.style.display = "none";
    try {
      const data = await apiLimitModule(p, q, {
        optimize: els.limitOptimize.checked,
        crossCheck: els.limitCrosscheck.checked,
        reductionDepth: clampInt(els.limitDepth.value, 1, 8, 3),
      });
      if (!data.nodes || !data.nodes.length) {
        els.limitStatus.textContent = data.error || "解算失败";
        return;
      }
      window._importGraph({ nodes: data.nodes, edges: data.edges });
      const info = data.info || {};
      const cost = info.cost || {};
      let text = `已生成限流模块 ${info.target}（策略 ${info.strategy}，${cost.nodes ?? "?"}节点/${cost.edges ?? "?"}边，rank ${info.rank}/${info.edgeCount}）`;
      if (info.crossCheck) {
        text += `｜Rust 复核 ${info.crossCheck.backend} ${info.crossCheck.status}`;
      }
      els.limitStatus.textContent = text;
      if (els.limitInfo) {
        const steps = (info.steps || []).join(" | ");
        els.limitInfo.textContent = `steps: ${steps}\nflow: ${info.flow}  fullRank: ${info.fullRank}`;
        els.limitInfo.style.display = "block";
      }
      showToast(`限流模块已生成: ${info.target}`, "info");
    } catch (error) {
      els.limitStatus.textContent = `解算失败: ${error.message}`;
      showToast(`解算失败: ${error.message}`, "error");
    } finally {
      state.backendBusy = false;
    }
  }

  // ── 主函数 ──────────────────────────────────────────────────────

  function main() {
    loadSavesFromStorage();
    rebuildSimulator("已载入示例图。");
    bindEvents();
    render();
    loadSolvers().then(() => applySolverModel());
  }

  async function rebuildSimulator(message) {
    state.flowGroups = [];
    state.currentFlowIdx = 0;
    state.currentStateIdx = 0;
    state.solveAttempted = false;
    stopSimPlay();
    state.simFrames = [];
    state.simFrame = 0;
    state.simCycle = null;
    ensureSelectedObject();
    state.message = message;
    render();
  }

  // ── 事件绑定 ──────────────────────────────────────────────────

  function bindEvents() {
    els.clear.addEventListener("click", clearGraphToEndpoints);

    // 求解按钮（按当前模型分发：rust → 原生求解；其余 → MILP）
    els.solveBtn.addEventListener("click", () => {
      if (state.solverModel.startsWith("rust-")) triggerNativeSolve();
      else triggerMilpSolve();
    });
    // 模型切换
    els.solverModel.addEventListener("change", applySolverModel);

    function navigateFlow(delta) {
      const n = state.flowGroups.length;
      if (n === 0) return;
      state.currentFlowIdx = (state.currentFlowIdx + delta + n) % n;
      state.currentStateIdx = 0;
      render();
    }
    function navigateState(delta) {
      const g = state.flowGroups[state.currentFlowIdx];
      if (!g || !g.states || g.states.length === 0) return;
      state.currentStateIdx = (state.currentStateIdx + delta + g.states.length) % g.states.length;
      render();
    }
    els.flowPrevBtn.addEventListener("click", () => navigateFlow(-1));
    els.flowNextBtn.addEventListener("click", () => navigateFlow(1));
    els.statePrevBtn.addEventListener("click", () => navigateState(-1));
    els.stateNextBtn.addEventListener("click", () => navigateState(1));
    els.saveList.addEventListener("change", () => {
      state.activeSaveId = els.saveList.value || null;
      syncSelectedSaveName();
    });
    els.saveList.addEventListener("click", () => {
      state.activeSaveId = els.saveList.value || null;
      syncSelectedSaveName();
    });
    els.saveList.addEventListener("pointerup", () => {
      state.activeSaveId = els.saveList.value || null;
      syncSelectedSaveName();
    });
    els.save.addEventListener("click", saveCurrentGraph);
    els.load.addEventListener("click", loadSelectedGraph);
    els.rename.addEventListener("click", renameSelectedGraph);
    els.removeSave.addEventListener("click", deleteSelectedSave);

    for (const button of els.toolButtons) {
      button.addEventListener("click", () => {
        setMode(button.getAttribute("data-mode-button"));
      });
    }

    els.svg.addEventListener("pointerdown", onSvgPointerDown);
    els.svg.addEventListener("pointermove", onSvgPointerMove);
    els.svg.addEventListener("wheel", onSvgWheel, { passive: false });
    els.svg.addEventListener("contextmenu", onSvgContextMenu);
    document.addEventListener("pointermove", onDocumentPointerMove);
    document.addEventListener("pointerup", onDocumentPointerUp);
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("keyup", onKeyUp);

    // 导入
    els.importBtn.addEventListener("click", onImportJson);

    // 物理布局（调用内置布局求解器）
    els.layoutBtn.addEventListener("click", triggerTopoflowLayout);
    els.layoutToggleBtn.addEventListener("click", toggleLayoutView);
    els.layoutExportBtn.addEventListener("click", onLayoutExport);
    els.layoutImportBtn.addEventListener("click", onLayoutImport);
    els.layoutImportFile.addEventListener("change", handleLayoutImportFile);
    // 给预览页加版本参数，绕过浏览器对旧版 layout-editor.html 的缓存
    els.layoutFrame.src = `/layout-editor.html?v=${Date.now()}`;
    // 等 iframe 内预览页加载完成后，注入缓存的物理解
    els.layoutFrame.addEventListener("load", () => {
      layoutFrameLoaded = true;
      if (layoutPendingMeta) {
        postLayoutToFrame(layoutPendingMeta);
        layoutPendingMeta = null;
      }
    });

    // wg 格式导入/导出
    els.wgExportBtn.addEventListener("click", onWgExport);
    els.wgImportBtn.addEventListener("click", onWgImport);
    els.wgMergeBtn.addEventListener("click", onWgImportMerge);
    els.ratioBtn.addEventListener("click", triggerRatioSplit);

    // 离散仿真
    els.simRunBtn.addEventListener("click", triggerSimulate);
    els.simPlayBtn.addEventListener("click", toggleSimPlay);
    els.simPrevBtn.addEventListener("click", () => { stopSimPlay(); stepSim(-1); });
    els.simNextBtn.addEventListener("click", () => { stopSimPlay(); stepSim(1); });
    els.simFrameSlider.addEventListener("input", () => {
      stopSimPlay();
      setSimFrame(parseInt(els.simFrameSlider.value, 10) || 0);
    });
    els.simFps.addEventListener("input", () => {
      state.simFps = clampInt(els.simFps.value, 1, 240, 8);
    });

    // 标准限流计算
    els.limitModuleBtn.addEventListener("click", triggerLimitModule);
  }

  // ── 键盘事件 ──────────────────────────────────────────────────

  function onKeyDown(event) {
    if (isEditableTarget(event.target)) return;
    const keyMap = {
      "1": "add-S", "2": "add-C", "3": "add-In", "4": "add-Out",
    };
    if (keyMap[event.key]) {
      event.preventDefault();
      setMode(keyMap[event.key]);
      return;
    }
    if (event.key.toLowerCase() === "e") {
      event.preventDefault();
      setMode("edge");
      return;
    }
    if (event.key.toLowerCase() === "x" || event.key === "Escape") {
      event.preventDefault();
      setMode("select");
      return;
    }
    if (["Delete", "Backspace"].includes(event.key) || event.key.toLowerCase() === "f") {
      if (state.selected) {
        event.preventDefault();
        deleteSelectedObject();
      }
    }
    // WASD / 方向键平移画布（按键集合 + rAF 连续移动）
    if (PAN_KEYS[event.key.toLowerCase()]) {
      event.preventDefault();
      panKeys.add(event.key.toLowerCase());
      startPanMove();
    }
  }

  /** 松键时从按键集合移除对应方向；集合为空时 rAF 循环自动停止。 */
  function onKeyUp(event) {
    panKeys.delete(event.key.toLowerCase());
  }

  /** 启动/保持 rAF 循环：向目标方向渐进加速，松键后惯性滑行减速。 */
  function startPanMove() {
    if (panRaf) return;
    let last = performance.now();
    const loop = (now) => {
      // dt 按 60fps 归一，保证不同帧率下速度一致
      const dt = Math.min((now - last) / (1000 / 60), 3);
      last = now;
      let dx = 0, dy = 0;
      for (const k of panKeys) {
        const v = PAN_KEYS[k];
        if (v) { dx += v.dx; dy += v.dy; }
      }
      const len = Math.hypot(dx, dy) || 1;
      const tx = (dx / len) * MAX_PAN_SPEED; // 目标速度
      const ty = (dy / len) * MAX_PAN_SPEED;
      if (dx !== 0 || dy !== 0) {
        // 有按键：指数逼近目标速度（渐进加速）
        panVX += (tx - panVX) * PAN_ACCEL * dt;
        panVY += (ty - panVY) * PAN_ACCEL * dt;
      } else {
        // 无按键：惯性滑行，速度按指数衰减
        panVX *= Math.pow(PAN_DECAY, dt);
        panVY *= Math.pow(PAN_DECAY, dt);
      }
      state.viewport.x += panVX * state.viewport.width * dt;
      state.viewport.y += panVY * state.viewport.height * dt;
      els.svg.setAttribute("viewBox",
        `${state.viewport.x} ${state.viewport.y} ${state.viewport.width} ${state.viewport.height}`);
      // 无按键且速度几乎归零时停止
      if (panKeys.size === 0 && Math.abs(panVX) < 0.0002 && Math.abs(panVY) < 0.0002) {
        panVX = 0; panVY = 0; panRaf = null; return;
      }
      panRaf = requestAnimationFrame(loop);
    };
    panRaf = requestAnimationFrame(loop);
  }

  // ── 指针事件 ──────────────────────────────────────────────────

  function onSvgPointerDown(event) {
    blurActiveElement();
    const point = getSvgPoint(event);
    state.pointerGraph = point;
    const nodeEl = event.target.closest("[data-node-id]");

    if (nodeEl) {
      const nodeId = nodeEl.getAttribute("data-node-id");
      if (state.mode === "edge") {
        handleEdgeNodeClick(nodeId);
        render();
        return;
      }
      state.selected = { kind: "node", id: nodeId };
      const node = getNode(nodeId);
      const rect = els.svg.getBoundingClientRect();
      // 记录起始图坐标 + 屏幕比例，拖拽时用屏幕像素差换算，避免 getScreenCTM 的反馈振荡
      state.drag = {
        nodeId,
        startNodeX: node.x,
        startNodeY: node.y,
        startClientX: event.clientX,
        startClientY: event.clientY,
        scaleX: state.viewport.width / rect.width,
        scaleY: state.viewport.height / rect.height,
      };
      render();
      event.preventDefault();
      return;
    }

    const edgeEl = event.target.closest("[data-edge-id]");
    if (edgeEl) {
      state.selected = { kind: "edge", id: edgeEl.getAttribute("data-edge-id") };
      state.message = `已选中边 ${state.selected.id}。`;
      render();
      return;
    }

    if (state.mode === "edge") {
      state.draftEdgeFrom = null;
      state.message = "连边模式：先点起点，再点终点。";
      render();
      return;
    }

    if (state.mode === "select") {
      state.blankHold = true;
      const rect = els.svg.getBoundingClientRect();
      state.pan = {
        startClientX: event.clientX,
        startClientY: event.clientY,
        scaleX: state.viewport.width / rect.width,
        scaleY: state.viewport.height / rect.height,
        startViewportX: state.viewport.x,
        startViewportY: state.viewport.y,
      };
      state.selected = null;
      state.message = "已取消选择，可拖动画布。";
      event.preventDefault();
      render();
      return;
    }

    if (state.mode.startsWith("add-")) {
      const type = modeToNodeType(state.mode);
      addNode(type, point);
    }
  }

  function onSvgPointerMove(event) {
    state.pointerGraph = getSvgPoint(event);
    if (state.mode === "edge" && state.draftEdgeFrom) render();
  }

  function onDocumentPointerMove(event) {
    if (state.drag) {
      const node = getNode(state.drag.nodeId);
      // 用 pointerdown 时固定的比例 × 屏幕像素差换算，避免 getScreenCTM 反馈振荡；不限图边界
      node.x = state.drag.startNodeX + (event.clientX - state.drag.startClientX) * state.drag.scaleX;
      node.y = state.drag.startNodeY + (event.clientY - state.drag.startClientY) * state.drag.scaleY;
      // 轻量更新 DOM：不重建全部 SVG，只更新节点位置和相连边的路径
      _updateNodeDom(state.drag.nodeId);
      _updateConnectedEdgesDom(state.drag.nodeId);
      return;
    }
    if (!state.pan) return;
    state.viewport.x =
      state.pan.startViewportX - (event.clientX - state.pan.startClientX) * state.pan.scaleX;
    state.viewport.y =
      state.pan.startViewportY - (event.clientY - state.pan.startClientY) * state.pan.scaleY;
    // 轻量更新 viewBox，不触发全量 render
    els.svg.setAttribute("viewBox",
      `${state.viewport.x} ${state.viewport.y} ${state.viewport.width} ${state.viewport.height}`);
  }

  function onDocumentPointerUp() {
    state.drag = null;
    state.pan = null;
    state.blankHold = false;
  }

  function onSvgWheel(event) {
    event.preventDefault();
    const rect = els.svg.getBoundingClientRect();
    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;
    const anchorX = state.viewport.x + (localX / rect.width) * state.viewport.width;
    const anchorY = state.viewport.y + (localY / rect.height) * state.viewport.height;
    const currentScale = GRAPH_BOUNDS.width / state.viewport.width;
    const zoomFactor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    const nextScale = clamp(currentScale * zoomFactor, 1.05, 9);
    const nextWidth = GRAPH_BOUNDS.width / nextScale;
    const nextHeight = GRAPH_BOUNDS.height / nextScale;
    state.viewport.width = nextWidth;
    state.viewport.height = nextHeight;
    state.viewport.x = anchorX - (localX / rect.width) * nextWidth;
    state.viewport.y = anchorY - (localY / rect.height) * nextHeight;
    render();
  }

  function onSvgContextMenu(event) {
    event.preventDefault();
    blurActiveElement();
    if (state.mode !== "select") setMode("select");
  }

  function setMode(mode) {
    state.mode = mode;
    state.draftEdgeFrom = null;
    state.drag = null;
    state.message = getModeHint(mode);
    render();
  }

  function addNode(type, point) {
    const node = {
      id: nextId(type),
      type,
      x: point.x,
      y: point.y,
    };
    state.graph.nodes.push(node);
    state.selected = { kind: "node", id: node.id };
    rebuildSimulator(`已添加 ${node.id}。`);
  }

  function deleteSelectedObject() {
    if (!state.selected) return;
    if (state.selected.kind === "node") {
      const nodeId = state.selected.id;
      state.drag = null;
      state.pan = null;
      if (state.draftEdgeFrom === nodeId) state.draftEdgeFrom = null;
      state.graph.nodes = state.graph.nodes.filter((n) => n.id !== nodeId);
      state.graph.edges = state.graph.edges.filter(
        (e) => e.from !== nodeId && e.to !== nodeId
      );
      state.selected = state.graph.nodes[0]
        ? { kind: "node", id: state.graph.nodes[0].id }
        : null;
      rebuildSimulator(`节点 ${nodeId} 已删除。`);
    } else {
      const edgeId = state.selected.id;
      state.graph.edges = state.graph.edges.filter((e) => e.id !== edgeId);
      state.selected = state.graph.nodes[0]
        ? { kind: "node", id: state.graph.nodes[0].id }
        : null;
      rebuildSimulator(`边 ${edgeId} 已删除。`);
    }
  }

  function handleEdgeNodeClick(nodeId) {
    state.selected = { kind: "node", id: nodeId };
    if (!state.draftEdgeFrom) {
      state.draftEdgeFrom = nodeId;
      state.message = `连边模式：已选择起点 ${nodeId}，请点击终点。`;
      return;
    }
    if (state.draftEdgeFrom === nodeId) {
      state.draftEdgeFrom = null;
      state.message = "连边模式：已取消当前起点。";
      return;
    }
    const result = tryAddEdge(state.draftEdgeFrom, nodeId);
    state.draftEdgeFrom = null;
    state.message = result.message;
  }

  function tryAddEdge(fromId, toId) {
    const fromNode = getNode(fromId);
    const toNode = getNode(toId);
    if (fromId === toId) return { ok: false, message: "不允许创建自环。" };
    const fromShape = NODE_CAPACITY[fromNode.type];
    const toShape = NODE_CAPACITY[toNode.type];
    if (fromShape.maxOut === 0) return { ok: false, message: `${fromNode.id} 不能作为边的起点。` };
    if (toShape.maxIn === 0) return { ok: false, message: `${toNode.id} 不能作为边的终点。` };

    const fromSlot = findFirstEmptySlot(
      fromShape.maxOut,
      state.graph.edges.filter((e) => e.from === fromId).map((e) => e.fromSlot ?? 0)
    );
    if (fromSlot === -1) return { ok: false, message: `${fromNode.id} 的输出口已满。` };
    const toSlot = findFirstEmptySlot(
      toShape.maxIn,
      state.graph.edges.filter((e) => e.to === toId).map((e) => e.toSlot ?? 0)
    );
    if (toSlot === -1) return { ok: false, message: `${toNode.id} 的输入口已满。` };

    state.graph.edges.push({
      id: nextId("edge"),
      from: fromId,
      to: toId,
      fromSlot,
      toSlot,
    });
    rebuildSimulator(`已连接 ${fromId} -> ${toId}。`);
    return { ok: true, message: `已连接 ${fromId} -> ${toId}。` };
  }

  function ensureSelectedObject() {
    if (state.selected?.kind === "node" && state.graph.nodes.some((n) => n.id === state.selected.id)) return;
    if (state.selected?.kind === "edge" && state.graph.edges.some((e) => e.id === state.selected.id)) return;
    state.selected = state.graph.nodes[0] ? { kind: "node", id: state.graph.nodes[0].id } : null;
  }

  // ── 渲染 ──────────────────────────────────────────────────────

  function render() {
    renderControls();
    renderGraph();
  }

  function renderControls() {
    const fg = state.flowGroups;
    const flowCount = fg.length;
    const flowIdx = flowCount > 0 ? state.currentFlowIdx + 1 : 0;
    let flowDisplay;
    if (flowCount > 0) {
      flowDisplay = `${flowIdx} / ${flowCount}`;
    } else if (state.solveAttempted) {
      flowDisplay = "0";
    } else {
      flowDisplay = "手动求解";
    }
    els.flowCounter.textContent = flowDisplay;

    const curFlow = flowCount > 0 ? fg[state.currentFlowIdx] : null;
    const stateCount = curFlow ? curFlow.states.length : 0;
    const stateIdx = stateCount > 0 ? state.currentStateIdx + 1 : 0;
    let stateDisplay;
    if (stateCount > 0) {
      stateDisplay = `${stateIdx} / ${stateCount}`;
    } else {
      stateDisplay = flowCount > 0 ? "无状态" : "0";
    }
    els.stateCounter.textContent = stateDisplay;
    const simN = state.simFrames.length;
    if (els.simFrameSlider) {
      els.simFrameSlider.max = String(Math.max(0, simN - 1));
      els.simFrameSlider.value = String(simN > 0 ? state.simFrame : 0);
    }
    if (els.simFrameValue) {
      els.simFrameValue.textContent = simN > 0 ? `${state.simFrame + 1} / ${simN}` : "0 / 0";
    }
    els.graphStats.textContent = `${state.graph.nodes.length} 个节点 / ${state.graph.edges.length} 条边`;
    renderSaveControls();

    for (const button of els.toolButtons) {
      const active = button.getAttribute("data-mode-button") === state.mode;
      button.classList.toggle("is-active", active);
    }
  }

  function renderGraph() {
    const currSol = getCurrentSolution();
    const simSnap = getSimSnapshot();

    const edgesMarkup = state.graph.edges.map((edge) => {
      const path = computeEdgePath(edge);
      const selectedClass =
        state.selected?.kind === "edge" && state.selected.id === edge.id ? " is-selected" : "";
      const stateClass = currSol?.edgeBlocked?.[edge.id] === "fb" ? " is-blocked"
        : currSol?.edgeBlocked?.[edge.id] === "sb" ? " is-semi" : "";
      const continuousFlow = currSol?.edgeFlows?.[edge.id] || null;
      const continuousLabel = continuousFlow
        ? computeEdgeLabelPosition(edge, 0, 0)
        : null;
      const simEdge = simSnap?.edges?.[edge.id] || null;
      const simQueue = simEdge ? (simEdge.queue || []).length : 0;
      const simActiveClass = simQueue > 0 ? " is-active" : "";
      const simLabel = simSnap
        ? computeEdgeLabelPosition(edge, continuousFlow ? 1 : 0, 0)
        : null;
      return `
        <g class="edge-group${selectedClass}${stateClass}${simActiveClass}" data-edge-id="${edge.id}">
          <path class="edge-hit" d="${path}"></path>
          <path class="edge-line" d="${path}"></path>
          ${
            continuousFlow && continuousLabel
              ? `<text class="edge-flow edge-flow-continuous" x="${continuousLabel.x}" y="${continuousLabel.y}" text-anchor="middle">${escapeHtml(continuousFlow.text)}</text>`
              : ""
          }
          ${
            simSnap && simLabel
              ? `<text class="edge-flow edge-flow-sim" x="${simLabel.x}" y="${simLabel.y}" text-anchor="middle">${simQueue}</text>`
              : ""
          }
        </g>
      `;
    });

    const draftMarkup = renderDraftEdge();

    const nodesMarkup = state.graph.nodes.map((node) => {
      const selectedClass =
        state.selected?.kind === "node" && state.selected.id === node.id ? " is-selected" : "";
      const edgeStartClass = state.draftEdgeFrom === node.id ? " is-edge-start" : "";
      const continuousNodeFlow = currSol?.nodeFlows?.[node.id] || null;
      const simNode = simSnap?.nodes?.[node.id] || null;
      const busyClass = simNode?.hasItem ? " has-item" : "";
      return `
        <g class="node${selectedClass}${edgeStartClass}${busyClass}" data-node-id="${node.id}" transform="translate(${node.x}, ${node.y})">
          <circle r="${NODE_RADIUS}" fill="${NODE_COLORS[node.type]}"></circle>
          <text text-anchor="middle" dy="-2">${escapeHtml(node.id)}</text>
          <text class="node-type" text-anchor="middle" dy="14">${TYPE_LABELS[node.type]}</text>
          ${
            continuousNodeFlow
              ? `<text class="node-flow node-flow-continuous" text-anchor="middle" dy="46">${escapeHtml(continuousNodeFlow.text)}</text>`
              : ""
          }
        </g>
      `;
    });

    els.svg.setAttribute(
      "viewBox",
      `${state.viewport.x} ${state.viewport.y} ${state.viewport.width} ${state.viewport.height}`
    );
    els.svg.innerHTML = `
      <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#667085"></path>
        </marker>
      </defs>
      <g class="edges">${edgesMarkup.join("")}${draftMarkup}</g>
      <g class="nodes">${nodesMarkup.join("")}</g>
    `;
  }

  function renderDraftEdge() {
    if (state.mode !== "edge" || !state.draftEdgeFrom) return "";
    const from = state.graph.nodes.find((n) => n.id === state.draftEdgeFrom);
    if (!from) { state.draftEdgeFrom = null; return ""; }
    const pts = computeEdgePoints(from, state.pointerGraph);
    return `<path class="edge-line draft-edge" d="M ${pts.x1} ${pts.y1} L ${pts.x2} ${pts.y2}"></path>`;
  }

  // ── 几何工具 ──────────────────────────────────────────────────

  function computeEdgePoints(fromNode, toNodeOrPoint, offset = 0) {
    const dx = toNodeOrPoint.x - fromNode.x;
    const dy = toNodeOrPoint.y - fromNode.y;
    const length = Math.hypot(dx, dy) || 1;
    const ux = dx / length;
    const uy = dy / length;
    const px = -uy;
    const py = ux;
    return {
      x1: fromNode.x + ux * NODE_RADIUS + px * offset,
      y1: fromNode.y + uy * NODE_RADIUS + py * offset,
      x2: toNodeOrPoint.x - ux * NODE_RADIUS + px * offset,
      y2: toNodeOrPoint.y - uy * NODE_RADIUS + py * offset,
      mx: (fromNode.x + toNodeOrPoint.x) / 2 + px * offset,
      my: (fromNode.y + toNodeOrPoint.y) / 2 + py * offset,
    };
  }

  function computeEdgePath(edge) {
    const from = getNode(edge.from);
    const to = getNode(edge.to);
    const siblings = state.graph.edges
      .filter(
        (c) =>
          (c.from === edge.from && c.to === edge.to) ||
          (c.from === edge.to && c.to === edge.from)
      )
      .sort((a, b) => a.id.localeCompare(b.id));
    const idx = siblings.findIndex((c) => c.id === edge.id);
    const center = (siblings.length - 1) / 2;
    const offset = (idx - center) * 18;
    const pts = computeEdgePoints(from, to, offset);
    return `M ${pts.x1} ${pts.y1} Q ${pts.mx} ${pts.my} ${pts.x2} ${pts.y2}`;
  }

  function computeEdgeLabelPosition(edge, lane = 0, verticalOffset = 0) {
    const from = getNode(edge.from);
    const to = getNode(edge.to);
    const siblings = state.graph.edges
      .filter(
        (c) =>
          (c.from === edge.from && c.to === edge.to) ||
          (c.from === edge.to && c.to === edge.from)
      )
      .sort((a, b) => a.id.localeCompare(b.id));
    const idx = siblings.findIndex((c) => c.id === edge.id);
    const center = (siblings.length - 1) / 2;
    const offset = (idx - center) * 18;
    const pts = computeEdgePoints(from, to, offset);
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const len = Math.hypot(dx, dy) || 1;
    const perpX = -dy / len;
    const perpY = dx / len;
    const lo = lane * 13;
    return { x: pts.mx + perpX * lo, y: pts.my + perpY * lo - 8 + verticalOffset };
  }

  function getSvgPoint(event) {
    const matrix = els.svg.getScreenCTM();
    if (!matrix) return { x: 0, y: 0 };
    const pt = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    return { x: pt.x, y: pt.y };
  }

  function modeToNodeType(mode) { return mode.replace("add-", ""); }

  function getModeHint(mode) {
    const hints = {
      select: "选择模式：点击选中，拖动节点，空白处拖动画布，Delete/F 删除。",
      "add-In": "放置 In（输入）：点击空白处创建，点击已有节点仍可选中和拖动。",
      "add-Out": "放置 Out（输出）：点击空白处创建，点击已有节点仍可选中和拖动。",
      "add-S": "放置分流器 S：按 1 进入，点击空白处创建。",
      "add-C": "放置汇流器 C：按 2 进入，点击空白处创建。",
      edge: "连边模式：先点击起点，再点击终点；Esc 返回选择模式。",
    };
    return hints[mode] || "";
  }

  // ── 轻量 DOM 更新（拖拽时避免全量重绘） ─────────────────────

  function _updateNodeDom(nodeId) {
    const node = getNode(nodeId);
    const el = els.svg.querySelector(`[data-node-id="${nodeId}"]`);
    if (el) el.setAttribute("transform", `translate(${node.x}, ${node.y})`);
  }

  function _updateConnectedEdgesDom(nodeId) {
    const connected = state.graph.edges.filter(
      (e) => e.from === nodeId || e.to === nodeId);
    for (const edge of connected) {
      const path = computeEdgePath(edge);
      // 更新主路径（.edge-line）和点击区域（.edge-hit）
      const group = els.svg.querySelector(`[data-edge-id="${edge.id}"]`);
      if (!group) continue;
      const hit = group.querySelector(".edge-hit");
      const line = group.querySelector(".edge-line");
      if (hit) hit.setAttribute("d", path);
      if (line) line.setAttribute("d", path);
    }
  }

  function findFirstEmptySlot(size, occupiedSlots) {
    const occupied = new Set(occupiedSlots);
    for (let i = 0; i < size; i++) {
      if (!occupied.has(i)) return i;
    }
    return -1;
  }

  function nextId(kind) {
    const next = (state.counters[kind] ?? 0) + 1;
    state.counters[kind] = next;
    return `${kind}${next}`;
  }

  function buildCounters(graph) {
    const counters = {};
    for (const item of [...graph.nodes, ...graph.edges]) {
      const m = item.id.match(/^([a-zA-Z]+)(\d+)$/);
      if (!m) continue;
      counters[m[1]] = Math.max(counters[m[1]] ?? 0, Number(m[2]));
    }
    return counters;
  }

  function getNode(nodeId) {
    const node = state.graph.nodes.find((n) => n.id === nodeId);
    if (!node) throw new Error(`Unknown node ${nodeId}`);
    return node;
  }

  function cloneGraph(graph) {
    return {
      nodes: graph.nodes.map((n) => ({ ...n })),
      edges: graph.edges.map((e) => ({ ...e })),
    };
  }

  // ── 存档 ──────────────────────────────────────────────────────

  function renderSaveControls() {
    els.saveList.innerHTML = state.saves
      .map(
        (s) =>
          `<option value="${s.id}" ${s.id === state.activeSaveId ? "selected" : ""}>${escapeHtml(s.name)}</option>`
      )
      .join("");
  }

  function syncSelectedSaveName() {
    const rec = state.saves.find((s) => s.id === state.activeSaveId) || null;
    if (rec) els.saveName.value = rec.name;
  }

  function saveCurrentGraph() {
    const name = (els.saveName.value || "").trim();
    if (!name) { state.message = "请输入存档名。"; render(); return; }
    const existing = state.saves.find((s) => s.name === name) || null;
    const id = existing?.id || `save-${Date.now()}-${Math.floor(Math.random() * 100000)}`;
    state.saves = [
      ...state.saves.filter((s) => s.id !== id),
      { id, name, graph: cloneGraph(state.graph) },
    ].sort((a, b) => a.name.localeCompare(b.name));
    state.activeSaveId = id;
    persistSaves();
    state.message = existing ? `已覆盖本地存档: ${name}` : `图已保存到本地存档: ${name}`;
    render();
  }

  function loadSelectedGraph() {
    const rec = state.saves.find((s) => s.id === els.saveList.value);
    if (!rec) { state.message = "请选择要加载的图。"; render(); return; }
    state.graph = cloneGraph(rec.graph);
    state.counters = buildCounters(state.graph);
    state.activeSaveId = rec.id;
    state.selected = state.graph.nodes[0] ? { kind: "node", id: state.graph.nodes[0].id } : null;
    state.viewport = { ...DEFAULT_VIEWPORT };
    rebuildSimulator(`已加载图: ${rec.name}`);
  }

  function renameSelectedGraph() {
    const rec = state.saves.find((s) => s.id === els.saveList.value);
    const name = (els.saveName.value || "").trim();
    if (!rec) { state.message = "请选择要重命名的图。"; render(); return; }
    if (!name) { state.message = "请输入新的图名称。"; render(); return; }
    rec.name = name;
    state.saves.sort((a, b) => a.name.localeCompare(b.name));
    state.activeSaveId = rec.id;
    persistSaves();
    state.message = `已重命名为: ${name}`;
    render();
  }

  function deleteSelectedSave() {
    const rec = state.saves.find((s) => s.id === els.saveList.value);
    if (!rec) { state.message = "请选择要删除的图。"; render(); return; }
    state.saves = state.saves.filter((s) => s.id !== rec.id);
    state.activeSaveId = state.saves[0]?.id || null;
    persistSaves();
    state.message = `已删除图: ${rec.name}`;
    render();
  }

  function clearGraphToEndpoints() {
    state.graph = cloneGraph(BLANK_GRAPH);
    state.counters = buildCounters(state.graph);
    state.selected = { kind: "node", id: "In1" };
    state.viewport = { ...DEFAULT_VIEWPORT };
    state.draftEdgeFrom = null;
    rebuildSimulator("已清空为仅起点和终点。");
  }

  function loadSavesFromStorage() {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      state.saves = raw ? JSON.parse(raw) : [];
      state.activeSaveId = state.saves[0]?.id || null;
    } catch {
      state.saves = [];
      state.activeSaveId = null;
    }
  }

  function persistSaves() {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state.saves));
  }

  // ── 导入 JSON 图 ─────────────────────────────────────────────

  /** 暴露给内联按钮的导入函数 */
  window._importGraph = function _importGraph(graph) {
    const ng = { nodes: [], edges: [] };
    for (const n of graph.nodes) {
      ng.nodes.push({ id: n.id, type: n.type, x: n.x, y: n.y });
    }
    for (const e of graph.edges) {
      const fromSlot = e.fromSlot ?? findFirstEmptySlot(
        NODE_CAPACITY[_getNodeType(e.from, ng)].maxOut,
        ng.edges.filter((c) => c.from === e.from).map((c) => c.fromSlot ?? 0)
      );
      const toSlot = e.toSlot ?? findFirstEmptySlot(
        NODE_CAPACITY[_getNodeType(e.to, ng)].maxIn,
        ng.edges.filter((c) => c.to === e.to).map((c) => c.toSlot ?? 0)
      );
      ng.edges.push({ id: e.id, from: e.from, to: e.to, fromSlot, toSlot });
    }
    state.graph = ng;
    state.counters = buildCounters(ng);
    state.selected = ng.nodes.length > 0 ? { kind: "node", id: ng.nodes[0].id } : null;
    state.viewport = { ...DEFAULT_VIEWPORT };
    rebuildSimulator(`已导入图`);
  };

  function onImportJson() {
    const raw = (els.importJsonInput.value || "").trim();
    if (!raw) {
      els.importStatus.textContent = "请先粘贴 JSON。";
      return;
    }
    els.importStatus.textContent = "解析中...";
    let graph;
    try {
      const data = JSON.parse(raw);
      if (Array.isArray(data.nodes) && Array.isArray(data.edges)) {
        graph = data;
      } else if (data.graph && Array.isArray(data.graph.nodes)) {
        graph = data.graph;
      } else if (data.ratios && typeof data.ratios === "object") {
        const entries = Object.values(data.ratios);
        if (entries.length === 0) throw new Error("ratios 为空");
        graph = entries[0]?.graph;
        if (!graph) throw new Error("ratios 首条无 graph 字段");
      } else {
        throw new Error("格式不识别，需 {nodes:[],edges:[]}");
      }
      if (!Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
        throw new Error("nodes/edges 非数组");
      }
    } catch (e) {
      els.importStatus.textContent = `解析失败：${e.message}`;
      return;
    }
    // 构建新图
    const ng = { nodes: [], edges: [] };
    for (const n of graph.nodes) {
      ng.nodes.push({ id: n.id, type: n.type, x: n.x, y: n.y });
    }
    for (const e of graph.edges) {
      const fromSlot = e.fromSlot ?? findFirstEmptySlot(
        NODE_CAPACITY[_getNodeType(e.from, ng)].maxOut,
        ng.edges.filter((c) => c.from === e.from).map((c) => c.fromSlot ?? 0)
      );
      const toSlot = e.toSlot ?? findFirstEmptySlot(
        NODE_CAPACITY[_getNodeType(e.to, ng)].maxIn,
        ng.edges.filter((c) => c.to === e.to).map((c) => c.toSlot ?? 0)
      );
      ng.edges.push({ id: e.id, from: e.from, to: e.to, fromSlot, toSlot });
    }
    state.graph = ng;
    state.counters = buildCounters(ng);
    state.selected = ng.nodes.length > 0 ? { kind: "node", id: ng.nodes[0].id } : null;
    state.viewport = { ...DEFAULT_VIEWPORT };
    els.importStatus.textContent = `已导入（${ng.nodes.length}节点 / ${ng.edges.length}条边）`;
    rebuildSimulator(`已导入图`);
  }

  // ── wg 格式导入/导出 ─────────────────────────────────────────

  /** 推断 wg 节点类型：前缀 S/C 或全名 In/Out。 */
  function inferWgType(id, deg) {
    if (id.startsWith("In")) return "In";
    if (id.startsWith("Out")) return "Out";
    if (id.startsWith("S")) return "S";
    if (id.startsWith("C")) return "C";
    // 非标准命名（无 In/Out/S/C 前缀）：按出入度推断
    return inferTypeFromDegree(deg);
  }

  /** 根据节点的出入度推断类型（In/Out/S/C），不匹配返回 null。 */
  function inferTypeFromDegree(deg) {
    if (deg.in === 0 && deg.out === 1) return "In";     // 无入一出一 → 入口
    if (deg.in === 1 && deg.out === 0) return "Out";    // 一入无出 → 出口
    if (deg.in === 1 && deg.out >= 1 && deg.out <= 3) return "S"; // 一入 1~3 出 → 分流器
    if (deg.in >= 2 && deg.in <= 3 && deg.out === 1) return "C";  // 2~3 入一出 → 汇流器
    return null; // 其他情况视为无法推断，交由上层报异常
  }

  /** 把当前图导出为 wg 文本（每行一条 "from -> to"）。 */
  function graphToWg(graph) {
    const lines = [];
    for (const e of graph.edges) {
      lines.push(`${e.from} -> ${e.to}`);
    }
    return lines.join("\n");
  }

  /** 解析 wg 文本为 {nodes, edges} 图（自动推断类型并分配坐标）。 */
  function wgToGraph(text) {
    // 第一遍：收集边并统计每个节点的出入度（供非标准命名节点推断类型）
    const degreeMap = new Map();
    const edges = [];
    let edgeIdx = 0;
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.trim();
      if (!line || line.startsWith("#")) continue;
      if (!line.includes("->")) continue;
      const [from, to] = line.split("->", 2).map((s) => s.trim());
      if (!from || !to) continue;
      for (const nid of [from, to]) {
        if (!degreeMap.has(nid)) degreeMap.set(nid, { in: 0, out: 0 });
      }
      degreeMap.get(from).out += 1;
      degreeMap.get(to).in += 1;
      edges.push({ id: `e${edgeIdx++}`, from, to });
    }
    if (edges.length === 0) throw new Error("没有解析到任何 wg 边");
    // 第二遍：推断类型并分配坐标
    const nodes = [];
    let i = 0;
    for (const [id, deg] of degreeMap) {
      const type = inferWgType(id, deg);
      if (!type) throw new Error(`无法推断节点类型: ${id}（入度 ${deg.in} / 出度 ${deg.out}）`);
      const col = i % 4;
      const row = Math.floor(i / 4);
      nodes.push({ id, type, x: 80 + col * 110, y: 80 + row * 110 });
      i++;
    }
    return { nodes, edges };
  }

  function onWgExport() {
    if (state.graph.edges.length === 0) {
      els.wgStatus.textContent = "当前图为空，无可导出的边。";
      return;
    }
    els.wgInput.value = graphToWg(state.graph);
    els.wgStatus.textContent =
      `已导出 ${state.graph.edges.length} 条边到上方文本框，可复制分享。`;
  }

  function onWgImport() {
    const text = (els.wgInput.value || "").trim();
    if (!text) {
      els.wgStatus.textContent = "请先粘贴 wg 文本。";
      return;
    }
    els.wgStatus.textContent = "解析中…";
    let graph;
    try {
      graph = wgToGraph(text);
    } catch (e) {
      els.wgStatus.textContent = `解析失败: ${e.message}`;
      return;
    }
    window._importGraph(graph);
    els.wgStatus.textContent =
      `已导入（${graph.nodes.length}节点 / ${graph.edges.length}条边）`;
  }

  /** 把 wg 文本中的图【合并】进现有图：冲突节点重命名、整体放到最右侧，不覆盖现有内容。 */
  function onWgImportMerge() {
    const text = (els.wgInput.value || "").trim();
    if (!text) {
      els.wgStatus.textContent = "请先粘贴 wg 文本。";
      return;
    }
    let parsed;
    try {
      parsed = wgToGraph(text);
    } catch (e) {
      els.wgStatus.textContent = `解析失败: ${e.message}`;
      return;
    }
    if (state.graph.edges.length === 0 && state.graph.nodes.length === 0) {
      // 空图退化为直接导入
      window._importGraph(parsed);
      els.wgStatus.textContent =
        `当前图为空，已直接导入（${parsed.nodes.length}节点 / ${parsed.edges.length}条边）`;
      return;
    }

    // 节点重命名，避免与现有图 id 冲突
    const existingIds = new Set(state.graph.nodes.map((n) => n.id));
    const renameMap = {};
    const newNodes = parsed.nodes.map((n) => {
      let newId = n.id;
      if (existingIds.has(newId)) {
        let k = 2;
        while (existingIds.has(`${n.id}_${k}`)) k++;
        newId = `${n.id}_${k}`;
      }
      renameMap[n.id] = newId;
      existingIds.add(newId);
      return { id: newId, type: n.type, x: n.x, y: n.y };
    });

    // 新图整体右移到现有图最右侧（保持内部相对布局）
    const minNewX = Math.min(...parsed.nodes.map((n) => n.x));
    const maxCurX = state.graph.nodes.reduce((m, n) => Math.max(m, n.x), 0);
    const shiftX = maxCurX + 140 - minNewX;
    for (const n of newNodes) n.x += shiftX;

    // 新边：重命名端点 + 唯一 id + 分配输入/输出槽位
    const mergedNodes = [...state.graph.nodes, ...newNodes];
    const getType = (id) => _getNodeType(id, { nodes: mergedNodes });
    const usedEdgeIds = new Set(state.graph.edges.map((e) => e.id));
    const newEdges = parsed.edges.map((e, i) => {
      let id = e.id || `m${i}`;
      let k = 2;
      while (usedEdgeIds.has(id)) id = `m${i}_${k++}`;
      usedEdgeIds.add(id);
      const from = renameMap[e.from] ?? e.from;
      const to = renameMap[e.to] ?? e.to;
      const fromSlot = findFirstEmptySlot(
        NODE_CAPACITY[getType(from)].maxOut,
        state.graph.edges.filter((c) => c.from === from).map((c) => c.fromSlot ?? 0)
      );
      const toSlot = findFirstEmptySlot(
        NODE_CAPACITY[getType(to)].maxIn,
        state.graph.edges.filter((c) => c.to === to).map((c) => c.toSlot ?? 0)
      );
      return { id, from, to, fromSlot, toSlot };
    });

    // 合并进现有图
    state.graph.nodes.push(...newNodes);
    state.graph.edges.push(...newEdges);
    state.counters = buildCounters(state.graph);
    state.selected = newNodes.length > 0 ? { kind: "node", id: newNodes[0].id } : state.selected;
    state.viewport = { ...DEFAULT_VIEWPORT };
    rebuildSimulator("已向现有图加入 wg 内容");
    els.wgStatus.textContent =
      `已加入（${newNodes.length}节点 / ${newEdges.length}条边）到现有图右侧`;
  }

  function _getNodeType(nodeId, graph) {
    const n = graph.nodes.find((n) => n.id === nodeId);
    return n ? n.type : "S";
  }

  // ── 工具函数 ──────────────────────────────────────────────────

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function isEditableTarget(target) {
    return target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement;
  }

  function blurActiveElement() {
    const active = document.activeElement;
    if (active instanceof HTMLElement) active.blur();
  }

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function clampInt(value, min, max, fallback) {
    const n = parseInt(value, 10);
    if (!Number.isFinite(n)) return fallback;
    return Math.min(Math.max(n, min), max);
  }

  let toastTimer = null;
  function showToast(message, kind = "info") {
    let toast = document.getElementById("app-toast");
    if (!toast) {
      toast = document.createElement("div");
      toast.id = "app-toast";
      toast.className = "app-toast";
      document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.classList.remove("app-toast-error", "app-toast-info");
    toast.classList.add(kind === "error" ? "app-toast-error" : "app-toast-info");
    toast.classList.add("app-toast-visible");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("app-toast-visible"), 5000);
  }

  // ── 启动 ──────────────────────────────────────────────────────
  main();
})();
