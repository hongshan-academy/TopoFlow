(function () {

  const GRAPH_BOUNDS = { width: 2700, height: 1680, padding: 36 };
  const SAFE_CLAMP = { min: -32768, max: 32767 };
  const DEFAULT_VIEWPORT = { x: 0, y: 0, width: 900, height: 560 };
  const NODE_RADIUS = 28;
  const HIGH_FPS_THRESHOLD = 120;
  const HIGH_FPS_RENDER_INTERVAL = 120;
  const STORAGE_KEY = "topoflow-sim-graphs-v1";
  const MIN_VIEW_SCALE = 1.05;
  const MAX_VIEW_SCALE = 9;
  const MODE_LABELS = {
    select: "选择",
    "add-source": "放置起点",
    "add-sink": "放置终点",
    "add-splitter": "放置分流器",
    "add-merger": "放置汇流器",
    edge: "连边",
  };
  const TYPE_LABELS = {
    source: "起点",
    sink: "终点",
    splitter: "分流器",
    merger: "汇流器",
  };
  const NODE_COLORS = {
    source: "#2e8b57",
    sink: "#c84c3a",
    splitter: "#2f6fed",
    merger: "#d98a1f",
  };
  const NODE_CAPACITY = {
    source: { maxIn: 0, maxOut: 1 },
    sink: { maxIn: 1, maxOut: 0 },
    splitter: { maxIn: 1, maxOut: 3 },
    merger: { maxIn: 3, maxOut: 1 },
  };
  const INITIAL_GRAPH = {
    nodes: [
      { id: "source1", type: "source", x: 90, y: 280 },
      { id: "merger1", type: "merger", x: 270, y: 280 },
      { id: "splitter1", type: "splitter", x: 450, y: 280 },
      { id: "sink1", type: "sink", x: 760, y: 280 },
      { id: "sink2", type: "sink", x: 760, y: 430 },
    ],
    edges: [
      { id: "edge1", from: "source1", to: "merger1", toSlot: 0 },
      { id: "edge2", from: "merger1", to: "splitter1" },
      { id: "edge3", from: "splitter1", to: "sink1", fromSlot: 0 },
      { id: "edge4", from: "splitter1", to: "sink2", fromSlot: 1 },
    ],
  };
  const BLANK_GRAPH = {
    nodes: [
      { id: "source1", type: "source", x: 180, y: 280 },
      { id: "sink1", type: "sink", x: 720, y: 280 },
    ],
    edges: [],
  };

  const state = {
    graph: cloneGraph(INITIAL_GRAPH),
    frames: [],
    simFrame: 0,
    cycleInfo: null,
    selected: { kind: "node", id: "merger1" },
    mode: "select",
    fps: 100,
    runner: null,
    draftEdgeFrom: null,
    pointerGraph: { x: 0, y: 0 },
    drag: null,
    viewport: { ...DEFAULT_VIEWPORT },
    pan: null,
    blankHold: false,
    showEdgeFlowLabels: true,
    autoSolveEnabled: true,
    autoSolveFrames: 100000,
    continuousSolveEnabled: true,
    continuousSolution: null,
    continuousSolutions: [],
    activeSolutionIndex: 0,
    continuousSolveVersion: 0,
    provedInfeasible: false,
    counters: buildCounters(INITIAL_GRAPH),
    saves: [],
    activeSaveId: null,
    message: "选择模式下可选中、拖动和删除节点。",
  };

  const els = {
    svg: document.getElementById("graph"),
    frame: document.getElementById("frame-value"),
    fps: document.getElementById("fps-input"),
    fpsLabel: document.getElementById("fps-value"),
    play: document.getElementById("play-btn"),
    pause: document.getElementById("pause-btn"),
    step: document.getElementById("step-btn"),
    reset: document.getElementById("reset-btn"),
    clear: document.getElementById("clear-btn"),
    save: document.getElementById("save-btn"),
    load: document.getElementById("load-btn"),
    rename: document.getElementById("rename-btn"),
    removeSave: document.getElementById("delete-save-btn"),
    showEdgeFlow: document.getElementById("show-edge-flow-toggle"),
    autoSolveToggle: document.getElementById("auto-solve-toggle"),
    autoSolveFrames: document.getElementById("auto-solve-frames"),
    continuousSolveToggle: document.getElementById("continuous-solve-toggle"),
    prevSolutionBtn: document.getElementById("prev-solution-btn"),
    nextSolutionBtn: document.getElementById("next-solution-btn"),
    solutionNav: document.getElementById("solution-nav"),
    solutionIndex: document.getElementById("solution-index"),
    saveName: document.getElementById("save-name-input"),
    saveList: document.getElementById("save-list"),
    exportBtn: document.getElementById("export-btn"),
    importBtn: document.getElementById("import-btn"),
    importFileInput: document.getElementById("import-file-input"),
    modeValue: document.getElementById("mode-value"),
    graphStats: document.getElementById("graph-stats"),
    selected: document.getElementById("selected-node"),
    type: document.getElementById("selected-type"),
    endpoints: document.getElementById("selected-endpoints"),
    hasItem: document.getElementById("selected-item"),
    ratio: document.getElementById("selected-ratio"),
    cycle: document.getElementById("selected-cycle"),
    history: document.getElementById("selected-history"),
    historyMeta: document.getElementById("history-meta"),
    edgeDraft: document.getElementById("edge-draft"),
    toolButtons: [...document.querySelectorAll("[data-mode-button]")],
    statusText: document.getElementById("status-text"),
  };

  function main() {
    loadSavesFromStorage();
    bindEvents();
    fetchConfig();
    rebuildSimulator("已载入示例图。");
    render();
  }

  async function fetchConfig() {
    try {
      var resp = await fetch("/api/config");
      var cfg = await resp.json();
      if (cfg.max_frames) {
        state.autoSolveFrames = cfg.max_frames;
        els.autoSolveFrames.value = String(cfg.max_frames);
      }
    } catch (_) {}
  }

  function bindEvents() {
    els.play.addEventListener("click", startRunning);
    els.pause.addEventListener("click", stopRunning);
    els.step.addEventListener("click", () => {
      if (!state.frames.length) {
        return;
      }
      stepOnceLocal();
      render();
    });
    els.reset.addEventListener("click", () => {
      stopRunning();
      rebuildSimulator("模拟已重置。");
      render();
    });
    els.clear.addEventListener("click", clearGraphToEndpoints);
    els.showEdgeFlow.addEventListener("change", () => {
      state.showEdgeFlowLabels = els.showEdgeFlow.checked;
      render();
    });
    els.autoSolveToggle.addEventListener("change", () => {
      state.autoSolveEnabled = els.autoSolveToggle.checked;
      render();
    });
    els.continuousSolveToggle.addEventListener("change", () => {
      state.continuousSolveEnabled = els.continuousSolveToggle.checked;
      state.continuousSolution = null;
      state.continuousSolutions = [];
      state.activeSolutionIndex = 0;
      state.provedInfeasible = false;
      state.continuousSolveVersion += 1;
      if (state.continuousSolveEnabled) {
        requestContinuousSolve(state.continuousSolveVersion);
      } else {
        state.message = "已关闭连续求解器。";
        render();
      }
    });
    els.prevSolutionBtn.addEventListener("click", prevSolution);
    els.nextSolutionBtn.addEventListener("click", nextSolution);
    els.autoSolveFrames.addEventListener("input", () => {
      const value = Number.parseInt(els.autoSolveFrames.value || "0", 10);
      state.autoSolveFrames = Number.isFinite(value) && value >= 1 ? value : 100000;
    });
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
    els.exportBtn.addEventListener("click", exportGraphAsText);
    els.importBtn.addEventListener("click", () => els.importFileInput.click());
    els.importFileInput.addEventListener("change", importGraphFromFile);
    els.fps.addEventListener("input", () => {
      state.fps = Number(els.fps.value);
      els.fpsLabel.textContent = String(state.fps);
      if (state.runner !== null) {
        stopRunning();
        startRunning();
      }
    });

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
  }

  function onKeyDown(event) {
    if (isEditableTarget(event.target)) {
      return;
    }

    if (event.key === "1") {
      event.preventDefault();
      setMode("add-splitter");
      return;
    }
    if (event.key === "2") {
      event.preventDefault();
      setMode("add-merger");
      return;
    }
    if (event.key === "3") {
      event.preventDefault();
      setMode("add-source");
      return;
    }
    if (event.key === "4") {
      event.preventDefault();
      setMode("add-sink");
      return;
    }
    if (event.key.toLowerCase() === "e") {
      event.preventDefault();
      setMode("edge");
      return;
    }
    if (event.key.toLowerCase() === "x") {
      event.preventDefault();
      setMode("select");
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      setMode("select");
      return;
    }
    if (
      event.key === "Delete" ||
      event.key === "Backspace" ||
      event.key.toLowerCase() === "f"
    ) {
      if (state.selected) {
        event.preventDefault();
        deleteSelectedObject();
      }
    }
  }

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
      state.drag = {
        nodeId,
        offsetX: point.x - node.x,
        offsetY: point.y - node.y,
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
      const screenMatrix = els.svg.getScreenCTM();
      state.blankHold = true;
      state.pan = {
        startPoint: point,
        startScreenInverse: screenMatrix ? screenMatrix.inverse() : null,
        startViewportX: state.viewport.x,
        startViewportY: state.viewport.y,
      };
      state.selected = null;
      state.message = "已取消选择，可拖动画布。";
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
    if (state.mode === "edge" && state.draftEdgeFrom) {
      render();
    }
  }

  function onDocumentPointerMove(event) {
    if (state.drag) {
      const point = getSvgPoint(event);
      moveNode(state.drag.nodeId, {
        x: point.x - state.drag.offsetX,
        y: point.y - state.drag.offsetY,
      });
      render();
      return;
    }

    if (!state.pan) {
      return;
    }
    const point = state.pan.startScreenInverse
      ? new DOMPoint(event.clientX, event.clientY).matrixTransform(state.pan.startScreenInverse)
      : getSvgPoint(event);
    state.viewport.x = state.pan.startViewportX - (point.x - state.pan.startPoint.x);
    state.viewport.y = state.pan.startViewportY - (point.y - state.pan.startPoint.y);
    render();
  }

  function onDocumentPointerUp() {
    state.drag = null;
    state.pan = null;
    state.blankHold = false;
  }

  function onSvgWheel(event) {
    if (!state.blankHold) {
      return;
    }

    event.preventDefault();
    const rect = els.svg.getBoundingClientRect();
    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;
    const anchorX = state.viewport.x + (localX / rect.width) * state.viewport.width;
    const anchorY = state.viewport.y + (localY / rect.height) * state.viewport.height;
    const currentScale = GRAPH_BOUNDS.width / state.viewport.width;
    const zoomFactor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    const nextScale = clamp(currentScale * zoomFactor, MIN_VIEW_SCALE, MAX_VIEW_SCALE);
    const nextWidth = GRAPH_BOUNDS.width / nextScale;
    const nextHeight = GRAPH_BOUNDS.height / nextScale;
    const ratioX = localX / rect.width;
    const ratioY = localY / rect.height;

    state.viewport.width = nextWidth;
    state.viewport.height = nextHeight;
    state.viewport.x = anchorX - ratioX * nextWidth;
    state.viewport.y = anchorY - ratioY * nextHeight;
    render();
  }

  function onSvgContextMenu(event) {
    event.preventDefault();
    blurActiveElement();
    if (state.mode !== "select") {
      setMode("select");
    }
  }

  function startRunning() {
    if (state.runner !== null || !state.frames.length) {
      return;
    }
    state.runner = {
      rafId: 0,
      lastTime: performance.now(),
      carryFrames: 0,
      lastRenderTime: 0,
    };
    state.runner.rafId = window.requestAnimationFrame(runLoop);
    state.message = "模拟运行中。";
    render();
  }

  function runLoop(now) {
    if (!state.runner || !state.frames.length) {
      return;
    }

    const elapsedMs = Math.max(0, now - state.runner.lastTime);
    state.runner.lastTime = now;
    state.runner.carryFrames += (elapsedMs * state.fps) / 1000;
    const framesToAdvance = Math.min(5000, Math.floor(state.runner.carryFrames));
    if (framesToAdvance > 0) {
      state.runner.carryFrames -= framesToAdvance;
      stepLocal(framesToAdvance);
    }

    const shouldRender =
      state.fps <= HIGH_FPS_THRESHOLD || now - state.runner.lastRenderTime >= HIGH_FPS_RENDER_INTERVAL;
    if (shouldRender) {
      state.runner.lastRenderTime = now;
      render();
    }

    state.runner.rafId = window.requestAnimationFrame(runLoop);
  }

  function stopRunning() {
    if (state.runner === null) {
      return;
    }
    window.cancelAnimationFrame(state.runner.rafId);
    state.runner = null;
    state.message = "模拟已暂停。";
    render();
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
      x: clamp(point.x, SAFE_CLAMP.min, SAFE_CLAMP.max),
      y: clamp(point.y, SAFE_CLAMP.min, SAFE_CLAMP.max),
    };
    state.graph.nodes.push(node);
    state.selected = { kind: "node", id: node.id };
    rebuildSimulator(`${TYPE_LABELS[type]} ${node.id} 已添加。`);
    render();
  }

  function moveNode(nodeId, point) {
    const node = getNode(nodeId);
    node.x = clamp(point.x, SAFE_CLAMP.min, SAFE_CLAMP.max);
    node.y = clamp(point.y, SAFE_CLAMP.min, SAFE_CLAMP.max);
  }

  function deleteSelectedObject() {
    if (!state.selected) {
      return;
    }

    stopRunning();
    if (state.selected.kind === "node") {
      const nodeId = state.selected.id;
      state.drag = null;
      state.pan = null;
      if (state.draftEdgeFrom === nodeId) {
        state.draftEdgeFrom = null;
      }
      state.graph.nodes = state.graph.nodes.filter((node) => node.id !== nodeId);
      state.graph.edges = state.graph.edges.filter((edge) => edge.from !== nodeId && edge.to !== nodeId);
      state.selected = state.graph.nodes[0] ? { kind: "node", id: state.graph.nodes[0].id } : null;
      rebuildSimulator(`节点 ${nodeId} 已删除。`);
    } else {
      const edgeId = state.selected.id;
      state.graph.edges = state.graph.edges.filter((edge) => edge.id !== edgeId);
      state.selected = state.graph.nodes[0] ? { kind: "node", id: state.graph.nodes[0].id } : null;
      rebuildSimulator(`边 ${edgeId} 已删除。`);
    }
    render();
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

    if (fromId === toId) {
      return { ok: false, message: "不允许创建自环。" };
    }

    const fromShape = NODE_CAPACITY[fromNode.type];
    const toShape = NODE_CAPACITY[toNode.type];
    if (fromShape.maxOut === 0) {
      return { ok: false, message: `${fromNode.id} 不能作为边的起点。` };
    }
    if (toShape.maxIn === 0) {
      return { ok: false, message: `${toNode.id} 不能作为边的终点。` };
    }

    const fromSlot = findFirstEmptySlot(
      fromShape.maxOut,
      state.graph.edges.filter((edge) => edge.from === fromId).map((edge) => edge.fromSlot ?? 0)
    );
    if (fromSlot === -1) {
      return { ok: false, message: `${fromNode.id} 的输出口已满。` };
    }

    const toSlot = findFirstEmptySlot(
      toShape.maxIn,
      state.graph.edges.filter((edge) => edge.to === toId).map((edge) => edge.toSlot ?? 0)
    );
    if (toSlot === -1) {
      return { ok: false, message: `${toNode.id} 的输入口已满。` };
    }

    stopRunning();
    state.graph.edges.push({
      id: nextId("edge"),
      from: fromId,
      to: toId,
      fromSlot,
      toSlot,
    });
    rebuildSimulator(`已连接 ${fromId} -> ${toId}。`);
    render();
    return { ok: true, message: `已连接 ${fromId} -> ${toId}。` };
  }

  function rebuildSimulator(message) {
    stopRunning();
    state.frames = [];
    state.simFrame = 0;
    state.cycleInfo = null;
    state.continuousSolution = null;
    state.continuousSolutions = [];
    state.activeSolutionIndex = 0;
    state.provedInfeasible = false;
    state.continuousSolveVersion += 1;
    ensureSelectedObject();
    state.message = message;
    if (state.autoSolveEnabled) {
      requestDiscreteSimulate(message);
    } else {
      render();
    }
    if (state.continuousSolveEnabled) {
      requestContinuousSolve(state.continuousSolveVersion);
    }
  }

  function buildSimulateRequest() {
    const connectedIds = new Set();
    state.graph.edges.forEach(function (e) { connectedIds.add(e.from); connectedIds.add(e.to); });
    const activeNodes = state.graph.nodes.filter(function (n) { return connectedIds.has(n.id); });
    const activeEdges = state.graph.edges;
    return {
      nodes: activeNodes.map(function (n) { return { node_id: n.id, node_type: n.type, x: n.x, y: n.y }; }),
      edges: activeEdges.map(function (e) { return { id: e.id, from: e.from, to: e.to }; }),
      options: { max_frames: state.autoSolveFrames || null },
    };
  }

  async function requestDiscreteSimulate(message) {
    state.message = "离散模拟中...";
    render();
    try {
      const response = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildSimulateRequest()),
      });
      const data = await response.json();
      if (!response.ok || data.error) {
        if (!state.autoSolveEnabled) return;
        state.frames = [];
        state.simFrame = 0;
        state.cycleInfo = null;
        state.message = data.error || "模拟请求失败";
        render();
        return;
      }
      if (!state.autoSolveEnabled) {
        return;
      }
      state.frames = data.frames || [];
      state.simFrame = 0;
      state.cycleInfo = data.cycleInfo || null;
      if (state.cycleInfo) {
        const ci = state.cycleInfo;
        state.message = `${message} 周期 ${ci.period} 帧，暖机 ${ci.warmupFrames} 帧。`;
      } else {
        state.message = message;
      }
    } catch (error) {
      if (!state.autoSolveEnabled) {
        return;
      }
      state.frames = [];
      state.simFrame = 0;
      state.cycleInfo = null;
      state.message = "离散模拟失败：" + error.message;
    }
    render();
  }

  async function requestContinuousSolve(version) {
    state.message = "连续求解中...";
    render();
    try {
      var connectedIds = new Set();
      state.graph.edges.forEach(function (e) { connectedIds.add(e.from); connectedIds.add(e.to); });
      var activeNodes = state.graph.nodes.filter(function (n) { return connectedIds.has(n.id); });
      var activeEdges = state.graph.edges;
      const response = await fetch("/api/solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          nodes: activeNodes.map(function (n) { return { node_id: n.id, node_type: n.type, x: n.x, y: n.y }; }),
          edges: activeEdges.map(function (e) { return { id: e.id, from: e.from, to: e.to }; }),
        }),
      });
      const data = await response.json();
      if (!response.ok || !data.feasible) {
        throw new Error(data.error || "求解失败");
      }
      if (version !== state.continuousSolveVersion || !state.continuousSolveEnabled) {
        return;
      }

      state.continuousSolutions = (data.solutions || []).map(function (sol) {
        var edgeFlowEntries = (sol.edgeFlows || []).map(function (item) {
          return [item.id, item.flow];
        });
        var edgeBlockedEntries = (sol.edgeFlows || []).map(function (item) {
          return [item.id, item.isBlocked];
        });
        var nodeFlowEntries = (sol.nodeFlows || []).map(function (item) {
          return [item.id, item.flow];
        });
        return {
          edgeFlows: new Map(edgeFlowEntries),
          edgeBlocked: new Map(edgeBlockedEntries),
          nodeFlows: new Map(nodeFlowEntries),
        };
      });
      state.activeSolutionIndex = 0;
      state.continuousSolution = state.continuousSolutions[0] || null;
      state.provedInfeasible = data.provedInfeasible || false;

      var count = state.continuousSolutions.length;
      if (state.provedInfeasible) {
        state.message = count > 0
          ? "已证明仅有 " + count + " 种阻塞/满带组合。"
          : "已证明无可行解。";
      } else if (count > 1) {
        state.message = "找到 " + count + " 个解（可能还有更多）。";
      } else {
        state.message = "连续求解完成，流量已按近似分数显示。";
      }
    } catch (error) {
      if (version !== state.continuousSolveVersion || !state.continuousSolveEnabled) {
        return;
      }
      state.continuousSolutions = [];
      state.activeSolutionIndex = 0;
      state.continuousSolution = null;
      state.message = "连续求解失败：" + error.message;
    }
    render();
  }

  function prevSolution() {
    if (state.activeSolutionIndex > 0) {
      state.activeSolutionIndex -= 1;
      state.continuousSolution = state.continuousSolutions[state.activeSolutionIndex];
      render();
    }
  }

  function nextSolution() {
    if (state.activeSolutionIndex < state.continuousSolutions.length - 1) {
      state.activeSolutionIndex += 1;
      state.continuousSolution = state.continuousSolutions[state.activeSolutionIndex];
      render();
    }
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
    if (toastTimer) {
      clearTimeout(toastTimer);
    }
    toastTimer = setTimeout(() => {
      toast.classList.remove("app-toast-visible");
    }, 5000);
  }

  function ensureSelectedObject() {
    if (state.selected?.kind === "node" && state.graph.nodes.some((node) => node.id === state.selected.id)) {
      return;
    }
    if (state.selected?.kind === "edge" && state.graph.edges.some((edge) => edge.id === state.selected.id)) {
      return;
    }
    state.selected = state.graph.nodes[0] ? { kind: "node", id: state.graph.nodes[0].id } : null;
  }

  function render() {
    renderControls();
    renderGraph();
    renderSidebar();
  }

  function renderControls() {
    els.frame.textContent = String(state.simFrame);
    els.fpsLabel.textContent = String(state.fps);
    els.autoSolveToggle.checked = state.autoSolveEnabled;
    els.autoSolveFrames.value = String(state.autoSolveFrames);
    els.continuousSolveToggle.checked = state.continuousSolveEnabled;
    var solCount = state.continuousSolutions.length;
    if (solCount > 1 || state.provedInfeasible) {
      els.solutionNav.style.display = "flex";
      var label = "解 " + (state.activeSolutionIndex + 1) + " / " + solCount;
      if (state.provedInfeasible) label += "  ✓";
      els.solutionIndex.textContent = label;
    } else {
      els.solutionNav.style.display = "none";
    }
    if (els.modeValue)
      els.modeValue.textContent = MODE_LABELS[state.mode];
    if (els.graphStats)
      els.graphStats.textContent = `${state.graph.nodes.length} 个节点 / ${state.graph.edges.length} 条边`;
    if (els.edgeDraft)
      els.edgeDraft.textContent = state.draftEdgeFrom ? `起点: ${state.draftEdgeFrom}` : "起点: 未选择";
    renderSaveControls();
    els.statusText.textContent = state.message;

    for (const button of els.toolButtons) {
      const active = button.getAttribute("data-mode-button") === state.mode;
      button.classList.toggle("is-active", active);
    }
  }

  function renderGraph() {
    const snapshot = getCurrentSnapshot();
    const nodeMap = snapshot.nodes || {};
    const edgeMap = snapshot.edges || {};
    const showOccupiedEdges = !(state.runner && state.fps > HIGH_FPS_THRESHOLD);

    const edgesMarkup = state.graph.edges.map((edge) => {
      const pathInfo = computeEdgePath(edge);
      const edgeRt = edgeMap[edge.id];
      const hasItem = edgeRt && edgeRt.queue && edgeRt.queue.length > 0;
      const activeClass = showOccupiedEdges && hasItem ? " is-active" : "";
      const selectedClass =
        state.selected?.kind === "edge" && state.selected.id === edge.id ? " is-selected" : "";
      const blockedClass = state.continuousSolution?.edgeBlocked.get(edge.id) ? " is-blocked" : "";
      const edgeAnalysis =
        state.showEdgeFlowLabels && state.autoSolveEnabled && state.cycleInfo
          ? analyzeEdgeLocal(edge.id)
          : null;
      const continuousFlow = state.continuousSolution?.edgeFlows.get(edge.id) || null;
      const discreteLabel = edgeAnalysis ? computeEdgeLabelPosition(edge, continuousFlow ? -1 : 0) : null;
      const continuousLabel = continuousFlow ? computeEdgeLabelPosition(edge, edgeAnalysis ? 1 : 0, 10) : null;
      return `
        <g class="edge-group${activeClass}${selectedClass}${blockedClass}" data-edge-id="${edge.id}">
          <path class="edge-hit" d="${pathInfo.d}"></path>
          <path class="edge-line" d="${pathInfo.d}"></path>
          ${
            edgeAnalysis?.flowRatio && discreteLabel
               ? `<text class="edge-flow" x="${discreteLabel.x}" y="${discreteLabel.y}" text-anchor="middle">${edgeAnalysis.flowRatio.textReduced ?? edgeAnalysis.flowRatio.text}</text>`
              : ""
          }
          ${
            continuousFlow && continuousLabel
              ? `<text class="edge-flow edge-flow-continuous" x="${continuousLabel.x}" y="${continuousLabel.y}" text-anchor="middle">${continuousFlow.text}</text>`
              : ""
          }
        </g>
      `;
    });

    const draftMarkup = renderDraftEdge();

    const nodesMarkup = state.graph.nodes.map((node) => {
      const runtime = nodeMap[node.id] || { hasItem: false };
      const selectedClass =
        state.selected?.kind === "node" && state.selected.id === node.id ? " is-selected" : "";
      const busyClass = runtime.hasItem ? " has-item" : "";
      const edgeStartClass = state.draftEdgeFrom === node.id ? " is-edge-start" : "";
      const sinkAnalysis =
        node.type === "sink" && state.cycleInfo ? analyzeNodeLocal(node.id) : null;
      const continuousNodeFlow = node.type === "sink" ? state.continuousSolution?.nodeFlows.get(node.id) : null;
      return `
        <g
          class="node${selectedClass}${busyClass}${edgeStartClass}"
          data-node-id="${node.id}"
          transform="translate(${node.x}, ${node.y})"
        >
          <circle r="${NODE_RADIUS}" fill="${NODE_COLORS[node.type]}"></circle>
          <text text-anchor="middle" dy="-2">${node.id}</text>
          <text class="node-type" text-anchor="middle" dy="14">${TYPE_LABELS[node.type]}</text>
          ${
            sinkAnalysis?.flowRatio
               ? `<text class="node-flow" text-anchor="middle" dy="46">${sinkAnalysis.flowRatio.textReduced ?? sinkAnalysis.flowRatio.text}</text>`
              : ""
          }
          ${
            continuousNodeFlow
              ? `<text class="node-flow node-flow-continuous" text-anchor="middle" dy="59">${continuousNodeFlow.text}</text>`
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
    if (state.mode !== "edge" || !state.draftEdgeFrom) {
      return "";
    }
    const from = state.graph.nodes.find((node) => node.id === state.draftEdgeFrom);
    if (!from) {
      state.draftEdgeFrom = null;
      return "";
    }
    const points = computeEdgePoints(from, state.pointerGraph);
    return `
      <path class="edge-line draft-edge" d="M ${points.x1} ${points.y1} L ${points.x2} ${points.y2}"></path>
    `;
  }

  function renderSidebar() {
    if (!state.selected || !state.cycleInfo) {
      els.selected.textContent = state.selected ? state.selected.id : "-";
      els.type.textContent = "-";
      els.endpoints.textContent = "-";
      els.hasItem.textContent = "-";
      els.ratio.textContent = "-";
      els.cycle.textContent = "-";
      els.history.textContent = "-";
      els.historyMeta.textContent = "未求解";
      return;
    }

    const analysis =
      state.selected.kind === "node"
        ? analyzeNodeLocal(state.selected.id)
        : analyzeEdgeLocal(state.selected.id);
    const continuousFlow =
      state.selected.kind === "node"
        ? state.continuousSolution?.nodeFlows.get(state.selected.id)
        : state.continuousSolution?.edgeFlows.get(state.selected.id);

    const snapshot = getCurrentSnapshot();
    const hasItem = state.selected.kind === "node"
      ? Boolean(snapshot.nodes[state.selected.id]?.hasItem)
      : Boolean((snapshot.edges[state.selected.id]?.queue || []).length > 0);

    els.selected.textContent = analysis.id;
    if (state.selected.kind === "node") {
      els.type.textContent = TYPE_LABELS[analysis.type];
      els.endpoints.textContent = "-";
    } else {
      els.type.textContent = "传送带";
      els.endpoints.textContent = `${analysis.from} -> ${analysis.to}`;
    }
    els.hasItem.textContent = hasItem ? "1" : "0";
    els.ratio.textContent = analysis.flowRatio?.text ?? "-";
    els.cycle.textContent = analysis.cycleInfo ? String(analysis.cycleInfo.period) : "-";
    els.history.textContent = "-";

    const discreteMeta = analysis.cycleInfo
      ? `离散：${analysis.flowRatio.text} / 周期 ${analysis.cycleInfo.period} 帧 / ` +
        `暖机 ${analysis.cycleInfo.warmupFrames} 帧 / ` +
        `全长 ${analysis.cycleInfo.totalFrames} 帧`
      : `离散：${analysis.flowRatio?.text ?? "-"} / 未求解`;
    if (continuousFlow) {
      els.historyMeta.innerHTML =
        `${escapeHtml(discreteMeta)}<br>连续 MILP：${escapeHtml(continuousFlow.text)}`;
    } else {
      els.historyMeta.textContent = discreteMeta;
    }
  }

  function buildParallelIndex() {
    const groups = new Map();
    for (var i = 0; i < state.graph.edges.length; i++) {
      var edge = state.graph.edges[i];
      var a = edge.from;
      var b = edge.to;
      var key = a < b ? a + "<->" + b : b + "<->" + a;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(edge.id);
    }
    return groups;
  }

  function computeEdgePoints(fromNode, toNodeOrPoint, offset) {
    offset = offset || 0;
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
    var from = getNode(edge.from);
    var to = getNode(edge.to);

    var parallelGroups = buildParallelIndex();
    var lowId = edge.from < edge.to ? edge.from : edge.to;
    var highId = edge.from < edge.to ? edge.to : edge.from;
    var groupKey = lowId + "<->" + highId;
    var ids = parallelGroups.get(groupKey) || [edge.id];
    var idx = ids.indexOf(edge.id);
    var offset = (idx - (ids.length - 1) / 2) * 78;

    var low = getNode(lowId);
    var high = getNode(highId);
    var cdx = high.x - low.x;
    var cdy = high.y - low.y;
    var clen = Math.hypot(cdx, cdy) || 1;
    var px = -cdy / clen;
    var py = cdx / clen;

    var dx = to.x - from.x;
    var dy = to.y - from.y;
    var len = Math.hypot(dx, dy) || 1;
    var ux = dx / len;
    var uy = dy / len;

    var sx = from.x + ux * NODE_RADIUS;
    var sy = from.y + uy * NODE_RADIUS;
    var tx = to.x - ux * (NODE_RADIUS + 5);
    var ty = to.y - uy * (NODE_RADIUS + 5);
    var cx = (sx + tx) / 2 + px * offset;
    var cy = (sy + ty) / 2 + py * offset;

    var labelX = (sx + 2 * cx + tx) / 4 + px * 8;
    var labelY = (sy + 2 * cy + ty) / 4 + py * 8;

    return {
      d: "M " + sx + " " + sy + " Q " + cx + " " + cy + " " + tx + " " + ty,
      labelX: labelX,
      labelY: labelY,
    };
  }

  function computeEdgeLabelPosition(edge, lane, verticalOffset) {
    lane = lane || 0;
    verticalOffset = verticalOffset || 0;
    var pathInfo = computeEdgePath(edge);
    return {
      x: pathInfo.labelX + lane * 13,
      y: pathInfo.labelY - 8 + verticalOffset,
    };
  }

  function getSvgPoint(event) {
    const screenMatrix = els.svg.getScreenCTM();
    if (!screenMatrix) {
      return { x: 0, y: 0 };
    }
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(screenMatrix.inverse());
    return { x: point.x, y: point.y };
  }

  function modeToNodeType(mode) {
    return mode.replace("add-", "");
  }

  function getModeHint(mode) {
    switch (mode) {
      case "select":
        return "选择模式：点击选中，拖动节点，空白处拖动画布，Delete/F 删除。";
      case "add-source":
        return "放置起点：点击空白处创建，点击已有节点仍可选中和拖动。";
      case "add-sink":
        return "放置终点：点击空白处创建，点击已有节点仍可选中和拖动。";
      case "add-splitter":
        return "放置分流器：按 1 进入，点击空白处创建。";
      case "add-merger":
        return "放置汇流器：按 2 进入，点击空白处创建。";
      case "edge":
        return "连边模式：先点击起点，再点击终点；Esc 返回选择模式。";
      default:
        return "";
    }
  }

  function findFirstEmptySlot(size, occupiedSlots) {
    const occupied = new Set(occupiedSlots);
    for (let i = 0; i < size; i += 1) {
      if (!occupied.has(i)) {
        return i;
      }
    }
    return -1;
  }

  function nextId(kind) {
    const nextValue = (state.counters[kind] ?? 0) + 1;
    state.counters[kind] = nextValue;
    return `${kind}${nextValue}`;
  }

  function buildCounters(graph) {
    const counters = {};
    for (const item of [...graph.nodes, ...graph.edges]) {
      const match = item.id.match(/^([a-zA-Z]+)(\d+)$/);
      if (!match) {
        continue;
      }
      const key = match[1];
      const value = Number(match[2]);
      counters[key] = Math.max(counters[key] ?? 0, value);
    }
    return counters;
  }

  function getNode(nodeId) {
    const node = state.graph.nodes.find((item) => item.id === nodeId);
    if (!node) {
      throw new Error(`Unknown node ${nodeId}`);
    }
    return node;
  }

  function getEdge(edgeId) {
    const edge = state.graph.edges.find((item) => item.id === edgeId);
    if (!edge) {
      throw new Error(`Unknown edge ${edgeId}`);
    }
    return edge;
  }

  function cloneGraph(graph) {
    return {
      nodes: graph.nodes.map((node) => ({ ...node })),
      edges: graph.edges.map((edge) => ({ ...edge })),
    };
  }

  function stepOnceLocal() {
    state.simFrame += 1;
  }

  function stepLocal(n) {
    state.simFrame += n;
  }

  function effectiveFrameIndex() {
    if (!state.cycleInfo) {
      return Math.min(state.simFrame, state.frames.length - 1);
    }
    if (state.simFrame < state.cycleInfo.warmupFrames) {
      return state.simFrame;
    }
    return state.cycleInfo.warmupFrames
      + (state.simFrame - state.cycleInfo.warmupFrames) % state.cycleInfo.period;
  }

  function getCurrentSnapshot() {
    if (!state.frames.length) {
      return { frame: 0, nodes: {}, edges: {} };
    }
    var idx = effectiveFrameIndex();
    return state.frames[idx] || state.frames[0];
  }

  function analyzeNodeLocal(nodeId) {
    if (!state.cycleInfo) return null;
    const ratio = state.cycleInfo.nodeRatios[nodeId] || null;
    const node = getNode(nodeId);
    return {
      id: nodeId,
      type: node.type,
      flowRatio: ratio || null,
      cycleInfo: state.cycleInfo,
    };
  }

  function analyzeEdgeLocal(edgeId) {
    if (!state.cycleInfo) return null;
    const ratio = state.cycleInfo.edgeRatios[edgeId] || null;
    const edge = getEdge(edgeId);
    return {
      id: edgeId,
      from: edge.from,
      to: edge.to,
      flowRatio: ratio || null,
      cycleInfo: state.cycleInfo,
    };
  }

  function renderSaveControls() {
    els.saveList.innerHTML = state.saves
      .map(
        (item) => `
          <option value="${item.id}" ${item.id === state.activeSaveId ? "selected" : ""}>
            ${escapeHtml(item.name)}
          </option>
        `
      )
      .join("");
  }

  function syncSelectedSaveName() {
    const record = state.saves.find((item) => item.id === state.activeSaveId) || null;
    if (record) {
      els.saveName.value = record.name;
    }
  }

  function saveCurrentGraph() {
    const name = (els.saveName.value || "").trim();
    if (!name) {
      state.message = "请输入存档名。";
      render();
      return;
    }

    const existing = state.saves.find((item) => item.name === name) || null;
    const id = existing?.id || `save-${Date.now()}-${Math.floor(Math.random() * 100000)}`;
    const record = {
      id,
      name,
      graph: cloneGraph(state.graph),
    };
    state.saves = [...state.saves.filter((item) => item.id !== id), record].sort((a, b) =>
      a.name.localeCompare(b.name)
    );
    state.activeSaveId = id;
    persistSaves();
    state.message = existing ? `已覆盖本地存档：${name}` : `图已保存到本地存档：${name}`;
    render();
  }

  function loadSelectedGraph() {
    const id = els.saveList.value;
    const record = state.saves.find((item) => item.id === id);
    if (!record) {
      state.message = "请选择要加载的图。";
      render();
      return;
    }

    stopRunning();
    state.graph = cloneGraph(record.graph);
    state.counters = buildCounters(state.graph);
    state.activeSaveId = record.id;
    state.selected = state.graph.nodes[0] ? { kind: "node", id: state.graph.nodes[0].id } : null;
    state.viewport = { ...DEFAULT_VIEWPORT };
    rebuildSimulator(`已加载图：${record.name}`);
    render();
  }

  function renameSelectedGraph() {
    const id = els.saveList.value;
    const record = state.saves.find((item) => item.id === id);
    const name = (els.saveName.value || "").trim();
    if (!record) {
      state.message = "请选择要重命名的图。";
      render();
      return;
    }
    if (!name) {
      state.message = "请输入新的图名称。";
      render();
      return;
    }

    record.name = name;
    state.saves.sort((a, b) => a.name.localeCompare(b.name));
    state.activeSaveId = record.id;
    persistSaves();
    state.message = `已重命名为：${name}`;
    render();
  }

  function deleteSelectedSave() {
    const id = els.saveList.value;
    const record = state.saves.find((item) => item.id === id);
    if (!record) {
      state.message = "请选择要删除的图。";
      render();
      return;
    }

    state.saves = state.saves.filter((item) => item.id !== id);
    state.activeSaveId = state.saves[0]?.id || null;
    persistSaves();
    state.message = `已删除图：${record.name}`;
    render();
  }

  function clearGraphToEndpoints() {
    stopRunning();
    state.graph = cloneGraph(BLANK_GRAPH);
    state.counters = buildCounters(state.graph);
    state.selected = { kind: "node", id: "source1" };
    state.viewport = { ...DEFAULT_VIEWPORT };
    state.draftEdgeFrom = null;
    rebuildSimulator("已清空为仅起点和终点。");
    render();
  }

  function exportGraphAsText() {
    if (state.graph.edges.length === 0) {
      state.message = "图中没有边，无法导出。";
      render();
      return;
    }
    var lines = state.graph.edges.map(function (e) { return e.from + " -> " + e.to; });
    var text = lines.join("\n");
    var blob = new Blob([text], { type: "text/plain" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = "graph.txt";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    state.message = "已导出 graph.txt（" + state.graph.edges.length + " 条边）。";
    render();
  }

  function importGraphFromFile(event) {
    var file = event.target.files[0];
    if (!file) {
      return;
    }
    var reader = new FileReader();
    reader.onload = function (e) {
      try {
        var text = e.target.result;
        var imported = parseGraphText(text);
        if (!imported) {
          return;
        }
        stopRunning();
        state.graph = imported;
        state.counters = buildCounters(state.graph);
        state.selected = state.graph.nodes[0] ? { kind: "node", id: state.graph.nodes[0].id } : null;
        state.viewport = { ...DEFAULT_VIEWPORT };
        state.draftEdgeFrom = null;
        rebuildSimulator("已从 graph.txt 导入图。");
        render();
      } catch (err) {
        state.message = "导入失败：" + err.message;
        render();
      }
    };
    reader.readAsText(file);
    event.target.value = "";
  }

  function parseGraphText(text) {
    var edges = [];
    var nodeIds = new Set();
    var lines = text.split(/\r?\n/);
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i].trim();
      if (!line || line.indexOf("->") === -1) {
        continue;
      }
      var parts = line.split("->");
      if (parts.length < 2) {
        continue;
      }
      var from = parts[0].trim();
      var to = parts[1].trim();
      if (!from || !to) {
        continue;
      }
      nodeIds.add(from);
      nodeIds.add(to);
      edges.push({ from: from, to: to });
    }

    if (edges.length === 0) {
      throw new Error("未找到有效的边定义。");
    }

    var inDeg = {};
    var outDeg = {};
    nodeIds.forEach(function (nid) { inDeg[nid] = 0; outDeg[nid] = 0; });
    edges.forEach(function (e) {
      outDeg[e.from] += 1;
      inDeg[e.to] += 1;
    });

    function inferNodeType(nid) {
      var inD = inDeg[nid] || 0;
      var outD = outDeg[nid] || 0;
      if (inD === 0) return "source";
      if (outD === 0) return "sink";
      if (inD <= outD) return "splitter";
      return "merger";
    }

    var depth = {};
    var cycleEdges = new Set();
    nodeIds.forEach(function (nid) { depth[nid] = undefined; });

    var adj = {};
    nodeIds.forEach(function (nid) { adj[nid] = []; });
    edges.forEach(function (e) { adj[e.from].push(e.to); });

    function dfs(nodeId, path) {
      path.add(nodeId);
      var tos = adj[nodeId] || [];
      for (var i = 0; i < tos.length; i++) {
        var to = tos[i];
        if (path.has(to)) {
          cycleEdges.add(nodeId + "->" + to);
          continue;
        }
        var nd = depth[nodeId] + 1;
        if (depth[to] === undefined || nd > depth[to]) {
          depth[to] = nd;
        }
        dfs(to, path);
      }
      path.delete(nodeId);
    }

    nodeIds.forEach(function (nid) {
      if (inDeg[nid] === 0) {
        depth[nid] = 0;
        dfs(nid, new Set());
      }
    });

    var changed = true;
    while (changed) {
      changed = false;
      edges.forEach(function (e) {
        if (depth[e.to] !== undefined && depth[e.from] === undefined) {
          depth[e.from] = Math.max(depth[e.to] - 1, 0);
          changed = true;
        }
      });
    }
    nodeIds.forEach(function (nid) {
      if (depth[nid] === undefined) depth[nid] = 0;
    });

    var layerNodes = {};
    var maxLayerSize = 0;
    nodeIds.forEach(function (nid) {
      var d = depth[nid];
      if (!layerNodes[d]) layerNodes[d] = [];
      layerNodes[d].push(nid);
      if (layerNodes[d].length > maxLayerSize) maxLayerSize = layerNodes[d].length;
    });

    var layerKeys = Object.keys(layerNodes).map(Number).sort(function (a, b) { return a - b; });
    var spacingX = GRAPH_BOUNDS.width / Math.max(layerKeys.length + 1, 2);
    var spacingY = GRAPH_BOUNDS.height / Math.max(maxLayerSize + 1, 2);

    var nodes = [];
    nodeIds.forEach(function (nid) {
      var d = depth[nid];
      var col = layerKeys.indexOf(d);
      var row = layerNodes[d].indexOf(nid);
      var x = GRAPH_BOUNDS.padding + spacingX * (col + 1);
      var y = GRAPH_BOUNDS.padding + spacingY * (row + 1);
      nodes.push({
        id: nid,
        type: inferNodeType(nid),
        x: Math.round(x),
        y: Math.round(y),
      });
    });

    forceLayout(nodes, edges);

    return {
      nodes: nodes,
      edges: edges.map(function (e, idx) {
        return { id: "edge" + (idx + 1), from: e.from, to: e.to };
      }),
    };
  }

  function forceLayout(nodes, edgeList, iterations) {
    var k1 = 10000;
    var k2 = 200;
    var damp = 0.88;
    var dt = 0.5;
    var idealLength = 400;
    var minDist = 5;
    var cap = 500;

    var nodeMap = {};
    for (var i = 0; i < nodes.length; i++) {
      nodeMap[nodes[i].id] = nodes[i];
    }

    var vel = {};
    for (var k = 0; k < nodes.length; k++) {
      vel[nodes[k].id] = { vx: 0, vy: 0 };
    }

    iterations = iterations || 500;
    for (var iter = 0; iter < iterations; iter++) {
      var accel = {};
      for (var k = 0; k < nodes.length; k++) {
        accel[nodes[k].id] = { ax: 0, ay: 0 };
      }

      for (var m = 0; m < edgeList.length; m++) {
        var edge = edgeList[m];
        var u = nodeMap[edge.from];
        var v = nodeMap[edge.to];
        if (!u || !v) continue;
        var dx = v.x - u.x;
        var dy = v.y - u.y;
        var dist = Math.sqrt(dx * dx + dy * dy) || 1e-6;
        var force = k2 * (dist - idealLength) / (dist * idealLength);
        accel[edge.from].ax += force * dx;
        accel[edge.from].ay += force * dy;
        accel[edge.to].ax -= force * dx;
        accel[edge.to].ay -= force * dy;
      }

      for (var a = 0; a < nodes.length; a++) {
        for (var b = a + 1; b < nodes.length; b++) {
          var na = nodes[a];
          var nb = nodes[b];
          var rdx = nb.x - na.x;
          var rdy = nb.y - na.y;
          var rdist = Math.max(Math.sqrt(rdx * rdx + rdy * rdy), minDist);
          var rforce = k1 / (rdist * rdist * rdist);
          accel[na.id].ax -= rforce * rdx;
          accel[na.id].ay -= rforce * rdy;
          accel[nb.id].ax += rforce * rdx;
          accel[nb.id].ay += rforce * rdy;
        }
      }

      for (var p = 0; p < nodes.length; p++) {
        var n = nodes[p];
        var a = accel[n.id];
        var am = Math.sqrt(a.ax * a.ax + a.ay * a.ay);
        if (am > cap) { a.ax *= cap / am; a.ay *= cap / am; }
        vel[n.id].vx = damp * (vel[n.id].vx + a.ax * dt);
        vel[n.id].vy = damp * (vel[n.id].vy + a.ay * dt);
        n.x = clamp(n.x + vel[n.id].vx * dt, SAFE_CLAMP.min, SAFE_CLAMP.max);
        n.y = clamp(n.y + vel[n.id].vy * dt, SAFE_CLAMP.min, SAFE_CLAMP.max);
      }
    }
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
    if (active instanceof HTMLElement) {
      active.blur();
    }
  }

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  main();
})();
