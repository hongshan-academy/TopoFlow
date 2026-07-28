function gcd(a, b) {
  let x = Math.abs(a);
  let y = Math.abs(b);
  while (y !== 0) {
    const t = x % y;
    x = y;
    y = t;
  }
  return x || 1;
}

function reduceFraction(numerator, denominator) {
  if (denominator === 0) {
    return { numerator: 0, denominator: 0, text: "0/0" };
  }
  const d = gcd(numerator, denominator);
  return {
    numerator: numerator / d,
    denominator: denominator / d,
    text: `${numerator / d}/${denominator / d}`,
  };
}

function getNodeShape(type) {
  switch (type) {
    case "source":
      return { maxIn: 0, maxOut: 1 };
    case "sink":
      return { maxIn: 1, maxOut: 0 };
    case "splitter":
      return { maxIn: 1, maxOut: 3 };
    case "merger":
      return { maxIn: 3, maxOut: 1 };
    default:
      throw new Error(`Unknown node type: ${type}`);
  }
}

function buildGraph(rawGraph) {
  const nodes = new Map();
  const edges = new Map();
  const nodeOrder = [];

  for (const rawNode of rawGraph.nodes || []) {
    if (!rawNode || !rawNode.id) {
      throw new Error("Node id is required.");
    }
    if (nodes.has(rawNode.id)) {
      throw new Error(`Duplicate node id: ${rawNode.id}`);
    }

    const shape = getNodeShape(rawNode.type);
    const node = {
      id: rawNode.id,
      type: rawNode.type,
      x: rawNode.x ?? 0,
      y: rawNode.y ?? 0,
      initialFull: rawNode.initialFull,
      inSlots: new Array(shape.maxIn).fill(null),
      outSlots: new Array(shape.maxOut).fill(null),
      inEdges: [],
      outEdges: [],
    };
    nodes.set(rawNode.id, node);
    nodeOrder.push(node.id);
  }

  for (const rawEdge of rawGraph.edges || []) {
    if (!rawEdge || !rawEdge.id) {
      throw new Error("Edge id is required.");
    }
    if (edges.has(rawEdge.id)) {
      throw new Error(`Duplicate edge id: ${rawEdge.id}`);
    }

    const fromNode = nodes.get(rawEdge.from);
    const toNode = nodes.get(rawEdge.to);
    if (!fromNode) {
      throw new Error(`Edge ${rawEdge.id} references missing source node: ${rawEdge.from}`);
    }
    if (!toNode) {
      throw new Error(`Edge ${rawEdge.id} references missing target node: ${rawEdge.to}`);
    }

    const fromSlot = resolveSlot(fromNode.outSlots, rawEdge.fromSlot, `edge ${rawEdge.id} source`);
    const toSlot = resolveSlot(toNode.inSlots, rawEdge.toSlot, `edge ${rawEdge.id} target`);

    const edge = {
      id: rawEdge.id,
      from: rawEdge.from,
      to: rawEdge.to,
      fromSlot,
      toSlot,
      initialFull: rawEdge.initialFull,
    };

    fromNode.outSlots[fromSlot] = edge.id;
    toNode.inSlots[toSlot] = edge.id;
    fromNode.outEdges.push(edge.id);
    toNode.inEdges.push(edge.id);
    edges.set(edge.id, edge);
  }

  return { nodes, edges, nodeOrder };
}

function resolveSlot(slotList, requestedSlot, label) {
  if (slotList.length === 0) {
    throw new Error(`${label} has no available slots.`);
  }

  if (requestedSlot != null) {
    if (!Number.isInteger(requestedSlot) || requestedSlot < 0 || requestedSlot >= slotList.length) {
      throw new Error(`${label} slot ${requestedSlot} is out of range.`);
    }
    if (slotList[requestedSlot] !== null) {
      throw new Error(`${label} slot ${requestedSlot} is already occupied.`);
    }
    return requestedSlot;
  }

  const emptyIndex = slotList.findIndex((slot) => slot === null);
  if (emptyIndex === -1) {
    throw new Error(`${label} has no empty slot.`);
  }
  return emptyIndex;
}

function computeCycleInfo(bits) {
  const seq = Array.isArray(bits) ? bits : [];
  for (let totalLength = seq.length; totalLength >= 2; totalLength -= 1) {
    const start = seq.length - totalLength;
    for (let periodLength = 1; periodLength * 2 <= totalLength; periodLength += 1) {
      if (totalLength % periodLength !== 0) {
        continue;
      }

      let matches = true;
      for (let i = periodLength; i < totalLength; i += 1) {
        if (seq[start + i] !== seq[start + (i % periodLength)]) {
          matches = false;
          break;
        }
      }

      if (matches) {
        const pattern = seq.slice(start, start + periodLength);
        const ones = pattern.reduce((sum, bit) => sum + bit, 0);
        let extraPrefixLength = 0;
        let probe = start - 1;
        while (probe >= 0) {
          const patternIndex = (probe - start + totalLength) % periodLength;
          if (seq[probe] !== pattern[patternIndex]) {
            break;
          }
          extraPrefixLength += 1;
          probe -= 1;
        }
        return {
          periodLength,
          repeatedLength: totalLength + extraPrefixLength,
          repeatCount: totalLength / periodLength,
          pattern,
          ones,
          extraPrefixLength,
        };
      }
    }
  }

  return null;
}

function computeSuffixCycleLength(bits) {
  const info = computeCycleInfo(bits);
  return info ? info.periodLength : 0;
}

function computeUpdateOrder(nodes, nodeOrder, edges) {
  const visited = new Set();
  const order = [];
  const queue = [];
  const pushNode = (nodeId) => {
    if (visited.has(nodeId)) {
      return;
    }
    visited.add(nodeId);
    queue.push(nodeId);
  };

  for (const nodeId of nodeOrder) {
    const node = nodes.get(nodeId);
    if (node.type === "source") {
      pushNode(nodeId);
    }
  }

  while (queue.length > 0) {
    const nodeId = queue.shift();
    order.push(nodeId);
    const node = nodes.get(nodeId);
    for (const edgeId of node.outEdges) {
      const edge = edges.get(edgeId);
      pushNode(edge.to);
    }
  }

  for (const nodeId of nodeOrder) {
    if (visited.has(nodeId)) {
      continue;
    }
    pushNode(nodeId);
    while (queue.length > 0) {
      const currentId = queue.shift();
      order.push(currentId);
      const node = nodes.get(currentId);
      for (const edgeId of node.outEdges) {
        const edge = edges.get(edgeId);
        pushNode(edge.to);
      }
    }
  }

  return order;
}

function computeDeliverableNodeIds(nodes, edges) {
  const deliverable = new Set();
  const queue = [];
  for (const node of nodes.values()) {
    if (node.type === "sink") {
      deliverable.add(node.id);
      queue.push(node.id);
    }
  }

  while (queue.length > 0) {
    const nodeId = queue.shift();
    const node = nodes.get(nodeId);
    for (const edgeId of node.inEdges) {
      const edge = edges.get(edgeId);
      if (!deliverable.has(edge.from)) {
        deliverable.add(edge.from);
        queue.push(edge.from);
      }
    }
  }
  return deliverable;
}

function pickRoundRobinSlot(slotCount, currentSlot, hasCandidate) {
  if (slotCount <= 0) {
    return -1;
  }

  for (let offset = 1; offset <= slotCount; offset += 1) {
    const slotIndex = (currentSlot + offset + slotCount) % slotCount;
    if (hasCandidate(slotIndex)) {
      return slotIndex;
    }
  }

  return -1;
}

class TopoFlowSimulator {
  constructor(rawGraph, options = {}) {
    const built = buildGraph(rawGraph);
    this.nodes = built.nodes;
    this.edges = built.edges;
    this.nodeOrder = built.nodeOrder;
    this.updateOrder = computeUpdateOrder(this.nodes, this.nodeOrder, this.edges);
    this.deliverableNodeIds = computeDeliverableNodeIds(this.nodes, this.edges);
    this.orderIndex = new Map(this.updateOrder.map((nodeId, index) => [nodeId, index]));
    this.historyLimit = options.historyLimit ?? 256;
    this.initialFull = Boolean(options.initialFull);
    this.frame = 0;
    this.edgeRuntime = new Map();
    this.runtime = new Map();
    this.reset();
  }

  reset() {
    this.frame = 0;
    this.runtime.clear();
    this.edgeRuntime.clear();

    for (const node of this.nodes.values()) {
      const nodeInitialFull =
        node.initialFull !== undefined ? node.initialFull : this.initialFull;
      const startsWithItem =
        nodeInitialFull && node.type !== "source" && node.type !== "sink";
      this.runtime.set(node.id, {
        hasItem: startsWithItem,
        rrInIndex: -1,
        rrOutIndex: -1,
        flowHistory: [],
        displayHistory: [],
        flowOnes: 0,
        flowTotal: 0,
      });
    }
    for (const edge of this.edges.values()) {
      const edgeInitialFull =
        edge.initialFull !== undefined ? edge.initialFull : this.initialFull;
      this.edgeRuntime.set(edge.id, {
        hasItem: edgeInitialFull,
        flowHistory: [],
        displayHistory: [],
        flowOnes: 0,
        flowTotal: 0,
      });
    }
  }

  setNodeItem(nodeId, hasItem) {
    const runtime = this.getRuntime(nodeId);
    const node = this.getNode(nodeId);
    if (node.type === "source" || node.type === "sink") {
      throw new Error(`Cannot set stored item on ${node.type} node ${nodeId}.`);
    }
    runtime.hasItem = Boolean(hasItem);
  }

  setEdgeItem(edgeId, hasItem) {
    const runtime = this.edgeRuntime.get(edgeId);
    if (!runtime) {
      throw new Error(`Unknown edge id: ${edgeId}`);
    }
    runtime.hasItem = Boolean(hasItem);
  }

  step(count = 1) {
    if (!Number.isInteger(count) || count < 1) {
      throw new Error("Step count must be a positive integer.");
    }
    for (let i = 0; i < count; i += 1) {
      this.stepOnce();
    }
    return this.getSnapshot();
  }

  stepOnce() {
    const nodeStart = new Map();
    const edgeStart = new Map();
    const nodeState = new Map();
    const edgeState = new Map();
    const nodeSent = new Map();
    const nodeReceived = new Map();
    const edgeFilled = new Set();
    const queue = [];
    const inQueue = new Set();

    for (const [nodeId, runtime] of this.runtime.entries()) {
      const snapshot = {
        hasItem: runtime.hasItem,
        rrInIndex: runtime.rrInIndex,
        rrOutIndex: runtime.rrOutIndex,
      };
      nodeStart.set(nodeId, snapshot);
      nodeState.set(nodeId, { ...snapshot });
      nodeSent.set(nodeId, false);
      nodeReceived.set(nodeId, false);
    }

    for (const [edgeId, runtime] of this.edgeRuntime.entries()) {
      const snapshot = { hasItem: runtime.hasItem };
      edgeStart.set(edgeId, snapshot);
      edgeState.set(edgeId, { ...snapshot });
    }

    for (const nodeId of [...this.updateOrder].reverse()) {
      this.enqueueNode(queue, inQueue, nodeId);
    }

    while (queue.length > 0) {
      const nodeId = queue.shift();
      inQueue.delete(nodeId);
      this.processNodeFrame(nodeId, {
        nodeStart,
        edgeStart,
        nodeState,
        edgeState,
        nodeSent,
        nodeReceived,
        edgeFilled,
        queue,
        inQueue,
      });
    }

    const occupiedEdges = [];
    for (const node of this.nodes.values()) {
      const runtime = this.runtime.get(node.id);
      const current = nodeState.get(node.id);
      if (node.type !== "source" && node.type !== "sink") {
        runtime.hasItem = current.hasItem;
      }
      runtime.rrInIndex = current.rrInIndex;
      runtime.rrOutIndex = current.rrOutIndex;

      const flowBit = nodeReceived.get(node.id) ? 1 : 0;
      runtime.flowHistory.push(flowBit);
      runtime.displayHistory.push(flowBit);
      if (runtime.displayHistory.length > this.historyLimit) {
        runtime.displayHistory.shift();
      }
      runtime.flowOnes += flowBit;
      runtime.flowTotal += 1;
    }

    for (const edge of this.edges.values()) {
      const runtime = this.edgeRuntime.get(edge.id);
      const current = edgeState.get(edge.id);
      runtime.hasItem = current.hasItem;
      const flowBit = edgeFilled.has(edge.id) ? 1 : 0;
      runtime.flowHistory.push(flowBit);
      runtime.displayHistory.push(flowBit);
      if (runtime.displayHistory.length > this.historyLimit) {
        runtime.displayHistory.shift();
      }
      runtime.flowOnes += flowBit;
      runtime.flowTotal += 1;
      if (current.hasItem) {
        occupiedEdges.push(edge.id);
      }
    }

    this.occupiedEdges = occupiedEdges;
    this.frame += 1;
    return this.getSnapshot();
  }

  processNodeFrame(nodeId, frame) {
    const node = this.getNode(nodeId);
    const start = frame.nodeStart.get(nodeId);
    const current = frame.nodeState.get(nodeId);
    const startedWithSupply = node.type === "source" ? true : start.hasItem;

    if (startedWithSupply && !frame.nodeSent.get(nodeId)) {
      const sendEdge = this.chooseOutgoingEdge(node, current, frame.edgeState);
      if (sendEdge) {
        frame.edgeState.get(sendEdge.id).hasItem = true;
        frame.edgeFilled.add(sendEdge.id);
        frame.nodeSent.set(nodeId, true);
        if (node.type !== "source") {
          current.hasItem = false;
        }
        if (node.type === "splitter") {
          current.rrOutIndex = sendEdge.fromSlot;
        }
      }
    }

    if (!frame.nodeReceived.get(nodeId) && this.nodeHasReceiveSpace(node, current)) {
      const incomingEdge = this.chooseIncomingEdge(node, current, frame.edgeStart, frame.edgeState);
      if (incomingEdge) {
        frame.edgeState.get(incomingEdge.id).hasItem = false;
        frame.nodeReceived.set(nodeId, true);
        if (node.type !== "sink") {
          current.hasItem = true;
        }
        if (node.type === "merger") {
          current.rrInIndex = incomingEdge.toSlot;
        }
        this.enqueueNode(frame.queue, frame.inQueue, incomingEdge.from);
      }
    }
  }

  chooseOutgoingEdge(node, nodeState, edgeState) {
    if (node.type === "sink" || node.outEdges.length === 0) {
      return null;
    }

    if (node.type === "splitter") {
      const slotIndex = pickRoundRobinSlot(node.outSlots.length, nodeState.rrOutIndex, (candidateSlot) => {
        const edgeId = node.outSlots[candidateSlot];
        const edge = edgeId ? this.edges.get(edgeId) : null;
        return Boolean(edge) && this.canEdgeDeliver(edge) && !edgeState.get(edgeId).hasItem;
      });
      return slotIndex === -1 ? null : this.edges.get(node.outSlots[slotIndex]);
    }

    const edgeId = node.outSlots[0];
    const edge = edgeId ? this.edges.get(edgeId) : null;
    if (!edge || !this.canEdgeDeliver(edge) || edgeState.get(edgeId).hasItem) {
      return null;
    }
    return edge;
  }

  canEdgeDeliver(edge) {
    return this.deliverableNodeIds.has(edge.to);
  }

  chooseIncomingEdge(node, nodeState, edgeStart, edgeState) {
    if (node.type === "source" || node.inEdges.length === 0) {
      return null;
    }

    const available = node.inEdges
      .map((edgeId) => this.edges.get(edgeId))
      .filter((edge) => edgeStart.get(edge.id).hasItem && edgeState.get(edge.id).hasItem);

    if (available.length === 0) {
      return null;
    }

    if (node.type === "merger") {
      const slotIndex = pickRoundRobinSlot(node.inSlots.length, nodeState.rrInIndex, (candidateSlot) =>
        available.some((edge) => edge.toSlot === candidateSlot)
      );
      return slotIndex === -1 ? null : available.find((edge) => edge.toSlot === slotIndex) || null;
    }

    return available[0];
  }

  nodeHasReceiveSpace(node, nodeState) {
    if (node.type === "source") {
      return false;
    }
    if (node.type === "sink") {
      return true;
    }
    if (node.outEdges.length === 0) {
      return false;
    }
    return !nodeState.hasItem;
  }

  enqueueNode(queue, inQueue, nodeId) {
    if (inQueue.has(nodeId)) {
      return;
    }
    inQueue.add(nodeId);
    queue.push(nodeId);
  }

  analyzeNode(nodeId) {
    const node = this.getNode(nodeId);
    const runtime = this.getRuntime(nodeId);
    const cycleInfo = computeCycleInfo(runtime.flowHistory);
    const flowRatio = cycleInfo
      ? reduceFraction(cycleInfo.ones, cycleInfo.periodLength)
      : reduceFraction(runtime.flowOnes, runtime.flowTotal);
    return {
      id: node.id,
      type: node.type,
      hasItem: runtime.hasItem,
      frame: this.frame,
      flowHistory: [...runtime.displayHistory],
      fullHistoryLength: runtime.flowHistory.length,
      suffixCycleLength: cycleInfo ? cycleInfo.periodLength : 0,
      cycleInfo,
      warmupFrames: runtime.flowHistory.length - (cycleInfo ? cycleInfo.repeatedLength : 0),
      flowRatio,
      flowOnes: runtime.flowOnes,
      flowTotal: runtime.flowTotal,
    };
  }

  analyzeEdge(edgeId) {
    const edge = this.edges.get(edgeId);
    if (!edge) {
      throw new Error(`Unknown edge id: ${edgeId}`);
    }
    const runtime = this.edgeRuntime.get(edgeId);
    const cycleInfo = computeCycleInfo(runtime.flowHistory);
    const flowRatio = cycleInfo
      ? reduceFraction(cycleInfo.ones, cycleInfo.periodLength)
      : reduceFraction(runtime.flowOnes, runtime.flowTotal);
    return {
      id: edge.id,
      from: edge.from,
      to: edge.to,
      hasItem: runtime.hasItem,
      frame: this.frame,
      flowHistory: [...runtime.displayHistory],
      fullHistoryLength: runtime.flowHistory.length,
      suffixCycleLength: cycleInfo ? cycleInfo.periodLength : 0,
      cycleInfo,
      warmupFrames: runtime.flowHistory.length - (cycleInfo ? cycleInfo.repeatedLength : 0),
      flowRatio,
      flowOnes: runtime.flowOnes,
      flowTotal: runtime.flowTotal,
    };
  }

  getSnapshot() {
    return {
      frame: this.frame,
      nodes: [...this.nodes.values()].map((node) => {
        const runtime = this.runtime.get(node.id);
        return {
          id: node.id,
          type: node.type,
          hasItem: runtime.hasItem,
          rrInIndex: runtime.rrInIndex,
          rrOutIndex: runtime.rrOutIndex,
          flowOnes: runtime.flowOnes,
          flowTotal: runtime.flowTotal,
        };
      }),
      edges: [...this.edges.values()].map((edge) => ({
        id: edge.id,
        from: edge.from,
        to: edge.to,
        hasItem: this.edgeRuntime.get(edge.id).hasItem,
        flowOnes: this.edgeRuntime.get(edge.id).flowOnes,
        flowTotal: this.edgeRuntime.get(edge.id).flowTotal,
      })),
      occupiedEdges: [...(this.occupiedEdges || [])],
    };
  }

  getNode(nodeId) {
    const node = this.nodes.get(nodeId);
    if (!node) {
      throw new Error(`Unknown node id: ${nodeId}`);
    }
    return node;
  }

  getRuntime(nodeId) {
    const runtime = this.runtime.get(nodeId);
    if (!runtime) {
      throw new Error(`Missing runtime for node id: ${nodeId}`);
    }
    return runtime;
  }
}

const simulatorApi = {
  TopoFlowSimulator,
  buildGraph,
  computeCycleInfo,
  computeSuffixCycleLength,
  reduceFraction,
};

if (typeof module !== "undefined") {
  module.exports = simulatorApi;
}

if (typeof window !== "undefined") {
  window.TopoFlowSimulatorApi = simulatorApi;
}
