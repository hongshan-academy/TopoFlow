"""Merger builders: chained merging.

Chains a set of leaf outputs into a merge chain, placing small shares
upstream. The closer a merger is to the target, the larger the total share it
carries (downstream larger, upstream smaller). Each merger has in-degree
``previous link + at most two new leaves`` (<= 3), so when the limit-module's
S1 connects to the topmost merger, all its other input edges carry smaller
shares.
"""

from __future__ import annotations


def merge_to(nodes: list[dict], edges: list[dict], emit, outputs: list[tuple[str, int]],
             target: str) -> None:
    """Merge a set of leaf outputs into a single stream ending at ``target``.

    ``emit(ntype)`` creates a node and returns its id (appends to ``nodes``).
    ``outputs`` is ``[(upstream splitter id, port share w)]`` sorted ascending
    by share. ``target`` is the merge target node id.
    """
    cur = sorted(outputs, key=lambda x: x[1])  # small shares first (upstream)
    if not cur:
        return
    src0, w0 = cur[0]
    idx = 1
    if len(cur) >= 2:
        # Chain head: merge the two most upstream shares (the two same-target
        # edges that are "small enough").
        cnode = emit("C")
        edges.append({"from": src0, "to": cnode, "w": w0})
        edges.append({"from": cur[1][0], "to": cnode, "w": cur[1][1]})
        prev = (cnode, w0 + cur[1][1])
        idx = 2
    else:
        prev = (src0, w0)
    # Each following merger = previous link + at most two new leaves.
    while idx < len(cur):
        take = min(2, len(cur) - idx)
        cnode = emit("C")
        wsum = prev[1]
        edges.append({"from": prev[0], "to": cnode, "w": wsum})
        for k in range(take):
            src, w = cur[idx + k]
            edges.append({"from": src, "to": cnode, "w": w})
            wsum += w
        prev = (cnode, wsum)
        idx += take
    edges.append({"from": prev[0], "to": target, "w": prev[1]})
