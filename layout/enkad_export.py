"""Export enriched TopoFlow layouts to EnKA Beta v4, without native solver/MCP.

Catalog verified 2026-10-18. This exports geometry; simulation equivalence is
not implied. See enkad-control-port-notes.md for CU assumptions.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import re


FAMILY = re.compile(r'^(S[23]|C[23]|CU[1-4])_\d+(?:__\d+)?$')
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # xy outward
IDS = {'belt': {'S': 35, 'C': 34, 'CU': 37, 'bridge': 36, 'path': 33},
       'pipe': {'S': 39, 'C': 38, 'CU': 41, 'bridge': 40, 'path': 42}}


def integer(value, label, low=0, high=2**32-1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{label} must be an integer in {low}..{high}')
    return value


def varint(value):
    value &= (1 << 64) - 1
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def uint(number, value):
    return varint(number * 8) + varint(value) if value else b''


def nested(number, value):
    return varint(number * 8 + 2) + varint(len(value)) + value


def encode(layout):
    """Encode the supported native Factory subset (all integers validated upstream)."""
    result = uint(1, 4) + uint(2, layout['area_width']) + uint(3, layout['area_height'])
    for layer in layout['layers']:
        body = uint(1, layer['layer_tag'])
        for e in layer['entities']:
            payload = (uint(1, e['type']) + uint(2, e['entity_id']) + uint(3, e['x'])
                       + uint(4, e['y']) + uint(9, e['uid']))
            if e['type'] == 0:
                b = e['building']
                data = uint(1, b['rotation']) + uint(8, b.get('value', 0))
                for p in b.get('port_items', []):
                    data += nested(2, uint(1, p['port_index']) + uint(2, p['item_id']))
                payload += nested(5, data)
            else:
                data = b''.join(nested(1, uint(1, p['x']) + uint(2, p['y']) + uint(3, p['type']))
                                for p in e['points'])
                payload += nested(6 if e['type'] == 1 else 7, data)
            body += nested(2, payload)
        result += nested(4, body)
    return base64.b64encode(result).decode('ascii')


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def delta(a, b):
    return b[0] - a[0], b[1] - a[1]


def ordered_route(edges, source, target, cell):
    adjacency = defaultdict(list)
    seen = set()
    for pair in edges:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError('route edges must contain cell pairs')
        a, b = map(cell, pair)
        key = frozenset((a, b))
        if sum(abs(v) for v in delta(a, b)) != 1 or key in seen:
            raise ValueError('route has duplicate or non-unit grid edges')
        seen.add(key)
        adjacency[a].append(b)
        adjacency[b].append(a)
    if source == target or len(adjacency[source]) != 1 or len(adjacency[target]) != 1:
        raise ValueError('route endpoints do not match its facilities')
    if any(len(v) != (1 if p in (source, target) else 2) for p, v in adjacency.items()):
        raise ValueError('route branches or has extra endpoints')
    path = [source]
    visited = {source}
    while path[-1] != target:
        candidates = [p for p in adjacency[path[-1]] if p not in visited]
        if len(candidates) != 1:
            raise ValueError('route is cyclic or disconnected')
        path.append(candidates[0])
        visited.add(path[-1])
    if len(path) != len(seen) + 1:
        raise ValueError('route contains disconnected edges')
    return path


def export_layout(solution, *, item_id=None, control_items=None,
                  full_rate_per_minute=None, offset=(8, 8)):
    """Return blueprint, native JSON, endpoint manifest and validation warnings.

control_items maps original TopoFlow node IDs to item IDs; item_id is a fallback
for single-item networks. Pipe CU exports require an explicit rate baseline.
"""
    if not isinstance(solution, dict):
        raise ValueError('solution must be an object')
    solution = solution.get('solution', solution)
    if not isinstance(solution, dict):
        raise ValueError('solution must be an object')
    rows = integer(solution['n'], 'n', 1, 300)
    cols = integer(solution['m'], 'm', 1, 300)
    meta = solution['topoflow']
    if not isinstance(meta, dict):
        raise ValueError('topoflow metadata must be an object')
    for key in ('machs', 'belts'):
        if not isinstance(solution.get(key), list) or any(not isinstance(v, dict) for v in solution[key]):
            raise ValueError(f'{key} must be a list of objects')
    external = meta.get('external_connections', [])
    if not isinstance(external, list) or any(not isinstance(v, dict) for v in external):
        raise ValueError('external_connections must be a list of objects')
    transport = meta['transport']
    if transport not in IDS:
        raise ValueError('transport must be belt or pipe')
    kind = 1 if transport == 'belt' else 2
    catalog = IDS[transport]
    if not isinstance(offset, (list, tuple)) or len(offset) != 2:
        raise ValueError('offset must contain x and y')
    ox, oy = (integer(v, 'offset', 6, 300) for v in offset)

    def cell(raw):
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError('cell must be [row, column]')
        row = integer(raw[0], 'row', 0, rows-1)
        col = integer(raw[1], 'column', 0, cols-1)
        return col + ox, row + oy

    machines, occupied, entities, routes, endpoints = {}, {}, [], [], []
    assignments_used = set()
    has_cu = False
    control_items = {} if control_items is None else control_items
    if not isinstance(control_items, dict):
        raise ValueError('control_items must map CU node IDs to item IDs')
    cu_nodes = set()
    for machine in solution['machs']:
        if machine.get('size', [1, 1]) != [1, 1] or machine.get('type', 'logi') != 'logi':
            raise ValueError('only 1x1 logistics facilities are supported')
        mid = integer(machine['id'], 'facility id')
        node = machine['topoflow_node_id']
        match = FAMILY.fullmatch(node)
        if not match:
            raise ValueError(f'unsupported facility {node}')
        family = match[1]
        if machine.get('topoflow_node_type', family) != family:
            raise ValueError(f'{node}: declared family mismatch')
        rot = integer(machine['rot'], f'{node}.rot', 0, 3)
        pos = cell(machine['pos'])
        if mid in machines or pos in occupied or any(m['node'] == node for m in machines.values()):
            raise ValueError('duplicate facility ID, node or occupied cell')
        group = 'CU' if family.startswith('CU') else family[0]
        inputs = {(d+rot) % 4 for d in ([0, 1, 3] if group == 'C' else [2])}
        outputs = {(d+rot) % 4 for d in ([0, 1, 3] if group == 'S' else [2] if group == 'C' else [0])}
        building = {'rotation': ((0 if group == 'C' else 180) - 90*rot) % 360}
        if group == 'CU':
            has_cu = True
            cu_nodes.add(node)
            gear = int(family[2])
            if machine.get('control_gear', gear) != gear:
                raise ValueError(f'{node}: control_gear mismatch')
            selected = control_items.get(node, item_id)
            integer(selected, f'{node} item_id (required)', 1)
            baseline = full_rate_per_minute
            if baseline is None:
                if transport == 'pipe':
                    raise ValueError('pipe CU requires explicit full_rate_per_minute')
                baseline = 30
            try:
                rate = Fraction(str(baseline)) * gear / 5
            except (ValueError, ZeroDivisionError):
                raise ValueError('full_rate_per_minute must be a positive number') from None
            maximum = 30 if kind == 1 else 120
            if not 0 < rate <= maximum or (rate * 2).denominator != 1:
                raise ValueError(f'{node}: rate must be a positive half-unit <= {maximum}/min')
            building.update(value=0 if rate == maximum else int(rate*2),
                            port_items=[{'port_index': 0, 'item_id': selected}])
        uid = len(entities) + 1
        machines[mid] = {'node': node, 'pos': pos, 'inputs': inputs, 'outputs': outputs, 'uid': uid}
        occupied[pos] = uid
        entities.append({'type': 0, 'entity_id': catalog[group], 'uid': uid,
                         'x': pos[0], 'y': pos[1], 'building': building})
    if not machines:
        raise ValueError('no facilities to export')
    if set(control_items) - cu_nodes:
        raise ValueError('control_items references unknown CU nodes')

    def port(mid, assignment, output):
        if not isinstance(assignment, dict):
            raise ValueError('port assignment must be an object')
        m = machines[mid]
        d = integer(assignment['dir'], 'port.dir', 0, 3)
        if (cell(assignment['cell']) != m['pos'] or
                assignment['kind'] != (-kind if output else kind) or
                d not in m['outputs' if output else 'inputs']):
            raise ValueError(f'{m["node"]}: port geometry mismatch')
        key = mid, output, d
        if key in assignments_used:
            raise ValueError(f'{m["node"]}: physical port reused')
        assignments_used.add(key)
        return m['pos'], DIRECTIONS[d]

    for index, belt in enumerate(solution['belts']):
        if belt['type'] != kind:
            raise ValueError('mixed transport kinds are not supported')
        source, out = port(belt['frm'], belt['port_assignment']['frm'], True)
        target, inward = port(belt['to'], belt['port_assignment']['to'], False)
        path = ordered_route(belt['edges'], source, target, cell)
        if path[1] != add(source, out) or path[-2] != add(target, inward):
            raise ValueError('route does not enter/leave the assigned ports')
        routes.append((belt.get('id', f'edge-{index}'), path))

    for ext in meta.get('external_connections', []):
        if ext['role'] not in ('input', 'output'):
            raise ValueError('external role must be input or output')
        output = ext['role'] == 'output'
        pos, outward = port(ext['facility_id'], ext['port_assignment'], output)
        anchor = ext.get('anchor')
        path = [pos]
        if anchor is not None:
            if not isinstance(anchor, dict):
                raise ValueError('anchor must be an object')
            side = anchor['side']
            if side not in ('top', 'right', 'bottom', 'left'):
                raise ValueError('invalid anchor side')
            off = integer(anchor['offset'], 'anchor.offset', 0,
                          (rows if side in ('left', 'right') else cols)-1)
            end = {'left': (ox-1, oy+off), 'right': (ox+cols, oy+off),
                   'top': (ox+off, oy-1), 'bottom': (ox+off, oy+rows)}[side]
            while path[-1] != end and len(path) <= rows+cols+1:
                path.append(add(path[-1], outward))
            if path[-1] != end:
                raise ValueError('external anchor is not aligned with its assigned port')
        else:
            # Leave a free attachment cell directly outside an unanchored port.
            path.append(add(pos, outward))
        endpoint = path[-1]
        if not output:
            path.reverse()
        routes.append((ext['id'], path))
        endpoints.append({'id': ext['id'], 'role': ext['role'], 'node': machines[ext['facility_id']]['node'],
                          'attachment_cell': list(endpoint), 'direction': list(outward)})

    # Detect overlap and construct explicit bridges; endpoint cells are not belts.
    owners = defaultdict(list)
    edge_use = set()
    attachments = {tuple(e['attachment_cell']) for e in endpoints}
    if len(attachments) != len(endpoints) or attachments & set(occupied):
        raise ValueError('external attachment cells overlap each other or facilities')
    for rid, path in routes:
        for a, b in zip(path, path[1:]):
            key = frozenset((a, b))
            if key in edge_use:
                raise ValueError('routes share a grid edge')
            edge_use.add(key)
        for i in range(1, len(path)-1):
            p = path[i]
            if p in occupied or p in attachments:
                raise ValueError('route traverses a facility or external attachment cell')
            owners[p].append((rid, delta(path[i-1], p), delta(p, path[i+1])))
    bridges = {}
    for p, hits in sorted(owners.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if len(hits) == 1:
            continue
        if (len(hits) != 2 or any(a != b for _, a, b in hits) or
                (hits[0][1][0] == 0) == (hits[1][1][0] == 0)):
            raise ValueError('crossing must contain exactly two straight perpendicular routes')
        uid = len(entities)+1
        bridges[p] = uid
        occupied[p] = uid
        entities.append({'type': 0, 'entity_id': catalog['bridge'], 'uid': uid,
                         'x': p[0], 'y': p[1], 'building': {'rotation': 0}})
    route_manifest = []
    for rid, path in routes:
        cuts = [0] + [i for i in range(1, len(path)-1) if path[i] in bridges] + [len(path)-1]
        segments, direct = [], []
        for a, b in zip(cuts, cuts[1:]):
            if b == a+1:
                direct.append([list(path[a]), list(path[b])])
                continue
            cells = path[a+1:b]
            points = [path[a], cells[0]]
            for i in range(1, len(cells)-1):
                if delta(cells[i-1], cells[i]) != delta(cells[i], cells[i+1]):
                    points.append(cells[i])
            if points[-1] != cells[-1]:
                points.append(cells[-1])
            points.append(path[b])
            x, y = cells[0]
            uid = len(entities)+1
            entities.append({'type': kind, 'entity_id': catalog['path'], 'uid': uid, 'x': x, 'y': y,
                             'points': [{'x': px-x, 'y': py-y,
                                         'type': 1 if i == 0 else 2 if i == len(points)-1 else 0}
                                        for i, (px, py) in enumerate(points)]})
            segments.append(uid)
        route_manifest.append({'id': rid, 'path': [list(p) for p in path],
                               'segments': segments, 'direct_connections': direct})
    layout = {'version': 4, 'area_width': max(50, ox+cols), 'area_height': max(50, oy+rows),
              'layers': [{'layer_tag': 3 if kind == 1 else 5, 'entities': entities}]}
    warnings = ['已验证导出几何；尚未在 EnKA 中验证导入和生产仿真。',
                '物品 ID 需来自当前 EnKA 目录；本接口不验证物品名称、类型或版本。']
    if has_cu:
        warnings.append('CU 仅按平均流量映射，未验证 EnKA 与五 tick 接收窗口等价；'
                        '物品带缺省基准为 30/m，counter=0。')
    return {'blueprint': encode(layout), 'layout': layout,
            'manifest': {'offset': [ox, oy], 'transport': transport,
                         'facilities': {m['node']: m['uid'] for m in machines.values()},
                         'endpoints': endpoints, 'routes': route_manifest, 'bridges': len(bridges)},
            'warnings': warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('solution', type=Path)
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--item-id', type=int)
    parser.add_argument('--full-rate-per-minute')
    args = parser.parse_args()
    try:
        result = export_layout(json.loads(args.solution.read_text(encoding='utf-8')),
                               item_id=args.item_id, full_rate_per_minute=args.full_rate_per_minute)
        args.output.write_text(result['blueprint']+'\n', encoding='utf-8')
        args.output.with_suffix('.manifest.json').write_text(
            json.dumps({k: v for k, v in result.items() if k != 'blueprint'}, ensure_ascii=False, indent=2),
            encoding='utf-8')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f'Export failed: {exc}\n')


if __name__ == '__main__':
    main()
