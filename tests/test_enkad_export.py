"""Tests use handcrafted layouts, independent of Z3 or EnKA runtime."""
import base64
import copy
import importlib.util
import struct
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from layout.enkad_export import export_layout
from layout import adapter


def fields(data):
    """Independent small protobuf wire reader (not the encoder under test)."""
    result = {}
    offset = 0

    def read_varint():
        nonlocal offset
        value, shift = 0, 0
        while True:
            b = data[offset]
            offset += 1
            value |= (b & 127) << shift
            if not b & 128:
                return value
            shift += 7
            assert shift < 70

    while offset < len(data):
        tag = read_varint()
        if tag % 8 == 0:
            value = read_varint()
        else:
            assert tag % 8 == 2
            length = read_varint()
            value = data[offset:offset+length]
            assert len(value) == length
            offset += length
        result.setdefault(tag >> 3, []).append(value)
    return result


def machine(mid, family, pos, rot):
    return {'id': mid, 'topoflow_node_id': f'{family}_{mid}',
            'topoflow_node_type': family, 'pos': pos, 'rot': rot}


def assignment(m, direction, output=False, kind=1):
    return {'cell': m['pos'], 'dir': direction, 'kind': -kind if output else kind}


def single(gear=2, rot=0, transport='belt', prefix=None):
    kind = 1 if transport == 'belt' else 2
    if prefix is None:
        prefix = 'PCU' if transport == 'pipe' else 'CU'
    m = machine(0, f'{prefix}{gear}', [1, 1], rot)
    sides = ['bottom', 'right', 'top', 'left']
    return {'n': 3, 'm': 3, 'machs': [m], 'belts': [], 'topoflow': {
        'transport': transport, 'external_connections': [
            {'id': 'in', 'role': 'input', 'facility_id': 0,
             'anchor': {'side': sides[(2+rot) % 4], 'offset': 1},
             'port_assignment': assignment(m, (2+rot) % 4, kind=kind)},
            {'id': 'out', 'role': 'output', 'facility_id': 0,
             'anchor': {'side': sides[rot], 'offset': 1},
             'port_assignment': assignment(m, rot, True, kind)}]}}


@pytest.mark.parametrize('gear', [1, 2, 3, 4])
@pytest.mark.parametrize('rot', range(4))
def test_cu_native_fields_and_rotations(gear, rot):
    result = export_layout(single(gear, rot), item_id=123)
    raw = base64.b64decode(result['blueprint'], validate=True)
    factory = fields(raw)
    assert factory[1] == [4]
    layer = fields(factory[4][0])
    assert layer[1] == [3]
    entity = fields(layer[2][0])
    assert entity[2] == [37]
    body = fields(entity[5][0])
    assert body.get(1, [0]) == [(180-90*rot) % 360]
    assert body[8] == [12*gear]
    port = fields(body[2][0])
    assert port.get(1, [0]) == [0]
    assert port[2] == [123]
    assert len(layer[2]) == 3  # CU and two one-cell belts
    # Decode negative offsets independently to confirm signed int32 representation.
    coordinates = []
    for entity_data in layer[2][1:]:
        for p in fields(fields(entity_data)[6][0])[1]:
            for number in (1, 2):
                value = fields(p).get(number, [0])[0]
                coordinates.append(struct.unpack('<i', struct.pack('<I', value & 0xffffffff))[0])
    assert -1 in coordinates


def test_pipe_control_requires_explicit_baseline_and_uses_pipe_layer():
    with pytest.raises(ValueError, match='explicit'):
        export_layout(single(transport='pipe'), item_id=123)
    result = export_layout(single(transport='pipe'), item_id=123,
                           full_rate_per_minute=120)
    layer = result['layout']['layers'][0]
    assert layer['layer_tag'] == 5
    assert layer['entities'][0]['entity_id'] == 41
    # PCU2 = 120 * 2 / 20 = 12/min -> value 24 (not 96).
    assert layer['entities'][0]['building']['value'] == 24
    assert all(e['entity_id'] == 42 for e in layer['entities'][1:])


def test_icu_uses_item_layer_and_rate():
    result = export_layout(single(gear=2, prefix='ICU'), item_id=123)
    layer = result['layout']['layers'][0]
    assert layer['layer_tag'] == 3
    assert layer['entities'][0]['entity_id'] == 37
    assert layer['entities'][0]['building']['value'] == 24


def test_pcu_nine_rate():
    result = export_layout(single(gear=9, transport='pipe', prefix='PCU'),
                           item_id=123, full_rate_per_minute=120)
    entity = result['layout']['layers'][0]['entities'][0]
    # PCU9 = 120 * 9 / 20 = 54/min -> value 108.
    assert entity['building']['value'] == 108


@pytest.mark.parametrize('prefix,transport,match', [
    ('PCU', 'belt', 'pipe control unit requires pipe transport'),
    ('ICU', 'pipe', 'item control unit requires belt transport'),
    ('CU', 'pipe', 'item control unit requires belt transport'),
])
def test_control_unit_transport_mismatch(prefix, transport, match):
    with pytest.raises(ValueError, match=match):
        export_layout(single(gear=2, transport=transport, prefix=prefix), item_id=123,
                      full_rate_per_minute=120)


def crossing():
    a, b, c, d = [machine(i, 'CU2', pos, rot) for i, (pos, rot) in enumerate(
        [([2, 0], 1), ([2, 4], 1), ([0, 2], 0), ([4, 2], 0)])]
    def route(src, dst, cells, direction):
        # Reverse edge order and edge orientation to exercise actual reconstruction.
        return {'type': 1, 'frm': src['id'], 'to': dst['id'],
                'edges': [[y, x] for x, y in zip(cells, cells[1:])][::-1],
                'port_assignment': {'frm': assignment(src, direction, True),
                                    'to': assignment(dst, (direction+2) % 4)}}
    return {'n': 5, 'm': 5, 'machs': [a, b, c, d], 'topoflow': {'transport': 'belt'},
            'belts': [route(a, b, [[2, x] for x in range(5)], 1),
                      route(c, d, [[x, 2] for x in range(5)], 0)]}


def test_crossing_splits_paths_and_preserves_cell_spacing():
    result = export_layout(crossing(), item_id=123)
    entities = result['layout']['layers'][0]['entities']
    assert len(entities) == 9  # four CU, one bridge, four belts
    assert entities[4]['entity_id'] == 36
    assert (entities[4]['x'], entities[4]['y']) == (10, 10)
    assert result['manifest']['routes'][0]['path'] == [[x, 10] for x in range(8, 13)]
    assert result['manifest']['routes'][1]['path'] == [[10, y] for y in range(8, 13)]
    for e in entities[5:]:
        assert len(e['points']) == 3
        points = [(p['x']+e['x'], p['y']+e['y']) for p in e['points']]
        assert (10, 10) in (points[0], points[-1])
        assert points[1] != (10, 10)


@pytest.mark.parametrize('family,expected', [('S2', 35), ('S3', 35), ('C2', 34), ('C3', 34)])
@pytest.mark.parametrize('rot', range(4))
def test_router_rotations(family, expected, rot):
    m = machine(0, family, [1, 1], rot)
    s = {'n': 3, 'm': 3, 'machs': [m], 'belts': [], 'topoflow': {'transport': 'belt'}}
    entity = export_layout(s)['layout']['layers'][0]['entities'][0]
    assert entity['entity_id'] == expected
    assert entity['building']['rotation'] == ((180 if family[0] == 'S' else 0)-rot*90) % 360


def test_adjacent_facilities_use_direct_connection():
    s = crossing()
    s['machs'] = s['machs'][:2]
    s['machs'][1]['pos'] = [2, 1]
    s['belts'] = s['belts'][:1]
    s['belts'][0]['edges'] = [[[2, 0], [2, 1]]]
    s['belts'][0]['port_assignment']['to']['cell'] = [2, 1]
    result = export_layout(s, item_id=123)
    assert len(result['layout']['layers'][0]['entities']) == 2
    assert result['manifest']['routes'][0]['segments'] == []
    assert len(result['manifest']['routes'][0]['direct_connections']) == 1


def parallel_router_layout(transport='belt', distance=1):
    """Two routed branches leave the facing S/C ports unused."""
    kind = 1 if transport == 'belt' else 2
    column = 2 + distance
    s = machine(0, 'S2', [1, 2], 0)
    c = machine(1, 'C2', [1, column], 0)
    paths = [
        ([s['pos'], *[[2, x] for x in range(2, column+1)], c['pos']], 0, 0),
        ([s['pos'], [1, 1], [2, 1], *[[3, x] for x in range(1, column+2)],
          [2, column+1], [1, column+1], c['pos']], 3, 1),
    ]
    belts = []
    for cells, out_dir, in_dir in paths:
        frm, to = assignment(s, out_dir, True, kind), assignment(c, in_dir, kind=kind)
        frm['index'] = [0, 1, 3].index(out_dir)
        to['index'] = [0, 1, 3].index(in_dir)
        belts.append({'type': kind, 'frm': 0, 'to': 1,
                      'edges': [[a, b] for a, b in zip(cells, cells[1:])],
                      'port_assignment': {'frm': frm, 'to': to}})
    graph = {'nodes': ['In', s['topoflow_node_id'], c['topoflow_node_id'], 'Out'],
             'edges': [['In', s['topoflow_node_id']],
                       [s['topoflow_node_id'], c['topoflow_node_id']],
                       [s['topoflow_node_id'], c['topoflow_node_id']],
                       [c['topoflow_node_id'], 'Out']]}
    adapted = adapter.adapt_topoflow(
        graph, rows=5, columns=column+3, transport=transport,
        terminal_anchors={'In': {'side': 'top', 'offset': 2},
                          'Out': {'side': 'top', 'offset': column}})
    return adapter.enrich_solution(adapted, {'n': 5, 'm': column+3,
                                             'machs': [s, c], 'belts': belts})


@pytest.mark.parametrize('transport', ['belt', 'pipe'])
def test_unused_facing_router_ports_reject_extra_branch(transport):
    with pytest.raises(ValueError, match='unintended direct connection: S2_0 -> C2_1'):
        export_layout(parallel_router_layout(transport))


@pytest.mark.parametrize('transport', ['belt', 'pipe'])
def test_separated_unused_ports_preserve_two_branches(transport):
    result = export_layout(parallel_router_layout(transport, distance=2))
    assert len(result['manifest']['routes']) == 4  # two branches plus In/Out
    assert all(not r['direct_connections'] for r in result['manifest']['routes'])


@pytest.mark.parametrize('transport', ['belt', 'pipe'])
def test_declared_router_direct_branch_is_allowed(transport):
    s = parallel_router_layout(transport)
    source, target = s['machs']
    kind = 1 if transport == 'belt' else 2
    s['belts'].append({'type': kind, 'frm': 0, 'to': 1,
                       'edges': [[source['pos'], target['pos']]],
                       'port_assignment': {'frm': assignment(source, 1, True, kind),
                                           'to': assignment(target, 3, kind=kind)}})
    for m, family in zip(s['machs'], ['S3', 'C3']):
        m['topoflow_node_type'] = family
    result = export_layout(s)
    assert len(result['manifest']['routes'][2]['direct_connections']) == 1


@pytest.mark.parametrize('rot', range(4))
def test_adjacent_incompatible_ports_do_not_count_as_connections(rot):
    positions = [([1, 1], [2, 1]), ([1, 1], [1, 2]),
                 ([2, 1], [1, 1]), ([1, 2], [1, 1])]
    a, b = positions[rot]
    s = {'n': 4, 'm': 4, 'machs': [machine(0, 'CU2', a, rot),
                                   machine(1, 'CU2', b, (rot+2) % 4)],
         'belts': [], 'topoflow': {'transport': 'belt'}}
    assert len(export_layout(s, item_id=123)['layout']['layers'][0]['entities']) == 2


@pytest.mark.parametrize('names', [('S3_0', 'C3_1'), ('C3_0', 'S3_1')])
def test_export_uses_adapter_inferred_types_and_retains_original_ids(names):
    s = parallel_router_layout(distance=2)
    old = [m['topoflow_node_id'] for m in s['machs']]
    renamed = dict(zip(old, names))
    graph = s['topoflow']['graph']
    graph['nodes'] = [renamed.get(n, n) for n in graph['nodes']]
    graph['edges'] = [[renamed.get(n, n) for n in edge] for edge in graph['edges']]
    graph['inferNodeTypesFromDegrees'] = True
    adapted = adapter.adapt_topoflow(graph, rows=s['n'], columns=s['m'])
    s = adapter.enrich_solution(adapted, s)
    result = export_layout(s)
    entities = result['layout']['layers'][0]['entities']
    assert [e['entity_id'] for e in entities[:2]] == [35, 34]
    assert [e['building']['rotation'] for e in entities[:2]] == [180, 0]
    assert result['manifest']['facilities'] == {names[0]: 1, names[1]: 2}


@pytest.mark.parametrize('family', ['CU1', 'ICU2', 'PCU2', 'S2', 'C2'])
def test_cu_identity_cannot_be_repaired_to_another_family(family):
    s = single()
    s['machs'][0]['topoflow_node_type'] = family
    with pytest.raises(ValueError, match='declared family mismatch'):
        export_layout(s, item_id=123)


def test_router_cannot_be_repaired_to_cu():
    s = single()
    s['machs'][0]['topoflow_node_id'] = 'S2_0'
    with pytest.raises(ValueError, match='declared family mismatch'):
        export_layout(s, item_id=123)


@pytest.mark.parametrize('family', ['S4', None, 3])
def test_invalid_resolved_family_is_rejected(family):
    s = single()
    s['machs'][0]['topoflow_node_type'] = family
    with pytest.raises(ValueError, match='unsupported facility family'):
        export_layout(s, item_id=123)


def test_adjacent_bridges_remain_adjacent_without_spacer_belts():
    s = crossing()
    s['m'] = 6
    s['machs'][1]['pos'] = [2, 5]
    s['belts'][0]['edges'] = [[[2, x], [2, x+1]] for x in range(5)]
    s['belts'][0]['port_assignment']['to']['cell'] = [2, 5]
    a, b = machine(4, 'CU2', [0, 3], 0), machine(5, 'CU2', [4, 3], 0)
    s['machs'].extend([a, b])
    s['belts'].append({'type': 1, 'frm': 4, 'to': 5,
                       'edges': [[[y, 3], [y+1, 3]] for y in range(4)],
                       'port_assignment': {'frm': assignment(a, 0, True), 'to': assignment(b, 2)}})
    result = export_layout(s, item_id=123)
    assert result['manifest']['bridges'] == 2
    route = result['manifest']['routes'][0]
    assert route['direct_connections'] == [[[10, 10], [11, 10]]]
    assert len(route['segments']) == 2
    assert route['path'] == [[x, 10] for x in range(8, 14)]


@pytest.mark.skipif(shutil.which('node') is None, reason='requires Node for JavaScript syntax check')
def test_export_ui_javascript_syntax():
    static = Path(__file__).resolve().parents[1] / 'static'
    sources = [(static / name).read_text(encoding='utf-8') for name in ('app.js', 'enkad-export.js')]
    html = (static / 'layout-editor.html').read_text(encoding='utf-8')
    sources.extend(re.findall(r'<script\b[^>]*>(.*?)</script>', html, flags=re.S))
    for source in sources:
        result = subprocess.run([shutil.which('node'), '--check'], input=source,
                                text=True, encoding='utf-8', capture_output=True)
        assert result.returncode == 0, result.stderr


def test_invalid_settings_rejected():
    with pytest.raises(ValueError, match='item_id'):
        export_layout(single())
    with pytest.raises(ValueError, match='unknown control-unit'):
        export_layout(single(), item_id=123, control_items={'CU1_99': 123})
    for value in [0, -30, 'nan', 31, 1000]:
        with pytest.raises(ValueError):
            export_layout(single(), item_id=123, full_rate_per_minute=value)
    s = single()
    s['topoflow']['external_connections'][0]['anchor']['offset'] = 2
    with pytest.raises(ValueError, match='aligned'):
        export_layout(s, item_id=123)


def test_invalid_routes_rejected():
    s = crossing()
    s['belts'][0]['edges'].append(copy.deepcopy(s['belts'][0]['edges'][0]))
    with pytest.raises(ValueError, match='duplicate'):
        export_layout(s, item_id=123)
    s = crossing()
    s['machs'][0]['rot'] = 0
    with pytest.raises(ValueError, match='port geometry'):
        export_layout(s, item_id=123)


@pytest.mark.skipif(any(importlib.util.find_spec(m) is None for m in
                        ('fastapi', 'httpx', 'topoflow_native')),
                    reason='requires FastAPI/httpx and native extension to import server')
def test_http_export_endpoint():
    from fastapi.testclient import TestClient
    from server import app
    with TestClient(app) as client:
        ok = client.post('/api/export/enkad', json={'solution': single(), 'item_id': 123})
        assert ok.status_code == 200
        assert base64.b64decode(ok.json()['blueprint'], validate=True)
        invalid = client.post('/api/export/enkad', json={'solution': single()})
        assert invalid.status_code == 400
