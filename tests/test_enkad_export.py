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


def single(gear=2, rot=0, transport='belt'):
    kind = 1 if transport == 'belt' else 2
    m = machine(0, f'CU{gear}', [1, 1], rot)
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


def test_pipe_cu_requires_explicit_baseline_and_uses_pipe_layer():
    with pytest.raises(ValueError, match='explicit'):
        export_layout(single(transport='pipe'), item_id=123)
    result = export_layout(single(transport='pipe'), item_id=123, full_rate_per_minute=120)
    layer = result['layout']['layers'][0]
    assert layer['layer_tag'] == 5
    assert layer['entities'][0]['entity_id'] == 41
    assert layer['entities'][0]['building']['value'] == 96
    assert all(e['entity_id'] == 42 for e in layer['entities'][1:])


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
    with pytest.raises(ValueError, match='unknown CU'):
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
