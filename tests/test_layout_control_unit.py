"""CU geometry and metadata checks; native integration runs when available."""
import importlib.util

import pytest

from layout import adapter


def single_cu(gear, suffix=""):
    node = f"CU{gear}_0{suffix}"
    return {"nodes": ["In", node, "Out"],
            "edges": [["In", node], [node, "Out"]]}


@pytest.mark.parametrize("gear", [1, 2, 3, 4])
@pytest.mark.parametrize("infer", [False, True])
@pytest.mark.parametrize("transport,kind", [("belt", 1), ("pipe", 2)])
def test_cu_adaptation_preserves_gear(gear, infer, transport, kind):
    graph = single_cu(gear, "__123")
    graph["inferNodeTypesFromDegrees"] = infer
    adapted = adapter.adapt_topoflow(graph, rows=3, columns=3, transport=transport)
    node = graph["nodes"][1]
    assert adapted.node_families[node] == f"CU{gear}"
    ports = adapted.problem.facilities[0].ports
    assert ports[kind] == ((2, 0),)
    assert ports[-kind] == ((0, 0),)
    assert len(adapted.external_edges) == 2


@pytest.mark.parametrize("rotation", range(4))
def test_result_retains_cu_and_rotated_external_ports(rotation):
    adapted = adapter.adapt_topoflow(single_cu(2), rows=3, columns=3)
    solution = {"machs": [{"id": 0, "pos": [1, 1], "rot": rotation}], "belts": []}
    result = adapter.enrich_solution(adapted, solution)
    machine = result["machs"][0]
    assert machine["control_gear"] == 2
    assert machine["topoflow_node_type"] == "CU2"
    external = result["topoflow"]["external_connections"]
    assert external[0]["port_assignment"]["dir"] == (2 + rotation) % 4
    assert external[1]["port_assignment"]["dir"] == rotation


@pytest.mark.parametrize("infer", [False, True])
def test_wrong_cu_degree_is_never_repaired(infer):
    graph = single_cu(2)
    graph["nodes"].append("C2_0")
    graph["edges"] = [["In", "CU2_0"], ["CU2_0", "C2_0"],
                      ["CU2_0", "C2_0"], ["C2_0", "Out"]]
    graph["inferNodeTypesFromDegrees"] = infer
    with pytest.raises(ValueError, match=r"CU2_0 must have degree"):
        adapter.adapt_topoflow(graph, rows=3, columns=3)


def test_mixed_network_keeps_distinct_facilities():
    graph = {"nodes": ["In", "S2_0", "CU2_0", "C2_0", "Out"],
             "edges": [["In", "S2_0"], ["S2_0", "CU2_0"],
                       ["CU2_0", "C2_0"], ["S2_0", "C2_0"], ["C2_0", "Out"]]}
    adapted = adapter.adapt_topoflow(graph, rows=4, columns=4,
                                    require_splitter_merger_belt_cell=True)
    assert adapted.node_families == {"S2_0": "S2", "CU2_0": "CU2", "C2_0": "C2"}
    assert len(adapted.internal_edges) == 3
    assert len(adapted.problem.facilities[0].ports[-1]) == 3
    assert len(adapted.problem.facilities[1].ports[-1]) == 1
    assert len(adapted.problem.facilities[2].ports[1]) == 3


@pytest.mark.parametrize("gear", [0, 5])
def test_invalid_gear(gear):
    with pytest.raises(ValueError, match="unsupported TopoFlow node"):
        adapter.adapt_topoflow(single_cu(gear), rows=3, columns=3)


@pytest.mark.skipif(importlib.util.find_spec("topoflow_native") is None,
                    reason="requires compiled topoflow_native Z3 extension")
@pytest.mark.parametrize("gear", [1, 2, 3, 4])
def test_native_cu_layout(gear):
    adapted = adapter.adapt_topoflow(single_cu(gear), rows=3, columns=3,
                                    terminal_anchors={"In": {"side": "left", "offset": 1},
                                                      "Out": {"side": "right", "offset": 1}})
    solution, _, _ = adapter.solve_topoflow(adapted, time_limit=10, workers=1)
    assert solution["machs"][0]["control_gear"] == gear
    assert len(solution["topoflow"]["external_connections"]) == 2
