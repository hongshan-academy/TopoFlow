"""End-to-end regressions for native solvers with multiple boundaries."""

from __future__ import annotations

import unittest

from solver import solve_native


MULTI_BOUNDARY_NODES = [
    {"id": "InA", "type": "In"},
    {"id": "InB", "type": "In"},
    {"id": "Merge", "type": "C"},
    {"id": "Split", "type": "S"},
    {"id": "OutA", "type": "Out"},
    {"id": "OutB", "type": "Out"},
]
MULTI_BOUNDARY_EDGES = [
    {"id": "e1", "from": "InA", "to": "Merge"},
    {"id": "e2", "from": "InB", "to": "Merge"},
    {"id": "e3", "from": "Merge", "to": "Split"},
    {"id": "e4", "from": "Split", "to": "OutA"},
    {"id": "e5", "from": "Split", "to": "OutB"},
]

EXAMPLE_NODES = [
    {"id": "In1", "type": "In"},
    {"id": "C1", "type": "C"},
    {"id": "S1", "type": "S"},
    {"id": "Out1", "type": "Out"},
    {"id": "C2", "type": "C"},
    {"id": "S2", "type": "S"},
]
EXAMPLE_EDGES = [
    {"id": "e1", "from": "In1", "to": "C1"},
    {"id": "e2", "from": "C1", "to": "S1"},
    {"id": "e3", "from": "S1", "to": "Out1"},
    {"id": "e4", "from": "S1", "to": "C2"},
    {"id": "e5", "from": "S1", "to": "S2"},
    {"id": "e6", "from": "S2", "to": "C2"},
    {"id": "e7", "from": "C2", "to": "C1"},
    {"id": "e8", "from": "S2", "to": "C1"},
]


class NativeMultiBoundarySolverTests(unittest.TestCase):
    def test_non_reserved_terminal_ids_and_extra_input(self) -> None:
        cases = [
            (EXAMPLE_NODES, EXAMPLE_EDGES, "2/5"),
            (
                [*EXAMPLE_NODES, {"id": "In2", "type": "In"}],
                [*EXAMPLE_EDGES, {"id": "e9", "from": "In2", "to": "C2"}],
                "4/9",
            ),
        ]
        for nodes, edges, expected_total in cases:
            for engine in ("rank-smt", "karzanov"):
                with self.subTest(engine=engine, total=expected_total):
                    result = solve_native(engine, nodes, edges, workers=1)
                    self.assertEqual(result["totalText"], expected_total)

    def test_connected_multi_input_multi_output_graph(self) -> None:
        for engine in ("rank-smt", "karzanov"):
            with self.subTest(engine=engine):
                result = solve_native(engine, MULTI_BOUNDARY_NODES, MULTI_BOUNDARY_EDGES, workers=1)
                self.assertTrue(result["feasible"])
                self.assertEqual(result["totalText"], "1/1")
                self.assertEqual(
                    [edge["text"] for edge in result["edgeFlows"]],
                    ["1/2", "1/2", "1/1", "1/2", "1/2"],
                )

    def test_disconnected_boundary_components_aggregate_total(self) -> None:
        nodes = [
            {"id": "InputA", "type": "In"},
            {"id": "OutputA", "type": "Out"},
            {"id": "InputB", "type": "In"},
            {"id": "OutputB", "type": "Out"},
        ]
        edges = [
            {"id": "a", "from": "InputA", "to": "OutputA"},
            {"id": "b", "from": "InputB", "to": "OutputB"},
        ]
        for engine in ("rank-smt", "karzanov"):
            with self.subTest(engine=engine):
                result = solve_native(engine, nodes, edges, workers=1)
                self.assertEqual(result["totalText"], "2/1")
                self.assertEqual([edge["text"] for edge in result["edgeFlows"]], ["1/1", "1/1"])

    def test_terminal_type_must_match_boundary_degree(self) -> None:
        nodes = [
            {"id": "In", "type": "In"},
            {"id": "UndeclaredInput", "type": "S"},
            {"id": "Merge", "type": "C"},
            {"id": "Out", "type": "Out"},
        ]
        edges = [
            {"id": "e1", "from": "In", "to": "Merge"},
            {"id": "e2", "from": "UndeclaredInput", "to": "Merge"},
            {"id": "e3", "from": "Merge", "to": "Out"},
        ]
        with self.assertRaisesRegex(ValueError, "declared In"):
            solve_native("karzanov", nodes, edges, workers=1)
