from __future__ import annotations

import unittest
from fractions import Fraction

from constructor import construct_fraction
from graph import Graph
from simulator import TopoFlowSimulator, simulate_frames


def _boundary(cycle, graph: Graph) -> Fraction:
    return sum(
        (
            Fraction(num, cycle["period"])
            for node, (num, _den) in cycle["node_ratios"].items()
            if node in graph.sinks
        ),
        Fraction(0),
    )


class StateSerializationTests(unittest.TestCase):
    def test_queue_distribution_is_part_of_the_state(self) -> None:
        graph = construct_fraction(1, 2).certificate.graph
        sim = TopoFlowSimulator(graph)
        edge_a, edge_b = sim._all_edges[0], sim._all_edges[1]

        sim.edge_queues[edge_a].append(1)
        state_a = sim.serialize_state()
        sim.edge_queues[edge_a].clear()
        sim.edge_queues[edge_b].append(1)
        state_b = sim.serialize_state()

        self.assertNotEqual(state_a, state_b)

    def test_sub_cell_position_is_part_of_the_state(self) -> None:
        graph = construct_fraction(1, 2).certificate.graph
        sim = TopoFlowSimulator(graph)
        transport = sim._transports[sim._all_edges[0]]

        transport.entries.append(1)
        state_at_tail = sim.serialize_state()
        # Positions 1 and 4 are in the same belt cell (eight positions per
        # cell) but are different physical states.
        transport.entries[0] = 4
        state_inside_cell = sim.serialize_state()

        self.assertNotEqual(state_at_tail, state_inside_cell)

    def test_true_tick_period_is_reported_separately_from_frames(self) -> None:
        # A zero-length pipe handshake with the two-tick source cadence has a
        # two-tick period.  At frame resolution that aliases to one frame.
        graph = Graph.from_edges([("In", "Out")])
        result = simulate_frames(
            graph, max_frames=10, transport="pipe", edge_length=0
        )
        cycle = result["cycle"]
        self.assertEqual(cycle["period_ticks"], 2)
        self.assertEqual(cycle["period"], 1)
        self.assertEqual(cycle["node_ratios"]["Out"], (4, 1))

    def test_pipe_control_unit_admits_nine_items_per_five_frames(self) -> None:
        graph = Graph.from_edges([("In", "PCU9_0"), ("PCU9_0", "Out")])
        result = simulate_frames(graph, max_frames=200, transport="pipe")
        cycle = result["cycle"]
        self.assertEqual(cycle["period_ticks"], 40)
        self.assertEqual(cycle["period"], 5)
        self.assertEqual(cycle["node_ratios"]["PCU9_0"], (9, 5))
        self.assertEqual(cycle["node_ratios"]["Out"], (9, 5))

    def test_325_799_reaches_the_exact_boundary(self) -> None:
        # Regression for the old ``extend(deque)`` serialization: the false
        # 5-frame cycle reported 2/5 instead of 325/799.
        graph = construct_fraction(325, 799).certificate.graph
        result = simulate_frames(graph, max_frames=None)
        self.assertEqual(_boundary(result["cycle"], graph), Fraction(325, 799))


if __name__ == "__main__":
    unittest.main()
