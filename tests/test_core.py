from __future__ import annotations

import unittest
from fractions import Fraction

from constructor.operators.composition import boundary_flow
from constructor.service.core import ConstructionCore


class ConstructionCoreTests(unittest.TestCase):
    def test_unit_modules_are_cached(self) -> None:
        core = ConstructionCore()
        self.assertIs(core.unit(7), core.unit(7))

    def test_operator_facade_uses_certified_primitives(self) -> None:
        core = ConstructionCore()
        half = core.unit(2)
        third = core.unit(3)
        self.assertEqual(boundary_flow(core.add(half, third)), Fraction(5, 6))
        self.assertEqual(boundary_flow(core.multiply(third, half)), Fraction(1, 6))
        self.assertEqual(
            boundary_flow(core.sub(core.full(), third)), Fraction(2, 3)
        )

    def test_certificate_graph_and_size_are_cached(self) -> None:
        certificate = ConstructionCore().unit(3)
        self.assertIs(certificate.graph, certificate.graph)
        self.assertEqual(certificate.node_count, len(certificate.graph.nodes))
        self.assertEqual(certificate.edge_count, len(certificate.edges))


if __name__ == "__main__":
    unittest.main()
