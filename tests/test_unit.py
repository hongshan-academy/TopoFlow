from __future__ import annotations

import unittest
from fractions import Fraction

from constructor.leaves.unit import (
    candidate_pair_count,
    construct_unit,
    plan_unit,
    realize_unit,
    search_odd_unit,
)
from constructor.operators.composition import boundary_flow


class UnitSearchTests(unittest.TestCase):
    def test_candidate_counts(self) -> None:
        self.assertEqual(candidate_pair_count(6), 259_560)
        self.assertEqual(candidate_pair_count(7), 12_703_320)

    def test_mixed_uses_exhaustion_for_small_layer(self) -> None:
        result = search_odd_unit(23, max_k=4)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.k, 4)
        self.assertEqual(result.backend, "exhaustion")

    def test_mixed_can_force_milp_by_threshold(self) -> None:
        result = search_odd_unit(5, max_k=2, exhaustion_pair_limit=1)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.backend, "milp")

    def test_large_layer_can_hit_in_exhaustion_prefix(self) -> None:
        result = search_odd_unit(799, max_k=7)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.k, 7)
        self.assertEqual(result.backend, "exhaustion-prefix")

    def test_even_and_odd_factorization(self) -> None:
        for denominator in (2, 3, 6, 8):
            with self.subTest(denominator=denominator):
                certificate = construct_unit(denominator)
                self.assertEqual(boundary_flow(certificate), Fraction(1, denominator))
                self.assertEqual(len(certificate.fixed), 1)
                self.assertTrue(certificate.check().full_rank)

    def test_bound_can_report_no_witness(self) -> None:
        self.assertIsNone(search_odd_unit(23, max_k=3))

    def test_plan_and_realization_have_the_same_cost(self) -> None:
        plan = plan_unit(25)
        self.assertEqual(plan.topology_size, (10, 17))
        certificate = realize_unit(plan)
        self.assertEqual(
            (certificate.node_count, certificate.edge_count), plan.topology_size
        )


if __name__ == "__main__":
    unittest.main()
