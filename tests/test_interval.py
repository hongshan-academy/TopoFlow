from __future__ import annotations

import unittest
from fractions import Fraction

from constructor.leaves.interval import (
    _merge_arities,
    construct_interval,
    plan_interval,
    realize_interval,
)
from constructor.leaves.unit import (
    construct_unit,
    half_certificate,
    third_certificate,
)
from constructor.operators.composition import (
    boundary_flow,
    full_certificate,
    parallel_sum,
    restricted_sub,
)


class IntervalConstructionTests(unittest.TestCase):
    def test_guaranteed_four_ninths(self) -> None:
        result = construct_interval(Fraction(4, 9))
        self.assertFalse(result.optimized)
        self.assertEqual(boundary_flow(result.certificate), Fraction(4, 9))
        self.assertTrue(result.certificate.check().full_rank)

    def test_optimizer_returns_a_certified_graph(self) -> None:
        result = construct_interval(Fraction(4, 9), optimize=True)
        self.assertTrue(result.optimized)
        self.assertEqual(boundary_flow(result.certificate), Fraction(4, 9))
        self.assertTrue(result.certificate.check().full_rank)

    def test_mixed_radix_constructs_325_over_799_directly(self) -> None:
        result = construct_interval(Fraction(325, 799))
        self.assertEqual(result.strategy, "mixed-radix")
        self.assertEqual(len(result.certificate.graph.nodes), 17)
        self.assertEqual(len(result.certificate.edges), 28)

    def test_symbolic_interval_cost_matches_realized_graph(self) -> None:
        targets = {Fraction(325, 799)}
        targets.update(
            Fraction(p, q)
            for q in range(2, 41)
            for p in range(1, q)
            if Fraction(1, 3) <= Fraction(p, q) <= Fraction(1, 2)
        )
        for target in sorted(targets):
            with self.subTest(target=target):
                plan = plan_interval(target)
                certificate = realize_interval(plan)
                self.assertEqual(
                    plan.topology_size,
                    (certificate.node_count, certificate.edge_count),
                )

    def test_merge_arities_cover_every_chain_shape(self) -> None:
        self.assertEqual(_merge_arities(0, has_base=False), ())
        self.assertEqual(_merge_arities(1, has_base=False), ())
        self.assertEqual(_merge_arities(2, has_base=False), (2,))
        self.assertEqual(_merge_arities(3, has_base=False), (3,))
        self.assertEqual(_merge_arities(6, has_base=False), (3, 3, 2))
        self.assertEqual(_merge_arities(0, has_base=True), ())
        self.assertEqual(_merge_arities(1, has_base=True), (2,))
        self.assertEqual(_merge_arities(2, has_base=True), (3,))
        self.assertEqual(_merge_arities(5, has_base=True), (3, 3, 2))

    def test_full_sub_tile(self) -> None:
        inner = construct_interval(Fraction(3, 7)).certificate
        outer = restricted_sub(full_certificate(), inner)
        self.assertEqual(boundary_flow(outer), Fraction(4, 7))
        self.assertTrue(outer.check().full_rank)

    def test_restricted_sub_and_strict_boundary(self) -> None:
        minuend = construct_interval(Fraction(2, 5)).certificate
        result = restricted_sub(minuend, construct_unit(6))
        self.assertEqual(boundary_flow(result), Fraction(7, 30))
        self.assertEqual((result.node_count, result.edge_count), (12, 18))
        self.assertTrue(result.check().full_rank)
        with self.assertRaisesRegex(ValueError, "a > 2b"):
            restricted_sub(minuend, construct_unit(5))

    def test_parallel_sum_operator(self) -> None:
        result = parallel_sum(half_certificate(), third_certificate())
        self.assertEqual(boundary_flow(result), Fraction(5, 6))
        self.assertTrue(result.check().full_rank)

    def test_parallel_sum_rejects_full_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "x \\+ y < 1"):
            parallel_sum(half_certificate(), half_certificate())

    def test_interval_validation(self) -> None:
        for target in (Fraction(1, 4), Fraction(3, 5)):
            with (
                self.subTest(target=target),
                self.assertRaisesRegex(ValueError, "1/3 <= x <= 1/2"),
            ):
                construct_interval(target)


if __name__ == "__main__":
    unittest.main()
