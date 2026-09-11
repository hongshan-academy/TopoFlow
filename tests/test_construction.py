from __future__ import annotations

import math
import unittest
from fractions import Fraction
from unittest.mock import patch

from constructor import boundary_flow, construct_fraction
from constructor.model import certificate as certificate_module


class UniversalConstructionTests(unittest.TestCase):
    def assert_constructs(self, p: int, q: int, **kwargs):
        result = construct_fraction(p, q, **kwargs)
        self.assertEqual(result.target, Fraction(p, q))
        self.assertEqual(boundary_flow(result.certificate), Fraction(p, q))
        self.assertTrue(result.validation.valid, result.validation.errors)
        self.assertTrue(result.validation.full_rank)
        self.assertEqual(
            result.reduction.cost.score,
            (result.certificate.node_count, result.certificate.edge_count),
        )
        return result

    def test_interval_and_endpoints(self) -> None:
        for p, q in ((1, 3), (4, 9), (1, 2), (2, 3)):
            with self.subTest(target=f"{p}/{q}"):
                self.assert_constructs(p, q, reduction_depth=1)

    def test_unit_fraction_is_a_hard_terminal(self) -> None:
        result = self.assert_constructs(1, 25)
        self.assertEqual(result.reduction.root.kind, "unit")
        self.assertEqual(str(result.reduction.root), "U(25)")
        self.assertEqual(result.reduction.steps(), ("U(25)",))

    def test_full_sub_of_unit_is_a_fast_terminal(self) -> None:
        result = self.assert_constructs(324, 325)
        self.assertEqual(result.reduction.strategy, "sub-full-unit")
        self.assertEqual(str(result.reduction.root), "(1) - (U(325))")
        self.assertEqual(
            result.reduction.steps(),
            ("1", "U(325)", "(1) - (U(325))"),
        )
        self.assertNotIn("109", " ".join(result.reduction.steps()))

    def test_restricted_sub_improves_known_target(self) -> None:
        result = self.assert_constructs(7, 30)
        self.assertEqual(result.reduction.cost.score, (12, 18))
        self.assertEqual(result.reduction.root.kind, "sub")

    def test_rank_solver_runs_only_for_the_final_graph(self) -> None:
        original = certificate_module._matrix_rank
        with patch(
            "constructor.model.certificate._matrix_rank", wraps=original
        ) as rank:
            self.assert_constructs(7, 30)
        self.assertEqual(rank.call_count, 1)

    def test_known_smart_reduction_improvements(self) -> None:
        expected = {
            (28, 39): (12, 19),
            (26, 35): (14, 22),
            (16, 21): (10, 16),
            (11, 39): (14, 23),
            (31, 40): (13, 20),
            (29, 35): (12, 19),
        }
        for (p, q), cost in expected.items():
            with self.subTest(target=f"{p}/{q}"):
                result = self.assert_constructs(p, q)
                self.assertEqual(result.reduction.cost.score, cost)

    def test_existing_compact_examples_do_not_regress(self) -> None:
        for p, q, maximum in ((325, 799, (17, 28)), (9, 14, (10, 15))):
            with self.subTest(target=f"{p}/{q}"):
                result = self.assert_constructs(p, q)
                self.assertLessEqual(result.reduction.cost.score, maximum)

    def test_timing_breakdown_is_reported(self) -> None:
        timing = self.assert_constructs(10, 21).timing
        self.assertGreaterEqual(timing.planning_seconds, 0)
        self.assertGreaterEqual(timing.unit_search_seconds, 0)
        self.assertGreaterEqual(timing.realization_seconds, 0)
        self.assertGreaterEqual(timing.validation_seconds, 0)
        self.assertGreaterEqual(timing.total_seconds, 0)

    def test_scale_unit_dependency_is_resolved_before_realization(self) -> None:
        result = self.assert_constructs(2, 69)
        self.assertTrue(result.reduction.cost.exact)
        self.assertIn("U(23)", result.reduction.steps())
        self.assertGreaterEqual(result.reduction.stats.unit_resolutions, 1)

    def test_small_reduced_fractions(self) -> None:
        for q in range(2, 16):
            for p in range(1, q):
                if math.gcd(p, q) != 1:
                    continue
                with self.subTest(target=f"{p}/{q}"):
                    self.assert_constructs(p, q, reduction_depth=1)

    def test_invalid_options(self) -> None:
        with self.assertRaisesRegex(ValueError, "reduction_depth"):
            construct_fraction(1, 3, reduction_depth=-1)
        with self.assertRaisesRegex(ValueError, "reduction_state_limit"):
            construct_fraction(1, 3, reduction_state_limit=0)

    def test_search_limit_reports_no_construction(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "state limit"):
            construct_fraction(2, 7, reduction_state_limit=1)

    def test_invalid_targets(self) -> None:
        for p, q in ((0, 3), (3, 3), (4, 3), (1, 0), (1, -2)):
            with (
                self.subTest(target=f"{p}/{q}"),
                self.assertRaisesRegex(ValueError, "0 < p < q"),
            ):
                construct_fraction(p, q)


if __name__ == "__main__":
    unittest.main()
