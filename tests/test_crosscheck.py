from __future__ import annotations

import unittest
from fractions import Fraction

import topoflow_native
from constructor.leaves.unit import half_certificate
from constructor.verification.polynomial import crosscheck_polynomial


class BundledPolynomialCrossCheckTests(unittest.TestCase):
    def test_native_extension_is_importable(self) -> None:
        self.assertTrue(topoflow_native.native_version())

    def test_half_graph_cross_checks_without_external_runtime(self) -> None:
        result = crosscheck_polynomial(half_certificate(), Fraction(1, 2))
        self.assertTrue(result.exact)
        self.assertEqual(result.flow, Fraction(1, 2))
        self.assertGreaterEqual(result.elapsed_seconds, 0)


if __name__ == "__main__":
    unittest.main()
