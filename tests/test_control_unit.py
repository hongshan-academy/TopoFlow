from __future__ import annotations

import unittest
from fractions import Fraction

from constructor import boundary_flow, construct_fraction
from constructor.leaves.unit import control_certificate
from constructor.model.certificate import EdgeState, FlowCertificate
from constructor.verification.karzanov import crosscheck_karzanov
from graph import NodeKind, control_gear


class ControlUnitNodeTests(unittest.TestCase):
    def test_node_kind_and_gear_parsing(self) -> None:
        for gear in range(1, 5):
            node = f"CU{gear}_7"
            self.assertIs(NodeKind.from_node(node), getattr(NodeKind, f"CU{gear}"))
            self.assertEqual(NodeKind.from_node(node).degrees, (1, 1))
            self.assertEqual(control_gear(node), gear)
        for plain in ("In", "Out", "S2_0", "C3_12", "U1_0"):
            self.assertIsNone(control_gear(plain))

    def test_invalid_control_unit_names_are_rejected(self) -> None:
        for bad in ("CU0_0", "CU5_0", "CU_0", "CU1a_0", "CU1"):
            self.assertIsNone(control_gear(bad))
            with self.assertRaises(ValueError):
                NodeKind.from_node(bad)

    def test_control_module_is_exact_and_full_rank(self) -> None:
        for gear in range(1, 5):
            certificate = control_certificate(gear)
            self.assertEqual(certificate.validation_errors(), ())
            check = certificate.check()
            self.assertTrue(check.valid, check.errors)
            self.assertTrue(check.full_rank)
            self.assertEqual(boundary_flow(certificate), Fraction(gear, 5))

    def test_cross_check_matches_control_unit_speed(self) -> None:
        for gear in range(1, 5):
            result = crosscheck_karzanov(control_certificate(gear), Fraction(gear, 5))
            self.assertTrue(result.exact)
            self.assertEqual(result.flow, Fraction(gear, 5))

    def test_flow_above_control_unit_speed_is_rejected(self) -> None:
        node = "CU2_0"
        certificate = FlowCertificate(
            (("In", node), (node, "Out")),
            (Fraction(1, 2), Fraction(1, 2)),
            (EdgeState.FULL, EdgeState.EMPTY),
            frozenset({0}),
        )
        errors = certificate.validation_errors()
        self.assertTrue(any("outside" in error for error in errors), errors)

    def test_native_rejects_invalid_gear(self) -> None:
        import topoflow_native as native

        with self.assertRaises(ValueError):
            native.solve_karzanov(
                [("In", "CU0_0"), ("CU0_0", "Out")], 1_000, 1e-10
            )

    def test_old_u_naming_is_not_a_control_unit(self) -> None:
        # A plain ``(1, 1)`` node whose name is not a CU is a generic pass-through
        # splitter in TopoFlow, not a control unit.
        self.assertIsNone(control_gear("U1_0"))


class ControlUnitPlannerTests(unittest.TestCase):
    def test_five_family_values_use_a_single_control_module(self) -> None:
        for gear in range(1, 5):
            result = construct_fraction(gear, 5)
            self.assertEqual(result.reduction.root.kind, "control")
            self.assertEqual(result.reduction.cost.score, (3, 2))
            self.assertEqual(result.target, Fraction(gear, 5))
            self.assertEqual(result.reduction.steps(), (f"CU({gear})",))
            checked = crosscheck_karzanov(result.certificate, Fraction(gear, 5))
            self.assertTrue(checked.exact)

    def test_control_module_replaces_expensive_five_unit(self) -> None:
        legacy = {1: (6, 9), 4: (8, 12)}
        for gear, old in legacy.items():
            result = construct_fraction(gear, 5)
            self.assertLess(result.reduction.cost.score, old)

    def test_control_module_is_used_as_a_scaled_inner_factor(self) -> None:
        result = construct_fraction(1, 25)
        self.assertIn("CU(1)", " | ".join(result.reduction.steps()))
        self.assertLessEqual(result.reduction.cost.nodes, 7)
        self.assertTrue(
            crosscheck_karzanov(result.certificate, Fraction(1, 25)).exact
        )

    def test_control_module_is_used_as_an_addend(self) -> None:
        result = construct_fraction(7, 10)
        self.assertIn("CU(1)", " | ".join(result.reduction.steps()))
        self.assertEqual(result.reduction.cost.score, (7, 8))
        self.assertTrue(crosscheck_karzanov(result.certificate, Fraction(7, 10)).exact)

    def test_switch_disables_control_modules(self) -> None:
        for p, q in ((1, 5), (2, 5), (3, 5), (4, 5)):
            with self.subTest(target=f"{p}/{q}"):
                result = construct_fraction(p, q, control_units=False)
                self.assertNotEqual(result.reduction.root.kind, "control")
                self.assertNotIn("CU(", " | ".join(result.reduction.steps()))
                self.assertEqual(result.target, Fraction(p, q))

    def test_switch_restores_legacy_unit_terminal(self) -> None:
        result = construct_fraction(1, 25, control_units=False)
        self.assertEqual(result.reduction.root.kind, "unit")
        self.assertEqual(str(result.reduction.root), "U(25)")
        self.assertEqual(result.reduction.cost.score, (10, 17))

    def test_switch_is_enabled_by_default(self) -> None:
        default = construct_fraction(1, 25)
        explicit = construct_fraction(1, 25, control_units=True)
        self.assertEqual(default.reduction.cost.score, explicit.reduction.cost.score)
        self.assertEqual(default.reduction.root.kind, "scale")


if __name__ == "__main__":
    unittest.main()
