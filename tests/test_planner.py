from __future__ import annotations

import unittest
from fractions import Fraction

from constructor.leaves.interval import plan_interval
from constructor.model.recipe import ReductionRecipe, TopologyCost
from constructor.search.dp import (
    ADDITION_CANDIDATE_LIMIT,
    SIMPLE_FRACTIONS,
    SIMPLE_PRIME_DENOMINATORS,
    SUB_CANDIDATE_LIMIT,
    ReductionPlanner,
    _prime_power_blocks,
)
from constructor.service.core import ConstructionCore


class ReductionPlannerTests(unittest.TestCase):
    def test_simple_fraction_catalog_uses_only_small_prime_denominators(self) -> None:
        self.assertEqual(SIMPLE_PRIME_DENOMINATORS, (2, 3, 5, 7, 11, 13))
        self.assertTrue(SIMPLE_FRACTIONS)
        self.assertTrue(
            all(
                value.denominator in SIMPLE_PRIME_DENOMINATORS
                for value in SIMPLE_FRACTIONS
            )
        )

    def test_addition_splits_are_bounded_exact_and_denominator_safe(self) -> None:
        planner = ReductionPlanner(ConstructionCore())
        for target in (
            Fraction(28, 39),
            Fraction(26, 35),
            Fraction(31, 40),
            Fraction(64, 799),
        ):
            with self.subTest(target=target):
                splits = planner._addition_splits(target)
                self.assertLessEqual(len(splits), ADDITION_CANDIDATE_LIMIT)
                self.assertTrue(splits)
                self.assertTrue(all(left + right == target for left, right in splits))
                self.assertTrue(
                    all(
                        max(left.denominator, right.denominator) <= target.denominator
                        for left, right in splits
                    )
                )

    def test_sub_splits_are_ordered_and_denominator_safe(self) -> None:
        planner = ReductionPlanner(ConstructionCore())
        target = Fraction(7, 30)
        splits = planner._sub_splits(target)
        self.assertLessEqual(len(splits), SUB_CANDIDATE_LIMIT)
        self.assertIn((Fraction(2, 5), Fraction(1, 6)), splits)
        self.assertIn((Fraction(1, 3), Fraction(1, 10)), splits)
        self.assertTrue(
            all(
                minuend - subtrahend == target
                and minuend > 2 * subtrahend
                and max(minuend.denominator, subtrahend.denominator)
                <= target.denominator
                for minuend, subtrahend in splits
            )
        )

    def test_factor_aware_splits_find_prime_power_partitions(self) -> None:
        planner = ReductionPlanner(ConstructionCore())
        self.assertEqual(_prime_power_blocks(799), (17, 47))
        self.assertIn(
            (Fraction(1, 47), Fraction(1, 17)),
            planner._addition_splits(Fraction(64, 799)),
        )
        self.assertIn(
            (Fraction(1, 3), Fraction(5, 13)),
            planner._addition_splits(Fraction(28, 39)),
        )
        self.assertIn(
            (Fraction(3, 8), Fraction(2, 5)),
            planner._addition_splits(Fraction(31, 40)),
        )

    def test_every_planned_binary_recipe_obeys_the_denominator_rule(self) -> None:
        planner = ReductionPlanner(ConstructionCore())

        def check(recipe: ReductionRecipe) -> None:
            if recipe.kind in ("add", "sub"):
                self.assertLessEqual(
                    max(child.target.denominator for child in recipe.children),
                    recipe.target.denominator,
                    str(recipe),
                )
            for child in recipe.children:
                check(child)

        for target in (
            Fraction(7, 30),
            Fraction(28, 39),
            Fraction(650, 799),
            Fraction(17, 32),
        ):
            with self.subTest(target=target):
                check(planner._solve(target, 3))

    def test_binary_remainder_skips_denominator_growth(self) -> None:
        planner = ReductionPlanner(ConstructionCore())
        self.assertIsNone(planner._binary_remainder(Fraction(12, 23)))
        self.assertIsNotNone(planner._binary_remainder(Fraction(17, 32)))

    def test_recipe_factories_track_dependencies_and_validate_shape(self) -> None:
        third = ReductionRecipe.for_unit(3, TopologyCost(4, 5, 1))
        full_sub = ReductionRecipe.for_sub(
            ReductionRecipe.for_full(), third, label="sub-full"
        )
        twenty_third = ReductionRecipe.for_unit(23, TopologyCost(8, 12, 1, exact=False))
        scaled = ReductionRecipe.for_scale(twenty_third, full_sub, label="scale")
        self.assertEqual(scaled.factor, 23)
        self.assertEqual(
            scaled.steps(),
            (
                "U(23)",
                "1",
                "U(3)",
                "(1) - (U(3))",
                "((1) - (U(3)))/23",
            ),
        )
        fifth = ReductionRecipe.for_unit(5, TopologyCost(6, 8, 1))
        added = ReductionRecipe.for_add(third, fifth, label="add")
        self.assertEqual(str(added), "(U(3)) + (U(5))")
        with self.assertRaisesRegex(ValueError, "invalid add"):
            ReductionRecipe.for_add(
                ReductionRecipe.for_unit(2, TopologyCost(4, 4, 1)),
                ReductionRecipe.for_unit(2, TopologyCost(4, 4, 1)),
                label="invalid",
            )
        with self.assertRaisesRegex(ValueError, "requires a unit"):
            ReductionRecipe.for_scale(full_sub, third, label="invalid")
        with self.assertRaisesRegex(ValueError, "invalid sub"):
            ReductionRecipe.for_sub(
                plan_interval_recipe(Fraction(2, 5)), fifth, label="invalid"
            )

    def test_interval_notation_has_one_rendering_source(self) -> None:
        recipe = plan_interval_recipe(Fraction(5, 13))
        self.assertEqual(str(recipe), "I(5, 13)[mixed-radix]")
        self.assertEqual(recipe.steps(), (str(recipe),))


def plan_interval_recipe(target: Fraction) -> ReductionRecipe:
    return ReductionRecipe.for_interval(plan_interval(target))


if __name__ == "__main__":
    unittest.main()
