from __future__ import annotations

import unittest

from constructor.model.arithmetic import ceil_log3, factor_power_of_two


class ArithmeticTests(unittest.TestCase):
    def test_factor_power_of_two(self) -> None:
        self.assertEqual(factor_power_of_two(1), (0, 1))
        self.assertEqual(factor_power_of_two(40), (3, 5))

    def test_ceil_log3(self) -> None:
        self.assertEqual(ceil_log3(1), 0)
        self.assertEqual(ceil_log3(9), 2)
        self.assertEqual(ceil_log3(10), 3)

    def test_positive_inputs_are_required(self) -> None:
        for function in (factor_power_of_two, ceil_log3):
            with (
                self.subTest(function=function.__name__),
                self.assertRaises(ValueError),
            ):
                function(0)


if __name__ == "__main__":
    unittest.main()
