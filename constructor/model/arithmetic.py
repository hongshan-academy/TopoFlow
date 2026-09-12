"""Small exact-integer helpers shared by planning and admissible bounds."""


def factor_power_of_two(value: int) -> tuple[int, int]:
    """Return ``(exponent, odd_part)`` for a positive integer."""
    if value < 1:
        raise ValueError("value must be positive")
    exponent = 0
    odd_part = value
    while odd_part % 2 == 0:
        exponent += 1
        odd_part //= 2
    return exponent, odd_part


def ceil_log3(value: int) -> int:
    """Return the least non-negative ``k`` satisfying ``3**k >= value``."""
    if value < 1:
        raise ValueError("value must be positive")
    exponent = 0
    capacity = 1
    while capacity < value:
        exponent += 1
        capacity *= 3
    return exponent


def unit_topology_size(twos: int, odd_k: int) -> tuple[int, int]:
    """Return ``(nodes, edges)`` of a unit module with ``2**twos * 3**odd_k`` parts."""
    return 2 * (twos + odd_k) + 2, 3 * twos + 4 * odd_k + 1
