"""Independent verification with this package's Rust stable-polynomial backend."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction
from time import perf_counter

from ..model.certificate import FlowCertificate

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PolynomialCheck:
    target: Fraction
    flow: Fraction
    backend: str
    status: str
    iterations: int
    elapsed_seconds: float = 0.0

    @property
    def exact(self) -> bool:
        return self.status == "Optimal" and self.flow == self.target

    def require_exact(self) -> None:
        if not self.exact:
            raise RuntimeError(
                f"polynomial cross-check disagrees: target={self.target} "
                f"flow={self.flow} backend={self.backend} status={self.status}"
            )


def crosscheck_polynomial(
    certificate: FlowCertificate,
    target: Fraction,
    *,
    max_iterations: int = 100_000,
    tolerance: float = 1e-10,
) -> PolynomialCheck:
    """Run the bundled Rust stable-polynomial backend after exact certification."""
    started = perf_counter()
    errors = certificate.validation_errors()
    if errors:
        raise ValueError(f"cross-check requires a valid exact certificate: {errors}")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("max_iterations and tolerance must be positive")
    try:
        import topoflow_native as _native
    except ImportError as exc:
        raise RuntimeError(
            "topoflow_native Rust extension is not installed; run `uv sync` to build it"
        ) from exc
    logger.info(
        "polynomial cross-check start target=%s nodes=%d edges=%d",
        target,
        certificate.node_count,
        certificate.edge_count,
    )
    native = _native.solve_stable_polynomial(
        list(certificate.edges), max_iterations, tolerance
    )
    try:
        flow = Fraction(
            int(native.total_numerator),
            int(native.total_denominator),
        )
    except AttributeError as exc:
        raise RuntimeError(
            "topoflow_native is outdated and does not expose exact rational output; "
            "rebuild it with `uv sync`"
        ) from exc
    result = PolynomialCheck(
        target=target,
        flow=flow,
        backend=str(native.backend),
        status=str(native.status),
        iterations=int(native.iterations),
        elapsed_seconds=perf_counter() - started,
    )
    result.require_exact()
    logger.info(
        "polynomial cross-check passed backend=%s iterations=%d flow=%s elapsed=%.3fs",
        result.backend,
        result.iterations,
        result.flow,
        result.elapsed_seconds,
    )
    return result
