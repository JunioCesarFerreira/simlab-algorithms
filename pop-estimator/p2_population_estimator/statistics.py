"""Small statistics helpers used by the estimator.

All functions accept ``list[float]`` or any iterable of numbers. They raise
``ValueError`` on degenerate inputs (e.g. n < 2 for variance), so the caller
can decide how to handle the failure (e.g. mark a block as
``insufficient_samples``).
"""

from __future__ import annotations

import math
import statistics as stdstats
from typing import Iterable


def _as_list(values: Iterable[float]) -> list[float]:
    out = [float(v) for v in values]
    return out


def sample_mean(values: Iterable[float]) -> float:
    xs = _as_list(values)
    if not xs:
        raise ValueError("sample_mean requires at least one value")
    return sum(xs) / len(xs)


def sample_variance(values: Iterable[float], ddof: int = 1) -> float:
    """Sample variance with default Bessel correction."""
    xs = _as_list(values)
    n = len(xs)
    if n - ddof <= 0:
        raise ValueError(
            f"sample_variance requires n > ddof; got n={n}, ddof={ddof}"
        )
    m = sum(xs) / n
    return sum((x - m) ** 2 for x in xs) / (n - ddof)


def sample_std(values: Iterable[float], ddof: int = 1) -> float:
    return math.sqrt(sample_variance(values, ddof=ddof))


def standard_error(values: Iterable[float]) -> float:
    xs = _as_list(values)
    n = len(xs)
    if n < 2:
        raise ValueError("standard_error requires n >= 2")
    return sample_std(xs) / math.sqrt(n)


def confidence_interval_95(values: Iterable[float]) -> tuple[float, float]:
    """Two-sided 95% CI using a normal approximation (z=1.96)."""
    xs = _as_list(values)
    m = sample_mean(xs)
    se = standard_error(xs)
    return (m - 1.96 * se, m + 1.96 * se)


def delta_samples(F_star: Iterable[float], F_local: Iterable[float]) -> list[float]:
    """Compute element-wise Delta_i^{(r)} = F_star^{(r)} - F_local^{(r)}."""
    a = _as_list(F_star)
    b = _as_list(F_local)
    if len(a) != len(b):
        raise ValueError(
            f"delta_samples: F_star and F_local must have the same length "
            f"({len(a)} vs {len(b)})"
        )
    return [x - y for x, y in zip(a, b)]


def d_hat(F_star: Iterable[float], F_local: Iterable[float]) -> float:
    return sample_mean(delta_samples(F_star, F_local))


def sigma_BB_hat(F_star: Iterable[float], F_local: Iterable[float]) -> float:
    """sigma_BB_i_hat is the sample standard deviation (ddof=1) of the deltas."""
    d = delta_samples(F_star, F_local)
    return sample_std(d, ddof=1)
