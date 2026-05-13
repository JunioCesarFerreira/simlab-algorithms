"""Population-size estimator.

We implement two estimators, both derived from the gambler-ruin
approximation:

  - Uniform-initialisation estimate (sec. 2.11 of the theoretical note):

        n_i_hat = -ln(alpha) * 2^{k_i - 1} * sigma_BB_i * sqrt(2m) / d_i

  - Bernoulli-initialisation estimate, when bits are drawn iid Bernoulli(rho):

        n_i_hat = (-ln(alpha) / (2 * pi_i(H_i*))) * sigma_BB_i * sqrt(2m) / d_i

  with ``pi_i(H_i^*) = rho^{s_i} (1-rho)^{k_i - s_i}``.

The block-level estimator returns a :class:`BlockComparisonResult`. The
global estimator returns ``max_i ceil(n_i_hat)`` over **valid** blocks.

Degenerate cases are handled explicitly:

  - d_i_hat <= 0          -> ``invalid_non_positive_d``
  - sigma_BB == 0         -> ``degenerate_zero_variance`` (warning;
                              we still output n_i_hat using SIGMA_FLOOR)
  - R < 2                 -> ``insufficient_samples``
  - alpha not in (0,1)    -> ValueError
  - pi_i == 0             -> ValueError
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

from p2_population_estimator.config import (
    D_POSITIVE_EPS,
    SIGMA_FLOOR,
    STATUS_DEGENERATE_VARIANCE,
    STATUS_INSUFFICIENT_SAMPLES,
    STATUS_NON_POSITIVE_D,
    STATUS_OK,
)
from p2_population_estimator.logging_utils import get_logger
from p2_population_estimator.models import (
    BlockComparisonResult,
    BlockPattern,
)
from p2_population_estimator.statistics import (
    d_hat,
    delta_samples,
    sigma_BB_hat,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Closed-form estimators
# ---------------------------------------------------------------------------
def estimate_uniform(
    alpha: float, k_i: int, m: int, sigma_BB_i: float, d_i: float
) -> float:
    """n_i_hat for uniform initialisation."""
    _check_alpha(alpha)
    if k_i <= 0:
        raise ValueError("k_i must be >= 1")
    if m <= 0:
        raise ValueError("m must be >= 1")
    if d_i <= 0:
        raise ValueError("d_i must be > 0; check the caller for status handling")
    sigma = max(sigma_BB_i, SIGMA_FLOOR)
    return -math.log(alpha) * (2 ** (k_i - 1)) * sigma * math.sqrt(2 * m) / d_i


def estimate_bernoulli(
    alpha: float, pi_i: float, m: int, sigma_BB_i: float, d_i: float
) -> float:
    """n_i_hat for Bernoulli(rho) initialisation."""
    _check_alpha(alpha)
    if not (0.0 < pi_i <= 1.0):
        raise ValueError("pi_i must be in (0, 1]")
    if m <= 0:
        raise ValueError("m must be >= 1")
    if d_i <= 0:
        raise ValueError("d_i must be > 0; check the caller for status handling")
    sigma = max(sigma_BB_i, SIGMA_FLOOR)
    return (-math.log(alpha) / (2.0 * pi_i)) * sigma * math.sqrt(2 * m) / d_i


def _check_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}")


# ---------------------------------------------------------------------------
# Block-level orchestration
# ---------------------------------------------------------------------------
def estimate_block(
    *,
    block_id: int,
    k_i: int,
    m: int,
    alpha: float,
    h_star: BlockPattern,
    F_star: list[float],
    F_local: list[float],
    rho: Optional[float] = None,
) -> BlockComparisonResult:
    """Compute d_i, sigma_BB_i, n_i for one block.

    Always returns a result object (status field reports degenerate cases).
    """
    warnings: list[str] = []
    deltas = delta_samples(F_star, F_local)
    R = len(deltas)
    s_i_star = h_star.s

    # Insufficient samples ---------------------------------------------------
    if R < 2:
        return BlockComparisonResult(
            block_id=block_id,
            k_i=k_i,
            s_i_star=s_i_star,
            alpha=alpha,
            pi_i_star=_pi(h_star, rho),
            d_i_hat=float("nan"),
            sigma_BB_i_hat=float("nan"),
            delta_samples=deltas,
            F_star_samples=list(F_star),
            F_local_samples=list(F_local),
            n_i_uniform=None,
            n_i_uniform_ceil=None,
            n_i_bernoulli=None,
            n_i_bernoulli_ceil=None,
            status=STATUS_INSUFFICIENT_SAMPLES,
            warnings=["R<2; cannot estimate sample variance."],
        )

    d_i = d_hat(F_star, F_local)
    sigma = sigma_BB_hat(F_star, F_local)
    pi_i = _pi(h_star, rho)

    status = STATUS_OK
    if d_i <= D_POSITIVE_EPS:
        status = STATUS_NON_POSITIVE_D
        warnings.append(
            f"d_i_hat={d_i:.4g} is not positive; H_i^* was not consistently "
            "better than H_i^L. Consider revising the H_i^* heuristic, the "
            "H_i^L competitor, or the scalarisation."
        )
        return BlockComparisonResult(
            block_id=block_id,
            k_i=k_i,
            s_i_star=s_i_star,
            alpha=alpha,
            pi_i_star=pi_i,
            d_i_hat=d_i,
            sigma_BB_i_hat=sigma,
            delta_samples=deltas,
            F_star_samples=list(F_star),
            F_local_samples=list(F_local),
            n_i_uniform=None,
            n_i_uniform_ceil=None,
            n_i_bernoulli=None,
            n_i_bernoulli_ceil=None,
            status=status,
            warnings=warnings,
        )

    if sigma <= SIGMA_FLOOR:
        status = STATUS_DEGENERATE_VARIANCE
        warnings.append(
            f"sigma_BB_i_hat={sigma:.4g} is effectively zero; "
            "n_i_hat is computed using a small floor for stability."
        )

    n_uniform = estimate_uniform(alpha, k_i, m, sigma, d_i)
    n_uniform_ceil = math.ceil(n_uniform)
    n_bernoulli: Optional[float] = None
    n_bernoulli_ceil: Optional[int] = None
    if pi_i is not None:
        if pi_i <= 0.0:
            raise ValueError("pi_i(H_i^*) is zero; cannot use Bernoulli estimator.")
        n_bernoulli = estimate_bernoulli(alpha, pi_i, m, sigma, d_i)
        n_bernoulli_ceil = math.ceil(n_bernoulli)

    return BlockComparisonResult(
        block_id=block_id,
        k_i=k_i,
        s_i_star=s_i_star,
        alpha=alpha,
        pi_i_star=pi_i,
        d_i_hat=d_i,
        sigma_BB_i_hat=sigma,
        delta_samples=deltas,
        F_star_samples=list(F_star),
        F_local_samples=list(F_local),
        n_i_uniform=n_uniform,
        n_i_uniform_ceil=n_uniform_ceil,
        n_i_bernoulli=n_bernoulli,
        n_i_bernoulli_ceil=n_bernoulli_ceil,
        status=status,
        warnings=warnings,
    )


def _pi(h_star: BlockPattern, rho: Optional[float]) -> Optional[float]:
    if rho is None:
        return None
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0, 1) when provided")
    return (rho ** h_star.s) * ((1.0 - rho) ** (h_star.k - h_star.s))


# ---------------------------------------------------------------------------
# Global aggregation
# ---------------------------------------------------------------------------
def aggregate_global(
    block_results: Iterable[BlockComparisonResult],
) -> dict[str, object]:
    """Aggregate per-block results into a global estimate."""
    valid = [b for b in block_results if b.status in (STATUS_OK, STATUS_DEGENERATE_VARIANCE)]
    invalid = [b for b in block_results if b not in valid]

    n_uniform_values = [b.n_i_uniform_ceil for b in valid if b.n_i_uniform_ceil is not None]
    n_bern_values = [b.n_i_bernoulli_ceil for b in valid if b.n_i_bernoulli_ceil is not None]

    if n_uniform_values:
        n_hat_uniform = max(n_uniform_values)
        most_difficult = max(valid, key=lambda b: (b.n_i_uniform_ceil or -1)).block_id
    else:
        n_hat_uniform = None
        most_difficult = None

    n_hat_bernoulli = max(n_bern_values) if n_bern_values else None

    return {
        "n_hat_uniform": n_hat_uniform,
        "n_hat_bernoulli": n_hat_bernoulli,
        "num_valid_blocks": len(valid),
        "num_invalid_blocks": len(invalid),
        "most_difficult_block_id": most_difficult,
    }
