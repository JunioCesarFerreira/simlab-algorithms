"""Default constants and small helpers for configuration.

Heavy configuration is represented by :class:`models.ExperimentConfig` and
built in :mod:`cli`. This module holds project-wide constants and the small
numerical safeguards used by the estimator.
"""

from __future__ import annotations

from typing import Final

# Default SSH ports for the 6 Cooja containers
DEFAULT_SSH_PORTS: Final[tuple[int, ...]] = (2231, 2232, 2233, 2234, 2235, 2236)

# Numerical guard rails ------------------------------------------------------
# Floor for sigma_BB to avoid division-by-zero. Treated as "degenerate" but the
# estimator can still produce a (very small) value if requested.
SIGMA_FLOOR: Final[float] = 1e-12

# Floor for d_i_hat above which a block is considered "useful" (d_i > 0).
D_POSITIVE_EPS: Final[float] = 1e-12

# Default surrogate trajectory sampling
SURROGATE_TIME_SAMPLES: Final[int] = 24
SURROGATE_TIME_HORIZON: Final[float] = 1.0  # parametric t in [0, T]

# Default sigma for k-means when sklearn missing (informative error)
KMEANS_MISSING_HINT: Final[str] = (
    "scikit-learn is not installed. Install with `pip install scikit-learn` "
    "or pass --partition-method grid / radial_to_sink."
)

# Status codes for BlockComparisonResult ------------------------------------
STATUS_OK: Final[str] = "ok"
STATUS_NON_POSITIVE_D: Final[str] = "invalid_non_positive_d"
STATUS_DEGENERATE_VARIANCE: Final[str] = "degenerate_zero_variance"
STATUS_INSUFFICIENT_SAMPLES: Final[str] = "insufficient_samples"

# Disclaimer added to every result --------------------------------------------
RESULT_DISCLAIMER: Final[str] = (
    "n_hat is a heuristic-statistical ESTIMATE based on a gambler-ruin "
    "approximation over structurally-defined blocks, comparing a candidate "
    "block-optimum H_i* against a deceptive competitor H_i^L. The choice of "
    "H_i* and H_i^L is heuristic; the value of n_hat must not be interpreted "
    "as an absolute convergence guarantee for NSGA-III."
)
