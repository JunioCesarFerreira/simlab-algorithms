"""Shared infrastructure for the four population-estimation methods.

This module centralises everything the four method scripts and the combined
estimator need:

* ``sys.path`` wiring so the in-tree modules import cleanly regardless of the
  current working directory:
    - ``p2_population_estimator``  (lives under ``pop-estimator/``)
    - ``adjacency_builder`` / ``path_builder``  (repo root)
* canonical paths (repo root, the fixed ``ind2`` instance, the ``results/``
  tree);
* a surrogate-fitness factory used by the genetic algorithm and by Method 4;
* the :class:`NHat` record returned by every method;
* small, dependency-light numerical helpers (a saturating-exponential fit and a
  bootstrap confidence interval) implemented with **NumPy only** — no SciPy or
  scikit-learn required, which keeps the whole pipeline on a single interpreter.

All methods are evaluated on a single common instance (``ind2``) so that the
four estimates N̂₁..N̂₄ and the combined estimate are directly comparable.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Paths and import wiring
# ---------------------------------------------------------------------------
REPO_ROOT         = Path(__file__).resolve().parent.parent
POP_ESTIMATOR_DIR = REPO_ROOT / "pop-estimator"
INSTANCE_PATH     = POP_ESTIMATOR_DIR / "examples" / "ind2.json"
MILP_SUMMARY_PATH = REPO_ROOT / "experiments" / "p2-milp-sweep" / "results" / "milp_run_summary.json"
RESULTS_DIR       = REPO_ROOT / "results"
GA_RUNS_DIR       = RESULTS_DIR / "ga_runs"

for _p in (str(REPO_ROOT), str(POP_ESTIMATOR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Result record shared by all methods
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class NHat:
    """A population-size estimate produced by one method.

    Attributes
    ----------
    method      : short method id ("M1".."M4").
    instance    : instance name the estimate was computed on.
    n_hat       : the recommended (integer, ceil) population size.
    n_hat_raw   : the unrounded estimate.
    ci_low,
    ci_high     : a confidence/uncertainty interval on ``n_hat_raw``.
    sigma       : a 1-σ uncertainty used by the combined estimator
                  (inverse-variance weighting and Bayesian fusion).
    params      : method-specific parameters/diagnostics (JSON-serialisable).
    """

    method: str
    instance: str
    n_hat: int
    n_hat_raw: float
    ci_low: float
    ci_high: float
    sigma: float
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=_json_default)
    return p


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Matplotlib (headless) helper
# ---------------------------------------------------------------------------
def get_plt():
    """Return a headless-configured ``matplotlib.pyplot`` module."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    return plt


# ---------------------------------------------------------------------------
# Surrogate fitness factory (P2 instance -> callable on a binary chromosome)
# ---------------------------------------------------------------------------
def make_surrogate_fitness(
    instance_path: str | Path = INSTANCE_PATH,
    weights: dict[str, float] | None = None,
) -> tuple[Callable[[np.ndarray], dict[str, float]], dict[str, Any]]:
    """Build a deterministic P2 fitness function over binary chromosomes.

    Returns ``(fitness_fn, info)`` where ``fitness_fn(bits) -> metrics`` with
    keys ``F`` (scalar objective, higher is better), ``relay_count``,
    ``connected_ratio``, ``mean_hop_count``.  ``info`` carries ``n_bits``,
    ``instance_name`` and the candidate count ``N``.

    The fitness is the SimLab surrogate evaluator (``connected_ratio``,
    ``relay_count``, hop/distance/redundancy penalties) — the same structural
    score the MILP minimises, which makes the GA and MILP solutions directly
    comparable.
    """
    from p2_population_estimator.io import load_instance
    from p2_population_estimator.models import FullSolution, ScalarizationWeights
    from p2_population_estimator.evaluation.surrogate import SurrogateEvaluator

    inst = load_instance(str(instance_path))
    problem = inst.problem
    w = ScalarizationWeights(**weights) if weights else ScalarizationWeights()
    evaluator = SurrogateEvaluator(problem, w)
    n_bits = len(problem.candidates)

    def fitness_fn(bits: np.ndarray) -> dict[str, float]:
        sol = FullSolution(solution_id="x", bits=[int(b) for b in bits])
        res = evaluator.evaluate(sol, [0])
        m = res.aggregated.mean
        return {
            "F": float(res.F),
            "relay_count": int(m.relay_count or 0),
            "connected_ratio": float(m.connected_ratio or 0.0),
            "mean_hop_count": float(m.mean_hop_count or 0.0),
        }

    info = {
        "n_bits": n_bits,
        "N": n_bits,
        "instance_name": problem.name,
        "radius_of_reach": problem.radius_of_reach,
        "n_mobiles": len(problem.mobile_nodes),
        "weights": asdict(w),
    }
    return fitness_fn, info


# ---------------------------------------------------------------------------
# NumPy-only numerical helpers
# ---------------------------------------------------------------------------
def fit_saturating_exponential(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_tau: int = 400,
    increasing: bool = True,
) -> dict[str, float]:
    """Fit ``y ≈ c0 + c1 · exp(-x / tau)`` by a 1-D grid search over ``tau``.

    For each candidate ``tau`` the model is **linear** in ``(c0, c1)``, so the
    optimal ``(c0, c1)`` is the closed-form least-squares solution on the basis
    ``[1, exp(-x/tau)]``.  We scan ``tau`` on a log grid spanning the data range
    and keep the ``(tau, c0, c1)`` with the smallest residual sum of squares.

    Returns a dict with ``y_inf`` (= c0, the saturation level), ``amp`` (= |c1|),
    ``tau``, ``rss`` and the raw ``c1`` sign.

    Notes
    -----
    * ``increasing=True``  models a saturating *growth* curve (used for
      best-fitness vs population size): the asymptote ``y_inf`` is approached
      from below, so ``c1 < 0``.
    * ``increasing=False`` models a saturating *decay* curve (used for the
      quality gap vs population size): ``c1 > 0`` and ``y_inf`` is the residual
      floor.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        raise ValueError("need at least 3 points for a 3-parameter fit")

    span = max(x.max() - x.min(), 1e-9)
    taus = np.logspace(np.log10(span / 50.0), np.log10(span * 5.0), n_tau)

    best = {"rss": np.inf, "tau": float(taus[0]), "c0": 0.0, "c1": 0.0}
    for tau in taus:
        basis = np.column_stack([np.ones_like(x), np.exp(-x / tau)])
        coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
        resid = y - basis @ coef
        rss = float(resid @ resid)
        if rss < best["rss"]:
            best = {"rss": rss, "tau": float(tau), "c0": float(coef[0]), "c1": float(coef[1])}

    return {
        "y_inf": best["c0"],
        "amp": abs(best["c1"]),
        "c1": best["c1"],
        "tau": best["tau"],
        "rss": best["rss"],
    }


def bootstrap_ci(
    estimator: Callable[[np.ndarray], float],
    data: np.ndarray,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for ``estimator`` applied to rows of ``data``.

    Returns ``(point_estimate, ci_low, ci_high)``.  ``data`` is resampled along
    axis 0 with replacement ``n_boot`` times.
    """
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    point = float(estimator(data))
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boots[b] = estimator(data[idx])
        except Exception:
            boots[b] = np.nan
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return point, point, point
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two equal-length binary vectors."""
    return int(np.count_nonzero(np.asarray(a) != np.asarray(b)))
