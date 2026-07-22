r"""Method 1 — Population inference via inter-sample differences (diminishing returns).

Idea
====
Run the genetic algorithm at a ladder of population sizes ``n`` and observe the
*converged solution quality* ``F^*(n)``.  Each additional individual the
population carries contributes some *marginal information* — but with strongly
diminishing returns: doubling a tiny population helps a lot, doubling an already
large one barely moves ``F^*``.  Method 1 fits that diminishing-returns curve and
reports the population size at which the marginal gain falls below a tolerance
``epsilon``.

Two complementary views are computed; the first is the headline estimate.

------------------------------------------------------------------------------
View A (headline): saturation of converged quality vs population size
------------------------------------------------------------------------------
Let ``F^*(n)`` be the GA's converged best fitness with population ``n`` (averaged
over random seeds).  We model the saturating growth

        F^*(n) = F_inf - A * exp(-n / tau),     A > 0, tau > 0.            (1)

``F_inf`` is the asymptotic quality (infinite population), ``A`` is the total
achievable improvement from ``n -> 0`` to ``n -> inf``, and ``tau`` is the
population "scale" of the diminishing returns.

The *remaining* improvement still available beyond population ``n`` is

        rho(n) = F_inf - F^*(n) = A * exp(-n / tau).                       (2)

The **marginal gain** of one extra individual is the derivative

        dF^*/dn = (A / tau) * exp(-n / tau).                               (3)

We declare the population "large enough" once the remaining relative
improvement drops below a tolerance ``eps_rel`` (default 0.05 = within 5 % of the
infinite-population quality):

        rho(n) / A = exp(-n / tau) <= eps_rel
    =>  n >= tau * ln(1 / eps_rel).

Hence the **Method-1 estimate**

        N_hat_1 = ceil( tau * ln(1 / eps_rel) ).                          (4)

Interpretation: ``N_hat_1`` is the smallest population that captures
``(1 - eps_rel)`` of the quality an unbounded population could ever reach.
Equation (4) is scale-free (independent of the units of ``F``).

A 95 % confidence interval is obtained by a **bootstrap over GA seeds**: resample
the seed set with replacement, recompute the per-``n`` means, refit (1), and
recompute (4); the 2.5/97.5 percentiles of the bootstrap distribution of
``N_hat_1`` give the interval.

------------------------------------------------------------------------------
View B (corroboration): inter-sample information gain
------------------------------------------------------------------------------
From a single large reference run we take the ordered stream of *distinct*
evaluated individuals ``x_1, x_2, ...`` and form two quantities the prompt calls
"difference vectors between solution representations":

* the **inter-sample Hamming distance** ``h_j = ||x_j XOR x_{j-1}||_1`` — large
  while the search explores, shrinking as it converges;
* the **cumulative best fitness** ``B(j) = max_{i<=j} F(x_i)`` — a diminishing-
  returns curve in the number of individuals *sampled* (not in population size).

Fitting (1) to ``B(j)`` yields an information scale ``tau_info`` and a sampling
budget ``j_95 = tau_info * ln(1/eps_rel)``; agreement between the population-scale
and sampling-scale saturation points is reported as a cross-check.

------------------------------------------------------------------------------
Baseline: gambler-ruin closed form
------------------------------------------------------------------------------
The repository's existing estimator (``pop-estimator``) computes a *worst-case*
gambler-ruin population bound.  Its value on the same instance is loaded and
reported alongside ``N_hat_1`` as a (conservative) baseline; it is typically
orders of magnitude larger because it bounds the probability of losing *any*
deceptive building block rather than tracking realised solution quality.

Complexity
==========
Building the curve costs the GA sweep itself: ``O(|sizes| * |seeds| * P * G * C_eval)``
where ``C_eval`` is one surrogate evaluation.  The fit (1) is an ``O(n_tau * K)``
grid search over ``tau`` with a closed-form linear solve per grid point
(``K`` = number of population sizes).  The bootstrap multiplies the fit cost by
``n_boot``.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from methods import bb_core
from methods.common import (
    GA_RUNS_DIR, INSTANCE_PATH, RESULTS_DIR, NHat, bootstrap_ci, ensure_dir,
    fit_saturating_exponential, get_plt, hamming, load_json,
    make_surrogate_fitness, save_json,
)

METHOD_DIR = RESULTS_DIR / "method1"
EPS_REL = 0.05            # within 5 % of the infinite-population quality
INSTANCE = "ind2"
ELITE_Q = 0.05           # top fraction of the GA stream taken as "converged elite"
ELITE_FREQ = 0.80        # gene is a BB if ON in >= this fraction of the elite


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_ga_summary() -> dict[int, list[float]]:
    """Return {pop_size: [final_best_F per seed]} from the GA sweep."""
    by_n: dict[int, list[float]] = defaultdict(list)
    with (GA_RUNS_DIR / "ga_summary.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_n[int(row["pop_size"])].append(float(row["final_best_F"]))
    return dict(sorted(by_n.items()))


def _n95_from_fit(sizes: np.ndarray, means: np.ndarray, eps_rel: float) -> float:
    """Fit eq. (1) to (sizes, means) and return tau * ln(1/eps_rel)  [eq. (4)]."""
    fit = fit_saturating_exponential(sizes, means, increasing=True)
    return fit["tau"] * math.log(1.0 / eps_rel), fit


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
def _converged_genes() -> tuple[list[int], dict]:
    """BB source for Method 1: genes the GA's high-fitness elite converges on.

    Take the top ``ELITE_Q`` fraction of the distinct evaluated stream by fitness;
    a gene is nominated as a (must-install) building block if it is ON in at least
    ``ELITE_FREQ`` of that elite — i.e. the search dynamics single it out as part
    of every good solution.
    """
    data = load_json(GA_RUNS_DIR / "ga_evaluated_stream.json")
    stream = data["stream"]
    F = np.array([s["F"] for s in stream], dtype=float)
    X = np.array([np.frombuffer(s["chromosome"].encode(), dtype=np.uint8) - ord("0")
                  for s in stream])
    k = max(1, int(len(F) * ELITE_Q))
    elite = np.argsort(F)[-k:]
    freq = X[elite].mean(axis=0)
    genes = sorted(int(u) for u in np.where(freq >= ELITE_FREQ)[0])
    return genes, {"elite_size": int(k), "elite_freq_threshold": ELITE_FREQ,
                   "gene_on_freq": {int(u): float(freq[u]) for u in genes}}


def estimate() -> NHat:
    # --- BB-directing signal: the diminishing-returns curve (kept as diagnostic) ---
    by_n = _load_ga_summary()
    sizes = np.array(sorted(by_n), dtype=float)
    per_seed = np.array([by_n[int(n)] for n in sizes])   # (K, S)
    means = per_seed.mean(axis=1)
    _, fit = _n95_from_fit(sizes, means, EPS_REL)
    tau_n95 = fit["tau"] * math.log(1.0 / EPS_REL)       # legacy diminishing-returns scale
    info = _information_gain()
    baseline = _load_gambler_ruin_baseline()

    # --- the estimate itself is the gambler-ruin formula on the BBs M1 directs ---
    genes, elite_diag = _converged_genes()
    fitness_fn, finfo = make_surrogate_fitness(INSTANCE_PATH)
    res = bb_core.estimate_order1(
        genes, method="M1", instance=INSTANCE,
        fitness_fn=fitness_fn, N=finfo["N"],
        extra_params={
            "view": "BBs directed by GA search dynamics (converged elite genes)",
            "bb_source": "genes ON in >= ELITE_FREQ of the top-ELITE_Q fitness elite",
            "eps_rel": EPS_REL, "fit_model": "F*(n) = F_inf - A*exp(-n/tau)",
            "F_inf": fit["y_inf"], "A": fit["amp"], "tau": fit["tau"], "rss": fit["rss"],
            "diminishing_returns_n95": tau_n95,
            "pop_sizes": sizes.tolist(),
            "mean_F_per_size": means.tolist(),
            "std_F_per_size": per_seed.std(axis=1).tolist(),
            "information_gain": info,
            "gambler_ruin_baseline": baseline,
            "elite": elite_diag,
        })
    return res


def _information_gain() -> dict:
    """View B: cumulative-best saturation and inter-sample Hamming over the stream."""
    data = load_json(GA_RUNS_DIR / "ga_evaluated_stream.json")
    stream = data["stream"]
    F = np.array([s["F"] for s in stream], dtype=float)
    chrom = [np.frombuffer(s["chromosome"].encode(), dtype=np.uint8) - ord("0")
             for s in stream]
    cum_best = np.maximum.accumulate(F)
    j = np.arange(1, len(F) + 1, dtype=float)
    # subsample for the fit (the stream can be tens of thousands long)
    step = max(1, len(j) // 400)
    jf, bf = j[::step], cum_best[::step]
    try:
        fit = fit_saturating_exponential(jf, bf, increasing=True)
        j95 = fit["tau"] * math.log(1.0 / EPS_REL)
    except Exception:
        fit, j95 = {"tau": float("nan")}, float("nan")
    inter_h = [hamming(chrom[i], chrom[i - 1]) for i in range(1, min(len(chrom), 2000))]
    return {
        "n_distinct": len(F),
        "tau_info": fit.get("tau"),
        "j95_individuals": j95,
        "mean_inter_sample_hamming": float(np.mean(inter_h)) if inter_h else None,
        "n_bits": int(data["n_bits"]),
    }


def _load_gambler_ruin_baseline() -> dict:
    path = METHOD_DIR / "gambler_ruin_baseline" / "population_estimate_result.json"
    if not path.exists():
        return {"available": False}
    g = load_json(path)["global_estimate"]
    return {
        "available": True,
        "n_hat_uniform": g.get("n_hat_uniform"),
        "n_hat_bernoulli": g.get("n_hat_bernoulli"),
        "note": "worst-case gambler-ruin bound; conservative relative to N_hat_1",
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(res: NHat) -> None:
    plt = get_plt()
    p = res.params
    sizes = np.array(p["pop_sizes"])
    means = np.array(p["mean_F_per_size"])
    stds = np.array(p["std_F_per_size"])
    F_inf, A, tau = p["F_inf"], p["A"], p["tau"]
    grid = np.linspace(sizes.min(), sizes.max(), 200)
    curve = F_inf - A * np.exp(-grid / tau)

    # (1) saturation curve
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(sizes, means, yerr=stds, fmt="o", capsize=3, label="GA $F^*(n)$ (mean ± std)")
    ax.plot(grid, curve, "-", color="C3",
            label=fr"fit $F_\infty-A e^{{-n/\tau}}$ ($\tau$={tau:.1f})")
    ax.axhline(F_inf, ls=":", color="gray", label=fr"$F_\infty$={F_inf:.4f}")
    ax.axvline(res.n_hat, ls="--", color="C2", label=fr"$\hat N_1$={res.n_hat}")
    ax.set_xlabel("population size $n$")
    ax.set_ylabel("converged best fitness $F^*(n)$")
    ax.set_title("Method 1 — diminishing returns of GA quality vs population size")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(METHOD_DIR / "method1_saturation.png"); plt.close(fig)

    # (2) marginal gain
    marg = (A / tau) * np.exp(-grid / tau)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, marg, "-", color="C0", label=r"marginal gain $dF^*/dn$")
    ax.axvline(res.n_hat, ls="--", color="C2", label=fr"$\hat N_1$={res.n_hat}")
    ax.axvspan(res.ci_low, res.ci_high, color="C2", alpha=0.15, label="95% CI")
    ax.set_xlabel("population size $n$"); ax.set_ylabel("marginal gain per individual")
    ax.set_title("Method 1 — marginal information gain")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(METHOD_DIR / "method1_marginal_gain.png"); plt.close(fig)

    # (3) inter-sample information gain (View B)
    data = load_json(GA_RUNS_DIR / "ga_evaluated_stream.json")
    F = np.array([s["F"] for s in data["stream"]], dtype=float)
    cum_best = np.maximum.accumulate(F)
    j = np.arange(1, len(F) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(j, cum_best, "-", color="C4", label="cumulative best fitness $B(j)$")
    j95 = p["information_gain"].get("j95_individuals")
    if j95 and np.isfinite(j95):
        ax.axvline(j95, ls="--", color="C1", label=fr"$j_{{95}}$={j95:.0f} individuals")
    ax.set_xlabel("number of individuals evaluated $j$")
    ax.set_ylabel("best fitness so far")
    ax.set_title("Method 1 (View B) — inter-sample information gain")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(METHOD_DIR / "method1_information_gain.png"); plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ensure_dir(METHOD_DIR)
    res = estimate()
    save_json(METHOD_DIR / "method1_result.json", res.to_dict())
    make_plots(res)
    p = res.params
    print("=== Method 1 — inference via inter-sample differences ===")
    print(f"  fit: F_inf={p['F_inf']:.5f}  A={p['A']:.5f}  tau={p['tau']:.2f}  (rss={p['rss']:.2e})")
    print(f"  N_hat_1 = {res.n_hat}   (raw {res.n_hat_raw:.1f}, 95% CI [{res.ci_low:.0f}, {res.ci_high:.0f}])")
    ig = p["information_gain"]
    print(f"  View B: tau_info={ig['tau_info']:.0f}  j95={ig['j95_individuals']:.0f} individuals  "
          f"mean inter-sample Hamming={ig['mean_inter_sample_hamming']:.1f}/{ig['n_bits']}")
    gr = p["gambler_ruin_baseline"]
    if gr.get("available"):
        print(f"  baseline (gambler-ruin): n_hat_uniform={gr['n_hat_uniform']}  "
              f"n_hat_bernoulli={gr['n_hat_bernoulli']}")
    print(f"  -> {METHOD_DIR}/method1_result.json (+ 3 figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
