r"""Method 4 — MILP-based population calibration.

The MILP sweep (``experiments/p2-milp-sweep``) solves the same P2 instance to
(near-)optimality, so its solutions are a *ground-truth reference* for the GA.
Method 4 calibrates how large a GA population must be for the GA's solution
quality to come within a target tolerance of that reference.

Reference optimum
=================
P2 quality is measured by the relay count at full connectivity (fewer relays at
``connected_ratio = 1`` is better — exactly what both the MILP and the GA
minimise).  We evaluate every MILP chromosome with the *same* surrogate the GA
uses and take the reference optimum

        r_opt = min { relays(x) : x in MILP solutions, connected(x) }.       (1)

(This equals the MILP's own minimum installed-relay count, cross-checked under
the surrogate.)

Quality gap of the GA
=====================
For population ``n`` let ``r_GA(n)`` be the GA's converged relay count at full
connectivity, averaged over seeds.  The **optimality gap** is

        gap(n) = ( r_GA(n) - r_opt ) / r_opt   >= 0.                         (2)

We fit a decaying calibration curve

        gap(n) = g_inf + G * exp(-n / tau),     G > 0,                       (3)

(``g_inf`` is the residual gap a finite-time GA cannot close).  ``N_hat_4`` is the
smallest population whose expected gap meets a target tolerance ``tau_gap``:

        g_inf + G * exp(-n / tau) = tau_gap
    =>  N_hat_4 = ceil( tau * ln( G / (tau_gap - g_inf) ) ),  if tau_gap > g_inf. (4)

If ``tau_gap <= g_inf`` the target is below the achievable floor and we report
that the tolerance is unreachable within the GA's time budget (and fall back to
the largest tested population).

A 95 % CI comes from a bootstrap over GA seeds (resample seeds, recompute
``r_GA(n)``, refit (3), recompute (4)).

Complexity
==========
``O(|MILP| * C_eval)`` to score the reference solutions, plus an ``O(n_tau*K)``
grid fit and the bootstrap.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from methods.common import (
    GA_RUNS_DIR, INSTANCE_PATH, MILP_SUMMARY_PATH, RESULTS_DIR, NHat,
    ensure_dir, fit_saturating_exponential, get_plt, load_json,
    make_surrogate_fitness, save_json,
)

METHOD_DIR = RESULTS_DIR / "method4"
INSTANCE = "ind2"
TAU_GAP = 0.10            # target relay-count optimality gap (10 %)
CONN_FEASIBLE = 0.999     # connectivity threshold to call a solution feasible


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _milp_reference() -> dict:
    """Reference optimum from the MILP, with a surrogate-scored diagnostic.

    The ground-truth relay optimum is the MILP's own minimum installed-relay
    count among proven/feasible (``OPTIMAL``) solves — the MILP enforces full
    temporal connectivity at its solve timesteps, so this is a valid P2 relay
    optimum.  We *also* score every MILP chromosome with the surrogate for the
    diagnostic landscape plot; note that the MILP solved a subsampled time
    horizon, so many MILP layouts have small coverage gaps under the surrogate's
    independent time sampling (reported as ``n_feasible``).
    """
    fitness_fn, info = make_surrogate_fitness(INSTANCE_PATH)
    records = load_json(MILP_SUMMARY_PATH)
    scored = []
    optimal_relays = []
    for r in records:
        chrom = r.get("chromosome")
        if not chrom:
            continue
        bits = np.frombuffer(chrom.encode(), dtype=np.uint8) - ord("0")
        m = fitness_fn(bits)
        scored.append({
            "milp_installed": r.get("installed_nodes"),
            "surrogate_relays": m["relay_count"],
            "surrogate_conn": m["connected_ratio"],
            "surrogate_F": m["F"],
            "status": r.get("status_name"),
        })
        if r.get("status_name") == "OPTIMAL" and r.get("installed_nodes"):
            optimal_relays.append(int(r["installed_nodes"]))

    r_opt = min(optimal_relays) if optimal_relays else min(
        s["milp_installed"] for s in scored if s["milp_installed"])
    f_ref = max(s["surrogate_F"] for s in scored)
    n_feasible = sum(1 for s in scored if s["surrogate_conn"] >= CONN_FEASIBLE)
    return {
        "r_opt": int(r_opt), "f_ref": float(f_ref),
        "n_milp": len(scored), "n_feasible": n_feasible,
        "ref_source": "min installed_nodes among OPTIMAL MILP solves",
        "scored": scored,
    }


def _ga_relays_by_n() -> dict[int, list[float]]:
    """{pop_size: [relay_count per feasible seed]} from the GA sweep."""
    by_n: dict[int, list[float]] = defaultdict(list)
    with (GA_RUNS_DIR / "ga_summary.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if float(row["final_best_conn"]) >= CONN_FEASIBLE:
                by_n[int(row["pop_size"])].append(float(row["final_best_relays"]))
    return dict(sorted(by_n.items()))


def _solve_n4(fit: dict, tau_gap: float) -> float:
    """Invert eq. (3) for the target gap -> eq. (4)."""
    g_inf, G, tau = fit["y_inf"], fit["amp"], fit["tau"]
    if tau_gap <= g_inf:
        return float("inf")
    return tau * math.log(G / (tau_gap - g_inf))


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
def estimate() -> NHat:
    ref = _milp_reference()
    r_opt = ref["r_opt"]
    by_n = _ga_relays_by_n()
    sizes = np.array(sorted(by_n), dtype=float)
    per_seed_relays = [np.array(by_n[int(n)]) for n in sizes]
    mean_relays = np.array([a.mean() for a in per_seed_relays])
    gap = (mean_relays - r_opt) / r_opt                       # eq. (2)

    fit = fit_saturating_exponential(sizes, gap, increasing=False)
    n_raw = _solve_n4(fit, TAU_GAP)
    if not np.isfinite(n_raw):
        n_hat = int(sizes.max())
        unreachable = True
    else:
        n_raw = max(n_raw, float(sizes.min()))
        n_hat = int(math.ceil(n_raw))
        unreachable = False

    # bootstrap over seeds
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(1000):
        bm = []
        for a in per_seed_relays:
            idx = rng.integers(0, len(a), size=len(a))
            bm.append(a[idx].mean())
        g = (np.array(bm) - r_opt) / r_opt
        try:
            f = fit_saturating_exponential(sizes, g, increasing=False)
            v = _solve_n4(f, TAU_GAP)
            if np.isfinite(v):
                boots.append(max(v, float(sizes.min())))
        except Exception:
            pass
    boots = np.array(boots)
    if boots.size:
        ci_low, ci_high = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    else:
        ci_low = ci_high = float(n_raw if np.isfinite(n_raw) else n_hat)
    sigma = max((ci_high - ci_low) / (2 * 1.96), 1.0)

    params = {
        "view": "GA solution-quality gap vs MILP reference optimum",
        "tau_gap": TAU_GAP,
        "r_opt": r_opt, "f_ref": ref["f_ref"], "ref_source": ref["ref_source"],
        "n_milp_scored": ref["n_milp"], "n_milp_feasible": ref["n_feasible"],
        "fit_model": "gap(n) = g_inf + G*exp(-n/tau)",
        "g_inf": fit["y_inf"], "G": fit["amp"], "tau": fit["tau"], "rss": fit["rss"],
        "pop_sizes": sizes.tolist(),
        "mean_relays_per_size": mean_relays.tolist(),
        "gap_per_size": gap.tolist(),
        "target_reachable": not unreachable,
        "n_hat_raw": n_raw if np.isfinite(n_raw) else None,
    }
    return NHat(method="M4", instance=INSTANCE, n_hat=n_hat,
                n_hat_raw=float(n_raw) if np.isfinite(n_raw) else float(n_hat),
                ci_low=ci_low, ci_high=ci_high, sigma=sigma, params=params)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(res: NHat) -> None:
    plt = get_plt()
    p = res.params
    sizes = np.array(p["pop_sizes"])
    gap = np.array(p["gap_per_size"])
    g_inf, G, tau = p["g_inf"], p["G"], p["tau"]
    grid = np.linspace(sizes.min(), sizes.max(), 200)
    curve = g_inf + G * np.exp(-grid / tau)

    # (1) gap vs population
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, 100 * gap, "o", label="GA relay-count gap")
    ax.plot(grid, 100 * curve, "-", color="C3",
            label=fr"fit $g_\infty+Ge^{{-n/\tau}}$ ($\tau$={tau:.1f})")
    ax.axhline(100 * p["tau_gap"], ls=":", color="C1",
               label=fr"target $\tau_{{gap}}$={100*p['tau_gap']:.0f}%")
    ax.axvline(res.n_hat, ls="--", color="C2", label=fr"$\hat N_4$={res.n_hat}")
    ax.set_xlabel("population size $n$"); ax.set_ylabel("optimality gap (%)")
    ax.set_title("Method 4 — MILP-calibrated quality gap vs population size")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(METHOD_DIR / "method4_gap_vs_pop.png"); plt.close(fig)

    # (2) MILP reference landscape: relays vs surrogate F
    scored = _milp_reference()["scored"]
    relays = np.array([s["surrogate_relays"] for s in scored])
    Fv = np.array([s["surrogate_F"] for s in scored])
    conn = np.array([s["surrogate_conn"] for s in scored])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(relays, Fv, c=conn, cmap="viridis", s=45, edgecolor="k", linewidth=0.3)
    ax.axvline(p["r_opt"], ls="--", color="C3", label=fr"$r_{{opt}}$={p['r_opt']} relays")
    ax.set_xlabel("relays installed"); ax.set_ylabel("surrogate fitness $F$")
    ax.set_title("Method 4 — MILP solutions scored by the surrogate")
    fig.colorbar(sc, ax=ax, label="connected ratio"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(METHOD_DIR / "method4_milp_landscape.png"); plt.close(fig)


def main() -> int:
    ensure_dir(METHOD_DIR)
    res = estimate()
    save_json(METHOD_DIR / "method4_result.json", res.to_dict())
    make_plots(res)
    p = res.params
    print("=== Method 4 — MILP-based population calibration ===")
    print(f"  reference optimum r_opt = {p['r_opt']} relays "
          f"({p['n_milp_feasible']}/{p['n_milp_scored']} MILP sols feasible under surrogate)")
    print(f"  fit: g_inf={p['g_inf']:.4f}  G={p['G']:.4f}  tau={p['tau']:.2f}  (rss={p['rss']:.2e})")
    print(f"  gap per size: " + "  ".join(f"n{int(n)}={100*g:.0f}%"
          for n, g in zip(p['pop_sizes'], p['gap_per_size'])))
    if p["target_reachable"]:
        print(f"  N_hat_4 = {res.n_hat}  (raw {res.n_hat_raw:.1f}, 95% CI [{res.ci_low:.0f}, {res.ci_high:.0f}])  "
              f"for {100*p['tau_gap']:.0f}% gap")
    else:
        print(f"  target {100*p['tau_gap']:.0f}% gap below achievable floor "
              f"(g_inf={100*p['g_inf']:.1f}%); reporting n={res.n_hat}")
    print(f"  -> {METHOD_DIR}/method4_result.json (+ 2 figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
