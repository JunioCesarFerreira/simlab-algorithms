r"""Method 2 — Adjacency-matrix temporal dynamics.

This ports and extends ``p3-building-blocks.ipynb`` / ``adjacency_builder.py``.
All of the notebook's original computations (the per-timestep adjacency tensor
``A(t)``, the accumulated co-occurrence matrix ``A_total``, the per-candidate
coverage score, and the heatmap / bar-chart / spatial-graph figures) are
preserved; the additions are (i) a temporal-convergence stopping rule on the
accumulated matrix and (ii) the population estimator ``N_hat_2``.

Formulae
========
Canonical node layout ``K = 1 + N + M`` = ``[sink, candidates, mobiles]``.

Per-timestep adjacency (geometric reachability):

        A(t)[u,v] = 1   iff   u != v  and  ||p_u(t) - p_v(t)|| <= R.        (1)

Accumulated co-occurrence:

        A_total[u,v] = sum_{t=1..T} A(t)[u,v]   in [0, T].                  (2)

Mobile-coverage score of a fixed candidate ``u`` (how many mobile-timesteps it
was in range of some mobile):

        cov(u) = sum_{m in mobiles} A_total[u,m]   in [0, T*M].             (3)

Indispensable candidates at coverage fraction ``theta``:

        C_theta = #{ u in Q : cov(u) >= theta * T * M }.                    (4)

Temporal convergence
--------------------
Let the *partial* coverage after ``t`` steps be
``cov_t(u) = sum_{tau<=t} sum_m A(tau)[u,m]`` and
``|C_theta(t)| = #{u : cov_t(u) >= theta*T*M}`` (using the final threshold).
``|C_theta(t)|`` is non-decreasing and saturates once the last indispensable
candidate has accumulated enough coverage.  We declare convergence at the first
``t*`` after which ``|C_theta(t)|`` does not change for ``k_stab`` consecutive
steps; ``t*/T`` is reported as the *coverage convergence fraction* (the share of
the trajectory needed to reveal the indispensable structure).

Population estimator
--------------------
Following the structural hypothesis (POPULATION_ESTIMATOR.md §3): the minimum
number of relays needed to cover the whole trajectory is lower-bounded by the
number of indispensable candidates; spreading those ``C_theta`` indispensable
genes across blocks of mean size ``k_bar = N / B_blocks`` means a single random
individual carries all of them with low probability, so

        N_hat_2 = ceil( C_theta * (1/alpha)^(1/k_bar) ).                    (5)

The factor ``(1/alpha)^(1/k_bar)`` is the per-critical-region multiplier needed
so that, across ``k_bar``-sized blocks, the probability of missing a required
gene falls to ``alpha``.

Uncertainty
-----------
``theta`` is swept over ``[0.05, 0.20]``; the spread of ``N_hat_2`` over that
range gives the reported interval and the 1-sigma used by the combined
estimator.

Complexity
==========
``A(t)`` and ``A_total`` cost ``O(T*K^2)``; coverage, the threshold sweep and the
convergence scan are ``O(T*N)``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import adjacency_builder as ab
from methods.common import (
    INSTANCE_PATH, RESULTS_DIR, NHat, ensure_dir, get_plt, save_json,
)

METHOD_DIR = RESULTS_DIR / "method2"
INSTANCE = "ind2"
ALPHA = 0.05
N_BLOCKS = 8                       # k_bar = N / N_BLOCKS  (matches the M1 baseline grid)
THETA_FRACTION = 0.10              # headline coverage threshold (fraction of T*M)
THETA_SWEEP = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20)
K_STAB = 5                         # consecutive stable steps for convergence


def _build():
    inst = ab.load_instance(str(INSTANCE_PATH))
    res = ab.build_from_instance(inst)
    return inst, res


def _coverage(res) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return (cov over candidates, per-timestep mobile-coverage (T,N), T, M)."""
    tensor = res["tensor"]                 # (T, K, K) uint8
    layout = res["layout"]
    c_idx = np.asarray(layout.candidate_indices)
    m_idx = np.asarray(layout.mobile_indices)
    T, M = tensor.shape[0], layout.M
    # per-timestep, per-candidate coverage of mobiles: (T, N)
    cov_t = tensor[:, c_idx][:, :, m_idx].sum(axis=2).astype(np.int64)
    cov = cov_t.sum(axis=0)                # (N,)  == eq. (3)
    return cov, cov_t, T, M


def _n_hat(C_theta: int, k_bar: float, alpha: float) -> float:
    return C_theta * (1.0 / alpha) ** (1.0 / k_bar)


def estimate() -> NHat:
    inst, res = _build()
    layout = res["layout"]
    N = layout.N
    cov, cov_t, T, M = _coverage(res)
    TM = T * M
    k_bar = N / N_BLOCKS

    # headline C_theta and estimate
    thr = THETA_FRACTION * TM
    C_theta = int((cov >= thr).sum())
    n_raw = _n_hat(C_theta, k_bar, ALPHA)
    n_hat = int(math.ceil(n_raw))

    # temporal convergence of the indispensable set.
    # |C_theta(t)| is non-decreasing (fixed threshold, cumulative coverage), so
    # the meaningful stabilisation is the first timestep it reaches its final
    # value and never changes afterwards.
    partial = np.cumsum(cov_t, axis=0)             # (T, N)
    size_t = (partial >= thr).sum(axis=1)          # (T,)  == |C_theta(t)|
    final_size = int(size_t[-1])
    if final_size > 0:
        t_star = int(np.argmax(size_t >= final_size)) + 1   # first t reaching final
    else:
        t_star = T
    conv_fraction = t_star / T

    # theta sweep -> interval / sigma
    sweep = []
    for th in THETA_SWEEP:
        c = int((cov >= th * TM).sum())
        sweep.append({"theta": th, "C_theta": c, "n_hat": math.ceil(_n_hat(c, k_bar, ALPHA))})
    n_vals = np.array([s["n_hat"] for s in sweep], dtype=float)
    ci_low, ci_high = float(n_vals.min()), float(n_vals.max())
    sigma = max(float(n_vals.std(ddof=1)) if n_vals.size > 1 else 1.0, 1.0)

    params = {
        "view": "geometric coverage of mobile trajectories (adjacency)",
        "alpha": ALPHA, "N": N, "T": T, "M": M, "TM": TM,
        "k_bar": k_bar, "n_blocks": N_BLOCKS,
        "theta_fraction": THETA_FRACTION, "coverage_threshold": thr,
        "C_theta": C_theta,
        "estimator": "N_hat_2 = C_theta * (1/alpha)^(1/k_bar)",
        "cov_stats": {"min": int(cov.min()), "max": int(cov.max()),
                       "mean": float(cov.mean()), "n_nonzero": int((cov > 0).sum())},
        "convergence": {"t_star": t_star, "T": T, "conv_fraction": conv_fraction,
                         "C_theta_final": int(size_t[-1])},
        "theta_sweep": sweep,
        "n_hat_raw": n_raw,
    }
    return NHat(method="M2", instance=INSTANCE, n_hat=n_hat, n_hat_raw=n_raw,
                ci_low=ci_low, ci_high=ci_high, sigma=sigma, params=params)


# ---------------------------------------------------------------------------
# Plots (port of the notebook figures + the convergence curve)
# ---------------------------------------------------------------------------
def make_plots(res_nhat: NHat) -> None:
    plt = get_plt()
    inst, res = _build()
    layout = res["layout"]
    A_total = res["accumulated"]
    cov, cov_t, T, M = _coverage(res)
    N = layout.N
    c_idx = np.asarray(layout.candidate_indices)
    TM = T * M
    thr = res_nhat.params["coverage_threshold"]

    # (1) accumulated co-occurrence heatmap
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(A_total, cmap="viridis", aspect="auto")
    ax.set_title(r"Method 2 — accumulated co-occurrence $A_{\mathrm{total}}$")
    ax.set_xlabel("node index"); ax.set_ylabel("node index")
    fig.colorbar(im, ax=ax, label=r"$\sum_t A(t)$")
    fig.tight_layout(); fig.savefig(METHOD_DIR / "method2_Atotal_heatmap.png"); plt.close(fig)

    # (2) sorted coverage bar chart
    order = np.argsort(cov)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.bar(np.arange(N), cov[order], color="C0")
    ax.axhline(thr, ls="--", color="C3", label=fr"$\theta\,T M$={thr:.0f}")
    ax.set_xlabel("candidate (sorted by coverage)")
    ax.set_ylabel(r"$\mathrm{cov}(u)$")
    ax.set_title("Method 2 — per-candidate mobile-coverage score")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(METHOD_DIR / "method2_coverage_bar.png"); plt.close(fig)

    # (3) temporal convergence of |C_theta(t)|
    partial = np.cumsum(cov_t, axis=0)
    size_t = (partial >= thr).sum(axis=1)
    t_star = res_nhat.params["convergence"]["t_star"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, T + 1), size_t, "-", color="C2")
    ax.axvline(t_star, ls="--", color="C3", label=fr"$t^*$={t_star} ({t_star/T:.0%} of T)")
    ax.axhline(res_nhat.params["C_theta"], ls=":", color="gray",
               label=fr"$C_\theta$={res_nhat.params['C_theta']}")
    ax.set_xlabel("timestep $t$")
    ax.set_ylabel(r"$|C_\theta(t)|$ indispensable candidates")
    ax.set_title("Method 2 — temporal convergence of the indispensable set")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(METHOD_DIR / "method2_convergence.png"); plt.close(fig)

    # (4) spatial heat-graph of candidates coloured by coverage
    cand_xy = inst.candidates                       # (N, 2)
    sink = np.asarray(inst.sink, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5.6))
    sc = ax.scatter(cand_xy[:, 0], cand_xy[:, 1], c=cov, cmap="plasma",
                    s=60, edgecolor="k", linewidth=0.3)
    ax.scatter([sink[0]], [sink[1]], marker="*", s=260, color="red",
               edgecolor="k", label="sink", zorder=5)
    ax.set_aspect("equal")
    ax.set_title("Method 2 — spatial coverage heat-graph")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, label=r"$\mathrm{cov}(u)$")
    fig.tight_layout(); fig.savefig(METHOD_DIR / "method2_spatial.png"); plt.close(fig)


def main() -> int:
    ensure_dir(METHOD_DIR)
    res = estimate()
    save_json(METHOD_DIR / "method2_result.json", res.to_dict())
    make_plots(res)
    p = res.params
    print("=== Method 2 — adjacency-matrix temporal dynamics ===")
    print(f"  T={p['T']}  M={p['M']}  N={p['N']}  T*M={p['TM']}  k_bar={p['k_bar']:.1f}")
    print(f"  C_theta (theta={p['theta_fraction']}) = {p['C_theta']}  "
          f"-> N_hat_2 = {res.n_hat}  (raw {res.n_hat_raw:.1f})")
    c = p["convergence"]
    print(f"  coverage convergence: t*={c['t_star']}/{c['T']} ({c['conv_fraction']:.0%})")
    print(f"  theta sweep N_hat range: [{res.ci_low:.0f}, {res.ci_high:.0f}]  sigma={res.sigma:.1f}")
    print(f"  -> {METHOD_DIR}/method2_result.json (+ 4 figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
