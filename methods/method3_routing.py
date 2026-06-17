r"""Method 3 — Routing-matrix variant (shortest-path usage).

This ports and extends ``p3-path-blocks.ipynb`` / ``path_builder.py``.  It mirrors
Method 2 structurally but replaces the *geometric* adjacency matrix ``A(t)`` with
the *operational* routing matrix ``R(t)``: instead of asking "who is within
range of whom", it asks "who is actually used to carry a mobile's traffic back
to the sink".

A(t) vs R(t) — what each captures
=================================
* ``A(t)`` (Method 2): symmetric *reachability*.  ``A(t)[u,v]=1`` whenever two
  nodes are within range, whether or not any traffic uses that link.  It
  over-counts: a candidate can be "covered" yet never needed.
* ``R(t)`` (Method 3): directed *usage*.  We run one BFS from the sink in
  ``A(t)`` and keep only the edges/nodes lying on a chosen shortest mobile->sink
  path.  A candidate scores only when it is an actual relay hop.  ``R(t)`` is a
  sub-graph of ``A(t)``; ``route(u) <= cov(u)/(...)`` always, and the gap between
  the two diagnoses *redundant coverage* (high ``cov``, low ``route``) vs
  *routing bottlenecks* (moderate ``cov``, high ``route``).

Formulae
========
Per timestep ``t``: build ``A(t)`` (Method 2 eq. 1), BFS from the sink, trace
each mobile's shortest path.  Then

    node_count(t)[u] = #{ mobiles whose shortest path passes through u },
    node_accumulated[u] = sum_t node_count(t)[u]      in [0, T*M].          (1)

Routing score of a fixed candidate:

    route(u) = node_accumulated[u] / (T*M)            in [0, 1].            (2)

Critical relays at routing threshold ``phi``:

    R_phi = #{ u in Q : route(u) >= phi }.                                  (3)

Population estimator (analogous to Method 2 eq. 5, with the critical-routing set
in place of the indispensable-coverage set):

    N_hat_3 = ceil( |R_phi| * (1/alpha)^(1/k_bar) ),   k_bar = N / B_blocks. (4)

Temporal convergence is the first timestep at which the critical set
``|R_phi(t)|`` (from partial ``node_accumulated``) reaches its final size.

Uncertainty: ``phi`` is swept over ``[0.02, 0.20]``; the spread gives the
interval and 1-sigma.

Complexity
==========
One BFS per timestep: ``O(T*(K + E))`` with ``E`` the per-timestep edge count;
scoring and the threshold sweep are ``O(T*N)``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import adjacency_builder as ab
import path_builder as pb
from methods.common import (
    INSTANCE_PATH, RESULTS_DIR, NHat, ensure_dir, get_plt, save_json,
)

METHOD_DIR = RESULTS_DIR / "method3"
INSTANCE = "ind2"
ALPHA = 0.05
N_BLOCKS = 8
PHI = 0.05                                  # headline routing threshold
PHI_SWEEP = (0.02, 0.05, 0.075, 0.10, 0.15, 0.20)


def _build():
    inst = ab.load_instance(str(INSTANCE_PATH))
    res = pb.build_from_instance(inst)
    return inst, res


def _route(res) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return (route over candidates, partial node_acc for candidates (T,N), T, M)."""
    layout = res["layout"]
    c_idx = np.asarray(layout.candidate_indices)
    node_per_t = res["node_per_t"]              # (T, K)
    T, M = node_per_t.shape[0], layout.M
    cand_per_t = node_per_t[:, c_idx]           # (T, N)
    node_acc = cand_per_t.sum(axis=0)           # (N,)
    route = node_acc / (T * M)                  # (N,) in [0,1]   == eq. (2)
    return route, cand_per_t, T, M


def _n_hat(R_phi: int, k_bar: float, alpha: float) -> float:
    return R_phi * (1.0 / alpha) ** (1.0 / k_bar)


def estimate() -> NHat:
    inst, res = _build()
    layout = res["layout"]
    N = layout.N
    route, cand_per_t, T, M = _route(res)
    k_bar = N / N_BLOCKS

    R_phi = int((route >= PHI).sum())
    n_raw = _n_hat(R_phi, k_bar, ALPHA)
    n_hat = int(math.ceil(n_raw))

    # temporal convergence of the critical-routing set (monotone, first-reach)
    partial = np.cumsum(cand_per_t, axis=0)             # (T, N)
    route_t = partial / (T * M)
    size_t = (route_t >= PHI).sum(axis=1)               # |R_phi(t)|
    final_size = int(size_t[-1])
    t_star = int(np.argmax(size_t >= final_size)) + 1 if final_size > 0 else T
    conv_fraction = t_star / T

    # phi sweep -> interval / sigma
    sweep = []
    for ph in PHI_SWEEP:
        r = int((route >= ph).sum())
        sweep.append({"phi": ph, "R_phi": r, "n_hat": math.ceil(_n_hat(r, k_bar, ALPHA))})
    n_vals = np.array([s["n_hat"] for s in sweep], dtype=float)
    ci_low, ci_high = float(n_vals.min()), float(n_vals.max())
    sigma = max(float(n_vals.std(ddof=1)) if n_vals.size > 1 else 1.0, 1.0)

    n_active = res["n_active_per_t"]
    params = {
        "view": "operational routing usage (shortest paths)",
        "alpha": ALPHA, "N": N, "T": T, "M": M, "TM": T * M,
        "k_bar": k_bar, "n_blocks": N_BLOCKS, "phi": PHI,
        "R_phi": R_phi,
        "estimator": "N_hat_3 = |R_phi| * (1/alpha)^(1/k_bar)",
        "route_stats": {"max": float(route.max()), "mean": float(route.mean()),
                         "n_used": int((route > 0).sum())},
        "convergence": {"t_star": t_star, "T": T, "conv_fraction": conv_fraction,
                         "R_phi_final": int(size_t[-1])},
        "connectivity": {"full_connectivity_pct": float(100.0 * (n_active == M).mean()),
                          "mean_active": float(n_active.mean())},
        "phi_sweep": sweep,
        "n_hat_raw": n_raw,
    }
    return NHat(method="M3", instance=INSTANCE, n_hat=n_hat, n_hat_raw=n_raw,
                ci_low=ci_low, ci_high=ci_high, sigma=sigma, params=params)


# ---------------------------------------------------------------------------
# Plots (port of the notebook figures + convergence + connectivity)
# ---------------------------------------------------------------------------
def make_plots(res_nhat: NHat) -> None:
    plt = get_plt()
    inst, res = _build()
    layout = res["layout"]
    edge_acc = res["edge_accumulated"]
    route, cand_per_t, T, M = _route(res)
    N = layout.N
    n_active = res["n_active_per_t"]

    # (1) accumulated edge-usage heatmap
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(edge_acc, cmap="magma", aspect="auto")
    ax.set_title(r"Method 3 — accumulated edge usage $\sum_t R(t)$")
    ax.set_xlabel("node index"); ax.set_ylabel("node index")
    fig.colorbar(im, ax=ax, label="mobile-timesteps using edge")
    fig.tight_layout(); fig.savefig(METHOD_DIR / "method3_edge_heatmap.png"); plt.close(fig)

    # (2) sorted routing-score bar chart
    order = np.argsort(route)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.bar(np.arange(N), route[order], color="C1")
    ax.axhline(res_nhat.params["phi"], ls="--", color="C3",
               label=fr"$\phi$={res_nhat.params['phi']}")
    ax.set_xlabel("candidate (sorted by routing score)")
    ax.set_ylabel(r"$\mathrm{route}(u)$")
    ax.set_title("Method 3 — per-candidate routing score")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(METHOD_DIR / "method3_route_bar.png"); plt.close(fig)

    # (3) temporal convergence of |R_phi(t)| + connectivity
    partial = np.cumsum(cand_per_t, axis=0)
    size_t = (partial / (T * M) >= res_nhat.params["phi"]).sum(axis=1)
    t_star = res_nhat.params["convergence"]["t_star"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, T + 1), size_t, "-", color="C2", label=r"$|R_\phi(t)|$")
    ax.axvline(t_star, ls="--", color="C3", label=fr"$t^*$={t_star} ({t_star/T:.0%} of T)")
    ax.set_xlabel("timestep $t$"); ax.set_ylabel(r"critical relays $|R_\phi(t)|$")
    ax.set_title("Method 3 — temporal convergence of the critical-routing set")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(METHOD_DIR / "method3_convergence.png"); plt.close(fig)

    # (4) spatial heat-graph of candidates coloured by routing score
    cand_xy = inst.candidates
    sink = np.asarray(inst.sink, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5.6))
    sc = ax.scatter(cand_xy[:, 0], cand_xy[:, 1], c=route, cmap="plasma",
                    s=60, edgecolor="k", linewidth=0.3)
    ax.scatter([sink[0]], [sink[1]], marker="*", s=260, color="red",
               edgecolor="k", label="sink", zorder=5)
    ax.set_aspect("equal")
    ax.set_title("Method 3 — spatial routing heat-graph")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, label=r"$\mathrm{route}(u)$")
    fig.tight_layout(); fig.savefig(METHOD_DIR / "method3_spatial.png"); plt.close(fig)


def main() -> int:
    ensure_dir(METHOD_DIR)
    res = estimate()
    save_json(METHOD_DIR / "method3_result.json", res.to_dict())
    make_plots(res)
    p = res.params
    print("=== Method 3 — routing-matrix variant ===")
    print(f"  T={p['T']}  M={p['M']}  N={p['N']}  k_bar={p['k_bar']:.1f}")
    print(f"  R_phi (phi={p['phi']}) = {p['R_phi']}  -> N_hat_3 = {res.n_hat}  (raw {res.n_hat_raw:.1f})")
    c = p["convergence"]
    print(f"  routing convergence: t*={c['t_star']}/{c['T']} ({c['conv_fraction']:.0%})")
    print(f"  full connectivity for {p['connectivity']['full_connectivity_pct']:.0f}% of timesteps")
    print(f"  phi sweep N_hat range: [{res.ci_low:.0f}, {res.ci_high:.0f}]  sigma={res.sigma:.1f}")
    print(f"  -> {METHOD_DIR}/method3_result.json (+ 4 figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
