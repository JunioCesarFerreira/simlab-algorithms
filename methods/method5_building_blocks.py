r"""Method 5 — Gambler-ruin population sizing with *semantic* P2 building blocks.

Motivation
==========
The original proposal (``estimando-população-p2.md``) sized the GA population with
the Harik *gambler-ruin* closed form, applied **per building block** ``i``:

        n_i_hat = -ln(alpha) * 2^{k_i - 1} * sigma_BB_i * sqrt(2 m) / d_i,      (GR)
        N_hat   = max_i  ceil( n_i_hat ),

where ``m`` is the number of building blocks, ``k_i`` the block size, ``d_i`` the
signal (mean fitness advantage of the correct block instance over a deceptive
competitor) and ``sigma_BB_i`` the block-fitness noise.  The document left one
question open: *what is a building block in P2?*

The existing ``pop-estimator`` answered it with an **arbitrary spatial grid** that
forces all ``N`` candidates into 8 blocks, producing blocks as large as ``k=10``.
Because (GR) grows like ``2^{k-1}``, those oversized blocks inflate the estimate to
``N_hat = 2839``.  This module gives a **semantic** answer instead and recomputes
(GR) faithfully, so the result is directly comparable to Methods 1--4.

Semantic building block for P2 (order-1)
========================================
The decisive observation is that (GR) is dominated by the block-order term
``2^{k_i-1}``: it doubles for every extra bit in a block.  The earlier grid
partition forced ``k`` up to 10 (``2^9 = 512``) and a *connectivity-component*
partition is even worse (a 12-candidate redundancy region gives ``2^{11}``), so
both inflate the estimate to thousands.  But that high order is an **artefact**:
the ``2^{k-1}`` factor assumes a Trap-``k`` schema with a *single* correct setting
out of ``2^k``.  P2 has no such deception — a redundancy region admits *many*
acceptable coverings, so its effective order is **not** its candidate count.

The theoretically honest decomposition is therefore **order-1**: each
*indispensable relay* is its own building block — a single gene the GA must keep
ON, because removing it (with everything else random) demonstrably lowers fitness.
P2's difficulty is not one ``k``-bit trap but *many* scattered must-keep genes.

Construction
------------
1.  Critical candidates ``Crit`` = coverage-indispensable (Method 2,
    ``cov(u) >= theta T M``) OR routing-critical (Method 3, ``route(u) >= phi``).
2.  Each ``u in Crit`` is a candidate **order-1 block** ``Q_u = {u}`` with
      * ``H_u^*`` = bit ``u`` ON (correct instance, ``k_u = s_u = 1``);
      * ``H_u^L`` = bit ``u`` OFF (the gene is lost).
3.  Draw ``R`` complements ``x_{-u} ~ Bernoulli(rho)``; score both compositions and
    form ``Delta_u^{(r)} = F(H_u^*, x_{-u}) - F(H_u^L, x_{-u})``,
    ``d_u = mean_r Delta``, ``sigma_BB_u = std_r Delta``.
4.  A block is **binding** iff ``d_u > 0`` (turning ``u`` off really hurts).  Only
    binding blocks count, so substitutable relays (``d_u approx 0``) are filtered
    out automatically.  With ``k_u = 1`` the order term is ``2^{0} = 1`` and

        n_u = -ln(alpha) * sigma_BB_u * sqrt(2 m) / d_u,   m = #binding blocks,
        N_hat_5 = max_u ceil(n_u).

Block-order sensitivity (diagnostic)
------------------------------------
For comparison we also report the connectivity-component grouping (blocks =
connected components of ``Crit`` under reach ``R``) and its ``2^{k-1}`` blow-up,
to make explicit how fragile (GR) is to the block-order assumption.

Complexity
==========
Criticality reuses Methods 2/3; the estimate costs ``|Crit| * R`` surrogate
evaluations.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np

import adjacency_builder as ab
import path_builder as pb
from methods.common import (
    INSTANCE_PATH, RESULTS_DIR, NHat, ensure_dir, get_plt,
    make_surrogate_fitness, save_json,
)
from p2_population_estimator.estimator import estimate_bernoulli, estimate_uniform
from p2_population_estimator.statistics import d_hat, sigma_BB_hat

METHOD_DIR = RESULTS_DIR / "method5"
INSTANCE = "ind2"

ALPHA = 0.05
RHO = 0.5                      # Bernoulli(0.5) initialisation, as in the GA
THETA_FRACTION = 0.10          # coverage-indispensability threshold (Method 2)
PHI = 0.05                     # routing-criticality threshold (Method 3)
NUM_COMPLEMENTS = 60           # R random complements per block
SEED = 42


# ---------------------------------------------------------------------------
# Criticality (reuse Methods 2 and 3)
# ---------------------------------------------------------------------------
def _criticality(inst):
    """Return (cov, route, T, M, N, cand_xy, R) on the candidate index space 0..N-1."""
    adj = ab.build_from_instance(inst)
    rout = pb.build_from_instance(inst)
    layout = adj["layout"]
    c_idx = np.asarray(layout.candidate_indices)
    m_idx = np.asarray(layout.mobile_indices)
    T, M, N = adj["tensor"].shape[0], layout.M, layout.N

    A_total = adj["tensor"]                                   # (T,K,K)
    cov = A_total[:, c_idx][:, :, m_idx].sum(axis=(0, 2)).astype(np.int64)   # (N,)
    node_acc = rout["node_per_t"][:, c_idx].sum(axis=0)       # (N,)
    route = node_acc / (T * M)                               # (N,) in [0,1]

    cand_xy = np.asarray(inst.candidates, dtype=float)        # (N,2)
    R = float(inst.radius_of_reach)
    return cov, route, T, M, N, cand_xy, R


# ---------------------------------------------------------------------------
# Semantic building blocks: connected components of critical candidates
# ---------------------------------------------------------------------------
def _building_blocks(cov, route, T, M, cand_xy, R):
    """Return list of blocks; each is a sorted list of candidate indices (0..N-1)."""
    TM = T * M
    crit = np.where((cov >= THETA_FRACTION * TM) | (route >= PHI))[0]
    crit = list(map(int, crit))
    if not crit:
        return []
    # adjacency among critical candidates: edge iff within reach R
    cset = set(crit)
    nbr: dict[int, list[int]] = {u: [] for u in crit}
    for a in range(len(crit)):
        for b in range(a + 1, len(crit)):
            u, v = crit[a], crit[b]
            if np.linalg.norm(cand_xy[u] - cand_xy[v]) <= R:
                nbr[u].append(v)
                nbr[v].append(u)
    # connected components (BFS)
    seen: set[int] = set()
    blocks: list[list[int]] = []
    for u in crit:
        if u in seen:
            continue
        comp, q = [], deque([u])
        seen.add(u)
        while q:
            x = q.popleft()
            comp.append(x)
            for w in nbr[x]:
                if w not in seen:
                    seen.add(w)
                    q.append(w)
        blocks.append(sorted(comp))
    return blocks


def _block_signal(idx: int, N: int, fitness_fn, rng) -> tuple[float, float]:
    """Estimate (d_u, sigma_BB_u) for the order-1 block {u} via R complements.

    H_u^* = bit u ON, H_u^L = bit u OFF, complement ~ Bernoulli(rho).
    """
    F_star, F_local = [], []
    for _ in range(NUM_COMPLEMENTS):
        comp = (rng.random(N) < RHO).astype(int)         # x_{-u} ~ Bernoulli(rho)
        xs = comp.copy(); xs[idx] = 1
        xl = comp.copy(); xl[idx] = 0
        F_star.append(fitness_fn(xs)["F"])
        F_local.append(fitness_fn(xl)["F"])
    return d_hat(F_star, F_local), sigma_BB_hat(F_star, F_local)


# ---------------------------------------------------------------------------
# Estimation (order-1 building blocks)
# ---------------------------------------------------------------------------
def estimate() -> NHat:
    inst = ab.load_instance(str(INSTANCE_PATH))
    cov, route, T, M, N, cand_xy, R = _criticality(inst)
    TM = T * M
    crit = sorted(map(int, np.where((cov >= THETA_FRACTION * TM) | (route >= PHI))[0]))

    fitness_fn, info = make_surrogate_fitness(INSTANCE_PATH)
    assert info["N"] == N, f"candidate-count mismatch: surrogate {info['N']} vs builder {N}"
    rng = np.random.default_rng(SEED)

    # --- first pass: per-candidate signal/noise; keep only binding blocks ---
    # A gene is a binding order-1 block iff its mean advantage d_u is *significantly*
    # positive (one-sided 95%): d_u > z * SE, SE = sigma_u / sqrt(R).  Genes whose
    # signal is indistinguishable from zero are substitutable filler, not blocks —
    # this also stabilises max_i against the 1/d singularity at d -> 0.
    Z = 1.645
    se_scale = math.sqrt(NUM_COMPLEMENTS)
    measured = []
    for u in crit:
        d_u, sigma_u = _block_signal(u, N, fitness_fn, rng)
        se = sigma_u / se_scale
        binding = d_u > Z * se and d_u > 1e-6
        measured.append({"candidate": u, "d_i_hat": d_u, "sigma_BB_i_hat": sigma_u,
                         "se": se, "z_score": (d_u / se) if se > 0 else 0.0,
                         "cov": int(cov[u]), "route": float(route[u]),
                         "binding": binding})
    binding = [b for b in measured if b["binding"]]
    m_blocks = len(binding)                               # m = #order-1 blocks

    block_records, n_vals = [], []
    for bid, b in enumerate(binding):
        n_u = estimate_uniform(ALPHA, 1, m_blocks, b["sigma_BB_i_hat"], b["d_i_hat"])
        rec = {"block_id": bid, "candidate": b["candidate"], "k_i": 1, "s_star": 1,
               "d_i_hat": b["d_i_hat"], "sigma_BB_i_hat": b["sigma_BB_i_hat"],
               "cov": b["cov"], "route": b["route"],
               "snr": b["sigma_BB_i_hat"] / b["d_i_hat"],
               "n_i_uniform": math.ceil(n_u)}
        block_records.append(rec); n_vals.append(math.ceil(n_u))

    if n_vals:
        n_hat = int(max(n_vals))
        worst = block_records[int(np.argmax(n_vals))]["candidate"]
        arr = np.array(n_vals, dtype=float)
        ci_low, ci_high = float(arr.min()), float(n_hat)
        sigma = max(float(arr.std(ddof=1)) if arr.size > 1 else 1.0, 1.0)
    else:
        n_hat, worst = 0, None
        ci_low = ci_high = sigma = 1.0

    # --- diagnostic: connectivity-component grouping and its 2^{k-1} blow-up ---
    components = _building_blocks(cov, route, T, M, cand_xy, R)
    comp_diag = []
    for cid, comp in enumerate(components):
        # use the worst (max) binding signal inside the component as a proxy d,sigma
        inside = [b for b in measured if b["candidate"] in comp and b["binding"]]
        if inside:
            ref = max(inside, key=lambda z: z["sigma_BB_i_hat"] / max(z["d_i_hat"], 1e-9))
            k = len(comp)
            n_k = estimate_uniform(ALPHA, k, len(components),
                                   ref["sigma_BB_i_hat"], ref["d_i_hat"])
            comp_diag.append({"component_id": cid, "k": k,
                              "n_if_treated_as_order_k": math.ceil(n_k)})
        else:
            comp_diag.append({"component_id": cid, "k": len(comp),
                              "n_if_treated_as_order_k": None})

    params = {
        "view": "gambler-ruin closed form on ORDER-1 semantic P2 building blocks",
        "estimator": "N_hat_5 = max_u ceil(-ln(alpha) * sigma_BB_u * sqrt(2m) / d_u),  k_u=1",
        "block_definition": ("each indispensable candidate (cov>=theta*TM OR route>=phi) "
                             "is one order-1 building block (must-keep gene)"),
        "alpha": ALPHA, "rho": RHO, "theta_fraction": THETA_FRACTION, "phi": PHI,
        "num_complements": NUM_COMPLEMENTS, "N": N, "T": T, "M": M,
        "n_critical": len(crit), "m_blocks": m_blocks,
        "worst_candidate": worst,
        "n_per_block": n_vals,
        "blocks": block_records,
        "order_sensitivity": {
            "note": "same signal/noise, but treating each connectivity component as "
                    "one order-k Trap block -> 2^{k-1} blow-up (artefact)",
            "components": comp_diag,
        },
    }
    return NHat(method="M5", instance=INSTANCE, n_hat=n_hat, n_hat_raw=float(n_hat),
                ci_low=ci_low, ci_high=ci_high, sigma=sigma, params=params)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(res: NHat) -> None:
    plt = get_plt()
    p = res.params
    blocks = p["blocks"]

    # (1) per-block order-1 gambler-ruin estimate, ranked
    if blocks:
        ranked = sorted(blocks, key=lambda b: b["n_i_uniform"], reverse=True)
        labels = [str(b["candidate"]) for b in ranked]
        n_u = [b["n_i_uniform"] for b in ranked]
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        ax.bar(labels, n_u, color="C0")
        ax.axhline(res.n_hat, ls="--", color="C3",
                   label=fr"$\hat N_5=\max_u n_u={res.n_hat}$")
        ax.set_xlabel("indispensable candidate $u$ (order-1 block)")
        ax.set_ylabel(r"$n_u$ (gambler-ruin, $k_u{=}1$)")
        ax.set_title("Method 5 — per-gene order-1 gambler-ruin population bound")
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(METHOD_DIR / "method5_per_block.png"); plt.close(fig)

    # (2) spatial map: filler vs indispensable (order-1) genes, sized by n_u
    inst = ab.load_instance(str(INSTANCE_PATH))
    cand_xy = np.asarray(inst.candidates, dtype=float)
    sink = np.asarray(inst.sink, dtype=float)
    crit_idx = [b["candidate"] for b in blocks]
    n_map = {b["candidate"]: b["n_i_uniform"] for b in blocks}
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(cand_xy[:, 0], cand_xy[:, 1], s=25, color="lightgray",
               edgecolor="k", linewidth=0.2, label="filler candidates")
    if crit_idx:
        sizes = [30 + 4 * n_map[u] for u in crit_idx]
        sc = ax.scatter(cand_xy[crit_idx, 0], cand_xy[crit_idx, 1], s=sizes,
                        c=[n_map[u] for u in crit_idx], cmap="plasma",
                        edgecolor="k", linewidth=0.4, label="order-1 blocks")
        fig.colorbar(sc, ax=ax, label=r"$n_u$")
    ax.scatter([sink[0]], [sink[1]], marker="*", s=280, color="red",
               edgecolor="k", label="sink", zorder=5)
    ax.set_aspect("equal")
    ax.set_title("Method 5 — order-1 building blocks (indispensable genes)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(METHOD_DIR / "method5_blocks_spatial.png"); plt.close(fig)


def main() -> int:
    ensure_dir(METHOD_DIR)
    res = estimate()
    save_json(METHOD_DIR / "method5_result.json", res.to_dict())
    make_plots(res)
    p = res.params
    print("=== Method 5 — gambler-ruin on ORDER-1 semantic P2 building blocks ===")
    print(f"  critical candidates={p['n_critical']}  ->  m={p['m_blocks']} binding order-1 blocks")
    for b in sorted(p["blocks"], key=lambda z: z["n_i_uniform"], reverse=True):
        print(f"  gene {b['candidate']:>2}: d={b['d_i_hat']:.4f}  sigma={b['sigma_BB_i_hat']:.4f}  "
              f"snr={b['snr']:.2f}  -> n_u={b['n_i_uniform']}")
    print(f"  N_hat_5 = {res.n_hat}  (worst gene {p['worst_candidate']}, "
          f"range [{res.ci_low:.0f}, {res.ci_high:.0f}])")
    print("  -- order sensitivity (if components were Trap-k blocks) --")
    for c in p["order_sensitivity"]["components"]:
        print(f"     component {c['component_id']}: k={c['k']}  "
              f"n_if_order_k={c['n_if_treated_as_order_k']}")
    print(f"  -> {METHOD_DIR}/method5_result.json (+ 2 figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
