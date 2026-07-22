r"""Shared gambler-ruin estimator on order-1 P2 building blocks.

This is the *single* population-size estimator used by every method.  Each method
1..5 only differs in **which genes it nominates as building blocks** (its
``gene_set``) and, optionally, in the correct allele ``H_u^*`` of each gene.  The
estimate itself is always the Harik gambler-ruin closed form applied per order-1
block:

    n_u = -ln(alpha) * 2^{k_u - 1} * sigma_BB_u * sqrt(2 m) / d_u,   k_u = 1,
    N_hat = max_u ceil(n_u),

with ``m`` the number of *binding* blocks and the signal/noise estimated from
inter-sample differences over random complements:

    Delta_u^{(r)} = F(H_u^*, x_{-u}^{(r)}) - F(H_u^L, x_{-u}^{(r)}),
    d_u = mean_r Delta,   sigma_BB_u = sd_r Delta,    x_{-u} ~ Bernoulli(rho).

``H_u^*`` is the gene's correct allele (default ON, "this relay must be installed")
and ``H_u^L`` its complement (gene lost).  A gene is *binding* iff its mean
advantage is significantly positive (one-sided 95%): ``d_u > z * sigma_u/sqrt(R)``.
Genes that fail the test are substitutable filler and are dropped (this also tames
the ``1/d`` singularity as ``d -> 0``).

Because ``k_u = 1`` the order factor is ``2^0 = 1``; the method that nominates the
gene set is the only thing that varies, so the four methods become four
*BB-direction strategies* feeding one formula.
"""

from __future__ import annotations

import math

import numpy as np

from methods.common import NHat
from p2_population_estimator.estimator import estimate_uniform
from p2_population_estimator.statistics import d_hat, sigma_BB_hat

ALPHA = 0.05
RHO = 0.5
NUM_COMPLEMENTS = 60
SEED = 42
SIGNIF_Z = 1.645          # one-sided 95% binding test


def _signal(idx: int, h_star: int, N: int, fitness_fn, rng) -> tuple[float, float]:
    """(d_u, sigma_BB_u) for block {u} via R complements; H* = h_star, H^L = 1-h_star."""
    F_star, F_local = [], []
    for _ in range(NUM_COMPLEMENTS):
        comp = (rng.random(N) < RHO).astype(int)
        xs = comp.copy(); xs[idx] = h_star
        xl = comp.copy(); xl[idx] = 1 - h_star
        F_star.append(fitness_fn(xs)["F"])
        F_local.append(fitness_fn(xl)["F"])
    return d_hat(F_star, F_local), sigma_BB_hat(F_star, F_local)


def estimate_order1(
    gene_set,
    *,
    method: str,
    instance: str,
    fitness_fn,
    N: int,
    h_star: dict[int, int] | None = None,
    extra_params: dict | None = None,
    alpha: float = ALPHA,
    num_complements: int = NUM_COMPLEMENTS,
    seed: int = SEED,
) -> NHat:
    """Apply the gambler-ruin formula to a method-nominated set of order-1 blocks.

    Parameters
    ----------
    gene_set : iterable of candidate indices nominated as building blocks.
    method   : method id ("M1".."M5").
    fitness_fn, N : surrogate evaluator and chromosome length.
    h_star   : optional {gene: correct allele in {0,1}} (default: all ON).
    extra_params : method-specific diagnostics merged into the result params.
    """
    global NUM_COMPLEMENTS
    NUM_COMPLEMENTS = num_complements
    genes = sorted(set(int(u) for u in gene_set))
    h_star = h_star or {}
    rng = np.random.default_rng(seed)
    se_scale = math.sqrt(num_complements)

    measured = []
    for u in genes:
        hs = int(h_star.get(u, 1))
        d_u, sigma_u = _signal(u, hs, N, fitness_fn, rng)
        se = sigma_u / se_scale
        binding = d_u > SIGNIF_Z * se and d_u > 1e-6
        measured.append({"candidate": u, "h_star": hs, "d_i_hat": d_u,
                         "sigma_BB_i_hat": sigma_u, "se": se,
                         "z_score": (d_u / se) if se > 0 else 0.0,
                         "binding": binding})
    binding = [b for b in measured if b["binding"]]
    m_blocks = len(binding)

    blocks, n_vals = [], []
    for bid, b in enumerate(binding):
        n_u = estimate_uniform(alpha, 1, m_blocks, b["sigma_BB_i_hat"], b["d_i_hat"])
        blocks.append({"block_id": bid, "candidate": b["candidate"], "k_i": 1,
                       "h_star": b["h_star"], "d_i_hat": b["d_i_hat"],
                       "sigma_BB_i_hat": b["sigma_BB_i_hat"],
                       "snr": b["sigma_BB_i_hat"] / b["d_i_hat"],
                       "n_i_uniform": math.ceil(n_u)})
        n_vals.append(math.ceil(n_u))

    if n_vals:
        n_hat = int(max(n_vals))
        worst = blocks[int(np.argmax(n_vals))]["candidate"]
        arr = np.array(n_vals, dtype=float)
        ci_low, ci_high = float(arr.min()), float(n_hat)
        sigma = max(float(arr.std(ddof=1)) if arr.size > 1 else 1.0, 1.0)
    else:
        n_hat, worst = 0, None
        ci_low = ci_high = sigma = 1.0

    params = {
        "estimator": ("gambler-ruin order-1:  N_hat = max_u ceil(-ln(alpha) * "
                      "sigma_BB_u * sqrt(2m) / d_u),  k_u=1"),
        "alpha": alpha, "rho": RHO, "num_complements": num_complements,
        "N": N, "n_nominated": len(genes), "m_blocks": m_blocks,
        "worst_candidate": worst, "n_per_block": n_vals,
        "blocks": blocks, "measured": measured,
    }
    if extra_params:
        params.update(extra_params)
    return NHat(method=method, instance=instance, n_hat=n_hat, n_hat_raw=float(n_hat),
                ci_low=ci_low, ci_high=ci_high, sigma=sigma, params=params)
