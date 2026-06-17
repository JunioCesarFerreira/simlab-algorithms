"""A compact, well-instrumented binary genetic algorithm.

Representation
--------------
A chromosome is a binary vector ``x ∈ {0,1}^N`` (``N`` = number of candidate
relay positions).  ``x[j] = 1`` means candidate ``j`` is installed.  This is
exactly the encoding used by the MILP sweep, so GA and MILP solutions live in
the same space and can be compared bit-for-bit.

Operators
---------
* **Selection**  — k-tournament (default k=3), maximising the scalar fitness F.
* **Crossover**  — uniform crossover with per-gene swap (probability 0.5),
  applied to a pair with probability ``p_crossover``.  Uniform crossover is a
  natural choice for subset-selection problems where good genes are not
  contiguous on the chromosome.
* **Mutation**   — independent bit-flip with per-bit probability ``p_mutation``
  (default ``1/N``, i.e. ~1 flipped bit per child in expectation).
* **Elitism**    — the best ``elitism`` individuals survive unchanged.

Instrumentation
---------------
Every generation we record best/mean fitness, the best individual's relay count
and connectivity, and the population **diversity** (mean pairwise Hamming
distance normalised by ``N``).  Diversity is the quantity Method 1 turns into an
information-gain curve; the per-generation best-fitness trace and the final best
solution feed Methods 1 and 4.

A fitness cache (keyed by the chromosome bytes) avoids re-evaluating identical
individuals, which matters because elitism and converging populations produce
many repeats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(slots=True)
class GAConfig:
    n_bits: int
    pop_size: int
    generations: int = 80
    p_crossover: float = 0.9
    p_mutation: float | None = None      # default: 1 / n_bits
    tournament_k: int = 3
    elitism: int = 1
    init_rho: float = 0.5                # Bernoulli(rho) for the initial population
    seed: int = 0

    def resolved_p_mutation(self) -> float:
        return self.p_mutation if self.p_mutation is not None else 1.0 / self.n_bits


@dataclass(slots=True)
class GAResult:
    config: dict[str, Any]
    best_bits: list[int]
    best_F: float
    best_metrics: dict[str, float]
    history: list[dict[str, float]]       # one row per generation
    evaluations: int                      # number of *distinct* fitness evaluations
    all_evaluated: list[tuple[tuple[int, ...], float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Population diversity
# ---------------------------------------------------------------------------
def population_diversity(pop: np.ndarray, *, max_pairs: int = 4000,
                         rng: np.random.Generator | None = None) -> float:
    """Mean pairwise Hamming distance normalised by the chromosome length.

    For large populations the exact O(P²) sum is approximated by sampling up to
    ``max_pairs`` random pairs.
    """
    P, N = pop.shape
    if P < 2:
        return 0.0
    n_all = P * (P - 1) // 2
    if n_all <= max_pairs:
        # exact: sum of pairwise distances via column-wise counts
        # dist_sum = sum_j ones_j * zeros_j  (per bit position)
        ones = pop.sum(axis=0)
        zeros = P - ones
        dist_sum = float(np.dot(ones, zeros))
        return (dist_sum / n_all) / N
    rng = rng or np.random.default_rng(0)
    a = rng.integers(0, P, size=max_pairs)
    b = rng.integers(0, P, size=max_pairs)
    mask = a != b
    a, b = a[mask], b[mask]
    d = np.count_nonzero(pop[a] != pop[b], axis=1)
    return float(d.mean()) / N


# ---------------------------------------------------------------------------
# GA
# ---------------------------------------------------------------------------
def run_ga(
    fitness_fn: Callable[[np.ndarray], dict[str, float]],
    cfg: GAConfig,
    *,
    record_evaluated: bool = False,
) -> GAResult:
    """Run the GA and return a fully-instrumented :class:`GAResult`.

    ``fitness_fn(bits) -> {"F": float, "relay_count": int, "connected_ratio": ...}``
    must return a dict containing at least the scalar key ``"F"`` (maximised).
    """
    rng = np.random.default_rng(cfg.seed)
    N, P = cfg.n_bits, cfg.pop_size
    pm = cfg.resolved_p_mutation()

    cache: dict[bytes, dict[str, float]] = {}
    evaluated_log: list[tuple[tuple[int, ...], float]] = []

    def evaluate(ind: np.ndarray) -> dict[str, float]:
        key = ind.tobytes()
        hit = cache.get(key)
        if hit is None:
            hit = fitness_fn(ind)
            cache[key] = hit
            if record_evaluated:
                evaluated_log.append((tuple(int(b) for b in ind), float(hit["F"])))
        return hit

    # --- initial population: Bernoulli(rho) ---
    pop = (rng.random((P, N)) < cfg.init_rho).astype(np.int8)
    fit = np.array([evaluate(ind)["F"] for ind in pop])

    history: list[dict[str, float]] = []

    def log_generation(g: int) -> None:
        best_i = int(np.argmax(fit))
        m = cache[pop[best_i].tobytes()]
        history.append({
            "generation": g,
            "best_F": float(fit[best_i]),
            "mean_F": float(fit.mean()),
            "best_relay_count": float(m.get("relay_count", float("nan"))),
            "best_connected_ratio": float(m.get("connected_ratio", float("nan"))),
            "diversity": population_diversity(pop, rng=rng),
            "distinct_evaluations": len(cache),
        })

    log_generation(0)

    def tournament() -> np.ndarray:
        idx = rng.integers(0, P, size=cfg.tournament_k)
        winner = idx[int(np.argmax(fit[idx]))]
        return pop[winner].copy()

    for g in range(1, cfg.generations + 1):
        # elitism: carry over the top individuals
        elite_idx = np.argsort(fit)[::-1][: cfg.elitism]
        new_pop = [pop[i].copy() for i in elite_idx]

        while len(new_pop) < P:
            p1, p2 = tournament(), tournament()
            if rng.random() < cfg.p_crossover:
                swap = rng.random(N) < 0.5
                c1 = np.where(swap, p2, p1)
                c2 = np.where(swap, p1, p2)
            else:
                c1, c2 = p1, p2
            for child in (c1, c2):
                flip = rng.random(N) < pm
                child[flip] ^= 1
                if len(new_pop) < P:
                    new_pop.append(child)

        pop = np.array(new_pop, dtype=np.int8)
        fit = np.array([evaluate(ind)["F"] for ind in pop])
        log_generation(g)

    best_i = int(np.argmax(fit))
    best_bits = pop[best_i]
    best_metrics = cache[best_bits.tobytes()]

    return GAResult(
        config={
            "n_bits": N, "pop_size": P, "generations": cfg.generations,
            "p_crossover": cfg.p_crossover, "p_mutation": pm,
            "tournament_k": cfg.tournament_k, "elitism": cfg.elitism,
            "init_rho": cfg.init_rho, "seed": cfg.seed,
        },
        best_bits=[int(b) for b in best_bits],
        best_F=float(fit[best_i]),
        best_metrics={k: float(v) for k, v in best_metrics.items()},
        history=history,
        evaluations=len(cache),
        all_evaluated=evaluated_log,
    )
