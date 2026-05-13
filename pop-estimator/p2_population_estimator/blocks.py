"""Build H_i^* (block-optimum candidate) and H_i^L (deceptive competitor).

The actual optimal block configuration is unknown in general, so we expose a
collection of plug-in **heuristics** under the same interface. The choice of
heuristic is recorded in the experiment output, since it is part of what
makes the n_hat value *heuristic*.

Public entry points:

  - :func:`build_h_star(method, block, problem, ...)` -> BlockPattern
  - :func:`build_h_local(method, block, problem, h_star=..., ...)` -> BlockPattern
  - :func:`compose_full_solution(problem, blocks, block_pattern, complement_bits)`
        Build a full binary vector x by overlaying ``block_pattern`` on the
        active block and ``complement_bits`` on every position outside it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

from p2_population_estimator.geometry import (
    euclidean,
    sample_all_mobiles,
)
from p2_population_estimator.logging_utils import get_logger
from p2_population_estimator.models import (
    BlockPattern,
    CandidateBlock,
    FullSolution,
    P2Problem,
)

log = get_logger(__name__)

# Methods enum
H_STAR_METHODS = ("structural_greedy", "dense_local", "external")
H_LOCAL_METHODS = (
    "deceptive_low_cost",
    "redundant_local",
    "far_from_sink",
    "random_competitor",
)


# ---------------------------------------------------------------------------
# Utility scores
# ---------------------------------------------------------------------------
def _proximity_to_trajectories(
    block: CandidateBlock, problem: P2Problem, time_samples: int = 24
) -> dict[int, float]:
    """For each candidate in the block, mean distance to all sampled mobile points.

    Lower is better. Returns ``{idx: mean_distance}``. Empty trajectories
    yield +inf (so a tie-break by other criteria can still rank them).
    """
    samples = [pt for traj in sample_all_mobiles(problem, time_samples) for pt in traj]
    if not samples:
        return {idx: float("inf") for idx in block.indices}
    out: dict[int, float] = {}
    for idx in block.indices:
        c = problem.candidates[idx]
        s = sum(euclidean(c, p) for p in samples)
        out[idx] = s / len(samples)
    return out


def _proximity_to_sink(block: CandidateBlock, problem: P2Problem) -> dict[int, float]:
    return {idx: euclidean(problem.candidates[idx], problem.sink) for idx in block.indices}


def _local_degree(block: CandidateBlock, problem: P2Problem) -> dict[int, int]:
    """Degree of each candidate within the block under the reach radius."""
    R = problem.radius_of_reach
    deg: dict[int, int] = {idx: 0 for idx in block.indices}
    idxs = list(block.indices)
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            a, b = idxs[i], idxs[j]
            if euclidean(problem.candidates[a], problem.candidates[b]) <= R:
                deg[a] += 1
                deg[b] += 1
    return deg


# ---------------------------------------------------------------------------
# H_i^* heuristics
# ---------------------------------------------------------------------------
def _pick_s_size(block: CandidateBlock, requested: int | None) -> int:
    """Decide how many bits are active in H_i^*. By default ~ k/2, clamped."""
    if requested is not None:
        return max(1, min(block.k, requested))
    return max(1, min(block.k, max(1, block.k // 2)))


def h_star_structural_greedy(
    block: CandidateBlock,
    problem: P2Problem,
    *,
    s_size: int | None = None,
) -> BlockPattern:
    """Combine proximity-to-trajectories, proximity-to-sink, and local degree
    into a single ranking, then select the top ``s_size`` candidates.

    Score (lower = better):
        score(j) = z(d_traj(j)) + 0.5 * z(d_sink(j)) - 0.3 * z(deg(j))

    We normalise each term with z-scores within the block, so the weights can
    be roughly compared. The bias toward higher degree (- z(deg)) favours
    candidates that contribute more to local connectivity.
    """
    s = _pick_s_size(block, s_size)
    d_traj = _proximity_to_trajectories(block, problem)
    d_sink = _proximity_to_sink(block, problem)
    deg = _local_degree(block, problem)

    def zscores(d: dict[int, float]) -> dict[int, float]:
        vals = list(d.values())
        finite = [v for v in vals if v != float("inf")]
        if not finite:
            return {k: 0.0 for k in d}
        m = sum(finite) / len(finite)
        var = sum((v - m) ** 2 for v in finite) / max(1, len(finite) - 1)
        sd = var ** 0.5 or 1.0
        return {k: ((v - m) / sd) if v != float("inf") else 3.0 for k, v in d.items()}

    z_traj = zscores(d_traj)
    z_sink = zscores(d_sink)
    z_deg = zscores({k: float(v) for k, v in deg.items()})

    scores = {
        idx: z_traj[idx] + 0.5 * z_sink[idx] - 0.3 * z_deg[idx]
        for idx in block.indices
    }
    ranked = sorted(block.indices, key=lambda idx: (scores[idx], idx))
    chosen = set(ranked[:s])
    bits = [1 if idx in chosen else 0 for idx in block.indices]
    return BlockPattern(block_id=block.block_id, bits=bits, label="H_star")


def h_star_dense_local(
    block: CandidateBlock,
    problem: P2Problem,
    *,
    s_size: int | None = None,
) -> BlockPattern:
    """Select the ``s`` candidates with the highest in-block degree."""
    s = _pick_s_size(block, s_size)
    deg = _local_degree(block, problem)
    ranked = sorted(block.indices, key=lambda idx: (-deg[idx], idx))
    chosen = set(ranked[:s])
    bits = [1 if idx in chosen else 0 for idx in block.indices]
    return BlockPattern(block_id=block.block_id, bits=bits, label="H_star")


def h_star_external(
    block: CandidateBlock,
    problem: P2Problem,
    *,
    external_path: str,
) -> BlockPattern:
    """Load H_i^* from an external JSON file.

    Expected schema:
        { "blocks": [ { "block_id": <int>, "bits": [0,1,...] }, ... ] }
    The ``bits`` length must match ``block.k``.
    """
    data = json.loads(Path(external_path).read_text(encoding="utf-8"))
    for b in data.get("blocks", []):
        if int(b["block_id"]) == block.block_id:
            bits = [int(x) for x in b["bits"]]
            if len(bits) != block.k:
                raise ValueError(
                    f"external H_star block {block.block_id} has length {len(bits)}, expected {block.k}"
                )
            return BlockPattern(block_id=block.block_id, bits=bits, label="H_star")
    raise KeyError(f"External H_star file has no entry for block {block.block_id}")


# ---------------------------------------------------------------------------
# H_i^L heuristics
# ---------------------------------------------------------------------------
def h_local_deceptive_low_cost(
    block: CandidateBlock,
    problem: P2Problem,
    *,
    h_star: BlockPattern,
) -> BlockPattern:
    """Activate FEWER bits than ``h_star``. Picks the lowest-cost candidates
    (here cost ~ proximity to sink, so it picks ones that "look cheap" but
    likely break connectivity to far-away mobile nodes).
    """
    target_s = max(1, h_star.s - 1) if h_star.s >= 2 else 1
    d_sink = _proximity_to_sink(block, problem)
    ranked = sorted(block.indices, key=lambda idx: (d_sink[idx], idx))
    chosen = set(ranked[:target_s])
    bits = [1 if idx in chosen else 0 for idx in block.indices]
    return BlockPattern(block_id=block.block_id, bits=bits, label="H_local")


def h_local_redundant_local(
    block: CandidateBlock,
    problem: P2Problem,
    *,
    h_star: BlockPattern,
) -> BlockPattern:
    """Pick the ``s`` mutually-closest candidates -> high redundancy, low coverage."""
    s = max(1, h_star.s)
    idxs = list(block.indices)
    if not idxs:
        return BlockPattern(block_id=block.block_id, bits=[], label="H_local")
    # Greedy: start from the centroid, then add the next-closest candidate.
    cx = sum(problem.candidates[i].x for i in idxs) / len(idxs)
    cy = sum(problem.candidates[i].y for i in idxs) / len(idxs)
    ranked = sorted(idxs, key=lambda i: (problem.candidates[i].x - cx) ** 2 + (problem.candidates[i].y - cy) ** 2)
    chosen = set(ranked[:s])
    bits = [1 if idx in block.indices and idx in chosen else 0 for idx in block.indices]
    return BlockPattern(block_id=block.block_id, bits=bits, label="H_local")


def h_local_far_from_sink(
    block: CandidateBlock,
    problem: P2Problem,
    *,
    h_star: BlockPattern,
) -> BlockPattern:
    """Pick relays far from the sink (poor for sink-bound routing)."""
    s = max(1, h_star.s)
    d_sink = _proximity_to_sink(block, problem)
    ranked = sorted(block.indices, key=lambda idx: (-d_sink[idx], idx))
    chosen = set(ranked[:s])
    bits = [1 if idx in chosen else 0 for idx in block.indices]
    return BlockPattern(block_id=block.block_id, bits=bits, label="H_local")


def h_local_random_competitor(
    block: CandidateBlock,
    problem: P2Problem,
    *,
    h_star: BlockPattern,
    rng: random.Random,
) -> BlockPattern:
    """Random pattern with the same number of active bits as ``h_star`` but
    explicitly different (when possible)."""
    s = h_star.s
    idxs = list(block.indices)
    for _ in range(16):
        sample = rng.sample(idxs, k=min(s, len(idxs)))
        chosen = set(sample)
        bits = [1 if idx in chosen else 0 for idx in block.indices]
        if bits != h_star.bits:
            return BlockPattern(block_id=block.block_id, bits=bits, label="H_local")
    return BlockPattern(block_id=block.block_id, bits=bits, label="H_local")  # type: ignore[has-type]


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------
def build_h_star(
    method: str,
    block: CandidateBlock,
    problem: P2Problem,
    *,
    s_size: int | None = None,
    external_path: str | None = None,
) -> BlockPattern:
    if method == "structural_greedy":
        return h_star_structural_greedy(block, problem, s_size=s_size)
    if method == "dense_local":
        return h_star_dense_local(block, problem, s_size=s_size)
    if method == "external":
        if not external_path:
            raise ValueError("h_star_external requires --hstar-external-path")
        return h_star_external(block, problem, external_path=external_path)
    raise ValueError(f"Unknown h_star method: {method!r}. Choose from {H_STAR_METHODS}")


def build_h_local(
    method: str,
    block: CandidateBlock,
    problem: P2Problem,
    *,
    h_star: BlockPattern,
    rng: random.Random,
) -> BlockPattern:
    if method == "deceptive_low_cost":
        return h_local_deceptive_low_cost(block, problem, h_star=h_star)
    if method == "redundant_local":
        return h_local_redundant_local(block, problem, h_star=h_star)
    if method == "far_from_sink":
        return h_local_far_from_sink(block, problem, h_star=h_star)
    if method == "random_competitor":
        return h_local_random_competitor(block, problem, h_star=h_star, rng=rng)
    raise ValueError(f"Unknown h_local method: {method!r}. Choose from {H_LOCAL_METHODS}")


# ---------------------------------------------------------------------------
# Solution composition
# ---------------------------------------------------------------------------
def compose_full_solution(
    problem: P2Problem,
    block: CandidateBlock,
    block_pattern: BlockPattern,
    complement_bits: list[int],
    *,
    solution_id: str,
) -> FullSolution:
    """Build x in {0,1}^J by placing ``block_pattern`` on the block's positions
    and ``complement_bits`` everywhere else.

    ``complement_bits`` must have length ``J``; its values at the block's
    positions are IGNORED — they're overwritten by ``block_pattern``. This
    matches the mathematical statement: x = (H_i, x_{-i}).
    """
    J = len(problem.candidates)
    if len(complement_bits) != J:
        raise ValueError(
            f"complement_bits has length {len(complement_bits)}, expected J={J}"
        )
    if len(block_pattern.bits) != block.k:
        raise ValueError(
            f"block_pattern.bits has length {len(block_pattern.bits)}, expected k={block.k}"
        )
    bits = list(complement_bits)
    for pos_in_block, original_idx in enumerate(block.indices):
        bits[original_idx] = int(block_pattern.bits[pos_in_block])
    # Make sure bits are 0/1
    bits = [1 if b else 0 for b in bits]
    return FullSolution(
        solution_id=solution_id,
        bits=bits,
        provenance={
            "block_id": block.block_id,
            "block_pattern_label": block_pattern.label,
        },
    )


# ---------------------------------------------------------------------------
# Bernoulli prior probability of H_i^*
# ---------------------------------------------------------------------------
def bernoulli_pi(block_pattern: BlockPattern, rho: float) -> float:
    """pi_i(H_i^*) = rho^{s_i} (1-rho)^{k_i - s_i}."""
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0, 1).")
    s = block_pattern.s
    k = block_pattern.k
    return (rho ** s) * ((1.0 - rho) ** (k - s))
