"""Generate complements x_{-i}^{(r)} that fill the bits *outside* the block.

We produce ``R`` complement vectors of length ``J = |Q|``. The bits inside
the active block ``Q_i`` will later be overwritten by H_i^* / H_i^L, so
their values here do not matter — but we keep them random for parity of
length.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

from p2_population_estimator.geometry import build_relay_graph, shortest_hops_to_sink
from p2_population_estimator.models import CandidateBlock, P2Problem

ComplementMethod = Literal["bernoulli", "feasible_repair", "population_sample"]


def generate_complements(
    method: ComplementMethod,
    problem: P2Problem,
    block: CandidateBlock,
    *,
    num_complements: int,
    rho: float,
    rng: random.Random,
    external_path: str | None = None,
) -> list[list[int]]:
    if num_complements <= 0:
        raise ValueError("num_complements must be a positive integer.")
    if method == "bernoulli":
        return _bernoulli(problem, num_complements, rho, rng)
    if method == "feasible_repair":
        return _feasible_repair(problem, num_complements, rho, rng)
    if method == "population_sample":
        if not external_path:
            raise ValueError("complement_method=population_sample requires --complement-external-path")
        return _from_external(problem, num_complements, external_path, rng)
    raise ValueError(f"Unknown complement method: {method!r}")


def _bernoulli(
    problem: P2Problem, R: int, rho: float, rng: random.Random
) -> list[list[int]]:
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0,1) for Bernoulli complements.")
    J = len(problem.candidates)
    return [[1 if rng.random() < rho else 0 for _ in range(J)] for _ in range(R)]


def _feasible_repair(
    problem: P2Problem, R: int, rho: float, rng: random.Random
) -> list[list[int]]:
    """Generate Bernoulli complements then repair until the relay-graph is
    connected up to a sink-anchored component (when feasible).

    "Repair" here means: while the sink has no neighbours selected, activate
    the relay closest to the sink that is still off.
    """
    J = len(problem.candidates)
    R_radius = problem.radius_of_reach
    out: list[list[int]] = []
    for _ in range(R):
        bits = [1 if rng.random() < rho else 0 for _ in range(J)]
        for _ in range(J):  # bounded number of repair steps
            sel = [j for j, b in enumerate(bits) if b]
            adj = build_relay_graph(problem.candidates, problem.sink, R_radius, sel)
            hops = shortest_hops_to_sink(adj)
            reachable = {j for j in hops if j != -1}
            if reachable or not sel:
                # Either we have a sink-connected component, or there's nothing
                # to fix yet — accept.
                break
            # Activate the off-candidate closest to the sink.
            off = [j for j in range(J) if not bits[j]]
            if not off:
                break
            off.sort(
                key=lambda j: (
                    (problem.candidates[j].x - problem.sink.x) ** 2
                    + (problem.candidates[j].y - problem.sink.y) ** 2
                )
            )
            bits[off[0]] = 1
        out.append(bits)
    return out


def _from_external(
    problem: P2Problem, R: int, path: str, rng: random.Random
) -> list[list[int]]:
    """Load an external population file.

    Expected schema (flexible):
        { "population": [ [0,1,0,...], ... ] }
        or
        [ [0,1,0,...], ... ]
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "population" in data:
        pop = data["population"]
    elif isinstance(data, list):
        pop = data
    else:
        raise ValueError(f"Unrecognised external complement schema in {path}")
    J = len(problem.candidates)
    sane = []
    for row in pop:
        if not isinstance(row, list) or len(row) != J:
            raise ValueError(
                f"External population row has length {len(row) if isinstance(row, list) else 'N/A'}, expected {J}"
            )
        sane.append([1 if int(b) else 0 for b in row])
    if not sane:
        raise ValueError("External population is empty.")
    # Sample with replacement to reach R entries (deterministic via rng).
    return [list(rng.choice(sane)) for _ in range(R)]
