from __future__ import annotations

import math

from p2_population_estimator.blocks import (
    build_h_local,
    build_h_star,
    compose_full_solution,
)
from p2_population_estimator.evaluation.surrogate import SurrogateEvaluator
from p2_population_estimator.models import FullSolution, ScalarizationWeights
from p2_population_estimator.partitioning import partition


def test_surrogate_is_deterministic(small_problem):
    weights = ScalarizationWeights()
    ev = SurrogateEvaluator(small_problem, weights)
    J = len(small_problem.candidates)
    sol = FullSolution(solution_id="t", bits=[1] * J)
    r1 = ev.evaluate(sol, [1, 2, 3])
    r2 = ev.evaluate(sol, [1, 2, 3])
    assert math.isclose(r1.F, r2.F, rel_tol=1e-12)


def test_surrogate_more_relays_better_connectivity(small_problem):
    weights = ScalarizationWeights()
    ev = SurrogateEvaluator(small_problem, weights)
    J = len(small_problem.candidates)
    full = FullSolution(solution_id="full", bits=[1] * J)
    empty = FullSolution(solution_id="empty", bits=[0] * J)
    r_full = ev.evaluate(full, [1])
    r_empty = ev.evaluate(empty, [1])
    # Connectivity ratio of full >= empty
    assert (r_full.aggregated.mean.connected_ratio or 0) >= (
        r_empty.aggregated.mean.connected_ratio or 0
    )


def test_surrogate_pipeline_block_against_complement(small_problem):
    """Smoke test for the full block-level evaluation flow with surrogate."""
    weights = ScalarizationWeights()
    ev = SurrogateEvaluator(small_problem, weights)
    blocks = partition(small_problem, "grid", num_blocks=4)
    block = blocks[0]
    h_star = build_h_star("structural_greedy", block, small_problem)
    import random
    rng = random.Random(0)
    h_local = build_h_local("deceptive_low_cost", block, small_problem, h_star=h_star, rng=rng)
    J = len(small_problem.candidates)
    comp = [1 if rng.random() < 0.3 else 0 for _ in range(J)]
    x_star = compose_full_solution(small_problem, block, h_star, comp, solution_id="s")
    x_local = compose_full_solution(small_problem, block, h_local, comp, solution_id="l")
    r_s = ev.evaluate(x_star, [1])
    r_l = ev.evaluate(x_local, [1])
    assert isinstance(r_s.F, float)
    assert isinstance(r_l.F, float)
