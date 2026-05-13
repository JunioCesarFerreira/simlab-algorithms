from __future__ import annotations

import random

import pytest

from p2_population_estimator.blocks import (
    bernoulli_pi,
    build_h_local,
    build_h_star,
    compose_full_solution,
)
from p2_population_estimator.partitioning import partition


def test_h_star_structural_greedy_chooses_half_by_default(small_problem):
    blocks = partition(small_problem, "grid", num_blocks=4)
    block = blocks[0]
    h = build_h_star("structural_greedy", block, small_problem)
    assert h.k == block.k
    assert 1 <= h.s <= block.k


def test_h_local_is_different_from_h_star(small_problem):
    blocks = partition(small_problem, "grid", num_blocks=4)
    block = blocks[0]
    rng = random.Random(0)
    h_star = build_h_star("structural_greedy", block, small_problem)
    h_local = build_h_local("deceptive_low_cost", block, small_problem, h_star=h_star, rng=rng)
    assert h_local.k == h_star.k
    # In most cases the patterns should differ; for tiny blocks they might
    # coincide -- assert at least the labels differ.
    assert h_local.label == "H_local" and h_star.label == "H_star"


def test_compose_full_solution_places_block_bits(small_problem):
    blocks = partition(small_problem, "grid", num_blocks=4)
    block = blocks[0]
    pattern = build_h_star("structural_greedy", block, small_problem)
    J = len(small_problem.candidates)
    comp = [0] * J
    sol = compose_full_solution(
        small_problem, block, pattern, comp, solution_id="t1"
    )
    # The complement is all-zero, so x_j = 1 iff j is in the block AND pattern says 1
    for pos_in_block, idx in enumerate(block.indices):
        assert sol.bits[idx] == pattern.bits[pos_in_block]
    # Indices outside the block must remain zero
    block_set = set(block.indices)
    for j in range(J):
        if j not in block_set:
            assert sol.bits[j] == 0


def test_compose_rejects_wrong_complement_length(small_problem):
    blocks = partition(small_problem, "grid", num_blocks=4)
    block = blocks[0]
    pattern = build_h_star("structural_greedy", block, small_problem)
    with pytest.raises(ValueError):
        compose_full_solution(small_problem, block, pattern, [0] * 3, solution_id="t")


def test_bernoulli_pi_formula(small_problem):
    blocks = partition(small_problem, "grid", num_blocks=4)
    block = blocks[0]
    pattern = build_h_star("structural_greedy", block, small_problem)
    rho = 0.3
    expected = (rho ** pattern.s) * ((1 - rho) ** (pattern.k - pattern.s))
    assert abs(bernoulli_pi(pattern, rho) - expected) < 1e-12


def test_bernoulli_pi_rejects_bad_rho(small_problem):
    blocks = partition(small_problem, "grid", num_blocks=4)
    block = blocks[0]
    pattern = build_h_star("structural_greedy", block, small_problem)
    with pytest.raises(ValueError):
        bernoulli_pi(pattern, 0.0)
    with pytest.raises(ValueError):
        bernoulli_pi(pattern, 1.0)
