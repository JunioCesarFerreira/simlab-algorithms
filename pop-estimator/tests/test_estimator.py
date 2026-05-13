from __future__ import annotations

import math

import pytest

from p2_population_estimator.estimator import (
    aggregate_global,
    estimate_bernoulli,
    estimate_block,
    estimate_uniform,
)
from p2_population_estimator.models import BlockPattern


def test_estimate_uniform_formula():
    # n = -ln(alpha) * 2^{k-1} * sigma * sqrt(2m) / d
    alpha, k, m, sigma, d = 0.05, 4, 8, 0.3, 0.1
    expected = -math.log(alpha) * (2 ** (k - 1)) * sigma * math.sqrt(2 * m) / d
    got = estimate_uniform(alpha, k, m, sigma, d)
    assert math.isclose(got, expected, rel_tol=1e-12)


def test_estimate_bernoulli_formula():
    alpha, pi_i, m, sigma, d = 0.05, 0.04, 8, 0.3, 0.1
    expected = (-math.log(alpha) / (2 * pi_i)) * sigma * math.sqrt(2 * m) / d
    got = estimate_bernoulli(alpha, pi_i, m, sigma, d)
    assert math.isclose(got, expected, rel_tol=1e-12)


def test_estimate_uniform_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        estimate_uniform(1.0, 2, 4, 0.1, 0.5)
    with pytest.raises(ValueError):
        estimate_uniform(0.0, 2, 4, 0.1, 0.5)


def test_estimate_bernoulli_rejects_invalid_pi():
    with pytest.raises(ValueError):
        estimate_bernoulli(0.05, 0.0, 4, 0.1, 0.5)


def test_estimate_block_ok_case():
    h_star = BlockPattern(block_id=0, bits=[1, 0, 1, 0], label="H_star")
    # Non-zero variance in the deltas: deltas = [1.0, 0.8, 1.3, 0.6]
    F_star = [1.0, 1.0, 1.5, 1.0]
    F_local = [0.0, 0.2, 0.2, 0.4]
    res = estimate_block(
        block_id=0, k_i=4, m=8, alpha=0.05, h_star=h_star,
        F_star=F_star, F_local=F_local, rho=0.2,
    )
    assert res.status == "ok"
    assert res.d_i_hat > 0
    assert res.sigma_BB_i_hat > 0
    assert res.n_i_uniform is not None
    assert res.n_i_uniform_ceil is not None
    assert res.n_i_bernoulli is not None
    assert res.pi_i_star is not None


def test_estimate_block_non_positive_d():
    h_star = BlockPattern(block_id=0, bits=[1, 0, 1, 0], label="H_star")
    F_star = [0.0, 0.0, 0.0, 0.0]
    F_local = [1.0, 1.0, 1.0, 1.0]
    res = estimate_block(
        block_id=0, k_i=4, m=8, alpha=0.05, h_star=h_star,
        F_star=F_star, F_local=F_local, rho=0.2,
    )
    assert res.status == "invalid_non_positive_d"
    assert res.n_i_uniform is None
    assert res.n_i_bernoulli is None


def test_estimate_block_insufficient_samples():
    h_star = BlockPattern(block_id=0, bits=[1, 0], label="H_star")
    res = estimate_block(
        block_id=0, k_i=2, m=4, alpha=0.05, h_star=h_star,
        F_star=[1.0], F_local=[0.0], rho=0.2,
    )
    assert res.status == "insufficient_samples"


def test_estimate_block_degenerate_zero_variance():
    h_star = BlockPattern(block_id=0, bits=[1, 0, 1, 0], label="H_star")
    # Equal deltas -> zero variance, but positive d
    F_star = [1.0, 1.0, 1.0, 1.0]
    F_local = [0.0, 0.0, 0.0, 0.0]
    res = estimate_block(
        block_id=0, k_i=4, m=8, alpha=0.05, h_star=h_star,
        F_star=F_star, F_local=F_local, rho=0.2,
    )
    assert res.status == "degenerate_zero_variance"
    assert res.n_i_uniform is not None  # still computed using SIGMA_FLOOR


def test_aggregate_global():
    h_star = BlockPattern(block_id=0, bits=[1, 0], label="H_star")
    # Two blocks: one valid (with non-degenerate variance), one invalid
    ok = estimate_block(
        block_id=0, k_i=2, m=4, alpha=0.05, h_star=h_star,
        F_star=[1.0, 1.5, 0.8], F_local=[0.0, 0.2, 0.1], rho=0.2,
    )
    bad = estimate_block(
        block_id=1, k_i=2, m=4, alpha=0.05, h_star=h_star,
        F_star=[0.0, 0.0, 0.0], F_local=[1.0, 1.0, 1.0], rho=0.2,
    )
    g = aggregate_global([ok, bad])
    assert g["num_valid_blocks"] == 1
    assert g["num_invalid_blocks"] == 1
    assert g["n_hat_uniform"] == ok.n_i_uniform_ceil
    assert g["most_difficult_block_id"] == 0
