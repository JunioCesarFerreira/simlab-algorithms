from __future__ import annotations

import math

import pytest

from p2_population_estimator.statistics import (
    confidence_interval_95,
    d_hat,
    delta_samples,
    sample_mean,
    sample_std,
    sample_variance,
    sigma_BB_hat,
    standard_error,
)


def test_sample_mean():
    assert sample_mean([1, 2, 3, 4]) == 2.5


def test_sample_variance_bessel():
    # var of [1,2,3,4] with ddof=1 is 1.6666...
    v = sample_variance([1, 2, 3, 4])
    assert math.isclose(v, 5 / 3, rel_tol=1e-9)


def test_sample_variance_requires_ddof():
    with pytest.raises(ValueError):
        sample_variance([1.0])


def test_sample_std_matches_sqrt_variance():
    xs = [10.0, 12.0, 23.0, 23.0, 16.0, 23.0, 21.0, 16.0]
    assert math.isclose(sample_std(xs), math.sqrt(sample_variance(xs)))


def test_standard_error():
    xs = [1.0, 2.0, 3.0, 4.0]
    se = standard_error(xs)
    assert math.isclose(se, sample_std(xs) / math.sqrt(len(xs)))


def test_ci_contains_mean():
    xs = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    lo, hi = confidence_interval_95(xs)
    assert lo < sample_mean(xs) < hi


def test_delta_and_d_hat():
    a = [3.0, 4.0, 5.0]
    b = [1.0, 2.0, 4.0]
    assert delta_samples(a, b) == [2.0, 2.0, 1.0]
    assert math.isclose(d_hat(a, b), (2 + 2 + 1) / 3)


def test_delta_length_mismatch():
    with pytest.raises(ValueError):
        delta_samples([1.0, 2.0], [1.0])


def test_sigma_BB_zero_when_no_variance():
    a = [3.0, 3.0, 3.0]
    b = [1.0, 1.0, 1.0]
    assert sigma_BB_hat(a, b) == 0.0
