from __future__ import annotations

import pytest

from p2_population_estimator.partitioning import partition, summarise_partition


def test_grid_partition_covers_all_candidates(small_problem):
    blocks = partition(small_problem, "grid", num_blocks=4)
    seen: set[int] = set()
    for b in blocks:
        for idx in b.indices:
            assert idx not in seen, "indices must be partitioned (no duplicates)"
            seen.add(idx)
    assert seen == set(range(len(small_problem.candidates)))


def test_radial_partition_orders_by_distance(small_problem):
    blocks = partition(small_problem, "radial_to_sink", num_blocks=4)
    # Each block's d_max should be <= next block's d_min
    for a, b in zip(blocks, blocks[1:]):
        assert a.metadata["d_max"] <= b.metadata["d_min"] + 1e-9


def test_kmeans_partition(small_problem):
    blocks = partition(small_problem, "kmeans", num_blocks=2, random_seed=1)
    seen: set[int] = set()
    for b in blocks:
        for idx in b.indices:
            seen.add(idx)
    assert seen == set(range(len(small_problem.candidates)))


def test_invalid_method_raises(small_problem):
    with pytest.raises(ValueError):
        partition(small_problem, "foo", num_blocks=2)  # type: ignore[arg-type]


def test_summarise(small_problem):
    blocks = partition(small_problem, "grid", num_blocks=4)
    s = summarise_partition(blocks)
    assert s["num_blocks"] == len(blocks)
    assert s["k_min"] >= 1
