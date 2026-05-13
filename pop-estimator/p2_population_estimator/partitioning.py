"""Partition the candidate set ``Q`` into structural blocks ``Q_1, ..., Q_m``.

Implemented methods:

  - ``grid``           — axis-aligned spatial grid; one block per non-empty cell.
  - ``kmeans``         — k-means on (x, y); requires scikit-learn (with a
                         simple stdlib fallback when sklearn is missing).
  - ``radial_to_sink`` — bin candidates by their distance to the sink.

All methods return ``list[CandidateBlock]`` with stable, deterministic order.
"""

from __future__ import annotations

import math
import random
from typing import Literal

from p2_population_estimator.config import KMEANS_MISSING_HINT
from p2_population_estimator.geometry import euclidean
from p2_population_estimator.logging_utils import get_logger
from p2_population_estimator.models import CandidateBlock, P2Problem, Point

log = get_logger(__name__)

PartitionMethod = Literal["grid", "kmeans", "radial_to_sink"]


def partition(
    problem: P2Problem,
    method: PartitionMethod,
    num_blocks: int,
    *,
    random_seed: int = 0,
) -> list[CandidateBlock]:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be a positive integer.")
    if method == "grid":
        return _partition_grid(problem, num_blocks)
    if method == "kmeans":
        return _partition_kmeans(problem, num_blocks, random_seed=random_seed)
    if method == "radial_to_sink":
        return _partition_radial(problem, num_blocks)
    raise ValueError(f"Unknown partition method: {method!r}")


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------
def _partition_grid(problem: P2Problem, num_blocks: int) -> list[CandidateBlock]:
    """Use a roughly-square grid with at most ``num_blocks`` non-empty cells.

    We start from ``ceil(sqrt(num_blocks))`` rows/cols and merge empty cells
    out. If the result is fewer than ``num_blocks`` non-empty cells, that is
    fine — the spec says "each non-empty cell becomes a block".
    """
    side = max(1, math.ceil(math.sqrt(num_blocks)))
    rows = side
    cols = side
    r = problem.region
    cw = r.width / cols
    ch = r.height / rows
    if cw <= 0 or ch <= 0:
        raise ValueError("Region has non-positive dimensions.")

    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, p in enumerate(problem.candidates):
        cx = min(cols - 1, max(0, int((p.x - r.xmin) // cw)))
        cy = min(rows - 1, max(0, int((p.y - r.ymin) // ch)))
        buckets.setdefault((cx, cy), []).append(idx)

    # Deterministic ordering (row-major)
    ordered_keys = sorted(buckets.keys(), key=lambda k: (k[1], k[0]))
    blocks: list[CandidateBlock] = []
    for bid, key in enumerate(ordered_keys):
        cx, cy = key
        blocks.append(
            CandidateBlock(
                block_id=bid,
                indices=sorted(buckets[key]),
                method="grid",
                metadata={"cell": [cx, cy], "rows": rows, "cols": cols},
            )
        )
    return blocks


# ---------------------------------------------------------------------------
# kmeans
# ---------------------------------------------------------------------------
def _partition_kmeans(
    problem: P2Problem, num_blocks: int, *, random_seed: int
) -> list[CandidateBlock]:
    coords = [[p.x, p.y] for p in problem.candidates]
    try:
        from sklearn.cluster import KMeans  # type: ignore

        km = KMeans(n_clusters=num_blocks, n_init=10, random_state=random_seed)
        labels = km.fit_predict(coords)
        labels_list: list[int] = [int(lbl) for lbl in labels]
    except ImportError:
        log.warning(
            "scikit-learn not available; falling back to a Lloyd's-algorithm "
            "implementation (works but lacks sklearn's robustness)",
            kv={"hint": KMEANS_MISSING_HINT},
        )
        labels_list = _kmeans_fallback(coords, num_blocks, random_seed)

    buckets: dict[int, list[int]] = {}
    for idx, lbl in enumerate(labels_list):
        buckets.setdefault(lbl, []).append(idx)

    blocks: list[CandidateBlock] = []
    for bid, lbl in enumerate(sorted(buckets.keys())):
        blocks.append(
            CandidateBlock(
                block_id=bid,
                indices=sorted(buckets[lbl]),
                method="kmeans",
                metadata={"cluster_label": lbl, "k": num_blocks},
            )
        )
    return blocks


def _kmeans_fallback(
    coords: list[list[float]], k: int, seed: int, max_iter: int = 50
) -> list[int]:
    """Tiny Lloyd's k-means used only if sklearn is missing."""
    rng = random.Random(seed)
    n = len(coords)
    if k >= n:
        return list(range(n))
    # k-means++ init
    centers: list[list[float]] = [list(coords[rng.randrange(n)])]
    for _ in range(1, k):
        d2 = [min((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for c in centers) for p in coords]
        total = sum(d2) or 1.0
        r = rng.random() * total
        cum = 0.0
        chosen = 0
        for i, w in enumerate(d2):
            cum += w
            if cum >= r:
                chosen = i
                break
        centers.append(list(coords[chosen]))

    labels = [0] * n
    for _ in range(max_iter):
        # Assign
        new_labels = []
        for p in coords:
            best = 0
            best_d = float("inf")
            for i, c in enumerate(centers):
                d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2
                if d < best_d:
                    best_d = d
                    best = i
            new_labels.append(best)
        if new_labels == labels:
            break
        labels = new_labels
        # Update
        sums = [[0.0, 0.0] for _ in range(k)]
        counts = [0] * k
        for p, lbl in zip(coords, labels):
            sums[lbl][0] += p[0]
            sums[lbl][1] += p[1]
            counts[lbl] += 1
        for i in range(k):
            if counts[i] > 0:
                centers[i] = [sums[i][0] / counts[i], sums[i][1] / counts[i]]
    return labels


# ---------------------------------------------------------------------------
# radial_to_sink
# ---------------------------------------------------------------------------
def _partition_radial(problem: P2Problem, num_blocks: int) -> list[CandidateBlock]:
    """Bin candidates by distance to sink into ``num_blocks`` quantile-based bins."""
    distances = [(i, euclidean(p, problem.sink)) for i, p in enumerate(problem.candidates)]
    distances.sort(key=lambda t: t[1])
    n = len(distances)
    if num_blocks > n:
        num_blocks = n
    bucket_size = math.ceil(n / num_blocks)

    blocks: list[CandidateBlock] = []
    for bid in range(num_blocks):
        start = bid * bucket_size
        end = min(start + bucket_size, n)
        if start >= end:
            break
        indices = sorted(idx for idx, _ in distances[start:end])
        dmin = distances[start][1]
        dmax = distances[end - 1][1]
        blocks.append(
            CandidateBlock(
                block_id=bid,
                indices=indices,
                method="radial_to_sink",
                metadata={"d_min": dmin, "d_max": dmax},
            )
        )
    return blocks


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------
def summarise_partition(blocks: list[CandidateBlock]) -> dict[str, object]:
    sizes = [b.k for b in blocks]
    return {
        "num_blocks": len(blocks),
        "k_min": min(sizes) if sizes else 0,
        "k_max": max(sizes) if sizes else 0,
        "k_mean": (sum(sizes) / len(sizes)) if sizes else 0.0,
        "blocks": [
            {"block_id": b.block_id, "k": b.k, "method": b.method, "indices": list(b.indices)}
            for b in blocks
        ],
    }
