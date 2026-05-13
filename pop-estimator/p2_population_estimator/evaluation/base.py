"""Base evaluator interface, aggregation (Psi_a), and scalarisation (S).

The evaluator interface is intentionally narrow:

    evaluate(solution, seeds) -> EvaluationResult

The aggregation function and the scalarisation function are factored out as
free helpers so they can be reused by both surrogate and Cooja back-ends.
"""

from __future__ import annotations

import math
import statistics as stdstats
from abc import ABC, abstractmethod
from typing import Iterable

from p2_population_estimator.models import (
    AggregatedMetrics,
    EvaluationResult,
    FullSolution,
    ScalarizationWeights,
    SimulationMetrics,
)


# ---------------------------------------------------------------------------
# Evaluator base class
# ---------------------------------------------------------------------------
class BaseEvaluator(ABC):
    name: str = "base"

    @abstractmethod
    def evaluate(self, solution: FullSolution, seeds: list[int]) -> EvaluationResult:
        """Evaluate ``solution`` under the given seeds and return a result."""

    def shutdown(self) -> None:
        """Optional cleanup hook (close SSH connections, etc.)."""


# ---------------------------------------------------------------------------
# Aggregation (Psi_a)
# ---------------------------------------------------------------------------
_METRIC_FIELDS = (
    "latency",
    "energy",
    "throughput",
    "packet_delivery_ratio",
    "connected_ratio",
    "relay_count",
    "mean_hop_count",
    "mean_distance_to_mobile",
    "redundancy",
)


def _values(metrics: list[SimulationMetrics], field: str) -> list[float]:
    return [
        float(getattr(m, field))
        for m in metrics
        if getattr(m, field) is not None
    ]


def _agg_one(values: list[float], method: str) -> float | None:
    if not values:
        return None
    if method == "mean" or method == "mean_with_std":
        return sum(values) / len(values)
    if method == "median":
        return float(stdstats.median(values))
    if method == "trimmed_mean":
        if len(values) <= 2:
            return sum(values) / len(values)
        s = sorted(values)
        cut = max(1, len(s) // 10)
        s = s[cut:-cut] if len(s) > 2 * cut else s
        return sum(s) / len(s)
    raise ValueError(f"Unknown aggregation method: {method!r}")


def aggregate_metrics(
    metrics: list[SimulationMetrics], method: str = "mean_with_std"
) -> AggregatedMetrics:
    """Apply Psi_a across seeds for each metric field."""
    mean_kwargs: dict[str, float | None] = {}
    std_kwargs: dict[str, float | None] = {}
    se_kwargs: dict[str, float | None] = {}
    n = 0
    for f in _METRIC_FIELDS:
        vals = _values(metrics, f)
        if vals:
            n = max(n, len(vals))
            mean_val = _agg_one(vals, method)
            mean_kwargs[f] = mean_val
            if len(vals) >= 2:
                stdev = stdstats.stdev(vals)
                std_kwargs[f] = stdev
                se_kwargs[f] = stdev / math.sqrt(len(vals))
            else:
                std_kwargs[f] = 0.0
                se_kwargs[f] = 0.0
        else:
            mean_kwargs[f] = None
            std_kwargs[f] = None
            se_kwargs[f] = None

    return AggregatedMetrics(
        method=method,
        mean=SimulationMetrics(**mean_kwargs),  # type: ignore[arg-type]
        std=SimulationMetrics(**std_kwargs),  # type: ignore[arg-type]
        se=SimulationMetrics(**se_kwargs),  # type: ignore[arg-type]
        n=n,
    )


# ---------------------------------------------------------------------------
# Scalarisation (S)
# ---------------------------------------------------------------------------
def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def scalarize(
    aggregated: AggregatedMetrics,
    weights: ScalarizationWeights,
    *,
    num_candidates: int,
) -> float:
    """Combine the aggregated metrics into a single score F(x).

    Higher is better. Convention:

        F = + w_connected * connected_ratio
            - w_relays    * relay_count / num_candidates
            - w_hops      * mean_hop_count / max_hops_norm
            - w_dist      * mean_distance_to_mobile / dist_norm
            - w_redundancy* redundancy / num_candidates
            - w_latency   * latency / max_latency_norm   (Cooja-only)
            - w_energy    * energy  / max_energy_norm    (Cooja-only)
            + w_throughput* throughput / max_thr_norm    (Cooja-only)

    Normalisers are intentionally constant ("self-normalising" using the
    candidate count, which is what we know structurally). If a required
    metric is missing, raise a :class:`ValueError`.
    """
    m = aggregated.mean

    # Required-metric check
    for req in weights.required_metrics:
        if getattr(m, req, None) is None:
            raise ValueError(
                f"Required metric {req!r} is missing from aggregated metrics; "
                "check the evaluator output or relax weights.required_metrics."
            )

    score = 0.0
    if m.connected_ratio is not None:
        score += weights.w_connected * float(m.connected_ratio)
    if m.relay_count is not None and num_candidates > 0:
        score -= weights.w_relays * (float(m.relay_count) / num_candidates)
    if m.mean_hop_count is not None:
        score -= weights.w_hops * _safe_div(float(m.mean_hop_count), float(num_candidates))
    if m.mean_distance_to_mobile is not None:
        # Use a soft normaliser using num_candidates so the term remains bounded.
        score -= weights.w_dist * _safe_div(float(m.mean_distance_to_mobile), 100.0)
    if m.redundancy is not None and num_candidates > 0:
        score -= weights.w_redundancy * (float(m.redundancy) / num_candidates)
    if m.latency is not None:
        score -= weights.w_latency * _safe_div(float(m.latency), 1000.0)
    if m.energy is not None:
        score -= weights.w_energy * _safe_div(float(m.energy), 1000.0)
    if m.throughput is not None:
        score += weights.w_throughput * _safe_div(float(m.throughput), 1000.0)
    return float(score)


def make_evaluation_result(
    solution: FullSolution,
    per_seed: list[SimulationMetrics],
    weights: ScalarizationWeights,
    *,
    num_candidates: int,
    aggregation_method: str,
    duration_s: float,
) -> EvaluationResult:
    aggregated = aggregate_metrics(per_seed, method=aggregation_method)
    F = scalarize(aggregated, weights, num_candidates=num_candidates)
    return EvaluationResult(
        solution_id=solution.solution_id,
        per_seed=per_seed,
        aggregated=aggregated,
        F=F,
        duration_s=duration_s,
    )
