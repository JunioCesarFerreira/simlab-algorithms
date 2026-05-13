"""Surrogate evaluator: a Cooja-free, deterministic structural evaluator.

It computes:

  - relay_count
  - connected_ratio       (fraction of (mobile, time) pairs reachable to sink)
  - mean_hop_count        (over reachable pairs)
  - mean_distance_to_mobile (mean closest-relay-or-sink distance across pairs)
  - redundancy            (mean intra-cluster degree, normalised by num candidates)

The surrogate is deterministic for a given (solution, problem, time_samples)
triple. Seeds are accepted for interface parity, but the surrogate is
itself noise-free.
"""

from __future__ import annotations

import math
import time

from p2_population_estimator.config import SURROGATE_TIME_SAMPLES
from p2_population_estimator.evaluation.base import (
    BaseEvaluator,
    make_evaluation_result,
)
from p2_population_estimator.geometry import (
    build_relay_graph,
    euclidean,
    mobile_reachable_via_relay,
    sample_all_mobiles,
    shortest_hops_to_sink,
)
from p2_population_estimator.models import (
    EvaluationResult,
    FullSolution,
    P2Problem,
    ScalarizationWeights,
    SimulationMetrics,
)


class SurrogateEvaluator(BaseEvaluator):
    name = "surrogate"

    def __init__(
        self,
        problem: P2Problem,
        weights: ScalarizationWeights,
        *,
        aggregation_method: str = "mean_with_std",
        time_samples: int = SURROGATE_TIME_SAMPLES,
    ):
        self.problem = problem
        self.weights = weights
        self.aggregation_method = aggregation_method
        self.time_samples = time_samples
        # Pre-sample mobile trajectories; reused across solutions.
        self._mobile_samples = sample_all_mobiles(problem, time_samples)

    # ------------------------------------------------------------------ #
    # BaseEvaluator
    # ------------------------------------------------------------------ #
    def evaluate(self, solution: FullSolution, seeds: list[int]) -> EvaluationResult:
        t0 = time.perf_counter()
        seeds = seeds or [0]
        # Surrogate is deterministic; we still run once per seed to keep the
        # interface symmetric and to register noise = 0 in variance estimates.
        metrics_one = self._evaluate_once(solution)
        per_seed = [metrics_one for _ in seeds]
        duration = time.perf_counter() - t0
        return make_evaluation_result(
            solution=solution,
            per_seed=per_seed,
            weights=self.weights,
            num_candidates=len(self.problem.candidates),
            aggregation_method=self.aggregation_method,
            duration_s=duration,
        )

    # ------------------------------------------------------------------ #
    # Core computation
    # ------------------------------------------------------------------ #
    def _evaluate_once(self, solution: FullSolution) -> SimulationMetrics:
        problem = self.problem
        cands = problem.candidates
        R = problem.radius_of_reach
        sink = problem.sink

        selected = [j for j, b in enumerate(solution.bits) if b]
        relay_count = len(selected)

        adj = build_relay_graph(cands, sink, R, selected)
        hops_from_sink = shortest_hops_to_sink(adj)
        # `hops_from_sink` excludes -1's own entry from selected; ensure we use
        # the dict as-is (it contains -1: 0).

        # If there are no mobile nodes, define a degenerate score that still
        # rewards being connectivity-compatible.
        all_samples = [pt for traj in self._mobile_samples for pt in traj]
        total_pairs = len(all_samples)
        connected = 0
        sum_hops = 0
        sum_dist = 0.0
        for pos in all_samples:
            reachable, h, dmin = mobile_reachable_via_relay(
                pos, cands, sink, R, hops_from_sink
            )
            sum_dist += dmin
            if reachable:
                connected += 1
                sum_hops += h
        if total_pairs > 0:
            connected_ratio = connected / total_pairs
            mean_dist = sum_dist / total_pairs
            mean_hops = (sum_hops / connected) if connected > 0 else float("nan")
        else:
            # No mobile nodes. Treat the "fraction" as 1.0 when the sink itself
            # is reachable from the network (degenerate but well-defined).
            connected_ratio = 1.0 if relay_count == 0 or any(j for j in hops_from_sink if j != -1) else 1.0
            mean_dist = 0.0
            mean_hops = 0.0

        if math.isnan(mean_hops):
            mean_hops_val: float = float(len(cands))  # penalise unreachable
        else:
            mean_hops_val = mean_hops

        redundancy = self._redundancy(selected)

        return SimulationMetrics(
            relay_count=relay_count,
            connected_ratio=float(connected_ratio),
            mean_hop_count=float(mean_hops_val),
            mean_distance_to_mobile=float(mean_dist),
            redundancy=float(redundancy),
        )

    def _redundancy(self, selected: list[int]) -> float:
        """Mean pairwise overlap: how many relays sit within reach of each other.

        Returns the average degree (in the selected sub-graph), which roughly
        proxies redundancy. Higher = more redundant, fewer = more "spread".
        """
        if len(selected) <= 1:
            return 0.0
        R = self.problem.radius_of_reach
        cands = self.problem.candidates
        total_degree = 0
        for i, a in enumerate(selected):
            for b in selected[i + 1:]:
                if euclidean(cands[a], cands[b]) <= R:
                    total_degree += 2
        return total_degree / len(selected)
