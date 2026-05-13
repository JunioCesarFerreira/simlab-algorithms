"""Cooja-backed evaluator.

This module is intentionally a *thin adapter*: the heavy lifting (SSH,
queueing, retries) lives in :mod:`ssh_pool`, and the metric parsing lives in
:mod:`parser`. Customisation hooks are exposed as small methods that you can
override for your specific firmware/simulation glue:

  - ``generate_csc_file``
  - ``generate_positions_file``
  - ``generate_config_file``
  - ``generate_firmware_config``

The default implementations write **placeholder** files that are correct in
structure but not necessarily ready for your firmware. Adapt them in a
subclass or by passing a ``file_generator`` callable.

If ``paramiko`` is not installed, instantiating this evaluator raises a
helpful ``RuntimeError`` *only when ``evaluate`` is actually called*; the
class can be imported in tests without paramiko.
"""

from __future__ import annotations

import json
import time
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

from p2_population_estimator.evaluation.base import (
    BaseEvaluator,
    make_evaluation_result,
)
from p2_population_estimator.evaluation.parser import parse_log_files
from p2_population_estimator.evaluation.ssh_pool import (
    SSHPool,
    SimulationTask,
    TaskResult,
    new_task_id,
)
from p2_population_estimator.logging_utils import get_logger
from p2_population_estimator.models import (
    EvaluationResult,
    FullSolution,
    P2Problem,
    ScalarizationWeights,
    SimulationMetrics,
)

log = get_logger(__name__)


FileGenerator = Callable[[FullSolution, int, Path, P2Problem], list[Path]]


class CoojaEvaluator(BaseEvaluator):
    """Evaluate one solution at a time using a pool of remote Cooja containers."""

    name = "cooja"

    def __init__(
        self,
        problem: P2Problem,
        weights: ScalarizationWeights,
        *,
        pool: SSHPool,
        output_dir: Path,
        remote_workdir_root: PurePosixPath,
        command_template: str,
        aggregation_method: str = "mean_with_std",
        simulation_timeout: int = 900,
        file_generator: Optional[FileGenerator] = None,
    ):
        self.problem = problem
        self.weights = weights
        self.pool = pool
        self.output_dir = Path(output_dir)
        self.remote_workdir_root = remote_workdir_root
        self.command_template = command_template
        self.aggregation_method = aggregation_method
        self.simulation_timeout = simulation_timeout
        self.file_generator = file_generator or _default_file_generator

    # ------------------------------------------------------------------ #
    def evaluate(self, solution: FullSolution, seeds: list[int]) -> EvaluationResult:
        t0 = time.perf_counter()
        if not seeds:
            raise ValueError("Cooja evaluator requires at least one seed")

        sol_dir = self.output_dir / "evaluations" / solution.solution_id
        sol_dir.mkdir(parents=True, exist_ok=True)

        # Build one task per seed; submit to the pool.
        submitted: list[SimulationTask] = []
        seed_to_task_id: dict[int, str] = {}
        for seed in seeds:
            task_id = new_task_id(f"sol-{solution.solution_id}-seed-{seed}")
            seed_to_task_id[seed] = task_id
            local_wd = sol_dir / str(seed)
            local_wd.mkdir(parents=True, exist_ok=True)
            remote_wd = self.remote_workdir_root / f"{solution.solution_id}-{seed}"
            task = SimulationTask(
                task_id=task_id,
                solution_id=solution.solution_id,
                seed=seed,
                local_workdir=local_wd,
                remote_workdir=remote_wd,
                prepare_local=lambda workdir, _sol=solution, _seed=seed: self.file_generator(
                    _sol, _seed, workdir, self.problem
                ),
                command_template=self.command_template,
                placeholders={"seed": str(seed)},
                timeout_s=self.simulation_timeout,
            )
            submitted.append(task)
            self.pool.submit(task)

        # Wait only for *this evaluation's* tasks. We can't `join` the whole
        # queue because other evaluations may share the same pool; instead we
        # poll the result list.
        target_ids = {t.task_id for t in submitted}
        results_for_self: dict[str, TaskResult] = {}
        while target_ids - set(results_for_self):
            time.sleep(0.2)
            with self.pool._lock:  # noqa: SLF001 — explicit cross-module access
                snapshot = list(self.pool.results)
            for r in snapshot:
                if r.task_id in target_ids and r.task_id not in results_for_self:
                    results_for_self[r.task_id] = r

        # Collect per-seed metrics
        per_seed: list[SimulationMetrics] = []
        any_failed = False
        for seed in seeds:
            tid = seed_to_task_id[seed]
            res = results_for_self[tid]
            if res.status == "ok" and res.log_path is not None:
                try:
                    metrics = parse_log_files([str(res.log_path)])
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failed to parse Cooja log",
                        kv={"solution_id": solution.solution_id, "seed": seed, "error": str(exc)},
                    )
                    any_failed = True
                    metrics = SimulationMetrics()
            else:
                log.error(
                    "Cooja simulation task failed",
                    kv={
                        "solution_id": solution.solution_id,
                        "seed": seed,
                        "error": res.error,
                    },
                )
                any_failed = True
                metrics = SimulationMetrics()
            per_seed.append(metrics)

        eval_result = make_evaluation_result(
            solution=solution,
            per_seed=per_seed,
            weights=self.weights,
            num_candidates=len(self.problem.candidates),
            aggregation_method=self.aggregation_method,
            duration_s=time.perf_counter() - t0,
        )
        if any_failed:
            eval_result.status = "failed"
            eval_result.error = "At least one seed failed; metrics may be missing."
        return eval_result

    def shutdown(self) -> None:
        self.pool.shutdown()


# ---------------------------------------------------------------------------
# Default file generator (placeholder, intentionally minimal)
# ---------------------------------------------------------------------------
def _default_file_generator(
    solution: FullSolution, seed: int, workdir: Path, problem: P2Problem
) -> list[Path]:
    """Write a minimal set of input files that document the solution.

    The default generator emits:

    * ``solution.json``  — the binary mask and metadata
    * ``positions.txt``  — selected relay coordinates
    * ``config.json``    — simulation parameters
    * ``simulation.csc`` — placeholder Cooja file referencing the others

    This is **not** a complete Cooja simulation file; you are expected to
    override ``file_generator`` with a real generator tailored to your
    firmware setup. The default exists so that an end-to-end "dry run" with
    a mocked SSH backend can be performed.
    """
    files: list[Path] = []

    sol_path = workdir / "solution.json"
    sol_path.write_text(
        json.dumps(
            {
                "solution_id": solution.solution_id,
                "bits": solution.bits,
                "relay_count": solution.relay_count,
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    files.append(sol_path)

    pos_path = workdir / "positions.txt"
    selected = [
        (j, problem.candidates[j].x, problem.candidates[j].y)
        for j, b in enumerate(solution.bits)
        if b
    ]
    pos_lines = ["# idx\tx\ty"]
    pos_lines += [f"{idx}\t{x}\t{y}" for idx, x, y in selected]
    pos_path.write_text("\n".join(pos_lines) + "\n", encoding="utf-8")
    files.append(pos_path)

    cfg_path = workdir / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "radius_of_reach": problem.radius_of_reach,
                "radius_of_inter": problem.radius_of_inter,
                "sink": [problem.sink.x, problem.sink.y],
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    files.append(cfg_path)

    csc_path = workdir / "simulation.csc"
    csc_path.write_text(
        (
            "<!-- Placeholder Cooja simulation file. -->\n"
            "<!-- Override CoojaEvaluator.file_generator for your real setup. -->\n"
            f"<!-- solution_id={solution.solution_id} seed={seed} -->\n"
        ),
        encoding="utf-8",
    )
    files.append(csc_path)
    return files
