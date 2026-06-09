"""High-level orchestration: run the full population-estimation experiment.

Pipeline overview:

  1. Load + validate the P2 instance (``io.load_instance``).
  2. Partition Q into ``m`` structural blocks (``partitioning.partition``).
  3. For each block i:
       a. Build H_i^* and H_i^L (``blocks.build_h_star`` / ``build_h_local``).
       b. Generate R complements (``complements.generate_complements``).
       c. For each complement r:
            - Compose x_star = (H_i^*,  x_{-i}^{(r)})
            - Compose x_local= (H_i^L, x_{-i}^{(r)})
            - Evaluate both -> F_star^{(r)}, F_local^{(r)}
       d. Estimate (d_i_hat, sigma_BB_i_hat, n_i) (``estimator.estimate_block``).
  4. Aggregate to global n_hat (``estimator.aggregate_global``).
  5. Persist JSON + CSV (``io.write_json`` / ``write_csv``).
"""

from __future__ import annotations

import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from p2_population_estimator import __version__
from p2_population_estimator.blocks import (
    bernoulli_pi,
    build_h_local,
    build_h_star,
    compose_full_solution,
)
from p2_population_estimator.complements import generate_complements
from p2_population_estimator.config import RESULT_DISCLAIMER
from p2_population_estimator.estimator import aggregate_global, estimate_block
from p2_population_estimator.evaluation.base import BaseEvaluator
from p2_population_estimator.evaluation.cooja import CoojaEvaluator
from p2_population_estimator.evaluation.ssh_pool import SSHPool
from p2_population_estimator.evaluation.surrogate import SurrogateEvaluator
from p2_population_estimator.io import load_instance, write_csv, write_json
from p2_population_estimator.logging_utils import configure as configure_logging
from p2_population_estimator.logging_utils import get_logger
from p2_population_estimator.models import (
    BlockComparisonResult,
    CandidateBlock,
    ExperimentConfig,
    FullSolution,
    P2Instance,
    PopulationEstimateResult,
)
from p2_population_estimator.partitioning import partition, summarise_partition

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_experiment(cfg: ExperimentConfig) -> PopulationEstimateResult:
    """Execute the full pipeline and return the result bundle."""
    t0 = time.perf_counter()

    output_dir = Path(cfg.output_dir)
    _prepare_output_dir(output_dir, force=cfg.force_overwrite)
    configure_logging(log_file=output_dir / "experiment.log")
    log.info("Starting experiment", kv={"output_dir": str(output_dir)})

    instance = load_instance(cfg.instance_path)
    log.info(
        "Loaded instance",
        kv={
            "name": instance.problem.name,
            "num_candidates": len(instance.problem.candidates),
            "num_mobile": len(instance.problem.mobile_nodes),
        },
    )

    blocks = partition(
        instance.problem,
        cfg.partition_method,
        cfg.num_blocks,
        random_seed=cfg.random_seed,
    )
    log.info(
        "Partitioned candidates",
        kv={
            "method": cfg.partition_method,
            "requested_blocks": cfg.num_blocks,
            "actual_blocks": len(blocks),
        },
    )

    evaluator = _build_evaluator(cfg, instance)
    rng = random.Random(cfg.random_seed)

    block_results: list[BlockComparisonResult] = []
    failed_evaluations: list[dict[str, Any]] = []
    warnings: list[str] = []
    m = len(blocks)
    try:
        for block in blocks:
            log.info(
                "Processing block",
                kv={"block_id": block.block_id, "k": block.k},
            )
            br = _process_block(
                block=block,
                m=m,
                instance=instance,
                cfg=cfg,
                evaluator=evaluator,
                rng=rng,
                failed_evaluations=failed_evaluations,
            )
            block_results.append(br)
            warnings.extend(br.warnings)
    finally:
        try:
            evaluator.shutdown()
        except Exception as exc:  # noqa: BLE001
            log.warning("Evaluator shutdown raised", kv={"error": str(exc)})

    global_estimate = aggregate_global(block_results)
    warnings.append(RESULT_DISCLAIMER)

    result = PopulationEstimateResult(
        experiment_config=_config_to_dict(cfg),
        instance_summary=_instance_summary(instance),
        partition_summary=summarise_partition(blocks),
        block_results=block_results,
        global_estimate=global_estimate,
        failed_evaluations=failed_evaluations,
        warnings=warnings,
        reproducibility_info=_reproducibility_info(cfg, instance, time.perf_counter() - t0),
    )

    _persist(result, output_dir, force=cfg.force_overwrite)
    log.info(
        "Experiment finished",
        kv={
            "duration_s": f"{time.perf_counter() - t0:.2f}",
            "n_hat_uniform": global_estimate.get("n_hat_uniform"),
            "n_hat_bernoulli": global_estimate.get("n_hat_bernoulli"),
        },
    )
    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    main_json = output_dir / "population_estimate_result.json"
    main_csv = output_dir / "block_results.csv"
    if (main_json.exists() or main_csv.exists()) and not force:
        raise FileExistsError(
            f"Output dir {output_dir} already contains a previous run. "
            "Pass --force to overwrite or choose a new --output-dir."
        )


def _build_evaluator(cfg: ExperimentConfig, instance: P2Instance) -> BaseEvaluator:
    if cfg.mode == "surrogate":
        return SurrogateEvaluator(
            problem=instance.problem,
            weights=cfg.weights,
            aggregation_method=cfg.aggregation_method,
        )
    if cfg.mode == "cooja":
        if not cfg.ssh_user:
            raise ValueError("Cooja mode requires --ssh-user.")
        if not cfg.ssh_ports:
            raise ValueError("Cooja mode requires at least one --ssh-port.")
        pool = SSHPool(
            host=cfg.ssh_host,
            user=cfg.ssh_user,
            ports=cfg.ssh_ports,
            max_retries=cfg.max_retries,
            password=cfg.ssh_password,
        )
        pool.start()
        return CoojaEvaluator(
            problem=instance.problem,
            weights=cfg.weights,
            pool=pool,
            output_dir=Path(cfg.output_dir),
            remote_workdir_root=PurePosixPath(cfg.remote_workdir),
            command_template=cfg.cooja_command_template,
            aggregation_method=cfg.aggregation_method,
            simulation_timeout=cfg.simulation_timeout,
            simulation_duration_s=cfg.simulation_duration,
            remote_cooja_dir=cfg.remote_cooja_dir,
        )
    raise ValueError(f"Unknown mode: {cfg.mode!r}")


def _process_block(
    *,
    block: CandidateBlock,
    m: int,
    instance: P2Instance,
    cfg: ExperimentConfig,
    evaluator: BaseEvaluator,
    rng: random.Random,
    failed_evaluations: list[dict[str, Any]],
) -> BlockComparisonResult:
    # H_i^* / H_i^L
    h_star = build_h_star(
        cfg.hstar_method,
        block,
        instance.problem,
        external_path=cfg.hstar_external_path,
    )
    h_local = build_h_local(
        cfg.hlocal_method,
        block,
        instance.problem,
        h_star=h_star,
        rng=rng,
    )
    log.info(
        "Built block patterns",
        kv={
            "block_id": block.block_id,
            "s_i_star": h_star.s,
            "s_i_local": h_local.s,
            "k_i": block.k,
        },
    )

    # Complements
    complements = generate_complements(
        cfg.complement_method,
        instance.problem,
        block,
        num_complements=cfg.num_complements,
        rho=cfg.rho,
        rng=rng,
    )

    F_star_samples: list[float] = []
    F_local_samples: list[float] = []
    for r_idx, comp in enumerate(complements):
        x_star = compose_full_solution(
            instance.problem,
            block,
            h_star,
            comp,
            solution_id=f"b{block.block_id}-r{r_idx}-star",
        )
        x_local = compose_full_solution(
            instance.problem,
            block,
            h_local,
            comp,
            solution_id=f"b{block.block_id}-r{r_idx}-local",
        )
        F_star = _safe_evaluate(
            evaluator, x_star, cfg.seeds, failed_evaluations
        )
        F_local = _safe_evaluate(
            evaluator, x_local, cfg.seeds, failed_evaluations
        )
        if F_star is None or F_local is None:
            continue
        F_star_samples.append(F_star)
        F_local_samples.append(F_local)

    return estimate_block(
        block_id=block.block_id,
        k_i=block.k,
        m=m,
        alpha=cfg.alpha,
        h_star=h_star,
        F_star=F_star_samples,
        F_local=F_local_samples,
        rho=cfg.rho,
    )


def _safe_evaluate(
    evaluator: BaseEvaluator,
    solution: FullSolution,
    seeds: list[int],
    failed_evaluations: list[dict[str, Any]],
) -> Optional[float]:
    try:
        res = evaluator.evaluate(solution, seeds)
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Evaluator raised",
            kv={"solution_id": solution.solution_id, "error": str(exc)},
        )
        failed_evaluations.append(
            {
                "solution_id": solution.solution_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return None
    if res.status == "failed":
        failed_evaluations.append(
            {
                "solution_id": solution.solution_id,
                "error": res.error or "unknown",
            }
        )
        return None
    return res.F


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _persist(result: PopulationEstimateResult, output_dir: Path, *, force: bool) -> None:
    out_json = output_dir / "population_estimate_result.json"
    out_csv = output_dir / "block_results.csv"
    write_json(out_json, _result_to_dict(result), force=force)
    write_csv(out_csv, _block_rows(result.block_results), force=force)


def _result_to_dict(r: PopulationEstimateResult) -> dict[str, Any]:
    return {
        "experiment_config": r.experiment_config,
        "instance_summary": r.instance_summary,
        "partition_summary": r.partition_summary,
        "block_results": [_block_result_to_dict(b) for b in r.block_results],
        "global_estimate": r.global_estimate,
        "failed_evaluations": r.failed_evaluations,
        "warnings": r.warnings,
        "reproducibility_info": r.reproducibility_info,
    }


def _block_result_to_dict(b: BlockComparisonResult) -> dict[str, Any]:
    d = asdict(b)
    # Truncate huge sample lists in the JSON if they ever become enormous
    return d


def _block_rows(blocks: list[BlockComparisonResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for b in blocks:
        rows.append(
            {
                "block_id": b.block_id,
                "k_i": b.k_i,
                "s_i_star": b.s_i_star,
                "alpha": b.alpha,
                "pi_i_star": b.pi_i_star,
                "d_i_hat": b.d_i_hat,
                "sigma_BB_i_hat": b.sigma_BB_i_hat,
                "n_i_uniform": b.n_i_uniform,
                "n_i_uniform_ceil": b.n_i_uniform_ceil,
                "n_i_bernoulli": b.n_i_bernoulli,
                "n_i_bernoulli_ceil": b.n_i_bernoulli_ceil,
                "status": b.status,
            }
        )
    return rows


def _instance_summary(instance: P2Instance) -> dict[str, Any]:
    p = instance.problem
    return {
        "name": p.name,
        "num_candidates": len(p.candidates),
        "num_mobile_nodes": len(p.mobile_nodes),
        "radius_of_reach": p.radius_of_reach,
        "radius_of_inter": p.radius_of_inter,
        "region": [p.region.xmin, p.region.ymin, p.region.xmax, p.region.ymax],
        "sink": [p.sink.x, p.sink.y],
        "source_path": instance.source_path,
    }


def _config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    d = asdict(cfg)
    # ScalarizationWeights -> dict
    if isinstance(d.get("weights"), dict):
        pass
    return d


def _reproducibility_info(
    cfg: ExperimentConfig, instance: P2Instance, duration_s: float
) -> dict[str, Any]:
    instance_hash = _hash_file(instance.source_path) if instance.source_path else None
    return {
        "package_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "duration_s": duration_s,
        "random_seed": cfg.random_seed,
        "seeds_used": list(cfg.seeds),
        "instance_path": instance.source_path,
        "instance_sha256": instance_hash,
    }


def _hash_file(path: str) -> str:
    import hashlib

    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
