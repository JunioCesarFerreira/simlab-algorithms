"""Command-line interface.

Example (surrogate)::

    python -m p2_population_estimator.cli \\
        --instance ind2.json \\
        --output-dir results/p2_estimation_run_001 \\
        --mode surrogate \\
        --partition-method grid \\
        --num-blocks 8 \\
        --num-complements 30 \\
        --alpha 0.05 \\
        --rho 0.20 \\
        --seeds 336157 667370 35239 873465 \\
        --random-seed 42

Example (cooja)::

    python -m p2_population_estimator.cli \\
        --instance ind2.json \\
        --output-dir results/p2_cooja_run_001 \\
        --mode cooja \\
        --ssh-host localhost \\
        --ssh-user ${USER} \\
        --ssh-ports 2231 2232 2233 2234 2235 2236 \\
        --partition-method grid \\
        --num-blocks 8 \\
        --num-complements 30 \\
        --alpha 0.05 \\
        --rho 0.20 \\
        --seeds 336157 667370 35239 873465 \\
        --simulation-timeout 900 \\
        --random-seed 42
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from p2_population_estimator.experiment import run_experiment
from p2_population_estimator.logging_utils import configure as configure_logging
from p2_population_estimator.models import ExperimentConfig, ScalarizationWeights


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="p2-popest",
        description=(
            "Estimate the NSGA-III population size required for Problem P2 "
            "using a block-decomposition / gambler-ruin heuristic."
        ),
    )

    p.add_argument("--instance", required=True, help="Path to the P2 instance JSON.")
    p.add_argument("--output-dir", required=True, help="Where to write outputs.")
    p.add_argument(
        "--mode",
        choices=("surrogate", "cooja"),
        required=True,
        help="surrogate = no external simulator; cooja = run via SSH.",
    )

    # Partition / blocks
    p.add_argument(
        "--partition-method",
        choices=("grid", "kmeans", "radial_to_sink"),
        default="grid",
    )
    p.add_argument("--num-blocks", type=int, default=8)
    p.add_argument(
        "--hstar-method",
        choices=("structural_greedy", "dense_local", "external"),
        default="structural_greedy",
    )
    p.add_argument("--hstar-external-path", default=None)
    p.add_argument(
        "--hlocal-method",
        choices=("deceptive_low_cost", "redundant_local", "far_from_sink", "random_competitor"),
        default="deceptive_low_cost",
    )

    # Complements
    p.add_argument(
        "--complement-method",
        choices=("bernoulli", "feasible_repair", "population_sample"),
        default="bernoulli",
    )
    p.add_argument("--num-complements", type=int, default=30)
    p.add_argument("--rho", type=float, default=0.2)

    # Statistical
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--aggregation-method",
        choices=("mean", "median", "trimmed_mean", "mean_with_std"),
        default="mean_with_std",
    )

    # Reproducibility
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--random-seed", type=int, default=42)

    # Cooja
    p.add_argument("--ssh-host", default="localhost")
    p.add_argument("--ssh-user", default="")
    p.add_argument(
        "--ssh-ports", type=int, nargs="+", default=[2231, 2232, 2233, 2234, 2235, 2236]
    )
    p.add_argument("--remote-workdir", default="/tmp/popest")
    p.add_argument("--simulation-timeout", type=int, default=900)
    p.add_argument(
        "--cooja-command-template",
        default="java -Xms4g -Xmx4g -jar /opt/cooja/cooja.jar --no-gui {simulation_file}",
    )
    p.add_argument("--max-retries", type=int, default=2)

    # Output safety
    p.add_argument("--force", action="store_true", help="Overwrite existing output.")

    # Scalarisation weights
    grp = p.add_argument_group("scalarisation weights")
    grp.add_argument("--w-connected", type=float, default=1.0)
    grp.add_argument("--w-relays", type=float, default=0.05)
    grp.add_argument("--w-hops", type=float, default=0.05)
    grp.add_argument("--w-dist", type=float, default=0.05)
    grp.add_argument("--w-redundancy", type=float, default=0.02)
    grp.add_argument("--w-latency", type=float, default=0.0)
    grp.add_argument("--w-energy", type=float, default=0.0)
    grp.add_argument("--w-throughput", type=float, default=0.0)
    grp.add_argument(
        "--required-metrics",
        nargs="+",
        default=["connected_ratio", "relay_count"],
        help="Metrics that must be present in the aggregated output; "
        "evaluation fails with a clear error if any is missing.",
    )

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging()

    weights = ScalarizationWeights(
        w_connected=args.w_connected,
        w_relays=args.w_relays,
        w_hops=args.w_hops,
        w_dist=args.w_dist,
        w_redundancy=args.w_redundancy,
        w_latency=args.w_latency,
        w_energy=args.w_energy,
        w_throughput=args.w_throughput,
        required_metrics=tuple(args.required_metrics),
    )

    cfg = ExperimentConfig(
        instance_path=args.instance,
        output_dir=args.output_dir,
        mode=args.mode,
        partition_method=args.partition_method,
        num_blocks=args.num_blocks,
        num_complements=args.num_complements,
        alpha=args.alpha,
        rho=args.rho,
        seeds=list(args.seeds),
        random_seed=args.random_seed,
        hstar_method=args.hstar_method,
        hlocal_method=args.hlocal_method,
        complement_method=args.complement_method,
        aggregation_method=args.aggregation_method,
        weights=weights,
        ssh_host=args.ssh_host,
        ssh_user=args.ssh_user,
        ssh_ports=list(args.ssh_ports),
        remote_workdir=args.remote_workdir,
        simulation_timeout=args.simulation_timeout,
        cooja_command_template=args.cooja_command_template,
        max_retries=args.max_retries,
        hstar_external_path=args.hstar_external_path,
        force_overwrite=args.force,
    )

    result = run_experiment(cfg)
    g = result.global_estimate
    print(
        "\n=== Population estimate (HEURISTIC) ===\n"
        f"n_hat_uniform   = {g.get('n_hat_uniform')}\n"
        f"n_hat_bernoulli = {g.get('n_hat_bernoulli')}\n"
        f"valid_blocks    = {g.get('num_valid_blocks')}\n"
        f"invalid_blocks  = {g.get('num_invalid_blocks')}\n"
        f"most_difficult  = block {g.get('most_difficult_block_id')}\n"
        f"\nOutputs:\n"
        f"  {args.output_dir}/population_estimate_result.json\n"
        f"  {args.output_dir}/block_results.csv\n",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
