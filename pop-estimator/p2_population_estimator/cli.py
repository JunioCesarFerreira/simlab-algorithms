"""Command-line interface — JSON-driven.

Usage::

    # Print default config and save it
    p2-popest --dump-config > my_run.json

    # Edit my_run.json, then run
    p2-popest my_run.json

    # Override output directory and force overwrite from the command line
    p2-popest my_run.json --output-dir results/run_002 --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Firmware bundled with the package (pop-estimator/firmware/rpl-udp-csma/)
_BUNDLED_FIRMWARE_DIR = Path(__file__).parent.parent / "firmware" / "rpl-udp-csma"

from p2_population_estimator.experiment import run_experiment
from p2_population_estimator.logging_utils import configure as configure_logging
from p2_population_estimator.models import ExperimentConfig, ScalarizationWeights


# ---------------------------------------------------------------------------
# Default configuration (all keys match ExperimentConfig field names)
# ---------------------------------------------------------------------------
def _default_config() -> dict[str, Any]:
    return {
        # Required — must be set by the user
        "instance_path": "path/to/instance.json",
        "output_dir": "results/run_001",

        # Mode: "surrogate" (no external simulator) or "cooja" (SSH to Cooja containers)
        "mode": "surrogate",

        # Partition / blocks
        "partition_method": "grid",       # "grid" | "kmeans" | "radial_to_sink"
        "num_blocks": 8,
        "hstar_method": "structural_greedy",   # "structural_greedy" | "dense_local" | "external"
        "hstar_external_path": None,
        "hlocal_method": "deceptive_low_cost", # "deceptive_low_cost" | "redundant_local" | "far_from_sink" | "random_competitor"

        # Complements
        "complement_method": "bernoulli",  # "bernoulli" | "feasible_repair" | "population_sample"
        "num_complements": 30,
        "rho": 0.2,

        # Statistical
        "alpha": 0.05,
        "aggregation_method": "mean_with_std",  # "mean" | "median" | "trimmed_mean" | "mean_with_std"

        # Reproducibility
        "seeds": [42],
        "random_seed": 42,

        # Scalarisation weights
        "weights": {
            "w_connected": 1.0,
            "w_relays": 0.05,
            "w_hops": 0.05,
            "w_dist": 0.05,
            "w_redundancy": 0.02,
            "w_latency": 0.0,
            "w_energy": 0.0,
            "w_throughput": 0.0,
            "required_metrics": ["connected_ratio", "relay_count"],
        },

        # Cooja / SSH (only used when mode == "cooja")
        "ssh_host": "localhost",
        "ssh_user": "root",
        "ssh_password": "root",
        "ssh_ports": [2231, 2232, 2233, 2234, 2235, 2236],
        "remote_workdir": "/tmp/popest",
        "simulation_timeout": 900,
        "simulation_duration": 180,
        "remote_cooja_dir": "/opt/contiki-ng/tools/cooja",
        "cooja_command_template": (
            "cd {remote_cooja_dir} && "
            "/opt/java/openjdk/bin/java --enable-preview "
            "-Xms1g -Xmx2g "
            "-jar build/libs/cooja.jar --no-gui {simulation_file}"
        ),
        "max_retries": 2,
        "firmware_local_dir": str(_BUNDLED_FIRMWARE_DIR),

        # Overwrite protection (can also be set via --force flag)
        "force_overwrite": False,
    }


# ---------------------------------------------------------------------------
# Build ExperimentConfig from a (possibly partial) dict
# ---------------------------------------------------------------------------
def _build_config(raw: dict[str, Any], *, force_overwrite: bool = False) -> ExperimentConfig:
    defaults = _default_config()
    merged: dict[str, Any] = {**defaults, **raw}

    w_raw: dict[str, Any] = merged.get("weights", {})
    w_defaults = defaults["weights"]
    w_merged = {**w_defaults, **w_raw}
    weights = ScalarizationWeights(
        w_connected=float(w_merged["w_connected"]),
        w_relays=float(w_merged["w_relays"]),
        w_hops=float(w_merged["w_hops"]),
        w_dist=float(w_merged["w_dist"]),
        w_redundancy=float(w_merged["w_redundancy"]),
        w_latency=float(w_merged["w_latency"]),
        w_energy=float(w_merged["w_energy"]),
        w_throughput=float(w_merged["w_throughput"]),
        required_metrics=tuple(w_merged["required_metrics"]),
    )

    return ExperimentConfig(
        instance_path=merged["instance_path"],
        output_dir=merged["output_dir"],
        mode=merged["mode"],
        partition_method=merged["partition_method"],
        num_blocks=int(merged["num_blocks"]),
        num_complements=int(merged["num_complements"]),
        alpha=float(merged["alpha"]),
        rho=float(merged["rho"]),
        seeds=[int(s) for s in merged["seeds"]],
        random_seed=int(merged["random_seed"]),
        hstar_method=merged["hstar_method"],
        hlocal_method=merged["hlocal_method"],
        complement_method=merged["complement_method"],
        aggregation_method=merged["aggregation_method"],
        weights=weights,
        ssh_host=merged["ssh_host"],
        ssh_user=merged["ssh_user"],
        ssh_password=merged.get("ssh_password"),
        ssh_ports=[int(p) for p in merged["ssh_ports"]],
        remote_workdir=merged["remote_workdir"],
        simulation_timeout=int(merged["simulation_timeout"]),
        simulation_duration=int(merged["simulation_duration"]),
        remote_cooja_dir=merged["remote_cooja_dir"],
        cooja_command_template=merged["cooja_command_template"],
        max_retries=int(merged["max_retries"]),
        hstar_external_path=merged.get("hstar_external_path"),
        hlocal_external_path=merged.get("hlocal_external_path"),
        firmware_local_dir=merged.get("firmware_local_dir"),
        force_overwrite=force_overwrite or bool(merged.get("force_overwrite", False)),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="p2-popest",
        description=(
            "Estimate the NSGA-III population size required for Problem P2. "
            "All parameters are read from a JSON config file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  p2-popest --dump-config > my_run.json   # generate default config\n"
            "  p2-popest my_run.json                   # run experiment\n"
        ),
    )
    p.add_argument(
        "config",
        nargs="?",
        metavar="CONFIG_JSON",
        help="Path to JSON config file.",
    )
    p.add_argument(
        "--dump-config",
        action="store_true",
        help="Print the default config JSON to stdout and exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.dump_config:
        print(json.dumps(_default_config(), indent=2))
        return 0

    if not args.config:
        _build_parser().error("CONFIG_JSON is required (or use --dump-config to see defaults)")

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"error: config file not found: {config_path}", file=sys.stderr)
        return 1

    raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))

    configure_logging()
    cfg = _build_config(raw)

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
        f"  {cfg.output_dir}/population_estimate_result.json\n"
        f"  {cfg.output_dir}/block_results.csv\n",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
