"""P2 MILP Parameter Sweep — main orchestrator.

Usage
-----
    python sweep.py [--preset quick|medium|full] [--instance PATH]
                    [--results-dir PATH] [--time-limit SEC]
                    [--no-plots] [--verbose]

The script loops over (C0, kdecay, B) as defined in config.py (or overridden
by --preset), solves the mobile-coverage MILP for each combination, and writes:

results/
  milp_run_summary.json          -- one record per MILP solve
  pic_candidates.png             -- instance overview (generated once)
  <chromosome>/
    output.json                  -- simulation-ready JSON for unique solutions
    pic_installed_graph.png      -- static relay graph for this solution

Stopping rules (per k_decay block, matching wsn-design-space-exploration):
  - If a run returns INFEASIBLE, skip remaining B values for that (C0, k_decay)
  - If a run returns any other non-optimal status, break the B loop
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make sure the package root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from milp.instance_adapter import load_p2_instance, adapt
from milp.model import precompute_topology, solve, binary_chromosome, detect_solver_name
from utils.plot_utils import (
    plot_candidates_and_paths,
    plot_installed_graph,
    plot_sweep_summary,
)

# ---------------------------------------------------------------------------
# Status names that indicate a usable solution was found
# (solver-agnostic: the model maps every backend's codes to these strings)
# ---------------------------------------------------------------------------
_GOOD_STATUS_NAMES  = {"OPTIMAL", "SUBOPTIMAL", "TIME_LIMIT"}
_INFEASIBLE_NAME    = "INFEASIBLE"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_output_json(inp, y_val, run_record: dict) -> dict:
    """Assemble the simulation-ready JSON for one unique solution."""
    installed = [j for j, v in y_val.items() if v > 0.5]

    fixed_motes_out = [{
        "position": [float(inp.p_sink[0]), float(inp.p_sink[1])],
        "name": "root",
        "sourceCode": "node.c",
    }]
    for idx, j_node in enumerate(installed, start=1):
        pos = inp.p_cand[j_node]
        fixed_motes_out.append({
            "position": [float(pos[0]), float(pos[1])],
            "name": f"node{idx}",
            "sourceCode": "node.c",
        })

    mobile_motes_out = []
    for name in inp.mob_names:
        # Reconstruct minimal mobile entry (trajectory is regenerated at runtime)
        mobile_motes_out.append({"name": name})

    return {
        "simulationModel": {
            "duration":       inp.duration,
            "radiusOfReach":  inp.R_comm,
            "radiusOfInter":  inp.R_interf,
            "region":         inp.region,
            "simulationElements": {
                "fixedMotes":  fixed_motes_out,
                "mobileMotes": mobile_motes_out,
            },
        },
        "milpParameters": {
            "C0":       run_record["C0"],
            "k_decay":  run_record["k_decay"],
            "B":        run_record["B"],
            "w_install": run_record["w_install"],
        },
        "milpSolve": {
            "status":          run_record["status"],
            "status_name":     run_record["status_name"],
            "objective_value": run_record["objective_value"],
            "runtime_seconds": run_record["runtime_seconds"],
            "mip_gap":         run_record["mip_gap"],
            "installed_nodes": run_record["installed_nodes"],
            "variables":       run_record["variables"],
            "constraints":     run_record["constraints"],
        },
    }


def _save_summary(records: list[dict], summary_path: Path) -> None:
    # y_val has tuple keys that are not JSON-serialisable; strip it before saving.
    # The chromosome column already encodes the full installation vector.
    serialisable = [{k: v for k, v in r.items() if k != "y_val"} for r in records]
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    instance_path: Path,
    results_dir: Path,
    sweep_params: dict,
    w_install: float,
    time_limit: float | None,
    make_plots: bool,
    verbose: bool,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "milp_run_summary.json"

    # --- Load and adapt instance ---
    print(f"Loading instance: {instance_path}")
    raw      = load_p2_instance(instance_path)
    inp      = adapt(raw)
    print(
        f"  N={len(inp.J)} candidates, M={len(inp.mob_names)} mobiles, "
        f"T={inp.T} timesteps, R_comm={inp.R_comm}"
    )

    # --- Precompute topology (done once, reused for all sweep runs) ---
    sample_step = cfg.TIME_SAMPLE_STEP
    t_sampled   = len(range(1, inp.T + 1, sample_step))
    print(
        f"Precomputing network topology  "
        f"(sample_step={sample_step}, {t_sampled}/{inp.T} timesteps)...",
        flush=True,
    )
    t_topo = time.time()
    topo = precompute_topology(inp, sample_step=sample_step)
    total_edges = sum(len(v) for v in topo.E_t.values())
    print(f"  Done in {time.time()-t_topo:.1f}s  "
          f"({total_edges} directed edge-timestep pairs)")

    solver_name = detect_solver_name()
    print(f"  Solver: {solver_name}\n")

    # --- Instance overview plot (once) ---
    if make_plots:
        plot_candidates_and_paths(
            J=inp.J, p_cand=inp.p_cand, p_sink=inp.p_sink,
            R_comm=inp.R_comm, mob_names=inp.mob_names,
            r_mobile_fns=inp.r_mobile_fns, T=inp.T, region=inp.region,
            out_path=str(results_dir / "pic_candidates.png"),
        )
        print("  Saved pic_candidates.png")

    C0_vals     = sweep_params["C0"]
    kdecay_vals = sweep_params["kdecay"]
    B_vals      = sweep_params["B"]
    total       = len(C0_vals) * len(kdecay_vals) * len(B_vals)
    print(f"Starting sweep: {total} runs  "
          f"({len(C0_vals)} C0 × {len(kdecay_vals)} kdecay × {len(B_vals)} B)\n")

    seen_chromosomes: set[str] = set()
    run_records:      list[dict] = []
    run_idx = 0
    t_start = time.time()

    for C0 in C0_vals:
        for kdecay in kdecay_vals:
            for B in B_vals:
                run_idx += 1
                elapsed = time.time() - t_start
                print(f"[{run_idx}/{total}]  C0={C0:>5}  kdecay={kdecay}  B={B:>3} "
                      f"  (elapsed {elapsed:.0f}s)", end="  ", flush=True)

                record = solve(
                    inp, topo,
                    C0=C0, kdecay=kdecay, B=B,
                    w_install=w_install,
                    time_limit=time_limit,
                    verbose=verbose,
                )
                run_records.append(record)
                print(f"→ {record['status_name']}  "
                      f"relays={record['installed_nodes']}  "
                      f"t={record['runtime_seconds']:.1f}s")

                # Persist summary after every run (fault-tolerant)
                _save_summary(run_records, summary_path)

                # --- Save unique solution ---
                if record["y_val"] is not None:
                    chrom = record["chromosome"]
                    if chrom not in seen_chromosomes:
                        seen_chromosomes.add(chrom)
                        sol_dir = results_dir / chrom
                        sol_dir.mkdir(exist_ok=True)

                        out_json = _build_output_json(inp, record["y_val"], record)
                        with open(sol_dir / "output.json", "w", encoding="utf-8") as fh:
                            json.dump(out_json, fh, ensure_ascii=False, indent=2)

                        if make_plots:
                            installed = [j for j, v in record["y_val"].items()
                                         if v > 0.5]
                            plot_installed_graph(
                                installed=installed,
                                p_cand=inp.p_cand,
                                p_sink=inp.p_sink,
                                R_comm=inp.R_comm,
                                region=inp.region,
                                out_path=str(sol_dir / "pic_installed_graph.png"),
                            )

                # --- Stopping rules (per wsn-dse convention) ---
                sname = record["status_name"]
                if sname == _INFEASIBLE_NAME:
                    print(f"    ↳ INFEASIBLE at B={B}; skipping remaining B "
                          f"for (C0={C0}, kdecay={kdecay})")
                    break
                if sname not in _GOOD_STATUS_NAMES:
                    print(f"    ↳ {sname}; skipping remaining B "
                          f"for (C0={C0}, kdecay={kdecay})")
                    break

    # --- Summary plot ---
    if make_plots:
        plot_sweep_summary(
            records=run_records,
            out_path=str(results_dir / "sweep_summary.png"),
        )
        print("Saved sweep_summary.png")

    total_time = time.time() - t_start
    solved     = sum(1 for r in run_records if r["installed_nodes"] is not None)
    unique_sol = len(seen_chromosomes)
    print(
        f"\nDone.  {run_idx} runs in {total_time:.1f}s  "
        f"({solved} solved, {unique_sol} unique solutions)"
    )
    print(f"Results: {results_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run a MILP parameter sweep over the P2 WSN instance."
    )
    p.add_argument(
        "--preset", choices=list(cfg.PRESETS), default=None,
        help="Override SWEEP_PRESET from config.py.",
    )
    p.add_argument(
        "--instance", type=Path, default=cfg.INSTANCE_PATH,
        help="Path to P2 instance JSON (default: instance/ind2.json).",
    )
    p.add_argument(
        "--results-dir", type=Path, default=cfg.RESULTS_DIR,
        help="Directory to write results (default: results/).",
    )
    p.add_argument(
        "--time-limit", type=float, default=cfg.TIME_LIMIT,
        help="Solver time limit per run in seconds (default: %(default)s).",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip all matplotlib output.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Show solver output (verbose mode).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    preset = args.preset or cfg.SWEEP_PRESET
    sweep_params = cfg.PRESETS[preset]

    print(f"Preset: {preset}  "
          f"({len(sweep_params['C0'])} C0 × "
          f"{len(sweep_params['kdecay'])} kdecay × "
          f"{len(sweep_params['B'])} B = "
          f"{len(sweep_params['C0']) * len(sweep_params['kdecay']) * len(sweep_params['B'])} runs)")

    run_sweep(
        instance_path = args.instance,
        results_dir   = args.results_dir,
        sweep_params  = sweep_params,
        w_install     = cfg.W_INSTALL,
        time_limit    = args.time_limit,
        make_plots    = not args.no_plots,
        verbose       = args.verbose,
    )


if __name__ == "__main__":
    main()
