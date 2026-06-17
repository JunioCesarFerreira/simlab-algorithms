"""Run the P2 genetic algorithm across population sizes × seeds on ``ind2``.

Outputs (under ``results/ga_runs/``)
------------------------------------
* ``ga_summary.csv``          one row per (pop_size, seed): converged fitness,
                              relay count, connectivity, evaluations, and the
                              generation at which 95 % of the run's improvement
                              was reached.
* ``ga_runs.json``            full per-generation histories for every run
                              (used for the convergence plots).
* ``ga_evaluated_stream.json``ordered distinct (chromosome, F) evaluations from
                              the largest-population reference run — the raw
                              material for Method 1's inter-sample analysis.
* ``ga_best_overall.json``    the best solution found across all runs (fewest
                              relays at full connectivity) — a GA reference for
                              Method 4.

This script is the shared data source for Method 1 and Method 4.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ga.engine import GAConfig, run_ga
from methods.common import (
    GA_RUNS_DIR, INSTANCE_PATH, ensure_dir, make_surrogate_fitness, save_json,
)

# Population sizes span the diminishing-returns regime; seeds give the spread
# Method 1 / Method 4 turn into confidence intervals.
POP_SIZES   = [4, 8, 16, 32, 64, 128, 256]
SEEDS       = [0, 1, 2, 3, 4]
GENERATIONS = 80


def _gen_to_fraction(history: list[dict], frac: float = 0.95) -> int:
    """First generation reaching ``frac`` of the run's total fitness improvement."""
    f0 = history[0]["best_F"]
    ff = history[-1]["best_F"]
    if ff <= f0:
        return 0
    target = f0 + frac * (ff - f0)
    for row in history:
        if row["best_F"] >= target:
            return int(row["generation"])
    return int(history[-1]["generation"])


def main(argv: list[str] | None = None) -> int:
    fitness_fn, info = make_surrogate_fitness(INSTANCE_PATH)
    n_bits = info["n_bits"]
    print(f"Instance: {info['instance_name']}  N={n_bits}  M={info['n_mobiles']}  "
          f"R={info['radius_of_reach']}")
    print(f"GA sweep: pop_sizes={POP_SIZES}  seeds={SEEDS}  generations={GENERATIONS}\n")

    out_dir = ensure_dir(GA_RUNS_DIR)
    summary_rows: list[dict] = []
    all_runs: dict[str, list[dict]] = {}
    best_overall = {"relay_count": n_bits + 1, "connected_ratio": 0.0,
                    "F": -1e18, "bits": None, "pop_size": None, "seed": None}

    t_start = time.time()
    for P in POP_SIZES:
        for s in SEEDS:
            cfg = GAConfig(n_bits=n_bits, pop_size=P, generations=GENERATIONS, seed=s)
            res = run_ga(fitness_fn, cfg)
            key = f"pop{P}_seed{s}"
            all_runs[key] = res.history
            g95 = _gen_to_fraction(res.history, 0.95)
            summary_rows.append({
                "pop_size": P, "seed": s,
                "final_best_F": round(res.best_F, 6),
                "final_best_relays": int(res.best_metrics.get("relay_count", -1)),
                "final_best_conn": round(res.best_metrics.get("connected_ratio", 0.0), 4),
                "evaluations": res.evaluations,
                "gen_to_95pct": g95,
            })
            # track the fittest feasible (fully-connected) layout with fewest relays
            m = res.best_metrics
            cand = (m.get("connected_ratio", 0.0), -m.get("relay_count", 1e9))
            cur  = (best_overall["connected_ratio"], -best_overall["relay_count"])
            if cand > cur:
                best_overall = {
                    "relay_count": int(m.get("relay_count", -1)),
                    "connected_ratio": float(m.get("connected_ratio", 0.0)),
                    "F": float(res.best_F), "bits": res.best_bits,
                    "pop_size": P, "seed": s,
                }
            print(f"  pop={P:>3} seed={s}  F={res.best_F:.4f}  "
                  f"relays={int(m.get('relay_count',-1)):>2}  "
                  f"conn={m.get('connected_ratio',0):.3f}  "
                  f"evals={res.evaluations}  g95={g95}")

    # --- reference run: record the full evaluated stream for Method 1 ---
    print("\nReference run (largest pop, seed 0) with full evaluated-stream logging...")
    ref_cfg = GAConfig(n_bits=n_bits, pop_size=max(POP_SIZES), generations=GENERATIONS, seed=0)
    ref = run_ga(fitness_fn, ref_cfg, record_evaluated=True)
    stream = [{"chromosome": "".join(map(str, bits)), "F": F}
              for bits, F in ref.all_evaluated]
    print(f"  distinct evaluations logged: {len(stream)}")

    # --- persist ---
    csv_path = out_dir / "ga_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    save_json(out_dir / "ga_runs.json", {
        "instance": info, "pop_sizes": POP_SIZES, "seeds": SEEDS,
        "generations": GENERATIONS, "runs": all_runs,
    })
    save_json(out_dir / "ga_evaluated_stream.json", {
        "pop_size": max(POP_SIZES), "seed": 0, "n_bits": n_bits,
        "n_distinct": len(stream), "stream": stream,
    })
    save_json(out_dir / "ga_best_overall.json", best_overall)

    dt = time.time() - t_start
    print(f"\nDone in {dt:.1f}s.  Best overall: relays={best_overall['relay_count']} "
          f"conn={best_overall['connected_ratio']:.3f} "
          f"(pop={best_overall['pop_size']}, seed={best_overall['seed']})")
    print(f"Wrote: {csv_path}")
    print(f"       {out_dir/'ga_runs.json'}")
    print(f"       {out_dir/'ga_evaluated_stream.json'}")
    print(f"       {out_dir/'ga_best_overall.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
