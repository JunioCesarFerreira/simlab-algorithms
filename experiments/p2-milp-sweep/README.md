# P2 MILP Parameter Sweep

End-to-end MILP design-space exploration over the **P2 population-estimator
instance** (`ind2.json`), adapting the formulation and sweep framework from
[`wsn-milp`](../../wsn-milp) and
[`wsn-design-space-exploration`](../../wsn-design-space-exploration).

---

## Instance

**File:** `instance/ind2.json`
(copied from `pop-estimator/examples/ind2.json`)

| Property | Value |
|---|---|
| Candidates N | 64 |
| Mobile sensors M | 4 |
| Radius of reach R | 50 |
| Region | [−150, −150, 150, 150] |
| Sink | (0, 0) |
| Simulation duration | 180 s |
| Time step dt | 1 s → **T = 180 timesteps** |

The four mobile nodes follow piecewise-parametric trajectories (closed loops
and round trips) that cover all four quadrants of the region.

---

## MILP Formulation

The model is the **mobile-coverage MILP** (same as `wsn-milp/wsn-mobile`):

### Decision variables
| Variable | Domain | Meaning |
|---|---|---|
| `y_j` | {0, 1} | relay j installed |
| `z_{ij}(t)` | {0, 1} | edge (i,j) active at timestep t |
| `x_{ij}(t)` | ≥ 0 | flow on (i,j) at t |

### Objective
```
min  w_install · Σ_j y_j  +  Σ_t Σ_{(i,j)∈E_t} ‖pos_i(t)−pos_j(t)‖² · x_{ij}(t)
```

### Constraints
1. **Capacity** — `x_{ij}(t) ≤ C_{ij}(t) · z_{ij}(t)` where
   `C_{ij}(t) = C0 · max{0, 1 − k_decay · d_{ij}(t)/R}²`
2. **Installation guard** — a link can only be active if both fixed endpoints
   are installed
3. **Mobile flow conservation** — each mobile generates B units per timestep
4. **Fixed node flow conservation** — relay nodes are transit only
5. **Sink balance** — total inflow at sink equals total mobile demand

---

## Swept Parameters

Three parameters are varied jointly, matching `wsn-design-space-exploration`:

| Parameter | Meaning | Preset ranges |
|---|---|---|
| `C0` | Nominal link capacity | see table below |
| `k_decay` | Distance decay factor in capacity function | see table below |
| `B` | Demand (flow units) per mobile per timestep | see table below |

### Presets

| Preset | C0 | k_decay | B | Total runs |
|---|---|---|---|---|
| `quick` | 10, 110, 510, 1010 | 0.9, 0.5, 0.1 | 1, 10, 50, 99 | **48** |
| `medium` | 10, 210, …, 1010 (6 values) | 0.9, 0.75, 0.5, 0.25, 0.1 | 1, 11, …, 91 (10 values) | **300** |
| `full` | 10, 110, …, 1010 (11 values) | 0.9, 0.75, 0.5, 0.25, 0.1 | 1, 3, …, 99 (50 values) | **2 750** |

Edit `SWEEP_PRESET` in `config.py` to change the active preset.

---

## How to Run

### Requirements

- Python ≥ 3.10
- `pulp` (MILP modelling layer, bundled with a CBC solver)
- `numpy`, `matplotlib`

```bash
pip install -r requirements.txt
```

#### Solver options (in priority order)

| Solver | Install | Performance on this instance |
|---|---|---|
| **Gurobi** | `pip install gurobipy` + licence | Seconds per run, proven-optimal solutions |
| **HiGHS** | `pip install highspy` | ~60 s per run; finds good feasible solutions (hits time limit, not proven optimal) |
| **CBC** | bundled with `pulp` | Slower than HiGHS; not recommended for production sweeps |

With HiGHS (no Gurobi licence), the `quick` preset (48 runs, `TIME_LIMIT=60 s`)
completes in approximately **50 minutes** and produces valid feasible solutions
comparable across parameter settings.

### Quick start

```bash
cd experiments/p2-milp-sweep

# Default: quick preset (48 runs)
./run_experiments.sh

# Explicit preset
./run_experiments.sh --preset medium

# Full sweep, no plots
./run_experiments.sh --preset full --no-plots

# Override time limit per run (seconds)
./run_experiments.sh --preset quick --time-limit 120
```

All arguments after `./run_experiments.sh` are forwarded to `sweep.py`:

```
python sweep.py --help
```

### Direct Python invocation

```bash
python sweep.py --preset quick --results-dir results/my_run
```

---

## Outputs

```
results/
  milp_run_summary.json       # one record per MILP solve (all runs)
  pic_candidates.png          # instance overview: all candidates + trajectories
  sweep_summary.png           # scatter: installed relays vs B, coloured by C0
  <chromosome>/               # one directory per unique solution found
    output.json               # simulation-ready JSON (sink + installed relays)
    pic_installed_graph.png   # static relay connectivity graph
```

### `milp_run_summary.json` schema

Each record in the array:

```json
{
  "C0": 110,
  "k_decay": 0.5,
  "B": 10,
  "w_install": 1000000.0,
  "status": 2,
  "status_name": "OPTIMAL",
  "runtime_seconds": 4.3,
  "solution_count": 1,
  "variables": 9842,
  "constraints": 14721,
  "objective_value": 2000041.7,
  "mip_gap": 0.0,
  "installed_nodes": 2,
  "chromosome": "0001000000..."
}
```

### `<chromosome>/output.json` schema

Simulation-ready JSON containing only the installed relays + sink (ready for
Cooja or the pop-estimator surrogate):

```json
{
  "simulationModel": { ... },
  "milpParameters": { "C0": ..., "k_decay": ..., "B": ..., "w_install": ... },
  "milpSolve":      { "status": ..., "installed_nodes": ..., ... }
}
```

---

## Project Layout

```
experiments/p2-milp-sweep/
├── README.md                  this file
├── config.py                  sweep presets and global settings
├── sweep.py                   parameter sweep orchestrator
├── run_experiments.sh         shell entry point
├── instance/
│   └── ind2.json              P2 instance (copied from pop-estimator/examples/)
├── milp/
│   ├── __init__.py
│   ├── instance_adapter.py    P2 JSON → P2MilpInputs (field mapping)
│   └── model.py               mobile-coverage MILP (PuLP, solver-agnostic)
├── utils/
│   ├── __init__.py
│   ├── sim_utils.py           trajectory generation (ported from wsn-dse)
│   └── plot_utils.py          matplotlib helpers
└── results/                   created at runtime
```

---

## Source Repositories

| File | Ported from |
|---|---|
| `milp/model.py` | `wsn-milp/wsn-mobile/mobile.py` + `wsn-design-space-exploration/milp/mobile-model/runner.py` |
| `utils/sim_utils.py` | `wsn-design-space-exploration/milp/mobile-model/utils/sim_utils.py` |
| `utils/plot_utils.py` | `wsn-design-space-exploration/milp/mobile-model/utils/plot_utils.py` |
| Sweep loop | `wsn-design-space-exploration/milp/mobile-model/runner.py` |
| Instance format | `pop-estimator/examples/ind2.json` (P2 schema) |

### Key adaptations

1. **Instance format** — P2 uses `radius_of_reach`, `candidates` (array of
   `[x, y]`), `path_segments`, `is_round_trip` (snake_case); the adapter in
   `milp/instance_adapter.py` maps these to the internal key format used by the
   MILP model.
2. **Refactored MILP** — the monolithic `runner.py` loop was split into a
   reusable `solve()` function (`milp/model.py`) and a separate orchestrator
   (`sweep.py`), making the model independently testable.
3. **Explicit sink** — P2 declares the sink as `"sink": [x, y]` (separate from
   candidates); the adapter handles this without requiring a `"root"` mote entry.
4. **Solver-agnostic MILP** — the model uses PuLP instead of gurobipy directly.
   The `_select_solver()` helper picks Gurobi → HiGHS → CBC in priority order,
   so the same code works with any available solver.
5. **Timestep subsampling** — `TIME_SAMPLE_STEP=10` (set in `config.py`) reduces
   the effective T from 180 to 18, cutting the model to ~12 K variables.  This
   makes each run feasible for open-source solvers (HiGHS finds solutions in
   ~60 s; Gurobi proves optimality in seconds).
