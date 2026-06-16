"""Sweep configuration for the P2 MILP experiment.

Edit SWEEP_PRESET to choose between a quick test run and the full sweep
that mirrors wsn-design-space-exploration.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

INSTANCE_PATH = Path(__file__).parent / "instance" / "ind2.json"
RESULTS_DIR   = Path(__file__).parent / "results"

# ── MILP fixed parameters ─────────────────────────────────────────────────────

W_INSTALL  = 1_000_000.0   # penalty weight for each installed relay

# Solver time limit per run (seconds).
# With Gurobi each run completes in seconds; with HiGHS the solver hits this
# limit but still returns the best feasible solution found (status = OPTIMAL in
# PuLP, meaning "feasible" not "proven optimal").  60 s balances solution
# quality against wall-clock time for the quick preset (~50 min with HiGHS).
TIME_LIMIT = 60.0

# Timestep subsampling: solve only every TIME_SAMPLE_STEP-th timestep.
# The mobile trajectories are smooth/periodic, so a sample of T/step timesteps
# captures the connectivity patterns at a fraction of the MILP size.
# Tested empirically on this instance (T=180, M=4, N=64):
#   step=1  → T_eff=180, ~124K vars — too large for HiGHS; use Gurobi
#   step=10 → T_eff=18,   ~12K vars — good coverage, HiGHS finds solutions in 60s
#   step=30 → T_eff=6,    ~4K  vars — coarser sampling, same HiGHS time behaviour
TIME_SAMPLE_STEP = 10

# ── Sweep preset ──────────────────────────────────────────────────────────────
# "quick"  — 4 × 3 × 4 = 48 runs  (Gurobi: ~2 min; HiGHS: ~50 min)
# "medium" — 6 × 5 × 10 = 300 runs
# "full"   — 11 × 5 × 50 = 2 750 runs (mirrors wsn-design-space-exploration)

SWEEP_PRESET = "quick"

PRESETS: dict[str, dict] = {
    "quick": {
        "C0":     [10, 110, 510, 1010],
        "kdecay": [0.9, 0.5, 0.1],
        "B":      [1, 10, 50, 99],
    },
    "medium": {
        "C0":     list(range(10, 1110, 200)),    # 10, 210, 410, 610, 810, 1010
        "kdecay": [0.9, 0.75, 0.5, 0.25, 0.1],
        "B":      list(range(1, 101, 10)),        # 1, 11, 21, …, 91
    },
    "full": {
        "C0":     list(range(10, 1110, 100)),    # 10, 110, …, 1010  (11 values)
        "kdecay": [0.9, 0.75, 0.5, 0.25, 0.1],  # 5 values
        "B":      list(range(1, 101, 2)),         # 1, 3, …, 99       (50 values)
    },
}


def get_sweep_params() -> dict:
    """Return the (C0, kdecay, B) ranges for the active preset."""
    if SWEEP_PRESET not in PRESETS:
        raise ValueError(
            f"Unknown SWEEP_PRESET={SWEEP_PRESET!r}. "
            f"Choose from {list(PRESETS)}"
        )
    return PRESETS[SWEEP_PRESET]


def total_runs() -> int:
    p = get_sweep_params()
    return len(p["C0"]) * len(p["kdecay"]) * len(p["B"])
