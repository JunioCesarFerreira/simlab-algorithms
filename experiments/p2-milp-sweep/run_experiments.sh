#!/usr/bin/env bash
# run_experiments.sh — end-to-end entry point for the P2 MILP sweep.
#
# Usage:
#   ./run_experiments.sh [--preset quick|medium|full] [EXTRA_ARGS...]
#
# Examples:
#   ./run_experiments.sh                         # quick preset (default)
#   ./run_experiments.sh --preset medium
#   ./run_experiments.sh --preset full --no-plots
#   ./run_experiments.sh --preset quick --time-limit 120

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Python environment ─────────────────────────────────────────────────────────
# Prefer the pop-estimator venv (has numpy); fall back to repo venv or PATH python3.
if [[ -x "$SCRIPT_DIR/../../pop-estimator/.venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/../../pop-estimator/.venv/bin/python"
elif [[ -x "$SCRIPT_DIR/../../.venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/../../.venv/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

echo "Using Python: $($PYTHON --version 2>&1)"

# ── Dependency check ───────────────────────────────────────────────────────────
if ! "$PYTHON" -c "import pulp" 2>/dev/null; then
    echo ""
    echo "ERROR: pulp not found."
    echo "  Install dependencies with:"
    echo "    $PYTHON -m pip install -r requirements.txt"
    echo ""
    echo "  Solver options (in priority order):"
    echo "    - Gurobi: pip install gurobipy  (requires a licence)"
    echo "    - HiGHS:  pip install highspy   (free, fast open-source solver)"
    echo "    - CBC:    bundled with pulp      (free, slower for large models)"
    echo ""
    exit 1
fi

# ── Solver detection ───────────────────────────────────────────────────────────
SOLVER=$("$PYTHON" - <<'PYEOF'
import sys
sys.path.insert(0, ".")
try:
    from milp.model import detect_solver_name
    print(detect_solver_name())
except Exception as e:
    print(f"unknown ({e})")
PYEOF
)
echo "Solver detected: $SOLVER"

if [[ "$SOLVER" == "CBC" ]]; then
    echo ""
    echo "NOTE: Using CBC (bundled with pulp).  CBC is slower than Gurobi or HiGHS."
    echo "  The 'quick' preset (48 runs, time_limit=300s each) is recommended."
    echo "  For better performance install HiGHS:  pip install highspy"
    echo ""
fi

# ── Run sweep ──────────────────────────────────────────────────────────────────

echo ""
echo "========================================="
echo " P2 MILP Sweep — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
echo ""

"$PYTHON" sweep.py "$@"

echo ""
echo "Sweep complete. Results saved in: $SCRIPT_DIR/results/"
