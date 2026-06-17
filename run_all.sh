#!/usr/bin/env bash
# run_all.sh — regenerate every result (Phases 1-5) and build the paper (Phase 6).
#
# Pipeline:
#   0. GA sweep on the ind2 instance              -> results/ga_runs/
#   1. Gambler-ruin baseline (Method 1 baseline)  -> results/method1/gambler_ruin_baseline/
#   2. Method 1  inference / diminishing returns   -> results/method1/
#   3. Method 2  adjacency coverage                -> results/method2/
#   4. Method 3  routing criticality               -> results/method3/
#   5. Method 4  MILP calibration                  -> results/method4/
#   6. Combined estimator (3 fusions)              -> results/combined/
#   7. Copy figures + table into paper/, compile main.tex (if pdflatex present)
#
# Usage:  ./run_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# --- pick a Python with numpy + matplotlib (prefer the pop-estimator venv) ---
if [[ -x "$ROOT/pop-estimator/.venv/bin/python" ]]; then
    PYTHON="$ROOT/pop-estimator/.venv/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi
echo "Using Python: $($PYTHON --version 2>&1)  ($PYTHON)"

if ! "$PYTHON" -c "import numpy, matplotlib" 2>/dev/null; then
    echo "ERROR: numpy and matplotlib are required."
    echo "  Install with:  $PYTHON -m pip install -r requirements.txt"
    exit 1
fi

hr() { printf '\n========== %s ==========\n' "$1"; }

hr "0/7  GA sweep (data source for Methods 1 and 4)"
"$PYTHON" ga/run_ga.py

hr "1/7  Gambler-ruin baseline on ind2 (Method 1 baseline)"
( cd "$ROOT/pop-estimator" && "$PYTHON" -m p2_population_estimator \
    ../results/method1/gambler_ruin_ind2.json )

hr "2/7  Method 1 — inference via inter-sample differences"
"$PYTHON" -m methods.method1_inference

hr "3/7  Method 2 — adjacency-matrix temporal dynamics"
"$PYTHON" -m methods.method2_adjacency

hr "4/7  Method 3 — routing-matrix variant"
"$PYTHON" -m methods.method3_routing

hr "5/7  Method 4 — MILP-based calibration"
"$PYTHON" -m methods.method4_milp

hr "6/7  Combined estimator (envelope / weighted / Bayesian)"
"$PYTHON" -m methods.combined_estimator

hr "7/7  Paper assets + compile"
"$PYTHON" paper/build_paper_assets.py
if command -v pdflatex >/dev/null 2>&1; then
    ( cd "$ROOT/paper" && \
      pdflatex -interaction=nonstopmode main.tex >/dev/null && \
      ( bibtex main >/dev/null || true ) && \
      pdflatex -interaction=nonstopmode main.tex >/dev/null && \
      pdflatex -interaction=nonstopmode main.tex >/dev/null )
    echo "  Built paper/main.pdf"
else
    echo "  pdflatex not found — skipping compile."
    echo "  Upload the paper/ directory to Overleaf, or install TeX Live and re-run."
fi

hr "DONE"
echo "Results in:  $ROOT/results/"
echo "Paper in:    $ROOT/paper/  (main.tex; figures/ and tables/ regenerated)"
