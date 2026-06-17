# simlab-algorithms
Study and validation of auxiliary algorithms used in simlab for multi-objective optimization of wireless sensor networks.

---

## Population-size estimation for the P2 relay-placement problem

This repository contains **four complementary estimators** of the genetic-algorithm
population size needed to solve the P2 problem (place the fewest relays so every
mobile sensor stays connected to the sink), plus a **combined estimator** that
fuses them, and an **IEEE-style paper** that writes the whole study up.

All four methods are evaluated on one fixed instance (`pop-estimator/examples/ind2.json`:
N=64 candidates, M=4 mobiles, R=50) so the estimates are directly comparable.

| Method | File | Signal | N̂ (ind2) |
|---|---|---|---|
| **M1 — Inference** | [methods/method1_inference.py](methods/method1_inference.py) | Diminishing returns of converged GA fitness vs population size | 22 |
| **M2 — Adjacency** | [methods/method2_adjacency.py](methods/method2_adjacency.py) | Temporal co-occurrence matrix A(t), coverage indispensability | 24 |
| **M3 — Routing** | [methods/method3_routing.py](methods/method3_routing.py) | Shortest-path usage R(t), routing criticality | 14 |
| **M4 — MILP calibration** | [methods/method4_milp.py](methods/method4_milp.py) | GA solution-quality gap vs MILP optimum | 48 |
| **Combined** | [methods/combined_estimator.py](methods/combined_estimator.py) | Envelope / inverse-variance / Bayesian fusion | env 48, consensus 25 |

A classical worst-case **gambler-ruin** bound (the existing `pop-estimator`) is
reported as a baseline (≈2839 on this instance — two orders larger).

### Components

- [ga/](ga/) — a compact binary-chromosome GA (`engine.py`) and the sweep
  (`run_ga.py`) over population sizes × seeds. This is the data source for M1 and M4.
- [methods/](methods/) — the four estimators, shared infrastructure
  (`common.py`), and the combined estimator.
- [results/](results/) — all raw outputs (`*_result.json`, `summary_table.csv`)
  and the 14 figures.
- [paper/](paper/) — `main.tex` (IEEEtran), `references.bib`, `figures/`,
  `tables/`, and `build_paper_assets.py`.
- [REPO_MAP.md](REPO_MAP.md) — full reconnaissance of the repository.

## How to reproduce

### 1. Environment

The pipeline needs **NumPy** and **Matplotlib** only (curve fitting and
regression are implemented with NumPy — no SciPy/scikit-learn). The MILP sweep
additionally needs **PuLP**.

```bash
# use the existing pop-estimator venv, or create one:
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`run_all.sh` auto-detects `pop-estimator/.venv/bin/python` if present.

### 2. Run everything

```bash
./run_all.sh
```

This regenerates, in order: the GA sweep → the gambler-ruin baseline → Methods
1–4 → the combined estimator → the paper figures/tables, then compiles
`paper/main.tex` if `pdflatex` is available.

### 3. Run a single method

```bash
PY=pop-estimator/.venv/bin/python
$PY ga/run_ga.py                       # (once) produces results/ga_runs/
$PY -m methods.method1_inference
$PY -m methods.method2_adjacency
$PY -m methods.method3_routing
$PY -m methods.method4_milp
$PY -m methods.combined_estimator
```

### 4. Build the paper

```bash
pop-estimator/.venv/bin/python paper/build_paper_assets.py   # copy figures + table
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Or upload the `paper/` directory to **Overleaf** (it is self-contained:
`figures/` and `tables/` are populated by `build_paper_assets.py`).

### Where outputs go

```
results/
  ga_runs/        ga_summary.csv, ga_runs.json, ga_evaluated_stream.json, ga_best_overall.json
  method1/        method1_result.json + 3 figures + gambler_ruin_baseline/
  method2/        method2_result.json + 4 figures
  method3/        method3_result.json + 4 figures
  method4/        method4_result.json + 2 figures
  combined/       combined_result.json, summary_table.csv, combined_comparison.png
```

## Related experiments

- [experiments/p2-milp-sweep/](experiments/p2-milp-sweep/) — the MILP design-space
  sweep that produces the reference optima consumed by Method 4.
- [pop-estimator/](pop-estimator/) — the gambler-ruin population estimator
  (Method 1 baseline) with surrogate and Cooja back-ends.
