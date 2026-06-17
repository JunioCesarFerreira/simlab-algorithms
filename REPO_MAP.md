# REPO_MAP.md — Phase 0 Reconnaissance

Repository: `simlab-algorithms`
Generated for the *four-method population-estimation* task.
Scope: study/validation of auxiliary algorithms used in **SimLab** for
multi-objective optimisation of Wireless Sensor Networks (WSNs).

The optimisation problem under study is **P2**: choose the smallest subset
`P ⊆ Q` of candidate relay positions so that every mobile sensor stays
connected to the sink at all times. A solution is encoded as a **binary
chromosome** `B ∈ {0,1}^|Q|` (1 = candidate installed). "Population
estimation" means: *how large a population `n` does an evolutionary
algorithm need to find a good solution with probability ≥ 1−α?*

---

## 1. Directory tree and file roles

### 1.1 Top-level

| Path | Type | Role |
|---|---|---|
| `README.md` | doc | One-line project description. |
| `LICENSE` | doc | License. |
| `estimando-população-p2.md` | doc | P2 problem definition + chromosome model + gambler-ruin formula (the theory behind Method 1). |
| `adjacency_builder.py` | **module** | Builds temporal full-network adjacency `A(t)` (T,K,K), accumulated `A_total` (K,K), node layout `[sink, candidates, mobiles]`. **Source for Method 2.** |
| `path_builder.py` | **module** | Per-timestep BFS shortest-paths from sink; `node_accumulated[u]`, `edge_accumulated`. Depends on `adjacency_builder`. **Source for Method 3.** |
| `p3-building-blocks.ipynb` | notebook | Visual exploration of `A_total` (adjacency). **Adjacency-matrix estimator notebook.** |
| `p3-path-blocks.ipynb` | notebook | Visual exploration of `node_accumulated` (routing). **Routing-matrix estimator notebook.** |
| `p2-trajectory-coverage.ipynb` | notebook | Earlier P2 trajectory/coverage study (context). |
| `p1-init-and-repair.ipynb`, `filter-p1.ipynb`, `nddr.ipynb`, `test*.ipynb` | notebook | P1-stage / NDDR / scratch notebooks — **not in scope** for this task. |
| `cooja-statistical-significance.ipynb` | notebook | Statistical-significance study of Cooja runs (context for the surrogate). |
| `random-seed.ipynb` | notebook | Seed generator. |
| `exp-cooja.txt` | data | Raw Cooja experiment log (text). |
| `.venv/` | env | Root virtualenv (ignore). |

### 1.2 `pop-estimator/` — the existing population estimator (Method-1 home)

| Path | Role |
|---|---|
| `p2_population_estimator/estimator.py` | **Gambler-ruin closed-form estimators** (`estimate_uniform`, `estimate_bernoulli`, `estimate_block`, `aggregate_global`). |
| `p2_population_estimator/statistics.py` | `delta_samples`, `d_hat`, `sigma_BB_hat`, CIs — the **inter-sample-difference** machinery. |
| `p2_population_estimator/blocks.py` | Block heuristics `H*` (good) and `H_local` (deceptive competitor). |
| `p2_population_estimator/partitioning.py` | Partition `Q` into blocks (`grid`, `kmeans`, `radial_to_sink`). |
| `p2_population_estimator/complements.py` | Random complement sampling `x_{-i}`. |
| `p2_population_estimator/experiment.py` | Orchestrates the per-block evaluation loop. |
| `p2_population_estimator/evaluation/{surrogate,cooja,parser,ssh_pool,base}.py` | Two fitness back-ends: analytic **surrogate** and real **Cooja** (via SSH). |
| `p2_population_estimator/models.py` | Dataclasses (`BlockComparisonResult`, `BlockPattern`, …). |
| `p2_population_estimator/{cli,config,io,geometry,logging_utils}.py` | CLI, constants, I/O, geometry helpers. |
| `examples/ind2.json` | **P2 instance**: N=64 candidates, M=4 mobiles, R=50, sink=(0,0). |
| `examples/simple.json` | **Toy instance**: N=10, M=1, R=45, sink=(3,9). |
| `examples/example_output.json`, `examples/example_block_results.csv` | Reference outputs. |
| `results/simple_surrogate/` | Gambler-ruin run on `simple` (surrogate back-end). |
| `results/p2_cooja_run_simple_1/` | Gambler-ruin run on `simple` (Cooja back-end). |
| `run_surrogate.json`, `run_cooja.json`, `my_run.json` | Run configs (both real ones point at `examples/simple.json`). |
| `firmware/rpl-udp-csma/` | Contiki-NG firmware (`node.c`, `root.c`) for Cooja evaluation. |
| `tests/` | pytest suite for the estimator package. |
| `POPULATION_ESTIMATOR.md` | **Design doc** mapping the repo to 3 methods (M1 gambler-ruin, M2 adjacency, M3 routing) + an integration roadmap. |

### 1.3 `experiments/p2-milp-sweep/` — MILP design-space sweep (Method-4 home)

| Path | Role |
|---|---|
| `milp/model.py` | Mobile-coverage MILP (PuLP, solver-agnostic: Gurobi→HiGHS→CBC). |
| `milp/instance_adapter.py` | P2 JSON → `P2MilpInputs`. |
| `instance/ind2.json` | Copy of the P2 instance used for the sweep. |
| `config.py` | Sweep presets, `TIME_LIMIT`, `TIME_SAMPLE_STEP`. |
| `sweep.py`, `run_experiments.sh` | Orchestrator + entry point. |
| `utils/{sim_utils,plot_utils}.py` | Trajectory generation + plotting. |
| `results/milp_run_summary.json` | **42 MILP solves** (one record each). |
| `results/<chromosome>/output.json` + `pic_installed_graph.png` | Per-unique-solution artefacts (~40 dirs). |

---

## 2. "GA result files" — what actually exists

**There are no genetic-algorithm generation logs in this repository** — no
DEAP/NSGA traces, no per-generation fitness curves, no population dumps. The
project studies the *theory* of population sizing and the *building blocks* of
P2, not recorded GA executions.

The objects that can stand in for "a population of GA solutions" are:

| Surrogate for "GA data" | File(s) | Shape / fields | Instance |
|---|---|---|---|
| **MILP-optimal chromosomes** (a curated "population" of high-quality individuals across parameter settings) | `experiments/p2-milp-sweep/results/milp_run_summary.json` (+ per-solution `output.json`) | 42 records: `C0, k_decay, B, chromosome (64-bit), installed_nodes, objective_value, status_name, runtime_seconds, variables, constraints` | `ind2` |
| **Gambler-ruin Δ-samples** (inter-sample differences `F* − F_local` over random complements) | `pop-estimator/results/*/population_estimate_result.json` → `block_results[].delta_samples` / `F_star_samples` / `F_local_samples` | per block: `delta_samples`, `F_star_samples`, `F_local_samples`, `d_i_hat`, `sigma_BB_i_hat`, `n_i_uniform`, `n_i_bernoulli`, `status` | `simple` |
| **Per-block estimator table** | `pop-estimator/results/*/block_results.csv` | columns: `block_id,k_i,s_i_star,alpha,pi_i_star,d_i_hat,sigma_BB_i_hat,n_i_uniform,n_i_uniform_ceil,n_i_bernoulli,n_i_bernoulli_ceil,status` | `simple` |

**Important coverage gap:** the existing gambler-ruin results are on the
**`simple`** instance (N=10, M=1); the MILP sweep is on **`ind2`** (N=64,
M=4). No instance currently has *all four* methods computed on it.

---

## 3. The two estimator notebooks

Both share the canonical node layout `K = 1 + N + M` = `[sink, candidates,
mobiles]` and the SimLab trajectory discretisation (arc-length sampling of the
parametric `path_segments`, optional round-trip).

### 3.1 `p3-building-blocks.ipynb` — adjacency-matrix estimator (Method 2)

Logic (delegates to `adjacency_builder.py`):

- Per timestep `t`: `A(t)[u,v] = 1` iff `u≠v` and `‖pos_u(t)−pos_v(t)‖ ≤ R`
  (symmetric, zero diagonal), shape `(K,K)`.
- Accumulated co-occurrence: `A_total[u,v] = Σ_t A(t)[u,v]` ∈ `[0,T]`.
- **Mobile-coverage score** of a fixed candidate `u`:
  `cov(u) = Σ_{m∈mobiles} A_total[u,m]` ∈ `[0, T·M]`.

Figures produced (cells):
1. Heatmap of `A_total` (K×K). *(cell 6)*
2. Per-timestep density + per-mobile degree. *(cell 8)*
3. Sorted per-fixed-node `cov(u)` bar chart. *(cell 12)*
4. Spatial heat-graph of fixed nodes coloured by `cov`. *(cell 14)*

### 3.2 `p3-path-blocks.ipynb` — routing-matrix estimator (Method 3)

Logic (delegates to `path_builder.py`):

- Per timestep `t`: build `A(t)`, run **one BFS from the sink**, trace each
  mobile's shortest path back to the sink.
- `node_count(t)[u]` = #mobiles whose shortest path uses node `u`;
  `edge_count(t)[u,v]` likewise for edges.
- Accumulated: `node_accumulated[u] = Σ_t node_count(t)[u]` ∈ `[0, T·M]`;
  `edge_accumulated` analogously.
- **Routing score**: `route(u) = node_accumulated[u] / (T·M)` ∈ `[0,1]`.
- `n_active_per_t[t]` = #mobiles with a path to the sink at `t`.

Figures produced (cells):
1. Heatmap of `edge_accumulated`. *(cell 6)*
2. Per-timestep connectivity + total path length. *(cell 8)*
3. Spatial snapshot of routing tree at a chosen `t`. *(cell 10)*
4. Sorted per-fixed-node `route` bar chart. *(cell 12)*
5. Spatial heat-graph of fixed nodes coloured by `route`. *(cell 14)*

### 3.3 A(t) vs R(t) — what each captures

- **A(t) (adjacency):** *geometric reachability* — who is within range of whom.
  Symmetric, counts every in-range pair regardless of whether traffic flows.
- **R(t) (routing):** *operational usage* — who actually lies on a chosen
  shortest mobile→sink path. A candidate can have high `cov` but low `route`
  (in range but never needed) or be a routing bottleneck (moderate `cov`,
  high `route`).

---

## 4. MILP parameter-sweep results

- **Model:** mobile-coverage MILP (min installed relays + routing energy).
- **Parameters varied** (quick preset): `C0 ∈ {10,110,510,1010}` (nominal link
  capacity), `k_decay ∈ {0.9,0.5,0.1}` (distance-decay), `B ∈ {1,10,50,99}`
  (demand per mobile per timestep). 48 nominal combos; per-`(C0,k_decay)`
  B-loop stops on first INFEASIBLE.
- **Recorded:** `42` solves → 36 `OPTIMAL`, 6 `INFEASIBLE`.
  Each record: objective value, `installed_nodes` (relay count), 64-bit
  chromosome, runtime, model size (variables/constraints), solver (`HiGHS`).
- `installed_nodes` ranges ≈ 15–64 across the sweep; each unique chromosome is
  a feasible relay layout for the P2 instance.
- **Solver caveat:** HiGHS hits the 60 s time limit (feasible, not proven
  optimal); `mip_gap` is `null`. Gurobi would give proven optima in seconds.

---

## 5. Mapping to the four requested methods + status

| Requested method | Best-fit existing asset | Data available | Notes / gaps |
|---|---|---|---|
| **M1 — inference via inter-sample differences** | `pop-estimator` gambler-ruin (`statistics.delta_samples`, `estimator.estimate_*`) | Δ-samples on `simple` | The prompt describes a *diminishing-returns / information-gain* curve; the existing estimator is a *gambler-ruin* closed form (also built on inter-sample Δ). These are **related but not identical** — needs a decision. |
| **M2 — adjacency temporal dynamics** | `adjacency_builder.py` + `p3-building-blocks.ipynb` | computable on demand for any instance | The prompt's "ΣA(t) convergence over time" is a *new* angle; the notebook computes `A_total` and `cov(u)` but does not yet do a temporal-convergence stopping rule. |
| **M3 — routing-matrix variant** | `path_builder.py` + `p3-path-blocks.ipynb` | computable on demand | Same as M2 but for `node_accumulated` / `route(u)`. |
| **M4 — MILP calibration** | `experiments/p2-milp-sweep` | 42 solves on `ind2` | Maps GA population → expected gap-to-MILP-optimum; needs GA-quality data to regress against (see §2 gap). |
| **Combined estimator** | `POPULATION_ESTIMATOR.md` §5 (strategies A/B/C) | — | Requires all four `N̂` on a **common instance** (currently impossible: M1 on `simple`, M4 on `ind2`). |

---

## 6. Open decisions before Phase 1 (flagged per the output contract)

1. **What represents "GA runs"?** No real GA logs exist. Candidates: (a) treat
   the 42 MILP chromosomes as a solution population; (b) reuse the gambler-ruin
   Δ-sampling as the inter-sample differences; (c) actually run a GA to
   generate genuine generation traces.
2. **Method 1 definition:** build the *new* diminishing-returns estimator the
   prompt describes, or reframe the *existing* gambler-ruin estimator as M1?
3. **Common instance for combination:** M1 results are on `simple`, M4 on
   `ind2`. Pick `ind2`, `simple`, or both — and re-run the missing methods there
   so the four estimates are comparable.

These three are genuinely ambiguous and cascade through every later phase, so
per the task's output contract they are raised here before any code is written.
