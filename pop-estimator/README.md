# P2 Population Estimator

A heuristic-statistical procedure (and full Python package) to estimate the
**population size `n`** required by NSGA-III / genetic algorithms applied to
**Problem P2 (Discrete Coverage with Mobility)**.

> ⚠️ **Important** — `n_hat` is an **estimate**, not a guarantee. It comes
> from a gambler-ruin approximation over structurally-defined blocks,
> comparing a candidate block-optimum `H_i^*` against a deceptive competitor
> `H_i^L`. The choice of `H_i^*` and `H_i^L` is *heuristic*. Repeat the
> experiment with different partitions, scalarisations, and heuristics
> before trusting any single value.

## 1. Mathematical summary

Given the binary chromosome `x ∈ {0,1}^J` where `x_j = 1` selects candidate
`q_j` as a relay, we:

1. Partition `Q = {q_1,…,q_J}` into `m` structural blocks `Q_1,…,Q_m`
   (`partitioning.py`).
2. For each block `i`, build two patterns of `Q_i`:
   - `H_i^*` — a candidate block-optimum (heuristic).
   - `H_i^L` — a locally-attractive / deceptive competitor.
3. Generate `R` complements `x_{-i}^{(r)}` filling the rest of the chromosome.
4. Evaluate `F(H_i^*, x_{-i}^{(r)})` and `F(H_i^L, x_{-i}^{(r)})` either via
   the *surrogate* (`evaluation/surrogate.py`) or via Cooja on remote
   containers (`evaluation/cooja.py`).
5. Aggregate metrics with `Ψ_a` (`aggregate_metrics`) and scalarise to `F(x)`
   with `S` (`scalarize`).
6. Compute:

   - `d_i_hat   = mean_r [ F_star^{(r)} − F_local^{(r)} ]`
   - `σ_BB_i_hat = stdev_r [ Δ^{(r)} ]` (ddof = 1)

7. Estimate per block:

   - Uniform initialisation:

     ```
     n_i_hat = −ln(α) · 2^{k_i − 1} · σ_BB_i_hat · √(2m) / d_i_hat
     ```

   - Bernoulli(ρ) initialisation:

     ```
     π_i(H_i^*) = ρ^{s_i} (1−ρ)^{k_i − s_i}
     n_i_hat   = (−ln(α) / (2 π_i)) · σ_BB_i_hat · √(2m) / d_i_hat
     ```

8. Global estimate: `n_hat = max_i ⌈n_i_hat⌉` over **valid** blocks.

## 2. Input format

The package accepts the SimLab P2 JSON schema (both at the root or wrapped in
`parameters`). See [`examples/ind2.json`](examples/ind2.json) for a complete
working instance. The minimum required keys are:

```json
{
  "problem": {
    "name": "...",
    "radius_of_reach": 50,
    "radius_of_inter": 100,
    "region": [xmin, ymin, xmax, ymax],
    "sink": [x, y],
    "candidates": [[x1,y1], [x2,y2], ...],
    "mobile_nodes": [
      {
        "name": "...",
        "speed": 5,
        "time_step": 1,
        "is_closed": true,
        "is_round_trip": false,
        "path_segments": [["x(t)", "y(t)"], ...]
      }
    ]
  }
}
```

Trajectory expressions evaluate `t ∈ [0, 1]` within each segment and may use
`np.cos`, `np.sin`, `np.pi`, etc. They run in a restricted namespace
(no Python builtins).

## 3. Installation

```bash
cd pop-estimator
python -m pip install -e .[dev]            # surrogate + tests
python -m pip install -e .[ssh,kmeans,dev] # add paramiko & sklearn
```

Python ≥ 3.11 is required.

## 4. Running in surrogate mode

```bash
python -m p2_population_estimator.cli \
  --instance examples/ind2.json \
  --output-dir results/p2_estimation_run_001 \
  --mode surrogate \
  --partition-method grid \
  --num-blocks 8 \
  --num-complements 30 \
  --alpha 0.05 \
  --rho 0.20 \
  --seeds 336157 667370 35239 873465 \
  --random-seed 42
```

The surrogate is deterministic and Cooja-free — use it first to validate the
pipeline before paying the cost of real simulation.

## 5. Running with Cooja

```bash
python -m p2_population_estimator.cli \
  --instance examples/ind2.json \
  --output-dir results/p2_cooja_run_001 \
  --mode cooja \
  --ssh-host localhost \
  --ssh-user <USER> \
  --ssh-ports 2231 2232 2233 2234 2235 2236 \
  --partition-method grid \
  --num-blocks 8 \
  --num-complements 30 \
  --alpha 0.05 \
  --rho 0.20 \
  --seeds 336157 667370 35239 873465 \
  --simulation-timeout 900 \
  --random-seed 42
```

Each of the six SSH ports is handled by a dedicated worker thread; at most
one simulation runs per container at a time. Failures are retried up to
`--max-retries` times. **The default `CoojaEvaluator.file_generator` writes
placeholder artifacts** — you are expected to subclass `CoojaEvaluator` (or
pass a `file_generator` callable) to emit firmware-specific `.csc` /
`positions` / config files.

## 6. Interpreting `population_estimate_result.json`

Top-level keys:

| Key                     | Meaning                                                        |
| ----------------------- | -------------------------------------------------------------- |
| `experiment_config`     | Snapshot of the CLI/config arguments.                          |
| `instance_summary`      | Counts and metadata of the loaded instance.                    |
| `partition_summary`     | `num_blocks`, sizes, and indices per block.                    |
| `block_results`         | Per-block `d_i_hat`, `σ_BB_i_hat`, `n_i_hat`, status, samples. |
| `global_estimate`       | `n_hat_uniform`, `n_hat_bernoulli`, valid/invalid counts.      |
| `failed_evaluations`    | Any solution that raised during evaluation.                    |
| `warnings`              | Warnings (incl. the disclaimer).                               |
| `reproducibility_info`  | Versions, seeds, instance SHA-256.                             |

Per-block statuses:

- `ok` — valid estimate.
- `degenerate_zero_variance` — `n_i_hat` computed using a small floor.
- `invalid_non_positive_d` — `H_i^*` was not consistently better than
  `H_i^L`. Revise the heuristics.
- `insufficient_samples` — `R < 2`, variance cannot be estimated.

## 7. Limitations

- `H_i^*` and `H_i^L` are *heuristic* — the value of `n_hat` is only as
  meaningful as those heuristics. Run with multiple partitions / heuristics.
- The surrogate captures structural coverage, not Cooja's MAC/routing
  behaviour. Use it for pipeline validation and fast iteration, not as a
  faithful proxy for end-to-end performance.
- The gambler-ruin formula assumes block-level independence (no epistasis
  between blocks beyond `x_{-i}`). For strongly-coupled problems the
  estimate is optimistic.
- `α` is a desired failure probability per block; multiple-testing
  considerations are not corrected for here.

## 8. Project layout

```
p2_population_estimator/
  __init__.py
  __main__.py
  config.py
  models.py
  io.py
  geometry.py
  partitioning.py
  blocks.py
  complements.py
  evaluation/
    __init__.py
    base.py
    surrogate.py
    cooja.py
    ssh_pool.py
    parser.py
  statistics.py
  estimator.py
  experiment.py
  cli.py
  logging_utils.py
tests/
  conftest.py
  test_io.py
  test_geometry.py
  test_partitioning.py
  test_blocks.py
  test_statistics.py
  test_estimator.py
  test_surrogate.py
  test_parser.py
  test_experiment.py
examples/
  ind2.json
```

## 9. Running tests

```bash
cd pop-estimator
pytest -v
```
