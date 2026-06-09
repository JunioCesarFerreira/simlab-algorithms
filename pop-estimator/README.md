# P2 Population Estimator

A heuristic-statistical procedure (and full Python package) to estimate the
**population size `n`** required by NSGA-III / genetic algorithms applied to
**Problem P2 (Discrete Coverage with Mobility)**.

> **Important** — `n_hat` is an **estimate**, not a guarantee. It comes from a
> gambler-ruin approximation over structurally-defined blocks, comparing a
> candidate block-optimum `H_i^*` against a deceptive competitor `H_i^L`. The
> choice of those patterns is *heuristic*. Repeat the experiment with different
> partitions, scalarisations, and heuristics before trusting any single value.

---

## 1. Mathematical summary

Given the binary chromosome `x ∈ {0,1}^J` where `x_j = 1` selects candidate
`q_j` as a relay, we:

1. Partition `Q = {q_1,…,q_J}` into `m` structural blocks `Q_1,…,Q_m`.
2. For each block `i`, build two patterns:
   - `H_i^*` — a candidate block-optimum (heuristic).
   - `H_i^L` — a locally-attractive / deceptive competitor.
3. Generate `R` complements `x_{-i}^{(r)}` filling the rest of the chromosome.
4. Evaluate `F(H_i^*, x_{-i}^{(r)})` and `F(H_i^L, x_{-i}^{(r)})` via the
   surrogate or via Cooja on remote containers.
5. Aggregate metrics with `Ψ_a` and scalarise to `F(x)`.
6. Compute:
   - `d_i_hat    = mean_r [ F_star^{(r)} − F_local^{(r)} ]`
   - `σ_BB_i_hat = stdev_r [ Δ^{(r)} ]`  (ddof = 1)
7. Estimate per block:

   Uniform initialisation:
   ```
   n_i_hat = −ln(α) · 2^{k_i − 1} · σ_BB_i_hat · √(2m) / d_i_hat
   ```

   Bernoulli(ρ) initialisation:
   ```
   π_i(H_i^*) = ρ^{s_i} (1−ρ)^{k_i − s_i}
   n_i_hat    = (−ln(α) / (2 π_i)) · σ_BB_i_hat · √(2m) / d_i_hat
   ```

8. Global estimate: `n_hat = max_i ⌈n_i_hat⌉` over valid blocks.

---

## 2. Installation

```bash
cd pop-estimator
python -m pip install -e .[dev]             # surrogate + tests
python -m pip install -e .[ssh,kmeans,dev]  # add paramiko & scikit-learn
```

Python ≥ 3.11 required.

---

## 3. Configuration

All parameters live in a single JSON file. Generate the default template with:

```bash
p2-popest --dump-config > my_run.json
```

The output looks like:

```json
{
  "instance_path": "path/to/instance.json",
  "output_dir":    "results/run_001",
  "mode":          "surrogate",

  "partition_method": "grid",
  "num_blocks":       8,
  "hstar_method":     "structural_greedy",
  "hlocal_method":    "deceptive_low_cost",

  "complement_method": "bernoulli",
  "num_complements":   30,
  "rho":               0.2,

  "alpha":              0.05,
  "aggregation_method": "mean_with_std",

  "seeds":       [42],
  "random_seed": 42,

  "weights": {
    "w_connected":    1.0,
    "w_relays":       0.05,
    "w_hops":         0.05,
    "w_dist":         0.05,
    "w_redundancy":   0.02,
    "w_latency":      0.0,
    "w_energy":       0.0,
    "w_throughput":   0.0,
    "required_metrics": ["connected_ratio", "relay_count"]
  },

  "ssh_host":     "localhost",
  "ssh_user":     "",
  "ssh_password": null,
  "ssh_ports":    [2231, 2232, 2233, 2234, 2235, 2236],

  "remote_workdir":   "/tmp/popest",
  "simulation_timeout":  900,
  "simulation_duration": 180,
  "remote_cooja_dir": "/opt/contiki-ng/tools/cooja",
  "cooja_command_template": "cd {remote_cooja_dir} && /opt/java/openjdk/bin/java --enable-preview -Xms4g -Xmx4g -jar build/libs/cooja.jar --no-gui {simulation_file}",
  "max_retries":        2,
  "firmware_local_dir": null,

  "force_overwrite": false
}
```

The JSON is **partial-merge friendly**: you only need to include the keys you
want to override from the defaults.

### Key parameters

| Key | Description |
|-----|-------------|
| `instance_path` | Path to the P2 instance JSON (required). |
| `output_dir` | Directory where results are written. |
| `mode` | `"surrogate"` (fast, no simulator) or `"cooja"` (SSH to Cooja containers). |
| `partition_method` | `"grid"` \| `"kmeans"` \| `"radial_to_sink"` |
| `num_blocks` | Number of structural blocks `m`. |
| `hstar_method` | `"structural_greedy"` \| `"dense_local"` \| `"external"` |
| `hlocal_method` | `"deceptive_low_cost"` \| `"redundant_local"` \| `"far_from_sink"` \| `"random_competitor"` |
| `complement_method` | `"bernoulli"` \| `"feasible_repair"` \| `"population_sample"` |
| `num_complements` | Number of complement vectors `R` per block. |
| `seeds` | List of simulation seeds (one Cooja run per seed per solution). |
| `firmware_local_dir` | Local path to firmware directory. Files are uploaded to the remote workdir for each simulation. Omit to use firmware already installed on the remote host. |

---

## 4. Running

```bash
# Surrogate (no Cooja — fast validation)
p2-popest surrogate_run.json

# Cooja mode
p2-popest cooja_run.json

# Override output directory without editing the file
p2-popest cooja_run.json --output-dir results/run_002 --force
```

### Surrogate example config

```json
{
  "instance_path": "examples/ind2.json",
  "output_dir":    "results/surrogate_001",
  "mode":          "surrogate",
  "num_blocks":    8,
  "num_complements": 30,
  "seeds":         [336157, 667370, 35239, 873465],
  "random_seed":   42
}
```

### Cooja example config

```json
{
  "instance_path": "examples/simple.json",
  "output_dir":    "results/cooja_001",
  "mode":          "cooja",
  "num_blocks":    4,
  "num_complements": 30,
  "seeds":         [336157, 667370, 35239, 873465],
  "random_seed":   42,

  "ssh_host":     "localhost",
  "ssh_user":     "root",
  "ssh_password": "root",
  "ssh_ports":    [2231, 2232, 2233, 2234, 2235, 2236],

  "simulation_duration":  180,
  "simulation_timeout":   900,
  "firmware_local_dir":   "/path/to/wsn-dse/batch_runner/firmware/rpl-udp-csma"
}
```

The surrogate is Cooja-free and deterministic — run it first to validate the
pipeline before paying the cost of real simulation.

---

## 5. Input format

The package accepts the SimLab P2 JSON schema (root-level or wrapped in
`parameters`). See [`examples/ind2.json`](examples/ind2.json) for a complete
working instance.

```json
{
  "problem": {
    "name": "...",
    "radius_of_reach": 50,
    "radius_of_inter": 100,
    "region": [xmin, ymin, xmax, ymax],
    "sink": [x, y],
    "candidates": [[x1, y1], [x2, y2], "..."],
    "mobile_nodes": [
      {
        "name": "...",
        "speed": 5,
        "time_step": 1,
        "is_closed": true,
        "is_round_trip": false,
        "path_segments": [["x_expr(t)", "y_expr(t)"], "..."]
      }
    ]
  }
}
```

Trajectory expressions evaluate `t ∈ [0, 1]` per segment and may use
`np.cos`, `np.sin`, `np.pi`, etc. (restricted namespace, no Python builtins).

---

## 6. Cooja execution model

Each SSH port gets one dedicated worker thread. Simulations are dispatched via
a queue: at most one simulation runs per container at a time. Failures are
retried up to `max_retries` times before the task is marked failed.

Per simulation the worker:

1. Calls `prepare_local` to build local files (`simulation.csc`, `positions.dat`,
   and optionally firmware files copied from `firmware_local_dir`).
2. Opens an SFTP connection and uploads all files to a per-run remote workdir
   (e.g. `/tmp/popest/<solution_id>-<seed>/`).
3. Executes the Cooja command via SSH with `get_pty=True` and polls for
   completion.
4. Downloads `COOJA.testlog` from the remote host and saves it locally as
   `cooja.log`.
5. Parses the log for JSON metrics emitted by the firmware.

When `firmware_local_dir` is set, the generated `simulation.csc` points
`<source>` at the uploaded workdir and injects `CONTIKI=/opt/contiki-ng` into
the make command (overriding the Makefile's relative `CONTIKI=../..`).

---

## 7. Interpreting outputs

### `population_estimate_result.json`

| Key | Meaning |
|-----|---------|
| `experiment_config` | Snapshot of the config used. |
| `instance_summary` | Counts and metadata of the loaded instance. |
| `partition_summary` | Block sizes and candidate indices. |
| `block_results` | Per-block `d_i_hat`, `σ_BB_i_hat`, `n_i_hat`, status, samples. |
| `global_estimate` | `n_hat_uniform`, `n_hat_bernoulli`, valid/invalid counts. |
| `failed_evaluations` | Solutions that raised during evaluation. |
| `warnings` | Warnings, including the heuristic disclaimer. |
| `reproducibility_info` | Package version, seeds, instance SHA-256. |

### `block_results.csv`

One row per block with the main scalar outputs (no sample lists).

### Per-block statuses

| Status | Meaning |
|--------|---------|
| `ok` | Valid estimate. |
| `degenerate_zero_variance` | `n_i_hat` computed using a small variance floor. |
| `invalid_non_positive_d` | `H_i^*` was not consistently better than `H_i^L`. Revise heuristics. |
| `insufficient_samples` | `R < 2`, variance cannot be estimated. |

---

## 8. Limitations

- `H_i^*` and `H_i^L` are *heuristic*. Run with multiple partitions and
  heuristics before trusting a single `n_hat`.
- The surrogate captures structural coverage only, not MAC/routing behaviour.
  Use it for pipeline validation and fast iteration.
- The gambler-ruin formula assumes block-level independence. Strongly-coupled
  problems yield an optimistic estimate.
- `α` is a per-block failure probability. Multiple-testing correction is not
  applied.

---

## 9. Project layout

```
pop-estimator/
  p2_population_estimator/
    cli.py              # JSON-driven entrypoint
    experiment.py       # Pipeline orchestration
    models.py           # Dataclasses (ExperimentConfig, P2Problem, ...)
    io.py               # Instance loading, JSON/CSV output
    partitioning.py     # Grid / k-means / radial partition methods
    blocks.py           # H_star / H_local builders
    complements.py      # Complement generation
    estimator.py        # Gambler-ruin formula, global aggregation
    evaluation/
      base.py           # BaseEvaluator ABC
      surrogate.py      # Structural coverage surrogate
      cooja.py          # Cooja file generation + CoojaEvaluator
      ssh_pool.py       # SSH worker pool (paramiko)
      parser.py         # COOJA.testlog metric parser
    config.py
    geometry.py
    statistics.py
    logging_utils.py
  tests/
  examples/
    ind2.json
    simple.json
```

## 10. Running tests

```bash
cd pop-estimator
pytest -v
```
