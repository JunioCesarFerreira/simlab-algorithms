"""Mobile-coverage MILP model for a P2 WSN instance.

Uses PuLP as the modelling layer so the code is solver-agnostic.
Solver priority: Gurobi (if licence available) > HiGHS > bundled CBC.

Formulation
-----------
Decision variables
  y_j ∈ {0,1}          relay j installed  (j ∈ J = fixed candidates)
  z_{ij}(t) ∈ {0,1}    edge (i,j) active at timestep t
  x_{ij}(t) ≥ 0        flow on (i,j) at t

Objective
  min  w_install · Σ_j y_j  +  Σ_t Σ_{(i,j)∈E_t} d²_{ij}(t) · x_{ij}(t)

Constraints
  (1) Capacity      x ≤ C_{ij}(t) · z,  C = C0·max{0, 1−k·d/R}²
  (2) Installation  z_{ij}(t) ≤ y_i  (i∈J);  z_{ij}(t) ≤ y_j  (j∈J)
  (3) Mobile flow   outflow − inflow = B   ∀ mobile m, ∀ t
  (4) Fixed flow    outflow − inflow = 0   ∀ candidate j, ∀ t
  (5) Sink balance  Σ_in(sink,t) = M·B     ∀ t

Performance note
----------------
``precompute_topology`` builds E_t (edges within R_comm) once from the
geometry.  The topology never changes across the parameter sweep (E_t depends
only on positions and R_comm, not on C0/k_decay/B).  Pass the returned
``Topology`` object to every ``solve()`` call to avoid repeating this O(K²·T)
scan for each of the 48–2750 sweep points.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import pulp
except ImportError as exc:
    raise ImportError(
        "pulp is required.  Install with:  pip install 'pulp>=2.7'"
    ) from exc

from milp.instance_adapter import P2MilpInputs


# ---------------------------------------------------------------------------
# Solver selection
# ---------------------------------------------------------------------------

def _select_solver(time_limit: float | None, verbose: bool):
    """Return the best available PuLP solver in priority order.

    Priority: Gurobi Python API  →  Gurobi CMD  →  HiGHS  →  CBC (bundled).
    """
    msg = 1 if verbose else 0
    tl  = {"timeLimit": float(time_limit)} if time_limit is not None else {}

    candidates = [
        ("Gurobi",    lambda: pulp.GUROBI(msg=msg, **tl)),
        ("GurobiCMD", lambda: pulp.GUROBI_CMD(msg=msg, **tl)),
        ("HiGHS",     lambda: pulp.HiGHS(msg=msg, **tl)),       # Python API (highspy)
        ("HiGHS_CMD", lambda: pulp.HiGHS_CMD(msg=msg, **tl)),   # command-line fallback
        ("CBC",       lambda: pulp.PULP_CBC_CMD(msg=msg, **tl)),
    ]

    for name, factory in candidates:
        try:
            s = factory()
            if s.available():   # True = yes; None/False = no
                return s, name
        except Exception:
            pass

    raise RuntimeError(
        "No MILP solver found.  Install one of:  gurobipy, highspy, or pulp (bundles CBC)."
    )


def detect_solver_name() -> str:
    """Return the name of the solver that will be used (for display)."""
    try:
        _, name = _select_solver(None, False)
        return name
    except RuntimeError:
        return "none"


# ---------------------------------------------------------------------------
# Short node identifier for PuLP variable/constraint names
# ---------------------------------------------------------------------------

def _nid(n: tuple) -> str:
    """("j", "cand7") → "j7",  ("m", "node2") → "m2",  ("sink", "root") → "s"."""
    kind, label = n
    if kind == "sink":
        return "s"
    suffix = (label
              .replace("cand", "")
              .replace("node", "")
              .replace("root", ""))
    return f"{kind[0]}{suffix}"   # "j7", "m2"


# ---------------------------------------------------------------------------
# Topology (precomputed once, reused across all sweep runs)
# ---------------------------------------------------------------------------

@dataclass
class Topology:
    """Per-timestep edge structure derived purely from geometry and R_comm.

    E_t[t]      : list of directed pairs (i, j) with 0 < d ≤ R_comm
    dist[i,j,t] : Euclidean distance  (float)
    """
    E_t:  dict  = field(default_factory=dict)
    dist: dict  = field(default_factory=dict)


def precompute_topology(inp: P2MilpInputs, sample_step: int = 1) -> Topology:
    """Build E_t and distances for a (sub)sample of timesteps.

    Parameters
    ----------
    sample_step : int
        Include only every sample_step-th timestep (1 = full, 10 = every 10th).
        Mobile trajectories are smooth/periodic so a coarse sample captures the
        same relay-installation decisions at a fraction of the MILP size.
        The sampled timesteps are t = 1, 1+step, 1+2*step, …

    Call once per instance; pass the returned Topology to every solve() call.
    """
    sink      = ("sink", "root")
    all_nodes = [sink] + inp.J + [("m", name) for name in inp.mob_names]

    def pos(n, t):
        if n[0] == "sink": return inp.p_sink
        if n[0] == "j":    return inp.p_cand[n]
        return inp.r_mobile_fns[n[1]](t)

    E_t:  dict[int, list] = {}
    dist: dict            = {}

    step = max(1, int(sample_step))
    R    = inp.R_comm
    sampled = range(1, inp.T + 1, step)

    for t in sampled:
        E_t[t] = []
        for i in all_nodes:
            pi = pos(i, t)
            for j in all_nodes:
                if i == j:
                    continue
                pj = pos(j, t)
                d  = float(np.linalg.norm(pi - pj))
                if 0.0 < d <= R:
                    E_t[t].append((i, j))
                    dist[(i, j, t)] = d

    return Topology(E_t=E_t, dist=dist)


# ---------------------------------------------------------------------------
# Binary chromosome
# ---------------------------------------------------------------------------

def binary_chromosome(y_val: dict, J: list) -> str:
    J_sorted = sorted(J, key=lambda j: j[1])
    return "".join("1" if (y_val.get(j) or 0.0) > 0.5 else "0" for j in J_sorted)


# ---------------------------------------------------------------------------
# MILP solve
# ---------------------------------------------------------------------------

def solve(
    inp:  P2MilpInputs,
    topo: Topology,
    *,
    C0:         float,
    kdecay:     float,
    B:          float,
    w_install:  float = 1_000_000.0,
    time_limit: float | None = None,
    verbose:    bool  = False,
) -> dict[str, Any]:
    """Solve the mobile-coverage MILP for one (C0, kdecay, B) point.

    Parameters
    ----------
    inp, topo    : instance data and precomputed topology
    C0, kdecay   : capacity function parameters
    B            : demand per mobile per timestep
    w_install    : installation penalty weight
    time_limit   : seconds per run (None = unlimited)
    verbose      : show solver log

    Returns
    -------
    dict with keys: C0, k_decay, B, w_install, solver,
      status, status_name, runtime_seconds, solution_count,
      variables, constraints,
      objective_value, mip_gap, installed_nodes, chromosome, y_val
    """
    J          = inp.J
    mob_names  = inp.mob_names
    T          = inp.T
    R_comm     = inp.R_comm
    sink       = ("sink", "root")

    # Only iterate over the timesteps actually present in the topology
    # (with sample_step > 1 this is a subset of 1..T)
    sampled_ts = sorted(topo.E_t.keys())

    # Capacity values for this (C0, kdecay)
    C: dict[tuple, float] = {}
    for (i, j, t), d in topo.dist.items():
        cap = C0 * max(0.0, 1.0 - kdecay * d / R_comm) ** 2
        if cap > 0.0:
            C[(i, j, t)] = cap

    # Only keep edges that have positive capacity
    E_t_cap: dict[int, list] = {t: [] for t in sampled_ts}
    for (i, j, t), cap in C.items():
        E_t_cap[t].append((i, j))

    # ── PuLP model ────────────────────────────────────────────────────────────
    prob = pulp.LpProblem("WSN_P2", pulp.LpMinimize)

    # Decision variables
    y    = {j: pulp.LpVariable(f"y_{_nid(j)}", cat="Binary") for j in J}
    z    = {(i, j, t): pulp.LpVariable(f"z_{_nid(i)}_{_nid(j)}_t{t}", cat="Binary")
            for t in sampled_ts for (i, j) in E_t_cap[t]}
    xvar = {(i, j, t): pulp.LpVariable(f"x_{_nid(i)}_{_nid(j)}_t{t}", lowBound=0.0)
            for t in sampled_ts for (i, j) in E_t_cap[t]}

    # Objective: installation + routing energy
    prob += (
        w_install * pulp.lpSum(y[j] for j in J)
        + pulp.lpSum(
            topo.dist[(i, j, t)] ** 2 * xvar[(i, j, t)]
            for t in sampled_ts
            for (i, j) in E_t_cap[t]
        )
    )

    # (1) Capacity
    for t in sampled_ts:
        for (i, j) in E_t_cap[t]:
            prob += xvar[(i, j, t)] <= C[(i, j, t)] * z[(i, j, t)]

    # (2) Installation guards
    for t in sampled_ts:
        for (i, j) in E_t_cap[t]:
            if i[0] == "j":
                prob += z[(i, j, t)] <= y[i]
            if j[0] == "j":
                prob += z[(i, j, t)] <= y[j]

    # (3) Mobile flow conservation
    for t in sampled_ts:
        for name in mob_names:
            m = ("m", name)
            out_m = pulp.lpSum(xvar[(m, j, t)] for (i, j) in E_t_cap[t] if i == m)
            in_m  = pulp.lpSum(xvar[(i, m, t)] for (i, j) in E_t_cap[t] if j == m)
            prob += out_m - in_m == float(B)

    # (4) Fixed node flow conservation
    for t in sampled_ts:
        for j_node in J:
            out_j = pulp.lpSum(xvar[(j_node, v, t)] for (u, v) in E_t_cap[t] if u == j_node)
            in_j  = pulp.lpSum(xvar[(u, j_node, t)] for (u, v) in E_t_cap[t] if v == j_node)
            prob += out_j - in_j == 0.0

    # (5) Sink balance
    for t in sampled_ts:
        in_sink = pulp.lpSum(xvar[(i, sink, t)] for (i, j) in E_t_cap[t] if j == sink)
        prob += in_sink == float(B) * len(mob_names)

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver, solver_name = _select_solver(time_limit, verbose)

    t0        = time.perf_counter()
    lp_status = prob.solve(solver)
    rt        = time.perf_counter() - t0

    # Normalise PuLP status  (-1=Infeasible, 1=Optimal, 0=Not Solved, ...)
    # to Gurobi-compatible integers for the stopping rules in sweep.py:
    #   2 = OPTIMAL  |  3 = INFEASIBLE  |  other = non-productive
    _STATUS_MAP = {
         1: (2,  "OPTIMAL"),
        -1: (3,  "INFEASIBLE"),
        -2: (5,  "UNBOUNDED"),
        -3: (4,  "UNDEFINED"),
         0: (0,  "NOT_SOLVED"),
    }
    status_int, status_name = _STATUS_MAP.get(
        lp_status,
        (lp_status, pulp.LpStatus.get(lp_status, str(lp_status))),
    )

    has_sol = lp_status == 1 and pulp.value(prob.objective) is not None

    record: dict[str, Any] = {
        "C0":              float(C0),
        "k_decay":         float(kdecay),
        "B":               float(B),
        "w_install":       float(w_install),
        "solver":          solver_name,
        "status":          status_int,
        "status_name":     status_name,
        "runtime_seconds": round(rt, 3),
        "solution_count":  1 if has_sol else 0,
        "variables":       len(prob.variables()),
        "constraints":     len(prob.constraints),
        "objective_value": None,
        "mip_gap":         None,
        "installed_nodes": None,
        "chromosome":      None,
        "y_val":           None,
    }

    if has_sol:
        y_val = {j: float(pulp.value(y[j]) or 0.0) for j in J}
        chrom = binary_chromosome(y_val, J)
        record.update({
            "objective_value": float(pulp.value(prob.objective)),
            "installed_nodes": int(sum(1 for j in J if y_val[j] > 0.5)),
            "chromosome":      chrom,
            "y_val":           y_val,
        })

    return record
