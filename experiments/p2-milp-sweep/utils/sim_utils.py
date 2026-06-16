"""Trajectory generation utilities.

Ported from wsn-design-space-exploration/milp/mobile-model/utils/sim_utils.py
and adapted to accept the P2 field names (path_segments, is_round_trip).
"""

import json
import math
import numpy as np


# ---------------------------------------------------------------------------
# Expression evaluation
# ---------------------------------------------------------------------------

def _safe_eval_expr(expr: str, t: float) -> float:
    """Evaluate a parametric string expression at scalar t ∈ [0, 1].

    Allows np.* and math.*; no dangerous builtins.
    """
    try:
        return float(expr)
    except (ValueError, TypeError):
        pass
    allowed_globals = {"__builtins__": {}, "np": np, "math": math}
    allowed_locals  = {"t": float(t)}
    return float(eval(expr, allowed_globals, allowed_locals))  # noqa: S307


# ---------------------------------------------------------------------------
# Arc length
# ---------------------------------------------------------------------------

def _segment_length(function_pair, nsamples: int = 200) -> float:
    """Approximate arc length of the segment (x_expr, y_expr) with t ∈ [0, 1]."""
    x_expr, y_expr = function_pair
    ts  = np.linspace(0.0, 1.0, nsamples + 1)
    pts = np.stack(
        [[_safe_eval_expr(str(x_expr), tt), _safe_eval_expr(str(y_expr), tt)]
         for tt in ts],
        axis=0,
    )
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


# ---------------------------------------------------------------------------
# Integer proportional allocation
# ---------------------------------------------------------------------------

def _distribute_integer_proportions(total_steps: int, weights) -> list[int]:
    """Distribute total_steps across len(weights) buckets proportionally.

    Uses floor-then-largest-remainder to ensure the sum is exact.
    """
    w = np.asarray(weights, dtype=float)
    if len(w) == 0:
        return []
    if np.all(w <= 0):
        base = total_steps // len(w)
        rem  = total_steps % len(w)
        steps = [base] * len(w)
        for i in range(rem):
            steps[i] += 1
        return steps
    w   = np.maximum(w, 0.0)
    W   = float(np.sum(w))
    raw = (total_steps * w / W) if W > 0 else np.zeros_like(w)
    flo = np.floor(raw).astype(int)
    rem = int(total_steps - np.sum(flo))
    frac = raw - flo
    order = np.argsort(-frac)
    steps = flo.tolist()
    for k in range(rem):
        steps[order[k]] += 1
    return steps


# ---------------------------------------------------------------------------
# Trajectory function builder
# ---------------------------------------------------------------------------

def make_mobile_trajectory_fn(path_segments, is_closed: bool, is_roundtrip: bool,
                               T: int, speed: float):
    """Return r(tau: int) -> np.ndarray([x, y]) for tau = 1 … T.

    Parameters
    ----------
    path_segments : list of [x_expr, y_expr]
        Parametric segment expressions; t ∈ [0, 1] per segment.
    is_closed : bool
        If True the path cycles back to the start (only forward direction).
    is_roundtrip : bool
        If True (and not is_closed) the path is traversed forward then backward.
    T : int
        Total number of discrete timesteps.
    speed : float
        Travel speed (same units as the coordinate system).
    """
    K = len(path_segments)
    if K == 0:
        raise ValueError("path_segments is empty.")

    lens_by_k = [_segment_length(path_segments[k]) for k in range(K)]

    # Build the sequence of (segment_index, direction) legs
    if is_closed:
        seq = [(k, +1) for k in range(K)]
    elif is_roundtrip and K >= 1:
        seq  = [(k, +1) for k in range(K)]
        seq += [(k, -1) for k in range(K - 1, -1, -1)]
    else:
        seq = [(k, +1) for k in range(K)]

    spd = 1.0 if (speed is None or speed <= 0) else float(speed)
    times_eff = [lens_by_k[k] / spd for (k, _) in seq]

    steps_per_leg = _distribute_integer_proportions(T, times_eff)

    # Guarantee at least 1 step per leg (when T ≥ number of legs)
    if T >= len(seq):
        steps_per_leg = [max(1, s) for s in steps_per_leg]
        surplus = int(sum(steps_per_leg) - T)
        if surplus > 0:
            order = np.argsort(times_eff)
            for idx in order:
                if surplus == 0:
                    break
                if steps_per_leg[idx] > 1:
                    steps_per_leg[idx] -= 1
                    surplus -= 1
        elif surplus < 0:
            deficit = -surplus
            order   = np.argsort(-np.asarray(times_eff))
            for k in range(deficit):
                steps_per_leg[order[k % len(seq)]] += 1

    cut = np.cumsum([0] + steps_per_leg)
    S   = len(seq)

    def r_of_tau(tau: int) -> np.ndarray:
        u   = tau - 1
        leg = int(np.searchsorted(cut, u, side="right") - 1)
        leg = min(max(leg, 0), S - 1)
        local_len = steps_per_leg[leg]
        tloc = 1.0 if local_len <= 1 else (u - cut[leg]) / (local_len - 1)
        k, direc = seq[leg]
        teff = tloc if direc == +1 else (1.0 - tloc)
        x_expr, y_expr = path_segments[k]
        return np.array(
            [_safe_eval_expr(str(x_expr), teff),
             _safe_eval_expr(str(y_expr), teff)],
            dtype=float,
        )

    return r_of_tau
