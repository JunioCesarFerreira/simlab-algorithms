"""Convert a P2 instance JSON into the internal objects needed by the MILP model.

P2 instance schema (parameters.problem):
  radius_of_reach : float
  radius_of_inter : float
  region          : [xmin, ymin, xmax, ymax]
  sink            : [x, y]
  candidates      : [[x, y], ...]           -- N entries
  mobile_nodes    : list of {
      name, speed, time_step, is_closed, is_round_trip, path_segments
  }

Internal representation (P2MilpInputs):
  p_sink         : np.ndarray (2,)
  J              : list of ("j", f"cand{i}") keys
  p_cand         : dict  {("j", f"cand{i}"): np.ndarray([x, y])}
  mob_names      : list[str]
  r_mobile_fns   : dict  {name: callable(tau: int) -> np.ndarray([x, y])}
  T              : int   (time horizon = duration // min(time_step))
  R_comm         : float
  R_interf       : float
  region         : list[float]
  duration       : int
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from utils.sim_utils import make_mobile_trajectory_fn


@dataclass
class P2MilpInputs:
    p_sink:       np.ndarray
    J:            list
    p_cand:       dict
    mob_names:    list
    r_mobile_fns: dict
    T:            int
    R_comm:       float
    R_interf:     float
    region:       list
    duration:     int


def load_p2_instance(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def adapt(data: dict) -> P2MilpInputs:
    """Convert raw P2 JSON dict to P2MilpInputs.

    Accepts both the nested form {"parameters": {"problem": ...}}
    and the flat form {"problem": ...}.
    """
    if "parameters" in data:
        sim_params = data["parameters"].get("simulation", {})
        prob       = data["parameters"]["problem"]
    elif "problem" in data:
        sim_params = data.get("simulation", {})
        prob       = data["problem"]
    else:
        # Assume data IS the problem block (minimal form)
        sim_params = {}
        prob       = data

    duration = int(sim_params.get("duration", 180))
    R_comm   = float(prob["radius_of_reach"])
    R_interf = float(prob.get("radius_of_inter", R_comm * 2))
    region   = list(prob["region"])

    # --- Sink ---
    p_sink = np.array(prob["sink"], dtype=float)

    # --- Candidates ---
    cands_raw = prob["candidates"]
    J      = [("j", f"cand{i}") for i in range(len(cands_raw))]
    p_cand = {J[i]: np.array(cands_raw[i], dtype=float)
              for i in range(len(cands_raw))}

    # --- Mobile nodes ---
    mobile_nodes = prob["mobile_nodes"]
    time_steps   = [int(m.get("time_step", 1)) for m in mobile_nodes]
    dt           = max(1, min(time_steps) if time_steps else 1)
    T            = max(1, duration // dt)

    mob_names    = [m["name"] for m in mobile_nodes]
    r_mobile_fns: dict = {}
    for m in mobile_nodes:
        name         = m["name"]
        path_segs    = m["path_segments"]
        is_closed    = bool(m.get("is_closed",    False))
        is_roundtrip = bool(m.get("is_round_trip", False))
        speed        = float(m.get("speed", 1.0))
        r_mobile_fns[name] = make_mobile_trajectory_fn(
            path_segs, is_closed, is_roundtrip, T, speed
        )

    return P2MilpInputs(
        p_sink       = p_sink,
        J            = J,
        p_cand       = p_cand,
        mob_names    = mob_names,
        r_mobile_fns = r_mobile_fns,
        T            = T,
        R_comm       = R_comm,
        R_interf     = R_interf,
        region       = region,
        duration     = duration,
    )
