"""JSON I/O and instance validation.

We expect the SimLab-style schema:

    {
      "parameters": {
        "problem": {
          "name": "...",
          "radius_of_reach": <float>,
          "radius_of_inter": <float>,
          "region": [xmin, ymin, xmax, ymax],
          "sink": [x, y],
          "candidates": [[x,y], ...],
          "mobile_nodes": [ { ... }, ... ]
        }
      }
    }

For convenience we also accept a top-level "problem" key (i.e. without the
"parameters" wrapper), since several P2 example files exist in both shapes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from p2_population_estimator.models import (
    MobileNode,
    P2Instance,
    P2Problem,
    Point,
    Region,
)


class InstanceValidationError(ValueError):
    """Raised when the input JSON does not match the P2 schema."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _locate_problem_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Return the inner ``problem`` dict, accepting both shapes."""
    if "problem" in data and isinstance(data["problem"], dict):
        return data["problem"]
    if "parameters" in data and isinstance(data["parameters"], dict):
        params = data["parameters"]
        if "problem" in params and isinstance(params["problem"], dict):
            return params["problem"]
    raise InstanceValidationError(
        "Missing 'problem' key (looked at root and at 'parameters.problem')."
    )


def _as_point(value: Any, *, what: str) -> Point:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise InstanceValidationError(f"{what} must be a 2D point [x, y], got: {value!r}")
    try:
        return Point(float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise InstanceValidationError(f"{what} has non-numeric coordinates: {value!r}") from exc


def _as_region(value: Any) -> Region:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise InstanceValidationError(
            f"'region' must be a 4-element list [xmin, ymin, xmax, ymax], got: {value!r}"
        )
    try:
        xmin, ymin, xmax, ymax = (float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise InstanceValidationError(f"'region' has non-numeric entries: {value!r}") from exc
    if not (xmax > xmin and ymax > ymin):
        raise InstanceValidationError(
            f"'region' must satisfy xmax>xmin and ymax>ymin, got: {value!r}"
        )
    return Region(xmin, ymin, xmax, ymax)


def _as_mobile(node: dict[str, Any], *, idx: int) -> MobileNode:
    if not isinstance(node, dict):
        raise InstanceValidationError(f"mobile_nodes[{idx}] must be an object, got {type(node)}")
    if "path_segments" not in node:
        raise InstanceValidationError(f"mobile_nodes[{idx}] is missing 'path_segments'")
    segs_raw = node["path_segments"]
    if not isinstance(segs_raw, list):
        raise InstanceValidationError(
            f"mobile_nodes[{idx}].path_segments must be a list, got {type(segs_raw)}"
        )
    segments: list[tuple[str, str]] = []
    for j, seg in enumerate(segs_raw):
        if not isinstance(seg, (list, tuple)) or len(seg) != 2:
            raise InstanceValidationError(
                f"mobile_nodes[{idx}].path_segments[{j}] must be [expr_x, expr_y]"
            )
        sx, sy = seg
        if not isinstance(sx, str) or not isinstance(sy, str):
            raise InstanceValidationError(
                f"mobile_nodes[{idx}].path_segments[{j}] must contain string expressions"
            )
        segments.append((sx, sy))
    return MobileNode(
        name=str(node.get("name", f"mobile_{idx}")),
        speed=float(node.get("speed", 1.0)),
        time_step=float(node.get("time_step", 1.0)),
        is_closed=bool(node.get("is_closed", False)),
        is_round_trip=bool(node.get("is_round_trip", False)),
        path_segments=segments,
        source_code=node.get("source_code"),
        raw=dict(node),
    )


def parse_instance(data: dict[str, Any], *, source_path: str | None = None) -> P2Instance:
    """Validate the parsed JSON dict and return a :class:`P2Instance`."""
    problem_dict = _locate_problem_dict(data)

    # candidates -------------------------------------------------------------
    cands_raw = problem_dict.get("candidates")
    if not isinstance(cands_raw, list) or not cands_raw:
        raise InstanceValidationError("'candidates' must be a non-empty list of 2D points.")
    candidates = [_as_point(c, what=f"candidates[{i}]") for i, c in enumerate(cands_raw)]

    # sink -------------------------------------------------------------------
    if "sink" not in problem_dict:
        raise InstanceValidationError("'sink' is missing.")
    sink = _as_point(problem_dict["sink"], what="sink")

    # radius_of_reach --------------------------------------------------------
    if "radius_of_reach" not in problem_dict:
        raise InstanceValidationError("'radius_of_reach' is missing.")
    try:
        radius_reach = float(problem_dict["radius_of_reach"])
    except (TypeError, ValueError) as exc:
        raise InstanceValidationError("'radius_of_reach' must be numeric.") from exc
    if radius_reach <= 0:
        raise InstanceValidationError("'radius_of_reach' must be positive.")

    # radius_of_inter --------------------------------------------------------
    radius_inter = float(problem_dict.get("radius_of_inter", radius_reach))
    if radius_inter <= 0:
        raise InstanceValidationError("'radius_of_inter' must be positive when provided.")

    # region -----------------------------------------------------------------
    if "region" not in problem_dict:
        raise InstanceValidationError("'region' is missing.")
    region = _as_region(problem_dict["region"])

    # mobile_nodes -----------------------------------------------------------
    if "mobile_nodes" not in problem_dict:
        raise InstanceValidationError("'mobile_nodes' is missing (use [] for an empty list).")
    mobs_raw = problem_dict["mobile_nodes"]
    if not isinstance(mobs_raw, list):
        raise InstanceValidationError("'mobile_nodes' must be a list (possibly empty).")
    mobile_nodes = [_as_mobile(m, idx=i) for i, m in enumerate(mobs_raw)]

    # Extras -----------------------------------------------------------------
    known = {
        "name", "radius_of_reach", "radius_of_inter", "region", "sink",
        "candidates", "mobile_nodes",
    }
    extras = {k: v for k, v in problem_dict.items() if k not in known}

    problem = P2Problem(
        name=str(problem_dict.get("name", "p2-instance")),
        radius_of_reach=radius_reach,
        radius_of_inter=radius_inter,
        region=region,
        sink=sink,
        candidates=candidates,
        mobile_nodes=mobile_nodes,
        extras=extras,
    )
    return P2Instance(problem=problem, raw=data, source_path=source_path)


def load_instance(path: str | Path) -> P2Instance:
    """Load and validate a P2 instance from a JSON file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Instance file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise InstanceValidationError("Top-level JSON must be an object.")
    return parse_instance(data, source_path=str(p.resolve()))


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def write_json(path: str | Path, payload: Any, *, force: bool = False) -> Path:
    """Write JSON, refusing to overwrite by default."""
    p = Path(path)
    if p.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {p}. Pass force=True (or "
            "--force on the CLI) to overwrite."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=_json_default)
    return p


def _json_default(o: Any) -> Any:
    # Best-effort fallback so dataclasses serialise via dict()-like access.
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    if hasattr(o, "as_tuple"):
        return o.as_tuple()
    raise TypeError(f"Cannot JSON-serialise {type(o).__name__}")


def write_csv(path: str | Path, rows: list[dict[str, Any]], *, force: bool = False) -> Path:
    """Write a list of dicts as CSV. The header is the union of keys."""
    import csv

    p = Path(path)
    if p.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return p
