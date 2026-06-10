"""Temporal full-network adjacency matrices for WSN instances.

For each discrete timestep ``t`` of the simulation, we build a single
symmetric binary adjacency matrix ``A(t)`` that covers **the entire
network** — sink, candidate relays, and mobile sensors. The matrix is
square of shape ``(K, K)`` with

    K = 1 + N + M

where ``N`` is the number of candidate relays (``problem.candidates``)
and ``M`` is the number of mobile sensors (``problem.mobile_nodes``).

Node ordering
-------------
Both axes share the same canonical ordering::

    index 0             -> sink
    indices 1 .. N      -> candidate relays  (order of problem.candidates)
    indices N+1 .. N+M  -> mobile sensors    (order of problem.mobile_nodes)

For every timestep ``t`` and for every pair ``(u, v)``::

    A(t)[u, v] = 1   iff   u != v  and  dist(node_u(t), node_v(t)) <= R

with ``R = problem.radius_of_reach`` and Euclidean 2D distance. The
matrix is symmetric and has zero diagonal (no self-loops). Sink and
candidates are static, so the corresponding sub-blocks are constant
across time; the mobile rows/columns are the only ones that vary.

Pipeline outputs
----------------
    tensor       — (T, K, K) uint8   per-timestep adjacency
    accumulated  — (K, K)    int64   sum_t A(t), the co-occurrence matrix
    positions    — (T, K, 2) float   per-timestep node positions

JSON schema
-----------
Compatible with the SimLab P1/P2 instance format. Accepts both shapes
(``{"problem": ...}`` and ``{"parameters": {"problem": ...}}``):

    {
      "parameters": {
        "problem": {
          "radius_of_reach": <float>,
          "sink":            [x, y],
          "candidates":      [[x, y], ...],
          "mobile_nodes":    [
            {
              "name": ...,
              "speed": <float>,
              "time_step": <float>,
              "is_closed": <bool>,
              "is_round_trip": <bool>,
              "path_segments": [[x_expr, y_expr], ...]
            },
            ...
          ]
        }
      }
    }

Time discretisation
-------------------
For each mobile we follow the same procedure used by SimLab's Cooja
position generator (``simlab/pylib/cooja_builder/parse_json_pos_dat.py``):

    1. evaluate each parametric segment ``(x(t), y(t))`` at a dense
       reference grid (``_ARC_REF_SAMPLES`` points);
    2. estimate per-segment arc length and the total length;
    3. ``total_duration = total_length / speed``;
    4. ``n_steps = max(1, int(total_duration / time_step))``;
    5. distribute ``n_steps`` proportionally to each segment and
       interpolate the dense samples to obtain per-step positions;
    6. if ``is_round_trip`` is true, append the reversed sequence.

Mobiles may end up with different ``n_steps`` (different speeds or
trajectory lengths). The global horizon ``T`` is the maximum across
mobiles; shorter sequences are padded with their last position, which
keeps the sensor static after its trajectory ends. ``is_closed`` is
informational here and does **not** force wrap-around; if you need the
trajectory to repeat indefinitely, increase ``time_step`` or replicate
``path_segments`` in the instance.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


_ARC_REF_SAMPLES = 100

_SAFE_NS: dict[str, Any] = {
    "np": np,
    "math": math,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "pi": math.pi, "e": math.e,
    "abs": abs, "min": min, "max": max,
}


class InstanceError(ValueError):
    """Raised when the input JSON does not match the expected schema."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class MobileNodeSpec:
    name: str
    speed: float
    time_step: float
    is_closed: bool
    is_round_trip: bool
    path_segments: list[tuple[str, str]]


@dataclass(slots=True)
class ProblemInstance:
    """Minimal P1/P2-style instance needed to build adjacency matrices."""

    radius_of_reach: float
    candidates: np.ndarray            # shape (N, 2)
    mobile_nodes: list[MobileNodeSpec]
    sink: tuple[float, float] | None = None
    source_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def n_candidates(self) -> int:
        return int(self.candidates.shape[0])

    @property
    def n_mobiles(self) -> int:
        return len(self.mobile_nodes)


@dataclass(slots=True)
class NodeLayout:
    """Index mapping for the unified node ordering ``[sink, candidates, mobiles]``."""

    sink_index: int                  # always 0
    candidate_indices: list[int]     # length N
    mobile_indices: list[int]        # length M
    names: list[str]                 # length K = 1 + N + M

    @property
    def K(self) -> int:
        return len(self.names)

    @property
    def N(self) -> int:
        return len(self.candidate_indices)

    @property
    def M(self) -> int:
        return len(self.mobile_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sink_index": self.sink_index,
            "candidate_indices": list(self.candidate_indices),
            "mobile_indices": list(self.mobile_indices),
            "names": list(self.names),
        }


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------
def _locate_problem(data: dict[str, Any]) -> dict[str, Any]:
    if isinstance(data.get("problem"), dict):
        return data["problem"]
    params = data.get("parameters")
    if isinstance(params, dict) and isinstance(params.get("problem"), dict):
        return params["problem"]
    raise InstanceError(
        "Missing 'problem' key (looked at root and at 'parameters.problem')."
    )


def _as_xy(value: Any, *, what: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise InstanceError(f"{what} must be a 2D point [x, y], got: {value!r}")
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise InstanceError(f"{what} has non-numeric coordinates: {value!r}") from exc


def _as_mobile(node: dict[str, Any], idx: int) -> MobileNodeSpec:
    if not isinstance(node, dict):
        raise InstanceError(f"mobile_nodes[{idx}] must be an object")
    segs_raw = node.get("path_segments")
    if not isinstance(segs_raw, list) or not segs_raw:
        raise InstanceError(
            f"mobile_nodes[{idx}].path_segments must be a non-empty list"
        )
    segments: list[tuple[str, str]] = []
    for j, seg in enumerate(segs_raw):
        if not isinstance(seg, (list, tuple)) or len(seg) != 2:
            raise InstanceError(
                f"mobile_nodes[{idx}].path_segments[{j}] must be [expr_x, expr_y]"
            )
        sx, sy = seg
        if not isinstance(sx, str) or not isinstance(sy, str):
            raise InstanceError(
                f"mobile_nodes[{idx}].path_segments[{j}] must contain string expressions"
            )
        segments.append((sx, sy))
    return MobileNodeSpec(
        name=str(node.get("name", f"mobile_{idx}")),
        speed=float(node.get("speed", 1.0)),
        time_step=float(node.get("time_step", 1.0)),
        is_closed=bool(node.get("is_closed", False)),
        is_round_trip=bool(node.get("is_round_trip", False)),
        path_segments=segments,
    )


def load_instance(path: str | Path) -> ProblemInstance:
    """Load and validate a WSN instance JSON file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Instance file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise InstanceError("Top-level JSON must be an object.")

    problem = _locate_problem(data)

    if "radius_of_reach" not in problem:
        raise InstanceError("'radius_of_reach' is missing from problem.")
    radius = float(problem["radius_of_reach"])
    if radius <= 0:
        raise InstanceError("'radius_of_reach' must be positive.")

    cands_raw = problem.get("candidates")
    if not isinstance(cands_raw, list) or not cands_raw:
        raise InstanceError("'candidates' must be a non-empty list of 2D points.")
    candidates = np.array(
        [_as_xy(c, what=f"candidates[{i}]") for i, c in enumerate(cands_raw)],
        dtype=float,
    )

    mobs_raw = problem.get("mobile_nodes")
    if not isinstance(mobs_raw, list):
        raise InstanceError("'mobile_nodes' must be a list (possibly empty).")
    mobile_nodes = [_as_mobile(m, i) for i, m in enumerate(mobs_raw)]

    sink: tuple[float, float] | None = None
    if "sink" in problem:
        sink = _as_xy(problem["sink"], what="sink")

    return ProblemInstance(
        radius_of_reach=radius,
        candidates=candidates,
        mobile_nodes=mobile_nodes,
        sink=sink,
        source_path=str(p.resolve()),
        raw=data,
    )


# ---------------------------------------------------------------------------
# Trajectory discretisation
# ---------------------------------------------------------------------------
def _eval_expr(expr: str, t: float) -> float:
    try:
        return float(eval(  # noqa: S307 - trusted instance JSON, restricted namespace
            expr, {"__builtins__": {}}, {**_SAFE_NS, "t": float(t)}
        ))
    except Exception as exc:
        raise ValueError(
            f"Failed to evaluate trajectory expression {expr!r} at t={t}: {exc}"
        ) from exc


def _evaluate_segment(x_expr: str, y_expr: str, n: int) -> np.ndarray:
    """Evaluate a parametric segment at ``n`` uniformly spaced t values."""
    ts = np.linspace(0.0, 1.0, n)
    xs = np.array([_eval_expr(x_expr, float(t)) for t in ts], dtype=float)
    ys = np.array([_eval_expr(y_expr, float(t)) for t in ts], dtype=float)
    return np.column_stack([xs, ys])


def discretise_mobile(
    mobile: MobileNodeSpec, *, ref_samples: int = _ARC_REF_SAMPLES
) -> np.ndarray:
    """Return per-timestep positions for one mobile node, shape ``(n_steps, 2)``."""
    if mobile.speed <= 0:
        raise InstanceError(f"mobile {mobile.name!r}: speed must be positive.")
    if mobile.time_step <= 0:
        raise InstanceError(f"mobile {mobile.name!r}: time_step must be positive.")

    seg_dense: list[np.ndarray] = []
    seg_length: list[float] = []
    for x_expr, y_expr in mobile.path_segments:
        pts = _evaluate_segment(x_expr, y_expr, ref_samples)
        diffs = np.diff(pts, axis=0)
        length = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
        seg_dense.append(pts)
        seg_length.append(length)

    total_length = float(sum(seg_length))
    if total_length <= 0.0:
        raise InstanceError(
            f"mobile {mobile.name!r}: trajectory has zero arc length."
        )

    total_duration = total_length / mobile.speed
    n_steps = max(1, int(total_duration / mobile.time_step))

    per_seg_steps = [
        max(1, int(round(n_steps * (l / total_length))))
        for l in seg_length
    ]

    chunks: list[np.ndarray] = []
    for pts, n_here in zip(seg_dense, per_seg_steps):
        ref_t = np.linspace(0.0, 1.0, pts.shape[0])
        target_t = np.linspace(0.0, 1.0, n_here)
        xs = np.interp(target_t, ref_t, pts[:, 0])
        ys = np.interp(target_t, ref_t, pts[:, 1])
        chunks.append(np.column_stack([xs, ys]))

    positions = np.concatenate(chunks, axis=0)

    if mobile.is_round_trip:
        positions = np.concatenate([positions, positions[::-1]], axis=0)

    return positions


def build_mobile_positions(
    instance: ProblemInstance,
    *,
    ref_samples: int = _ARC_REF_SAMPLES,
) -> tuple[np.ndarray, list[int]]:
    """Return a ``(T, M, 2)`` array of mobile positions and per-mobile raw step counts.

    Shorter trajectories are padded with their final position.
    """
    if not instance.mobile_nodes:
        raise InstanceError("Instance has no mobile_nodes; cannot build trajectories.")

    per_mobile = [discretise_mobile(m, ref_samples=ref_samples) for m in instance.mobile_nodes]
    raw_lengths = [int(p.shape[0]) for p in per_mobile]
    T = max(raw_lengths)
    M = len(per_mobile)

    positions = np.empty((T, M, 2), dtype=float)
    for i, pts in enumerate(per_mobile):
        n_i = pts.shape[0]
        positions[:n_i, i, :] = pts
        if n_i < T:
            positions[n_i:, i, :] = pts[-1]
    return positions, raw_lengths


# ---------------------------------------------------------------------------
# Node layout / full-network positions
# ---------------------------------------------------------------------------
def build_node_layout(instance: ProblemInstance) -> NodeLayout:
    """Build the canonical node ordering ``[sink, candidates, mobiles]``."""
    if instance.sink is None:
        raise InstanceError(
            "Instance has no 'sink'; required for the unified (1+N+M) node ordering."
        )
    N = instance.n_candidates
    M = instance.n_mobiles
    names = (
        ["sink"]
        + [f"cand_{i}" for i in range(N)]
        + [m.name for m in instance.mobile_nodes]
    )
    return NodeLayout(
        sink_index=0,
        candidate_indices=list(range(1, 1 + N)),
        mobile_indices=list(range(1 + N, 1 + N + M)),
        names=names,
    )


def build_node_positions(
    instance: ProblemInstance,
    *,
    ref_samples: int = _ARC_REF_SAMPLES,
) -> tuple[np.ndarray, NodeLayout, list[int]]:
    """Return a ``(T, K, 2)`` array of per-timestep node positions.

    Axis 1 follows the canonical ordering ``[sink, candidates, mobiles]``
    (see :class:`NodeLayout`). The sink and candidates are constant across
    time; only mobile rows vary.
    """
    layout = build_node_layout(instance)
    mobile_pos, raw_lengths = build_mobile_positions(instance, ref_samples=ref_samples)
    T = mobile_pos.shape[0]
    K = layout.K
    N = layout.N

    positions = np.empty((T, K, 2), dtype=float)
    positions[:, layout.sink_index, :] = np.asarray(instance.sink, dtype=float)
    positions[:, 1:1 + N, :] = instance.candidates[None, :, :]
    positions[:, 1 + N:, :] = mobile_pos
    return positions, layout, raw_lengths


# ---------------------------------------------------------------------------
# Adjacency computation
# ---------------------------------------------------------------------------
def adjacency_at_time(
    node_positions: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Symmetric binary adjacency for one timestep, shape ``(K, K)``, dtype uint8.

    Parameters
    ----------
    node_positions : ndarray of shape (K, 2)
        All node positions at a single timestep, in the canonical
        ``[sink, candidates, mobiles]`` ordering.
    radius : float
        Communication radius R.

    Returns
    -------
    ndarray of shape (K, K), dtype=uint8, with entries in {0, 1} and zero
    diagonal (no self-loops). Symmetric.
    """
    if node_positions.ndim != 2 or node_positions.shape[1] != 2:
        raise ValueError(f"node_positions must be (K, 2), got {node_positions.shape}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    diff = node_positions[:, None, :] - node_positions[None, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    adj = (dist_sq <= radius * radius).astype(np.uint8)
    np.fill_diagonal(adj, 0)
    return adj


def build_adjacency_tensor(
    node_positions: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Stack per-timestep adjacency matrices into a ``(T, K, K)`` tensor.

    Each slice is symmetric with zero diagonal.
    """
    if node_positions.ndim != 3 or node_positions.shape[2] != 2:
        raise ValueError(f"node_positions must be (T, K, 2), got {node_positions.shape}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    T, K, _ = node_positions.shape
    diff = node_positions[:, :, None, :] - node_positions[:, None, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    tensor = (dist_sq <= radius * radius).astype(np.uint8)
    diag_idx = np.arange(K)
    tensor[:, diag_idx, diag_idx] = 0
    return tensor


def accumulate(tensor: np.ndarray) -> np.ndarray:
    """Sum the per-timestep matrices along the time axis. Returns ``(K, K)`` int64."""
    if tensor.ndim != 3 or tensor.shape[1] != tensor.shape[2]:
        raise ValueError(f"tensor must be (T, K, K), got {tensor.shape}")
    return tensor.sum(axis=0, dtype=np.int64)


# ---------------------------------------------------------------------------
# Pipeline / I/O
# ---------------------------------------------------------------------------
def build_from_instance(
    instance: ProblemInstance,
    *,
    radius: float | None = None,
    ref_samples: int = _ARC_REF_SAMPLES,
) -> dict[str, Any]:
    """Run the full pipeline on a parsed instance.

    Returns a dict with keys:
        ``positions``    — (T, K, 2) per-timestep node positions
                           (sink, candidates, mobiles in this order)
        ``tensor``       — (T, K, K) uint8 symmetric adjacency, zero diagonal
        ``accumulated``  — (K, K)    int64 co-occurrence matrix
        ``layout``       — :class:`NodeLayout` describing the axis ordering
        ``raw_lengths``  — list of per-mobile raw step counts (before padding)
        ``radius``       — communication radius used
    """
    if instance.n_candidates == 0:
        raise InstanceError("Instance has no candidates; nothing to evaluate.")

    R = float(instance.radius_of_reach if radius is None else radius)
    positions, layout, raw_lengths = build_node_positions(instance, ref_samples=ref_samples)
    tensor = build_adjacency_tensor(positions, R)
    total = accumulate(tensor)
    return {
        "positions": positions,
        "tensor": tensor,
        "accumulated": total,
        "layout": layout,
        "raw_lengths": raw_lengths,
        "radius": R,
    }


def _block_stats(total: np.ndarray, rows: list[int], cols: list[int]) -> dict[str, float]:
    if not rows or not cols:
        return {"min": 0, "max": 0, "mean": 0.0, "sum": 0, "density_pct": 0.0}
    sub = total[np.ix_(rows, cols)]
    return {
        "min": int(sub.min()),
        "max": int(sub.max()),
        "mean": float(sub.mean()),
        "sum": int(sub.sum()),
        "density_pct": float(100.0 * (sub > 0).mean()),
    }


def summarise(result: dict[str, Any], instance: ProblemInstance) -> dict[str, Any]:
    """Return a JSON-serialisable summary of a pipeline result."""
    tensor: np.ndarray = result["tensor"]
    total: np.ndarray = result["accumulated"]
    layout: NodeLayout = result["layout"]
    T, K, _ = tensor.shape

    density_per_t = tensor.reshape(T, -1).mean(axis=1)
    s_idx = [layout.sink_index]
    c_idx = layout.candidate_indices
    m_idx = layout.mobile_indices

    return {
        "instance_source": instance.source_path,
        "radius_of_reach": float(result["radius"]),
        "n_timesteps": int(T),
        "n_nodes": int(K),
        "n_candidates": int(layout.N),
        "n_mobiles": int(layout.M),
        "layout": layout.to_dict(),
        "raw_step_counts": list(result["raw_lengths"]),
        "tensor_shape": [int(T), int(K), int(K)],
        "accumulated_shape": [int(K), int(K)],
        "accumulated_stats": {
            "min": int(total.min()),
            "max": int(total.max()),
            "mean": float(total.mean()),
            "sum": int(total.sum()),
            "density_pct": float(100.0 * (total > 0).mean()),
        },
        "block_stats": {
            "sink_x_candidates": _block_stats(total, s_idx, c_idx),
            "sink_x_mobiles":    _block_stats(total, s_idx, m_idx),
            "cand_x_cand":       _block_stats(total, c_idx, c_idx),
            "cand_x_mobiles":    _block_stats(total, c_idx, m_idx),
            "mobile_x_mobile":   _block_stats(total, m_idx, m_idx),
        },
        "per_timestep_density_pct": {
            "min": float(100.0 * density_per_t.min()),
            "max": float(100.0 * density_per_t.max()),
            "mean": float(100.0 * density_per_t.mean()),
        },
    }


def export_results(
    out_dir: str | Path,
    result: dict[str, Any],
    summary: dict[str, Any],
    *,
    save_tensor: bool = True,
    save_positions: bool = False,
    force: bool = False,
) -> dict[str, Path]:
    """Persist the result to ``out_dir``.

    Writes:
        * ``adjacency_accumulated.npy`` — (K, K) int64
        * ``adjacency_tensor.npy``      — (T, K, K) uint8  (skip with save_tensor=False)
        * ``node_positions.npy``        — (T, K, 2) float  (only if save_positions)
        * ``summary.json``              — JSON summary with layout
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    paths = {
        "accumulated": out / "adjacency_accumulated.npy",
        "tensor":      out / "adjacency_tensor.npy",
        "positions":   out / "node_positions.npy",
        "summary":     out / "summary.json",
    }
    if not force:
        for key, p in paths.items():
            if key == "tensor" and not save_tensor:
                continue
            if key == "positions" and not save_positions:
                continue
            if p.exists():
                raise FileExistsError(f"Refusing to overwrite {p} (pass force=True).")

    np.save(paths["accumulated"], result["accumulated"])
    written["accumulated"] = paths["accumulated"]

    if save_tensor:
        np.save(paths["tensor"], result["tensor"])
        written["tensor"] = paths["tensor"]

    if save_positions:
        np.save(paths["positions"], result["positions"])
        written["positions"] = paths["positions"]

    with paths["summary"].open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    written["summary"] = paths["summary"]

    return written


def print_summary(summary: dict[str, Any]) -> None:
    """Pretty-print a summary dict to stdout."""
    s = summary
    acc = s["accumulated_stats"]
    den = s["per_timestep_density_pct"]
    bs = s["block_stats"]
    print(f"Instance:           {s['instance_source']}")
    print(f"Radius of reach:    {s['radius_of_reach']}")
    print(f"Timesteps (T):      {s['n_timesteps']}")
    print(f"Nodes (K=1+N+M):    {s['n_nodes']}  (N={s['n_candidates']}, M={s['n_mobiles']})")
    print(f"Raw step counts:    {s['raw_step_counts']}  (T = max)")
    print(f"Tensor shape:       {tuple(s['tensor_shape'])}  (T, K, K), uint8 symmetric, zero diag")
    print(f"Accumulated shape:  {tuple(s['accumulated_shape'])}  (K, K), int64")
    print("Accumulated A_total stats:")
    print(f"  min = {acc['min']}, max = {acc['max']}, mean = {acc['mean']:.3f}, "
          f"sum = {acc['sum']}, density = {acc['density_pct']:.2f}%")
    print("Per-block A_total density (% of pairs ever in range):")
    for name, label in [
        ("sink_x_candidates", "sink <-> cand "),
        ("sink_x_mobiles",    "sink <-> mob  "),
        ("cand_x_cand",       "cand <-> cand "),
        ("cand_x_mobiles",    "cand <-> mob  "),
        ("mobile_x_mobile",   "mob  <-> mob  "),
    ]:
        b = bs[name]
        print(f"  {label}  min={b['min']:>4}  max={b['max']:>4}  "
              f"mean={b['mean']:>7.2f}  density={b['density_pct']:>6.2f}%")
    print("Per-timestep density (fraction of pairs in range):")
    print(f"  min = {den['min']:.2f}%, max = {den['max']:.2f}%, "
          f"mean = {den['mean']:.2f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build temporal full-network (1+N+M) adjacency matrices."
    )
    parser.add_argument("instance", type=str, help="Path to the instance JSON file.")
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory. If omitted, results are only printed.",
    )
    parser.add_argument(
        "--radius", type=float, default=None,
        help="Override radius_of_reach from the instance.",
    )
    parser.add_argument(
        "--ref-samples", type=int, default=_ARC_REF_SAMPLES,
        help="Dense reference grid size for arc length estimation.",
    )
    parser.add_argument(
        "--no-tensor", action="store_true",
        help="Skip writing the full (T, K, K) tensor (only the accumulated matrix).",
    )
    parser.add_argument(
        "--save-positions", action="store_true",
        help="Also write the per-timestep node positions tensor.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    instance = load_instance(args.instance)
    result = build_from_instance(
        instance, radius=args.radius, ref_samples=args.ref_samples
    )
    summary = summarise(result, instance)
    print_summary(summary)
    if args.out_dir:
        written = export_results(
            args.out_dir, result, summary,
            save_tensor=not args.no_tensor,
            save_positions=args.save_positions,
            force=args.force,
        )
        print("\nWritten files:")
        for key, p in written.items():
            print(f"  {key:12s} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
