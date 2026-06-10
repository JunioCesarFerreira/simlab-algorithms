"""Shortest-path usage counters for WSN instances.

Same canonical ``(1 + N + M)`` node layout as :mod:`adjacency_builder`
(see :class:`adjacency_builder.NodeLayout`). The difference is what we
compute per timestep.

Per-timestep semantics
----------------------
For each timestep ``t``:

    1. build the radius-of-reach adjacency ``A(t)`` (as in
       :mod:`adjacency_builder`);
    2. run a single BFS from the **sink** in ``A(t)`` to obtain a
       shortest-hop tree;
    3. for each mobile ``m``, trace the BFS shortest path from ``m``
       back to the sink (if any);
    4. **increment by 1 every node on that path**, and (for the matrix
       view) every edge used by that path.

This yields, per timestep:

    node_count(t) : (K,) int
        node_count(t)[u] = #{ mobiles whose shortest path uses node u }
    edge_count(t) : (K, K) uint16
        edge_count(t)[u, v] = #{ mobiles whose shortest path uses edge (u, v) }
        Symmetric, zero diagonal.

Summing over time gives the accumulated quantities:

    node_accumulated : (K,) int
        node_accumulated[u] = total mobile-timesteps node u was on a
                              shortest mobile-to-sink path.
        Upper bound per entry: T * M.
    edge_accumulated : (K, K) int
        edge_accumulated[u, v] = total mobile-timesteps edge (u, v) was
                                 used by a shortest mobile-to-sink path.
        Upper bound per entry: T * M.

Implementation notes
--------------------
* BFS picks **one** shortest path per mobile (the canonical parent-tree
  path). When several shortest paths of equal length exist, only one is
  counted; the chosen one depends on the BFS visitation order. This
  matches typical routing semantics ("pick a shortest path"), not the
  enumeration of all shortest paths.
* The sink is on every successful path, so its node-score equals the
  total number of (mobile, timestep) pairs that had a path to the sink.
* Each mobile is on its own path, so its node-score equals the number of
  timesteps in which it was connected to the sink.
* For relay candidates, the node-score is the interesting quantity: how
  often the candidate was actually used as an intermediate hop.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

import adjacency_builder as ab


__all__ = [
    "shortest_paths_at_time",
    "build_path_tensors",
    "build_from_instance",
    "summarise",
    "export_results",
    "print_summary",
    "main",
]


# ---------------------------------------------------------------------------
# Per-timestep shortest-path counting
# ---------------------------------------------------------------------------
def shortest_paths_at_time(
    adj_t: np.ndarray,
    sink_index: int,
    mobile_indices: list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """For one timestep, count node/edge usage on shortest mobile-to-sink paths.

    A single BFS is run from the sink; each mobile's shortest path is
    obtained by tracing parent pointers backwards.

    Parameters
    ----------
    adj_t : ndarray of shape (K, K)
        Symmetric binary adjacency at this timestep, as produced by
        :func:`adjacency_builder.adjacency_at_time`.
    sink_index : int
        Index of the sink in the canonical node ordering.
    mobile_indices : sequence of int
        Indices of mobile sensors.

    Returns
    -------
    node_count : ndarray of shape (K,), dtype=int64
        ``node_count[u]`` = number of mobiles whose shortest path passes
        through node ``u``. Each successful path contributes +1 to every
        node it visits (including the mobile endpoint and the sink).
    edge_count : ndarray of shape (K, K), dtype=uint16
        ``edge_count[u, v]`` = number of mobiles whose shortest path uses
        edge ``(u, v)``. Symmetric, zero diagonal.
    n_active : int
        Number of mobiles that had a path to the sink at this timestep.
    """
    if adj_t.ndim != 2 or adj_t.shape[0] != adj_t.shape[1]:
        raise ValueError(f"adj_t must be (K, K), got {adj_t.shape}")
    K = adj_t.shape[0]

    parent = np.full(K, -1, dtype=np.int32)
    dist   = np.full(K, -1, dtype=np.int32)
    dist[sink_index] = 0
    queue: deque[int] = deque([int(sink_index)])
    while queue:
        u = queue.popleft()
        for v in np.flatnonzero(adj_t[u]):
            iv = int(v)
            if dist[iv] == -1:
                dist[iv] = dist[u] + 1
                parent[iv] = u
                queue.append(iv)

    node_count = np.zeros(K, dtype=np.int64)
    edge_count = np.zeros((K, K), dtype=np.uint16)
    n_active = 0
    for m in mobile_indices:
        im = int(m)
        if dist[im] == -1:
            continue
        n_active += 1
        cur = im
        while cur != -1:
            node_count[cur] += 1
            p = int(parent[cur])
            if p != -1:
                edge_count[cur, p] += 1
                edge_count[p, cur] += 1
            cur = p
    return node_count, edge_count, n_active


def build_path_tensors(
    positions: np.ndarray,
    radius: float,
    layout: ab.NodeLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-timestep shortest-path counters.

    Returns
    -------
    node_per_t : ndarray of shape (T, K), dtype=int64
        Per-timestep per-node hit counts.
    edge_per_t : ndarray of shape (T, K, K), dtype=uint16
        Per-timestep per-edge hit counts. Symmetric, zero diagonal.
    n_active_per_t : ndarray of shape (T,), dtype=int64
        Number of mobiles connected to the sink at each timestep.
    """
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError(f"positions must be (T, K, 2), got {positions.shape}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    T = positions.shape[0]
    K = positions.shape[1]
    node_per_t = np.zeros((T, K), dtype=np.int64)
    edge_per_t = np.zeros((T, K, K), dtype=np.uint16)
    n_active_per_t = np.zeros(T, dtype=np.int64)
    for t in range(T):
        adj_t = ab.adjacency_at_time(positions[t], radius)
        nc, ec, na = shortest_paths_at_time(
            adj_t, layout.sink_index, layout.mobile_indices
        )
        node_per_t[t] = nc
        edge_per_t[t] = ec
        n_active_per_t[t] = na
    return node_per_t, edge_per_t, n_active_per_t


# ---------------------------------------------------------------------------
# Pipeline / I/O
# ---------------------------------------------------------------------------
def build_from_instance(
    instance: ab.ProblemInstance,
    *,
    radius: float | None = None,
    ref_samples: int = ab._ARC_REF_SAMPLES,
) -> dict[str, Any]:
    """Run the full pipeline on a parsed instance.

    Returns a dict with keys:
        ``positions``         — (T, K, 2)   per-timestep node positions
        ``node_per_t``        — (T, K)      per-timestep node hit counts
        ``edge_per_t``        — (T, K, K)   per-timestep edge hit counts
        ``node_accumulated``  — (K,)        sum_t node_per_t  [PRIMARY]
        ``edge_accumulated``  — (K, K)      sum_t edge_per_t
        ``n_active_per_t``    — (T,)        # mobiles with a path per t
        ``layout``            — :class:`adjacency_builder.NodeLayout`
        ``raw_lengths``       — list of per-mobile raw step counts
        ``radius``            — communication radius used
    """
    if instance.n_candidates == 0:
        raise ab.InstanceError("Instance has no candidates; nothing to evaluate.")
    R = float(instance.radius_of_reach if radius is None else radius)
    positions, layout, raw_lengths = ab.build_node_positions(
        instance, ref_samples=ref_samples
    )
    node_per_t, edge_per_t, n_active_per_t = build_path_tensors(positions, R, layout)
    return {
        "positions": positions,
        "node_per_t": node_per_t,
        "edge_per_t": edge_per_t,
        "node_accumulated": node_per_t.sum(axis=0, dtype=np.int64),
        "edge_accumulated": edge_per_t.sum(axis=0, dtype=np.int64),
        "n_active_per_t": n_active_per_t,
        "layout": layout,
        "raw_lengths": raw_lengths,
        "radius": R,
    }


def _by_role(
    node_acc: np.ndarray, layout: ab.NodeLayout
) -> dict[str, dict[str, float]]:
    def stats(idx: list[int]) -> dict[str, float]:
        if not idx:
            return {"min": 0, "max": 0, "mean": 0.0, "sum": 0, "n_nonzero": 0}
        sub = node_acc[idx]
        return {
            "min":  int(sub.min()),
            "max":  int(sub.max()),
            "mean": float(sub.mean()),
            "sum":  int(sub.sum()),
            "n_nonzero": int((sub > 0).sum()),
        }
    return {
        "sink":       stats([layout.sink_index]),
        "candidates": stats(layout.candidate_indices),
        "mobiles":    stats(layout.mobile_indices),
    }


def summarise(result: dict[str, Any], instance: ab.ProblemInstance) -> dict[str, Any]:
    """Return a JSON-serialisable summary of a pipeline result."""
    node_acc: np.ndarray = result["node_accumulated"]
    edge_acc: np.ndarray = result["edge_accumulated"]
    n_active: np.ndarray = result["n_active_per_t"]
    layout: ab.NodeLayout = result["layout"]
    T = int(result["node_per_t"].shape[0])
    K = int(layout.K)

    return {
        "instance_source": instance.source_path,
        "radius_of_reach": float(result["radius"]),
        "n_timesteps": T,
        "n_nodes": K,
        "n_candidates": int(layout.N),
        "n_mobiles": int(layout.M),
        "layout": layout.to_dict(),
        "raw_step_counts": list(result["raw_lengths"]),
        "shapes": {
            "node_per_t": [T, K],
            "edge_per_t": [T, K, K],
            "node_accumulated": [K],
            "edge_accumulated": [K, K],
        },
        "upper_bound_per_node": int(T * layout.M),
        "upper_bound_per_edge": int(T * layout.M),
        "node_accumulated_stats": {
            "min":  int(node_acc.min()),
            "max":  int(node_acc.max()),
            "mean": float(node_acc.mean()),
            "sum":  int(node_acc.sum()),
        },
        "edge_accumulated_stats": {
            "min":  int(edge_acc.min()),
            "max":  int(edge_acc.max()),
            "mean": float(edge_acc.mean()),
            "sum":  int(edge_acc.sum()),
            "density_pct": float(100.0 * (edge_acc > 0).mean()),
        },
        "node_score_by_role": _by_role(node_acc, layout),
        "n_active_per_t_stats": {
            "min":  int(n_active.min()),
            "max":  int(n_active.max()),
            "mean": float(n_active.mean()),
            "full_connectivity_pct": float(100.0 * (n_active == layout.M).mean()),
            "no_connectivity_pct":   float(100.0 * (n_active == 0).mean()),
        },
    }


def export_results(
    out_dir: str | Path,
    result: dict[str, Any],
    summary: dict[str, Any],
    *,
    save_edge_tensor: bool = True,
    save_node_tensor: bool = True,
    save_positions: bool = False,
    force: bool = False,
) -> dict[str, Path]:
    """Persist the result to ``out_dir``.

    Writes:
        * ``path_node_accumulated.npy`` — (K,)     int64  [primary score]
        * ``path_edge_accumulated.npy`` — (K, K)   int64
        * ``path_node_per_t.npy``       — (T, K)   int64  (opt)
        * ``path_edge_per_t.npy``       — (T, K, K) uint16 (opt)
        * ``n_active_per_t.npy``        — (T,)     int64
        * ``node_positions.npy``        — (T, K, 2) float (opt)
        * ``summary.json``              — JSON summary
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    paths = {
        "node_acc":   out / "path_node_accumulated.npy",
        "edge_acc":   out / "path_edge_accumulated.npy",
        "node_t":     out / "path_node_per_t.npy",
        "edge_t":     out / "path_edge_per_t.npy",
        "n_active":   out / "n_active_per_t.npy",
        "positions":  out / "node_positions.npy",
        "summary":    out / "summary.json",
    }
    skipped = set()
    if not save_node_tensor:
        skipped.add("node_t")
    if not save_edge_tensor:
        skipped.add("edge_t")
    if not save_positions:
        skipped.add("positions")

    if not force:
        for key, p in paths.items():
            if key in skipped:
                continue
            if p.exists():
                raise FileExistsError(f"Refusing to overwrite {p} (pass force=True).")

    np.save(paths["node_acc"], result["node_accumulated"])
    written["node_acc"] = paths["node_acc"]
    np.save(paths["edge_acc"], result["edge_accumulated"])
    written["edge_acc"] = paths["edge_acc"]
    if save_node_tensor:
        np.save(paths["node_t"], result["node_per_t"])
        written["node_t"] = paths["node_t"]
    if save_edge_tensor:
        np.save(paths["edge_t"], result["edge_per_t"])
        written["edge_t"] = paths["edge_t"]
    np.save(paths["n_active"], result["n_active_per_t"])
    written["n_active"] = paths["n_active"]
    if save_positions:
        np.save(paths["positions"], result["positions"])
        written["positions"] = paths["positions"]
    with paths["summary"].open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    written["summary"] = paths["summary"]
    return written


def print_summary(summary: dict[str, Any]) -> None:
    """Pretty-print a summary dict to stdout."""
    s  = summary
    na = s["node_accumulated_stats"]
    ea = s["edge_accumulated_stats"]
    nb = s["node_score_by_role"]
    nc = s["n_active_per_t_stats"]
    print(f"Instance:           {s['instance_source']}")
    print(f"Radius of reach:    {s['radius_of_reach']}")
    print(f"Timesteps (T):      {s['n_timesteps']}")
    print(f"Nodes (K=1+N+M):    {s['n_nodes']}  (N={s['n_candidates']}, M={s['n_mobiles']})")
    print(f"Raw step counts:    {s['raw_step_counts']}  (T = max)")
    print(f"Upper bound:        T*M = {s['upper_bound_per_node']}  (per node and per edge)")
    print("Per-node accumulated stats:")
    print(f"  min={na['min']}, max={na['max']}, mean={na['mean']:.2f}, sum={na['sum']}")
    print("Per-edge accumulated stats:")
    print(f"  min={ea['min']}, max={ea['max']}, mean={ea['mean']:.3f}, "
          f"sum={ea['sum']}, density={ea['density_pct']:.2f}%")
    print("Node score by role:")
    for role, st in nb.items():
        print(f"  {role:<10s} min={st['min']:>4} max={st['max']:>4} "
              f"mean={st['mean']:>7.2f} nonzero={st['n_nonzero']}")
    print("Mobiles connected to sink per timestep:")
    print(f"  min={nc['min']}, max={nc['max']}, mean={nc['mean']:.2f}")
    print(f"  full connectivity = {nc['full_connectivity_pct']:.1f}% of t,  "
          f"no connectivity = {nc['no_connectivity_pct']:.1f}% of t")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build per-timestep shortest-path usage counters "
                    "(per-node and per-edge) over the (1+N+M) layout."
    )
    parser.add_argument("instance", type=str, help="Path to the instance JSON file.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory. If omitted, results are only printed.")
    parser.add_argument("--radius", type=float, default=None,
                        help="Override radius_of_reach from the instance.")
    parser.add_argument("--ref-samples", type=int, default=ab._ARC_REF_SAMPLES,
                        help="Dense reference grid size for arc length estimation.")
    parser.add_argument("--no-node-tensor", action="store_true",
                        help="Skip writing the (T, K) per-timestep node tensor.")
    parser.add_argument("--no-edge-tensor", action="store_true",
                        help="Skip writing the (T, K, K) per-timestep edge tensor.")
    parser.add_argument("--save-positions", action="store_true",
                        help="Also write the per-timestep node positions tensor.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    instance = ab.load_instance(args.instance)
    result = build_from_instance(
        instance, radius=args.radius, ref_samples=args.ref_samples
    )
    summary = summarise(result, instance)
    print_summary(summary)
    if args.out_dir:
        written = export_results(
            args.out_dir, result, summary,
            save_node_tensor=not args.no_node_tensor,
            save_edge_tensor=not args.no_edge_tensor,
            save_positions=args.save_positions,
            force=args.force,
        )
        print("\nWritten files:")
        for key, p in written.items():
            print(f"  {key:12s} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
