"""Parse Cooja log output into :class:`SimulationMetrics`.

Matches the firmware output format from ``root.c``:
  - Lines captured by the ScriptRunner script look like:
      [Mote:1] {"node":"ip", "rtt_latency":X, "total_energy_mj":X, "server_received":X, ...}
  - One JSON object per packet received by the root (mote 1).
  - Aggregation mirrors ``wsn-design-space-exploration/batch_runner/lib/csv_converter.py``.

Derived metrics:
  latency    <- mean of ``rtt_latency`` across all records
  energy     <- sum of last ``total_energy_mj`` per node (total energy at end of sim)
  throughput <- sum of (last - first) ``server_received`` per node (packets delivered)
  relay_count <- number of unique nodes that responded
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Iterable

from p2_population_estimator.models import SimulationMetrics

# Matches [Mote:1] lines that contain a JSON object (flat, no nested braces).
_MOTE1_RE = re.compile(r'\[Mote:1\].*?(\{[^{}]+\})')


def _parse_records(text: str) -> list[dict]:
    records = []
    for line in text.splitlines():
        m = _MOTE1_RE.search(line)
        if not m:
            continue
        try:
            rec = json.loads(m.group(1))
            if "node" in rec:
                records.append(rec)
        except json.JSONDecodeError:
            continue
    return records


def parse_cooja_log(text: str) -> SimulationMetrics:
    """Parse COOJA.testlog content and return aggregated SimulationMetrics."""
    records = _parse_records(text)

    if not records:
        return SimulationMetrics()

    # Group by node identifier
    by_node: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_node[str(r["node"])].append(r)

    # latency: mean rtt_latency across all records
    latency_vals = [float(r["rtt_latency"]) for r in records if "rtt_latency" in r]
    latency = sum(latency_vals) / len(latency_vals) if latency_vals else None

    # energy: sum of the LAST total_energy_mj reading per node
    energy = 0.0
    for node_records in by_node.values():
        sorted_recs = sorted(node_records, key=lambda r: r.get("root_time_now", 0))
        energy += float(sorted_recs[-1].get("total_energy_mj", 0))
    energy_val: float | None = energy if energy > 0 else None

    # throughput: sum of (last - first) server_received per node
    throughput = 0.0
    for node_records in by_node.values():
        sorted_recs = sorted(node_records, key=lambda r: r.get("root_time_now", 0))
        first = float(sorted_recs[0].get("server_received", 0))
        last = float(sorted_recs[-1].get("server_received", 0))
        throughput += max(0.0, last - first)

    # relay_count: number of unique nodes that responded
    relay_count = len(by_node)

    return SimulationMetrics(
        latency=latency,
        energy=energy_val,
        throughput=throughput,
        relay_count=relay_count,
    )


def parse_log_files(paths: Iterable[str]) -> SimulationMetrics:
    chunks: list[str] = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            chunks.append(fh.read())
    return parse_cooja_log("\n".join(chunks))
