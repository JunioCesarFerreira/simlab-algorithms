"""Parse Cooja log output into :class:`SimulationMetrics`.

The exact log format depends on the Contiki-NG application running inside
Cooja. This module is intentionally small and well-tested so that adapting
it to a different output format is a localised change.

We support two parsing flavours out of the box:

1. ``parse_keyvalue_log`` — picks up lines like ``KEY=VALUE`` (case-insensitive
   key match), tolerating prefixes / log timestamps.
2. ``parse_summary_json``  — looks for a single JSON object preceded by the
   marker ``__POPEST_RESULT__`` (a convention you can emit from your
   firmware/simulation glue).

If both are present, ``parse_summary_json`` wins because it is the explicit
contract.
"""

from __future__ import annotations

import json
import re
from typing import Iterable, Optional

from p2_population_estimator.models import SimulationMetrics

_KV_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(?P<val>-?\d+(?:\.\d+)?)")
_MARKER = "__POPEST_RESULT__"


_KNOWN_FIELDS = {
    "latency",
    "energy",
    "throughput",
    "packet_delivery_ratio",
    "connected_ratio",
    "relay_count",
    "mean_hop_count",
    "mean_distance_to_mobile",
    "redundancy",
}


def parse_summary_json(text: str) -> Optional[SimulationMetrics]:
    """Look for ``__POPEST_RESULT__ {...json...}`` and parse it."""
    idx = text.find(_MARKER)
    if idx < 0:
        return None
    rest = text[idx + len(_MARKER):]
    # find first '{' and try to parse JSON greedily until balanced
    start = rest.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    for i, ch in enumerate(rest[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    try:
        payload = json.loads(rest[start:end])
    except json.JSONDecodeError:
        return None
    return _from_dict(payload)


def parse_keyvalue_log(text: str) -> SimulationMetrics:
    """Collect last numeric occurrence of each known key."""
    collected: dict[str, float] = {}
    for m in _KV_RE.finditer(text):
        key = m.group("key").lower()
        if key in _KNOWN_FIELDS:
            try:
                collected[key] = float(m.group("val"))
            except ValueError:
                continue
    return _from_dict(collected)


def parse_cooja_log(text: str) -> SimulationMetrics:
    """Convenience: try summary JSON first, then KV parsing."""
    js = parse_summary_json(text)
    if js is not None:
        return js
    return parse_keyvalue_log(text)


def parse_log_files(paths: Iterable[str]) -> SimulationMetrics:
    chunks: list[str] = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            chunks.append(fh.read())
    return parse_cooja_log("\n".join(chunks))


def _from_dict(d: dict[str, object]) -> SimulationMetrics:
    def _f(key: str) -> Optional[float]:
        v = d.get(key)
        if v is None:
            return None
        try:
            return float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    def _i(key: str) -> Optional[int]:
        v = d.get(key)
        if v is None:
            return None
        try:
            return int(float(v))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    extras = {
        k: float(v)  # type: ignore[arg-type]
        for k, v in d.items()
        if k not in _KNOWN_FIELDS and isinstance(v, (int, float))
    }
    return SimulationMetrics(
        latency=_f("latency"),
        energy=_f("energy"),
        throughput=_f("throughput"),
        packet_delivery_ratio=_f("packet_delivery_ratio"),
        connected_ratio=_f("connected_ratio"),
        relay_count=_i("relay_count"),
        mean_hop_count=_f("mean_hop_count"),
        mean_distance_to_mobile=_f("mean_distance_to_mobile"),
        redundancy=_f("redundancy"),
        extras=extras,
    )
