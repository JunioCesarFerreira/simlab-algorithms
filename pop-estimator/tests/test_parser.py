from __future__ import annotations

from p2_population_estimator.evaluation.parser import (
    parse_cooja_log,
    parse_keyvalue_log,
    parse_summary_json,
)


def test_parse_summary_json_basic():
    text = """
    some log line
    __POPEST_RESULT__ {"latency": 12.5, "energy": 100, "throughput": 7.5,
                       "connected_ratio": 0.9, "relay_count": 8}
    more log
    """
    m = parse_summary_json(text)
    assert m is not None
    assert m.latency == 12.5
    assert m.energy == 100.0
    assert m.throughput == 7.5
    assert m.connected_ratio == 0.9
    assert m.relay_count == 8


def test_parse_keyvalue_log():
    text = """
    [boot] latency=12.5 energy=100
    [end] throughput=7.5 relay_count=8 connected_ratio=0.9
    """
    m = parse_keyvalue_log(text)
    assert m.latency == 12.5
    assert m.energy == 100.0
    assert m.throughput == 7.5
    assert m.connected_ratio == 0.9
    assert m.relay_count == 8


def test_parse_cooja_prefers_summary():
    text = "latency=999\n__POPEST_RESULT__ {\"latency\": 1.0}\n"
    m = parse_cooja_log(text)
    assert m.latency == 1.0


def test_parse_cooja_falls_back_to_kv():
    text = "latency=42\n"
    m = parse_cooja_log(text)
    assert m.latency == 42.0
