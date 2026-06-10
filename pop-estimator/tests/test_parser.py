from __future__ import annotations

import json

from p2_population_estimator.evaluation.parser import parse_cooja_log


def _mote1_line(**fields) -> str:
    return f"[Mote:1] {json.dumps(fields)}"


def test_parse_empty_log_returns_blank_metrics():
    m = parse_cooja_log("Random seed: 1\nInitializing simulation script\n")
    assert m.latency is None
    assert m.throughput is None
    assert m.relay_count is None


def test_parse_single_node_two_records():
    log = "\n".join([
        "Random seed: 35239",
        _mote1_line(node="fd00::206:6:6:6", total_energy_mj=18,
                    server_received=3, rtt_latency=95, root_time_now=30000),
        _mote1_line(node="fd00::206:6:6:6", total_energy_mj=36,
                    server_received=6, rtt_latency=91, root_time_now=60000),
        "Final simulation time: 180000000 ms",
    ])
    m = parse_cooja_log(log)
    assert m.relay_count == 1
    assert m.latency == 93.0          # mean(95, 91)
    assert m.energy == 36.0           # last total_energy_mj
    assert m.throughput == 3.0        # 6 - 3


def test_parse_multiple_nodes_aggregates_per_node():
    log = "\n".join([
        _mote1_line(node="A", total_energy_mj=10, server_received=2,
                    rtt_latency=100, root_time_now=10000),
        _mote1_line(node="A", total_energy_mj=20, server_received=5,
                    rtt_latency=80, root_time_now=20000),
        _mote1_line(node="B", total_energy_mj=15, server_received=1,
                    rtt_latency=60, root_time_now=15000),
        _mote1_line(node="B", total_energy_mj=30, server_received=4,
                    rtt_latency=40, root_time_now=25000),
    ])
    m = parse_cooja_log(log)
    assert m.relay_count == 2
    assert m.latency == 70.0          # mean(100, 80, 60, 40)
    assert m.energy == 50.0           # 20 (A last) + 30 (B last)
    assert m.throughput == 6.0        # (5-2) + (4-1)


def test_non_mote1_json_is_ignored():
    log = "\n".join([
        '[Mote:2] {"node":"X", "server_received":99, "root_time_now":1}',
        _mote1_line(node="Y", server_received=7, rtt_latency=50, root_time_now=1),
    ])
    m = parse_cooja_log(log)
    assert m.relay_count == 1         # only the [Mote:1] record counts
