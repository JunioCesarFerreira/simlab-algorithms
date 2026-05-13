from __future__ import annotations

import math

from p2_population_estimator.geometry import (
    build_relay_graph,
    euclidean,
    mobile_reachable_via_relay,
    sample_all_mobiles,
    sample_trajectory,
    shortest_hops_to_sink,
)
from p2_population_estimator.models import MobileNode, Point


def test_euclidean():
    assert euclidean(Point(0, 0), Point(3, 4)) == 5.0


def test_sample_trajectory_linear():
    node = MobileNode(
        name="m",
        speed=1.0,
        time_step=1.0,
        is_closed=False,
        is_round_trip=False,
        path_segments=[("0 + 10 * t", "0")],
    )
    pts = sample_trajectory(node, num_samples=5)
    assert len(pts) == 5
    assert pts[0].x == 0.0
    assert pts[-1].x == 10.0


def test_sample_trajectory_handles_np_in_expression():
    node = MobileNode(
        name="m",
        speed=1.0,
        time_step=1.0,
        is_closed=False,
        is_round_trip=False,
        path_segments=[("np.cos(2*np.pi*t)", "np.sin(2*np.pi*t)")],
    )
    pts = sample_trajectory(node, num_samples=4)
    assert len(pts) == 4
    # First sample at t=0 should be (cos 0, sin 0) = (1, 0)
    assert math.isclose(pts[0].x, 1.0, abs_tol=1e-9)
    assert math.isclose(pts[0].y, 0.0, abs_tol=1e-9)


def test_relay_graph_and_hops(small_problem):
    selected = [0, 1, 2]  # (10,0),(0,10),(-10,0) all within radius 30 of sink
    adj = build_relay_graph(
        small_problem.candidates, small_problem.sink, small_problem.radius_of_reach, selected
    )
    # All three are within 30 of sink, so each connects to -1
    assert -1 in adj[0]
    assert -1 in adj[1]
    assert -1 in adj[2]
    hops = shortest_hops_to_sink(adj)
    assert hops[-1] == 0
    assert hops[0] == 1


def test_mobile_reachable(small_problem):
    selected = [0, 1]
    adj = build_relay_graph(
        small_problem.candidates, small_problem.sink, small_problem.radius_of_reach, selected
    )
    hops = shortest_hops_to_sink(adj)
    mobile_pos = Point(15, 0)  # close to candidate 0=(10,0)
    reach, h, d = mobile_reachable_via_relay(
        mobile_pos, small_problem.candidates, small_problem.sink,
        small_problem.radius_of_reach, hops
    )
    assert reach is True
    assert h >= 1
    assert d < small_problem.radius_of_reach
