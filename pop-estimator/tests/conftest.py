"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from p2_population_estimator.io import parse_instance
from p2_population_estimator.models import (
    MobileNode,
    P2Problem,
    Point,
    Region,
)


def _small_problem_dict() -> dict[str, Any]:
    return {
        "problem": {
            "name": "tiny",
            "radius_of_reach": 30.0,
            "radius_of_inter": 60.0,
            "region": [-50, -50, 50, 50],
            "sink": [0, 0],
            "candidates": [
                [10, 0], [0, 10], [-10, 0], [0, -10],
                [20, 20], [-20, 20], [-20, -20], [20, -20],
            ],
            "mobile_nodes": [
                {
                    "name": "m1",
                    "speed": 1.0,
                    "time_step": 1.0,
                    "is_closed": False,
                    "is_round_trip": False,
                    "path_segments": [
                        ["-40 + 80 * t", "10"],
                    ],
                }
            ],
        }
    }


@pytest.fixture
def small_problem_dict() -> dict[str, Any]:
    return _small_problem_dict()


@pytest.fixture
def small_instance():
    return parse_instance(_small_problem_dict())


@pytest.fixture
def small_problem(small_instance) -> P2Problem:
    return small_instance.problem


@pytest.fixture
def tmp_instance_file(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.json"
    p.write_text(json.dumps(_small_problem_dict()), encoding="utf-8")
    return p
