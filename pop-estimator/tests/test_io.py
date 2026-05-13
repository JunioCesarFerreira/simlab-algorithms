from __future__ import annotations

import json

import pytest

from p2_population_estimator.io import (
    InstanceValidationError,
    load_instance,
    parse_instance,
)


def test_parse_minimal_instance(small_problem_dict):
    inst = parse_instance(small_problem_dict)
    assert inst.problem.name == "tiny"
    assert len(inst.problem.candidates) == 8
    assert inst.problem.sink.x == 0
    assert inst.problem.region.xmin == -50
    assert len(inst.problem.mobile_nodes) == 1


def test_parse_accepts_parameters_wrapper(small_problem_dict):
    wrapped = {"parameters": small_problem_dict}
    inst = parse_instance(wrapped)
    assert inst.problem.name == "tiny"


def test_parse_missing_problem():
    with pytest.raises(InstanceValidationError):
        parse_instance({"foo": "bar"})


def test_parse_missing_sink(small_problem_dict):
    del small_problem_dict["problem"]["sink"]
    with pytest.raises(InstanceValidationError, match="sink"):
        parse_instance(small_problem_dict)


def test_parse_radius_must_be_positive(small_problem_dict):
    small_problem_dict["problem"]["radius_of_reach"] = -1
    with pytest.raises(InstanceValidationError, match="radius_of_reach"):
        parse_instance(small_problem_dict)


def test_parse_empty_candidates(small_problem_dict):
    small_problem_dict["problem"]["candidates"] = []
    with pytest.raises(InstanceValidationError, match="candidates"):
        parse_instance(small_problem_dict)


def test_parse_region_must_be_four_floats(small_problem_dict):
    small_problem_dict["problem"]["region"] = [0, 0, 10]
    with pytest.raises(InstanceValidationError, match="region"):
        parse_instance(small_problem_dict)


def test_load_instance_roundtrip(tmp_instance_file):
    inst = load_instance(tmp_instance_file)
    assert inst.source_path is not None
    assert inst.source_path.endswith("tiny.json")


def test_parse_mobile_requires_path_segments(small_problem_dict):
    small_problem_dict["problem"]["mobile_nodes"] = [{"name": "x"}]
    with pytest.raises(InstanceValidationError, match="path_segments"):
        parse_instance(small_problem_dict)
