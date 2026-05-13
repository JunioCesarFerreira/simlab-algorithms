"""End-to-end test for ``run_experiment`` in surrogate mode."""

from __future__ import annotations

import json
from pathlib import Path

from p2_population_estimator.experiment import run_experiment
from p2_population_estimator.models import ExperimentConfig, ScalarizationWeights


def test_run_experiment_surrogate_end_to_end(tmp_instance_file, tmp_path: Path):
    out_dir = tmp_path / "run01"
    cfg = ExperimentConfig(
        instance_path=str(tmp_instance_file),
        output_dir=str(out_dir),
        mode="surrogate",
        partition_method="grid",
        num_blocks=4,
        num_complements=5,
        alpha=0.05,
        rho=0.3,
        seeds=[1, 2, 3],
        random_seed=42,
        weights=ScalarizationWeights(),
    )
    result = run_experiment(cfg)

    # Files produced
    assert (out_dir / "population_estimate_result.json").exists()
    assert (out_dir / "block_results.csv").exists()
    # Result structure
    data = json.loads((out_dir / "population_estimate_result.json").read_text(encoding="utf-8"))
    assert "global_estimate" in data
    assert "block_results" in data
    assert data["experiment_config"]["mode"] == "surrogate"
    assert isinstance(result.block_results, list)
    assert result.global_estimate["num_valid_blocks"] + result.global_estimate["num_invalid_blocks"] == len(result.block_results)
