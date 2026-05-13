"""Evaluation back-ends (surrogate, Cooja) and SSH pool."""

from p2_population_estimator.evaluation.base import BaseEvaluator
from p2_population_estimator.evaluation.surrogate import SurrogateEvaluator

__all__ = ["BaseEvaluator", "SurrogateEvaluator"]
