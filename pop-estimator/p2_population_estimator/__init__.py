"""P2 Population Estimator.

Heuristic-statistical procedure to estimate the population size required by
NSGA-III / genetic algorithms applied to Problem P2 (Discrete Coverage with
Mobility), based on a gambler-ruin approximation.

Public entry points: see ``p2_population_estimator.cli`` and
``p2_population_estimator.experiment``.
"""

from p2_population_estimator import models  # re-export for convenience

__all__ = ["models"]
__version__ = "0.1.0"
