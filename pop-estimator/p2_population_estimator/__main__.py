"""Allow ``python -m p2_population_estimator`` to invoke the CLI."""

from p2_population_estimator.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
