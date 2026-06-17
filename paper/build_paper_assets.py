"""Regenerate the paper's figures and tables from ``results/``.

Copies the 14 result figures into ``paper/figures/`` (clean names referenced by
``main.tex``) and rebuilds ``paper/tables/summary_table.tex`` from the combined
estimator's ``summary_table.csv``.  Run after the method scripts and before
compiling ``main.tex``.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent
RESULTS = REPO_ROOT / "results"
FIG_DIR = PAPER_DIR / "figures"
TAB_DIR = PAPER_DIR / "tables"

FIGURE_MAP = {
    "method1/method1_saturation.png":      "m1_saturation.png",
    "method1/method1_marginal_gain.png":   "m1_marginal.png",
    "method1/method1_information_gain.png": "m1_infogain.png",
    "method2/method2_Atotal_heatmap.png":  "m2_heatmap.png",
    "method2/method2_coverage_bar.png":    "m2_coverage.png",
    "method2/method2_convergence.png":     "m2_convergence.png",
    "method2/method2_spatial.png":         "m2_spatial.png",
    "method3/method3_edge_heatmap.png":    "m3_edge_heatmap.png",
    "method3/method3_route_bar.png":       "m3_route.png",
    "method3/method3_convergence.png":     "m3_convergence.png",
    "method3/method3_spatial.png":         "m3_spatial.png",
    "method4/method4_gap_vs_pop.png":      "m4_gap.png",
    "method4/method4_milp_landscape.png":  "m4_landscape.png",
    "combined/combined_comparison.png":    "combined.png",
}


def _esc(s: str) -> str:
    return s.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def build_table() -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((RESULTS / "combined" / "summary_table.csv").open()))
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Estimator & $\hat N$ & raw & CI low & CI high & $\sigma$ \\", r"\midrule"]
    for r in rows:
        if r["estimator"].startswith("COMBINED"):
            lines.append(r"\midrule")
        cells = [_esc(r["estimator"]), r["n_hat"], r["n_hat_raw"],
                 r["ci_low"], r["ci_high"], r["sigma"]]
        cells = [c if c not in ("", "None") else "--" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB_DIR / "summary_table.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {TAB_DIR/'summary_table.tex'}")


def copy_figures() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for src, dst in FIGURE_MAP.items():
        s = RESULTS / src
        if s.exists():
            shutil.copy2(s, FIG_DIR / dst)
            n += 1
        else:
            print(f"  WARNING: missing {s}")
    print(f"  copied {n}/{len(FIGURE_MAP)} figures into {FIG_DIR}")


def main() -> int:
    print("Building paper assets...")
    copy_figures()
    build_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
