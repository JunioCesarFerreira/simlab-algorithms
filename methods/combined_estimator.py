r"""Combined estimator — fuse N_hat_1 .. N_hat_4 into a single recommendation.

Each method views the population-sizing problem through a different lens
(search dynamics, geometric coverage, routing criticality, MILP-calibrated
quality), and each returns a point estimate ``N_hat_i`` with a 1-sigma
uncertainty ``sigma_i``.  Three fusion strategies are provided, from the most
conservative to the most principled.

1. Conservative envelope
------------------------
        N_env = max_i N_hat_i.                                              (1)

Guarantees every individual criterion is met (no view is under-served).  Used as
a safe upper recommendation.

2. Inverse-variance weighted average
------------------------------------
Weight each estimate by its precision ``w_i = 1 / sigma_i^2``:

        N_wavg = ( sum_i w_i * N_hat_i ) / ( sum_i w_i ),                   (2)
        Var(N_wavg) = 1 / sum_i w_i.                                        (3)

More reliable (lower-variance) methods dominate.

3. Bayesian fusion (Gaussian conjugate)
---------------------------------------
Treat each estimate as a noisy observation ``N_hat_i ~ Normal(N*, sigma_i^2)`` of
the true population ``N*``.  With a weakly-informative Gaussian prior
``N* ~ Normal(mu_0, sigma_0^2)`` and independent observations, the posterior is
Gaussian with

        precision_post = 1/sigma_0^2 + sum_i 1/sigma_i^2,                   (4)
        mu_post = ( mu_0/sigma_0^2 + sum_i N_hat_i/sigma_i^2 ) / precision_post,  (5)
        sigma_post^2 = 1 / precision_post.                                  (6)

The 95 % credible interval is ``mu_post +- 1.96 * sigma_post``.

Gaussian vs Poisson
-------------------
The prompt allows a Poisson conjugate model instead.  A Poisson model forces
``Var = mean``; we test it against the empirical dispersion of the four
estimates.  The four estimates are strongly **over-dispersed**
(``Var >> mean``), which violates the Poisson assumption, so the Gaussian model
is selected.  The dispersion ratio is reported.

Output
======
* ``summary_table.csv`` — every method + the three combined estimates.
* ``combined_comparison.png`` — all estimates with uncertainty bars (log scale,
  so the conservative gambler-ruin baseline is visible alongside the rest).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from methods.common import RESULTS_DIR, ensure_dir, get_plt, load_json, save_json

METHOD_DIR = RESULTS_DIR / "combined"
INSTANCE = "ind2"

METHOD_FILES = {
    "M1": RESULTS_DIR / "method1" / "method1_result.json",
    "M2": RESULTS_DIR / "method2" / "method2_result.json",
    "M3": RESULTS_DIR / "method3" / "method3_result.json",
    "M4": RESULTS_DIR / "method4" / "method4_result.json",
    "M5": RESULTS_DIR / "method5" / "method5_result.json",
}
METHOD_LABEL = {
    "M1": "M1 inference (diminishing returns)",
    "M2": "M2 adjacency coverage",
    "M3": "M3 routing criticality",
    "M4": "M4 MILP calibration",
    "M5": "M5 gambler-ruin (order-1 blocks)",
}


def _load_methods() -> list[dict]:
    out = []
    for mid, path in METHOD_FILES.items():
        d = load_json(path)
        out.append({
            "method": mid,
            "label": METHOD_LABEL[mid],
            "n_hat": float(d["n_hat"]),
            "n_hat_raw": float(d["n_hat_raw"]),
            "ci_low": float(d["ci_low"]),
            "ci_high": float(d["ci_high"]),
            "sigma": float(d["sigma"]),
        })
    return out


def combine() -> dict:
    methods = _load_methods()
    n = np.array([m["n_hat_raw"] for m in methods])
    sig = np.array([m["sigma"] for m in methods])
    w = 1.0 / sig**2

    # (1) conservative envelope
    env = float(np.max([m["n_hat"] for m in methods]))

    # (2) inverse-variance weighted average
    wavg = float(np.sum(w * n) / np.sum(w))
    wavg_var = float(1.0 / np.sum(w))
    wavg_sigma = math.sqrt(wavg_var)

    # (3) Bayesian Gaussian fusion with a weakly-informative prior
    mu0 = float(np.mean(n))
    sigma0 = float(max(np.std(n, ddof=1) * 3.0, np.max(sig) * 3.0))  # broad prior
    prec_post = 1.0 / sigma0**2 + np.sum(1.0 / sig**2)
    mu_post = (mu0 / sigma0**2 + np.sum(n / sig**2)) / prec_post
    sigma_post = math.sqrt(1.0 / prec_post)

    # Gaussian vs Poisson dispersion check
    emp_mean = float(np.mean(n))
    emp_var = float(np.var(n, ddof=1))
    dispersion = emp_var / emp_mean      # ~1 => Poisson plausible; >>1 => over-dispersed
    model_choice = "gaussian" if dispersion > 2.0 else "poisson"

    weights_norm = (w / w.sum()).tolist()
    return {
        "instance": INSTANCE,
        "methods": [
            {**m, "weight": weights_norm[i]} for i, m in enumerate(methods)
        ],
        "combined": {
            "conservative_envelope": {
                "n_hat": int(math.ceil(env)), "formula": "max_i N_hat_i",
            },
            "weighted_average": {
                "n_hat": int(math.ceil(wavg)), "n_hat_raw": wavg,
                "sigma": wavg_sigma,
                "ci_low": wavg - 1.96 * wavg_sigma, "ci_high": wavg + 1.96 * wavg_sigma,
                "formula": "sum(w_i N_i)/sum(w_i),  w_i=1/sigma_i^2",
            },
            "bayesian_fusion": {
                "n_hat": int(math.ceil(mu_post)), "mu_post": mu_post,
                "sigma_post": sigma_post,
                "ci_low": mu_post - 1.96 * sigma_post, "ci_high": mu_post + 1.96 * sigma_post,
                "prior_mu": mu0, "prior_sigma": sigma0,
                "model": model_choice,
                "formula": "Gaussian conjugate posterior (eqs. 4-6)",
            },
        },
        "model_selection": {
            "empirical_mean": emp_mean, "empirical_var": emp_var,
            "dispersion_var_over_mean": dispersion,
            "chosen": model_choice,
            "reason": ("over-dispersed (Var >> mean) -> Gaussian"
                       if model_choice == "gaussian"
                       else "dispersion ~ 1 -> Poisson plausible"),
        },
    }


def _write_table(summary: dict) -> Path:
    path = METHOD_DIR / "summary_table.csv"
    rows = []
    for m in summary["methods"]:
        rows.append({
            "estimator": m["label"], "n_hat": int(m["n_hat"]),
            "n_hat_raw": round(m["n_hat_raw"], 2),
            "ci_low": round(m["ci_low"], 1), "ci_high": round(m["ci_high"], 1),
            "sigma": round(m["sigma"], 2), "weight": round(m["weight"], 4),
        })
    c = summary["combined"]
    rows.append({"estimator": "COMBINED conservative envelope",
                 "n_hat": c["conservative_envelope"]["n_hat"], "n_hat_raw": "",
                 "ci_low": "", "ci_high": "", "sigma": "", "weight": ""})
    rows.append({"estimator": "COMBINED weighted average",
                 "n_hat": c["weighted_average"]["n_hat"],
                 "n_hat_raw": round(c["weighted_average"]["n_hat_raw"], 2),
                 "ci_low": round(c["weighted_average"]["ci_low"], 1),
                 "ci_high": round(c["weighted_average"]["ci_high"], 1),
                 "sigma": round(c["weighted_average"]["sigma"], 2), "weight": ""})
    rows.append({"estimator": "COMBINED Bayesian fusion",
                 "n_hat": c["bayesian_fusion"]["n_hat"],
                 "n_hat_raw": round(c["bayesian_fusion"]["mu_post"], 2),
                 "ci_low": round(c["bayesian_fusion"]["ci_low"], 1),
                 "ci_high": round(c["bayesian_fusion"]["ci_high"], 1),
                 "sigma": round(c["bayesian_fusion"]["sigma_post"], 2), "weight": ""})
    ensure_dir(METHOD_DIR)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    return path


def _plot(summary: dict) -> None:
    plt = get_plt()
    # gather baseline (gambler-ruin) for context
    m1 = load_json(METHOD_FILES["M1"])
    gr = m1["params"].get("gambler_ruin_baseline", {})
    gr_val = gr.get("n_hat_uniform") if gr.get("available") else None

    labels, vals, los, his, colors = [], [], [], [], []
    for m in summary["methods"]:
        labels.append(m["method"]); vals.append(m["n_hat_raw"])
        los.append(m["n_hat_raw"] - m["ci_low"]); his.append(m["ci_high"] - m["n_hat_raw"])
        colors.append("C0")
    c = summary["combined"]
    for key, lab, col in [("conservative_envelope", "envelope", "C3"),
                          ("weighted_average", "weighted", "C2"),
                          ("bayesian_fusion", "Bayesian", "C4")]:
        cc = c[key]
        labels.append(lab)
        vals.append(cc.get("n_hat_raw", cc.get("mu_post", cc["n_hat"])))
        lo = cc.get("ci_low", cc["n_hat"]); hi = cc.get("ci_high", cc["n_hat"])
        v = vals[-1]
        los.append(max(v - lo, 0)); his.append(max(hi - v, 0)); colors.append(col)

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.errorbar(vals, y, xerr=[los, his], fmt="o", capsize=4, color="none",
                ecolor="gray", zorder=1)
    ax.scatter(vals, y, c=colors, s=70, zorder=2, edgecolor="k", linewidth=0.4)
    if gr_val:
        ax.axvline(gr_val, ls=":", color="C1",
                   label=f"gambler-ruin baseline = {gr_val}")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel(r"estimated population size $\hat N$ (log scale)")
    ax.set_title("Population-size estimates: four methods + three fusions")
    ax.invert_yaxis(); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(METHOD_DIR / "combined_comparison.png"); plt.close(fig)


def main() -> int:
    ensure_dir(METHOD_DIR)
    summary = combine()
    save_json(METHOD_DIR / "combined_result.json", summary)
    table = _write_table(summary)
    _plot(summary)
    c = summary["combined"]
    print("=== Combined estimator ===")
    print("  individual:  " + "  ".join(
        f"{m['method']}={int(m['n_hat'])}(w={m['weight']:.2f})" for m in summary["methods"]))
    print(f"  conservative envelope : {c['conservative_envelope']['n_hat']}")
    print(f"  weighted average      : {c['weighted_average']['n_hat']}  "
          f"(95% CI [{c['weighted_average']['ci_low']:.0f}, {c['weighted_average']['ci_high']:.0f}])")
    print(f"  Bayesian fusion       : {c['bayesian_fusion']['n_hat']}  "
          f"(95% CrI [{c['bayesian_fusion']['ci_low']:.0f}, {c['bayesian_fusion']['ci_high']:.0f}], "
          f"model={c['bayesian_fusion']['model']})")
    ms = summary["model_selection"]
    print(f"  dispersion Var/mean = {ms['dispersion_var_over_mean']:.1f} -> {ms['chosen']}")
    print(f"  -> {table}")
    print(f"  -> {METHOD_DIR}/combined_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
