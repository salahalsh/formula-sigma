"""e02 case 3 - Boscolo 2023 BBD nanosuspension + lyophilisation.

Reproduces the published linear model: PS = beta0 + beta1*X1 + beta3*X3
(X2 amplitude not significant per the paper).
"""
from __future__ import annotations
import json, time, sys, csv as _csv
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import seeds, paths, manifest, data_io


def code(x, low, high):
    return (x - (low + high) / 2.0) / ((high - low) / 2.0)


def main():
    t0 = time.time()
    seed = seeds.set_seeds(42)
    OUT_DIR = paths.RESULTS / "e02" / "case3_boscolo"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    case = data_io.load_case("case3_lyophilization_bbd")
    d = case.design.copy()
    r = case.responses.copy()

    X1 = code(d["stabilizer_pct"].astype(float).to_numpy(), 0.3, 1.0)
    X2 = code(d["amplitude_W"].astype(float).to_numpy(), 30, 100)
    X3 = code(d["sonication_time_min"].astype(float).to_numpy(), 3, 10)
    y = r["particle_size_nm"].astype(float).to_numpy()

    # Full quadratic for fairness, then evaluate sig terms
    poly_terms = {
        "X1": X1, "X2": X2, "X3": X3,
        "X1^2": X1 ** 2, "X2^2": X2 ** 2, "X3^2": X3 ** 2,
        "X1*X2": X1 * X2, "X1*X3": X1 * X3, "X2*X3": X2 * X3,
    }
    Xfull = np.column_stack(list(poly_terms.values()))
    fit_full = sm.OLS(y, sm.add_constant(Xfull)).fit()

    # Linear-only (as published)
    Xlin = np.column_stack([X1, X2, X3])
    fit_lin = sm.OLS(y, sm.add_constant(Xlin)).fit()

    # Reduced: X1 + X3 only (as published with X2 dropped)
    Xred = np.column_stack([X1, X3])
    fit_red = sm.OLS(y, sm.add_constant(Xred)).fit()

    # Centre-point variance for PDS noise model
    center = (X1 == 0) & (X2 == 0) & (X3 == 0)
    y_center = y[center]
    s_center = float(y_center.std(ddof=1)) if len(y_center) > 1 else float("nan")

    rows = [["model", "n_terms", "R2", "RMSE", "X1_p", "X2_p", "X3_p"]]
    rows.append([
        "full_quadratic", fit_full.df_model, f"{fit_full.rsquared:.4f}",
        f"{float(np.sqrt(np.mean(fit_full.resid ** 2))):.3f}",
        f"{fit_full.pvalues[1]:.4f}",
        f"{fit_full.pvalues[2]:.4f}",
        f"{fit_full.pvalues[3]:.4f}",
    ])
    rows.append([
        "linear_X1_X2_X3", 3, f"{fit_lin.rsquared:.4f}",
        f"{float(np.sqrt(np.mean(fit_lin.resid ** 2))):.3f}",
        f"{fit_lin.pvalues[1]:.4f}", f"{fit_lin.pvalues[2]:.4f}",
        f"{fit_lin.pvalues[3]:.4f}",
    ])
    rows.append([
        "linear_X1_X3_only (published)", 2, f"{fit_red.rsquared:.4f}",
        f"{float(np.sqrt(np.mean(fit_red.resid ** 2))):.3f}",
        f"{fit_red.pvalues[1]:.4f}", "dropped", f"{fit_red.pvalues[2]:.4f}",
    ])

    detail = {
        "n_runs": len(y),
        "centre_point_runs": int(center.sum()),
        "centre_point_sigma_PS": s_center,
        "linear_model_coefficients": dict(zip(["const", "X1", "X2", "X3"],
                                              fit_lin.params.tolist())),
        "reduced_model_coefficients": dict(zip(["const", "X1", "X3"],
                                               fit_red.params.tolist())),
        "matches_published_finding_X2_not_significant":
            bool(fit_lin.pvalues[2] > 0.05),
        "matches_published_finding_X1_significant":
            bool(fit_lin.pvalues[1] < 0.05),
        "matches_published_finding_X3_significant":
            bool(fit_lin.pvalues[3] < 0.05),
    }

    csv_path = OUT_DIR / "case3_summary.csv"
    json_path = OUT_DIR / "case3_detail.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        _csv.writer(fh).writerows(rows)
    json_path.write_text(json.dumps(detail, indent=2))

    manifest.write_manifest(
        experiment="e02_case3_boscolo", out_dir=OUT_DIR, seed=seed,
        inputs=[paths.DATA_PROC / "case3_lyophilization_bbd" / f for f in
                ["case3_design.csv", "case3_responses.csv"]],
        outputs=[csv_path, json_path],
        runtime_sec=time.time() - t0)

    print("Boscolo 2023 reproduction:")
    for row in rows:
        print("  ", "  ".join(str(c) for c in row))
    print(f"  centre-point sigma (PS): {s_center:.2f} nm  (n={center.sum()})")
    print(f"  matches published 'X2 not sig': "
          f"{detail['matches_published_finding_X2_not_significant']}")
    print(f"  matches published 'X1 sig':      "
          f"{detail['matches_published_finding_X1_significant']}")
    print(f"  matches published 'X3 sig':      "
          f"{detail['matches_published_finding_X3_significant']}")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
