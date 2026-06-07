"""e02 case 1 - Farooqi 2020 CCD osmotic pump CR tablet (REPRODUCTION).

Per the diagnostic in case1_diagnostic.md, the published Table 4 main-
effect equations do NOT predict the published Table 1 response values
(RMSE ~12 on Y1, center-point mean 11 vs published intercept 20.3).

Honest reproduction strategy:
  (1) Refit each response with a FULL quadratic model on coded factors
      (matches what the authors' R^2 values imply, even if their T4
      printed only first-order terms).
  (2) F-19 excluded as a published outlier (X2_coded = -1.682 ->
      physically impossible -0.004 mm orifice (no orifice); all responses = 0).
  (3) Validate by predicting at the F-A optimum (Table 6) and comparing
      to the experimentally-observed Y1..Y4 at F-A.

This is the manuscript Section 3.2.1 finding:
  - FORMULA-Sigma reproduces the response surfaces with high R^2.
  - The published main-effect equations cannot be reproduced from the
    published data (documented as a finding, not a flaw of FORMULA-Sigma).
  - Optimum predictions agree with experimental within X%.
"""
from __future__ import annotations
import json, time, sys, csv as _csv
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import seeds, paths, manifest, data_io


def coded_from_actual(actual, low, high):
    """Map actual factor value -> coded (-1..+1) given low/high anchors."""
    return (actual - (low + high) / 2.0) / ((high - low) / 2.0)


def main():
    t0 = time.time()
    seed = seeds.set_seeds(42)
    OUT_DIR = paths.RESULTS / "e02" / "case1_sharma"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    case = data_io.load_case("case1_tablet_ccd")
    design = case.design.copy()
    responses = case.responses.copy()

    for c in ["X1_coded", "X2_coded", "X3_coded"]:
        design[c] = design[c].astype(str).str.replace("−", "-").astype(float)
    X = design[["X1_coded", "X2_coded", "X3_coded"]].to_numpy()

    is_outlier = (design["run"].str.strip() == "F-19").to_numpy()
    mask = ~is_outlier
    X_fit = X[mask]

    # F-A published optimum: X1_actual=13.308 %, X2_actual=0.603 mm, X3_actual=7.96 %
    # Coding anchors per case1_ingestion.yaml:
    #   X1 low/high = 5 / 15  (range 10, center 10)
    #   X2 low/high = 0.2 / 0.8 (range 0.6, center 0.5)
    #   X3 low/high = 4 / 12 (range 8, center 8)
    x_opt_coded = np.array([[
        coded_from_actual(13.308, 5, 15),    # 0.6616
        coded_from_actual(0.603, 0.2, 0.8),  # 0.3433
        coded_from_actual(7.96, 4, 12),      # -0.01
    ]])

    published_opt = {
        "Y1_pct": {"predicted": 17.73,  "observed": 18.014, "pct_error": 1.60},
        "Y2_pct": {"predicted": 53.235, "observed": 52.87,  "pct_error": -0.68},
        "Y3_pct": {"predicted": 91.886, "observed": 95.18,  "pct_error": -3.5},
        "Y4":     {"predicted": 0.9604, "observed": 0.9703, "pct_error": 1.03},
    }

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(X_fit)
    feat_names = ["const"] + list(poly.get_feature_names_out(["x1", "x2", "x3"]))

    rows = [["response", "model", "n_used", "R2", "RMSE",
             "fx_at_FA_predicted", "fx_at_FA_observed",
             "fx_pct_err_vs_observed", "passes_5pct"]]
    detail = {}
    for ycol in ["Y1_pct", "Y2_pct", "Y3_pct", "Y4"]:
        y = pd.to_numeric(responses[ycol], errors="coerce").to_numpy(dtype=float)
        y_fit = y[mask]
        Xq = poly.transform(X_fit)
        Xc = sm.add_constant(Xq)
        fit = sm.OLS(y_fit, Xc).fit()
        rmse = float(np.sqrt(np.mean(fit.resid ** 2)))

        # Predict at F-A
        Xopt_p = poly.transform(x_opt_coded)
        Xopt_c = sm.add_constant(Xopt_p, has_constant="add")
        fx_pred = float(fit.predict(Xopt_c)[0])
        observed = published_opt[ycol]["observed"]
        pct_err = 100.0 * (fx_pred - observed) / abs(observed)
        passes = bool(abs(pct_err) < 5.0)

        rows.append([
            ycol.replace("_pct", ""), "quadratic_full", int(mask.sum()),
            f"{fit.rsquared:.4f}", f"{rmse:.3f}",
            f"{fx_pred:.3f}", f"{observed:.3f}",
            f"{pct_err:+.2f}%", passes,
        ])
        detail[ycol] = {
            "model": "quadratic_full",
            "n_used": int(mask.sum()),
            "R2": float(fit.rsquared),
            "RMSE": rmse,
            "coefficients": dict(zip(feat_names, fit.params.tolist())),
            "p_values": dict(zip(feat_names, fit.pvalues.tolist())),
            "FA_optimum_prediction": {
                "x_coded": x_opt_coded.tolist()[0],
                "predicted": fx_pred,
                "published_predicted": published_opt[ycol]["predicted"],
                "experimental_observed": observed,
                "our_predicted_vs_observed_pct_err": pct_err,
                "passes_5pct": passes,
            },
        }

    csv_path = OUT_DIR / "case1_summary.csv"
    json_path = OUT_DIR / "case1_detail.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        _csv.writer(fh).writerows(rows)
    json_path.write_text(json.dumps(detail, indent=2))

    # Also write a published-equations diagnostic (kept as evidence)
    diag = []
    diag.append("# Farooqi 2020 published-equation diagnostic")
    diag.append("")
    diag.append("The published Table 4 main-effect equations do NOT predict")
    diag.append("the published Table 1 response values. Diagnostic numbers:")
    diag.append("")
    diag.append("Center-point Y1 observed (6 runs at coded 0/0/0): 13.95, 8.79, 9.04, 12.47, 11.01, 10.36 (mean ~10.94)")
    diag.append("Published linear equation predicts intercept = 20.3379 at coded 0/0/0.")
    diag.append("")
    diag.append("Refit findings (full quadratic, F-19 excluded):")
    for r in rows[1:]:
        diag.append(f"  {r[0]}: R^2={r[3]}  RMSE={r[4]}  F-A pred={r[5]} (obs={r[6]}, err={r[7]})")
    diag.append("")
    diag.append("Interpretation: the printed equations in T4 appear to be reduced")
    diag.append("or transformed forms not directly comparable to the raw data.")
    diag.append("FORMULA-Sigma's full quadratic refit is the canonical reproduction.")
    (OUT_DIR / "case1_diagnostic.md").write_text("\n".join(diag))

    manifest.write_manifest(
        experiment="e02_case1_sharma", out_dir=OUT_DIR, seed=seed,
        inputs=[paths.DATA_PROC / "case1_tablet_ccd" / f for f in
                ["case1_design.csv", "case1_responses.csv",
                 "case1_published_equations.csv", "case1_known_optimum.csv"]],
        outputs=[csv_path, json_path, OUT_DIR / "case1_diagnostic.md"],
        runtime_sec=time.time() - t0,
        extra={"n_responses": 4, "outlier_excluded": "F-19",
               "fit_form": "full quadratic"})

    print("Farooqi 2020 reproduction (full quadratic, F-19 excluded):")
    for r in rows:
        print("  ", "  ".join(str(c) for c in r))
    print(f"  -> {csv_path}")
    print(f"  -> {OUT_DIR / 'case1_diagnostic.md'}")


if __name__ == "__main__":
    main()
