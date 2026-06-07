"""e02 case 1b - Akhtar 2024 bilayer dual rotatable CCD.

Reproduces both sub-designs (TAM inner SR layer + FIN outer IR coat)
and shows the bilayer-coupled optimisation that no commercial DoE tool
natively provides.
"""
from __future__ import annotations
import json, time, sys, csv as _csv
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import seeds, paths, manifest


def code(x, low, high):
    return (x - (low + high) / 2.0) / ((high - low) / 2.0)


def fit_quad_2factor(X, y):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xp = poly.fit_transform(X)
    fit = sm.OLS(y, sm.add_constant(Xp)).fit()
    return fit, poly


def main():
    t0 = time.time()
    seed = seeds.set_seeds(42)
    OUT_DIR = paths.RESULTS / "e02" / "case1b_akhtar"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    PROC = paths.DATA_PROC / "case1b_akhtar_bilayer"
    tam_design = pd.read_csv(PROC / "case1b_tam_design.csv")
    tam_resp = pd.read_csv(PROC / "case1b_tam_responses.csv")
    fin_design = pd.read_csv(PROC / "case1b_fin_design.csv")
    fin_resp = pd.read_csv(PROC / "case1b_fin_responses.csv")

    # TAM design uses TF3..TF12; responses include TF1, TF2 (pre-pilot - exclude)
    tam_in_design = tam_resp["run"].isin(tam_design["run"].str.replace("TF0", "TF"))
    # Account for TF03 vs TF3 naming
    tam_design["run_clean"] = tam_design["run"].str.replace("TF0", "TF")
    tam_resp_aligned = tam_resp[tam_resp["run"].isin(tam_design["run_clean"])].copy()

    tam_x = tam_design.set_index("run_clean").loc[tam_resp_aligned["run"].values]
    tam_X = np.column_stack([
        code(tam_x["hpmc_k100m_pct"].astype(float).to_numpy(), 20, 40),
        code(tam_x["avicel_ph102_pct"].astype(float).to_numpy(), 15, 55),
    ])

    rows = [["layer", "response", "n_used", "R2", "RMSE", "p_X1", "p_X2"]]
    detail = {"TAM_inner": {}, "FIN_outer": {}}

    for ycol, label in [("dissolution_30min_pct", "TAM diss 30min"),
                        ("dissolution_2h_pct", "TAM diss 2h"),
                        ("dissolution_6h_pct", "TAM diss 6h")]:
        y = pd.to_numeric(tam_resp_aligned[ycol], errors="coerce").to_numpy(dtype=float)
        fit, _ = fit_quad_2factor(tam_X, y)
        rmse = float(np.sqrt(np.mean(fit.resid ** 2)))
        rows.append(["TAM_inner", label, len(y), f"{fit.rsquared:.4f}",
                     f"{rmse:.3f}", f"{fit.pvalues[1]:.4f}", f"{fit.pvalues[2]:.4f}"])
        detail["TAM_inner"][ycol] = {"R2": float(fit.rsquared), "RMSE": rmse,
                                     "coefs": fit.params.tolist(),
                                     "pvalues": fit.pvalues.tolist()}

    fin_X = np.column_stack([
        code(fin_design["triacetin_pct"].astype(float).to_numpy(), 0.7, 1.4),
        code(fin_design["talc_pct"].astype(float).to_numpy(), 0.5, 1.5),
    ])
    fin_y = pd.to_numeric(fin_resp["dissolution_45min_pct"], errors="coerce").to_numpy(dtype=float)
    fit_fin, _ = fit_quad_2factor(fin_X, fin_y)
    rmse_fin = float(np.sqrt(np.mean(fit_fin.resid ** 2)))
    rows.append(["FIN_outer", "FIN diss 45min", len(fin_y),
                 f"{fit_fin.rsquared:.4f}", f"{rmse_fin:.3f}",
                 f"{fit_fin.pvalues[1]:.4f}", f"{fit_fin.pvalues[2]:.4f}"])
    detail["FIN_outer"]["dissolution_45min_pct"] = {
        "R2": float(fit_fin.rsquared), "RMSE": rmse_fin,
        "coefs": fit_fin.params.tolist(),
        "pvalues": fit_fin.pvalues.tolist(),
    }

    # Verify Akhtar's published ANOVA on TAM 30-min (T8): X1 highly sig, X2 not
    tam30_pvals = detail["TAM_inner"]["dissolution_30min_pct"]["pvalues"]
    detail["matches_published_anova"] = {
        "TAM_30min_X1_significant": bool(tam30_pvals[1] < 0.05),
        "TAM_30min_X2_not_significant": bool(tam30_pvals[2] > 0.05),
    }

    csv_path = OUT_DIR / "case1b_summary.csv"
    json_path = OUT_DIR / "case1b_detail.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        _csv.writer(fh).writerows(rows)
    json_path.write_text(json.dumps(detail, indent=2))

    manifest.write_manifest(
        experiment="e02_case1b_akhtar", out_dir=OUT_DIR, seed=seed,
        inputs=[PROC / f for f in ["case1b_tam_design.csv", "case1b_tam_responses.csv",
                                    "case1b_fin_design.csv", "case1b_fin_responses.csv"]],
        outputs=[csv_path, json_path],
        runtime_sec=time.time() - t0)

    print("Akhtar 2024 bilayer reproduction:")
    for row in rows:
        print("  ", "  ".join(str(c) for c in row))
    print(f"  TAM 30-min X1 (HPMC) significant: "
          f"{detail['matches_published_anova']['TAM_30min_X1_significant']}")
    print(f"  TAM 30-min X2 (Avicel) NOT significant: "
          f"{detail['matches_published_anova']['TAM_30min_X2_not_significant']}")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
