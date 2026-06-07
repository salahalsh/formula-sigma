"""e02 case 2 - Arif 2022 D-optimal combined mixture-process.

Reproduces fits for PS, PDI, EE from the 22-run D-optimal combined
mixture-process design and compares against published ANOVA (Table 3
of the original paper).
"""
from __future__ import annotations
import json, time, sys, csv as _csv
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import seeds, paths, manifest, data_io


def main():
    t0 = time.time()
    seed = seeds.set_seeds(42)
    OUT_DIR = paths.RESULTS / "e02" / "case2_arif"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    case = data_io.load_case("case2_nanoparticle_mixture")
    d = case.design.copy()
    r = case.responses.copy()
    d.columns = [c.strip() for c in d.columns]
    r.columns = [c.strip() for c in r.columns]

    # Independent variables: Precirol %, Labrasol % (mixture A+B=100%) + surfactant C
    A = d["precirol_pct"].astype(float).to_numpy() / 100.0  # mixture fraction
    B = d["labrasol_pct"].astype(float).to_numpy() / 100.0
    C = d["surfactant_pct"].astype(float).to_numpy()

    # Published reduced model uses A, B (mixture) + C (process) with
    # interactions AB, AC, BC. Scheffe-quadratic + process variable.
    # Construct design matrix (no intercept; mixture components sum to 1):
    cols = [A, B, A * B, A * C, B * C]
    names = ["A_precirol", "B_labrasol", "A*B", "A*C", "B*C"]
    Z = np.column_stack(cols)

    published_R2 = {"particle_size_nm": 0.9454, "pdi": 0.9775, "ee_pct": 0.9788}
    rows = [["response", "model", "n_used", "R2_ours", "R2_published",
             "R2_match", "RMSE", "n_terms"]]
    detail = {}
    for ycol, label in [("particle_size_nm", "PS"), ("pdi", "PDI"), ("ee_pct", "EE")]:
        y = pd.to_numeric(r[ycol].astype(str).str.replace("±.*", "", regex=True),
                          errors="coerce").to_numpy(dtype=float)
        fit = sm.OLS(y, Z).fit()
        r2 = float(fit.rsquared)
        rmse = float(np.sqrt(np.mean(fit.resid ** 2)))
        r2p = published_R2[ycol]
        rows.append([label, "Scheffe + mix*process", len(y),
                     f"{r2:.4f}", f"{r2p:.4f}",
                     "Y" if abs(r2 - r2p) < 0.05 else "N",
                     f"{rmse:.4f}", len(names)])
        detail[ycol] = {
            "model": "Scheffe quadratic + mix*process interactions",
            "coefficients": dict(zip(names, fit.params.tolist())),
            "p_values": dict(zip(names, fit.pvalues.tolist())),
            "R2_ours": r2, "R2_published": r2p,
            "RMSE": rmse,
            "passes_R2_match_5pct": bool(abs(r2 - r2p) < 0.05),
        }

    # Verify published optimum: 80.55% A, 19.45% B, 5% C -> PS=157.9, PDI=0.241, EE=85
    x_opt = np.array([[0.8055, 0.1945, 5]])
    Z_opt = np.column_stack([
        x_opt[:, 0], x_opt[:, 1],
        x_opt[:, 0] * x_opt[:, 1],
        x_opt[:, 0] * x_opt[:, 2],
        x_opt[:, 1] * x_opt[:, 2],
    ])
    opt_predictions = {}
    for ycol in ["particle_size_nm", "pdi", "ee_pct"]:
        y = pd.to_numeric(r[ycol].astype(str).str.replace("±.*", "", regex=True),
                          errors="coerce").to_numpy(dtype=float)
        fit = sm.OLS(y, Z).fit()
        opt_predictions[ycol] = float(fit.predict(Z_opt)[0])
    detail["optimum_prediction"] = {
        "x_at_optimum": [0.8055, 0.1945, 5],
        "predicted_particle_size_nm": opt_predictions["particle_size_nm"],
        "predicted_pdi": opt_predictions["pdi"],
        "predicted_ee_pct": opt_predictions["ee_pct"],
        "published_predicted_PS": 157.9,
        "published_predicted_PDI": 0.241,
        "published_predicted_EE": 85,
        "observed_PS": 152.0, "observed_PDI": 0.230, "observed_EE": 88.0,
    }

    csv_path = OUT_DIR / "case2_summary.csv"
    json_path = OUT_DIR / "case2_detail.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        _csv.writer(fh).writerows(rows)
    json_path.write_text(json.dumps(detail, indent=2))

    manifest.write_manifest(
        experiment="e02_case2_arif", out_dir=OUT_DIR, seed=seed,
        inputs=[paths.DATA_PROC / "case2_nanoparticle_mixture" / f for f in
                ["case2_design.csv", "case2_responses.csv"]],
        outputs=[csv_path, json_path],
        runtime_sec=time.time() - t0)

    print("Arif 2022 reproduction:")
    for row in rows:
        print("  ", "  ".join(str(c) for c in row))
    print("  optimum prediction:")
    for k, v in opt_predictions.items():
        print(f"    {k}: ours={v:.3f}  published_pred={detail['optimum_prediction'].get(f'published_predicted_{k.split(chr(95))[0][:2].upper()}','-')}")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
