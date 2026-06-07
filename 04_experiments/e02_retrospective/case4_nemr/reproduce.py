"""e02 case 4 - Nemr 2022 D-optimal with categorical factor.

Reproduces the D-optimal fit for EE%, PS, ZP, PDI, Q2h, Q24h with
ea_type as a 3-level categorical. Excludes D22 ZP anomaly per
case4_ingestion.yaml.
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
    OUT_DIR = paths.RESULTS / "e02" / "case4_nemr"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    case = data_io.load_case("case4_optimization")
    d = case.design.copy()
    r = case.responses.copy()

    # Build full design matrix: PC:EA ratio + HA% + categorical (3 levels)
    pc_ea = d["pc_ea_ratio"].astype(float).to_numpy()
    ha = d["ha_pct"].astype(float).to_numpy()
    ea_type = d["ea_type"].astype(str).to_numpy()

    # One-hot encode ea_type (drop one for reference)
    ea_sc = (ea_type == "SC").astype(float)
    ea_sdc = (ea_type == "SDC").astype(float)
    # ea_stc = reference category

    # Full model: PC:EA + HA + cat + interactions PC:EA*HA + PC:EA^2 + HA^2
    X = np.column_stack([
        pc_ea, ha, ea_sc, ea_sdc,
        pc_ea * ha, pc_ea ** 2, ha ** 2,
    ])
    names = ["PC:EA", "HA", "EA_type_SC", "EA_type_SDC", "PC:EA*HA",
             "PC:EA^2", "HA^2"]

    rows = [["response", "n_used", "R2", "RMSE", "p_PC:EA", "p_HA", "p_EA_SC",
             "p_EA_SDC"]]
    detail = {}
    for ycol in ["ee_pct", "ps_nm", "zp_mV", "pdi", "q2h_pct", "q24h_pct"]:
        y = pd.to_numeric(r[ycol].astype(str).str.replace("−", "-"),
                          errors="coerce").to_numpy(dtype=float)
        # Exclude D22 zp anomaly only for zp fit
        mask = np.ones(len(y), dtype=bool)
        if ycol == "zp_mV":
            mask = (d["run"] != "D22").to_numpy()
        Xfit = X[mask]; yfit = y[mask]
        Xc = sm.add_constant(Xfit)
        fit = sm.OLS(yfit, Xc).fit()
        rmse = float(np.sqrt(np.mean(fit.resid ** 2)))
        rows.append([
            ycol, int(mask.sum()), f"{fit.rsquared:.4f}", f"{rmse:.3f}",
            f"{fit.pvalues[1]:.4f}", f"{fit.pvalues[2]:.4f}",
            f"{fit.pvalues[3]:.4f}", f"{fit.pvalues[4]:.4f}",
        ])
        detail[ycol] = {
            "n_used": int(mask.sum()),
            "R2": float(fit.rsquared),
            "RMSE": rmse,
            "coefficients": dict(zip(["const"] + names, fit.params.tolist())),
            "p_values": dict(zip(["const"] + names, fit.pvalues.tolist())),
        }

    # Check published optimum: theoretical EE=82.06, PS=414.95, Q2h=42.338, Q24h=74.723
    # The paper does not report the optimum x values explicitly in T2; we'd
    # need to re-derive via desirability. For this report we just note
    # the response targets the optimum should hit.
    detail["published_optimum_targets"] = {
        "ee_pct_theoretical": 82.06, "ee_pct_observed": 81.81,
        "ps_nm_theoretical": 414.95, "ps_nm_observed": 432.45,
        "q2h_theoretical": 42.338, "q2h_observed": 42.652,
        "q24h_theoretical": 74.723, "q24h_observed": 75.138,
    }

    csv_path = OUT_DIR / "case4_summary.csv"
    json_path = OUT_DIR / "case4_detail.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        _csv.writer(fh).writerows(rows)
    json_path.write_text(json.dumps(detail, indent=2))

    manifest.write_manifest(
        experiment="e02_case4_nemr", out_dir=OUT_DIR, seed=seed,
        inputs=[paths.DATA_PROC / "case4_optimization" / f for f in
                ["case4_design.csv", "case4_responses.csv"]],
        outputs=[csv_path, json_path],
        runtime_sec=time.time() - t0,
        extra={"data_anomaly": "D22 ZP excluded only from ZP fit"})

    print("Nemr 2022 reproduction:")
    for row in rows:
        print("  ", "  ".join(str(c) for c in row))
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
