"""e01-06: Build Table 2 (manuscript) from all e01 sub-experiments."""
from __future__ import annotations
import json, csv as _csv, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, manifest, seeds


def main():
    t0 = time.time()
    seed = seeds.set_seeds(42)

    e01 = paths.RESULTS / "e01"
    subs = [
        ("e01_01_ccd.json",            "Central composite (3F, rotatable)"),
        ("e01_02_bbd.json",            "Box-Behnken (3F)"),
        ("e01_03_scheffe.json",        "Scheffe simplex-lattice {3,3}"),
        ("e01_04_pb.json",             "Plackett-Burman (11F)"),
        ("e01_05_lhs.json",            "Latin hypercube (50, 4)"),
        ("e01_07_full_factorial.json", "Full factorial (multi-level)"),
        ("e01_08_fractional.json",     "Fractional factorial 2^(4-1)_IV"),
        ("e01_09_d_optimal.json",      "D-optimal (Fedorov exchange)"),
        ("e01_10_model_zoo.json",      "Model zoo (11 families + ensemble)"),
    ]
    rows = [["design", "n_runs", "fit_R2", "max_rel_coef_diff", "passes"]]
    for fname, label in subs:
        p = e01 / fname
        if not p.exists():
            rows.append([label, "MISSING", "", "", ""])
            continue
        d = json.loads(p.read_text())
        n_runs = d.get("n_runs", d.get("n_runs_observed", "n/a"))
        r2 = d.get("fit_R2", "n/a")
        if isinstance(r2, float):
            r2 = f"{r2:.6f}"
        rel = d.get("max_rel_coef_diff_sm_vs_numpy",
                    d.get("max_rel_coef_diff_ours_vs_numpy", "n/a"))
        if isinstance(rel, float):
            rel = f"{rel:.2e}"
        passes = d.get("passes_1e-6",
                       d.get("passes_all",
                             d.get("passes",
                                   d.get("passes_at_3x_ratio",
                                         d.get("passes_at_0.80_jaccard", "n/a")))))
        if isinstance(passes, dict):
            passes = "n/a"
        # Model zoo: count passing models
        if "model_zoo" in fname:
            n_pass = sum(1 for v in d.values()
                         if isinstance(v, dict) and v.get("passes_R2_above_0.5"))
            n_total = sum(1 for v in d.values() if isinstance(v, dict))
            passes = f"{n_pass}/{n_total}"
            r2 = f"max {max((v['R2'] for v in d.values() if isinstance(v, dict) and 'R2' in v), default=0):.4f}"
            rel = "model-fit, n/a"
            n_runs = "20 (CCD)"
        rows.append([label, n_runs, r2, rel, passes])

    out_csv = paths.TABLES_RAW / "Table2_e01_cross_tool_correctness.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        _csv.writer(fh).writerows(rows)

    manifest.write_manifest(
        experiment="e01_06_summary", out_dir=e01,
        seed=seed,
        inputs=[e01 / f for f, _ in subs if (e01 / f).exists()],
        outputs=[out_csv], runtime_sec=time.time() - t0)

    print("Table 2:")
    for r in rows:
        print("  ", "  ".join(str(c) for c in r))
    print(f"  -> {out_csv}")


if __name__ == "__main__":
    main()
