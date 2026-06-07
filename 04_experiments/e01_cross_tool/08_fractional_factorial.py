"""e01-08: Fractional factorial 2^(k-p) design correctness.

Tests a 2^(4-1) resolution IV design with generator d = abc.
Confirms (a) run count = 8, (b) levels are pm 1, (c) design is
orthogonal (X'X diagonal), (d) effect D is aliased with ABC.
"""
from __future__ import annotations
import json, time, sys, csv as _csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import seeds, designs, paths, manifest


def main():
    t0 = time.time()
    seed = seeds.set_seeds(42)
    OUT_DIR = paths.RESULTS / "e01" / "e01_08_fractional"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2^(4-1) generator: d = a b c (resolution IV)
    D = designs.fractional_factorial("a b c abc")
    runs, factors = D.shape

    levels_ok = bool(set(np.unique(D)) <= {-1.0, 1.0})
    runs_correct = (runs == 8)
    factors_correct = (factors == 4)

    # Orthogonality
    gram = D.T @ D
    diag = np.diag(gram)
    off_diag = gram - np.diag(diag)
    orthogonal = bool(np.allclose(off_diag, 0))
    balanced = bool(np.allclose(diag, runs))

    # Aliasing check: column D should equal column A*B*C (resolution IV)
    abc = D[:, 0] * D[:, 1] * D[:, 2]
    d_col = D[:, 3]
    aliased = bool(np.array_equal(abc, d_col))

    rows = [["metric", "value"],
            ["design", "2^(4-1)_IV (d=abc)"],
            ["n_runs", runs],
            ["n_factors", factors],
            ["levels_are_pm1", levels_ok],
            ["balanced", balanced],
            ["orthogonal", orthogonal],
            ["d_aliased_with_abc", aliased],
            ["passes_all", runs_correct and factors_correct and levels_ok
                           and balanced and orthogonal and aliased]]

    csv_path = OUT_DIR.parent / "e01_08_fractional.csv"
    json_path = OUT_DIR.parent / "e01_08_fractional.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        _csv.writer(fh).writerows(rows)
    json_path.write_text(json.dumps({
        "design": "2^(4-1)_IV",
        "generator": "d = abc",
        "n_runs": runs, "n_factors": factors,
        "levels_are_pm1": levels_ok,
        "balanced": balanced, "orthogonal": orthogonal,
        "d_aliased_with_abc": aliased,
        "passes_all": bool(runs_correct and factors_correct and levels_ok
                           and balanced and orthogonal and aliased),
    }, indent=2))

    manifest.write_manifest(
        experiment="e01_08_fractional", out_dir=OUT_DIR, seed=seed,
        inputs=[], outputs=[csv_path, json_path],
        runtime_sec=time.time() - t0)

    print("Fractional factorial 2^(4-1)_IV (d=abc):")
    for r in rows[1:]: print("  ", "  ".join(str(c) for c in r))
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
