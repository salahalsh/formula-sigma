"""e01-07: Full factorial design generation + agreement.

Confirms multi-level full factorial generation against pyDOE3.
Tests both 3-factor 3-level (27 runs) and 2-factor 5-level (25 runs).
"""
from __future__ import annotations
import json, time, sys, csv as _csv
from pathlib import Path

import numpy as np
import pyDOE3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import seeds, designs, paths, manifest


def main():
    t0 = time.time()
    seed = seeds.set_seeds(42)
    OUT_DIR = paths.RESULTS / "e01" / "e01_07_full_factorial"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        ("3F-3L", [3, 3, 3], 27),
        ("2F-5L", [5, 5],     25),
        ("4F-2L", [2, 2, 2, 2], 16),
    ]
    rows = [["case", "expected_runs", "ours_runs", "pyDOE3_runs",
             "ours_unique_per_factor", "set_equal", "passes"]]
    detail = {}

    for label, levels, expected in cases:
        D_ours  = designs.full_factorial(levels)
        D_pydoe = pyDOE3.fullfact(levels).astype(int)

        def canon(M):
            return np.array(sorted(map(tuple, M)))

        equal = bool(np.array_equal(canon(D_ours), canon(D_pydoe)))
        uniq = [len(np.unique(D_ours[:, j])) for j in range(D_ours.shape[1])]
        ok = (len(D_ours) == expected) and (len(D_pydoe) == expected) and equal
        rows.append([label, expected, len(D_ours), len(D_pydoe), uniq, equal, ok])
        detail[label] = {"expected": expected, "ours": len(D_ours),
                         "pyDOE3": len(D_pydoe), "set_equal": equal,
                         "unique_levels_per_factor": uniq, "passes": ok}

    csv_path = OUT_DIR.parent / "e01_07_full_factorial.csv"
    json_path = OUT_DIR.parent / "e01_07_full_factorial.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        _csv.writer(fh).writerows(rows)
    json_path.write_text(json.dumps(detail, indent=2))

    manifest.write_manifest(
        experiment="e01_07_full_factorial", out_dir=OUT_DIR, seed=seed,
        inputs=[], outputs=[csv_path, json_path],
        runtime_sec=time.time() - t0,
        extra={"n_cases": len(cases)})

    print("Full factorial verification:")
    for r in rows: print("  ", "  ".join(str(c) for c in r))
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
