"""Project path constants. All experiment outputs land under RESULTS/<eXX>/."""
from pathlib import Path

# Repo root resolved relative to this file (04_experiments/common/paths.py),
# so the harness runs from any clone location.
ROOT       = Path(__file__).resolve().parents[2]
DATA_RAW   = ROOT / "03_data" / "raw"
DATA_PROC  = ROOT / "03_data" / "processed"
EXPERIMENTS = ROOT / "04_experiments"
RESULTS    = ROOT / "05_results"
TABLES_RAW = RESULTS / "tables_raw"
FIGURES_RAW = RESULTS / "figures_raw"

for d in (RESULTS, TABLES_RAW, FIGURES_RAW,
          RESULTS / "e01", RESULTS / "e02", RESULTS / "e03",
          RESULTS / "e04"):
    d.mkdir(parents=True, exist_ok=True)
