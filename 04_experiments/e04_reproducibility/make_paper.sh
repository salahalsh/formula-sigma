#!/usr/bin/env bash
# Reproduce every figure and table in the FORMULA-Sigma paper from a
# clean environment. Documented in Section 3.4 of the manuscript.
#
# Usage:
#   bash make_paper.sh                      # run all
#   bash make_paper.sh e01                  # one phase only
#   bash make_paper.sh --check              # verify hashes only

set -e

# Resolve repo paths relative to this script (04_experiments/e04_reproducibility/),
# so the harness runs from any clone location.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "$HERE/.." && pwd)"
PROJECT="$(cd "$EXP/.." && pwd)"
PY="${PY:-python}"

cd "$EXP"

run_e01() {
  echo "=== e01 cross-tool correctness (8 designs + 11 model families) ==="
  "$PY" e01_cross_tool/01_ccd_correctness.py
  "$PY" e01_cross_tool/02_bbd_correctness.py
  "$PY" e01_cross_tool/03_scheffe_mixture.py
  "$PY" e01_cross_tool/04_pbdesign.py
  "$PY" e01_cross_tool/05_lhs_correctness.py
  "$PY" e01_cross_tool/07_full_factorial.py
  "$PY" e01_cross_tool/08_fractional_factorial.py
  "$PY" e01_cross_tool/09_d_optimal.py
  "$PY" e01_cross_tool/10_model_zoo.py
  "$PY" e01_cross_tool/06_summary_table.py
}

run_e02() {
  echo "=== e02 retrospective reproduction ==="
  "$PY" e02_retrospective/case1_sharma/reproduce.py
  "$PY" e02_retrospective/case1b_akhtar/reproduce.py
  "$PY" e02_retrospective/case2_arif/reproduce.py
  "$PY" e02_retrospective/case3_boscolo/reproduce.py
  "$PY" e02_retrospective/case4_nemr/reproduce.py
  "$PY" e02_retrospective/99_summary_table.py
}

run_e03() {
  echo "=== e03 synthetic benchmarks ==="
  "$PY" e03_synthetic/bo/run_bo_benchmarks.py
  "$PY" e03_synthetic/pds/run_pds_recovery.py
  "$PY" e03_synthetic/robust/run_robust_recovery.py
  "$PY" e03_synthetic/pareto/run_zdt1.py
}

run_e04() {
  echo "=== e04 reproducibility check ==="
  "$PY" e04_reproducibility/verify_hashes.py
}

run_figs() {
  echo "=== figures (main F1-F7 + anomaly + PRISMA) ==="
  # Figure 8 and the supplementary figures render from the proprietary
  # platform DB and are shipped as static assets (figures_static/); they
  # are not regenerated here.
  "$PY" figures/build_main_figures.py
  "$PY" figures/build_sharma_anomaly_figure.py
  "$PY" figures/build_prisma_figure.py
}

case "${1:-all}" in
  all)
    run_e01
    run_e02
    run_e03
    run_figs
    run_e04
    ;;
  e01) run_e01 ;;
  e02) run_e02 ;;
  e03) run_e03 ;;
  e04) run_e04 ;;
  figs) run_figs ;;
  --check)
    "$PY" e04_reproducibility/verify_hashes.py
    ;;
  *)
    echo "Usage: $0 [all|e01|e02|e03|e04|figs|--check]"
    exit 1
    ;;
esac

echo "DONE."
