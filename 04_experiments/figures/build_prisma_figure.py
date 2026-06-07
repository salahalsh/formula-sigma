"""Figure 4 - PRISMA-style case selection flow for the L1 literature scan.

Numbers come from L1_SCORING_REPORT.md and case_shortlist.md.

187 raw hits -> 60 metadata pulled -> 12 scored -> 4 primary + 1 supporting
                  (127 excluded)      (48 excluded)   (8 lower-scoring + 1 swap)

Five stages laid out vertically with right-side count chips and aside boxes
for exclusions. All elements live inside one set of axes for total control
over alignment.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, plotting

plotting.apply_paper_style()
W = plotting.WONG
FIG_DIR = paths.FIGURES_RAW
FIG_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(9.5, 10.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 14); ax.axis("off")
ax.set_aspect("auto")


# Geometry
MAIN_X = 4.5            # centre of main column
MAIN_W = 7.0            # main box width
ASIDE_X = 11.5          # centre of right-side count chip
ASIDE_W = 4.4
CHIP_X = 9.2            # right-side exclusion chip
CHIP_W = 4.0


def stage_box(y, header_h, body_h, header, body,
              header_fc, header_fs=11):
    """Draw a stage = header bar + body bar, stacked, no gap."""
    # Header
    ax.add_patch(mpatches.FancyBboxPatch(
        (MAIN_X - MAIN_W / 2, y + body_h / 2),
        MAIN_W, header_h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=header_fc, edgecolor="black", linewidth=1.0))
    ax.text(MAIN_X, y + body_h / 2 + header_h / 2, header,
             ha="center", va="center", fontsize=header_fs, weight="bold")
    # Body
    ax.add_patch(mpatches.FancyBboxPatch(
        (MAIN_X - MAIN_W / 2, y - body_h / 2),
        MAIN_W, body_h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor="white", edgecolor="#888", linewidth=0.8))
    ax.text(MAIN_X, y, body,
             ha="center", va="center", fontsize=9.5)
    return y + body_h / 2 + header_h, y - body_h / 2  # top, bottom


def count_chip(y, n_text, sub_text, fc=W["sky"]):
    ax.add_patch(mpatches.FancyBboxPatch(
        (ASIDE_X - ASIDE_W / 2, y - 0.45), ASIDE_W, 0.95,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=fc, edgecolor="black", linewidth=0.9))
    ax.text(ASIDE_X, y + 0.12, n_text,
             ha="center", va="center", fontsize=11, weight="bold")
    ax.text(ASIDE_X, y - 0.22, sub_text,
             ha="center", va="center", fontsize=8, color="#222")


def exclusion_chip(y, text):
    ax.add_patch(mpatches.FancyBboxPatch(
        (CHIP_X - CHIP_W / 2, y - 0.40), CHIP_W, 0.85,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor="#FCE4EC", edgecolor=W["red"], linewidth=0.9))
    ax.text(CHIP_X, y, text,
             ha="center", va="center", fontsize=8, color="#8E2222",
             style="italic")


def down_arrow(y_from, y_to):
    ax.annotate("", xy=(MAIN_X, y_to), xytext=(MAIN_X, y_from),
                 arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))


# --- Stage 1: Identification ---
top1, bot1 = stage_box(
    y=12.3, header_h=0.55, body_h=1.0,
    header="1.  Identification through PubMed",
    body="8 query strands covering CCD, BBD, mixture, D-optimal,\n"
         "PB screening, mixture-process, robust DoE, ICH Q8 PDS\n"
         "(2020-2025, English, full-text PMC available)",
    header_fc=W["yellow"])
count_chip(12.55, "187", "records identified", fc="#FFF59D")
down_arrow(bot1 - 0.05, bot1 - 0.55)

# --- Stage 2: Screening ---
top2, bot2 = stage_box(
    y=10.0, header_h=0.55, body_h=1.0,
    header="2.  Screening (title + abstract)",
    body="Top 15 per slot pulled for metadata review:\n"
         "year, design family, n_runs, responses, full-text\n"
         "availability, pharma dosage form",
    header_fc=W["yellow"])
count_chip(10.25, "60", "records reviewed", fc="#FFF59D")
exclusion_chip(9.30, "127 excluded\n(off-topic /\nno per-run data)")
down_arrow(bot2 - 0.05, bot2 - 0.55)

# --- Stage 3: Eligibility ---
top3, bot3 = stage_box(
    y=7.7, header_h=0.55, body_h=1.1,
    header="3.  Eligibility scoring (12-point rubric)",
    body="Top 3 candidates per slot scored on:\n"
         "design clarity (3 pts) - response transparency (3 pts) -\n"
         "PMC table machine-extractability (3 pts) -\n"
         "reviewer-defence value (3 pts)",
    header_fc=W["yellow"])
count_chip(7.95, "12", "records scored", fc="#FFF59D")
exclusion_chip(7.00, "48 excluded\n(figure-only\ndata)")
down_arrow(bot3 - 0.05, bot3 - 0.55)

# --- Stage 4: Inclusion ---
top4, bot4 = stage_box(
    y=5.0, header_h=0.55, body_h=1.6,
    header="4.  Inclusion in retrospective benchmark",
    body="4 primary picks (one per design slot):\n"
         "  - Kotamarthy 2022 (CCD)  ->  SWAPPED to Farooqi 2020\n"
         "       (Kotamarthy per-run data was figure-only;\n"
         "        Farooqi has both per-run table + polynomial T4)\n"
         "  - Arif 2022 (D-optimal mixture-process, NLC)\n"
         "  - Boscolo 2023 (BBD, UDCA freeze-dried NS)\n"
         "  - Nemr 2022 (D-optimal categorical, ocular bilosomes)",
    header_fc="#A5D6A7")
count_chip(5.25, "4", "primary picks", fc="#C8E6C9")
exclusion_chip(4.00, "8 excluded\n(4 paywalled,\n4 lower-scoring\nbackups)")
down_arrow(bot4 - 0.05, bot4 - 0.55)

# --- Stage 5: Final dataset ---
top5, bot5 = stage_box(
    y=2.4, header_h=0.55, body_h=1.0,
    header="5.  Final retrospective dataset",
    body="5 cases (4 primary + 1 supporting):\n"
         "Farooqi, Arif, Boscolo, Nemr +  Akhtar 2024 bilayer\n"
         "(supporting demonstration of coupled-design analysis)",
    header_fc=W["sky"])
count_chip(2.65, "5", "cases analysed", fc="#B3E5FC")

fig.suptitle("Figure 4. PRISMA-style flow of retrospective case selection  "
              "(L1 PubMed scan, 2020-2025)",
              fontsize=11.5, y=0.985)
fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02)
pdf, png = plotting.save_both(fig, "Figure_4_prisma_case_selection", FIG_DIR)
print(f"-> {pdf.name} + {png.name}")
