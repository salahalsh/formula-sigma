"""Generate the 6 manuscript main figures in this file (publication-quality layout).
PRISMA (Figure 4) and Farooqi anomaly (Figure 6) live in their own builders.

F1  Platform architecture (layered diagram)
F2  Competitor feature comparison heatmap
F3  Cross-tool agreement bar (-log10 scale)
F5  Retrospective case studies panel (2x2)
F7  Synthetic benchmarks (2x2 grid: BO + PDS + robust + Pareto)
F8  PDS "money shot" - real 2D probability heatmap regenerated from the service

Each figure is laid out so there is no text-bar overlap, no clipped
annotations, no truncated labels, and aspect ratios are matched to content
density. PDFs are saved as vector, PNGs at 300 dpi.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, plotting

plotting.apply_paper_style()
W = plotting.WONG
FIG_DIR = paths.FIGURES_RAW
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    pdf, png = plotting.save_both(fig, name, FIG_DIR)
    print(f"  -> {pdf.name} + {png.name}")


# ============================================================
# F1 - Platform architecture
# ============================================================
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_aspect("auto")

    layers = [
        ("Web UI  (React + Tailwind + DaisyUI)",       6.6, "#E0F2F1", W["green"]),
        ("REST API  (Django REST Framework)",          5.5, "#E0F7FA", W["sky"]),
        ("Service layer  (designs / models / opt / PDS / robust)",
                                                        4.4, "#FFF3E0", W["orange"]),
        ("Engine wrappers  (statsmodels / sklearn / GPyTorch / pymoo / pyDOE3)",
                                                        3.3, "#FFE0E6", W["red"]),
        ("Persistence  (PostgreSQL + Celery + Redis)",  2.2, "#F3E5F5", W["purple"]),
    ]
    box_x0, box_x1 = 0.6, 10.4
    for label, y, fill, edge in layers:
        ax.add_patch(mpatches.FancyBboxPatch(
            (box_x0, y - 0.45), box_x1 - box_x0, 0.9,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=fill, edgecolor=edge, linewidth=1.3))
        ax.text((box_x0 + box_x1) / 2, y, label,
                ha="center", va="center", fontsize=10, weight="bold")

    # Down arrows between layers (start/end at clear gap between boxes)
    for y_top, y_bot in zip([6.6, 5.5, 4.4, 3.3], [5.5, 4.4, 3.3, 2.2]):
        ax.annotate("", xy=((box_x0 + box_x1) / 2, y_bot + 0.50),
                    xytext=((box_x0 + box_x1) / 2, y_top - 0.50),
                    arrowprops=dict(arrowstyle="->", color="grey", lw=0.9))

    # Side annotation - inside axes bounds
    side_x = box_x1 + 0.6
    ax.text(side_x, 6.6, "Browser", ha="left", va="center",
            fontsize=9, color="#444", style="italic")
    ax.text(side_x, 5.5, "Auth + ACL", ha="left", va="center",
            fontsize=9, color="#444", style="italic")
    ax.text(side_x, 4.4, "Per-tool services", ha="left", va="center",
            fontsize=9, color="#444", style="italic")
    ax.text(side_x, 3.3, "Open libraries", ha="left", va="center",
            fontsize=9, color="#444", style="italic")
    ax.text(side_x, 2.2, "Async jobs", ha="left", va="center",
            fontsize=9, color="#444", style="italic")

    # Headline box (capability summary) - bottom row
    ax.add_patch(mpatches.FancyBboxPatch(
        (box_x0, 0.4), box_x1 - box_x0, 1.05,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="#F5F5F5", edgecolor="#888", linewidth=0.8))
    ax.text((box_x0 + box_x1) / 2, 0.92,
            "8 design families  |  11 surrogate model families  |  "
            "4 optimisers  |  Monte Carlo PDS (ICH Q8)",
            ha="center", va="center", fontsize=9.5, weight="bold", color="#222")

    fig.suptitle("Figure 1. FORMULA-Sigma platform architecture",
                 fontsize=11, y=0.99)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.03)
    save(fig, "Figure_1_architecture")


# ============================================================
# F2 - Competitor feature comparison heatmap
# ============================================================
def fig2_competitor_heatmap():
    tools = ["FORMULA-Sigma", "Design-Expert", "JMP Pro", "Modde", "pyDOE3", "sklearn"]
    rows = [
        ("Open source",                  [0, 0, 0, 0, 1, 1]),
        ("Web UI",                       [1, 0, 0, 0, 0, 0]),
        ("Programmatic API",             [1, 0.5, 0.5, 0, 1, 1]),
        ("8 design families",            [1, 1, 1, 1, 0.5, 0]),
        ("D-optimal w/ categorical",     [1, 1, 1, 1, 0, 0]),
        ("Mixture-process combined",     [1, 1, 0.5, 1, 0, 0]),
        ("Scheffe mixture fit",          [1, 1, 1, 1, 0, 0]),
        ("GP / RF / NN surrogates",      [1, 0, 0.5, 0, 0, 1]),
        ("Bayesian optimisation",        [1, 0, 0.5, 0, 0, 0]),
        ("Probabilistic design space",   [1, 0.5, 0.5, 0.5, 0, 0]),
        ("NSGA-II Pareto",               [1, 0.5, 0.5, 0.5, 0, 0]),
        ("Robust optimisation",          [1, 0.5, 0.5, 1, 0, 0]),
        ("Reproducibility (scripts)",    [1, 0, 0, 0, 1, 1]),
        ("Free / academic licence",      [0, 0.3, 0.3, 0, 1, 1]),
    ]
    labels = [r[0] for r in rows]
    M = np.array([r[1] for r in rows])

    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    im = ax.imshow(M, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(tools)))
    ax.set_xticklabels(tools, rotation=28, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(axis="x", which="both", length=0, pad=2)
    ax.tick_params(axis="y", which="both", length=0)

    # Highlight FORMULA-Sigma column with a thin frame
    ax.add_patch(mpatches.Rectangle((-0.5, -0.5), 1.0, len(labels),
                                     fill=False, ec=W["red"], lw=1.6,
                                     clip_on=False))

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = "yes" if v == 1 else ("part" if v in (0.3, 0.5) else "no")
            colour = "white" if v >= 0.7 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=colour, fontsize=7.5, weight="bold" if v == 1 else "normal")

    # Minor grid between cells (visual separation)
    ax.set_xticks(np.arange(-0.5, len(tools), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)

    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["no", "partial", "yes"], fontsize=7.5)
    cb.outline.set_visible(False)

    fig.suptitle("Figure 2. Feature parity: FORMULA-Sigma vs commercial and open competitors",
                 fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.30, right=0.93, top=0.93, bottom=0.14)
    save(fig, "Figure_2_comparison_heatmap")


# ============================================================
# F3 - Cross-tool agreement (-log10 scale)
# ============================================================
def fig3_cross_tool():
    e01 = paths.RESULTS / "e01"
    data = []
    for fname, label in [
        ("e01_01_ccd.json",     "Central composite\n(rotatable 3F)"),
        ("e01_02_bbd.json",     "Box-Behnken\n(3F)"),
        ("e01_03_scheffe.json", "Scheffe\n{3,3} mixture"),
    ]:
        d = json.loads((e01 / fname).read_text())
        rel = d.get("max_rel_coef_diff_sm_vs_numpy",
                    d.get("max_rel_coef_diff_ours_vs_numpy"))
        data.append((label, rel))

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    labels = [d[0] for d in data]
    vals = [max(d[1], 1e-16) for d in data]
    logvals = [-np.log10(v) for v in vals]
    bars = ax.barh(range(len(labels)), logvals,
                   color=W["green"], edgecolor="black", height=0.55)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(logvals) + 4)
    ax.set_xlabel(r"$-\log_{10}$  (max relative coefficient difference vs reference)")

    # Threshold marker - put label at top of axis to avoid x-tick collision
    ax.axvline(6, ls="--", color=W["red"], lw=1.1)
    ax.text(6.15, -0.55, r"$10^{-6}$ threshold",
            color=W["red"], fontsize=8, va="bottom")

    for i, (bar, val, lv) in enumerate(zip(bars, vals, logvals)):
        ax.text(lv + 0.25, bar.get_y() + bar.get_height() / 2,
                f"{val:.2e}", va="center", fontsize=8.5, color="#222")

    ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Figure 3. Cross-tool numerical agreement at machine precision",
                 fontsize=11, y=0.97)
    fig.subplots_adjust(left=0.22, right=0.96, top=0.86, bottom=0.20)
    save(fig, "Figure_3_cross_tool")


# ============================================================
# F4 - Retrospective case studies (2x2 panel)
# ============================================================
def fig4_case_panel():
    e02 = paths.RESULTS / "e02"
    cases = [
        ("case1_sharma",  "Case 1: Farooqi 2020 (CCD)", "case1_summary.csv"),
        ("case2_arif",    "Case 2: Arif 2022 (D-opt mix-proc)", "case2_summary.csv"),
        ("case3_boscolo", "Case 3: Boscolo 2023 (BBD)", "case3_summary.csv"),
        ("case4_nemr",    "Case 4: Nemr 2022 (D-opt categorical)", "case4_summary.csv"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))

    label_map = {
        "Y1": "Y1 (1h)", "Y2": "Y2 (5h)", "Y3": "Y3 (10h)", "Y4": "Y4 (f2)",
        "PS": "PS (nm)", "PDI": "PDI", "EE": "EE (%)",
        "ee_pct": "EE (%)", "ps_nm": "PS (nm)", "pdi": "PDI",
        "q2h_pct": "Q2h (%)", "q24h_pct": "Q24h (%)", "zp_mV": "ZP (mV)",
        "particle_size_nm": "PS (nm)",
    }

    for ax, (folder, title, csv) in zip(axes.ravel(), cases):
        p = e02 / folder / csv
        if not p.exists():
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center")
            ax.set_title(title); ax.axis("off"); continue
        df = pd.read_csv(p)
        # pick R2 column
        r2col = "R2" if "R2" in df.columns else ("R2_ours" if "R2_ours" in df.columns else None)
        if r2col is None:
            ax.text(0.5, 0.5, "(no R^2)", ha="center", va="center"); ax.axis("off")
            ax.set_title(title); continue
        d = df.dropna(subset=[r2col]).copy()
        d = d.sort_values(r2col, ascending=True)
        # Boscolo case has no "response" column - it's a single-response BBD; label by model variant
        if "response" in d.columns:
            label_src = d["response"].astype(str).tolist()
        elif "model" in d.columns:
            label_src = [m.replace("_", " ") for m in d["model"].astype(str).tolist()]
        else:
            label_src = [f"row{i}" for i in range(len(d))]
        labels = [label_map.get(r, r) for r in label_src]
        vals = d[r2col].astype(float).tolist()

        y = np.arange(len(labels))
        bars = ax.barh(y, vals, color=W["blue"], edgecolor="black",
                       height=0.55)
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.set_xlim(0, 1.06)
        ax.set_xlabel(r"$R^2$ of FORMULA-$\Sigma$ refit")
        ax.axvline(0.9, color=W["red"], ls="--", lw=1.0)
        # Threshold label below x-axis to avoid colliding with bar values
        ax.text(0.9, -0.85, r"$R^2 = 0.9$",
                color=W["red"], fontsize=7.5, ha="center", va="top")

        # Text positioned OUTSIDE bar end so no overlap with bar fill or axis edge
        for b, v in zip(bars, vals):
            tx = v + 0.012
            ha = "left"
            if tx > 1.0:
                tx = v - 0.012; ha = "right"
                colour = "white"
            else:
                colour = "#222"
            ax.text(tx, b.get_y() + b.get_height() / 2,
                    f"{v:.3f}", va="center", ha=ha,
                    fontsize=8, color=colour)
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(title, fontsize=10)

    fig.suptitle("Figure 5. Retrospective reproduction: per-response $R^2$ across four cases",
                 fontsize=11.5, y=0.995)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.08,
                         hspace=0.45, wspace=0.42)
    save(fig, "Figure_5_case_panel")


# ============================================================
# F5 - Synthetic benchmarks (2x2 grid)
# ============================================================
def fig5_synthetic():
    e03 = paths.RESULTS / "e03"
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))

    # 5a BO convergence (3 test functions, BO vs random)
    ax = axes[0, 0]
    bo = json.loads((e03 / "bo" / "bo_detail.json").read_text())
    colours = {"Branin": W["blue"], "Hartmann-6": W["orange"], "Ackley-4": W["green"]}
    for fn in ["Branin", "Hartmann-6", "Ackley-4"]:
        m_bo = bo[fn]["methods"]["BO"]["median_regret_history"]
        m_rd = bo[fn]["methods"]["RAND"]["median_regret_history"]
        x = np.arange(1, len(m_bo) + 1)
        ax.plot(x, m_bo, color=colours[fn], lw=1.6, label=f"{fn} (BO)")
        ax.plot(x, m_rd, color=colours[fn], lw=1.0, ls=":", alpha=0.75,
                label=f"{fn} (random)")
    ax.set_yscale("log")
    ax.set_xlabel("Evaluation budget")
    ax.set_ylabel("Simple regret  (median, 5 seeds)")
    ax.set_title("(a)  Bayesian optimisation vs random search", loc="left")
    ax.legend(fontsize=6.5, ncol=2, loc="upper right", handlelength=2.2)
    ax.grid(True, which="both", alpha=0.3)

    # 5b PDS recovery (3-bar)
    ax = axes[0, 1]
    pds = json.loads((e03 / "pds" / "pds_detail.json").read_text())
    cats = ["Analytic\n(ground truth)", "FORMULA-Sigma\n(recovered)", "Intersection"]
    vals = [pds["n_analytic"], pds["n_recovered"], pds["intersection_cells"]]
    cols = [W["green"], W["blue"], W["sky"]]
    bars = ax.bar(cats, vals, color=cols, edgecolor="black", width=0.6)
    ax.set_ylabel(r"Grid cells with $P \geq 0.95$")
    ax.set_ylim(0, max(vals) * 1.15)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.015,
                f"{int(v)}", ha="center", va="bottom", fontsize=8.5)
    ax.text(0.97, 0.97,
            f"Jaccard = {pds['jaccard_similarity']:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.25",
                                    fc="white", ec="#aaa", lw=0.6))
    ax.set_title("(b)  Probabilistic design space recovery", loc="left")
    ax.grid(axis="y", alpha=0.3)

    # 5c Robust gain (3-bar)
    ax = axes[1, 0]
    rob = json.loads((e03 / "robust" / "robust_detail.json").read_text())
    y_nom_clean = 0.80
    y_nom_noisy = rob["y_at_nominal_under_noise"]
    y_rob_noisy = rob["y_robust_expected_under_noise"]
    cats = ["Nominal\n(noise-free)", "Nominal\nunder noise", "Robust\nunder noise"]
    vals = [y_nom_clean, y_nom_noisy, y_rob_noisy]
    bars = ax.bar(cats, vals, color=[W["sky"], W["orange"], W["green"]],
                  edgecolor="black", width=0.6)
    ax.set_ylabel("E[y]  (higher is better)")
    ax.set_ylim(0, max(vals) * 1.18)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.text(0.97, 0.97,
            f"robust gain = +{rob['expected_gain_robust_over_nominal']:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.25",
                                    fc="white", ec="#aaa", lw=0.6))
    ax.set_title("(c)  Robust optimisation under input noise", loc="left")
    ax.grid(axis="y", alpha=0.3)

    # 5d Pareto HV (2-bar)
    ax = axes[1, 1]
    par = json.loads((e03 / "pareto" / "pareto_detail.json").read_text())
    cats = ["NSGA-II\nobserved", "Analytic\ntrue HV"]
    vals = [par["hypervolume_observed"], par["hypervolume_true"]]
    bars = ax.bar(cats, vals, color=[W["blue"], W["green"]],
                  edgecolor="black", width=0.5)
    ax.set_ylabel("Hypervolume  (ZDT1, ref = (1.1, 1.1))")
    ax.set_ylim(0, max(vals) * 1.18)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.text(0.97, 0.97,
            f"HV ratio = {par['hypervolume_ratio']:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.25",
                                    fc="white", ec="#aaa", lw=0.6))
    ax.set_title("(d)  NSGA-II Pareto on ZDT1", loc="left")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Figure 7. Synthetic benchmarks: BO, PDS, robust, Pareto recover ground truth",
                 fontsize=11.5, y=0.995)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.10,
                         hspace=0.50, wspace=0.30)
    save(fig, "Figure_7_synthetic")


# ============================================================
# Figure 8 (PDS probability heatmap) is rendered from the live
# Benchmarking project on the proprietary FORMULA-Sigma platform
# and is therefore supplied as a static asset (see figures_static/).
# It is intentionally not regenerated by this open harness.
# ============================================================


def main():
    print("Generating manuscript main figures...")
    fig1_architecture()
    fig2_competitor_heatmap()
    fig3_cross_tool()
    fig4_case_panel()
    fig5_synthetic()
    print("Done.")
    print(f"Outputs: {FIG_DIR}")


if __name__ == "__main__":
    main()
