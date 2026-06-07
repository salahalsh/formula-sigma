"""Figure 6 - Farooqi 2020 published-equation anomaly.

Compares predictions of three things for the same 19 design points:
  (a) Author's published Table 5 main-effect (Y1-Y3) and full quadratic (Y4)
      equation.
  (b) FORMULA-Sigma's full-quadratic OLS refit on the published Table 1
      per-run responses (R^2 ~ 0.96 for Y1, see e02).
  (c) Observed Table 1 response values.

The point of the figure: (a) does NOT predict (c) for Y1-Y3 even though it
was reported by the original authors. (b) does. This is the central piece of
the §3.2.1 anomaly subsection.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths, plotting

plotting.apply_paper_style()
W = plotting.WONG

CASE_DIR = paths.ROOT / "03_data" / "processed" / "case1_tablet_ccd"
FIG_DIR = paths.FIGURES_RAW
FIG_DIR.mkdir(parents=True, exist_ok=True)

design = pd.read_csv(CASE_DIR / "case1_design.csv")
resp = pd.read_csv(CASE_DIR / "case1_responses.csv")
df = design.merge(resp, on="run")

# Exclude F-19 to match the e02 reproduction (and the manuscript): F-19 is the
# alpha = -1.682 axial point on X2 (orifice size) which decodes to -0.004 mm (no orifice)
# (unphysical, all responses set to 0 in the published table). The same outlier
# mask is applied in e02 case1_sharma/reproduce.py.
df = df[df["run"].astype(str).str.strip() != "F-19"].reset_index(drop=True)

# Coded factors (X1, X2, X3) - sanitise unicode minus
X1 = df["X1_coded"].astype(str).str.replace("−", "-").astype(float).values
X2 = df["X2_coded"].astype(str).str.replace("−", "-").astype(float).values
X3 = df["X3_coded"].astype(str).str.replace("−", "-").astype(float).values


def yhat_pub(resp_name):
    if resp_name == "Y1_pct":
        return 20.3379 + 0.493808*X1 + 11.4995*X2 + -2.02491*X3
    if resp_name == "Y2_pct":
        return 25.1651 + 2.38424*X1 + 28.0141*X2 + -2.58589*X3
    if resp_name == "Y3_pct":
        return 49.6625 + 3.3092*X1 + 36.606*X2 + -3.00681*X3
    if resp_name == "Y4":
        return (0.493169 + -0.0388303*X1 + 2.33354*X2 + -0.0129157*X3
                + -0.00824167*X1*X2 + 0.00276812*X1*X3 + 0.0155729*X2*X3
                + 0.000998447*X1**2 + -2.1592*X2**2 + -0.000213218*X3**2)


def yhat_fs(resp_name):
    y = pd.to_numeric(df[resp_name], errors="coerce").values
    m = np.isfinite(y)
    X = np.column_stack([X1, X2, X3])
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xp = poly.fit_transform(X[m])
    lr = LinearRegression().fit(Xp, y[m])
    return lr.predict(poly.transform(X)), y, m


fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.4))
responses = [
    ("Y1_pct", "Y1: release 1 h (%)"),
    ("Y2_pct", "Y2: release 6 h (%)"),
    ("Y3_pct", "Y3: release 12 h (%)"),
    ("Y4",     "Y4: RSQ-zero"),
]

for ax, (rcol, label) in zip(axes, responses):
    yp = yhat_pub(rcol)
    yfs, yobs, mask = yhat_fs(rcol)
    yp = yp[mask]; yfs = yfs[mask]; yobs = yobs[mask]

    # Clip view to the observed-response range (with padding) plus FS refit
    # range, EXCLUDING extreme published-equation outliers (the whole point
    # of the figure is that those outliers exist). Off-panel Pub points get
    # annotated below.
    in_view = np.concatenate([yobs, yfs])
    lo = float(np.nanmin(in_view))
    hi = float(np.nanmax(in_view))
    span = (hi - lo) if hi > lo else 1.0
    pad = 0.10 * span
    lim = (lo - pad, hi + pad)

    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.55, label="Identity")
    ax.scatter(yobs, yp, s=48, color=W["red"], marker="o",
                edgecolor="black", lw=0.5, zorder=3, alpha=0.9,
                label="Published equation")
    ax.scatter(yobs, yfs, s=48, color=W["blue"], marker="^",
                edgecolor="black", lw=0.5, zorder=4, alpha=0.9,
                label="FORMULA-Sigma refit")

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    r2_p = r2_score(yobs, yp)
    r2_fs = r2_score(yobs, yfs)
    ax.set_title(label, fontsize=9.5, loc="left")

    # Count Pub outliers that fall off-panel
    n_off = int(np.sum((yp < lim[0]) | (yp > lim[1])))
    off_str = f"  ({n_off} Pub off-panel)" if n_off else ""

    ax.text(0.04, 0.96,
             f"Pub $R^2$ = {r2_p: .3f}{off_str}\nFS  $R^2$ = {r2_fs: .3f}",
             transform=ax.transAxes, ha="left", va="top",
             fontsize=8.5, family="monospace",
             bbox=dict(boxstyle="round,pad=0.30",
                        fc="white", ec="#aaa", lw=0.6))
    ax.grid(True, alpha=0.3)

# One shared legend, placed above the row of axes
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="upper center", ncol=3,
            fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.965))

fig.suptitle("Figure 6. Farooqi 2020 published equation vs FORMULA-Sigma refit "
              "vs observed data  (n = 19)",
              fontsize=11.5, y=0.998)
fig.subplots_adjust(left=0.05, right=0.98, top=0.83, bottom=0.13, wspace=0.32)
pdf, png = plotting.save_both(fig, "Figure_6_sharma_anomaly", FIG_DIR)
print(f"-> {pdf.name} + {png.name}")
